import cv2
import numpy as np
from ultralytics import YOLO
import time
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import threading
import queue
import os
import wave
import tempfile
import requests
from datetime import datetime
import logging
from fastapi import FastAPI
from starlette.responses import FileResponse
import uvicorn
import aiohttp
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpeechDetector")

app = FastAPI()

class SpeechDetector:
    """
    A class for detecting people in video streams using YOLOv8.
    Triggers events when 2 or more people are detected.
    Press 's' to stop the detection cleanly.
    """
    
    def __init__(self, video_model_path='edge_device/models/yolov8n.pt', audio_model_path='../models/audio_classifier_test.h5', confidence=0.5):
        """
        Initialize the person detector.
        
        Args:
            model_path: Path to YOLOv8 model file (.pt)
            confidence: Confidence threshold for detections
        """
        self.model = YOLO(video_model_path)
        self.confidence = confidence
        self.person_class_id = 0  # In COCO dataset, person is class 0
        self.current_count = 0
        self.current_frame = None
        self.cap = None

        # Initialize AudioDetectorModel components
        self.yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
        self.yamnet_model = hub.load(self.yamnet_model_handle)
        self.audio_model = tf.keras.models.load_model(audio_model_path)
        self.my_classes = ['vacuum_cleaner', 'door_wood_knock', 'footsteps', 'coughing', 'laughing', 'keyboard_typing',
                           'clock_alarm', 'speech']
        self.SAMPLE_RATE = 16000
        self.CHUNK_DURATION = 2.0  # 5 was giving a delay in predictions so changed to 2
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        self.SILENCE_THRESHOLD = 0.0001
        self.audio_queue = queue.Queue()
        self.current_noise_level = "Normal"

        # Control variables
        self.running = False
        self.recording = False
        self.inference_thread = None
        self.prediction_thread = None

        # FastAPI endpoint
        self.api_url = os.getenv("FASTAPI_URL", "http://localhost:8000/predict") # will be replaced with our cloud URL
        logger.info(f"Using API URL: {self.api_url}")
        self.last_prediction_sent = 0
        self.prediction_interval = 1.0

        # Directory for captured images
        self.image_dir = "captured_images"
        os.makedirs(self.image_dir, exist_ok=True)

    def load_wav_16k_mono(self, filename):
        """Load a WAV file, convert to float tensor, resample to 16 kHz single-channel."""
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)

        def true_fn(): return wav

        def false_fn():
            wav_len = tf.cast(tf.shape(wav)[0], tf.float32)
            sr = tf.cast(sample_rate, tf.float32)
            desired_length = tf.cast(wav_len * 16000.0 / sr, tf.int32)
            return tf.image.resize(wav[tf.newaxis, :, tf.newaxis], [desired_length, 1])[0, :, 0]

        wav = tf.cond(sample_rate == 16000, true_fn, false_fn)
        return wav

    def save_audio_chunk(self, audio_data, sample_rate=None):
        """Save audio data to a temporary WAV file."""
        if sample_rate is None:
            sample_rate = self.SAMPLE_RATE
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        try:
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            return temp_filename
        except Exception as e:
            logger.error(f"Error saving audio chunk: {e}")
            raise

    def predict_sound_class(self, audio_data):
        """Predict sound class from audio data."""
        audio_energy = np.mean(np.square(audio_data))
        if audio_energy < self.SILENCE_THRESHOLD:
            return 'silence'
        temp_filename = self.save_audio_chunk(audio_data)
        try:
            wav_data = self.load_wav_16k_mono(temp_filename)
            scores, embeddings, _ = self.yamnet_model(wav_data)
            predictions = self.audio_model(embeddings)
            prediction = tf.reduce_mean(predictions, axis=0)
            class_id = tf.argmax(prediction).numpy()
            return self.my_classes[class_id]
        finally:
            try:
                os.unlink(temp_filename)
            except:
                pass

    def noise_class(self, class_name):
        """Classify the noise level based on the detected sound class."""
        return "High" if class_name in ['laughing', 'speech'] else "Normal"

    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice stream."""

        if status:
            logger.error(f"Stream status: {status}")
        self.audio_queue.put(indata.copy())


    async def send_prediction(self):
        """Send noise_level, person_count, and timestamp to FastAPI /predict."""
        prediction = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "noise_level": self.current_noise_level,
            "person_count": self.current_count
        }
        logger.info(f"Sending prediction to {self.api_url}: {prediction}")
        for attempt in range(5):
            try:
                async with session.post(self.api_url, json=prediction, timeout=5) as response:
                    response = requests.post(self.api_url, json=prediction)
                    response.raise_for_status()
                    logger.info(f"Prediction sent successfully: {result}")
                    return

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 4:
                    await asyncio.sleep(1)
        logger.error(f"Failed to send prediction after 5 attempts: {prediction}")

    async def prediction_loop(self):
        async with aiohttp.ClientSession() as session:
            try:
                while self.running:
                    current_time = time.time()
                    if current_time - self.last_prediction_sent >= self.prediction_interval:
                        await self.send_prediction(session)
                        self.last_prediction_sent = current_time
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
            finally:
                logger.info("Prediction loop stopped")

    def start_prediction_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.prediction_loop())
        loop.close()

    def inference_loop(self):
        """Process audio chunks for speech detection."""
        audio_buffer = np.array([], dtype=np.float32)
        last_prediction_time = time.time()
        prediction_interval = 0.5 # 1 changed to 0.5 to get better preditions

        try:
            while self.recording:
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    audio_chunk = audio_chunk.flatten().astype(np.float32)
                    audio_buffer = np.append(audio_buffer, audio_chunk)
                    current_time = time.time()
                    if (current_time - last_prediction_time >= prediction_interval and
                            len(audio_buffer) >= self.CHUNK_SIZE):
                        chunk_to_process = audio_buffer[-self.CHUNK_SIZE:]
                        prediction_label = self.predict_sound_class(chunk_to_process)
                        self.current_noise_level = self.noise_class(prediction_label)
                        overlap_samples = int(self.SAMPLE_RATE * 1.0)
                        if len(audio_buffer) > overlap_samples:
                            audio_buffer = audio_buffer[-overlap_samples:]
                        last_prediction_time = current_time
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            logger.info("Stopping audio inference loop")

        finally:
            self.recording = False






    def register_multi_person_callback(self, callback):
        """
        Register a callback function to be called when 2+ people are detected.
        The callback will receive the number of people detected as an argument.
        
        Args:
            callback: Function to call when multiple people are detected
        """
        self.callbacks.append(callback)
        
    def _notify_callbacks(self, count):
        """
        Notify all registered callbacks about person count.
        
        Args:
            count: Number of people detected
        """
        for callback in self.callbacks:
            callback(count)
    
    def process_frame(self, frame):
        """
        Process a single frame and detect people.
        
        Args:
            frame: The image frame to process
            
        Returns:
            Annotated frame with detection boxes and person count
        """
        start_time = time.time()
        self.current_frame = frame.copy()
        resized_frame = cv2.resize(frame, (640, 480))
        results = self.model(resized_frame)
        
        # Get the detections
        result = results[0]
        
        # Count people (class 0 in COCO dataset)
        person_count = 0
        
        annotated_frame = frame.copy()

        orig_height, orig_width = frame.shape[:2]
        scale_x, scale_y = orig_width / 640, orig_height / 480
        
        for box in result.boxes:
            # Check if detection is a person and confidence is above threshold
            if int(box.cls) == self.person_class_id and box.conf >= self.confidence:
                person_count += 1
                
                # Get the bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

                # Draw rectangle around person
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display the count on the frame
        cv2.putText(annotated_frame, f"People count: {person_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Noise level: {self.current_noise_level}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        fps = 1 / (time.time() - start_time) if time.time() > start_time else 0
        cv2.putText(annotated_frame, "Press 's' to stop, 'c' to capture",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        self.current_count = person_count
        
        return  person_count,annotated_frame, fps


    def capture_image(self, output_path=None):
        """
        Capture the current frame and save it as an image.

        Args:
            output_path: Path to save the captured image

        Returns:
            Boolean indicating if capture was successful
        """
        if self.current_frame is None:
            logger.error("No frame available to capture")
            return False
        try:
            if output_path is None:
                timestamp = int(time.time())
                output_path = os.path.join(self.image_dir, f"capture_{timestamp}.jpg")
            cv2.imwrite(output_path, self.current_frame)
            logger.info(f"Image captured and saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False


    def start(self, camera_id=0):
        """
        Start detection with a camera feed.
        
        Args:
            camera_id: Camera device ID (default: 0 for primary camera)
        """
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Start audio inference
        self.recording = True
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()

        self.prediction_thread = threading.Thread(target=self.start_prediction_loop)
        self.prediction_thread.daemon = True
        self.prediction_thread.start()

        logger.info("Starting speech detection. Press 's' to stop, 'c' to capture image.")
        self.running = True
        
        try:
            with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1, dtype='float32', callback=self.audio_callback):
                while self.running:
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.error("Failed to capture frame")
                        break
                    person_count, annotated_frame, fps = self.process_frame(frame)
                    cv2.imshow("Speech Detection", annotated_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        logger.info("Stopping detection gracefully...")
                        self.running = False
                        break
                    elif key == ord('c'):
                        self.capture_image()
        except Exception as e:
            logger.error(f"Error during detection: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """
        Stop the detection process externally if needed.
        """
        self.running = False
        self.recording = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1)
        if self.prediction_thread and self.prediction_thread.is_alive():
            self.prediction_thread.join(timeout=1)
        logger.info("Detection stopped successfully.")


# FastAPI endpoints for capture
detector = None

@app.post("/capture")
async def capture_image_endpoint():
    global detector
    if detector is None or not detector.running:
        raise HTTPException(status_code=400, detail="Detector is not running")
    image_path = detector.capture_image()
    if image_path is None:
        raise HTTPException(status_code=500, detail="Failed to capture image")
    return {"image_path": image_path}

@app.get("/image/{filename}")
async def get_image(filename: str):
    image_path = os.path.join(detector.image_dir, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8001)

# Example usage:
if __name__ == "__main__":

    # Create detector instance
    detector = SpeechDetector()
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    detector.start()