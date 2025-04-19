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

        self.audio_queue.put(indata.copy())


    def send_prediction(self):
        """Send noise_level, person_count, and timestamp to FastAPI /predict."""
        prediction = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "noise_level": self.current_noise_level,
            "person_count": self.current_count
        }

        try:
            response = requests.post(self.api_url, json=prediction)
            response.raise_for_status()

        except:
            pass



    def inference_loop(self):
        """Process audio chunks for speech detection."""
        audio_buffer = np.array([], dtype=np.float32)
        last_prediction_time = time.time()
        prediction_interval = 1.0

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
                        overlap_samples = int(self.SAMPLE_RATE * 2.0)
                        if len(audio_buffer) > overlap_samples:
                            audio_buffer = audio_buffer[-overlap_samples:]
                        last_prediction_time = current_time
                except queue.Empty:
                    continue

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
        self.current_frame = frame.copy()

        results = self.model(frame)
        
        # Get the detections
        result = results[0]
        
        # Count people (class 0 in COCO dataset)
        person_count = 0
        
        annotated_frame = frame.copy()
        
        for box in result.boxes:
            # Check if detection is a person and confidence is above threshold
            if int(box.cls) == self.person_class_id and box.conf >= self.confidence:
                person_count += 1
                
                # Get the bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw rectangle around person
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display the count on the frame
        cv2.putText(annotated_frame, f"People count: {person_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Noise level: {self.current_noise_level}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(annotated_frame, "Press 's' to stop, 'c' to capture",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        self.current_count = person_count
        
        return  person_count,annotated_frame


    def capture_image(self, output_path="captured_image.jpg"):
        """
        Capture the current frame and save it as an image.

        Args:
            output_path: Path to save the captured image

        Returns:
            Boolean indicating if capture was successful
        """
        if self.current_frame is None:

            return False
        try:
            base, ext = os.path.splitext(output_path)
            if os.path.exists(output_path):
                output_path = f"{base}_{int(time.time())}{ext}"
            cv2.imwrite(output_path, self.current_frame)

            return True
        except Exception as e:

            return False


    def start_camera(self, camera_id=0):
        """
        Start detection with a camera feed.
        
        Args:
            camera_id: Camera device ID (default: 0 for primary camera)
        """
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("Starting person detection. Press 's' to stop cleanly.")
        # Start audio inference
        self.recording = True
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()


        self.running = True
        
        try:
            with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1, dtype='float32', callback=self.audio_callback):
                while self.running:
                    ret, frame = self.cap.read()
                    if not ret:

                        break
                    person_count, annotated_frame = self.process_frame(frame)
                    cv2.imshow("Speech Detection", annotated_frame)
                    self.send_prediction()  # Send prediction after processing frame
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):

                        self.running = False
                        break
                    elif key == ord('c'):
                        self.capture_image()

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
        print("Stop signal received. Detection will stop after current frame.")


# Example usage:
if __name__ == "__main__":

    # Create detector instance
    detector = SpeechDetector()
    # Start webcam detection
    detector.start_camera()