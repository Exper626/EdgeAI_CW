import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
from queue import Queue
from threading import Thread

# Constants
SAMPLE_RATE = 16000  # Yamnet expects 16kHz audio
CHUNK_DURATION = 1.0  # Process 1-second chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.001

# Load models
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
my_model = tf.keras.models.load_model('models/audio_classifier_no_silence.h5')
my_classes = ['vacuum_cleaner', 'door_wood_knock', 'footsteps', 'coughing', 'laughing', 'speech']

# Audio buffer and queue for processing
audio_buffer = np.array([], dtype=np.float32)

prediction_queue = Queue()


def preprocess_audio(audio_data):
    """Convert audio to format expected by Yamnet"""
    audio_data = audio_data.astype(np.float32)
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    return audio_data


def predict_audio_chunk(audio_chunk):
    """Process an audio chunk and return the predicted class"""
    # Check for silence
    audio_energy = np.mean(audio_chunk ** 2)
    if audio_energy < SILENCE_THRESHOLD:
        return "silence"

    # Get embeddings from Yamnet
    audio_tensor = tf.convert_to_tensor(audio_chunk, dtype=tf.float32)
    scores, embeddings, _ = yamnet_model(audio_tensor)

    # Make prediction
    predictions = my_model(embeddings)
    prediction = tf.reduce_mean(predictions, axis=0)
    class_id = tf.argmax(prediction).numpy()

    return my_classes[class_id]


def audio_callback(indata, frames, time, status):
    """Callback function for sounddevice to process incoming audio"""
    global audio_buffer
    audio_chunk = indata[:, 0]  # Get mono channel
    audio_buffer = np.concatenate([audio_buffer, audio_chunk])

    # Process when we have enough samples
    while len(audio_buffer) >= CHUNK_SIZE:
        chunk_to_process = audio_buffer[:CHUNK_SIZE]
        audio_buffer = audio_buffer[CHUNK_SIZE:]

        # Preprocess and predict
        processed_chunk = preprocess_audio(chunk_to_process)
        prediction = predict_audio_chunk(processed_chunk)
        prediction_queue.put(prediction)


def prediction_printer():
    """Thread function to print predictions from the queue"""
    while True:
        prediction = prediction_queue.get()
        print(f"Predicted sound: {prediction}")


def start_real_time_prediction():
    """Start real-time audio prediction"""
    # Start prediction printer thread
    printer_thread = Thread(target=prediction_printer, daemon=True)
    printer_thread.start()

    # Start audio stream
    print("Starting real-time prediction... Press Ctrl+C to stop.")
    try:
        with sd.InputStream(callback=audio_callback,
                            channels=1,
                            samplerate=SAMPLE_RATE,
                            blocksize=int(SAMPLE_RATE * 0.5)):  # 0.5 second blocks
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopping real-time prediction...")


if __name__ == "__main__":
    start_real_time_prediction()