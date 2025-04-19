import streamlit as st
import requests
import time
import logging
from datetime import datetime
import pytz
from google.cloud import firestore
import pandas as pd
from PIL import Image
import io
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamlitApp")

API_URL = "http://localhost:8000/live" # Changes according to the deployment(Ours Google CLoud)
CAPTURE_URL = "http://localhost:8001/capture"
IMAGE_URL = "http://localhost:8001/image/"

# Initialize Firestore
db = firestore.Client(project="cloudsqltest-457110") #changes according to our projectName

st.markdown("""
    <style>
    .live-prediction {
        border: 3px solid #00ffcc;
        padding: 20px;
        border-radius: 10px;
        background-color: #1a1a1a;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 20px #00ffcc; }
        50% { box-shadow: 0 0 30px #00ffcc; }
        100% { box-shadow: 0 0 20px #00ffcc; }
    }
    .live-status {
        font-size: 30px;
        color: #00ffcc;
        font-weight: bold;
        text-shadow: 0 0 10px #00ffcc;
    }
    .live-count {
        font-size: 22px;
        color: #ff007a;
    }
    .live-timestamp {
        font-size: 18px;
        color: #33ccff;
    }
    .loading {
        font-size: 20px;
        color: #ffffff;
        text-align: center;
    }
    .history-table {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .history-table h3 {
        color: #00ffcc;
        text-shadow: 0 0 10px #00ffcc;
    }
    .capture-button {
        background-color: #ff007a;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px;
    }
    .captured-image {
        border: 2px solid #33ccff;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Audio and Person Detection Dashboard")

def fetch_prediction():
    try:
        response = requests.get(API_URL, timeout=5)
        response.raise_for_status()
        prediction = response.json()
        logger.info(f"Fetched prediction: {prediction}")
        return prediction
    except requests.RequestException as e:
        logger.error(f"Error fetching prediction: {e}")
        st.error(f"Error fetching prediction: {e}")
        return {}

def capture_image():
    try:
        response = requests.post(CAPTURE_URL, timeout=5)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Capture response: {result}")
        return result.get("image_path")
    except requests.RequestException as e:
        logger.error(f"Error capturing image: {e}")
        st.error(f"Error capturing image: {e}")
        return None

def fetch_image(image_path):
    filename = os.path.basename(image_path)
    try:
        response = requests.get(IMAGE_URL + filename, timeout=5)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except requests.RequestException as e:
        logger.error(f"Error fetching image: {e}")
        st.error(f"Error fetching image: {e}")
        return None

def fetch_history():
    try:
        docs = db.collection("predictions").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        history = []
        for doc in docs:
            data = doc.to_dict()
            # Only include predictions that would have human speech detected
            # Since Firestore only saves when consecutive_high_count >= 4 and person_count >= 2
            data["status"] = "Human speech detected"
            history.append(data)
        return history
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        st.error(f"Error fetching history: {e}")
        return []

def format_timestamp(iso_timestamp):
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except:
        return iso_timestamp

# Session state for captured image
if "image_path" not in st.session_state:
    st.session_state.image_path = None

# Live Detection Section
with st.container():
    st.markdown('<div class="live-prediction">', unsafe_allow_html=True)
    st.write("Live Detection:")
    placeholder = st.empty()
    prediction = fetch_prediction()
    if prediction:
        logger.info(f"Displaying live prediction: {prediction}")
        status = prediction.get("status", "No human speech detected")
        person_count = prediction.get("person_count", 0)
        timestamp = format_timestamp(prediction.get("timestamp", "N/A"))
        with placeholder.container():
            st.markdown(f'<div class="live-status">{status}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="live-count">Persons Detected: {person_count}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="live-timestamp">Timestamp: {timestamp}</div>', unsafe_allow_html=True)
    else:
        placeholder.markdown('<div class="loading">Waiting for predictions...</div>', unsafe_allow_html=True)

    # Capture Button
    if st.button("Capture Image", key="capture_button", help="Capture a photo from the camera"):
        image_path = capture_image()
        if image_path:
            st.session_state.image_path = image_path
            logger.info(f"Image captured: {image_path}")

    # Display Captured Image
    if st.session_state.image_path:
        image = fetch_image(st.session_state.image_path)
        if image:
            st.image(image, caption="Captured Image", use_column_width=True, output_format="JPEG", channels="BGR")
            st.markdown('<div class="captured-image"></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# History Table
with st.container():
    st.markdown('<div class="history-table">', unsafe_allow_html=True)
    st.subheader("Prediction History (Human Speech Detected)")
    history = fetch_history()
    if history:
        df = pd.DataFrame(history)
        df["timestamp"] = df["timestamp"].apply(format_timestamp)
        df = df[["timestamp", "status", "person_count", "noise_level"]]
        df = df.rename(columns={"timestamp": "Timestamp", "status": "Status", "person_count": "Persons", "noise_level": "Noise Level"})
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No predictions with human speech detected yet.")
    st.markdown('</div>', unsafe_allow_html=True)

time.sleep(5)
st.rerun()