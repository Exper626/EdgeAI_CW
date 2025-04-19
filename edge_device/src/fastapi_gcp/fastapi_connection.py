from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import logging
from google.cloud import firestore
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API2_for_audio")

db = firestore.Client(project="cloudsqltest-457110")

app = FastAPI()

latest_image = None
capture_requested = False

class Prediction(BaseModel):
    timestamp: str
    noise_level: str
    person_count: int

    @validator("noise_level")
    def validate_noise_level(cls, v):
        if v not in ["High", "Normal"]:
            raise ValueError("noise_level must be 'High' or 'Normal'")
        return v

class ImageCapture(BaseModel):
    image_data: str

predictions = []
consecutive_high_count = 0
CONSECUTIVE_THRESHOLD = 4
MAX_PREDICTIONS = 100

@app.post("/predict")
async def save_prediction(prediction: Prediction):
    global consecutive_high_count
    pred_dict = prediction.dict()
    logger.info(f"Received prediction: {pred_dict}")

    predictions.append(pred_dict)
    if len(predictions) > MAX_PREDICTIONS:
        predictions[:] = predictions[-MAX_PREDICTIONS:]

    if pred_dict["noise_level"] == "High":
        consecutive_high_count += 1
    else:
        consecutive_high_count = 0
    logger.info(f"Consecutive High count: {consecutive_high_count}, Total predictions: {len(predictions)}")

    if consecutive_high_count >= CONSECUTIVE_THRESHOLD and pred_dict["person_count"] >= 2:
        try:
            doc_ref = db.collection("predictions").add(pred_dict)
            logger.info(f"Saved to Firestore with ID: {doc_ref[1].id}, Data: {pred_dict}")
        except Exception as e:
            logger.error(f"Firestore save error: {e}")
            raise HTTPException(status_code=500, detail=f"Firestore save error: {str(e)}")

    return {"status": "Prediction received"}

@app.get("/live")
async def get_live():
    if not predictions:
        logger.info("No predictions in memory")
        return {}
    latest_prediction = predictions[-1]
    status = "Human speech detected" if consecutive_high_count >= CONSECUTIVE_THRESHOLD else "No human speech detected"
    response = {
        "status": status,
        "person_count": latest_prediction["person_count"],
        "timestamp": latest_prediction["timestamp"]
    }
    logger.info(f"Serving live prediction: {response}")
    return response

@app.post("/capture")
async def capture_image(capture: ImageCapture):
    global latest_image
    try:
        latest_image = capture.image_data
        logger.info("Image received and stored in memory")
        return {"status": "Image received"}
    except Exception as e:
        logger.error(f"Error storing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error storing image: {str(e)}")

@app.get("/image")
async def get_image():
    global latest_image
    if latest_image is None:
        logger.info("No image available")
        raise HTTPException(status_code=404, detail="No image available")
    logger.info("Serving latest image")
    return {"image_data": latest_image}

@app.post("/capture_request")
async def request_capture():
    global capture_requested
    capture_requested = True
    logger.info("Capture requested")
    return {"status": "Capture requested"}

@app.get("/capture_request")
async def get_capture_request():
    global capture_requested
    if capture_requested:
        capture_requested = False
        logger.info("Capture request served")
        return {"capture": True}
    return {"capture": False}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting FastAPI on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)