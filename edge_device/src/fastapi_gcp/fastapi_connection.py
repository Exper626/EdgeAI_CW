from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import logging
from google.cloud import firestore
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API2_for_audio")

db = firestore.Client(project="cloudsqltest-457110") #changes according to our projectName

app = FastAPI()


class Prediction(BaseModel):
    timestamp: str
    noise_level: str
    person_count: int

    @validator("noise_level")
    def validate_noise_level(cls, v):
        if v not in ["High", "Normal"]:
            raise ValueError("noise_level must be 'High' or 'Normal'")
        return v


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

    # Save to Firestore only if human speech detected and person_count >= 2
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


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting FastAPI on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)