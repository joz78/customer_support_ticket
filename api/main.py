from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import mlflow 
import os

app = FastAPI(title="Customer Ticket Classifier API", version="1.0")

# Dynamically find the model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path to /api
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "model (1).pkl"))

model = joblib.load(MODEL_PATH)

# Optional: Track logs locally
mlflow.set_tracking_uri("file:./mlruns")

class TicketInput(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Customer Ticket Classifier API is running 🚀"}

@app.post("/predict")
def predict(data: TicketInput):
    text = data.message

    with mlflow.start_run():
        mlflow.log_param("input_text", text)

        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        classes = model.classes_

        mlflow.log_param("prediction", prediction)

        for cls, prob in zip(classes, probabilities):
            mlflow.log_metric(f"confidence_{cls}", round(prob, 3))

        confidence = dict(zip(classes, probabilities.round(3)))

    return {
        "input": text,
        "predicted_class" : prediction,
        "confidence_scores" : confidence
    }
