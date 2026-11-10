from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.logging_config import configure_logging
from src.model_inference import ModelInference, ModelInferenceError

# Configure application logging
logger = configure_logging(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "REST API for predicting equipment failures "
        "using a trained Random Forest model."
    ),
    version="1.0.0",
)


# Load the trained model when the API starts
try:
    inference_engine = ModelInference()
    MODEL_LOADED = True
    logger.info("Predictive maintenance model initialized.")

except ModelInferenceError:
    inference_engine = None
    MODEL_LOADED = False
    logger.exception("Could not initialize predictive maintenance model.")


# Input schema
class MachineInput(BaseModel):
    metric1: float
    metric2: float
    metric3: float
    metric4: float
    metric5: float
    metric6: float
    metric7: float
    metric9: float

    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    day_of_month: int = Field(ge=1, le=31)


# Request schema
class PredictionRequest(BaseModel):
    observations: List[MachineInput] = Field(min_length=1)


# Individual prediction response
class PredictionResult(BaseModel):
    failure_probability: float
    predicted_failure: int


# API response schema
class PredictionResponse(BaseModel):
    model_name: str
    classification_threshold: float
    total_observations: int
    predictions: List[PredictionResult]


# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Predictive Maintenance API is running.",
        "documentation": "/docs",
    }


# Health endpoint
@app.get("/health")
def health_check():
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Predictive maintenance model is unavailable.",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": "Random Forest",
    }


# Prediction endpoint
@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_failure(request: PredictionRequest):

    if not MODEL_LOADED or inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is unavailable.",
        )

    try:
        # Convert request observations into a DataFrame
        input_data = pd.DataFrame(
            [observation.model_dump() for observation in request.observations]
        )

        logger.info(
            "Received prediction request with %d observations.",
            len(input_data),
        )

        # Generate predictions
        predictions = inference_engine.predict(input_data)

        # Construct response
        prediction_records = [
            PredictionResult(
                failure_probability=float(row["failure_probability"]),
                predicted_failure=int(row["predicted_failure"]),
            )
            for _, row in predictions.iterrows()
        ]

        return PredictionResponse(
            model_name="Random Forest",
            classification_threshold=inference_engine.threshold,
            total_observations=len(input_data),
            predictions=prediction_records,
        )

    except ModelInferenceError as error:
        logger.error("Prediction input or inference error: %s", error)

        raise HTTPException(
            status_code=400,
            detail="Invalid prediction input.",
        ) from error

    except Exception as error:
        logger.exception("Unexpected API prediction error.")

        raise HTTPException(
            status_code=500,
            detail="Internal prediction service error.",
        ) from error
