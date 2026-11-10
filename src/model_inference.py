from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.logging_config import configure_logging

logger = configure_logging(__name__)


class ModelInferenceError(Exception):
    """Custom exception for model inference failures."""


class ModelInference:
    """
    Production-oriented inference module for predictive maintenance.
    """

    def __init__(self, model_path=None):

        self.project_root = Path(__file__).resolve().parents[1]

        if model_path is None:
            model_path = self.project_root / "models" / "selected_model.joblib"

        self.model_path = Path(model_path)

        self.model = None
        self.threshold = None
        self.feature_columns = None

        self.load_model()

    def load_model(self):
        """Load the trained model, threshold and feature metadata."""

        try:
            logger.info(
                "Loading production model: %s",
                self.model_path,
            )

            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            model_package = joblib.load(self.model_path)

            required_keys = [
                "model",
                "threshold",
                "feature_columns",
                "model_name",
            ]

            for key in required_keys:
                if key not in model_package:
                    raise ValueError(f"Model package is missing: {key}")

            self.model = model_package["model"]
            self.threshold = float(model_package["threshold"])
            self.feature_columns = model_package["feature_columns"]

            if not 0 <= self.threshold <= 1:
                raise ValueError("Invalid classification threshold.")

            if not self.feature_columns:
                raise ValueError("Feature list is empty.")

            logger.info(
                "Model loaded successfully: %s",
                model_package["model_name"],
            )

        except Exception as error:
            logger.exception("Model loading failed.")

            raise ModelInferenceError("Could not load production model.") from error

    def validate_input(self, input_data):
        """Validate incoming sensor observations."""

        try:
            if not isinstance(input_data, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")

            if input_data.empty:
                raise ValueError("Input data is empty.")

            missing_columns = [
                column
                for column in self.feature_columns
                if column not in input_data.columns
            ]

            if missing_columns:
                raise ValueError(f"Missing features: {missing_columns}")

            features = input_data[self.feature_columns].copy()

            # Ensure all model features are numeric.
            features = features.apply(
                pd.to_numeric,
                errors="coerce",
            )

            if features.isnull().any().any():
                raise ValueError(
                    "Input contains missing or non-numeric feature values."
                )

            if not np.isfinite(features.to_numpy()).all():
                raise ValueError("Input contains infinite feature values.")

            return features

        except Exception as error:
            logger.exception("Input validation failed.")

            raise ModelInferenceError("Invalid inference input.") from error

    def predict(self, input_data):
        """
        Generate failure probabilities and binary predictions.

        Output:
        failure_probability: predicted probability of failure
        predicted_failure: 0 or 1
        """

        try:
            logger.info(
                "Starting inference for %d observations.",
                len(input_data),
            )

            features = self.validate_input(input_data)

            probabilities = self.model.predict_proba(features)[:, 1]

            predictions = (probabilities >= self.threshold).astype(int)

            results = pd.DataFrame(
                {
                    "failure_probability": probabilities,
                    "predicted_failure": predictions,
                }
            )

            logger.info(
                "Inference completed successfully. " "Predicted failures: %d",
                int(predictions.sum()),
            )

            return results

        except ModelInferenceError:
            raise

        except Exception as error:
            logger.exception("Unexpected inference error.")

            raise ModelInferenceError("Model prediction failed.") from error


if __name__ == "__main__":

    inference = ModelInference()

    # Demonstration using a few records from the processed dataset.
    data_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
        / "processed_predictive_maintenance.csv"
    )

    sample_data = pd.read_csv(data_path)

    sample_features = sample_data[inference.feature_columns].head(5)

    predictions = inference.predict(sample_features)

    print("\nINFERENCE DEMONSTRATION")
    print("=" * 50)
    print(predictions.to_string(index=False))
