import json
from pathlib import Path

import pandas as pd

from src.logging_config import configure_logging

logger = configure_logging(__name__)


class FeatureEngineeringError(Exception):
    """Raised when feature engineering fails."""


class FeatureEngineering:
    """Clean data and create model-ready candidate features."""

    METRIC_COLUMNS = [f"metric{i}" for i in range(1, 8)] + ["metric9"]

    FEATURE_COLUMNS = METRIC_COLUMNS + [
        "day_of_week",
        "month",
        "day_of_month",
    ]

    TARGET_COLUMN = "failure"

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.removed_duplicates = 0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset and generate candidate model features."""

        try:
            logger.info("Starting feature engineering.")

            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame.")

            if df.empty:
                raise FeatureEngineeringError("Input dataset is empty.")

            required_columns = ["date", "device", "failure"] + [
                f"metric{i}" for i in range(1, 10)
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise FeatureEngineeringError(
                    f"Missing required columns: {missing_columns}"
                )

            processed_df = df.copy()

            # Convert date using the source dataset's month/day/year format.
            processed_df["date"] = pd.to_datetime(
                processed_df["date"],
                format="%m/%d/%Y",
                errors="raise",
            )

            # Remove exact duplicate records only.
            original_rows = len(processed_df)
            processed_df = processed_df.drop_duplicates().copy()
            removed_duplicates = original_rows - len(processed_df)
            self.removed_duplicates = removed_duplicates

            logger.info(
                "Removed %d exact duplicate records.",
                removed_duplicates,
            )

            # Ensure numerical metrics are numeric.
            all_metrics = [f"metric{i}" for i in range(1, 10)]

            for column in all_metrics:
                processed_df[column] = pd.to_numeric(
                    processed_df[column],
                    errors="raise",
                )

            # Create date-based features.
            processed_df["day_of_week"] = processed_df["date"].dt.dayofweek

            processed_df["month"] = processed_df["date"].dt.month

            processed_df["day_of_month"] = processed_df["date"].dt.day

            # Check target values.
            if not processed_df[self.TARGET_COLUMN].isin([0, 1]).all():
                raise FeatureEngineeringError(
                    "Target column must contain only 0 and 1."
                )

            # Check that candidate features contain no missing values.
            if processed_df[self.FEATURE_COLUMNS].isnull().any().any():
                raise FeatureEngineeringError("Missing values found in model features.")

            logger.info(
                "Feature engineering completed. Rows: %d, Features: %d",
                len(processed_df),
                len(self.FEATURE_COLUMNS),
            )

            logger.info(
                "Failure distribution: %s",
                processed_df[self.TARGET_COLUMN].value_counts().to_dict(),
            )

            return processed_df

        except Exception as error:
            logger.exception("Feature engineering failed: %s", error)

            if isinstance(error, FeatureEngineeringError):
                raise

            raise FeatureEngineeringError(
                "Could not complete feature engineering."
            ) from error

    def save_processed_data(self, df: pd.DataFrame) -> Path:
        """Save processed data and feature metadata."""

        try:
            output_directory = self.project_root / "data" / "processed"
            output_directory.mkdir(parents=True, exist_ok=True)

            output_path = output_directory / "processed_predictive_maintenance.csv"

            df.to_csv(output_path, index=False)

            metadata = {
                "dataset_rows": int(len(df)),
                "feature_columns": self.FEATURE_COLUMNS,
                "target_column": self.TARGET_COLUMN,
                "excluded_columns": [
                    "device",
                    "metric8",
                    "date",
                ],
                "removed_exact_duplicates": self.removed_duplicates,
            }

            metadata_path = output_directory / "feature_metadata.json"

            with metadata_path.open("w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=4)

            logger.info("Processed dataset saved: %s", output_path)
            logger.info("Feature metadata saved: %s", metadata_path)

            return output_path

        except OSError as error:
            logger.exception("Could not save processed data: %s", error)
            raise


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    raw_df = ingestion.load_data()

    engineer = FeatureEngineering()

    processed_df = engineer.transform(raw_df)
    output_path = engineer.save_processed_data(processed_df)

    print("\n--- FEATURE ENGINEERING SUMMARY ---")
    print("Original rows:", len(raw_df))
    print("Processed rows:", len(processed_df))
    print("Removed duplicates:", len(raw_df) - len(processed_df))
    print("Number of candidate features:", len(engineer.FEATURE_COLUMNS))
    print("Candidate features:", engineer.FEATURE_COLUMNS)
    print("\nTarget distribution:")
    print(processed_df["failure"].value_counts())
    print("\nDate range:")
    print(processed_df["date"].min(), "to", processed_df["date"].max())
    print("\nSaved dataset:", output_path)
