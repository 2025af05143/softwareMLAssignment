from pathlib import Path

import pandas as pd

from src.logging_config import configure_logging

logger = configure_logging(__name__)


class DataIngestionError(Exception):
    """Custom exception for data ingestion failures."""


class DataIngestion:

    REQUIRED_COLUMNS = [
        "date",
        "device",
        "failure",
        "metric1",
        "metric2",
        "metric3",
        "metric4",
        "metric5",
        "metric6",
        "metric7",
        "metric8",
        "metric9",
    ]

    def __init__(self, file_path=None):

        project_root = Path(__file__).resolve().parents[1]

        if file_path is None:
            file_path = (
                project_root / "data" / "raw" / "predictive_maintenance_dataset.csv"
            )

        self.file_path = Path(file_path)

    def load_data(self) -> pd.DataFrame:
        """
        Load the predictive maintenance dataset.
        """

        try:
            logger.info("Starting dataset ingestion: %s", self.file_path)

            if not self.file_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.file_path}")

            if self.file_path.suffix.lower() != ".csv":
                raise ValueError("Unsupported file format. Expected CSV.")

            df = pd.read_csv(self.file_path)

            if df.empty:
                raise DataIngestionError("Dataset is empty.")

            missing_columns = [
                column for column in self.REQUIRED_COLUMNS if column not in df.columns
            ]

            if missing_columns:
                raise DataIngestionError(f"Missing required columns: {missing_columns}")

            logger.info(
                "Dataset loaded successfully. Rows: %d, Columns: %d",
                df.shape[0],
                df.shape[1],
            )

            logger.info("Dataset ingestion completed successfully.")

            return df

        except FileNotFoundError:
            logger.error("Dataset file does not exist: %s", self.file_path)
            raise

        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            logger.error("Dataset reading or format error: %s", error)
            raise

        except DataIngestionError as error:
            logger.error("Data ingestion validation failed: %s", error)
            raise

        except Exception as error:
            logger.exception("Unexpected error during data ingestion: %s", error)
            raise DataIngestionError("Unexpected data ingestion failure.") from error
