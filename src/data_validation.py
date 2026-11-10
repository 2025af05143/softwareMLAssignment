import json
from pathlib import Path

import pandas as pd

from src.data_ingestion import DataIngestion
from src.logging_config import configure_logging

logger = configure_logging(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation failures."""


class DataValidation:

    REQUIRED_COLUMNS = DataIngestion.REQUIRED_COLUMNS

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate dataset structure and calculate data quality metrics.
        """

        try:
            logger.info("Starting data validation.")

            if not isinstance(df, pd.DataFrame):
                raise DataValidationError("Input must be a Pandas DataFrame.")

            if df.empty:
                raise DataValidationError("Cannot validate an empty dataset.")

            # 1. Schema validation
            missing_columns = [
                col for col in self.REQUIRED_COLUMNS if col not in df.columns
            ]

            schema_valid = len(missing_columns) == 0

            if not schema_valid:
                raise DataValidationError(
                    f"Missing required columns: {missing_columns}"
                )

            logger.info("Schema validation passed.")

            # 2. Missing value analysis
            missing_counts = df.isnull().sum()
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = int(missing_counts.sum())

            completeness = ((total_cells - missing_cells) / total_cells) * 100

            # 3. Duplicate analysis
            duplicate_count = int(df.duplicated().sum())

            duplicate_rate = (duplicate_count / len(df)) * 100

            # 4. Target variable validation
            target = df["failure"]

            valid_target_values = target.dropna().isin([0, 1])

            invalid_target_count = int((~valid_target_values).sum())

            missing_target_count = int(target.isnull().sum())

            valid_target_count = (
                len(target) - invalid_target_count - missing_target_count
            )

            target_validity = (valid_target_count / len(target)) * 100

            # 5. Date validation
            parsed_dates = pd.to_datetime(
                df["date"], format="%m/%d/%Y", errors="coerce"
            )

            invalid_date_count = int(parsed_dates.isnull().sum())

            # 6. Data type analysis
            data_types = {column: str(dtype) for column, dtype in df.dtypes.items()}

            # 7. Missing value report
            missing_value_report = {
                column: int(count) for column, count in missing_counts.items()
            }

            report = {
                "dataset_rows": int(df.shape[0]),
                "dataset_columns": int(df.shape[1]),
                "schema_valid": schema_valid,
                "missing_columns": missing_columns,
                "missing_value_counts": missing_value_report,
                "total_missing_cells": missing_cells,
                "data_completeness_percentage": round(completeness, 4),
                "duplicate_records": duplicate_count,
                "duplicate_rate_percentage": round(duplicate_rate, 4),
                "invalid_target_records": invalid_target_count,
                "missing_target_records": missing_target_count,
                "target_validity_percentage": round(target_validity, 4),
                "invalid_date_records": invalid_date_count,
                "data_types": data_types,
            }

            if missing_cells > 0:
                logger.warning("Dataset contains %d missing cells.", missing_cells)

            if duplicate_count > 0:
                logger.warning(
                    "Dataset contains %d duplicate records.", duplicate_count
                )

            if invalid_target_count > 0 or missing_target_count > 0:
                logger.warning("Target variable contains invalid or missing values.")

            if invalid_date_count > 0:
                logger.warning("Dataset contains %d invalid dates.", invalid_date_count)

            logger.info("Data validation completed successfully.")

            return report

        except DataValidationError:
            logger.exception("Data validation failed.")
            raise

        except Exception as error:
            logger.exception("Unexpected error during data validation.")

            raise DataValidationError("Unexpected data validation failure.") from error

    def save_report(self, report: dict) -> Path:
        """
        Save data quality report as JSON.
        """

        project_root = Path(__file__).resolve().parents[1]

        report_directory = project_root / "reports" / "quality_reports"

        report_directory.mkdir(parents=True, exist_ok=True)

        report_path = report_directory / "data_quality_report.json"

        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        logger.info("Data quality report saved to %s", report_path)

        return report_path
