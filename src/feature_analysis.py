import json
from pathlib import Path

import pandas as pd

from src.data_ingestion import DataIngestion
from src.logging_config import configure_logging

logger = configure_logging(__name__)


def analyze_features(df: pd.DataFrame) -> dict:
    """Analyze metric distributions and their relationship with failure."""

    try:
        logger.info("Starting feature analysis.")

        metric_columns = [f"metric{i}" for i in range(1, 10)]

        missing_metrics = [
            column for column in metric_columns if column not in df.columns
        ]

        if missing_metrics:
            raise ValueError(f"Missing metric columns: {missing_metrics}")

        results = {}

        # Percentage of zero values in each metric
        results["zero_percentage"] = (
            ((df[metric_columns] == 0).mean() * 100).round(2).to_dict()
        )

        # Check whether metric7 and metric8 are identical
        results["metric7_metric8_identical"] = bool(df["metric7"].equals(df["metric8"]))

        # Correlation with the target
        results["correlation_with_failure"] = (
            df[metric_columns + ["failure"]]
            .corr(numeric_only=True)["failure"]
            .drop("failure")
            .round(5)
            .to_dict()
        )

        # Device-date duplicate combinations
        results["duplicate_device_date_combinations"] = int(
            df.duplicated(subset=["device", "date"]).sum()
        )

        # Number of records per device
        results["records_per_device"] = {
            "minimum": int(df.groupby("device").size().min()),
            "maximum": int(df.groupby("device").size().max()),
            "average": round(float(df.groupby("device").size().mean()), 2),
        }

        logger.info("Feature analysis completed successfully.")

        return results

    except Exception as error:
        logger.exception("Feature analysis failed: %s", error)
        raise


def save_analysis(results: dict) -> Path:
    """Save feature analysis results as JSON."""

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "reports" / "quality_reports" / "feature_analysis.json"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)

        logger.info("Feature analysis report saved: %s", output_path)

        return output_path

    except OSError as error:
        logger.exception("Could not save feature analysis report: %s", error)
        raise


if __name__ == "__main__":
    dataframe = DataIngestion().load_data()

    analysis = analyze_features(dataframe)
    report_path = save_analysis(analysis)

    print("\n--- FEATURE ANALYSIS ---")

    for key, value in analysis.items():
        print(f"\n{key}:")
        print(value)

    print(f"\nReport saved: {report_path}")
