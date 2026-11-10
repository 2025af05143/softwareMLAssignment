from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.logging_config import configure_logging

logger = configure_logging(__name__)


class ModelTraining:
    """
    Loads processed data, creates device-level splits,
    trains classification models and evaluates them.
    """

    FEATURE_COLUMNS = [
        "metric1",
        "metric2",
        "metric3",
        "metric4",
        "metric5",
        "metric6",
        "metric7",
        "metric9",
        "day_of_week",
        "month",
        "day_of_month",
    ]

    TARGET_COLUMN = "failure"

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.models_directory = self.project_root / "models"
        self.reports_directory = self.project_root / "reports" / "quality_reports"

        self.models_directory.mkdir(parents=True, exist_ok=True)
        self.reports_directory.mkdir(parents=True, exist_ok=True)

    def load_processed_data(self):
        """Load the processed predictive maintenance dataset."""

        data_path = (
            self.project_root
            / "data"
            / "processed"
            / "processed_predictive_maintenance.csv"
        )

        try:
            logger.info("Loading processed dataset: %s", data_path)

            if not data_path.exists():
                raise FileNotFoundError(f"Processed dataset not found: {data_path}")

            df = pd.read_csv(data_path, parse_dates=["date"])

            if df.empty:
                raise ValueError("Processed dataset is empty.")

            required_columns = self.FEATURE_COLUMNS + [
                self.TARGET_COLUMN,
                "device",
                "date",
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            if df[self.FEATURE_COLUMNS].isnull().any().any():
                raise ValueError("Feature columns contain missing values.")

            if not df[self.TARGET_COLUMN].isin([0, 1]).all():
                raise ValueError("Target must contain only 0 and 1.")

            logger.info(
                "Dataset loaded successfully. Rows: %d, Columns: %d",
                df.shape[0],
                df.shape[1],
            )

            return df

        except Exception:
            logger.exception("Could not load processed dataset.")
            raise

    def device_group_split(self, df):
        """Create stratified training, validation and test splits by device."""

        try:
            logger.info("Starting stratified device-level split.")

            if df["device"].isnull().any():
                raise ValueError("Device column contains missing values.")

            device_summary = (
                df.groupby("device", as_index=False)["failure"]
                .max()
                .rename(columns={"failure": "has_failure"})
                .sort_values("device")
            )

            train_devices, temporary_devices = train_test_split(
                device_summary,
                test_size=0.30,
                random_state=42,
                stratify=device_summary["has_failure"],
            )

            validation_devices, test_devices = train_test_split(
                temporary_devices,
                test_size=0.50,
                random_state=42,
                stratify=temporary_devices["has_failure"],
            )

            train_df = df[df["device"].isin(train_devices["device"])].copy()

            validation_df = df[df["device"].isin(validation_devices["device"])].copy()

            test_df = df[df["device"].isin(test_devices["device"])].copy()

            train_ids = set(train_df["device"])
            validation_ids = set(validation_df["device"])
            test_ids = set(test_df["device"])

            if train_ids & validation_ids:
                raise ValueError("Training-validation device overlap detected.")

            if train_ids & test_ids:
                raise ValueError("Training-test device overlap detected.")

            if validation_ids & test_ids:
                raise ValueError("Validation-test device overlap detected.")

            logger.info("Device-level split completed successfully.")

            return train_df, validation_df, test_df

        except Exception:
            logger.exception("Device-level splitting failed.")
            raise

    def display_split_summary(self, datasets):
        """Display the summary of each dataset."""

        names = ["Training", "Validation", "Test"]

        print("\n" + "=" * 55)
        print("STRATIFIED DEVICE-LEVEL SPLIT SUMMARY")
        print("=" * 55)

        for name, data in zip(names, datasets):
            print(f"\n{name} dataset")
            print("-" * 40)
            print("Records:", len(data))
            print("Unique devices:", data["device"].nunique())
            print("Failure records:", int(data["failure"].sum()))
            print(
                "Non-failure records:",
                int((data["failure"] == 0).sum()),
            )
            print(
                "Failure percentage:",
                round(data["failure"].mean() * 100, 4),
                "%",
            )
            print("Start date:", data["date"].min())
            print("End date:", data["date"].max())

        train_ids = set(datasets[0]["device"])
        validation_ids = set(datasets[1]["device"])
        test_ids = set(datasets[2]["device"])

        print("\nDevice overlap checks:")
        print("Training vs Validation:", len(train_ids & validation_ids))
        print("Training vs Test:", len(train_ids & test_ids))
        print("Validation vs Test:", len(validation_ids & test_ids))

    def train_and_evaluate_models(self, train_df, validation_df):
        """
        Train candidate models and analyse multiple classification
        thresholds using validation data.
        """

        try:
            logger.info("Starting model training and threshold analysis.")

            X_train = train_df[self.FEATURE_COLUMNS]
            y_train = train_df[self.TARGET_COLUMN]

            X_validation = validation_df[self.FEATURE_COLUMNS]
            y_validation = validation_df[self.TARGET_COLUMN]

            if y_train.nunique() < 2 or y_validation.nunique() < 2:
                raise ValueError("Training and validation must contain both classes.")

            models = {
                "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
                "Logistic Regression": Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(
                                class_weight="balanced",
                                max_iter=2000,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            }

            validation_results = []
            threshold_results = []

            for model_name, model in models.items():

                logger.info("Training model: %s", model_name)

                model.fit(X_train, y_train)

                probabilities = model.predict_proba(X_validation)[:, 1]

                # PR-AUC is calculated independently of the threshold.
                pr_auc = average_precision_score(
                    y_validation,
                    probabilities,
                )

                # Evaluate the original threshold of 0.50.
                default_predictions = (probabilities >= 0.50).astype(int)

                default_precision = precision_score(
                    y_validation,
                    default_predictions,
                    zero_division=0,
                )

                default_recall = recall_score(
                    y_validation,
                    default_predictions,
                    zero_division=0,
                )

                default_f1 = f1_score(
                    y_validation,
                    default_predictions,
                    zero_division=0,
                )

                # Calculate precision and recall at all possible
                # classification thresholds.
                precisions, recalls, thresholds = precision_recall_curve(
                    y_validation,
                    probabilities,
                )

                # The last precision/recall point has no corresponding
                # threshold, so exclude it for threshold-based analysis.
                f1_values = (
                    2
                    * precisions[:-1]
                    * recalls[:-1]
                    / (precisions[:-1] + recalls[:-1] + 1e-12)
                )

                threshold_df = pd.DataFrame(
                    {
                        "Model": model_name,
                        "Threshold": thresholds,
                        "Precision": precisions[:-1],
                        "Recall": recalls[:-1],
                        "F1_Score": f1_values,
                    }
                )

                threshold_results.append(threshold_df)

                # Identify the threshold with the highest validation F1.
                best_index = f1_values.argmax()
                best_threshold = thresholds[best_index]

                best_predictions = (probabilities >= best_threshold).astype(int)

                tn, fp, fn, tp = confusion_matrix(
                    y_validation,
                    best_predictions,
                    labels=[0, 1],
                ).ravel()

                best_precision = precision_score(
                    y_validation,
                    best_predictions,
                    zero_division=0,
                )

                best_recall = recall_score(
                    y_validation,
                    best_predictions,
                    zero_division=0,
                )

                best_f1 = f1_score(
                    y_validation,
                    best_predictions,
                    zero_division=0,
                )

                validation_results.append(
                    {
                        "Model": model_name,
                        "PR_AUC": pr_auc,
                        "Default_Threshold": 0.50,
                        "Default_Precision": default_precision,
                        "Default_Recall": default_recall,
                        "Default_F1": default_f1,
                        "Best_Threshold": best_threshold,
                        "Best_Precision": best_precision,
                        "Best_Recall": best_recall,
                        "Best_F1": best_f1,
                        "True_Positives": tp,
                        "False_Positives": fp,
                        "False_Negatives": fn,
                        "True_Negatives": tn,
                    }
                )

                # Save the trained model.
                model_filename = model_name.lower().replace(" ", "_") + ".joblib"

                model_path = self.models_directory / model_filename
                joblib.dump(model, model_path)

                print(f"\n{model_name}")
                print("-" * 45)
                print(f"PR-AUC: {pr_auc:.6f}")

                print("\nAt default threshold 0.50:")
                print(f"Precision: {default_precision:.4f}")
                print(f"Recall:    {default_recall:.4f}")
                print(f"F1-score:  {default_f1:.4f}")

                print("\nAt best validation F1 threshold:")
                print(f"Threshold: {best_threshold:.6f}")
                print(f"Precision: {best_precision:.4f}")
                print(f"Recall:    {best_recall:.4f}")
                print(f"F1-score:  {best_f1:.4f}")
                print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

            # Save model comparison.
            results_df = pd.DataFrame(validation_results)

            results_path = self.reports_directory / "model_validation_results.csv"

            results_df.to_csv(results_path, index=False)

            # Save all evaluated threshold combinations.
            all_thresholds = pd.concat(
                threshold_results,
                ignore_index=True,
            )

            threshold_path = self.reports_directory / "threshold_analysis.csv"

            all_thresholds.to_csv(threshold_path, index=False)

            print("\n" + "=" * 65)
            print("MODEL COMPARISON - VALIDATION DATA")
            print("=" * 65)

            print(
                results_df[
                    [
                        "Model",
                        "PR_AUC",
                        "Best_Threshold",
                        "Best_Precision",
                        "Best_Recall",
                        "Best_F1",
                    ]
                ].to_string(index=False)
            )

            print("\nTOP 5 THRESHOLDS BY F1-SCORE FOR EACH MODEL")

            for model_name in models:

                model_thresholds = all_thresholds[
                    all_thresholds["Model"] == model_name
                ].nlargest(5, "F1_Score")

                print(f"\n{model_name}")
                print(
                    model_thresholds[
                        ["Threshold", "Precision", "Recall", "F1_Score"]
                    ].to_string(index=False)
                )

            print("\nValidation results saved:", results_path)
            print("Threshold analysis saved:", threshold_path)

            logger.info("Threshold analysis completed successfully.")

            return results_df

        except Exception:
            logger.exception("Model training or threshold analysis failed.")
            raise

    def evaluate_selected_model(self, test_df, validation_results):
        """
        Evaluate the selected Random Forest model on untouched
        test data using the threshold selected on validation data.
        """

        try:
            logger.info("Starting final test evaluation.")

            # Retrieve the threshold selected using validation data.
            selected_result = validation_results[
                validation_results["Model"] == "Random Forest"
            ]

            if selected_result.empty:
                raise ValueError("Random Forest results not found.")

            threshold = float(selected_result.iloc[0]["Best_Threshold"])

            # Load the trained Random Forest model.
            model_path = self.models_directory / "random_forest.joblib"

            if not model_path.exists():
                raise FileNotFoundError(f"Random Forest model not found: {model_path}")

            model = joblib.load(model_path)

            X_test = test_df[self.FEATURE_COLUMNS]
            y_test = test_df[self.TARGET_COLUMN]

            # Generate test probabilities.
            probabilities = model.predict_proba(X_test)[:, 1]

            # Apply the previously selected validation threshold.
            predictions = (probabilities >= threshold).astype(int)

            # Calculate final test metrics.
            precision = precision_score(y_test, predictions, zero_division=0)

            recall = recall_score(y_test, predictions, zero_division=0)

            f1 = f1_score(y_test, predictions, zero_division=0)

            pr_auc = average_precision_score(y_test, probabilities)

            tn, fp, fn, tp = confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1],
            ).ravel()

            print("\n" + "=" * 60)
            print("FINAL RANDOM FOREST - TEST RESULTS")
            print("=" * 60)

            print("Classification threshold:", round(threshold, 6))
            print("Test records:", len(test_df))
            print("Actual failures:", int(y_test.sum()))

            print("\nModel quality metrics")
            print("-" * 40)
            print(f"Precision : {precision:.4f}")
            print(f"Recall    : {recall:.4f}")
            print(f"F1-score  : {f1:.4f}")
            print(f"PR-AUC    : {pr_auc:.4f}")

            print("\nConfusion Matrix")
            print("-" * 40)
            print(f"True Positives : {tp}")
            print(f"False Positives: {fp}")
            print(f"False Negatives: {fn}")
            print(f"True Negatives : {tn}")

            # Save the final model together with its threshold
            # and feature list for consistent future inference.
            final_model_path = self.models_directory / "selected_model.joblib"

            joblib.dump(
                {
                    "model": model,
                    "threshold": threshold,
                    "feature_columns": self.FEATURE_COLUMNS,
                    "model_name": "Random Forest",
                },
                final_model_path,
            )

            # Save final test results.
            test_results = pd.DataFrame(
                [
                    {
                        "Model": "Random Forest",
                        "Threshold": threshold,
                        "Precision": precision,
                        "Recall": recall,
                        "F1_Score": f1,
                        "PR_AUC": pr_auc,
                        "True_Positives": tp,
                        "False_Positives": fp,
                        "False_Negatives": fn,
                        "True_Negatives": tn,
                    }
                ]
            )

            report_path = self.reports_directory / "final_test_results.csv"

            test_results.to_csv(report_path, index=False)

            print("\nFinal model saved:", final_model_path)
            print("Test report saved:", report_path)

            logger.info("Final test evaluation completed.")

            return test_results

        except Exception:
            logger.exception("Final test evaluation failed.")
            raise


if __name__ == "__main__":

    trainer = ModelTraining()

    dataframe = trainer.load_processed_data()

    datasets = trainer.device_group_split(dataframe)

    trainer.display_split_summary(datasets)

    train_df, validation_df, test_df = datasets

    # Train and compare candidate models.
    validation_results = trainer.train_and_evaluate_models(
        train_df,
        validation_df,
    )

    # Evaluate the selected model on untouched test data.
    trainer.evaluate_selected_model(
        test_df,
        validation_results,
    )
