import pandas as pd
import pytest

from src.model_training import ModelTraining


@pytest.fixture
def trainer(tmp_path):
    """Create a trainer with temporary output directories.

    This prevents automated tests from overwriting production models.
    """
    model_trainer = ModelTraining()

    model_trainer.models_directory = tmp_path / "models"
    model_trainer.reports_directory = tmp_path / "quality_reports"

    model_trainer.models_directory.mkdir(parents=True)
    model_trainer.reports_directory.mkdir(parents=True)

    return model_trainer


@pytest.fixture
def sample_data():
    """Create a synthetic dataset containing 60 devices."""
    records = []

    for device_idx in range(60):
        for observation in range(2):

            failure = int(device_idx < 12 and observation == 1)

            records.append(
                {
                    "device": f"D{device_idx:03d}",
                    "failure": failure,
                    "metric1": device_idx + observation * 0.1,
                    "metric2": device_idx % 7 + observation,
                    "metric3": (device_idx * 3) % 17 + observation,
                    "metric4": device_idx % 11,
                    "metric5": device_idx * 0.5 + observation,
                    "metric6": device_idx % 5 + observation,
                    "metric7": failure * 20 + device_idx % 3,
                    "metric9": device_idx % 13 + observation,
                    "day_of_week": device_idx % 7,
                    "month": 1 + device_idx % 11,
                    "day_of_month": 1 + device_idx % 28,
                }
            )

    return pd.DataFrame(records)


def test_device_group_split_has_no_overlap(trainer, sample_data):
    """Verify that devices do not appear in multiple datasets."""

    train_df, validation_df, test_df = trainer.device_group_split(sample_data)

    train_devices = set(train_df["device"])
    validation_devices = set(validation_df["device"])
    test_devices = set(test_df["device"])

    assert train_devices.isdisjoint(validation_devices)
    assert train_devices.isdisjoint(test_devices)
    assert validation_devices.isdisjoint(test_devices)

    assert len(train_df) + len(validation_df) + len(test_df) == len(sample_data)


def test_model_training_and_evaluation(
    trainer,
    sample_data,
):
    """Verify candidate model training and final test evaluation."""

    train_df, validation_df, test_df = trainer.device_group_split(sample_data)

    validation_results = trainer.train_and_evaluate_models(
        train_df,
        validation_df,
    )

    # Verify that all three candidate models were evaluated.
    expected_models = {
        "Dummy Classifier",
        "Logistic Regression",
        "Random Forest",
    }

    assert set(validation_results["Model"]) == expected_models

    # Verify that validation metrics are within valid ranges.
    assert validation_results["PR_AUC"].between(0, 1).all()
    assert validation_results["Best_F1"].between(0, 1).all()

    # Verify that trained model files were saved.
    assert (trainer.models_directory / "random_forest.joblib").exists()

    # Evaluate the selected model on test data.
    test_results = trainer.evaluate_selected_model(
        test_df,
        validation_results,
    )

    assert len(test_results) == 1
    assert test_results.iloc[0]["Model"] == "Random Forest"

    # Verify that the final production model was saved.
    assert (trainer.models_directory / "selected_model.joblib").exists()


def test_training_rejects_single_class_data(
    trainer,
    sample_data,
):
    """Verify that training rejects data containing only one class."""

    train_df, validation_df, _ = trainer.device_group_split(sample_data)

    single_class_train = train_df.copy()
    single_class_train["failure"] = 0

    with pytest.raises(ValueError):
        trainer.train_and_evaluate_models(
            single_class_train,
            validation_df,
        )
