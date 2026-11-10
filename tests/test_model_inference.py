import pandas as pd
import pytest

from src.model_inference import ModelInference, ModelInferenceError


@pytest.fixture
def inference():
    """Load the trained production model."""
    return ModelInference()


@pytest.fixture
def valid_input(inference):
    """Load five valid records from the processed dataset."""
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    data_path = (
        project_root / "data" / "processed" / "processed_predictive_maintenance.csv"
    )

    df = pd.read_csv(data_path)
    return df[inference.feature_columns].head(5)


def test_prediction_output_shape(inference, valid_input):
    result = inference.predict(valid_input)

    assert len(result) == len(valid_input)
    assert list(result.columns) == [
        "failure_probability",
        "predicted_failure",
    ]


def test_probability_range(inference, valid_input):
    result = inference.predict(valid_input)

    assert result["failure_probability"].between(0, 1).all()


def test_predictions_are_binary(inference, valid_input):
    result = inference.predict(valid_input)

    assert set(result["predicted_failure"].unique()).issubset({0, 1})


def test_missing_feature_rejected(inference, valid_input):
    invalid_input = valid_input.drop(columns=[inference.feature_columns[0]])

    with pytest.raises(ModelInferenceError):
        inference.predict(invalid_input)


def test_empty_input_rejected(inference):
    empty_input = pd.DataFrame(columns=inference.feature_columns)

    with pytest.raises(ModelInferenceError):
        inference.predict(empty_input)
