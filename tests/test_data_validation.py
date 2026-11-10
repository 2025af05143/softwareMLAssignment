import pandas as pd
import pytest

from src.data_validation import DataValidation, DataValidationError


@pytest.fixture
def validator():
    return DataValidation()


@pytest.fixture
def valid_data():
    """Create a small valid dataset with the required schema."""
    return pd.DataFrame(
        {
            "date": ["1/1/2015", "1/2/2015"],
            "device": ["S1", "S2"],
            "failure": [0, 1],
            "metric1": [10, 20],
            "metric2": [100, 200],
            "metric3": [5, 6],
            "metric4": [15, 16],
            "metric5": [25, 26],
            "metric6": [35, 36],
            "metric7": [45, 46],
            "metric8": [45, 46],
            "metric9": [55, 56],
        }
    )


def test_valid_dataset_schema(validator, valid_data):
    report = validator.validate_data(valid_data)

    assert report["schema_valid"] is True
    assert report["missing_columns"] == []


def test_missing_values_detected(validator, valid_data):
    valid_data.loc[0, "metric1"] = None

    report = validator.validate_data(valid_data)

    assert report["total_missing_cells"] == 1
    assert report["data_completeness_percentage"] < 100


def test_duplicate_records_detected(validator, valid_data):
    duplicate_data = pd.concat([valid_data, valid_data.iloc[[0]]], ignore_index=True)

    report = validator.validate_data(duplicate_data)

    assert report["duplicate_records"] == 1


def test_invalid_target_values_detected(validator, valid_data):
    valid_data.loc[0, "failure"] = 2

    report = validator.validate_data(valid_data)

    assert report["invalid_target_records"] == 1
    assert report["target_validity_percentage"] == 50.0


def test_missing_required_column_rejected(validator, valid_data):
    invalid_data = valid_data.drop(columns=["metric1"])

    with pytest.raises(DataValidationError):
        validator.validate_data(invalid_data)


def test_empty_dataset_rejected(validator):
    empty_data = pd.DataFrame()

    with pytest.raises(DataValidationError):
        validator.validate_data(empty_data)


def test_invalid_date_detected(validator, valid_data):
    valid_data.loc[0, "date"] = "invalid-date"

    report = validator.validate_data(valid_data)

    assert report["invalid_date_records"] == 1
