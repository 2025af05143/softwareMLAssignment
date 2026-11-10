from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def valid_prediction_payload():
    return {
        "observations": [
            {
                "metric1": 10,
                "metric2": 100,
                "metric3": 5,
                "metric4": 15,
                "metric5": 25,
                "metric6": 35,
                "metric7": 45,
                "metric9": 55,
                "day_of_week": 2,
                "month": 3,
                "day_of_month": 15,
            }
        ]
    }


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "Predictive Maintenance API" in response.json()["message"]


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json=valid_prediction_payload(),
    )

    assert response.status_code == 200

    result = response.json()

    assert result["model_name"] == "Random Forest"
    assert result["total_observations"] == 1
    assert len(result["predictions"]) == 1

    prediction = result["predictions"][0]

    assert 0 <= prediction["failure_probability"] <= 1
    assert prediction["predicted_failure"] in [0, 1]


def test_prediction_rejects_missing_features():
    payload = valid_prediction_payload()
    del payload["observations"][0]["metric1"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_rejects_empty_observations():
    response = client.post(
        "/predict",
        json={"observations": []},
    )

    assert response.status_code == 422
