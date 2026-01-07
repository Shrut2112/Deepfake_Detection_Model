import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_accepts_input_and_returns_output():
    with open("tests/assets/test.jpg", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    assert response.json() is not None
