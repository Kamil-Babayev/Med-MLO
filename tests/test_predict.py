import io
from PIL import Image


def make_test_image() -> bytes:
    img = Image.new("RGB", (64, 64), color=(128, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_predict_valid_image(client):
    image_bytes = make_test_image()
    response = client.post(
        "/predict",
        files={"file": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200

    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["filename"] == "test.png"


def test_predict_invalid_file_type(client):
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 422


def test_predict_logs_to_db(client):
    image_bytes = make_test_image()
    client.post(
        "/predict",
        files={"file": ("logged.png", image_bytes, "image/png")},
    )

    history = client.get("/history").json()
    filenames = [r["filename"] for r in history["results"]]
    assert "logged.png" in filenames