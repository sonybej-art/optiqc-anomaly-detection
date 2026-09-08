import importlib
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"


@pytest.fixture
def client(monkeypatch):
    fake_engine_module = types.ModuleType("engine")

    class FakeAnomalyInferenceEngine:
        image_threshold = 0.2036
        device = "cpu"

        def __init__(self, checkpoint_path=None, device="cpu"):
            pass

        def inspect_image(self, image_bytes, custom_threshold=None):
            return {
                "image_score": 0.05,
                "threshold": 0.2036,
                "is_defective": False,
                "defect_type": "NONE",
                "latency_ms": 12.5,
                "heatmap_base64": "dummy_base64",
            }

    fake_engine_module.AnomalyInferenceEngine = FakeAnomalyInferenceEngine
    monkeypatch.setitem(sys.modules, "engine", fake_engine_module)
    monkeypatch.syspath_prepend(str(BACKEND_DIR))

    main = importlib.import_module("main")
    monkeypatch.chdir(BACKEND_DIR)
    return TestClient(main.app)


def test_health_returns_healthy_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_demo_is_served(client):
    response = client.get("/demo")

    assert response.status_code == 200


def test_inspect_rejects_non_image_upload(client):
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file must be an image."