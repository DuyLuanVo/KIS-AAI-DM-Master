"""
Cấu hình test: Qdrant in-memory, embedder "lite", lưu ảnh vào thư mục tạm.
Không cần Docker, GPU hay tải mô hình.
"""
import io
import os
import sys
import tempfile
from pathlib import Path

import pytest

TMP = tempfile.mkdtemp(prefix="kis_test_")
os.environ.update({
    "EMBEDDER_BACKEND": "lite",
    "QDRANT_MODE": "memory",
    "STORAGE_BACKEND": "local",
    "DATASET_DIR": str(Path(TMP) / "dataset"),
    "MEDIA_DIR": str(Path(TMP) / "media"),
    "SERVE_FRONTEND": "false",
})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402


def make_png(color, size=(64, 48), stripe=None) -> bytes:
    img = Image.new("RGB", size, color)
    if stripe:
        for x in range(0, size[0], 8):
            for y in range(size[1]):
                img.putpixel((x, y), stripe)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


SAMPLES = [
    ("red_tower.png", (180, 60, 40), "A weathered red brick Cham tower with a stepped pyramidal roof under a blue sky."),
    ("green_pagoda.png", (40, 140, 60), "A traditional pagoda with curved green tiled roofs surrounded by bamboo trees."),
    ("grey_church.png", (120, 120, 125), "A grey stone Gothic church with twin bell towers and a large rose window."),
    ("yellow_house.png", (220, 190, 60), "A yellow colonial villa with green wooden shutters and a tiled roof."),
]


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def seeded(client):
    files = [("files", (n, make_png(c), "image/png")) for n, c, _ in SAMPLES]
    import json

    r = client.post("/api/v1/ingest/upload", files=files,
                    data={"captions": json.dumps([cap for _, _, cap in SAMPLES]), "batch_name": ""})
    assert r.status_code == 200, r.text
    return r.json()
