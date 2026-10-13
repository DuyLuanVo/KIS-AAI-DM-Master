import base64
import json

from conftest import SAMPLES, make_png


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"


def test_ingest_indexes_all(seeded):
    assert seeded["indexed_count"] == len(SAMPLES)
    assert seeded["skipped_count"] == 0
    assert all(f["status"] == "indexed" for f in seeded["files"])


def test_ingest_requires_caption(client):
    files = [("files", ("a.png", make_png((1, 2, 3)), "image/png")), ("files", ("b.png", make_png((9, 9, 9)), "image/png"))]
    r = client.post("/api/v1/ingest/upload", files=files,
                    data={"captions": json.dumps({"a.png": "A small shrine with a red roof near a lake."})})
    body = r.json()
    assert body["indexed_count"] == 1
    skipped = [f for f in body["files"] if f["status"] == "skipped"]
    assert skipped[0]["file_name"] == "b.png" and "caption" in skipped[0]["reason"].lower()
    client.delete(f"/api/v1/ingest/batches/{body['batch_id']}")


def test_caption_count_mismatch_rejected(client):
    files = [("files", ("a.png", make_png((1, 2, 3)), "image/png"))]
    r = client.post("/api/v1/ingest/upload", files=files, data={"captions": json.dumps(["x" * 20, "y" * 20])})
    assert r.status_code == 400


def test_invalid_image_skipped(client):
    files = [("files", ("bad.png", b"not an image at all" * 10, "image/png"))]
    r = client.post("/api/v1/ingest/upload", files=files, data={"captions": json.dumps(["A valid caption text here."])})
    assert r.json()["indexed_count"] == 0


def test_text_search_queries_captions(client, seeded):
    r = client.post("/api/v1/search/text", json={"query": "brick tower", "limit": 10})
    assert r.status_code == 200
    d = r.json()
    assert d["search_type"] == "text"
    top = d["results"][0]
    assert top["file_name"] == "red_tower.png"
    assert set(top["matched_terms"]) == {"brick", "tower"}
    assert "keyword" in top["match_sources"]


def test_text_search_plural_matches_singular(client, seeded):
    d = client.post("/api/v1/search/text", json={"query": "pagodas bamboo", "limit": 5}).json()
    assert d["results"][0]["file_name"] == "green_pagoda.png"


def test_image_search_returns_most_similar(client, seeded):
    b64 = base64.b64encode(make_png((182, 62, 42))).decode()
    d = client.post("/api/v1/search/image", json={"image_base64": "data:image/png;base64," + b64, "limit": 3}).json()
    assert d["search_type"] == "image"
    assert d["results"][0]["file_name"] == "red_tower.png"
    assert d["results"][0]["match_sources"] == ["image"]


def test_image_upload_search(client, seeded):
    r = client.post("/api/v1/search/image/upload", files={"file": ("q.png", make_png((40, 140, 60)), "image/png")})
    assert r.json()["results"][0]["file_name"] == "green_pagoda.png"


def test_image_search_rejects_non_image(client):
    b64 = base64.b64encode(b"hello world, not an image").decode()
    assert client.post("/api/v1/search/image", json={"image_base64": b64}).status_code == 400


def test_similar_excludes_itself(client, seeded):
    key = next(f["key"] for f in seeded["files"] if f["file_name"] == "yellow_house.png")
    d = client.get("/api/v1/images/similar", params={"key": key, "limit": 10}).json()
    assert d["search_type"] == "similar"
    assert key not in [r["key"] for r in d["results"]]


def test_media_served(client, seeded):
    url = seeded["files"][0]["image_url"]
    r = client.get(url)
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/")


def test_stats(client, seeded):
    s = client.get("/api/v1/stats").json()
    assert s["total_images"] >= len(SAMPLES)
    assert s["by_source"].get("upload", 0) >= len(SAMPLES)


def test_delete_batch(client):
    files = [("files", ("tmp.png", make_png((5, 5, 200)), "image/png"))]
    body = client.post("/api/v1/ingest/upload", files=files,
                       data={"captions": json.dumps(["A blue temporary test image of a gate."])}).json()
    before = client.get("/api/v1/stats").json()["total_images"]
    r = client.delete(f"/api/v1/ingest/batches/{body['batch_id']}")
    assert r.json()["deleted"] == 1
    assert client.get("/api/v1/stats").json()["total_images"] == before - 1
    assert client.get(body["files"][0]["image_url"]).status_code == 404
