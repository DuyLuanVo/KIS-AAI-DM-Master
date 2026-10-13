def test_list_batch(client, seeded):
    d = client.get("/api/v1/images/list", params={"batch_id": seeded["batch_id"]}).json()
    assert d["search_type"] == "browse"
    assert {r["file_name"] for r in d["results"]} == {f["file_name"] for f in seeded["files"]}


def test_get_item_and_404(client, seeded):
    key = seeded["files"][0]["key"]
    assert client.get("/api/v1/images/item", params={"key": key}).json()["key"] == key
    assert client.get("/api/v1/images/item", params={"key": "dataset/khong_co.png"}).status_code == 404
