def test_history_empty(client):
    response = client.get("/history")
    assert response.status_code == 200

    data = response.json()
    assert "total" in data
    assert "results" in data
    assert isinstance(data["results"], list)


def test_history_pagination(client):
    response = client.get("/history?page=1&page_size=5")
    assert response.status_code == 200

    data = response.json()
    assert data["page"] == 1
    assert data["page_size"] == 5
    assert len(data["results"]) <= 5


def test_history_invalid_page(client):
    response = client.get("/history?page=0")
    assert response.status_code == 422