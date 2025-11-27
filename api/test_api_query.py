from fastapi.testclient import TestClient
from unittest.mock import patch
from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

@patch("api.main.execute_query")
def test_query_forwarding(mock_exec):
    mock_exec.return_value = {"rows": [{"id": 1, "name": "alice"}]}
    payload = {"query_str": "dummy", "params": {"x": 1}}
    r = client.post("/query", json=payload)
    assert r.status_code == 200
    assert r.json() == {"rows": [{"id": 1, "name": "alice"}]}
    mock_exec.assert_called_once_with("dummy", {"x": 1})