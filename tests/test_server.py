import json
import threading
import urllib.request
from memory_mesh.engine import MemoryMeshEngine
from memory_mesh.server import create_server


def test_http_rest_api_server():
    engine = MemoryMeshEngine()
    server = create_server(host="127.0.0.1", port=8991, engine=engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = "http://127.0.0.1:8991"

    try:
        # 1. Test /stats
        with urllib.request.urlopen(f"{base_url}/stats") as res:
            assert res.status == 200
            data = json.loads(res.read().decode("utf-8"))
            assert "total_episodes" in data

        # 2. Test POST /ingest
        req_data = json.dumps({"content": "Minh runs memory mesh benchmarks", "importance": 0.9, "entities": ["Minh", "Benchmarks"]}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/ingest", data=req_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as res:
            assert res.status == 201
            data = json.loads(res.read().decode("utf-8"))
            assert data["status"] == "ok"
            assert data["episode"]["importance"] == 0.9

        # 3. Test POST /recall
        recall_data = json.dumps({"query": "Minh", "top_k": 1}).encode("utf-8")
        req_rec = urllib.request.Request(f"{base_url}/recall", data=recall_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_rec) as res:
            assert res.status == 200
            res_json = json.loads(res.read().decode("utf-8"))
            assert len(res_json["episodes"]) == 1
    finally:
        server.shutdown()
        server.server_close()
