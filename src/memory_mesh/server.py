from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict
from memory_mesh.engine import MemoryMeshEngine


class MemoryMeshHTTPHandler(BaseHTTPRequestHandler):
    engine: MemoryMeshEngine

    def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/stats":
            self._send_json(200, self.engine.stats())
        elif self.path == "/graph":
            self._send_json(200, self.engine.graph.to_dict())
        else:
            self._send_json(404, {"error": "Not Found", "endpoints": ["/stats", "/graph", "/ingest", "/recall", "/reflect"]})

    def do_POST(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        if self.path == "/ingest":
            content = payload.get("content")
            if not content:
                self._send_json(400, {"error": "Missing 'content' in request body"})
                return
            importance = float(payload.get("importance", 0.5))
            entities = payload.get("entities")
            episode = self.engine.ingest(content=content, importance=importance, entities=entities)
            self._send_json(201, {"status": "ok", "episode": episode.model_dump(mode="json")})

        elif self.path == "/recall":
            query = payload.get("query")
            if not query:
                self._send_json(400, {"error": "Missing 'query' in request body"})
                return
            top_k = int(payload.get("top_k", 3))
            res = self.engine.recall(query=query, top_k=top_k)
            self._send_json(200, res.model_dump(mode="json"))

        elif self.path == "/reflect":
            insights = self.engine.reflect()
            self._send_json(200, {
                "reflected_count": len(insights),
                "insights": [ep.model_dump(mode="json") for ep in insights],
            })

        else:
            self._send_json(404, {"error": "Not Found"})

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy standard request logging in tests
        pass


def create_server(host: str = "127.0.0.1", port: int = 8000, engine: MemoryMeshEngine | None = None) -> HTTPServer:
    handler = MemoryMeshHTTPHandler
    handler.engine = engine or MemoryMeshEngine()
    return HTTPServer((host, port), handler)
