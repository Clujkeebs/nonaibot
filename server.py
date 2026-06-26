"""
Health/status HTTP server.

Endpoints:
  GET /health  — lightweight liveness check (Railway uses this)
  GET /status  — full portfolio + risk state as JSON

Runs in a daemon thread; the main loop owns the process lifecycle.
Shared state is updated by the main loop via update_status().
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Shared state — main loop writes, HTTP handler reads (no lock needed for dict reads in CPython)
_status: Dict[str, Any] = {
    "bot": "starting",
    "timestamp": None,
    "equity": None,
    "buying_power": None,
    "open_positions": 0,
    "risk_status": "UNKNOWN",
    "risk_reason": "",
    "paper_mode": True,
    "last_equity_scan": None,
    "last_crypto_scan": None,
    "last_exit_check": None,
}


def update_status(updates: Dict[str, Any]) -> None:
    """Called by main loop to push fresh state."""
    _status.update(updates)
    _status["timestamp"] = datetime.utcnow().isoformat() + "Z"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress access log to avoid log noise

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "timestamp": _status.get("timestamp")})
        elif self.path == "/status":
            self._respond(200, _status)
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code: int, body: dict) -> None:
        payload = json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the HTTP server in a background daemon thread."""
    server = HTTPServer((host, port), _Handler)

    def _run():
        logger.info("Status server listening on %s:%d", host, port)
        server.serve_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
