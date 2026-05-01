"""
Unit tests for renderer/server.py — the FastAPI render API.

These tests run WITHOUT a GPU (stub-mode) so they pass in CI:
    pytest renderer/test_server.py -v

For a real GPU end-to-end test, run against a pod-hosted server manually:
    RENDERER_SERVER_URL=http://<pod>:8080 pytest renderer/test_server.py -v -m gpu
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    """FastAPI TestClient with an isolated OUTPUT_DIR and no auth."""
    monkeypatch.setenv("RENDERER_OUTPUT_DIR", str(tmp_path / "renders"))
    monkeypatch.delenv("RENDERER_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RENDERER_PUBLIC_BASE_URL", raising=False)

    # Reload the module so env-var reads take effect
    import importlib
    import renderer.server as server  # noqa: PLC0415

    importlib.reload(server)
    return TestClient(server.app)


@pytest.fixture
def silent_wav(tmp_path) -> Path:
    """A 1-second 16kHz silent mono wav (for tests that don't actually render)."""
    out = tmp_path / "silent.wav"
    # 1 second of silence: 16000 samples × 2 bytes = 32000 bytes of zeros + wav header
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-t",
            "1",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture
def silent_wav_b64(silent_wav) -> str:
    return base64.b64encode(silent_wav.read_bytes()).decode()


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------
class TestHealthz:
    def test_healthz_returns_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["engine"] == "musetalk-v1.5"
        assert "cuda" in data
        assert "cached_engines" in data

    def test_healthz_no_auth_required(self, client, monkeypatch):
        """/healthz should work even when bearer auth is enabled."""
        monkeypatch.setenv("RENDERER_AUTH_TOKEN", "secret")
        resp = client.get("/healthz")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
class TestAuth:
    def test_render_rejects_missing_token_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RENDERER_OUTPUT_DIR", str(tmp_path / "renders"))
        monkeypatch.setenv("RENDERER_AUTH_TOKEN", "secret")
        import importlib
        import renderer.server as server

        importlib.reload(server)
        tc = TestClient(server.app)

        resp = tc.post(
            "/render",
            json={"audio": "x", "portrait": "x"},
        )
        assert resp.status_code == 401

    def test_render_accepts_valid_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RENDERER_OUTPUT_DIR", str(tmp_path / "renders"))
        monkeypatch.setenv("RENDERER_AUTH_TOKEN", "secret")
        import importlib
        import renderer.server as server

        importlib.reload(server)
        tc = TestClient(server.app)

        # This will fail for a different reason (bad audio/portrait) but it
        # should get past the auth check first (i.e. NOT be 401).
        resp = tc.post(
            "/render",
            json={"audio": "x", "portrait": "x"},
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# /render input validation
# ---------------------------------------------------------------------------
class TestRenderValidation:
    def test_render_rejects_missing_audio(self, client):
        resp = client.post("/render", json={"portrait": "https://example.com/x.jpg"})
        assert resp.status_code == 422  # pydantic validation

    def test_render_rejects_missing_portrait(self, client, silent_wav_b64):
        resp = client.post("/render", json={"audio": silent_wav_b64})
        assert resp.status_code == 422

    def test_render_rejects_bad_fps(self, client, silent_wav_b64):
        resp = client.post(
            "/render",
            json={"audio": silent_wav_b64, "portrait": "x", "fps": 1000},
        )
        assert resp.status_code == 422

    def test_render_bubbles_up_bad_audio(self, client):
        resp = client.post(
            "/render",
            json={"audio": "not-valid-base64-@@@", "portrait": "x"},
        )
        # Bad audio can fail at resolve time (400) or pydantic (422)
        assert resp.status_code in {400, 422}


# ---------------------------------------------------------------------------
# /renders/{filename}
# ---------------------------------------------------------------------------
class TestGetRender:
    def test_get_render_404_for_missing(self, client):
        resp = client.get("/renders/does-not-exist.mp4")
        assert resp.status_code == 404

    def test_get_render_404_for_wrong_extension(self, client, tmp_path, monkeypatch):
        output = Path(os.environ["RENDERER_OUTPUT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "hello.txt").write_text("hi")
        resp = client.get("/renders/hello.txt")
        assert resp.status_code == 404

    def test_get_render_serves_mp4(self, client):
        output = Path(os.environ["RENDERER_OUTPUT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        fake_mp4 = output / "fake.mp4"
        fake_mp4.write_bytes(b"fake mp4 bytes")
        resp = client.get("/renders/fake.mp4")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"
        assert resp.content == b"fake mp4 bytes"


# ---------------------------------------------------------------------------
# End-to-end (marked gpu; skipped by default)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
class TestEndToEnd:
    """Only run when a real server URL is provided via env."""

    @pytest.fixture(autouse=True)
    def skip_without_url(self):
        if not os.environ.get("RENDERER_SERVER_URL"):
            pytest.skip("RENDERER_SERVER_URL not set — skipping real-GPU E2E test")

    def test_healthz_reports_cuda(self):
        import urllib.request

        url = os.environ["RENDERER_SERVER_URL"].rstrip("/")
        with urllib.request.urlopen(f"{url}/healthz", timeout=5) as r:
            data = json.loads(r.read())
        assert data["cuda"] is True
        assert data["engine"] == "musetalk-v1.5"
