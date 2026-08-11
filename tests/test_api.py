"""Tests de l'API FastAPI (src/manga_access/api/).

Le lancement réel de scripts/demo.py (sous-processus, pipeline ML complet)
n'est jamais exercé ici : jobs.start_job est monkeypatché pour les tests
qui passent par POST /api/produce. La plomberie sous-processus elle-même
est vérifiée manuellement (uvicorn + curl, cf. plan de la tâche).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from manga_access.api import app as app_module
from manga_access.api import jobs as jobs_module


@pytest.fixture(autouse=True)
def clear_jobs() -> Any:
    """Vide le registre de jobs en mémoire avant et après chaque test (isolation)."""
    jobs_module._JOBS.clear()
    yield
    jobs_module._JOBS.clear()


@pytest.fixture
def manga_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Un faux dossier Manga109-s avec un manga "TestManga" de 2 pages."""
    root = tmp_path / "manga109s"
    manga_dir = root / "TestManga"
    manga_dir.mkdir(parents=True)
    (manga_dir / "000.jpg").write_bytes(b"fake-image-bytes-0")
    (manga_dir / "001.jpg").write_bytes(b"fake-image-bytes-1")
    monkeypatch.setattr(jobs_module, "_MANGA109_IMAGES_ROOT", root)
    return root


@pytest.fixture
def output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "outputs"
    monkeypatch.setattr(jobs_module, "_OUTPUT_ROOT", root)
    return root


@pytest.fixture
def client() -> TestClient:
    return TestClient(app_module.app)


def _make_job(job_id: str, output_dir: Path, **overrides: Any) -> jobs_module.Job:
    defaults: dict[str, Any] = {
        "job_id": job_id,
        "manga_title": "TestManga",
        "output_dir": output_dir,
        "image_paths": [],
        "status": "done",
    }
    defaults.update(overrides)
    job = jobs_module.Job(**defaults)
    jobs_module._JOBS[job_id] = job
    return job


# --- jobs.py : résolution / sécurité ---


def test_list_manga_titles_sorted(manga_root: Path) -> None:
    (manga_root / "AnotherManga").mkdir()
    assert jobs_module.list_manga_titles() == ["AnotherManga", "TestManga"]


def test_resolve_manga_dir_unknown_title_raises(manga_root: Path) -> None:
    """Un titre qui ne correspond à aucun dossier réel lève ValueError (garde-fou traversée de chemin)."""
    with pytest.raises(ValueError):
        jobs_module.resolve_manga_dir("../../etc")


def test_character_bank_for_title_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    banks_dir = tmp_path / "banks"
    banks_dir.mkdir()
    (banks_dir / "testmanga.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(jobs_module, "_CHARACTER_BANKS_DIR", banks_dir)

    assert jobs_module._character_bank_for_title("TestManga") == banks_dir / "testmanga.json"


def test_character_bank_for_title_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    banks_dir = tmp_path / "banks"
    banks_dir.mkdir()
    monkeypatch.setattr(jobs_module, "_CHARACTER_BANKS_DIR", banks_dir)

    assert jobs_module._character_bank_for_title("Unknown") is None


# --- GET /api/mangas ---


def test_list_mangas_returns_titles_and_page_counts(client: TestClient, manga_root: Path) -> None:
    response = client.get("/api/mangas")

    assert response.status_code == 200
    assert response.json() == [{"title": "TestManga", "page_count": 2}]


# --- GET /api/mangas/{title}/pages ---


def test_manga_pages_lists_filenames_sorted(client: TestClient, manga_root: Path) -> None:
    response = client.get("/api/mangas/TestManga/pages")

    assert response.status_code == 200
    assert response.json() == {"title": "TestManga", "pages": ["000.jpg", "001.jpg"]}


def test_manga_pages_unknown_title_404(client: TestClient, manga_root: Path) -> None:
    response = client.get("/api/mangas/DoesNotExist/pages")

    assert response.status_code == 404


# --- POST /api/produce ---


def test_produce_returns_job_id(
    client: TestClient, manga_root: Path, output_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_start_job(manga_title: str, pages: int | None, narration_lang: str) -> jobs_module.Job:
        return _make_job("fake-job-id", output_root / "fake-job-id", manga_title=manga_title, status="running")

    monkeypatch.setattr(jobs_module, "start_job", fake_start_job)

    response = client.post("/api/produce", json={"manga_title": "TestManga"})

    assert response.status_code == 200
    assert response.json() == {"job_id": "fake-job-id"}


def test_produce_unknown_manga_returns_400(client: TestClient, manga_root: Path) -> None:
    """manga_title inconnu -> 400, sans jamais tenter de lancer scripts/demo.py."""
    response = client.post("/api/produce", json={"manga_title": "Nope"})

    assert response.status_code == 400


# --- GET /api/status/{job_id} (SSE) ---


def _parse_sse_events(body: str) -> list[dict[str, Any]]:
    """Parse un corps SSE (lignes 'data: {...}') en liste de payloads JSON décodés."""
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def test_status_streams_buffered_logs_then_final_status(client: TestClient, output_root: Path) -> None:
    _make_job(
        "job-done",
        output_root / "job-done",
        status="done",
        log_lines=["🔍 Détection structure...", "✅ Audio généré"],
    )

    response = client.get("/api/status/job-done")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    assert events == [
        {"type": "log", "message": "🔍 Détection structure..."},
        {"type": "log", "message": "✅ Audio généré"},
        {"type": "status", "status": "done"},
    ]


def test_status_unknown_job_404(client: TestClient) -> None:
    response = client.get("/api/status/does-not-exist")

    assert response.status_code == 404


# --- GET /api/audio/{job_id} ---


def test_audio_endpoint_serves_file_when_done(client: TestClient, output_root: Path) -> None:
    job_dir = output_root / "job-audio"
    job_dir.mkdir(parents=True)
    (job_dir / "demo.opus").write_bytes(b"fake-opus-bytes")
    _make_job("job-audio", job_dir, status="done")

    response = client.get("/api/audio/job-audio")

    assert response.status_code == 200
    assert response.content == b"fake-opus-bytes"


def test_audio_endpoint_404_when_not_done(client: TestClient, output_root: Path) -> None:
    job_dir = output_root / "job-running"
    job_dir.mkdir(parents=True)
    _make_job("job-running", job_dir, status="running")

    response = client.get("/api/audio/job-running")

    assert response.status_code == 404


# --- GET /api/timeline/{job_id} ---


def test_timeline_endpoint_serves_file_when_done(client: TestClient, output_root: Path) -> None:
    job_dir = output_root / "job-timeline"
    job_dir.mkdir(parents=True)
    (job_dir / "demo.timeline.json").write_text('{"segments": []}', encoding="utf-8")
    _make_job("job-timeline", job_dir, status="done")

    response = client.get("/api/timeline/job-timeline")

    assert response.status_code == 200
    assert response.json() == {"segments": []}


def test_timeline_endpoint_404_when_not_done(client: TestClient, output_root: Path) -> None:
    job_dir = output_root / "job-running"
    job_dir.mkdir(parents=True)
    _make_job("job-running", job_dir, status="running")

    response = client.get("/api/timeline/job-running")

    assert response.status_code == 404


# --- GET /api/page/{job_id}/{page_index} ---


def test_page_endpoint_serves_correct_image(
    client: TestClient, manga_root: Path, output_root: Path
) -> None:
    image_path = manga_root / "TestManga" / "000.jpg"
    _make_job("job-page", output_root / "job-page", image_paths=[image_path], status="done")

    response = client.get("/api/page/job-page/0")

    assert response.status_code == 200
    assert response.content == image_path.read_bytes()


def test_page_endpoint_out_of_range_404(client: TestClient, output_root: Path) -> None:
    _make_job("job-page-oob", output_root / "job-page-oob", image_paths=[], status="done")

    response = client.get("/api/page/job-page-oob/0")

    assert response.status_code == 404


def test_page_endpoint_unknown_job_404(client: TestClient) -> None:
    response = client.get("/api/page/does-not-exist/0")

    assert response.status_code == 404
