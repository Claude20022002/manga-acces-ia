"""API FastAPI : pilotage de scripts/demo.py sur un manga Manga109-s et service des résultats.

Lancement : `uv run uvicorn manga_access.api.app:app` (depuis la racine du
dépôt ; PAS `src.manga_access.api.app` — le package installé s'appelle
`manga_access`, cf. pyproject.toml `packages = ["src/manga_access"]`).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from manga_access.api import jobs
from manga_access.pipeline.image_source import find_images

_APP_HTML_PATH = Path(__file__).resolve().parents[1] / "app.html"

app = FastAPI(title="Manga Access AI — Démonstration")


class MangaSummary(BaseModel):
    """Résumé d'un manga disponible pour production."""

    title: str
    page_count: int


class MangaPages(BaseModel):
    """Liste ordonnée des fichiers image d'un manga."""

    title: str
    pages: list[str]


class ProduceRequest(BaseModel):
    """Corps de POST /api/produce."""

    manga_title: str
    pages: int | None = None
    narration_lang: str = "fr"
    start_page: int = 0


class ProduceResponse(BaseModel):
    """Réponse de POST /api/produce."""

    job_id: str


@app.get("/")
def index() -> FileResponse:
    """Sert app.html à la racine (même origine que /api/... : pas de configuration CORS nécessaire)."""
    return FileResponse(_APP_HTML_PATH)


@app.get("/api/mangas")
def list_mangas() -> list[MangaSummary]:
    """Liste les mangas disponibles dans Manga109-s, avec leur nombre de pages."""
    summaries = []
    for title in jobs.list_manga_titles():
        images_dir = jobs.resolve_manga_dir(title)
        summaries.append(MangaSummary(title=title, page_count=len(find_images(images_dir))))
    return summaries


def _resolve_manga_dir_or_404(title: str) -> Path:
    try:
        return jobs.resolve_manga_dir(title)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/mangas/{title}/pages")
def manga_pages(title: str) -> MangaPages:
    """Liste les fichiers image d'un manga, dans l'ordre de lecture (tri alphabétique)."""
    images_dir = _resolve_manga_dir_or_404(title)
    return MangaPages(title=title, pages=[p.name for p in find_images(images_dir)])


@app.post("/api/produce")
async def produce(request: ProduceRequest) -> ProduceResponse:
    """Lance scripts/demo.py sur le manga demandé, en arrière-plan (sous-processus)."""
    try:
        job = await jobs.start_job(
            request.manga_title, request.pages, request.narration_lang, request.start_page
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ProduceResponse(job_id=job.job_id)


def _get_job_or_404(job_id: str) -> jobs.Job:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job inconnu : {job_id!r}")
    return job


@app.get("/api/status/{job_id}")
async def status(job_id: str) -> StreamingResponse:
    """Flux SSE : une ligne de log par événement, puis un événement final {type: status}."""
    job = _get_job_or_404(job_id)

    async def event_stream() -> AsyncIterator[str]:
        async for line in jobs.subscribe(job):
            yield f"data: {json.dumps({'type': 'log', 'message': line})}\n\n"
        final: dict[str, str] = {"type": "status", "status": job.status}
        if job.error_message is not None:
            final["error"] = job.error_message
        yield f"data: {json.dumps(final)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/audio/{job_id}")
def audio(job_id: str) -> FileResponse:
    """Sert demo.opus produit par le job, une fois terminé."""
    job = _get_job_or_404(job_id)
    audio_path = job.output_dir / "demo.opus"
    if job.status != "done" or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="audio pas encore disponible")
    return FileResponse(audio_path, media_type="audio/ogg")


@app.get("/api/timeline/{job_id}")
def timeline(job_id: str) -> FileResponse:
    """Sert demo.timeline.json produit par le job, une fois terminé."""
    job = _get_job_or_404(job_id)
    timeline_path = job.output_dir / "demo.timeline.json"
    if job.status != "done" or not timeline_path.is_file():
        raise HTTPException(status_code=404, detail="timeline pas encore disponible")
    return FileResponse(timeline_path, media_type="application/json")


@app.get("/api/page/{job_id}/{page_index}")
def page(job_id: str, page_index: int) -> FileResponse:
    """Sert l'image source de la page `page_index` (0-indexée) du manga traité par le job."""
    job = _get_job_or_404(job_id)
    if page_index < 0 or page_index >= len(job.image_paths):
        raise HTTPException(status_code=404, detail="page_index hors bornes")
    return FileResponse(job.image_paths[page_index])
