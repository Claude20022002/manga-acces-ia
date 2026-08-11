"""État et exécution des jobs de production (pilotage de scripts/demo.py en sous-processus).

Registre en mémoire, process-local : suivi éphémère (perdu au redémarrage
du serveur), pas une donnée persistante — cf. plan Phase "interface web"
pour la justification vs SQLite (stack décidé, réservé au contenu durable).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from manga_access.pipeline.image_source import find_images

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MANGA109_IMAGES_ROOT = _REPO_ROOT / "data" / "manga109s" / "Manga109s_released_2026_05_21" / "images"
_OUTPUT_ROOT = _REPO_ROOT / "data" / "outputs" / "jobs"
_CHARACTER_BANKS_DIR = _REPO_ROOT / "data" / "character_banks"
_DEMO_SCRIPT = _REPO_ROOT / "scripts" / "demo.py"


@dataclass
class Job:
    """État d'un job de production, mis à jour au fil de l'exécution du sous-processus."""

    job_id: str
    manga_title: str
    output_dir: Path
    image_paths: list[Path]
    status: str = "running"  # "running" | "done" | "error"
    log_lines: list[str] = field(default_factory=list)
    subscribers: list[asyncio.Queue[str | None]] = field(default_factory=list)
    error_message: str | None = None


_JOBS: dict[str, Job] = {}


def list_manga_titles() -> list[str]:
    """Retourne les titres de mangas disponibles (dossiers sous data/manga109s/.../images), triés."""
    if not _MANGA109_IMAGES_ROOT.is_dir():
        return []
    return sorted(p.name for p in _MANGA109_IMAGES_ROOT.iterdir() if p.is_dir())


def resolve_manga_dir(title: str) -> Path:
    """Résout `title` vers son dossier d'images, en validant contre `list_manga_titles()`.

    Lève `ValueError` si `title` ne correspond à aucun dossier réel — c'est
    le garde-fou anti-traversée de chemin : `title` vient de l'URL, jamais
    utilisé pour construire un chemin avant cette vérification.
    """
    if title not in list_manga_titles():
        raise ValueError(f"manga inconnu : {title!r}")
    return _MANGA109_IMAGES_ROOT / title


def get_job(job_id: str) -> Job | None:
    """Retourne le job `job_id`, ou None s'il n'existe pas (jamais créé, ou serveur redémarré)."""
    return _JOBS.get(job_id)


def _character_bank_for_title(title: str) -> Path | None:
    """Résout automatiquement une character_bank pour `title` si `data/character_banks/{title}.json` existe."""
    candidate = _CHARACTER_BANKS_DIR / f"{title.lower()}.json"
    return candidate if candidate.is_file() else None


async def start_job(manga_title: str, pages: int | None, narration_lang: str) -> Job:
    """Valide `manga_title`, crée un job et lance scripts/demo.py dessus en sous-processus.

    Le sous-processus (pas d'appel direct à ChapterProcessor) isole la RAM
    et les crashs du pipeline ML (Magiv2/manga-ocr/Kokoro, contrainte 12 Go
    du projet) du process API. `image_paths` est résolu ici via le même
    `find_images` que demo.py utilisera lui-même en interne — les deux
    doivent produire exactement la même liste pour que `page_index` dans la
    timeline corresponde à la bonne image servie par `/api/page`.
    """
    images_dir = resolve_manga_dir(manga_title)
    image_paths = find_images(images_dir, limit=pages)

    job_id = uuid.uuid4().hex
    output_dir = _OUTPUT_ROOT / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    job = Job(
        job_id=job_id,
        manga_title=manga_title,
        output_dir=output_dir,
        image_paths=image_paths,
    )
    _JOBS[job_id] = job

    command = [
        "uv",
        "run",
        "python",
        str(_DEMO_SCRIPT),
        str(images_dir),
        "--output-dir",
        str(output_dir),
        "--narration-lang",
        narration_lang,
    ]
    if pages is not None:
        command += ["--pages", str(pages)]
    character_bank_path = _character_bank_for_title(manga_title)
    if character_bank_path is not None:
        command += ["--character-bank", str(character_bank_path)]

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(_REPO_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    asyncio.create_task(_pump_logs(job, process))
    return job


async def _pump_logs(job: Job, process: asyncio.subprocess.Process) -> None:
    """Lit la sortie du sous-processus ligne par ligne, alimente log_lines et les abonnés SSE."""
    assert process.stdout is not None
    async for raw_line in process.stdout:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
        if not line:
            continue
        job.log_lines.append(line)
        for queue in job.subscribers:
            queue.put_nowait(line)

    returncode = await process.wait()
    job.status = "done" if returncode == 0 else "error"
    if job.status == "error":
        job.error_message = f"scripts/demo.py a terminé avec le code de sortie {returncode}"
    for queue in job.subscribers:
        queue.put_nowait(None)  # sentinelle : fin du flux pour ce job


async def subscribe(job: Job) -> AsyncIterator[str]:
    """Fait défiler les lignes de log de `job`, déjà connues puis en direct, jusqu'à la fin du job.

    Rejoue d'abord `job.log_lines` (utile à un client SSE qui se connecte
    en cours de job), puis attend les nouvelles lignes via une Queue
    dédiée tant que `job.status == "running"`.
    """
    for line in job.log_lines:
        yield line

    if job.status != "running":
        return

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    job.subscribers.append(queue)
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        job.subscribers.remove(queue)
