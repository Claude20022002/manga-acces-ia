#!/usr/bin/env python3
"""Smoketest Qwen3-VL-4B-Instruct GGUF via llama-cpp-python (MTMDChatHandler).

Script de validation, PAS du code de production. Objectif (Phase 6,
prérequis documenté dans docs/phases/phase-5.md, section "État Qwen3-VL") :
vérifier que le GGUF Q4_K_M tourne sur ce CPU (Ryzen 5 5500U, 12 Go RAM,
chargement séquentiel) avant d'écrire tout backend QwenVLBackend de
production. Mesure temps de chargement, RAM pic, temps d'inférence, et
affiche la sortie texte pour évaluation qualitative manuelle.

Le paquet mainline llama-cpp-python n'a pas de handler dédié Qwen3-VL (voir
issues ouvertes non résolues abetlen/llama-cpp-python#2080 et #2098) mais
expose un handler générique MTMDChatHandler basé sur libmtmd, qui supporte
Qwen3-VL au niveau du cœur C++ llama.cpp depuis la PR ggml-org/llama.cpp#16780
(mergée). Ce script utilise ce chemin générique ; un échec ici (en particulier
une AttributeError autour de 'sampler', cf. issue #2098) est un résultat de
smoketest valide à documenter, pas un bug de ce script.

Fichiers GGUF attendus (non fournis, à télécharger séparément) :
    Qwen3VL-4B-Instruct-Q4_K_M.gguf        (~2.5 Go)
    mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf   (~454 Mo)
    depuis https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF

Usage:
    python benchmarks/qwen_vl_smoketest.py \
        --model /chemin/vers/Qwen3VL-4B-Instruct-Q4_K_M.gguf \
        --mmproj /chemin/vers/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf \
        [--image data/manga_jpg/1-1.jpg] [--prompt "..."] \
        [--n-ctx 4096] [--n-threads 6] [--max-tokens 256] \
        [--output data/outputs/qwen_vl_smoketest.txt]
"""

from __future__ import annotations

import argparse
import base64
import gc
import resource
import sys
import time
from pathlib import Path

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MTMDChatHandler
except ImportError as exc:
    print(
        "Erreur : le paquet 'llama-cpp-python' n'est pas installé.\n"
        "Installe-le (ex. 'uv add llama-cpp-python' ou "
        "'pip install llama-cpp-python') avant d'exécuter ce script.\n"
        "Note : si 'MTMDChatHandler' n'existe pas dans ta version installée, "
        "vérifie le CHANGELOG de llama-cpp-python pour le nom de classe "
        "correspondant au handler multimodal générique dans ta version.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

_DEFAULT_PROMPT = (
    "Décris cette image de manga en une ou deux phrases, en français, pour "
    "une personne aveugle : action, ambiance, éléments visuels importants. "
    "Ne décris pas le texte des bulles, seulement l'image."
)


def _peak_rss_mb() -> float:
    """RAM résidente pic du processus courant, en Mo (Linux : ru_maxrss en Ko)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _image_to_data_uri(image_path: Path) -> str:
    """Encode une image locale en data URI base64 (format attendu par llama-cpp-python)."""
    suffix = image_path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix == "jpg" else suffix
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def run_smoketest(
    model_path: Path,
    mmproj_path: Path,
    image_path: Path,
    prompt: str,
    n_ctx: int,
    n_threads: int,
    max_tokens: int,
) -> str:
    """Charge le modèle, lance une inférence sur une image, retourne un rapport texte.

    Toutes les mesures (temps, RAM) sont imprimées au fur et à mesure sur
    stdout en plus d'être incluses dans le rapport retourné, pour visibilité
    immédiate en cas d'échec en cours de route.
    """
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    log(f"Modèle    : {model_path}")
    log(f"Mmproj    : {mmproj_path}")
    log(f"Image     : {image_path}")
    log(f"RAM avant chargement : {_peak_rss_mb():.0f} Mo")

    load_start = time.perf_counter()
    chat_handler = MTMDChatHandler(clip_model_path=str(mmproj_path))
    llm = Llama(
        model_path=str(model_path),
        chat_handler=chat_handler,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False,
    )
    load_elapsed = time.perf_counter() - load_start
    log(f"Temps de chargement   : {load_elapsed:.2f}s")
    log(f"RAM après chargement  : {_peak_rss_mb():.0f} Mo")

    image_data_uri = _image_to_data_uri(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        }
    ]

    inference_start = time.perf_counter()
    response = llm.create_chat_completion(messages=messages, max_tokens=max_tokens)
    inference_elapsed = time.perf_counter() - inference_start

    output_text = response["choices"][0]["message"]["content"]

    log(f"Temps d'inférence     : {inference_elapsed:.2f}s")
    log(f"RAM après inférence   : {_peak_rss_mb():.0f} Mo")
    log("")
    log("=== Sortie du modèle ===")
    log(output_text)

    del llm
    del chat_handler
    gc.collect()

    return "\n".join(lines)


def main() -> None:
    """Point d'entrée CLI du smoketest Qwen3-VL."""
    parser = argparse.ArgumentParser(
        description="Smoketest Qwen3-VL-4B-Instruct GGUF via llama-cpp-python."
    )
    parser.add_argument("--model", type=Path, required=True, help="Chemin du GGUF Q4_K_M")
    parser.add_argument("--mmproj", type=Path, required=True, help="Chemin du GGUF mmproj")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/manga_jpg/1-1.jpg"),
        help="Image du corpus à décrire (défaut : data/manga_jpg/1-1.jpg)",
    )
    parser.add_argument("--prompt", type=str, default=_DEFAULT_PROMPT, help="Prompt utilisateur")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Taille du contexte (défaut 4096)")
    parser.add_argument(
        "--n-threads", type=int, default=6, help="Threads CPU (défaut 6, Ryzen 5 5500U 6c/12t)"
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="Tokens max générés")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/qwen_vl_smoketest.txt"),
        help="Fichier rapport (défaut : data/outputs/qwen_vl_smoketest.txt)",
    )
    args = parser.parse_args()

    for path, label in ((args.model, "--model"), (args.mmproj, "--mmproj"), (args.image, "--image")):
        if not path.is_file():
            print(f"Erreur : fichier introuvable pour {label} : {path}", file=sys.stderr)
            raise SystemExit(1)

    try:
        report = run_smoketest(
            model_path=args.model,
            mmproj_path=args.mmproj,
            image_path=args.image,
            prompt=args.prompt,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            max_tokens=args.max_tokens,
        )
    except AttributeError as exc:
        print(
            f"\nÉchec (AttributeError : {exc}).\n"
            "Si le message mentionne 'sampler', c'est probablement le bug connu "
            "abetlen/llama-cpp-python#2098 (chargement Qwen3-VL cassé au niveau "
            "des bindings Python, indépendant du cœur C++ llama.cpp qui supporte "
            "ce modèle depuis ggml-org/llama.cpp#16780). Résultat de smoketest "
            "valide à documenter dans docs/phases/ : llama-cpp-python n'est pas "
            "encore utilisable tel quel pour Qwen3-VL, envisager llama-server/"
            "llama-mtmd-cli natifs en repli.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\nRapport écrit dans {args.output}")


if __name__ == "__main__":
    main()
