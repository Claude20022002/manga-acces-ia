#!/usr/bin/env python3
"""Smoketest traduction Qwen3-VL-4B-Instruct GGUF (texte seul, pas d'image).

Script de validation, PAS du code de production. Étape 0 du plan de
traduction des dialogues (Phase 3, roadmap original) : mesurer la latence
par segment et juger la qualité de traduction AVANT de brancher
QwenVLBackend.translate() dans scripts/demo.py — même discipline que
benchmarks/qwen_vl_smoketest.py pour l'usage vision (description de scène),
cf. docs/phases/phase-5.md.

Traduit une dizaine de vraies lignes de dialogue OCR (corpus Manga109-s,
Arisa — déjà observées dans docs/sessions et data/outputs/demo/demo.txt
cette session) vers la langue cible, un seul chargement du modèle.

Usage:
    python benchmarks/qwen_vl_translate_smoketest.py \
        --model models/qwen3vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf \
        --mmproj models/qwen3vl/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf \
        [--target-lang fr] [--n-ctx 4096] [--n-threads 6] [--max-tokens 128] \
        [--output data/outputs/qwen_vl_translate_smoketest.txt]
"""

from __future__ import annotations

import argparse
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
        "Lance 'uv sync' pour installer les dépendances du projet avant "
        "d'exécuter ce script.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

_SAMPLE_DIALOGUE_LINES = (
    "ありがとうあります。",
    "そういえば、これを原作すれば、原作は１１月２０日",
    "ううっ",
    "しんちゃん。",
    "私に",
    "ないよぅ．．．",
    "泣くなって！",
    "やるからさ",
    "今度会うときはしんちゃんがびっくりするくらい元気な子になってる",
    "起きて兄さんっ！！",
)

_TARGET_LANG_NAMES = {"fr": "français", "en": "anglais"}

_TRANSLATION_PROMPT_TEMPLATE = (
    "Traduis ce texte japonais en {lang_name}. Réponds uniquement avec la "
    "traduction, sans explication ni guillemets : {text}"
)


def _peak_rss_mb() -> float:
    """RAM résidente pic du processus courant, en Mo (Linux : ru_maxrss en Ko)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_smoketest(
    model_path: Path,
    mmproj_path: Path,
    target_lang: str,
    lines: tuple[str, ...],
    n_ctx: int,
    n_threads: int,
    max_tokens: int,
) -> str:
    """Charge le modèle une fois, traduit chaque ligne de `lines`, retourne un rapport texte."""
    report_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        report_lines.append(msg)

    log(f"Modèle       : {model_path}")
    log(f"Mmproj       : {mmproj_path}")
    log(f"Langue cible : {target_lang}")
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
    log(f"Temps de chargement  : {load_elapsed:.2f}s")
    log(f"RAM après chargement : {_peak_rss_mb():.0f} Mo")
    log("")

    lang_name = _TARGET_LANG_NAMES.get(target_lang, target_lang)
    latencies: list[float] = []
    for i, text in enumerate(lines):
        prompt = _TRANSLATION_PROMPT_TEMPLATE.format(lang_name=lang_name, text=text)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        start = time.perf_counter()
        response = llm.create_chat_completion(messages=messages, max_tokens=max_tokens)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        output = response["choices"][0]["message"]["content"].strip()
        log(f"[{i}] {elapsed:.2f}s")
        log(f"    JA          : {text}")
        log(f"    {target_lang.upper():<11} : {output}")
        log("")

    log(f"RAM après {len(lines)} traduction(s) : {_peak_rss_mb():.0f} Mo")
    log(
        f"Latence min/moy/max : {min(latencies):.2f}s / "
        f"{sum(latencies) / len(latencies):.2f}s / {max(latencies):.2f}s"
    )

    del llm
    del chat_handler
    gc.collect()

    return "\n".join(report_lines)


def main() -> None:
    """Point d'entrée CLI du smoketest de traduction Qwen3-VL."""
    parser = argparse.ArgumentParser(
        description="Smoketest traduction Qwen3-VL-4B-Instruct GGUF (texte seul)."
    )
    parser.add_argument("--model", type=Path, required=True, help="Chemin du GGUF Q4_K_M")
    parser.add_argument("--mmproj", type=Path, required=True, help="Chemin du GGUF mmproj")
    parser.add_argument(
        "--target-lang", choices=["fr", "en"], default="fr", help="Langue cible (défaut: fr)"
    )
    parser.add_argument("--n-ctx", type=int, default=4096, help="Taille du contexte (défaut 4096)")
    parser.add_argument(
        "--n-threads", type=int, default=6, help="Threads CPU (défaut 6, Ryzen 5 5500U 6c/12t)"
    )
    parser.add_argument("--max-tokens", type=int, default=128, help="Tokens max générés par ligne")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/qwen_vl_translate_smoketest.txt"),
        help="Fichier rapport (défaut : data/outputs/qwen_vl_translate_smoketest.txt)",
    )
    args = parser.parse_args()

    for path, label in ((args.model, "--model"), (args.mmproj, "--mmproj")):
        if not path.is_file():
            print(f"Erreur : fichier introuvable pour {label} : {path}", file=sys.stderr)
            raise SystemExit(1)

    try:
        report = run_smoketest(
            model_path=args.model,
            mmproj_path=args.mmproj,
            target_lang=args.target_lang,
            lines=_SAMPLE_DIALOGUE_LINES,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            max_tokens=args.max_tokens,
        )
    except AttributeError as exc:
        print(
            f"\nÉchec (AttributeError : {exc}).\n"
            "Si le message mentionne 'sampler', c'est probablement le bug connu "
            "abetlen/llama-cpp-python#2098, cf. benchmarks/qwen_vl_smoketest.py.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"\nRapport écrit dans {args.output}")


if __name__ == "__main__":
    main()
