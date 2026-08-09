"""Génération de description de scène minimaliste par règles, sans VLM."""

from __future__ import annotations

from collections import Counter

from manga_access.schemas.manga_page import Panel

_TYPE_ORDER = ("dialogue", "narration", "sfx", "thought")
_TYPE_LABELS: dict[str, tuple[str, str]] = {
    "dialogue": ("dialogue", "dialogues"),
    "narration": ("narration", "narrations"),
    "sfx": ("effet sonore", "effets sonores"),
    "thought": ("pensée", "pensées"),
}


def _pluralize(count: int, singular: str, plural: str) -> str:
    """Retourne la forme singulier/pluriel adaptée à `count`."""
    return singular if count == 1 else plural


def _characters_sentence(n_characters: int) -> str | None:
    """Phrase sur le nombre de personnages détectés sur la page, ou None si aucun."""
    if n_characters == 0:
        return None
    word = _pluralize(n_characters, "personnage", "personnages")
    verb = _pluralize(n_characters, "détecté", "détectés")
    return f"{n_characters} {word} {verb}."


def _elements_sentence(panel: Panel) -> str:
    """Phrase sur les éléments de texte du panel : 'Aucun texte.' ou liste par type."""
    if not panel.elements:
        return "Aucun texte."

    counts = Counter(element.type for element in panel.elements)
    parts = [
        f"{counts[type_]} {_pluralize(counts[type_], *_TYPE_LABELS[type_])}"
        for type_ in _TYPE_ORDER
        if counts[type_] > 0
    ]
    return ", ".join(parts) + "."


def describe_panel(panel: Panel, n_characters: int) -> str | None:
    """Génère une description minimaliste du panel à partir des données MangaPage.

    Ne s'appuie sur aucun VLM : uniquement le nombre de personnages détectés
    sur la page entière (Magiv2 ne lie pas les personnages aux panels) et les
    éléments de texte propres au panel. Retourne None quand il n'y a
    strictement rien à dire (aucun élément, aucun personnage sur la page).
    """
    if n_characters == 0 and not panel.elements:
        return None

    characters_sentence = _characters_sentence(n_characters)
    elements_sentence = _elements_sentence(panel)

    if characters_sentence is None:
        return elements_sentence
    return f"{characters_sentence} {elements_sentence}"
