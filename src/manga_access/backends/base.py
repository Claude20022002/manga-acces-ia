"""Interfaces abstraites des backends du pipeline.

Chaque backend concret (OCR, structuration, TTS, etc.) doit implémenter une de
ces interfaces. La logique métier ne doit jamais importer un modèle
directement — toujours passer par ces contrats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from manga_access.schemas.manga_page import BBox, TextElement


class OCRBackend(ABC):
    """Contrat pour un backend de reconnaissance de texte japonais."""

    @abstractmethod
    def load(self) -> None:
        """Charge le modèle en mémoire. Doit être appelé avant `recognize`."""
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """Décharge le modèle de la mémoire (contrainte RAM CPU du projet)."""
        raise NotImplementedError

    @abstractmethod
    def recognize(self, image: Image.Image, bbox: BBox) -> TextElement:
        """Reconnaît le texte contenu dans la région `bbox` de `image`.

        L'implémentation est responsable de recadrer l'image selon `bbox`
        avant reconnaissance et de renseigner `confidence` dans le
        `TextElement` retourné.
        """
        raise NotImplementedError


class StructureBackend(ABC):
    """Contrat pour un backend de structuration (détection cases/bulles/personnages)."""

    @abstractmethod
    def load(self) -> None:
        """Charge le modèle en mémoire. Doit être appelé avant `detect`."""
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """Décharge le modèle de la mémoire (contrainte RAM CPU du projet)."""
        raise NotImplementedError

    @abstractmethod
    def detect(self, image: np.ndarray) -> dict[str, Any]:
        """Détecte la structure de `image` (planche complète).

        # Any : la structure retournée est hétérogène par nature (listes de
        # bbox, paires d'indices d'association, labels de cluster, booléens).
        # Elle reflète le format brut de sortie du modèle de structuration
        # sous-jacent (ex. Magiv2), avant assemblage dans le schéma MangaPage.
        Retourne un dict incluant au minimum les clés "panels" et "texts"
        (listes de bbox [x1, y1, x2, y2]).
        """
        raise NotImplementedError

    @abstractmethod
    def detect_chapter(
        self,
        images: Sequence[np.ndarray | Image.Image],
        character_bank: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Détecte la structure de tout un chapitre (`images`, dans l'ordre de lecture).

        Contrairement à `detect`, traite toutes les pages en un seul appel
        modèle : c'est ce qui permet l'identification nominative des
        personnages par `character_bank` (dict `{"images": [...], "names": [...]}`
        de portraits de référence) — le matching s'appuie sur des contraintes
        de cohérence *entre* les pages du chapitre, impossibles à reproduire
        page par page. Si `character_bank` est `None` ou vide, aucun nom
        n'est assigné (comportement équivalent à `detect` appelé page par
        page). Retourne une liste de dicts, un par page, dans le même format
        que `detect` plus la clé "character_names".
        """
        raise NotImplementedError


class SceneDescriptionBackend(ABC):
    """Contrat pour un backend de description de scène (VLM)."""

    @abstractmethod
    def load(self) -> None:
        """Charge le modèle en mémoire. Doit être appelé avant `describe`."""
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """Décharge le modèle de la mémoire (contrainte RAM CPU du projet)."""
        raise NotImplementedError

    @abstractmethod
    def describe(self, image: Image.Image, bbox: BBox) -> str | None:
        """Décrit la scène contenue dans la région `bbox` de `image`.

        Retourne None si le backend n'a rien de pertinent à décrire (dégradation
        gracieuse, cohérent avec `Panel.scene_description: str | None`).
        """
        raise NotImplementedError


class TTSBackend(ABC):
    """Contrat pour un backend de synthèse vocale."""

    @abstractmethod
    def synthesize(self, text: str, voice_id: str, lang: str = "en-us") -> bytes:
        """Synthétise `text` avec la voix `voice_id` et retourne l'audio brut.

        `lang` pilote la phonémisation (ex. "en-us", "ja" une fois la
        traduction Qwen3-VL intégrée) — indépendant de `voice_id`.
        """
        raise NotImplementedError
