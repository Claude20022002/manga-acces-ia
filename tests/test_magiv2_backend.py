"""Tests de Magiv2Backend.detect_chapter (chapitre-entier, identification nominative)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from manga_access.backends.magiv2_backend import Magiv2Backend


def _make_backend_with_mock_model() -> tuple[Magiv2Backend, MagicMock]:
    """Construit un Magiv2Backend dont `_model` est un mock (pas de téléchargement HuggingFace)."""
    backend = Magiv2Backend()
    mock_model = MagicMock()
    backend._model = mock_model
    return backend, mock_model


def test_detect_chapter_raises_when_not_loaded() -> None:
    """detect_chapter() avant load() lève RuntimeError, comme detect()."""
    backend = Magiv2Backend()

    with pytest.raises(RuntimeError):
        backend.detect_chapter([np.zeros((10, 10, 3), dtype=np.uint8)])


def test_detect_chapter_calls_do_chapter_wide_prediction_with_ocr_disabled() -> None:
    """do_ocr=False systématiquement : manga-ocr reste la seule source d'OCR du projet."""
    backend, mock_model = _make_backend_with_mock_model()
    mock_model.do_chapter_wide_prediction.return_value = [{"characters": []}]
    images = [np.zeros((10, 10, 3), dtype=np.uint8)]
    bank = {"images": [np.zeros((5, 5, 3), dtype=np.uint8)], "names": ["Naruto"]}

    backend.detect_chapter(images, character_bank=bank)

    mock_model.do_chapter_wide_prediction.assert_called_once()
    call_args = mock_model.do_chapter_wide_prediction.call_args
    assert call_args.args[1] is bank
    assert call_args.kwargs["do_ocr"] is False


def test_detect_chapter_defaults_to_empty_bank_when_none() -> None:
    """character_bank=None est transmis comme banque vide, pas comme None (contrat Magiv2)."""
    backend, mock_model = _make_backend_with_mock_model()
    mock_model.do_chapter_wide_prediction.return_value = [{"characters": []}]

    backend.detect_chapter([np.zeros((10, 10, 3), dtype=np.uint8)], character_bank=None)

    call_args = mock_model.do_chapter_wide_prediction.call_args
    assert call_args.args[1] == {"images": [], "names": []}


def test_detect_chapter_converts_pil_images_to_ndarray() -> None:
    """Les images PIL sont converties en np.ndarray avant l'appel modèle (cohérent avec detect())."""
    backend, mock_model = _make_backend_with_mock_model()
    mock_model.do_chapter_wide_prediction.return_value = [{"characters": []}]
    pil_image = Image.new("RGB", (10, 10))

    backend.detect_chapter([pil_image])

    passed_images = mock_model.do_chapter_wide_prediction.call_args.args[0]
    assert isinstance(passed_images[0], np.ndarray)


def test_detect_chapter_returns_per_page_results_unchanged() -> None:
    """Le résultat de do_chapter_wide_prediction est retourné tel quel (passthrough)."""
    backend, mock_model = _make_backend_with_mock_model()
    expected = [
        {"characters": [], "character_names": []},
        {"characters": [], "character_names": ["Naruto"]},
    ]
    mock_model.do_chapter_wide_prediction.return_value = expected

    result = backend.detect_chapter([Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10))])

    assert result is expected
