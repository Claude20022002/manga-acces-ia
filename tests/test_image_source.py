"""Tests de find_images (résolution ordonnée des pages d'un dossier manga)."""

from __future__ import annotations

from pathlib import Path

import pytest

from manga_access.pipeline.image_source import find_images


def _touch(path: Path) -> None:
    path.write_bytes(b"")


def test_find_images_sorted_alphabetically(tmp_path: Path) -> None:
    """Les images sont triées par nom, indépendamment de l'ordre de création sur disque."""
    _touch(tmp_path / "002.jpg")
    _touch(tmp_path / "000.jpg")
    _touch(tmp_path / "001.jpg")

    result = find_images(tmp_path)

    assert [p.name for p in result] == ["000.jpg", "001.jpg", "002.jpg"]


def test_find_images_filters_non_image_extensions(tmp_path: Path) -> None:
    """Les fichiers non-image (ex. mapping.json) sont ignorés."""
    _touch(tmp_path / "000.jpg")
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / "mapping.json")

    result = find_images(tmp_path)

    assert [p.name for p in result] == ["000.jpg"]


def test_find_images_accepts_common_extensions(tmp_path: Path) -> None:
    """jpg/jpeg/png/webp/bmp (et leurs majuscules) sont tous reconnus."""
    names = ["a.jpg", "b.jpeg", "c.png", "d.webp", "e.bmp", "f.JPG"]
    for name in names:
        _touch(tmp_path / name)

    result = find_images(tmp_path)

    assert {p.name for p in result} == set(names)


def test_find_images_limit_truncates_after_sort(tmp_path: Path) -> None:
    """limit tronque après le tri, pas avant (toujours les N premières pages)."""
    for name in ["002.jpg", "000.jpg", "001.jpg", "003.jpg"]:
        _touch(tmp_path / name)

    result = find_images(tmp_path, limit=2)

    assert [p.name for p in result] == ["000.jpg", "001.jpg"]


def test_find_images_limit_none_returns_all(tmp_path: Path) -> None:
    """limit=None (défaut) retourne toutes les images, sans troncature."""
    for name in ["000.jpg", "001.jpg", "002.jpg"]:
        _touch(tmp_path / name)

    result = find_images(tmp_path, limit=None)

    assert len(result) == 3


def test_find_images_empty_folder_raises(tmp_path: Path) -> None:
    """Un dossier sans image lève ValueError plutôt que de retourner silencieusement []."""
    with pytest.raises(ValueError, match="aucune image"):
        find_images(tmp_path)
