from __future__ import annotations

from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def iter_image_paths(input_folder: str | Path, limit: int | None = None) -> list[Path]:
    folder = Path(input_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    paths = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.stem.endswith("_prediction")
    )
    return paths if limit is None else paths[:limit]
