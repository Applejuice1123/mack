"""Turning a file into text for a batch item.

Plain text is read directly. Images and PDFs go through the Mack app's own Vision
and PDFKit code (the same recognizer the screen explainer uses), invoked as
`Mack --ocr <image>` and `Mack --pdf-text <pdf>`; it is not reimplemented here.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Optional

from .config import BASE, ROOT

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".heic", ".tif", ".tiff", ".gif", ".bmp", ".webp"}
PDF_EXT = {".pdf"}
MIN_OCR_CHARS = 20


def _find_helper() -> Path:
    if os.environ.get("MACK_HELPER"):
        return Path(os.environ["MACK_HELPER"])
    apps = (ROOT / "app" / "Mack.app", BASE / "app" / "Mack.app", Path("/Applications/Mack.app"))
    app = next((a for a in apps if a.exists()), apps[0])
    return app / "Contents" / "MacOS" / "Mack"


HELPER = _find_helper()


class OCRUnavailable(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _helper(flag: str, path: Path) -> str:
    if not HELPER.exists():
        raise OCRUnavailable("the Mack app is not built (run app/build.sh); it does the text recognition")
    out = subprocess.run([str(HELPER), flag, str(path)], capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        raise OCRUnavailable(out.stdout.strip() or out.stderr.strip() or f"{flag} failed")
    return out.stdout


def ocr_image(path: Path) -> str:
    return _helper("--ocr", path)


def pdf_text(path: Path) -> str:
    """Text layer first; OCR of the first 3 pages if there is none (done by the helper)."""
    return _helper("--pdf-text", path)


def looks_binary(head: bytes) -> bool:
    if b"\x00" in head:
        return True
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as e:
        return e.start < len(head) - 4  # a multi-byte character cut at the boundary is fine
    return False


def read_text(path: Path, max_bytes: int, ocr: bool) -> tuple[Optional[str], Optional[str]]:
    """Returns (text, None) or (None, skip reason)."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXT or ext in PDF_EXT:
        if not ocr:
            return None, "unsupported_type: images and PDFs need a recipe with OCR (e.g. rename)"
        try:
            text = ocr_image(path) if ext in IMAGE_EXT else pdf_text(path)
        except (OCRUnavailable, subprocess.TimeoutExpired) as e:
            return None, f"ocr_unavailable: {e}"
        if len(text.strip()) < MIN_OCR_CHARS:
            return None, "no_text"
        return text, None
    size = path.stat().st_size
    if size > max_bytes:
        return None, f"too_large: {size / 1e6:.1f} MB, limit {max_bytes // 1024} KB"
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        return None, "binary: not a text file"
    if looks_binary(data[:8192]):
        return None, "not_utf8: text in another encoding; re-save it as UTF-8 to include it"
    text = data.decode("utf-8", errors="replace")
    if not text.strip():
        return None, "empty"
    return text, None
