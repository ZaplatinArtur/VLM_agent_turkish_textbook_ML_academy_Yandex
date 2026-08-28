import os
import shutil
import sys
import urllib.request
from pathlib import Path

import pytesseract

from paths import TESSDATA_DIR, ensure_data_dirs

_TUR_URL = "https://github.com/tesseract-ocr/tessdata/raw/main/tur.traineddata"


def _from_registry() -> Path | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:
        return None

    for hive, subkey in (
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Tesseract-OCR"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Tesseract-OCR"),
    ):
        try:
            with winreg.OpenKey(hive, subkey) as key:
                install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
        except OSError:
            continue
        exe = Path(install_dir) / "tesseract.exe"
        if exe.exists():
            return exe
    return None


def _resolve_cmd() -> str:
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if cmd:
        return cmd

    registry_cmd = _from_registry()
    if registry_cmd is not None:
        return str(registry_cmd)

    raise RuntimeError(
        "Tesseract OCR not found. Install it, add to PATH, or set TESSERACT_CMD."
    )


def _tessdata_dir(tesseract_cmd: str) -> Path:
    prefix = os.environ.get("TESSDATA_PREFIX")
    if prefix:
        return Path(prefix)
    return Path(tesseract_cmd).resolve().parent / "tessdata"


def _install_turkish(tessdata_dir: Path) -> None:
    tessdata_dir.mkdir(parents=True, exist_ok=True)
    tur_path = tessdata_dir / "tur.traineddata"
    if not tur_path.exists():
        urllib.request.urlretrieve(_TUR_URL, tur_path)
    os.environ["TESSDATA_PREFIX"] = str(tessdata_dir.resolve()) + os.sep


def _ensure_turkish(tesseract_cmd: str) -> None:
    ensure_data_dirs()
    if "tur" in set(pytesseract.get_languages(config="")):
        return

    candidates = [
        TESSDATA_DIR,
        _tessdata_dir(tesseract_cmd),
        Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "tessdata",
    ]
    for tessdata_dir in candidates:
        try:
            _install_turkish(tessdata_dir)
        except OSError:
            continue
        if "tur" in set(pytesseract.get_languages(config="")):
            return

    raise RuntimeError(
        "Turkish tessdata (tur) is missing. Download tur.traineddata into data/corpus/tessdata "
        "or set TESSDATA_PREFIX."
    )


def configure_tesseract() -> None:
    cmd = _resolve_cmd()
    pytesseract.pytesseract.tesseract_cmd = cmd
    _ensure_turkish(cmd)
