"""CLI entry for ColQwen LoRA fine-tuning (GPU host)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visual_retrive.train.train_colqwen_lora import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
