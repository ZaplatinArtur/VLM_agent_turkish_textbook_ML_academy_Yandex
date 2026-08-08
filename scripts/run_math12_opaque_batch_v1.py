from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from evidence_os.math12_opaque_batch import run_opaque_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Math12 source resolution over an opaque image JSONL; "
            "the output contains no benchmark labels or score."
        )
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument(
        "--render-page-root",
        type=Path,
        help="Directory containing the manifest-pinned page-NNN.png files",
    )
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_opaque_batch(
        input_jsonl=args.input_jsonl,
        asset_root=args.asset_root,
        inventory_path=args.inventory,
        render_manifest_path=args.render_manifest,
        render_page_root=args.render_page_root,
        pdf_path=args.pdf,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
