#!/usr/bin/env python3
"""Run frozen task-ID-free MCQ source resolution on an opaque input bundle.

Only the observable prompt and pinned image bytes reach the resolver.  The
opaque ``input_id`` is retained solely to align source-only output records.
This command does not accept gold, labels, predictions, a scorer or a map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_PACKAGES = REPO_ROOT / "tmp" / "portfolio_official_sources" / "python_pkgs"
for candidate in (PINNED_PACKAGES, REPO_ROOT / "src"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evidence_os.mcq_fullpage_source import McqSourceError  # noqa: E402
from evidence_os.mcq_opaque_batch import (  # noqa: E402
    McqOpaqueBatchError,
    run_mcq_opaque_batch,
)
from evidence_os.visual_coordinate_binding import (  # noqa: E402
    VisualCoordinateBindingError,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--key-index", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--page-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = run_mcq_opaque_batch(
            input_jsonl=args.input_jsonl,
            asset_root=args.asset_root,
            inventory_path=args.inventory,
            key_index_path=args.key_index,
            render_manifest_path=args.render_manifest,
            page_root=args.page_root,
            output_dir=args.output_dir,
        )
    except (
        McqOpaqueBatchError,
        McqSourceError,
        VisualCoordinateBindingError,
        OSError,
    ) as exc:
        print(f"MCQ opaque source batch failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
