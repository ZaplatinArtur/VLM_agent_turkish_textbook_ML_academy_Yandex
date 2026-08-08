#!/usr/bin/env python3
"""Build and run the source-only Math12 activity adapter.

This CLI has no benchmark loader and no correctness/scoring command.  The
``resolve`` command accepts arbitrary image bytes, sweeps every content page,
and writes a source-address certificate or an explicit abstention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_PACKAGES = REPO_ROOT / "tmp" / "portfolio_official_sources" / "python_pkgs"
for candidate in (PINNED_PACKAGES, REPO_ROOT / "src"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evidence_os.math12_activity_source import (  # noqa: E402
    Math12SourceError,
    build_math12_inventory,
    extract_official_solution,
    load_math12_inventory,
    load_math12_render_manifest,
    load_math12_source_certificate,
    resolve_math12_image_bytes,
    write_canonical_json,
)
from evidence_os.official_ogm import canonical_json_sha256, sha256_file  # noqa: E402
from evidence_os.visual_coordinate_binding import (  # noqa: E402
    SiftRuntimeProfile,
    VisualCoordinateBindingError,
)


RENDER_SCHEMA = "math12-poppler-content-render-manifest-v1"
EXPECTED_POPPLER_VERSION = "26.05.0"
EXPECTED_OPENCV_VERSION = "5.0.0"
_PAGE_NAME = re.compile(r"^page-(\d+)\.png$")


def _poppler_version(executable: Path) -> str:
    result = subprocess.run(
        [str(executable), "-v"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"pdftoppm version ([0-9.]+)", text)
    if result.returncode != 0 or match is None:
        raise Math12SourceError("cannot attest the Poppler runtime")
    return match.group(1)


def _build_command(args: argparse.Namespace) -> None:
    inventory = build_math12_inventory(args.pdf)
    write_canonical_json(args.output, inventory.to_mapping())
    print(
        json.dumps(
            {
                "activities": len(inventory.activities),
                "pdf_sha256": inventory.pdf_sha256,
                "inventory_projection_sha256": inventory.inventory_projection_sha256,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


def _render_command(args: argparse.Namespace) -> None:
    inventory = load_math12_inventory(args.inventory)
    if sha256_file(args.pdf) != inventory.pdf_sha256:
        raise Math12SourceError("render input PDF differs from the inventory pin")
    version = _poppler_version(args.pdftoppm)
    if version != EXPECTED_POPPLER_VERSION:
        raise Math12SourceError(
            f"Poppler {version} differs from pinned {EXPECTED_POPPLER_VERSION}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "page"
    command = [
        str(args.pdftoppm),
        "-f",
        str(inventory.content_page_start),
        "-l",
        str(inventory.content_page_end),
        "-r",
        str(args.dpi),
        "-gray",
        "-png",
        "--",
        str(args.pdf.resolve()),
        str(prefix.resolve()),
    ]
    subprocess.run(command, check=True)
    pages: dict[int, dict[str, Any]] = {}
    for path in args.output_dir.glob("page-*.png"):
        match = _PAGE_NAME.fullmatch(path.name)
        if match is None:
            continue
        page_number = int(match.group(1))
        if inventory.content_page_start <= page_number <= inventory.content_page_end:
            pages[page_number] = {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    expected = set(range(inventory.content_page_start, inventory.content_page_end + 1))
    if set(pages) != expected:
        raise Math12SourceError("Poppler did not render every content page exactly once")
    projection = {
        "schema_version": RENDER_SCHEMA,
        "document_id": inventory.document_id,
        "pdf_sha256": inventory.pdf_sha256,
        "inventory_projection_sha256": inventory.inventory_projection_sha256,
        "render_dpi": args.dpi,
        "color_mode": "gray_png",
        "poppler_version": version,
        "pages": {str(page): pages[page] for page in sorted(pages)},
    }
    projection["render_manifest_projection_sha256"] = canonical_json_sha256(projection)
    write_canonical_json(args.manifest, projection)
    print(
        json.dumps(
            {
                "rendered_pages": len(pages),
                "render_manifest_projection_sha256": projection[
                    "render_manifest_projection_sha256"
                ],
                "manifest": str(args.manifest.resolve()),
            },
            ensure_ascii=False,
        )
    )


def _resolve_command(args: argparse.Namespace) -> None:
    inventory = load_math12_inventory(args.inventory)
    manifest = load_math12_render_manifest(args.render_manifest, inventory)
    profile = SiftRuntimeProfile(
        render_dpi=manifest.render_dpi,
        expected_opencv_version=EXPECTED_OPENCV_VERSION,
    )
    certificate = resolve_math12_image_bytes(
        args.image.read_bytes(), inventory, manifest, runtime_profile=profile
    )
    write_canonical_json(args.output, certificate.to_mapping())
    print(
        json.dumps(
            {
                "accepted": certificate.decision.accepted,
                "reason": certificate.decision.reason,
                "content_page": certificate.decision.selected_content_page,
                "activity_number": certificate.decision.selected_activity_number,
                "key_page_start": certificate.decision.key_page_start,
                "key_page_end": certificate.decision.key_page_end,
                "certificate_projection_sha256": certificate.certificate_projection_sha256,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


def _extract_solution_command(args: argparse.Namespace) -> None:
    inventory = load_math12_inventory(args.inventory)
    certificate = load_math12_source_certificate(args.certificate)
    solution = extract_official_solution(args.pdf, inventory, certificate)
    write_canonical_json(args.output, solution.to_mapping())
    print(
        json.dumps(
            {
                "activity_number": solution.activity_number,
                "key_page_start": solution.key_page_start,
                "key_page_end": solution.key_page_end,
                "official_solution_text_sha256": solution.official_solution_text_sha256,
                "answer_bound_certificate_projection_sha256": (
                    solution.answer_bound_certificate_projection_sha256
                ),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build-inventory")
    build.add_argument("--pdf", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(handler=_build_command)

    render = commands.add_parser("render-pages")
    render.add_argument("--pdf", type=Path, required=True)
    render.add_argument("--inventory", type=Path, required=True)
    render.add_argument("--pdftoppm", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--manifest", type=Path, required=True)
    render.add_argument("--dpi", type=int, default=144, choices=(144,))
    render.set_defaults(handler=_render_command)

    resolve = commands.add_parser("resolve")
    resolve.add_argument("--inventory", type=Path, required=True)
    resolve.add_argument("--render-manifest", type=Path, required=True)
    resolve.add_argument("--image", type=Path, required=True)
    resolve.add_argument("--output", type=Path, required=True)
    resolve.set_defaults(handler=_resolve_command)

    extract = commands.add_parser("extract-solution")
    extract.add_argument("--pdf", type=Path, required=True)
    extract.add_argument("--inventory", type=Path, required=True)
    extract.add_argument("--certificate", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.set_defaults(handler=_extract_solution_command)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        args.handler(args)
    except (Math12SourceError, VisualCoordinateBindingError, subprocess.CalledProcessError) as exc:
        print(f"math12 source adapter failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
