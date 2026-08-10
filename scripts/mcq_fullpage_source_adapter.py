#!/usr/bin/env python3
"""Build, render, resolve and replay the bounded Bio9/Physics12 source adapter.

The command surface accepts official source PDFs or observable prompt/image
bytes.  It deliberately has no benchmark IDs, selection map, gold labels,
predictions, scorer or accuracy command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_PACKAGES = REPO_ROOT / "tmp" / "portfolio_official_sources" / "python_pkgs"
for candidate in (PINNED_PACKAGES, REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from build_mcq_fullpage_source_v1 import build_source  # noqa: E402
from evidence_os.mcq_fullpage_source import (  # noqa: E402
    EXPECTED_NUMPY_VERSION,
    EXPECTED_OPENCV_VERSION,
    EXPECTED_PDFPLUMBER_VERSION,
    EXPECTED_PDFTOPPM_SHA256,
    EXPECTED_POPPLER_VERSION,
    EXPECTED_PYTHON_VERSION,
    FROZEN_SIFT_RUNTIME_PROFILE,
    McqRenderManifest,
    McqRenderedPage,
    McqSourceError,
    RENDER_MANIFEST_SCHEMA,
    assert_frozen_mcq_bundle,
    assert_mcq_runtime,
    load_mcq_inventory,
    load_mcq_render_manifest,
    load_mcq_source_certificate,
    resolve_mcq_image_bytes,
    verify_mcq_source_certificate,
    write_canonical_json,
)
from evidence_os.official_ogm import canonical_json_sha256, sha256_file  # noqa: E402
from evidence_os.mcq_opaque_batch import (  # noqa: E402
    McqOpaqueBatchError,
    assert_mcq_v11_code_freeze,
)
from evidence_os.visual_coordinate_binding import (  # noqa: E402
    VisualCoordinateBindingError,
)


_POPPLER_VERSION = re.compile(r"pdftoppm version ([0-9.]+)")


def _poppler_pin(executable: Path) -> tuple[str, str]:
    executable = executable.resolve()
    if not executable.is_file():
        raise McqSourceError("pdftoppm executable is missing")
    executable_sha = sha256_file(executable)
    if executable_sha != EXPECTED_PDFTOPPM_SHA256:
        raise McqSourceError("pdftoppm executable differs from the frozen SHA-256")
    result = subprocess.run(
        [str(executable), "-v"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    match = _POPPLER_VERSION.search(f"{result.stdout}\n{result.stderr}")
    if result.returncode != 0 or match is None:
        raise McqSourceError("cannot attest the pdftoppm version")
    version = match.group(1)
    if version != EXPECTED_POPPLER_VERSION:
        raise McqSourceError("pdftoppm version differs from the frozen runtime")
    return version, executable_sha


def _png_meta(path: Path) -> tuple[int, int]:
    try:
        header = path.read_bytes()[:29]
    except OSError as exc:
        raise McqSourceError("rendered PNG cannot be read") from exc
    if (
        len(header) != 29
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
        or header[24] != 8
        or header[25] != 2
    ):
        raise McqSourceError(
            "pdftoppm did not produce its pinned 8-bit RGB PNG container"
        )
    return int.from_bytes(header[16:20], "big"), int.from_bytes(
        header[20:24], "big"
    )


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    try:
        return args.prompt_file.read_text(encoding="utf-8-sig").rstrip("\r\n")
    except (OSError, UnicodeError) as exc:
        raise McqSourceError("prompt file cannot be read as UTF-8") from exc


def _attest_v11_code(args: argparse.Namespace) -> None:
    assert_mcq_v11_code_freeze(
        freeze_manifest_path=args.v11_freeze_manifest,
        expected_freeze_sha256=args.expected_v11_freeze_sha256,
        expected_freeze_projection_sha256=(
            args.expected_v11_freeze_projection_sha256
        ),
    )


def _verify_source_command(args: argparse.Namespace) -> None:
    _attest_v11_code(args)
    bundle = assert_frozen_mcq_bundle(
        freeze_manifest_path=args.freeze_manifest,
        inventory_path=args.inventory,
        key_index_path=args.key_index,
        render_manifest_path=args.render_manifest,
        page_root=args.page_root,
    )
    observed_inventory = bundle.inventory
    observed_key_index = bundle.key_index
    rebuilt_inventory, rebuilt_key_index, audit = build_source(
        args.biology_pdf, args.physics_pdf
    )
    if (
        observed_inventory.to_mapping() != rebuilt_inventory.to_mapping()
        or observed_key_index.to_mapping() != rebuilt_key_index.to_mapping()
    ):
        raise McqSourceError("source artifacts do not reproduce from the pinned PDFs")
    print(
        json.dumps(
            {
                "verified": True,
                "protocol_addresses": len(observed_inventory.questions),
                "official_choice_records": len(observed_key_index.cells),
                "inventory_projection_sha256": (
                    observed_inventory.inventory_projection_sha256
                ),
                "key_index_projection_sha256": (
                    observed_key_index.key_index_projection_sha256
                ),
                "source_layout_heading_projection_sha256": audit[
                    "source_layout_heading_projection_sha256"
                ],
            },
            ensure_ascii=False,
        )
    )


def _render_command(args: argparse.Namespace) -> None:
    assert_mcq_runtime()
    inventory = load_mcq_inventory(args.inventory)
    output_dir = args.output_dir.resolve(strict=False)
    manifest_path = args.manifest.resolve(strict=False)
    try:
        manifest_path.relative_to(output_dir)
    except ValueError:
        pass
    else:
        raise McqSourceError("render manifest must be outside the page payload root")
    if args.output_dir.exists() or args.manifest.exists():
        raise McqSourceError("render output and manifest must both be absent")
    pdf_by_family = {
        "biology9_textbook": args.biology_pdf.resolve(),
        "physics12_textbook": args.physics_pdf.resolve(),
    }
    for document in inventory.documents:
        pdf_path = pdf_by_family[document.source_family]
        if (
            not pdf_path.is_file()
            or sha256_file(pdf_path) != document.pdf_sha256
            or pdf_path.stat().st_size != document.pdf_size_bytes
        ):
            raise McqSourceError("render source PDF differs from its inventory pins")
    version, executable_sha = _poppler_pin(args.pdftoppm)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix="mcq_pages_", dir=output_dir.parent)
    )
    try:
        pages: list[McqRenderedPage] = []
        for document in inventory.documents:
            document_dir = staging / document.document_id
            document_dir.mkdir(parents=True)
            pdf_path = pdf_by_family[document.source_family]
            for page_number in document.content_pages:
                relative_path = (
                    f"{document.document_id}/page-{page_number:04d}.png"
                )
                output_path = staging / Path(relative_path)
                prefix = output_path.with_suffix("")
                result = subprocess.run(
                    [
                        str(args.pdftoppm.resolve()),
                        "-f",
                        str(page_number),
                        "-l",
                        str(page_number),
                        "-r",
                        str(FROZEN_SIFT_RUNTIME_PROFILE.render_dpi),
                        "-gray",
                        "-png",
                        "-singlefile",
                        "--",
                        str(pdf_path),
                        str(prefix),
                    ],
                    check=False,
                    capture_output=True,
                )
                if result.returncode != 0 or not output_path.is_file():
                    raise McqSourceError(
                        f"pdftoppm failed for {document.document_id} p{page_number}"
                    )
                width, height = _png_meta(output_path)
                pages.append(
                    McqRenderedPage(
                        document_id=document.document_id,
                        page_number=page_number,
                        relative_path=relative_path,
                        sha256=sha256_file(output_path),
                        size_bytes=output_path.stat().st_size,
                        width=width,
                        height=height,
                        resolved_path=output_path,
                    )
                )
        pages.sort(key=lambda item: (item.document_id, item.page_number))
        projection: dict[str, Any] = {
            "schema_version": RENDER_MANIFEST_SCHEMA,
            "inventory_projection_sha256": inventory.inventory_projection_sha256,
            "render_dpi": FROZEN_SIFT_RUNTIME_PROFILE.render_dpi,
            "color_mode": "poppler_gray_rgb_png",
            "poppler_version": version,
            "poppler_executable_sha256": executable_sha,
            "pages": [item.to_mapping() for item in pages],
        }
        projection_sha = canonical_json_sha256(projection)
        manifest = McqRenderManifest(
            inventory_projection_sha256=inventory.inventory_projection_sha256,
            render_dpi=FROZEN_SIFT_RUNTIME_PROFILE.render_dpi,
            color_mode="poppler_gray_rgb_png",
            poppler_version=version,
            poppler_executable_sha256=executable_sha,
            pages=tuple(pages),
            render_manifest_projection_sha256=projection_sha,
        )
        pending_manifest = staging / "_render_manifest.pending.json"
        write_canonical_json(pending_manifest, manifest.to_mapping())
        staging.replace(output_dir)
        (output_dir / pending_manifest.name).replace(manifest_path)
        load_mcq_render_manifest(manifest_path, inventory, page_root=output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "rendered_pages": len(pages),
                "render_manifest_projection_sha256": projection_sha,
                "page_root": str(output_dir),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )


def _resolve_command(args: argparse.Namespace) -> None:
    _attest_v11_code(args)
    bundle = assert_frozen_mcq_bundle(
        freeze_manifest_path=args.freeze_manifest,
        inventory_path=args.inventory,
        key_index_path=args.key_index,
        render_manifest_path=args.render_manifest,
        page_root=args.page_root,
    )
    prompt = _load_prompt(args)
    image_bytes = args.image.read_bytes()
    certificate = resolve_mcq_image_bytes(
        prompt,
        image_bytes,
        bundle.inventory,
        bundle.render_manifest,
        bundle.key_index,
    )
    verify_mcq_source_certificate(
        prompt,
        bundle.inventory,
        bundle.render_manifest,
        bundle.key_index,
        certificate,
        expected_task_image_bytes=image_bytes,
    )
    write_canonical_json(args.output, certificate.to_mapping())
    print(
        json.dumps(
            {
                "accepted": certificate.decision.accepted,
                "reason": certificate.decision.reason,
                "source_family": certificate.decision.selected_source_family,
                "content_page": certificate.decision.selected_page_number,
                "unit": certificate.decision.selected_unit_number,
                "question": certificate.decision.selected_question_number,
                "answer": certificate.answer,
                "certificate_projection_sha256": (
                    certificate.certificate_projection_sha256
                ),
            },
            ensure_ascii=False,
        )
    )


def _verify_certificate_command(args: argparse.Namespace) -> None:
    _attest_v11_code(args)
    bundle = assert_frozen_mcq_bundle(
        freeze_manifest_path=args.freeze_manifest,
        inventory_path=args.inventory,
        key_index_path=args.key_index,
        render_manifest_path=args.render_manifest,
        page_root=args.page_root,
    )
    image_bytes = args.image.read_bytes()
    certificate = load_mcq_source_certificate(args.certificate)
    decision = verify_mcq_source_certificate(
        _load_prompt(args),
        bundle.inventory,
        bundle.render_manifest,
        bundle.key_index,
        certificate,
        expected_task_image_bytes=image_bytes,
    )
    print(
        json.dumps(
            {
                "verified": True,
                "accepted": decision.accepted,
                "reason": decision.reason,
                "certificate_projection_sha256": (
                    certificate.certificate_projection_sha256
                ),
            },
            ensure_ascii=False,
        )
    )


def _preflight_command(args: argparse.Namespace) -> None:
    observed = assert_mcq_runtime(require_pdfplumber=True, require_visual=True)
    poppler, executable_sha = _poppler_pin(args.pdftoppm)
    observed["poppler"] = poppler
    observed["pdftoppm_sha256"] = executable_sha
    print(
        json.dumps(
            {
                "passed": True,
                "observed": observed,
                "expected": {
                    "python": EXPECTED_PYTHON_VERSION,
                    "pdfplumber": EXPECTED_PDFPLUMBER_VERSION,
                    "numpy": EXPECTED_NUMPY_VERSION,
                    "opencv": EXPECTED_OPENCV_VERSION,
                    "poppler": EXPECTED_POPPLER_VERSION,
                    "pdftoppm_sha256": EXPECTED_PDFTOPPM_SHA256,
                },
            },
            ensure_ascii=False,
        )
    )


def _add_source_artifacts(command: argparse.ArgumentParser) -> None:
    command.add_argument("--v11-freeze-manifest", type=Path, required=True)
    command.add_argument("--expected-v11-freeze-sha256", required=True)
    command.add_argument(
        "--expected-v11-freeze-projection-sha256", required=True
    )
    command.add_argument("--freeze-manifest", type=Path, required=True)
    command.add_argument("--inventory", type=Path, required=True)
    command.add_argument("--key-index", type=Path, required=True)


def _add_render_artifacts(command: argparse.ArgumentParser) -> None:
    command.add_argument("--render-manifest", type=Path, required=True)
    command.add_argument("--page-root", type=Path, required=True)


def _add_prompt(command: argparse.ArgumentParser) -> None:
    prompt = command.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    verify_source = commands.add_parser("verify-source")
    verify_source.add_argument("--biology-pdf", type=Path, required=True)
    verify_source.add_argument("--physics-pdf", type=Path, required=True)
    _add_source_artifacts(verify_source)
    _add_render_artifacts(verify_source)
    verify_source.set_defaults(handler=_verify_source_command)

    render = commands.add_parser("render-pages")
    render.add_argument("--biology-pdf", type=Path, required=True)
    render.add_argument("--physics-pdf", type=Path, required=True)
    render.add_argument("--inventory", type=Path, required=True)
    render.add_argument("--pdftoppm", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--manifest", type=Path, required=True)
    render.set_defaults(handler=_render_command)

    resolve = commands.add_parser("resolve")
    _add_source_artifacts(resolve)
    _add_render_artifacts(resolve)
    _add_prompt(resolve)
    resolve.add_argument("--image", type=Path, required=True)
    resolve.add_argument("--output", type=Path, required=True)
    resolve.set_defaults(handler=_resolve_command)

    verify = commands.add_parser("verify-certificate")
    _add_source_artifacts(verify)
    _add_render_artifacts(verify)
    _add_prompt(verify)
    verify.add_argument("--certificate", type=Path, required=True)
    verify.add_argument("--image", type=Path, required=True)
    verify.set_defaults(handler=_verify_certificate_command)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--pdftoppm", type=Path, required=True)
    preflight.set_defaults(handler=_preflight_command)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        args.handler(args)
    except (
        McqOpaqueBatchError,
        McqSourceError,
        VisualCoordinateBindingError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"MCQ source adapter failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
