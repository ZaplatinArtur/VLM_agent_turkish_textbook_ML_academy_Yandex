from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_maxim_evidence_os_image_judge_v1 as image_judge  # noqa: E402
import compose_maxim_official_ogm_failclosed_v2 as composer  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def replay_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    root = tmp_path / "repo"
    generator = root / "scripts" / "build_maxim_activity_visual_binding_v1.py"
    package_root = root / "runtime_packages"
    task_image_dir = root / "task_images"
    temp_parent = root / "tmp"
    for directory in (
        generator.parent,
        package_root,
        task_image_dir,
        temp_parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    generator.write_text("# synthetic generator\n", encoding="utf-8")

    pdftoppm = root / "tools" / "pdftoppm.exe"
    pdfinfo = root / "tools" / "pdfinfo.exe"
    pdftoppm.parent.mkdir(parents=True)
    pdftoppm.write_bytes(b"synthetic pdftoppm")
    pdfinfo.write_bytes(b"synthetic pdfinfo")

    paths: dict[str, Path] = {}
    for name in ("profile", "parser", "locator", "source_index", "document"):
        path = root / "inputs" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        paths[name] = path
    visual_path = root / "frozen" / "visual.json"
    visual_path.parent.mkdir(parents=True)
    visual_path.write_bytes(b'{"frozen":true}\n')
    paths["visual"] = visual_path

    monkeypatch.setattr(composer, "REPO_ROOT", root)
    payload = {
        "runtime": {
            "python": {"executable": sys.executable},
            "poppler": {
                "pdftoppm": {
                    "path": str(pdftoppm),
                    "sha256": _sha256(pdftoppm),
                },
                "pdfinfo": {
                    "path": str(pdfinfo),
                    "sha256": _sha256(pdfinfo),
                },
            },
            "package_root": str(package_root),
        },
        "inputs": {
            "task_image_dir": str(task_image_dir),
            "documents": {"synthetic_document": {}},
        },
    }
    kwargs = {
        "profile_path": paths["profile"],
        "parser_path": paths["parser"],
        "locator_path": paths["locator"],
        "source_index_path": paths["source_index"],
        "visual_path": visual_path,
        "expected_visual_sha256": _sha256(visual_path),
        "visual_payload": payload,
        "document_pdf_paths": {"synthetic_document": paths["document"]},
    }
    return {
        "root": root,
        "generator": generator,
        "pdftoppm": pdftoppm,
        "pdfinfo": pdfinfo,
        "payload": payload,
        "kwargs": kwargs,
        "frozen_bytes": visual_path.read_bytes(),
    }


def _completed_run(
    *,
    rebuilt_bytes: bytes,
    report_sha: str | Callable[[Path], str] | None = None,
    stdout: str | None = None,
    returncode: int = 0,
    stderr: str = "",
) -> Callable[..., SimpleNamespace]:
    def run(command: list[str], **options: Any) -> SimpleNamespace:
        output_path = Path(command[command.index("--output-json") + 1])
        if returncode == 0:
            output_path.write_bytes(rebuilt_bytes)
        if stdout is not None:
            rendered_stdout = stdout
        elif returncode == 0:
            if callable(report_sha):
                output_sha = report_sha(output_path)
            elif report_sha is not None:
                output_sha = report_sha
            else:
                output_sha = _sha256(output_path)
            rendered_stdout = json.dumps(
                {
                    "output_json": str(output_path),
                    "output_sha256": output_sha,
                    "summary": {"raw_page_evidences": 1},
                }
            )
        else:
            rendered_stdout = ""
        return SimpleNamespace(
            returncode=returncode,
            stdout=rendered_stdout,
            stderr=stderr,
        )

    return run


def test_fresh_rebuild_returns_exact_source_only_attestation(
    replay_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_bytes = replay_case["frozen_bytes"]
    monkeypatch.setattr(
        composer.subprocess,
        "run",
        _completed_run(rebuilt_bytes=frozen_bytes),
    )

    result = composer._fresh_rebuild_activity_visual_artifact(
        **replay_case["kwargs"]
    )

    expected_sha = replay_case["kwargs"]["expected_visual_sha256"]
    assert result["mode"] == "fresh_source_only_poppler_sift_exact_bytes_v1"
    assert result["exact_byte_identity"] is True
    assert result["benchmark_answer_candidate_outcome_artifacts_read"] is False
    assert result["frozen_artifact"] == {
        "path": str(replay_case["kwargs"]["visual_path"]),
        "sha256": expected_sha,
        "size_bytes": len(frozen_bytes),
    }
    assert result["reproduced_artifact"] == {
        "sha256": expected_sha,
        "size_bytes": len(frozen_bytes),
    }
    assert result["summary"] == {"raw_page_evidences": 1}
    assert result["generator"]["path"] == str(replay_case["generator"])
    assert result["generator"]["sha256"] == _sha256(replay_case["generator"])
    assert set(result["runtime"]) == {
        "python_executable_sha256",
        "pdftoppm_sha256",
        "pdfinfo_sha256",
        "runtime_projection_sha256",
    }
    assert all(len(value) == 64 for value in result["runtime"].values())
    assert len(result["command_projection_sha256"]) == 64


def test_fresh_rebuild_rejects_nonzero_generator_exit(
    replay_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composer.subprocess,
        "run",
        _completed_run(
            rebuilt_bytes=b"",
            returncode=7,
            stderr="synthetic generator failure\n",
        ),
    )

    with pytest.raises(composer.CompositionError, match="synthetic generator failure"):
        composer._fresh_rebuild_activity_visual_artifact(**replay_case["kwargs"])


def test_fresh_rebuild_rejects_timeout(
    replay_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(command: list[str], **options: Any) -> None:
        raise subprocess.TimeoutExpired(command, 1_800)

    monkeypatch.setattr(composer.subprocess, "run", timeout)

    with pytest.raises(composer.CompositionError, match="exceeded 1800 seconds"):
        composer._fresh_rebuild_activity_visual_artifact(**replay_case["kwargs"])


def test_fresh_rebuild_rejects_malformed_stdout(
    replay_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composer.subprocess,
        "run",
        _completed_run(
            rebuilt_bytes=replay_case["frozen_bytes"],
            stdout="not JSON",
        ),
    )

    with pytest.raises(composer.CompositionError, match="returned malformed JSON"):
        composer._fresh_rebuild_activity_visual_artifact(**replay_case["kwargs"])


def test_fresh_rebuild_rejects_reported_sha_mismatch(
    replay_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        composer.subprocess,
        "run",
        _completed_run(
            rebuilt_bytes=replay_case["frozen_bytes"],
            report_sha="0" * 64,
        ),
    )

    with pytest.raises(
        composer.CompositionError,
        match="generator report disagrees with rebuilt bytes",
    ):
        composer._fresh_rebuild_activity_visual_artifact(**replay_case["kwargs"])


def test_fresh_rebuild_rejects_byte_mismatch_even_if_sha_checks_are_spoofed(
    replay_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_sha = replay_case["kwargs"]["expected_visual_sha256"]
    real_sha256_file = composer.sha256_file

    def sha256_file_with_synthetic_collision(path: Path) -> str:
        if Path(path).name == "rebuilt.json":
            return expected_sha
        return real_sha256_file(path)

    monkeypatch.setattr(composer, "sha256_file", sha256_file_with_synthetic_collision)
    monkeypatch.setattr(
        composer.subprocess,
        "run",
        _completed_run(
            rebuilt_bytes=b'{"rebuilt":true}\n',
            report_sha=expected_sha,
        ),
    )

    with pytest.raises(composer.CompositionError, match="is not byte-identical"):
        composer._fresh_rebuild_activity_visual_artifact(**replay_case["kwargs"])


def test_fresh_rebuild_rejects_different_python_executable(
    replay_case: dict[str, Any],
) -> None:
    other_python = replay_case["root"] / "other-python.exe"
    other_python.write_bytes(b"not the current interpreter")
    replay_case["payload"]["runtime"]["python"]["executable"] = str(other_python)

    with pytest.raises(
        composer.CompositionError,
        match="frozen current Python executable",
    ):
        composer._fresh_rebuild_activity_visual_artifact(**replay_case["kwargs"])


def test_fresh_rebuild_rejects_poppler_hash_mismatch(
    replay_case: dict[str, Any],
) -> None:
    replay_case["payload"]["runtime"]["poppler"]["pdftoppm"]["sha256"] = "0" * 64

    with pytest.raises(
        composer.CompositionError,
        match="activity visual pdftoppm SHA-256 mismatch",
    ):
        composer._fresh_rebuild_activity_visual_artifact(**replay_case["kwargs"])


def test_image_only_rebuild_uses_its_isolated_profile_and_index(
    replay_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = replay_case["root"]
    image_only_generator = (
        root / "scripts" / "build_maxim_image_only_activity_visual_binding_v1.py"
    )
    image_only_generator.write_text(
        "# synthetic image-only generator\n", encoding="utf-8"
    )
    paths = replay_case["kwargs"]
    replay_case["payload"]["inputs"].update(
        {
            "profile": {
                "path": str(paths["profile_path"]),
                "sha256": _sha256(paths["profile_path"]),
            },
            "parser_observations": {
                "path": str(paths["parser_path"]),
                "sha256": _sha256(paths["parser_path"]),
            },
            "source_locators": {
                "path": str(paths["locator_path"]),
                "sha256": _sha256(paths["locator_path"]),
            },
            "source_index": {
                "path": str(paths["source_index_path"]),
                "sha256": _sha256(paths["source_index_path"]),
            },
        }
    )
    monkeypatch.setattr(
        composer.subprocess,
        "run",
        _completed_run(rebuilt_bytes=replay_case["frozen_bytes"]),
    )

    result = composer._fresh_rebuild_image_only_activity_visual_artifact(
        parser_path=paths["parser_path"],
        locator_path=paths["locator_path"],
        visual_path=paths["visual_path"],
        expected_visual_sha256=paths["expected_visual_sha256"],
        visual_payload=replay_case["payload"],
        document_pdf_paths=paths["document_pdf_paths"],
    )

    assert result["mode"] == (
        "fresh_source_only_poppler_sift_image_only_activity_exact_bytes_v1"
    )
    assert result["generator"]["path"] == str(image_only_generator)
    assert result["exact_byte_identity"] is True


def _valid_reproduction(visual_path: Path, visual_sha: str) -> dict[str, Any]:
    return {
        "mode": "fresh_source_only_poppler_sift_exact_bytes_v1",
        "generator": {"path": "scripts/synthetic_generator.py", "sha256": "1" * 64},
        "frozen_artifact": {
            "path": str(visual_path),
            "sha256": visual_sha,
            "size_bytes": 17,
        },
        "reproduced_artifact": {"sha256": visual_sha, "size_bytes": 17},
        "exact_byte_identity": True,
        "command_projection_sha256": "2" * 64,
        "runtime": {
            "python_executable_sha256": "3" * 64,
            "pdftoppm_sha256": "4" * 64,
            "pdfinfo_sha256": "5" * 64,
            "runtime_projection_sha256": "6" * 64,
        },
        "summary": {"raw_page_evidences": 1},
        "benchmark_answer_candidate_outcome_artifacts_read": False,
    }


def test_image_judge_accepts_valid_visual_reproduction(tmp_path: Path) -> None:
    visual_path = tmp_path / "visual.json"
    visual_sha = "a" * 64

    image_judge._require_activity_visual_reproduction(
        profile_visual_spec={"sha256": visual_sha},
        composition_manifest={
            "activity_visual_reproduction": _valid_reproduction(
                visual_path, visual_sha
            )
        },
        visual_path=visual_path,
    )


def test_image_judge_rejects_missing_visual_reproduction(tmp_path: Path) -> None:
    with pytest.raises(image_judge.JudgeBuildError, match="exact source-only"):
        image_judge._require_activity_visual_reproduction(
            profile_visual_spec={"sha256": "a" * 64},
            composition_manifest={},
            visual_path=tmp_path / "visual.json",
        )


def test_image_judge_rejects_false_byte_identity(tmp_path: Path) -> None:
    visual_path = tmp_path / "visual.json"
    visual_sha = "a" * 64
    reproduction = _valid_reproduction(visual_path, visual_sha)
    reproduction["exact_byte_identity"] = False

    with pytest.raises(image_judge.JudgeBuildError, match="exact source-only"):
        image_judge._require_activity_visual_reproduction(
            profile_visual_spec={"sha256": visual_sha},
            composition_manifest={"activity_visual_reproduction": reproduction},
            visual_path=visual_path,
        )


def test_image_judge_rejects_mismatched_visual_sha(tmp_path: Path) -> None:
    visual_path = tmp_path / "visual.json"
    reproduction = _valid_reproduction(visual_path, "a" * 64)

    with pytest.raises(image_judge.JudgeBuildError, match="attestation is malformed"):
        image_judge._require_activity_visual_reproduction(
            profile_visual_spec={"sha256": "b" * 64},
            composition_manifest={"activity_visual_reproduction": reproduction},
            visual_path=visual_path,
        )


def test_image_judge_rejects_source_guard_true(tmp_path: Path) -> None:
    visual_path = tmp_path / "visual.json"
    visual_sha = "a" * 64
    reproduction = _valid_reproduction(visual_path, visual_sha)
    reproduction["benchmark_answer_candidate_outcome_artifacts_read"] = True

    with pytest.raises(image_judge.JudgeBuildError, match="exact source-only"):
        image_judge._require_activity_visual_reproduction(
            profile_visual_spec={"sha256": visual_sha},
            composition_manifest={"activity_visual_reproduction": reproduction},
            visual_path=visual_path,
        )


def test_image_judge_rejects_stray_reproduction_for_nonvisual_profile(
    tmp_path: Path,
) -> None:
    visual_path = tmp_path / "visual.json"
    reproduction = _valid_reproduction(visual_path, "a" * 64)

    with pytest.raises(image_judge.JudgeBuildError, match="stray visual reproduction"):
        image_judge._require_activity_visual_reproduction(
            profile_visual_spec=None,
            composition_manifest={"activity_visual_reproduction": reproduction},
            visual_path=None,
        )


def test_image_only_reproduction_is_independent_when_normal_visual_is_absent(
    tmp_path: Path,
) -> None:
    image_only_path = tmp_path / "image-only.json"
    image_only_sha = "b" * 64
    reproduction = _valid_reproduction(image_only_path, image_only_sha)
    reproduction["mode"] = (
        "fresh_source_only_poppler_sift_image_only_activity_exact_bytes_v1"
    )
    composition_manifest = {
        "image_only_activity_visual_reproduction": reproduction
    }

    image_judge._require_activity_visual_reproduction(
        profile_visual_spec=None,
        composition_manifest=composition_manifest,
        visual_path=None,
    )
    image_judge._require_activity_visual_reproduction(
        profile_visual_spec={"sha256": image_only_sha},
        composition_manifest=composition_manifest,
        visual_path=image_only_path,
        manifest_key="image_only_activity_visual_reproduction",
        expected_mode=(
            "fresh_source_only_poppler_sift_image_only_activity_exact_bytes_v1"
        ),
    )
