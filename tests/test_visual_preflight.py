import json

import numpy as np
import pytest

from mla_baseline.visual_preflight import (
    check_adapter,
    check_backend_selected,
    check_index,
    check_prompt_profile,
    check_threshold,
    report,
    run_checks,
)


@pytest.fixture
def index_dir(tmp_path):
    root = tmp_path / "colqwen25_turkish_full_index"
    root.mkdir()
    adapter = tmp_path / "models" / "colqwen25_turkish"
    adapter.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps({"model": "vidore/colqwen2.5-base", "adapter": str(adapter),
                    "scoring": "pooled_then_maxsim"}),
        encoding="utf-8",
    )
    (root / "pages.jsonl").write_text("{}\n", encoding="utf-8")
    for name in ("offsets.npy", "tokens.f16.npy", "pooled.f16.npy"):
        np.save(root / name, np.zeros(1, dtype=np.float16), allow_pickle=False)
    return root


def test_backend_must_be_switched_on(monkeypatch):
    monkeypatch.delenv("MLA_RETRIEVAL_BACKEND", raising=False)
    assert check_backend_selected().ok is False
    monkeypatch.setenv("MLA_RETRIEVAL_BACKEND", "visual")
    assert check_backend_selected().ok is True


def test_threshold_must_be_set_and_numeric(monkeypatch):
    monkeypatch.delenv("MLA_VISUAL_MIN_SCORE", raising=False)
    failed = check_threshold()
    assert failed.ok is False
    assert "calibrate_visual" in failed.detail

    monkeypatch.setenv("MLA_VISUAL_MIN_SCORE", "не число")
    assert check_threshold().ok is False

    monkeypatch.setenv("MLA_VISUAL_MIN_SCORE", "12.5")
    assert check_threshold().ok is True


def test_index_files_are_checked_by_name(index_dir, monkeypatch):
    monkeypatch.setenv("MLA_VISUAL_INDEX_DIR", str(index_dir))
    assert check_index().ok is True

    (index_dir / "tokens.f16.npy").unlink()
    broken = check_index()
    assert broken.ok is False
    assert "tokens.f16.npy" in broken.detail


def test_missing_index_variable_is_named(monkeypatch):
    monkeypatch.delenv("MLA_VISUAL_INDEX_DIR", raising=False)
    assert "MLA_VISUAL_INDEX_DIR" in check_index().detail


def test_adapter_recorded_in_meta_must_exist(index_dir, monkeypatch):
    monkeypatch.setenv("MLA_VISUAL_INDEX_DIR", str(index_dir))
    assert check_adapter().ok is True

    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    meta["adapter"] = str(index_dir / "gone")
    (index_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    stale = check_adapter()
    assert stale.ok is False
    assert "которого нет на диске" in stale.detail


def test_prompt_profile_is_a_warning_not_a_blocker(monkeypatch):
    monkeypatch.delenv("MLA_PROMPT_VERSION", raising=False)
    check = check_prompt_profile()
    assert check.ok is False
    assert check.fatal is False

    monkeypatch.setenv("MLA_PROMPT_VERSION", "v2_cot_text_rag_v1")
    assert check_prompt_profile().ok is True


def test_live_probe_is_skipped_while_config_is_broken(monkeypatch):
    monkeypatch.delenv("MLA_RETRIEVAL_BACKEND", raising=False)
    monkeypatch.delenv("MLA_VISUAL_MIN_SCORE", raising=False)
    names = [check.name for check in run_checks()]
    assert "пробный запрос" not in names


def test_report_exit_code_follows_fatal_checks(index_dir, monkeypatch, capsys):
    monkeypatch.setenv("MLA_RETRIEVAL_BACKEND", "visual")
    monkeypatch.setenv("MLA_VISUAL_MIN_SCORE", "12.5")
    monkeypatch.setenv("MLA_VISUAL_INDEX_DIR", str(index_dir))
    monkeypatch.delenv("MLA_PROMPT_VERSION", raising=False)

    checks = run_checks(live=False)
    code = report(checks)
    output = capsys.readouterr().out
    # Промпт-профиль не задан, но это предупреждение: код возврата нулевой,
    # только если нет ни одного фатального провала.
    assert "WARN" in output
    assert code == (0 if all(c.ok for c in checks if c.fatal) else 1)
