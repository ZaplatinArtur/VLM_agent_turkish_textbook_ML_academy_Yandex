from pathlib import Path
import subprocess
import sys

from experiments.maxim_9b_ykslop_teslov_retrieval_ablation_v1_20260812.run_base_bge_census import _select
from experiments.maxim_9b_ykslop_teslov_retrieval_ablation_v1_20260812.teslov_retrieval import TheoryChunk


HERE = Path(__file__).resolve().parent


def chunk(name: str, size: int) -> TheoryChunk:
    return TheoryChunk(name, "x" * size, 9, "matematik")


def test_context_budget_matches_frozen_skip_not_stop_policy():
    selected = _select(
        [(chunk("first", 4000), 3.0), (chunk("skip", 2000), 2.0), (chunk("fit", 1000), 1.0)]
    )
    assert [item[0].chunk_id for item in selected] == ["first", "fit"]


def test_first_context_may_exceed_budget_but_result_is_bounded():
    selected = _select([(chunk("huge", 6000), 2.0), (chunk("later", 10), 1.0)])
    assert [item[0].chunk_id for item in selected] == ["huge"]


def test_census_script_imports_when_launched_directly(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(HERE / "run_base_bge_census.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--device" in completed.stdout
