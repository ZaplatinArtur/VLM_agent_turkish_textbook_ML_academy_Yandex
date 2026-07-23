from pathlib import Path

from mla_baseline.paired_eval import build_report, render_markdown


def _task(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "subject": "Math",
        "grade": 7,
        "question": "Soru",
        "reference_answer": "A",
        "answer_type": "choice",
    }


def _result(task_id: str, answer: str, *, tool_calls=None) -> dict:
    return {
        "task_id": task_id,
        "condition": "agent_rag",
        "final_answer": answer,
        "tool_calls": tool_calls or [],
        "error": None,
    }


def _judge(task_id: str, score: int) -> dict:
    return {"task_id": task_id, "verdict": {"score": score, "rationale": "ok"}}


def test_report_uses_full_judge_denominator_and_counts_answer_flips(
    tmp_path: Path,
) -> None:
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "7-sinif-matematik-ders-kitabi.jsonl").write_text("")
    tasks = [_task(task_id) for task_id in ("a", "b", "c", "d")]
    baseline = [
        _result("a", "A"),
        _result("b", "B"),
        _result("c", "A"),
        _result("d", "B"),
    ]
    rag = [
        _result("a", "A"),
        _result(
            "b",
            "A",
            tool_calls=[
                {
                    "returned_chunk_ids": ["book:1"],
                    "latency_ms": 5.0,
                    "error": None,
                }
            ],
        ),
        _result("c", "B"),
        _result("d", "B"),
    ]
    report = build_report(
        tasks,
        baseline,
        rag,
        baseline_judge=[_judge("a", 1), _judge("b", 0), _judge("c", 1)],
        rag_judge=[_judge("a", 1), _judge("b", 1), _judge("c", 0), _judge("d", 0)],
        chunks_dir=chunks,
    )

    judge = report["judge_full"]
    assert judge["denominator"] == 4
    assert judge["baseline_correct"] == 2
    assert judge["rag_correct"] == 2
    assert judge["baseline_evaluated"] == 3
    assert judge["fixed_by_rag"] == 1
    assert judge["regressed_with_rag"] == 1
    assert report["corpus_coverage"]["counts"] == {"covered": 4}
    assert report["tool_usage"]["tasks_with_retrieval_hits"] == 1
    assert report["judge_flip_cases"][0]["direction"] == "fixed_by_rag"
    assert report["judge_flip_cases"][0]["task_id"] == "b"
    assert "RAG исправил 1 ответов и ухудшил 1" in render_markdown(report)


def test_automatic_metric_counts_missing_agent_result_as_wrong() -> None:
    report = build_report(
        [_task("a"), _task("b")],
        [_result("a", "A")],
        [_result("a", "A"), _result("b", "A")],
    )

    assert report["automatic"]["denominator"] == 2
    assert report["automatic"]["baseline_correct"] == 1
    assert report["automatic"]["rag_correct"] == 2
    assert report["task_set"]["missing_baseline_results"] == 1
