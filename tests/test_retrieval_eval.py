import json
import tempfile
import unittest
from pathlib import Path

from vlm_judge.retrieval import build_bm25_index
from vlm_judge.retrieval_eval import evaluate_retrieval, prepare_qrels_template


class RetrievalEvaluationTests(unittest.TestCase):
    def test_metrics_and_qrels_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks.jsonl"
            chunks.write_text(
                "\n".join(
                    json.dumps(value)
                    for value in [
                        {
                            "chunk_id": "c1",
                            "page_id": "p1",
                            "kind": "text",
                            "text": "fraction addition equal denominators",
                            "metadata": {"subject": "math", "grade": 4},
                        },
                        {
                            "chunk_id": "c2",
                            "page_id": "p2",
                            "kind": "text",
                            "text": "triangle area base height",
                            "metadata": {"subject": "math", "grade": 5},
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            index = root / "bm25.sqlite"
            build_bm25_index(chunks, index)
            qrels = root / "qrels.jsonl"
            qrels.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "query": "fraction denominators",
                        "subject": "math",
                        "relevant_page_ids": ["p1"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = evaluate_retrieval(index, qrels, ks=(1, 2))
            self.assertEqual(report["queries"], 1)
            self.assertEqual(report["metrics"]["hit_rate_at_1"], 1.0)
            self.assertEqual(report["metrics"]["mrr_at_2"], 1.0)
            self.assertEqual(report["metrics"]["ndcg_at_2"], 1.0)

            benchmark = root / "benchmark.jsonl"
            benchmark.write_text(
                json.dumps(
                    {
                        "task_id": "t1",
                        "question_text": "Find the area",
                        "subject": "math",
                        "grade": 5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            template = root / "template.jsonl"
            summary = prepare_qrels_template(benchmark, template)
            self.assertEqual(summary["records"], 1)
            self.assertEqual(json.loads(template.read_text(encoding="utf-8"))["query"], "Find the area")


if __name__ == "__main__":
    unittest.main()
