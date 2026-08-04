from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROFILE = (
    REPO_ROOT
    / "reports"
    / "maxim_final_meta_verifier_v2_20260803"
    / "preregistered_profile.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load_module(
    "maxim_final_meta_prepare_v2_test",
    SCRIPT_DIR / "prepare_maxim_final_meta_verifier_v2.py",
)
runner = load_module(
    "maxim_final_meta_runner_v2_test",
    SCRIPT_DIR / "run_maxim_final_meta_verifier_v2.py",
)


class ExpandedMetaVerifierTests(unittest.TestCase):
    def test_v1_preregistered_code_is_immutable(self) -> None:
        expected = {
            "prepare_maxim_final_meta_verifier_v1.py": "57210c3408df4701da27d34442cd9c96384c4f6ad5cc7216685e7355c01bbf85",
            "run_maxim_final_meta_verifier_v1.py": "abcc5996b6c6555e2052fc37af474a0140d317c43382fdee9938500f837db181",
            "test_maxim_final_meta_verifier_v1.py": "5cf44c8a9bc3571e36e41d3caacf6e95a1566d9a492022338e765ed287c5a7be",
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((SCRIPT_DIR / name).read_bytes()).hexdigest(), digest)

    def test_exact_ten_candidate_contract(self) -> None:
        self.assertEqual(
            prep.REQUIRED_CANDIDATE_SLOTS,
            (
                "subject_router",
                "raw_verifier",
                "structural_rag",
                "mi_rag",
                "active_vision",
                "tiled_vision",
                "native_thinking_v4",
                "budgeted_thinking_v5",
                "stronger_27b_hard86_composite",
                "stronger_27b_direct",
            ),
        )
        self.assertEqual(prep.OPAQUE_IDS, tuple(f"C{i}" for i in range(1, 11)))
        profile = prep.load_json(PROFILE)
        prep.validate_profile(profile)

    def test_v2_shuffle_is_deterministic_and_distinct_from_v1_seed(self) -> None:
        first = prep.blind_order("val_0001", prep.DEFAULT_BLINDING_SEED)
        self.assertEqual(first, prep.blind_order("val_0001", prep.DEFAULT_BLINDING_SEED))
        self.assertEqual(set(first), set(prep.REQUIRED_CANDIDATE_SLOTS))
        orders = {
            tuple(prep.blind_order(f"val_{index:04d}", prep.DEFAULT_BLINDING_SEED))
            for index in range(30)
        }
        self.assertGreater(len(orders), 1)
        self.assertTrue(prep.DEFAULT_BLINDING_SEED.endswith("-v2"))

    def test_candidate_payload_respects_16k_budget_contract(self) -> None:
        row = {
            "final_answer": "A" * 500,
            "condition": "maxim_active_visual_crop_sketchpad_v1",
            "model": "Qwen/Qwen3.5-27B",
            "prompt_version": "active_vision",
            "reasoning": "active_vision " + "r" * 3000,
            "solution_steps": "Qwen/Qwen3.5-27B " + "s" * 3000,
            "generation": {
                "gold_access": False,
                "visual_facts": ["e" * 500, "f" * 500, "g" * 500],
            },
        }
        payload = prep.bounded_candidate_payload(row, "active_vision")
        self.assertLessEqual(len(payload["final_answer"]), 120)
        self.assertLessEqual(len(payload["bounded_reasoning"]), 650)
        self.assertLessEqual(len(payload["bounded_evidence"]), 2)
        self.assertTrue(all(len(item) <= 160 for item in payload["bounded_evidence"]))
        serialized = json.dumps(payload).casefold()
        self.assertNotIn("active_vision", serialized)
        worst_case = 10 * (120 + 650 + 2 * 160)
        self.assertEqual(worst_case, 10900)
        self.assertEqual(
            prep.load_json(PROFILE)["prompt_policy"][
                "maximum_candidate_payload_chars_excluding_labels"
            ],
            worst_case,
        )

    def test_prepare_separates_ten_way_key_without_gold_or_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark.jsonl"
            benchmark.write_text(
                json.dumps(
                    {
                        "task_id": "val_0001",
                        "subject": "Math",
                        "question": "visible question",
                        "answer_type": "choice",
                        "question_images": [{"data": "images/x.png"}],
                        "reference_answer": "A",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            candidates = {}
            for index, slot in enumerate(prep.REQUIRED_CANDIDATE_SLOTS):
                path = root / f"candidate_{index}.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "task_id": "val_0001",
                            "condition": slot,
                            "model": f"hidden-model-{index}",
                            "prompt_version": f"hidden-prompt-{index}",
                            "final_answer": chr(ord("A") + index % 5),
                            "reasoning": f"reason {slot}",
                            "solution_steps": "check the image",
                            "generation": {"gold_access": False},
                            "error": None,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                candidates[slot] = path
            queue = root / "queue.jsonl"
            key = root / "key.jsonl"
            manifest = prep.prepare_queue(
                benchmark_path=benchmark,
                candidate_paths=candidates,
                profile_path=PROFILE,
                queue_path=queue,
                private_key_path=key,
                manifest_path=root / "manifest.json",
                enforce_frozen=False,
            )
            public_text = queue.read_text(encoding="utf-8")
            public = json.loads(public_text)
            private = json.loads(key.read_text(encoding="utf-8"))
            self.assertEqual(len(public["candidates"]), 10)
            self.assertEqual(
                [row["candidate_id"] for row in public["candidates"]],
                list(prep.OPAQUE_IDS),
            )
            self.assertNotIn("reference_answer", public_text)
            for slot in prep.REQUIRED_CANDIDATE_SLOTS:
                self.assertNotIn(slot, public_text)
            self.assertEqual(
                set(private["opaque_to_source_slot"].values()),
                set(prep.REQUIRED_CANDIDATE_SLOTS),
            )
            self.assertFalse(manifest["candidate_scores_loaded"])
            self.assertFalse(manifest["judge_artifacts_loaded"])

    def test_dynamic_verdict_schema_requires_all_ten_checks(self) -> None:
        schema = runner.verdict_schema(prep.OPAQUE_IDS)
        checks = schema["properties"]["candidate_checks"]
        self.assertEqual(checks["minItems"], 10)
        self.assertEqual(checks["maxItems"], 10)
        self.assertEqual(
            checks["items"]["properties"]["candidate_id"]["enum"],
            list(prep.OPAQUE_IDS),
        )

    def test_original_image_still_precedes_all_ten_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "x.png").write_bytes(b"placeholder")
            row = {
                "task_id": "val_0001",
                "subject": "Math",
                "question": "solve",
                "answer_type": "choice",
                "question_images": [{"data": "images/x.png"}],
                "candidates": [
                    {
                        "candidate_id": opaque,
                        "final_answer": "A",
                        "bounded_reasoning": "bounded",
                        "bounded_evidence": ["visible evidence"],
                    }
                    for opaque in prep.OPAQUE_IDS
                ],
            }
            content = runner.build_messages(
                row, image_root=root, image_url_root="file:///images"
            )[1]["content"]
            self.assertEqual([block["type"] for block in content], ["text", "image_url", "text"])
            self.assertIn("ORIGINAL QUESTION", content[0]["text"])
            self.assertEqual(sum(f"{opaque}\n" in content[2]["text"] for opaque in prep.OPAQUE_IDS), 10)

    def test_failclosed_policy_and_content_exact_router_copy_are_unchanged(self) -> None:
        router_row = {
            "task_id": "val_0001",
            "condition": "maxim_subject_router_v1",
            "final_answer": "A",
            "generation": {"gold_access": False},
            "error": None,
        }
        solver, audit = runner.compose_solver_row(
            result={
                "task_id": "val_0001",
                "verdict": None,
                "error": "timeout",
            },
            router_row=router_row,
            min_confidence=0.7,
            min_evidence=2,
            queue_sha256="q" * 64,
        )
        self.assertEqual(solver, router_row)
        self.assertIsNot(solver, router_row)
        self.assertEqual(audit["selection"]["selected_source"], "router")
        self.assertEqual(runner.CONDITION, "maxim_final_gold_blind_meta_verifier_v2")


if __name__ == "__main__":
    unittest.main()
