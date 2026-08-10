from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PREP_SCRIPT = SCRIPT_DIR / "prepare_maxim_final_meta_verifier_v1.py"
RUN_SCRIPT = SCRIPT_DIR / "run_maxim_final_meta_verifier_v1.py"
PROFILE = (
    SCRIPT_DIR.parent
    / "reports"
    / "maxim_final_meta_verifier_v1_20260803"
    / "preregistered_profile.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load_module("maxim_final_meta_prep_test", PREP_SCRIPT)
runner = load_module("maxim_final_meta_runner_test", RUN_SCRIPT)


def valid_verdict() -> dict:
    return {
        "question_reconstruction": "The image asks which option satisfies the condition.",
        "decisive_evidence": ["The stem contains a negation.", "Option B matches the visible table."],
        "candidate_checks": [
            {
                "candidate_id": candidate_id,
                "status": "supported" if candidate_id == "C2" else "uncertain",
                "verification": "Checked directly against the original image.",
            }
            for candidate_id in prep.OPAQUE_IDS
        ],
        "independent_reasoning": "After re-solving the visible question, option B is the only match.",
        "final_answer": "B",
        "confidence": 0.86,
        "answer_format_verified": True,
        "abstain": False,
    }


class FakePool:
    model = "Qwen/Qwen3.5-27B"

    def complete(self, **_kwargs):
        return {
            "parsed": valid_verdict(),
            "raw": json.dumps(valid_verdict()),
            "endpoint": "http://example.invalid/v1",
            "finish_reason": "stop",
            "attempt": 1,
            "latency_s": 1.0,
            "input_tokens": 10,
            "output_tokens": 20,
            "recovered_partial": False,
            "parse_error": None,
        }


class MetaVerifierTests(unittest.TestCase):
    def test_blind_order_is_deterministic_and_task_dependent(self) -> None:
        first = prep.blind_order("val_0001", prep.DEFAULT_BLINDING_SEED)
        self.assertEqual(first, prep.blind_order("val_0001", prep.DEFAULT_BLINDING_SEED))
        self.assertEqual(set(first), set(prep.REQUIRED_CANDIDATE_SLOTS))
        orders = {
            tuple(prep.blind_order(f"val_{index:04d}", prep.DEFAULT_BLINDING_SEED))
            for index in range(30)
        }
        self.assertGreater(len(orders), 1)

    def test_queue_audit_rejects_gold_scores_and_source_identity(self) -> None:
        for payload in (
            {"reference_answer": "A"},
            {"score": 0.9},
            {"candidates": [{"system_id": "router"}]},
        ):
            with self.assertRaises(prep.PreparationError):
                prep.audit_gold_free(payload)
        with self.assertRaises(prep.PreparationError):
            prep.audit_candidate_solver_row(
                {
                    "generation": {"gold_access": False},
                    "judge_verdict": "OK",
                },
                slot="subject_router",
                task_id="val_0001",
            )

    def test_bounded_payload_redacts_identity_and_enforces_limits(self) -> None:
        row = {
            "final_answer": "A",
            "condition": "maxim_active_visual_crop_sketchpad_v1",
            "model": "Qwen/Qwen3.5-27B",
            "prompt_version": "active_vision",
            "reasoning": "active_vision says " + "x" * 3000,
            "solution_steps": "Qwen/Qwen3.5-27B checked the image",
            "generation": {"visual_facts": ["visible 12", "visible minus sign"]},
        }
        payload = prep.bounded_candidate_payload(row, "active_vision")
        self.assertEqual(payload["final_answer"], "A")
        self.assertLessEqual(len(payload["bounded_reasoning"]), 1200)
        self.assertNotIn("active_vision", payload["bounded_reasoning"].casefold())
        self.assertNotIn("qwen/qwen3.5-27b", payload["bounded_reasoning"].casefold())
        self.assertEqual(payload["bounded_evidence"][:2], ["visible 12", "visible minus sign"])

    def test_prepare_writes_public_queue_and_separate_private_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark.jsonl"
            benchmark.write_text(
                json.dumps(
                    {
                        "task_id": "val_0001",
                        "subject": "Math",
                        "grade": 7,
                        "question": "question in image",
                        "answer_type": "choice",
                        "question_images": [
                            {"data": "images/val_0001.png", "mime_type": "image/png"}
                        ],
                        "reference_answer": "A",
                        "reference_solution": "secret",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            candidates = {}
            for index, slot in enumerate(prep.REQUIRED_CANDIDATE_SLOTS):
                path = root / f"{slot}.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "task_id": "val_0001",
                            "condition": slot,
                            "model": f"hidden-{index}",
                            "final_answer": chr(ord("A") + index % 5),
                            "reasoning": f"reason from {slot}",
                            "solution_steps": "visible check",
                            "generation": {"gold_access": False},
                            "error": None,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                candidates[slot] = path
            queue = root / "queue.jsonl"
            key = root / "private_key.jsonl"
            manifest_path = root / "manifest.json"
            manifest = prep.prepare_queue(
                benchmark_path=benchmark,
                candidate_paths=candidates,
                profile_path=PROFILE,
                queue_path=queue,
                private_key_path=key,
                manifest_path=manifest_path,
                enforce_frozen=False,
            )
            public_text = queue.read_text(encoding="utf-8")
            public_row = json.loads(public_text)
            private_row = json.loads(key.read_text(encoding="utf-8"))
            self.assertNotIn("reference_answer", public_text)
            self.assertNotIn("reference_solution", public_text)
            for slot in prep.REQUIRED_CANDIDATE_SLOTS:
                self.assertNotIn(slot, public_text)
            self.assertEqual(
                [row["candidate_id"] for row in public_row["candidates"]],
                list(prep.OPAQUE_IDS),
            )
            self.assertEqual(
                set(private_row["opaque_to_source_slot"].values()),
                set(prep.REQUIRED_CANDIDATE_SLOTS),
            )
            self.assertFalse(manifest["candidate_scores_loaded"])
            self.assertFalse(manifest["judge_artifacts_loaded"])
            self.assertTrue(
                manifest["private_routing_key"]["must_not_be_mounted_into_or_loaded_by_model_runner"]
            )

    def test_prepare_rejects_incomplete_or_error_solver(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark.jsonl"
            benchmark.write_text(
                json.dumps(
                    {
                        "task_id": "val_0001",
                        "question_images": [{"data": "images/x.png"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            paths = {}
            for slot in prep.REQUIRED_CANDIDATE_SLOTS:
                path = root / f"{slot}.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "task_id": "val_0001",
                            "final_answer": "A",
                            "generation": {"gold_access": False},
                            "error": "failed" if slot == "tiled_vision" else None,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                paths[slot] = path
            with self.assertRaises(prep.PreparationError):
                prep.prepare_queue(
                    benchmark_path=benchmark,
                    candidate_paths=paths,
                    profile_path=PROFILE,
                    queue_path=root / "queue",
                    private_key_path=root / "key",
                    manifest_path=root / "manifest",
                    enforce_frozen=False,
                )

    def test_original_content_precedes_anonymous_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "val_0001.png").write_bytes(b"image-placeholder")
            row = {
                "task_id": "val_0001",
                "subject": "Math",
                "grade": 7,
                "question": "Solve the visible equation",
                "answer_type": "choice",
                "question_images": [{"data": "images/val_0001.png"}],
                "candidates": [
                    {
                        "candidate_id": opaque,
                        "final_answer": "A",
                        "bounded_reasoning": "bounded",
                        "bounded_evidence": ["evidence"],
                    }
                    for opaque in prep.OPAQUE_IDS
                ],
            }
            messages = runner.build_messages(
                row, image_root=root, image_url_root="file:///images"
            )
            content = messages[1]["content"]
            self.assertEqual([block["type"] for block in content], ["text", "image_url", "text"])
            self.assertIn("ORIGINAL QUESTION", content[0]["text"])
            self.assertIn("ANONYMOUS", content[2]["text"])

    def test_strict_verdict_validation_and_choice_format(self) -> None:
        value = runner.validate_verdict(
            valid_verdict(), candidate_ids=prep.OPAQUE_IDS, answer_type="choice"
        )
        self.assertEqual(value["final_answer"], "B")
        broken = valid_verdict()
        broken["candidate_checks"][1]["candidate_id"] = "C1"
        with self.assertRaises(runner.MetaVerifierError):
            runner.validate_verdict(
                broken, candidate_ids=prep.OPAQUE_IDS, answer_type="choice"
            )
        broken_answer = valid_verdict()
        broken_answer["final_answer"] = "option B"
        with self.assertRaises(runner.MetaVerifierError):
            runner.validate_verdict(
                broken_answer, candidate_ids=prep.OPAQUE_IDS, answer_type="choice"
            )

    def test_frozen_policy_uses_meta_only_above_all_gates(self) -> None:
        result = {"verdict": valid_verdict(), "error": None}
        self.assertEqual(
            runner.apply_frozen_policy(result, min_confidence=0.7, min_evidence=2),
            ("meta_verifier", "B", "valid_supported_meta_answer"),
        )
        errored = {"verdict": None, "error": "timeout"}
        self.assertEqual(
            runner.apply_frozen_policy(errored, min_confidence=0.7, min_evidence=2)[0],
            "router",
        )
        abstained = valid_verdict()
        abstained["abstain"] = True
        self.assertEqual(
            runner.apply_frozen_policy(
                {"verdict": abstained, "error": None},
                min_confidence=0.7,
                min_evidence=2,
            )[0],
            "router",
        )

    def test_error_fallback_is_content_exact_router_row(self) -> None:
        router_row = {
            "task_id": "val_0001",
            "condition": "maxim_subject_router_v1",
            "final_answer": "A",
            "reasoning": "router",
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

    def test_run_one_accepts_valid_structured_independent_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "x.png").write_bytes(b"image-placeholder")
            payload = {
                "schema_version": prep.QUEUE_SCHEMA_VERSION,
                "queue_index": 0,
                "task_id": "val_0001",
                "subject": "Math",
                "grade": 7,
                "question": "question",
                "answer_type": "choice",
                "question_images": [{"data": "images/x.png"}],
                "candidates": [
                    {
                        "candidate_id": opaque,
                        "final_answer": "A",
                        "bounded_reasoning": "bounded",
                        "bounded_evidence": ["evidence"],
                    }
                    for opaque in prep.OPAQUE_IDS
                ],
            }
            row = {**payload, "request_sha256": prep.stable_sha256(payload)}
            result = runner.run_one(
                row,
                pool=FakePool(),
                image_root=root,
                image_url_root="file:///images",
                max_tokens=4096,
                base_seed=20260803,
                semantic_attempts=2,
            )
            self.assertIsNone(result["error"])
            self.assertEqual(result["verdict"]["final_answer"], "B")


if __name__ == "__main__":
    unittest.main()
