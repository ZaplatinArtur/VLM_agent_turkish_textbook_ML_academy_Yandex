from __future__ import annotations

import hashlib
import importlib.util
import json
import copy
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
REPORT_DIR = REPO_ROOT / "reports" / "maxim_final_meta_verifier_v3_20260803"
PROFILE = REPORT_DIR / "preregistered_profile.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load_module(
    "maxim_final_meta_prepare_v3_test",
    SCRIPT_DIR / "prepare_maxim_final_meta_verifier_v3.py",
)
runner = load_module(
    "maxim_final_meta_runner_v3_test",
    SCRIPT_DIR / "run_maxim_final_meta_verifier_v3.py",
)


class TwelveCandidateMetaVerifierTests(unittest.TestCase):
    def test_v1_and_v2_code_remain_immutable(self) -> None:
        expected = {
            "prepare_maxim_final_meta_verifier_v1.py": "57210c3408df4701da27d34442cd9c96384c4f6ad5cc7216685e7355c01bbf85",
            "run_maxim_final_meta_verifier_v1.py": "abcc5996b6c6555e2052fc37af474a0140d317c43382fdee9938500f837db181",
            "test_maxim_final_meta_verifier_v1.py": "5cf44c8a9bc3571e36e41d3caacf6e95a1566d9a492022338e765ed287c5a7be",
            "prepare_maxim_final_meta_verifier_v2.py": "42e869b6e70b5f3872498a3befc4500ebb2bc22d090bd223cf4d80d72cdd0b21",
            "run_maxim_final_meta_verifier_v2.py": "dad95227c7002f3601a94c3b48da709e958e7bca9610241683db8469d8dd91e6",
            "test_maxim_final_meta_verifier_v2.py": "6f2f78eae478c8cc85be9cad1fcf9d9a68611df4bc9daa7df60b18de551624e4",
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((SCRIPT_DIR / name).read_bytes()).hexdigest(), digest)

    def test_exact_twelve_candidate_contract(self) -> None:
        self.assertEqual(
            prep.REQUIRED_CANDIDATE_SLOTS[-2:],
            (
                "paired_rag_norag_semantic_support_on_pinned_structural_context_v1",
                "literal_parallel8_lowconf_v1",
            ),
        )
        self.assertEqual(len(prep.REQUIRED_CANDIDATE_SLOTS), 12)
        self.assertEqual(prep.OPAQUE_IDS, tuple(f"C{i}" for i in range(1, 13)))
        prep.validate_profile(prep.load_json(PROFILE))

    def test_new_shuffle_is_deterministic_per_task(self) -> None:
        first = prep.blind_order("val_0001", prep.DEFAULT_BLINDING_SEED)
        self.assertEqual(first, prep.blind_order("val_0001", prep.DEFAULT_BLINDING_SEED))
        self.assertEqual(set(first), set(prep.REQUIRED_CANDIDATE_SLOTS))
        self.assertNotEqual(
            prep.DEFAULT_BLINDING_SEED,
            "maxim-final-meta-verifier-order-20260803-v2",
        )
        orders = {
            tuple(prep.blind_order(f"val_{index:04d}", prep.DEFAULT_BLINDING_SEED))
            for index in range(40)
        }
        self.assertGreater(len(orders), 1)

    def test_total_rendered_content_cap_and_decisive_reasoning_priority(self) -> None:
        row = {
            "final_answer": "A" * 500,
            "condition": "literal_parallel8_lowconf_v1",
            "model": "Qwen/Qwen3.5-27B",
            "prompt_version": "parallel8",
            "decisive_reasoning": "KEEP-FIRST " + "d" * 2000,
            "reasoning": "literal_parallel8_lowconf_v1 " + "r" * 2000,
            "solution_steps": "s" * 2000,
            "generation": {
                "gold_access": False,
                "decisive_evidence": ["e" * 400, "f" * 400, "g" * 400],
            },
        }
        payload = prep.bounded_candidate_payload(row, "literal_parallel8_lowconf_v1")
        self.assertTrue(payload["bounded_reasoning"].startswith("Decisive: KEEP-FIRST"))
        self.assertLessEqual(len(payload["final_answer"]), 120)
        self.assertLessEqual(len(payload["bounded_reasoning"]), 500)
        self.assertLessEqual(len(payload["bounded_evidence"]), 2)
        self.assertTrue(all(len(item) <= 90 for item in payload["bounded_evidence"]))
        dynamic = (
            len(payload["final_answer"])
            + len(payload["bounded_reasoning"])
            + sum(len(item) for item in payload["bounded_evidence"])
        )
        self.assertLessEqual(dynamic, 800)
        self.assertEqual(prep.TOTAL_RENDERED_CANDIDATE_CHARS_CAP, 10800)
        self.assertLessEqual(prep.TOTAL_RENDERED_CANDIDATE_CHARS_CAP, 10900)
        self.assertNotIn("literal_parallel8_lowconf_v1", json.dumps(payload).casefold())
        worst_rendered = runner.implementation._candidate_prompt(
            {
                "candidates": [
                    {
                        "candidate_id": opaque,
                        "final_answer": "A" * 120,
                        "bounded_reasoning": "R" * 500,
                        "bounded_evidence": ["E" * 90, "F" * 90],
                    }
                    for opaque in prep.OPAQUE_IDS
                ]
            }
        )
        self.assertLessEqual(len(worst_rendered), 10900)

    def test_preregistration_validation_reads_no_absent_candidate_file(self) -> None:
        profile = prep.load_json(PROFILE)
        prep.validate_profile(profile)
        parser = prep.build_parser()
        option_names = {option for action in parser._actions for option in action.option_strings}
        self.assertIn("--candidate", option_names)
        # Keep this preregistration check hermetic.  The real report directory
        # may legitimately contain these files after the frozen branch has
        # run; validation and argument parsing must still perform no I/O on
        # candidate or output paths.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "absent_candidate.jsonl"
            queue = root / "public_queue.jsonl"
            private_key = root / "private_identity_key.jsonl"
            manifest = root / "manifest.json"
            parser.parse_args(
                [
                    "--benchmark",
                    str(root / "absent_benchmark.jsonl"),
                    "--profile",
                    str(root / "absent_profile.json"),
                    "--candidate",
                    f"{prep.REQUIRED_CANDIDATE_SLOTS[0]}={candidate}",
                    "--queue",
                    str(queue),
                    "--private-key",
                    str(private_key),
                    "--manifest",
                    str(manifest),
                ]
            )
            for path in (candidate, queue, private_key, manifest):
                self.assertFalse(path.exists())
        runner_options = {
            option
            for action in runner.build_parser()._actions
            for option in action.option_strings
        }
        self.assertTrue(
            {"--private-key", "--gold", "--reference", "--judge"}.isdisjoint(
                runner_options
            )
        )

    def test_profile_mutation_after_freeze_is_rejected(self) -> None:
        profile = prep.load_json(PROFILE)
        mutated = copy.deepcopy(profile)
        mutated["generation"]["temperature"] = 0.1
        with self.assertRaises(prep.PreparationError):
            prep.validate_profile(mutated)
        mutated = copy.deepcopy(profile)
        mutated["selection_policy"]["min_confidence"] = 0.69
        with self.assertRaises(prep.PreparationError):
            prep.validate_profile(mutated)
        mutated = copy.deepcopy(profile)
        mutated["prompt_policy"][
            "absolute_rendered_candidate_content_limit_chars"
        ] = 10901
        with self.assertRaises(prep.PreparationError):
            prep.validate_profile(mutated)

    def test_synthetic_full274_by_12_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark.jsonl"
            benchmark_rows = [
                {
                    "task_id": f"val_{index:04d}",
                    "subject": "Math",
                    "grade": 7,
                    "question": f"synthetic question {index}",
                    "answer_type": "choice",
                    "question_images": [{"data": "images/x.png"}],
                    "reference_answer": "A",
                }
                for index in range(274)
            ]
            benchmark.write_text(
                "".join(json.dumps(row) + "\n" for row in benchmark_rows),
                encoding="utf-8",
            )
            candidate_paths = {}
            for candidate_index, slot in enumerate(prep.REQUIRED_CANDIDATE_SLOTS):
                path = root / f"candidate_{candidate_index:02d}.jsonl"
                rows = [
                    {
                        "task_id": f"val_{index:04d}",
                        "condition": slot,
                        "model": f"hidden-model-{candidate_index}",
                        "prompt_version": f"hidden-prompt-{candidate_index}",
                        "final_answer": chr(ord("A") + (index + candidate_index) % 5),
                        "decisive_reasoning": f"visible arithmetic for task {index}",
                        "solution_steps": "verify against original pixels",
                        "generation": {
                            "gold_access": False,
                            "decisive_evidence": ["visible value", "format check"],
                        },
                        "error": None,
                    }
                    for index in range(274)
                ]
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                candidate_paths[slot] = path

            queue = root / "public_queue.jsonl"
            key = root / "private_key.jsonl"
            manifest = prep.prepare_queue(
                benchmark_path=benchmark,
                candidate_paths=candidate_paths,
                profile_path=PROFILE,
                queue_path=queue,
                private_key_path=key,
                manifest_path=root / "manifest.json",
                enforce_frozen=False,
            )
            public_rows = prep.load_jsonl(queue)
            private_rows = prep.load_jsonl(key)
            runner.validate_queue(public_rows)
            self.assertEqual(len(public_rows), 274)
            self.assertEqual(sum(len(row["candidates"]) for row in public_rows), 274 * 12)
            self.assertEqual(len(private_rows), 274)
            self.assertEqual(
                [candidate["candidate_id"] for candidate in public_rows[0]["candidates"]],
                list(prep.OPAQUE_IDS),
            )
            self.assertEqual(
                set(private_rows[0]["opaque_to_source_slot"].values()),
                set(prep.REQUIRED_CANDIDATE_SLOTS),
            )
            public_text = queue.read_text(encoding="utf-8")
            self.assertNotIn("reference_answer", public_text)
            for slot in prep.REQUIRED_CANDIDATE_SLOTS:
                self.assertNotIn(slot, public_text)
            self.assertFalse(manifest["candidate_scores_loaded"])
            self.assertFalse(manifest["judge_artifacts_loaded"])

    def test_dynamic_schema_requires_all_twelve_checks(self) -> None:
        schema = runner.verdict_schema(prep.OPAQUE_IDS)
        checks = schema["properties"]["candidate_checks"]
        self.assertEqual(checks["minItems"], 12)
        self.assertEqual(checks["maxItems"], 12)
        self.assertEqual(
            checks["items"]["properties"]["candidate_id"]["enum"],
            list(prep.OPAQUE_IDS),
        )

    def test_original_image_precedes_all_twelve_candidates(self) -> None:
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
                        "bounded_reasoning": "bounded decisive reasoning",
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
            self.assertEqual(
                sum(f"{opaque}\n" in content[2]["text"] for opaque in prep.OPAQUE_IDS),
                12,
            )

    def test_frozen_gate_and_content_exact_router_fallback(self) -> None:
        profile = prep.load_json(PROFILE)
        generation = profile["generation"]
        policy = profile["selection_policy"]
        self.assertEqual(generation["model"], "Qwen/Qwen3.5-27B")
        self.assertEqual(generation["temperature"], 0.0)
        self.assertFalse(generation["enable_thinking"])
        self.assertEqual(generation["max_tokens"], 3072)
        self.assertEqual(policy["min_confidence"], 0.7)
        self.assertEqual(policy["min_decisive_evidence"], 2)
        router_row = {
            "task_id": "val_0001",
            "condition": "maxim_subject_router_v1",
            "final_answer": "A",
            "generation": {"gold_access": False},
            "error": None,
        }
        solver, audit = runner.compose_solver_row(
            result={"task_id": "val_0001", "verdict": None, "error": "timeout"},
            router_row=router_row,
            min_confidence=0.7,
            min_evidence=2,
            queue_sha256="q" * 64,
        )
        self.assertEqual(solver, router_row)
        self.assertIsNot(solver, router_row)
        self.assertEqual(audit["selection"]["selected_source"], "router")
        self.assertEqual(runner.CONDITION, "maxim_final_gold_blind_meta_verifier_v3")


if __name__ == "__main__":
    unittest.main()
