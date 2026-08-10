from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .adapter import ArtifactError, _source_evidence
from .model import RunSummary, TaskTrace, TraceDataset, latency_summary, split_solution_steps
from .replay_aggregate import EXPECTED_MODEL, FrozenReplayComparison


NINE_B_PIPELINE_PROVENANCE = (
    "9B Query Active Crop V2 anchor + deterministic source-adjudicated layers"
)
NINE_B_USAGE_SCOPE = (
    "recorded inherited Query Active Crop 9B anchor usage; deterministic source "
    "lookup/certification/composition and end-to-end wall clock are excluded"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ArtifactError(f"expected object at {path}:{line_number}")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read 9B trace artifact {path}: {exc}") from exc
    return rows


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in result:
            raise ArtifactError(f"{label} has a missing or duplicate task_id")
        result[task_id] = row
    return result


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _judge_correct(row: dict[str, Any], task_id: str) -> bool:
    for key in ("correct", "new_correct", "is_correct"):
        if isinstance(row.get(key), bool):
            return bool(row[key])
    raise ArtifactError(f"{task_id}: 9B judge row has no explicit boolean correctness")


def _derived_display_asset_root(benchmark_path: Path) -> Path:
    """Derive a bounded project root from the already SHA-verified benchmark."""

    resolved = benchmark_path.resolve()
    for candidate in tuple(resolved.parents)[:6]:
        try:
            relative = resolved.relative_to(candidate)
        except ValueError:
            continue
        if len(relative.parts) >= 3 and relative.parts[:2] == ("artifacts", "baselines"):
            return candidate
    return resolved.parent


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


class NineBV7ArtifactAdapter:
    """Join the validated final 9B milestone without importing archived 27B rows."""

    def __init__(
        self,
        comparison: FrozenReplayComparison,
        display_asset_root: Path | str | None = None,
    ):
        self.comparison = comparison
        self.final = comparison.final
        derived_root = _derived_display_asset_root(comparison.benchmark_path)
        self.display_asset_root = (
            Path(display_asset_root).expanduser().resolve()
            if display_asset_root is not None
            else derived_root
        )
        if not _within(comparison.benchmark_path.resolve(), self.display_asset_root):
            raise ArtifactError(
                "display asset root does not contain the verified 9B benchmark"
            )
        self._question_image_index: dict[str, Path] | None = None
        if self.final.milestone_id != "source_v7_rebase_9b":
            raise ArtifactError("9B comparison does not end in source_v7_rebase_9b")
        if self.final.model != EXPECTED_MODEL:
            raise ArtifactError("final replay does not satisfy the Qwen3.5-9B closure")
        if (
            self.final.provenance_status != "new_profile_bound_replay"
            or self.final.bound_before_score is not True
        ):
            raise ArtifactError("final 9B replay is not a bound new-profile replay")

    def _display_image_roots(self) -> tuple[Path, ...]:
        candidates = (
            self.display_asset_root / "tmp" / "blind_visual_binding" / "task_images",
            self.display_asset_root / "artifacts" / "images",
            self.display_asset_root / "images",
            self.comparison.benchmark_path.parent / "images",
        )
        roots: list[Path] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if (
                resolved.is_dir()
                and _within(resolved, self.display_asset_root)
                and resolved not in roots
            ):
                roots.append(resolved)
        return tuple(roots)

    def _build_question_image_index(self) -> dict[str, Path]:
        if self._question_image_index is not None:
            return self._question_image_index
        index: dict[str, Path] = {}
        for root in self._display_image_roots():
            for candidate in root.iterdir():
                if not candidate.is_file() or candidate.suffix.casefold() not in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                }:
                    continue
                resolved = candidate.resolve()
                if not _within(resolved, root) or not candidate.stem:
                    continue
                index.setdefault(candidate.stem.casefold(), resolved)
        self._question_image_index = index
        return index

    def _resolve_question_image(
        self,
        task_id: str,
        row: dict[str, Any],
    ) -> Path | None:
        values: list[Any] = []
        if row.get("question_image"):
            values.append(row["question_image"])
        values.extend(row.get("question_images") or [])
        saw_safe_task_locator = False
        for value in values:
            locator = (
                value.get("path") or value.get("data")
                if isinstance(value, dict)
                else value
            )
            if not isinstance(locator, str) or not locator or locator.startswith("data:"):
                continue
            unresolved = Path(locator)
            if unresolved.is_absolute() or ".." in unresolved.parts:
                continue
            if (
                unresolved.stem.casefold() != task_id.casefold()
                or unresolved.suffix.casefold() not in {".png", ".jpg", ".jpeg"}
            ):
                continue
            saw_safe_task_locator = True
            candidate = (self.comparison.benchmark_path.parent / unresolved).resolve()
            if (
                candidate.is_file()
                and _within(candidate, self.display_asset_root)
            ):
                return candidate
        if not saw_safe_task_locator:
            return None
        return self._build_question_image_index().get(task_id.casefold())

    def load(self) -> TraceDataset:
        base = self.comparison.manifest_path.parent.resolve()
        benchmark = _index(_read_jsonl(self.comparison.benchmark_path), "benchmark")
        solver = _index(_read_jsonl(self.final.solver_path), "solver")
        image_judge = _index(_read_jsonl(self.final.judge_path), "image judge")
        try:
            score_value = json.loads(self.final.score_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"cannot read 9B score outcomes: {exc}") from exc
        score_outcomes = score_value.get("task_outcomes") if isinstance(score_value, dict) else None
        judge = (
            _index(score_outcomes, "score task_outcomes")
            if isinstance(score_outcomes, list)
            else image_judge
        )
        anchor = (
            _index(_read_jsonl(self.final.anchor_path), "9B anchor")
            if self.final.anchor_path is not None
            else {}
        )
        if set(benchmark) != set(solver) or set(benchmark) != set(judge):
            raise ArtifactError("9B benchmark/solver/score-outcome task sets differ")
        if anchor and set(anchor) != set(benchmark):
            raise ArtifactError("9B anchor task set differs from benchmark")

        certificates: dict[str, dict[str, Any]] = {}
        for path in self.final.certificate_paths:
            for task_id, row in _index(_read_jsonl(path), f"certificates:{path.name}").items():
                evidence = _source_evidence(row)
                if evidence.accepted:
                    certificates[task_id] = row

        tasks: list[TaskTrace] = []
        by_subject: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "new_correct": 0})
        for task_id in benchmark:
            prompt = benchmark[task_id]
            solved = solver[task_id]
            verdict = judge[task_id]
            anchor_row = anchor.get(task_id, {})
            base_model = str(solved.get("base_row_model") or solved.get("model") or "")
            if base_model != EXPECTED_MODEL:
                raise ArtifactError(f"{task_id}: base row is not Qwen3.5-9B")
            final_answer = str(solved.get("final_answer") or "")
            anchor_answer = str(
                solved.get("anchor_answer")
                or anchor_row.get("final_answer")
                or final_answer
            )
            origin_code = str(solved.get("final_origin") or "")
            if not origin_code:
                generation = solved.get("generation") or {}
                override = generation.get("official_source_override")
                if not isinstance(override, dict):
                    override = generation.get("fill_blank_page_activity_override")
                origin_code = (
                    "deterministic_source_replacement"
                    if isinstance(override, dict) and final_answer != anchor_answer
                    else "model_anchor"
                )
            if origin_code == "deterministic_source_replacement":
                decision_action = "replace_anchor"
                final_origin = "deterministic source-adjudicated replacement"
            elif origin_code == "model_anchor":
                decision_action = "keep_anchor"
                final_origin = "inherited Query Active Crop V2 9B anchor"
            else:
                raise ArtifactError(f"{task_id}: unsupported final_origin {origin_code!r}")
            if decision_action == "replace_anchor" and not anchor_answer:
                raise ArtifactError(f"{task_id}: source replacement has no anchor answer")
            if decision_action == "keep_anchor" and final_answer != anchor_answer:
                raise ArtifactError(f"{task_id}: keep-anchor bytes differ")
            evidence = _source_evidence(certificates.get(task_id, {}))
            if decision_action == "replace_anchor" and not evidence.accepted:
                raise ArtifactError(f"{task_id}: source final lacks an accepted certificate")

            subject = str(verdict.get("subject") or prompt.get("subject") or "Unknown")
            correct = _judge_correct(verdict, task_id)
            by_subject[subject]["n"] += 1
            by_subject[subject]["new_correct"] += int(correct)
            usage = solved.get("usage") or {}
            reasoning = str(solved.get("reasoning") or "")
            steps = split_solution_steps(solved.get("solution_steps")) or split_solution_steps(reasoning)
            # This is a display-only join from a bounded local image directory.
            # No archived solver answer, judge verdict or provenance row is read.
            question_image = self._resolve_question_image(task_id, prompt)
            task = TaskTrace(
                task_id=task_id,
                subject=subject,
                grade=str(prompt.get("grade") or verdict.get("grade") or "—"),
                answer_type=str(prompt.get("answer_type") or verdict.get("answer_type") or ""),
                question_text=str(prompt.get("question_text") or prompt.get("question") or ""),
                question_image=question_image,
                final_answer=final_answer,
                anchor_answer=anchor_answer,
                challenger_answer=(final_answer if decision_action == "replace_anchor" else ""),
                correct=correct,
                baseline_correct=(
                    bool(verdict["baseline_correct"])
                    if self.final.native_adapter == "normalized_v2"
                    and isinstance(verdict.get("baseline_correct"), bool)
                    else None
                ),
                score_source=str(verdict.get("score_source") or verdict.get("evaluator_origin") or ""),
                score_method=str(verdict.get("score_method") or self.final.evaluator["semantics"]),
                transition=str(verdict.get("transition") or ""),
                reasoning=reasoning,
                solution_steps=steps,
                base_row_model=base_model,
                final_origin=final_origin,
                reasoning_origin=f"recorded Query Active Crop V2 anchor trace · {base_model}",
                usage_origin=NINE_B_USAGE_SCOPE,
                prompt_version=str(solved.get("prompt_version") or ""),
                latency_s=_number(usage.get("latency_s")),
                input_tokens=_integer(usage.get("input_tokens")),
                output_tokens=_integer(usage.get("output_tokens")),
                decision_action=decision_action,
                decision_reason=str(solved.get("decision_reason") or origin_code),
                source=evidence,
                attention_regions=(),
                candidates=(),
                raw={
                    "provenance": {
                        "pipeline": NINE_B_PIPELINE_PROVENANCE,
                        "base_row_model": base_model,
                        "final_origin": final_origin,
                        "usage_origin": NINE_B_USAGE_SCOPE,
                        "aggregate_sha256": self.final.aggregate_sha256,
                        "benchmark_sha256": self.final.benchmark_sha256,
                        "question_image_origin": (
                            "bounded display-only local asset; no archived answers/provenance"
                            if question_image is not None
                            else "no local image in the bounded display asset roots"
                        ),
                    },
                    "solver": solved,
                    "score_outcome": verdict,
                    "image_judge": image_judge.get(task_id, {}),
                    "anchor": anchor_row,
                    "certificate": certificates.get(task_id, {}),
                },
            )
            tasks.append(task)

        tasks.sort(key=lambda item: (0 if item.decision_action == "replace_anchor" else 1, item.task_id))
        latencies = [task.latency_s for task in tasks if task.latency_s is not None]
        p50, p95, maximum = latency_summary(latencies)
        math = self.final.slices["math"]
        active_crop = next(
            item
            for item in self.comparison.milestones
            if item.milestone_id == "query_active_crop_v2_9b"
        )
        for values in by_subject.values():
            values["new_accuracy"] = values["new_correct"] / values["n"]
        active_comparison = next(
            (
                item
                for item in self.final.comparisons
                if item["baseline_milestone_id"] == "query_active_crop_v2_9b"
            ),
            None,
        )
        summary = RunSummary(
            label="9B V7 · new profile-bound source-adjudicated replay",
            rows=self.final.rows,
            correct=self.final.correct,
            accuracy=self.final.accuracy,
            math_rows=int(math["rows"]),
            math_correct=int(math["correct"]),
            math_accuracy=float(math["accuracy"]),
            baseline_accuracy=active_crop.accuracy,
            source_certificates=int(self.final.source_union["size"]),
            answer_overrides=int(self.final.final_origin_counts["deterministic_source_replacement"]),
            direct_gain_vs_v6=(int(active_comparison["fixes"]) if active_comparison else 0),
            evaluator_corrections_vs_v6=0,
            latency_median_s=p50,
            latency_p95_s=p95,
            latency_max_s=maximum,
            pipeline_provenance=NINE_B_PIPELINE_PROVENANCE,
            base_row_models=(EXPECTED_MODEL,),
            recorded_usage_scope=NINE_B_USAGE_SCOPE,
            source_adjudicated_image_rows=int(
                self.final.evaluator.get("source_adjudicated_image_rows", 0)
            ),
            original_9b_judge_rows=int(
                self.final.evaluator.get("original_9b_judge_rows", 0)
            ),
            by_subject={key: dict(value) for key, value in by_subject.items()},
            limitations=tuple(self.final.caveats),
        )
        dataset = TraceDataset(
            summary=summary,
            tasks=tuple(tasks),
            artifact_root=base,
            source_files=(
                self.comparison.manifest_path,
                self.comparison.benchmark_path,
                self.final.aggregate_path,
                self.final.solver_path,
                self.final.score_path,
                self.final.judge_path,
                *((self.final.anchor_path,) if self.final.anchor_path else ()),
                *self.final.certificate_paths,
            ),
        )
        try:
            dataset.validate()
        except ValueError as exc:
            raise ArtifactError(str(exc)) from exc
        return dataset
