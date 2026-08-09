from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .model import (
    AttentionRegion,
    RunSummary,
    SourceEvidence,
    TaskTrace,
    TraceDataset,
    latency_summary,
    split_solution_steps,
)


class ArtifactError(RuntimeError):
    """Raised when a V7 artifact bundle is missing or internally inconsistent."""


REPORT_DIR = Path("reports/maxim_official_exact_source_v2_20260805")
POST_SCORE = REPORT_DIR / "V7_POST_SCORE_RESULT.json"
FINAL_COMPOSED = REPORT_DIR / "fill_blank_page_activity_history_v7_composed"
FINAL_EVALUATION = REPORT_DIR / "fill_blank_page_activity_history_v7_evaluation"
FINAL_RESOLVER = REPORT_DIR / "fill_blank_page_activity_history_v7_resolver"
MAIN_COMPOSED = REPORT_DIR / "public_workbook_samsung_philosophy_biology_imageonly_v7_composed"
MAIN_RESOLVER = REPORT_DIR / "public_workbook_samsung_philosophy_biology_imageonly_v7_resolver"
V6_COMPOSED = REPORT_DIR / "public_workbook_primary_layout_sociology_biology_global_visual_v6_composed"
PUBLIC_QUEUE = Path("reports/maxim_final_meta_verifier_v3_20260803/run/public_queue.jsonl")
PARSER_RESULTS = Path(
    "reports/maxim_document_parser_v1_20260803/"
    "parser_augmented_solver_v1/parser_artifacts/parser_results_274.jsonl"
)
SPEED_ANALYSIS = Path(
    "reports/maxim_v7_source_first_speed_v1_20260808/analysis.json"
)
PIPELINE_PROVENANCE = "META-27B anchor + deterministic source layers"
RECORDED_USAGE_SCOPE = (
    "recorded inherited-anchor usage only; excludes deterministic lookup, "
    "certificate/composer work and end-to-end wall clock"
)


def _looks_like_v7_root(path: Path) -> bool:
    return (path / POST_SCORE).is_file()


def discover_artifact_root(explicit: Path | str | None = None) -> Path:
    """Find the real V7 report tree without depending on a machine-specific cwd."""

    if explicit:
        resolved = Path(explicit).expanduser().resolve()
        if _looks_like_v7_root(resolved):
            return resolved
        raise ArtifactError(
            f"explicit --artifact-root does not contain frozen V7 artifacts: {resolved}"
        )
    configured = os.environ.get("VLM_TRACE_ARTIFACT_ROOT")
    if configured:
        resolved = Path(configured).expanduser().resolve()
        if _looks_like_v7_root(resolved):
            return resolved
        raise ArtifactError(
            "VLM_TRACE_ARTIFACT_ROOT does not contain frozen V7 artifacts: "
            f"{resolved}"
        )

    candidates: list[Path] = []

    # A packaged copy can live several levels below the project root, for
    # example target-repo/apps/vlm-analytics. Search a bounded set of ancestors
    # from both cwd and this module, and the expected sibling at each level.
    # Explicit CLI/env roots were handled above and remain canonical.
    sibling_name = "VLM_agent_turkish_textbook_basic_rag"
    for anchor in (Path.cwd(), Path(__file__).resolve().parent):
        current = anchor.resolve()
        for _ in range(8):
            candidates.extend((current, current / sibling_name))
            if current.parent == current:
                break
            current = current.parent
    candidates.append(Path.home() / "PycharmProjects" / sibling_name)

    seen: set[Path] = set()
    searched_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        searched_candidates.append(resolved)
        if _looks_like_v7_root(resolved):
            return resolved
    searched = "\n".join(f"  - {candidate}" for candidate in searched_candidates)
    raise ArtifactError(
        "V7 artifacts were not found. Pass --artifact-root or set "
        f"VLM_TRACE_ARTIFACT_ROOT. Searched:\n{searched}"
    )


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ArtifactError(f"required artifact is missing: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            raise ArtifactError(f"required artifact is missing: {path}")
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ArtifactError(
                        f"expected object at {path}:{line_number}, got {type(value).__name__}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSONL artifact {path}: {exc}") from exc
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactError(f"cannot hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        if task_id in result:
            raise ArtifactError(f"duplicate task_id {task_id!r} in artifact join")
        result[task_id] = row
    return result


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _source_evidence(certificate: dict[str, Any]) -> SourceEvidence:
    if not certificate:
        return SourceEvidence()
    trace = certificate.get("trace") or {}
    source = trace.get("source") or {}
    match = trace.get("match") or {}
    checks = trace.get("checks") or {}
    return SourceEvidence(
        status=str(certificate.get("status") or "absent"),
        strength=str(certificate.get("strength") or "none"),
        verifier=str(certificate.get("verifier") or trace.get("verifier") or ""),
        document_id=str(source.get("document_id") or ""),
        document_name=str(source.get("name") or ""),
        public_locator=str(source.get("public_locator") or ""),
        matched_page=_as_int(source.get("matched_page_number")),
        key_page=_as_int(source.get("key_page_number")),
        question_number=_as_int(source.get("question_number")),
        record_id=str(source.get("record_id") or ""),
        key_bbox=_parse_bbox(source.get("key_bbox")),
        checks=tuple((str(key), bool(value)) for key, value in checks.items()),
        page_coverage=_as_float(match.get("page_idf_coverage")),
        page_margin=_as_float(match.get("page_margin")),
        trace_fingerprint=str(certificate.get("trace_fingerprint") or ""),
        pdf_sha256=str(source.get("pdf_sha256") or ""),
    )


def _attention_regions(parser_row: dict[str, Any]) -> tuple[AttentionRegion, ...]:
    images = parser_row.get("images") or []
    if not images:
        return ()
    image = images[0]
    width = _as_int(image.get("width")) or 1
    height = _as_int(image.get("height")) or 1
    regions: list[AttentionRegion] = []
    for block in image.get("parsing_res_list") or []:
        bbox = _parse_bbox(block.get("block_bbox"))
        if not bbox:
            continue
        regions.append(
            AttentionRegion(
                bbox=bbox,
                label=str(block.get("block_label") or "text"),
                text=str(block.get("block_content") or ""),
                image_width=width,
                image_height=height,
            )
        )
    return tuple(regions)


def _question_from_queue(
    queue_row: dict[str, Any], regions: tuple[AttentionRegion, ...]
) -> str:
    question = str(queue_row.get("question") or "").strip()
    placeholders = {"", "(soru görselde)", "(soru gГ¶rselde)", "question in image"}
    if question.casefold() not in {item.casefold() for item in placeholders}:
        return question
    visible = [region.text.strip() for region in regions if region.text.strip()]
    return "\n".join(visible[:12])


class V7ArtifactAdapter:
    """Join the frozen V7 solver, score, OCR and source-certificate artifacts."""

    def __init__(self, artifact_root: Path | str | None = None):
        self.root = discover_artifact_root(artifact_root)
        self._image_index: dict[str, Path] | None = None

    def _path(self, relative: Path) -> Path:
        return self.root / relative

    def _build_image_index(self) -> dict[str, Path]:
        if self._image_index is not None:
            return self._image_index
        index: dict[str, Path] = {}
        search_roots = (
            self.root / "tmp" / "blind_visual_binding" / "task_images",
            self.root / "artifacts" / "images",
            self.root / "images",
        )
        for search_root in search_roots:
            if not search_root.is_dir():
                continue
            for pattern in ("val_*.png", "val_*.jpg", "val_*.jpeg"):
                for path in search_root.rglob(pattern):
                    index.setdefault(path.name.casefold(), path.resolve())
        self._image_index = index
        return index

    def _resolve_question_image(self, queue_row: dict[str, Any]) -> Path | None:
        values = queue_row.get("question_images") or []
        for value in values:
            locator = value.get("data") if isinstance(value, dict) else value
            if not locator or not isinstance(locator, str) or locator.startswith("data:"):
                continue
            path = Path(locator)
            candidates = (
                path,
                self.root / path,
                self._path(PUBLIC_QUEUE).parent / path,
                self.root / "reports" / "maxim_final_meta_verifier_v3_20260803" / "run" / path,
            )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate.resolve()
            indexed = self._build_image_index().get(path.name.casefold())
            if indexed:
                return indexed
        return None

    @staticmethod
    def _merge_decision(
        main: dict[str, Any], final: dict[str, Any]
    ) -> dict[str, Any]:
        # The final history wave is layered over the main V7 source wave. Preserve
        # the earlier certificate trace when the later wave had no challenger.
        if final.get("source_override") is True:
            return {
                **final,
                "action": "replace_anchor",
                "anchor_answer": main.get("selected_answer") or main.get("anchor_answer"),
                "reason": "strongly_verified_challenger",
            }
        if final and (
            final.get("action") == "replace_anchor"
            or final.get("certificate_trace_fingerprint")
            or final.get("reason") not in (None, "", "no_challengers")
        ):
            return final
        return main or final

    @staticmethod
    def _compact_candidates(queue_row: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        result: list[dict[str, Any]] = []
        for candidate in queue_row.get("candidates") or []:
            reasoning = str(candidate.get("bounded_reasoning") or "")
            evidence = candidate.get("bounded_evidence") or []
            result.append(
                {
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "final_answer": str(candidate.get("final_answer") or ""),
                    "reasoning": reasoning[:1200],
                    "evidence": [str(value)[:400] for value in evidence[:4]],
                }
            )
        return tuple(result)

    def load(self) -> TraceDataset:
        final_solver_path = self._path(FINAL_COMPOSED / "solver.jsonl")
        if not final_solver_path.is_file():
            final_solver_path = self._path(MAIN_COMPOSED / "solver.jsonl")

        score_path = self._path(FINAL_EVALUATION / "score.json")
        post_score_path = self._path(POST_SCORE)
        score = _read_json(score_path)
        post_score = _read_json(post_score_path)
        speed_analysis_path = self._path(SPEED_ANALYSIS)
        speed_analysis = _read_json(speed_analysis_path, required=False)

        solver_rows = _index(_read_jsonl(final_solver_path))
        v6_solver_rows = _index(
            _read_jsonl(self._path(V6_COMPOSED / "solver.jsonl"), required=False)
        )
        main_decisions = _index(
            _read_jsonl(self._path(MAIN_COMPOSED / "decisions.jsonl"), required=False)
        )
        final_decisions = _index(
            _read_jsonl(self._path(FINAL_COMPOSED / "decisions.jsonl"), required=False)
        )

        main_candidates = _index(
            _read_jsonl(self._path(MAIN_RESOLVER / "candidate.jsonl"), required=False)
        )
        final_candidates = _index(
            _read_jsonl(self._path(FINAL_RESOLVER / "candidate.jsonl"), required=False)
        )
        main_audits = _index(
            _read_jsonl(self._path(MAIN_RESOLVER / "audit.jsonl"), required=False)
        )
        final_audits = _index(
            _read_jsonl(self._path(FINAL_RESOLVER / "audit.jsonl"), required=False)
        )

        certificates: dict[str, dict[str, Any]] = {}
        for path in (
            self._path(MAIN_RESOLVER / "certificates.jsonl"),
            self._path(FINAL_RESOLVER / "certificates.jsonl"),
        ):
            certificates.update(_index(_read_jsonl(path, required=False)))

        queue_rows = _index(_read_jsonl(self._path(PUBLIC_QUEUE), required=False))
        parser_rows = _index(_read_jsonl(self._path(PARSER_RESULTS), required=False))
        outcome_rows = _index(score.get("task_outcomes") or [])

        if not solver_rows:
            raise ArtifactError("V7 final solver contains no rows")

        # The UI labels this run as frozen. Check the content-addressed links
        # recorded at score time instead of validating only row counts.
        score_snapshot = post_score.get("score") or {}
        expected_score_hash = str(score_snapshot.get("sha256") or "").casefold()
        if expected_score_hash and _sha256(score_path).casefold() != expected_score_hash:
            raise ArtifactError(
                "V7 score hash differs from V7_POST_SCORE_RESULT; refusing to label "
                "the joined artifacts as frozen"
            )
        speed_solver = ((speed_analysis.get("inputs") or {}).get("final_solver") or {})
        expected_solver_hash = str(speed_solver.get("sha256") or "").casefold()
        if expected_solver_hash and _sha256(final_solver_path).casefold() != expected_solver_hash:
            raise ArtifactError(
                "source-first analysis references a different final solver artifact"
            )

        if set(solver_rows) != set(outcome_rows):
            missing_score = sorted(set(solver_rows) - set(outcome_rows))
            missing_solver = sorted(set(outcome_rows) - set(solver_rows))
            raise ArtifactError(
                "V7 solver/score task sets differ: "
                f"missing_score={missing_score[:5]}, missing_solver={missing_solver[:5]}"
            )
        if score_snapshot:
            snapshot_rows = _as_int(score_snapshot.get("rows"))
            snapshot_correct = _as_int(score_snapshot.get("correct"))
            overall = score.get("overall") or {}
            if snapshot_rows is not None and snapshot_rows != _as_int(overall.get("n")):
                raise ArtifactError("post-score row count differs from score.json")
            if snapshot_correct is not None and snapshot_correct != _as_int(
                overall.get("new_correct")
            ):
                raise ArtifactError("post-score correct count differs from score.json")

        if speed_analysis:
            speed_rows = _as_int(speed_analysis.get("rows"))
            shortcuts = _as_int(speed_analysis.get("source_shortcuts")) or 0
            fallbacks = _as_int(speed_analysis.get("anchor_fallbacks")) or 0
            equivalents = _as_int(speed_analysis.get("answer_equivalent_rows")) or 0
            if speed_rows != len(solver_rows):
                raise ArtifactError(
                    "source-first analysis row count differs from the frozen V7 solver"
                )
            if shortcuts + fallbacks != speed_rows:
                raise ArtifactError("source-first shortcut/fallback counts do not cover all rows")
            if not (0 <= equivalents <= shortcuts <= speed_rows):
                raise ArtifactError("source-first shortcut counts are internally inconsistent")

        unknown_certificates = sorted(set(certificates) - set(solver_rows))
        if unknown_certificates:
            raise ArtifactError(
                "source certificates contain unknown task ids: "
                f"{unknown_certificates[:5]}"
            )

        tasks: list[TaskTrace] = []
        for task_id, solver in solver_rows.items():
            outcome = outcome_rows[task_id]
            queue = queue_rows.get(task_id, {})
            parser = parser_rows.get(task_id, {})
            regions = _attention_regions(parser)
            main_decision = main_decisions.get(task_id, {})
            final_decision = final_decisions.get(task_id, {})
            decision = self._merge_decision(main_decision, final_decision)
            source_candidate = final_candidates.get(task_id) or main_candidates.get(task_id) or {}
            if source_candidate.get("abstain"):
                source_candidate = main_candidates.get(task_id, {})
            evidence = _source_evidence(certificates.get(task_id, {}))
            v6_solver = v6_solver_rows.get(task_id, {})

            anchor_answer = str(
                decision.get("anchor_answer")
                or v6_solver.get("final_answer")
                or solver.get("final_answer")
                or ""
            )
            challenger_answer = ""
            if source_candidate and not source_candidate.get("abstain"):
                challenger_answer = str(source_candidate.get("final_answer") or "")

            usage = solver.get("usage") or {}
            final_answer = str(
                solver.get("final_answer") or decision.get("selected_answer") or ""
            )
            base_row_model = str(solver.get("model") or "")
            if decision.get("action") == "replace_anchor":
                if not evidence.accepted or not challenger_answer:
                    raise ArtifactError(
                        f"{task_id}: source replacement lacks an accepted certificate "
                        "or a non-abstaining challenger"
                    )
                final_origin = "deterministic official-source replacement"
            else:
                if final_answer != anchor_answer:
                    raise ArtifactError(
                        f"{task_id}: final answer differs from its anchor without a "
                        "recorded deterministic replacement"
                    )
                final_origin = "inherited pre-V7 composite anchor"
            reasoning_origin = (
                "recorded inherited-anchor trace · "
                + (base_row_model or "model metadata absent")
            )
            solution_steps = split_solution_steps(solver.get("solution_steps"))
            if not solution_steps:
                solution_steps = split_solution_steps(solver.get("reasoning"))
            task = TaskTrace(
                task_id=task_id,
                subject=str(outcome.get("subject") or queue.get("subject") or "Unknown"),
                grade=str(queue.get("grade") or outcome.get("grade") or "—"),
                answer_type=str(outcome.get("answer_type") or queue.get("answer_type") or ""),
                question_text=_question_from_queue(queue, regions),
                question_image=self._resolve_question_image(queue),
                final_answer=final_answer,
                anchor_answer=anchor_answer,
                challenger_answer=challenger_answer,
                correct=bool(outcome.get("new_correct")),
                baseline_correct=(
                    bool(outcome.get("baseline_correct"))
                    if outcome.get("baseline_correct") is not None
                    else None
                ),
                score_source=str(outcome.get("score_source") or ""),
                score_method=str(outcome.get("score_method") or ""),
                transition=str(outcome.get("transition") or ""),
                reasoning=str(solver.get("reasoning") or ""),
                solution_steps=solution_steps,
                base_row_model=base_row_model,
                final_origin=final_origin,
                reasoning_origin=reasoning_origin,
                usage_origin=RECORDED_USAGE_SCOPE,
                prompt_version=str(solver.get("prompt_version") or ""),
                latency_s=_as_float(usage.get("latency_s")),
                input_tokens=_as_int(usage.get("input_tokens")),
                output_tokens=_as_int(usage.get("output_tokens")),
                decision_action=str(decision.get("action") or "keep_anchor"),
                decision_reason=str(decision.get("reason") or "no_challengers"),
                source=evidence,
                attention_regions=regions,
                candidates=self._compact_candidates(queue),
                raw={
                    "provenance": {
                        "pipeline": PIPELINE_PROVENANCE,
                        "base_row_model": base_row_model,
                        "final_origin": final_origin,
                        "reasoning_origin": reasoning_origin,
                        "usage_origin": RECORDED_USAGE_SCOPE,
                    },
                    "composed_solver_row": solver,
                    "decision": decision,
                    "main_decision": main_decision,
                    "final_wave_decision": final_decision,
                    "source_candidate": source_candidate,
                    "certificate": certificates.get(task_id, {}),
                    "audit": final_audits.get(task_id) or main_audits.get(task_id) or {},
                    "score": outcome,
                },
            )
            tasks.append(task)

        # Put the two real source replacements first; the remaining ordering stays stable.
        tasks.sort(
            key=lambda task: (
                0 if task.decision_action == "replace_anchor" else 1,
                0 if task.has_certificate else 1,
                task.task_id,
            )
        )

        overall = score.get("overall") or {}
        by_subject = score.get("by_subject") or {}
        math = by_subject.get("Math") or by_subject.get("Mathematics") or {}
        operational_latency = (score.get("operational") or {}).get("latency") or {}
        post_comparison = post_score.get("comparison_to_v6") or {}
        changed = post_score.get("changed_correctness_vs_v6") or []
        direct_gain = sum(
            1 for row in changed if "answer replacement" in str(row.get("mechanism") or "")
        )
        evaluator_corrections = sum(
            1 for row in changed if "judge verdict" in str(row.get("mechanism") or "")
        )
        accepted_certificates = sum(
            1 for value in certificates.values() if _source_evidence(value).accepted
        )
        combined_overrides = sum(task.decision_action == "replace_anchor" for task in tasks)
        measured_latencies = [task.latency_s for task in tasks if task.latency_s is not None]
        measured_median, measured_p95, measured_max = latency_summary(measured_latencies)
        base_row_models = tuple(sorted({task.base_row_model for task in tasks if task.base_row_model}))

        summary = RunSummary(
            label="V7 · official-certificate-adjudicated development replay",
            rows=int(overall.get("n") or len(tasks)),
            correct=int(overall.get("new_correct") or 0),
            accuracy=float(overall.get("new_accuracy") or 0.0),
            math_rows=int(math.get("n") or 0),
            math_correct=int(math.get("new_correct") or 0),
            math_accuracy=float(math.get("new_accuracy") or 0.0),
            baseline_accuracy=float(overall.get("baseline_accuracy") or 0.0),
            source_certificates=accepted_certificates,
            answer_overrides=combined_overrides,
            direct_gain_vs_v6=direct_gain,
            evaluator_corrections_vs_v6=evaluator_corrections,
            latency_median_s=_as_float(operational_latency.get("latency_s_median")) or measured_median,
            latency_p95_s=_as_float(operational_latency.get("latency_s_p95_nearest_rank")) or measured_p95,
            latency_max_s=_as_float(operational_latency.get("latency_s_max")) or measured_max,
            pipeline_provenance=PIPELINE_PROVENANCE,
            base_row_models=base_row_models,
            recorded_usage_scope=RECORDED_USAGE_SCOPE,
            by_subject={str(key): value for key, value in by_subject.items()},
            limitations=tuple(str(value) for value in post_score.get("limitations") or ()),
            source_shortcuts=int(speed_analysis.get("source_shortcuts") or 0),
            anchor_fallbacks=int(speed_analysis.get("anchor_fallbacks") or 0),
            source_shortcut_rate=_as_float(speed_analysis.get("source_shortcut_rate")),
            answer_equivalent_shortcuts=int(
                speed_analysis.get("answer_equivalent_rows") or 0
            ),
            avoidable_recorded_latency_fraction=_as_float(
                (speed_analysis.get("recorded_anchor_usage") or {}).get(
                    "avoidable_latency_fraction"
                )
            ),
            avoidable_input_tokens_fraction=_as_float(
                (speed_analysis.get("recorded_anchor_usage") or {}).get(
                    "avoidable_input_fraction"
                )
            ),
            avoidable_output_tokens_fraction=_as_float(
                (speed_analysis.get("recorded_anchor_usage") or {}).get(
                    "avoidable_output_fraction"
                )
            ),
            speed_online_wall_clock_measured=bool(
                (speed_analysis.get("claims") or {}).get(
                    "online_wall_clock_speedup_measured"
                )
            ),
            speed_source_lookup_cost_included=bool(
                (speed_analysis.get("claims") or {}).get(
                    "source_lookup_cost_included"
                )
            ),
        )
        dataset = TraceDataset(
            summary=summary,
            tasks=tuple(tasks),
            artifact_root=self.root,
            source_files=tuple(
                path
                for path in (
                    final_solver_path,
                    score_path,
                    post_score_path,
                    self._path(FINAL_COMPOSED / "decisions.jsonl"),
                    self._path(MAIN_RESOLVER / "certificates.jsonl"),
                    self._path(FINAL_RESOLVER / "certificates.jsonl"),
                    self._path(PUBLIC_QUEUE),
                    self._path(PARSER_RESULTS),
                    speed_analysis_path,
                )
                if path.is_file()
            ),
        )
        try:
            dataset.validate()
        except ValueError as exc:
            raise ArtifactError(str(exc)) from exc
        return dataset

    def validation_report(self) -> dict[str, Any]:
        dataset = self.load()
        local_images = sum(task.question_image is not None for task in dataset.tasks)
        ocr_reconstructions = sum(bool(task.attention_regions) for task in dataset.tasks)
        return {
            "status": "ok",
            "label": dataset.summary.label,
            "artifact_root": str(dataset.artifact_root),
            "rows": dataset.summary.rows,
            "correct": dataset.summary.correct,
            "accuracy": dataset.summary.accuracy,
            "math_accuracy": dataset.summary.math_accuracy,
            "accepted_source_certificates": dataset.summary.source_certificates,
            "layered_answer_overrides": dataset.summary.answer_overrides,
            "provenance": {
                "pipeline": dataset.summary.pipeline_provenance,
                "base_row_models": list(dataset.summary.base_row_models),
                "final_origin_counts": {
                    "deterministic_official_source": sum(
                        task.decision_action == "replace_anchor"
                        for task in dataset.tasks
                    ),
                    "inherited_anchor": sum(
                        task.decision_action != "replace_anchor"
                        for task in dataset.tasks
                    ),
                },
                "usage_scope": dataset.summary.recorded_usage_scope,
                "latency_and_tokens_are_end_to_end": False,
            },
            "local_question_images": local_images,
            "ocr_reconstructable_questions": ocr_reconstructions,
            "source_first_projection": {
                "shortcuts": dataset.summary.source_shortcuts,
                "rows": dataset.summary.rows,
                "answer_equivalent_shortcuts": (
                    dataset.summary.answer_equivalent_shortcuts
                ),
                "avoidable_recorded_latency_fraction": (
                    dataset.summary.avoidable_recorded_latency_fraction
                ),
                "avoidable_input_tokens_fraction": (
                    dataset.summary.avoidable_input_tokens_fraction
                ),
                "online_wall_clock_measured": (
                    dataset.summary.speed_online_wall_clock_measured
                ),
                "source_lookup_cost_included": (
                    dataset.summary.speed_source_lookup_cost_included
                ),
            },
            "source_files": [str(path) for path in dataset.source_files],
        }
