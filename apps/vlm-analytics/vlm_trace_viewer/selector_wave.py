from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .adapter import ArtifactError, discover_artifact_root


MODEL = "Qwen/Qwen3.5-9B"

COMPARISON = Path("reports/maxim_9b_source_replay_v1_20260809/comparison.json")
SOURCE_V7_SCORE = Path(
    "reports/maxim_9b_source_replay_v1_20260809/active_crop/final_evaluation/score.json"
)
EXPERIMENT = Path("experiments/maxim_9b_baseline_selector_v1")
COMPLETION = EXPERIMENT / "evaluation_wave_v1/WAVE_COMPLETION_MANIFEST.json"
SELECTOR_SCORE = EXPERIMENT / "evaluation_wave_v1/results/v1_2_primary_score.json"
COMPOSITION = EXPERIMENT / "compositor_output_v1_2/composition_manifest_v1_2.json"
DECISIONS = EXPERIMENT / "compositor_output_v1_2/primary_decisions.jsonl"
PRIMARY_SOLVER = EXPERIMENT / "compositor_output_v1_2/primary_solver.jsonl"
PROPOSALS = EXPERIMENT / "output_v1_2/selection_proposals_v1_2.jsonl"
REPAIR = Path("experiments/maxim_9b_answer_contract_repair_v1_1")
REPAIR_RULE_FREEZE = REPAIR / "DEVELOPMENT_RULE_FREEZE.json"
REPAIR_OUTPUT_FREEZE = REPAIR / "candidate_output/CANDIDATE_OUTPUT_FREEZE.json"
REPAIR_SOLVER = REPAIR / (
    "candidate_output/"
    "on_v1_2_primary_240__exploratory_explicit_key_scalar__candidate_solver.jsonl"
)
REPAIR_DECISIONS = REPAIR / (
    "candidate_output/"
    "on_v1_2_primary_240__exploratory_explicit_key_scalar__decisions.jsonl"
)
REPAIR_SCORE = REPAIR / "evaluation_on_v1_2_primary_240/score.json"

PINNED_SHA256 = {
    COMPARISON: "27e4db797acd8fbd9818f54385c734059437c9a0c3ca804d12e9b8a0fbf77b35",
    SOURCE_V7_SCORE: "945c76f3f162c77d52595b4768c60cc48724dae728c0343851afa127733bc039",
    COMPLETION: "ea32a839ccf8dc256e69f3b994332ce32bae552c120d6a9d98bd7691cd950973",
    SELECTOR_SCORE: "ccf225c2d10f1719be2802585d063ea3aa106b7d0be7240abb5c6f93c03a0fc6",
    COMPOSITION: "0756f55b9ca2847fbee378070dca09909374851a7320d8c7125375483c157750",
    DECISIONS: "a889ea2cc2f8fdfd6c93ded8d6676257a7f50bdfd582f424ccf03f18c082b2af",
    PRIMARY_SOLVER: "09aa8d69e7de3a02bbc9b28b2b269b845a0dee1a40ef2d6aa55f7e966a779bef",
    PROPOSALS: "51a5ed6a8cb76677d17eb1e8a55319b57503069f0660735d2c7273624ba598ec",
    REPAIR_RULE_FREEZE: "8fc5607795cde3d4f97e174cdcebcdd8959a1f8420c9bcafb657460b3953628d",
    REPAIR_OUTPUT_FREEZE: "4bf354ad7ab2deb35ed89f136a2eeb9c69b5c3c95971b950a1ecadd1a0987067",
    REPAIR_SOLVER: "f5bc196340a400661e58c68ccdb684b5682317f4dda53a7487bb448fc825869e",
    REPAIR_DECISIONS: "d8c3ea9516a1e0b4da6f88ef324846131e357dea3f7eb3dee804591af274f47f",
    REPAIR_SCORE: "453970038673fb29b97d754b4ef980e19850a930499e89ccccfb9f9b8e6c9dc8",
}


@dataclass(frozen=True)
class Milestone:
    milestone_id: str
    label: str
    correct: int
    rows: int
    math_correct: int
    math_rows: int
    deterministic_correct: int
    deterministic_rows: int
    image_correct: int
    image_rows: int
    aggregate_sha256: str

    @property
    def accuracy(self) -> float:
        return self.correct / self.rows


@dataclass(frozen=True)
class SelectorTaskProvenance:
    task_id: str
    subject: str
    row_index: int
    anchor_answer: str
    selected_answer: str
    structural_answer: str
    native_answer: str
    native_v4_answer: str
    native_v5_answer: str
    parallel_answer: str
    route: str
    reason: str
    base_row_sha256: str
    output_row_sha256: str
    proposal_row_sha256: str
    source_row_sha256: str


@dataclass(frozen=True)
class SelectorWaveSummary:
    model: str
    milestones: tuple[Milestone, ...]
    correct: int
    rows: int
    math_correct: int
    math_rows: int
    history_correct: int
    history_rows: int
    deterministic_correct: int
    deterministic_rows: int
    image_correct: int
    image_rows: int
    fixes: int
    regressions: int
    passthrough_rows: int
    replacement_rows: int
    source_preserved_rows: int
    image_preserved_rows: int
    tasks: tuple[SelectorTaskProvenance, ...]
    comparison_sha256: str
    completion_sha256: str
    score_sha256: str
    composition_sha256: str
    solver_sha256: str
    repair_rule_sha256: str
    repair_output_freeze_sha256: str
    repair_solver_sha256: str
    repair_score_sha256: str
    repair_task_id: str
    repair_base_row_sha256: str
    repair_output_row_sha256: str
    verified_files: tuple[Path, ...]

    @property
    def accuracy(self) -> float:
        return self.correct / self.rows


MILESTONE_SPECS = (
    (
        "page_rag_9b",
        "Page-RAG",
        Path("reports/maxim_9b_source_replay_v1_20260809/legacy/page_rag/aggregate.json"),
        "477135933bc45a4f64e9405cd3f0696607773ba52abcbe03dbbf94708f03c8b4",
        (141, 274, 62, 139, 96, 177, 45, 97),
    ),
    (
        "no_tools_9b",
        "No tools",
        Path("reports/maxim_9b_source_replay_v1_20260809/legacy/no_tools/aggregate.json"),
        "fafbdf214db997bc0a37d8f5da5fd006f2ac2c453d00e354d20e28a56325e3e5",
        (193, 274, 103, 139, 132, 177, 61, 97),
    ),
    (
        "query_active_crop_v2_9b",
        "Active crop v2",
        Path("reports/maxim_9b_source_replay_v1_20260809/legacy/active_crop/aggregate.json"),
        "cd845f2d0f276233e3b3f963df6c06b3c7a6d2e2dc35e4e38225f702733daeb2",
        (194, 274, 104, 139, 133, 177, 61, 97),
    ),
    (
        "source_v1_rebase_9b",
        "Source v1",
        Path("reports/maxim_9b_source_replay_v1_20260809/active_crop/source_v1_aggregate/aggregate.json"),
        "caddf1c549845d08dc08dfa640270d59738b816a38152c79c776529991b3f4cf",
        (218, 274, 106, 139, 148, 177, 70, 97),
    ),
    (
        "source_v3_rebase_9b",
        "Source v3",
        Path("reports/maxim_9b_source_replay_v1_20260809/active_crop/source_v3_aggregate/aggregate.json"),
        "83d1fe7a0651b7b0e14493aa4adfb2ea2a6f4285db0e7b4ffa2a858de6b1a266",
        (227, 274, 106, 139, 152, 177, 75, 97),
    ),
    (
        "source_v6_rebase_9b",
        "Source v6",
        Path("reports/maxim_9b_source_replay_v1_20260809/active_crop/v6_aggregate/aggregate.json"),
        "3031a2a376168a5fadcf037465549084f61b86bd17a4b4c01b5e98e1ed0de9ae",
        (235, 274, 108, 139, 156, 177, 79, 97),
    ),
    (
        "source_v7_rebase_9b",
        "Source V7",
        Path("reports/maxim_9b_source_replay_v1_20260809/active_crop/source_v7_aggregate/aggregate.json"),
        "3de5129dee80d2f2fda544bdf7eecfa7d0f467d56bb7e43afc8eac89a6a5dacd",
        (238, 274, 108, 139, 156, 177, 82, 97),
    ),
)

TASK_SPECS = {
    "val_0089": {
        "subject": "History",
        "row_index": 87,
        "anchor": "A",
        "selected": "D",
        "base": "c86c93b840ced159195eacb554888e47ab3f3c9fd2fd5ad8de65d06ff14c5000",
        "output": "cb1b206959e31e33c4d08c2baa76493a3895dea7864bea23bb3bf9509404840a",
        "proposal": "4c401a7159bf75d5a85599a89ccf57d91dd400d26197f403d30996bec208c0fc",
        "source": "289332312cd591b63f570636cd1b34d2ff92f721a69cca408ca780f934dd6ec9",
    },
    "val_0251": {
        "subject": "Math",
        "row_index": 248,
        "anchor": "A",
        "selected": "B",
        "base": "dc2993c91c4ec6e9822b99d3b176596cf3be5a111d9a1f91d1d2807e19d5923f",
        "output": "35b18f350195379b19879dd99916787423f2d0337349c4216e2b9cf71f1553a8",
        "proposal": "9242818698640c08d2631ecef822fd68a2e706f671ecb5dacfb48b7ab830397f",
        "source": "869aafc636bc0aaa843512e69adede446ad0cd8b5e3350ac200de0b4ae0e8352",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactError(f"cannot hash selector artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactError(f"required selector artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read selector JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"expected object in selector artifact: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ArtifactError(f"required selector artifact is missing: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ArtifactError(f"expected object at {path}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read selector JSONL {path}: {exc}") from exc
    return rows


def _index(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in result:
            raise ArtifactError(f"missing or duplicate task_id in {label}: {task_id!r}")
        result[task_id] = row
    return result


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ArtifactError(f"selector contract mismatch for {label}: {actual!r} != {expected!r}")


def _declared_path_matches(value: Any, expected_relative: Path) -> bool:
    normalized = str(value or "").replace("\\", "/").casefold()
    return normalized.endswith(expected_relative.as_posix().casefold())


def _safe_manifest_path(base: Path, value: Any) -> Path:
    text = str(value or "")
    pure = PurePosixPath(text.replace("\\", "/"))
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or not pure.parts
        or pure.parts[0].endswith(":")
    ):
        raise ArtifactError(f"unsafe path in selector completion manifest: {text!r}")
    resolved_base = base.resolve()
    resolved = (resolved_base / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise ArtifactError(f"selector completion path escapes experiment root: {text!r}") from exc
    return resolved


def _metric_tuple(aggregate: dict[str, Any]) -> tuple[int, ...]:
    if isinstance(aggregate.get("metrics"), dict):
        metrics = aggregate["metrics"]
        slices = metrics.get("slices") or {}
        overall = metrics
        math = slices.get("math") or slices.get("Math") or {}
        deterministic = slices.get("deterministic") or {}
        image = slices.get("image") or slices.get("image_judge") or {}
        return (
            int(overall.get("correct") or 0),
            int(overall.get("rows") or overall.get("n") or 0),
            int(math.get("correct") or math.get("new_correct") or 0),
            int(math.get("rows") or math.get("n") or 0),
            int(deterministic.get("correct") or deterministic.get("new_correct") or 0),
            int(deterministic.get("rows") or deterministic.get("n") or 0),
            int(image.get("correct") or image.get("new_correct") or 0),
            int(image.get("rows") or image.get("n") or 0),
        )
    overall = aggregate.get("overall") or {}
    slices = aggregate.get("slices") or {}
    evaluator = aggregate.get("evaluator_split") or {}
    math = slices.get("Math") or slices.get("math") or {}
    deterministic = evaluator.get("deterministic") or {}
    image = evaluator.get("image_judge") or {}
    return (
        int(overall.get("new_correct") or 0),
        int(overall.get("n") or 0),
        int(math.get("new_correct") or 0),
        int(math.get("n") or 0),
        int(deterministic.get("new_correct") or 0),
        int(deterministic.get("n") or 0),
        int(image.get("new_correct") or 0),
        int(image.get("n") or 0),
    )


class SelectorWaveAdapter:
    """Load the audited all-9B selector wave without trusting machine-local paths."""

    def __init__(self, artifact_root: Path | str | None = None):
        self.root = discover_artifact_root(artifact_root)

    def _path(self, relative: Path) -> Path:
        root = self.root.resolve()
        result = (root / relative).resolve()
        try:
            result.relative_to(root)
        except ValueError as exc:
            raise ArtifactError(f"selector path escapes artifact root: {relative}") from exc
        return result

    def _verify_pin(self, relative: Path) -> Path:
        path = self._path(relative)
        expected = PINNED_SHA256[relative]
        actual = _sha256(path)
        if actual != expected:
            raise ArtifactError(
                f"selector frozen hash mismatch for {relative}: {actual} != {expected}"
            )
        return path

    def _load_milestones(self, comparison: dict[str, Any]) -> tuple[Milestone, ...]:
        _expect(comparison.get("schema_version"), "vlm-9b-milestone-comparison-v2", "comparison schema")
        _expect(comparison.get("model"), MODEL, "comparison model")
        rows = comparison.get("milestones")
        if not isinstance(rows, list):
            raise ArtifactError("comparison milestones must be a list")
        _expect(
            [row.get("milestone_id") for row in rows],
            [spec[0] for spec in MILESTONE_SPECS],
            "canonical seven milestone order",
        )
        result: list[Milestone] = []
        for row, spec in zip(rows, MILESTONE_SPECS):
            milestone_id, label, relative, expected_hash, expected_metrics = spec
            declared = row.get("aggregate") or {}
            _expect(declared.get("sha256"), expected_hash, f"{milestone_id} declared hash")
            if not _declared_path_matches(declared.get("path"), relative):
                raise ArtifactError(f"{milestone_id} declared path does not bind the canonical aggregate")
            aggregate_path = self._path(relative)
            _expect(_sha256(aggregate_path), expected_hash, f"{milestone_id} aggregate hash")
            aggregate = _read_json(aggregate_path)
            metrics = _metric_tuple(aggregate)
            _expect(metrics, expected_metrics, f"{milestone_id} metrics")
            result.append(Milestone(milestone_id, label, *metrics, expected_hash))
        return tuple(result)

    def _verify_completion(self, completion: dict[str, Any]) -> tuple[Path, ...]:
        _expect(completion.get("schema_version"), "maxim-9b-baseline-selector-multi-arm-completion-v1", "completion schema")
        _expect(completion.get("status"), "all_four_scores_completed_outputs_hash_frozen", "completion status")
        for key in (
            "all_four_processes_completed_before_manifest",
            "all_four_returncodes_zero",
            "same_wave_retuning_forbidden",
            "individual_score_content_was_not_parsed_or_printed_by_launcher",
        ):
            _expect(completion.get(key), True, f"completion {key}")
        artifacts = completion.get("artifacts") or {}
        _expect(sorted(artifacts), ["v1_1_primary", "v1_1_secondary", "v1_2_exploratory", "v1_2_primary"], "completion arms")
        experiment_root = self._path(EXPERIMENT)
        verified: list[Path] = []
        for arm, bundle in artifacts.items():
            if not isinstance(bundle, dict):
                raise ArtifactError(f"completion bundle {arm} must be an object")
            _expect(sorted(bundle), ["json", "md", "sha256"], f"completion files for {arm}")
            for kind, descriptor in bundle.items():
                if not isinstance(descriptor, dict):
                    raise ArtifactError(f"completion descriptor {arm}.{kind} must be an object")
                path = _safe_manifest_path(experiment_root, descriptor.get("path"))
                expected_hash = str(descriptor.get("sha256") or "")
                if len(expected_hash) != 64:
                    raise ArtifactError(f"invalid completion hash for {arm}.{kind}")
                _expect(_sha256(path), expected_hash, f"completion artifact {arm}.{kind}")
                verified.append(path)
        primary = artifacts["v1_2_primary"]["json"]
        _expect(primary.get("sha256"), PINNED_SHA256[SELECTOR_SCORE], "primary score completion pin")
        if not _declared_path_matches(primary.get("path"), Path("evaluation_wave_v1/results/v1_2_primary_score.json")):
            raise ArtifactError("completion manifest primary score path is not portable/bound")
        return tuple(verified)

    def load(self) -> SelectorWaveSummary:
        pinned_paths = {relative: self._verify_pin(relative) for relative in PINNED_SHA256}
        comparison = _read_json(pinned_paths[COMPARISON])
        milestones = self._load_milestones(comparison)
        if len(milestones) != 7 or milestones[-1].correct != 238:
            raise ArtifactError("canonical ladder must contain seven milestones and end at Source V7=238")

        completion = _read_json(pinned_paths[COMPLETION])
        completion_files = self._verify_completion(completion)
        score = _read_json(pinned_paths[SELECTOR_SCORE])
        composition = _read_json(pinned_paths[COMPOSITION])
        anchor_score = _read_json(pinned_paths[SOURCE_V7_SCORE])

        _expect(score.get("schema_version"), "maxim-full274-score-v1", "selector score schema")
        _expect(score.get("label"), "v1_2_primary", "selector score label")
        _expect(score.get("models"), [MODEL], "selector score model closure")
        overall = score.get("overall") or {}
        _expect((overall.get("new_correct"), overall.get("n")), (240, 274), "audited selector score (never infer 241)")
        _expect(overall.get("new_accuracy"), 0.875912, "audited selector accuracy")
        math = (score.get("by_subject") or {}).get("Math") or {}
        history = (score.get("by_subject") or {}).get("History") or {}
        deterministic = (score.get("by_source") or {}).get("deterministic") or {}
        image = (score.get("by_source") or {}).get("image_judge") or {}
        _expect((math.get("new_correct"), math.get("n")), (109, 139), "selector Math slice")
        _expect((history.get("new_correct"), history.get("n")), (10, 10), "selector History slice")
        _expect((deterministic.get("new_correct"), deterministic.get("n")), (158, 177), "selector deterministic slice")
        _expect((image.get("new_correct"), image.get("n")), (82, 97), "selector image slice")
        solver_provenance = score.get("provenance", {}).get("solver_results", {})
        _expect(solver_provenance.get("sha256"), PINNED_SHA256[PRIMARY_SOLVER], "scored solver hash")

        _expect(composition.get("schema_version"), "maxim-9b-baseline-selector-composition-manifest-v1.2", "composition schema")
        _expect(composition.get("status"), "composited_frozen_before_evaluation", "composition freeze status")
        _expect(composition.get("rows_per_solver"), 274, "composition rows")
        _expect(composition.get("model_closure"), [MODEL], "composition model closure")
        _expect(composition.get("runtime_outcome_access"), False, "composition runtime outcome access")
        _expect(composition.get("inputs", {}).get("base_source_solver_sha256"), "9d26067064ee07fe480391759782c86d66adbb76dbc0da0d86ccc1b3f035211e", "Source V7 anchor solver")
        _expect(composition.get("inputs", {}).get("selector_proposals_sha256"), PINNED_SHA256[PROPOSALS], "proposal binding")
        primary_artifacts = composition.get("artifacts", {}).get("primary", {})
        _expect(primary_artifacts.get("solver", {}).get("sha256"), PINNED_SHA256[PRIMARY_SOLVER], "composition primary solver")
        _expect(primary_artifacts.get("decisions", {}).get("sha256"), PINNED_SHA256[DECISIONS], "composition decisions")
        counts = composition.get("composition_counts", {}).get("primary", {})
        _expect(counts, {"base_passthrough_exact_bytes": 272, "bound_structural_selector_replacement": 2}, "primary composition counts")
        preservation = composition.get("preservation") or {}
        _expect((preservation.get("source_union_rows"), preservation.get("source_union_changes_primary")), (156, 0), "source preservation")
        _expect((preservation.get("image_judge_rows"), preservation.get("image_judge_changes_primary")), (97, 0), "image preservation")
        _expect(preservation.get("passthrough_representation"), "exact_original_base_jsonl_line_bytes", "passthrough representation")

        decisions = _index(_read_jsonl(pinned_paths[DECISIONS]), "primary decisions")
        proposals = _index(_read_jsonl(pinned_paths[PROPOSALS]), "selector proposals")
        _expect(len(decisions), 274, "decision row count")
        _expect(len(proposals), 274, "proposal row count")
        _expect(set(decisions), set(proposals), "decision/proposal task set")
        replacements = {
            task_id
            for task_id, row in decisions.items()
            if row.get("composition_action") == "bound_structural_selector_replacement"
        }
        _expect(replacements, set(TASK_SPECS), "replacement task ids")
        passthrough = {
            task_id
            for task_id, row in decisions.items()
            if row.get("composition_action") == "base_passthrough_exact_bytes"
        }
        _expect(len(passthrough), 272, "byte-exact passthrough row count")
        if replacements & passthrough or replacements | passthrough != set(decisions):
            raise ArtifactError("composition actions do not partition all 274 rows")

        anchor_outcomes = _index(anchor_score.get("task_outcomes") or [], "Source V7 outcomes")
        selector_outcomes = _index(score.get("task_outcomes") or [], "selector outcomes")
        _expect(set(anchor_outcomes), set(selector_outcomes), "anchor/selector outcome task set")
        fixes = {
            task_id
            for task_id in anchor_outcomes
            if not bool(anchor_outcomes[task_id].get("new_correct"))
            and bool(selector_outcomes[task_id].get("new_correct"))
        }
        regressions = {
            task_id
            for task_id in anchor_outcomes
            if bool(anchor_outcomes[task_id].get("new_correct"))
            and not bool(selector_outcomes[task_id].get("new_correct"))
        }
        _expect(fixes, set(TASK_SPECS), "fixes versus Source V7")
        _expect(regressions, set(), "regressions versus Source V7")

        task_provenance: list[SelectorTaskProvenance] = []
        for task_id, spec in TASK_SPECS.items():
            decision = decisions[task_id]
            proposal = proposals[task_id]
            outcome = selector_outcomes[task_id]
            anchor_outcome = anchor_outcomes[task_id]
            for field, expected in (
                ("row_index", spec["row_index"]),
                ("base_row_sha256", spec["base"]),
                ("output_row_sha256", spec["output"]),
                ("selector_proposal_row_sha256", spec["proposal"]),
                ("source_row_sha256", spec["source"]),
                ("authoritative_route", "deterministic"),
                ("runtime_outcome_access", False),
                ("protected_by_source_union", False),
            ):
                _expect(decision.get(field), expected, f"{task_id} decision {field}")
            primary = proposal.get("primary") or {}
            native_members = proposal.get("native_member_answers") or {}
            _expect(proposal.get("anchor_answer"), spec["anchor"], f"{task_id} anchor answer")
            _expect(proposal.get("structural_challenger"), spec["selected"], f"{task_id} structural vote")
            _expect(proposal.get("native_group_answer"), spec["selected"], f"{task_id} native vote")
            _expect(native_members.get("v4"), spec["selected"], f"{task_id} native v4 vote")
            _expect(native_members.get("v5"), spec["selected"], f"{task_id} native v5 vote")
            _expect(proposal.get("parallel_group_answer"), spec["selected"], f"{task_id} parallel vote")
            _expect(primary.get("selected_answer"), spec["selected"], f"{task_id} selected answer")
            _expect(primary.get("action"), "propose_challenger", f"{task_id} selector action")
            _expect(primary.get("arm"), "three_group_unanimity", f"{task_id} selector arm")
            _expect(primary.get("reason"), "all_three_preregistered_groups_agree", f"{task_id} selector reason")
            _expect((anchor_outcome.get("new_correct"), outcome.get("new_correct")), (False, True), f"{task_id} correctness change")
            _expect(outcome.get("subject"), spec["subject"], f"{task_id} subject")
            _expect(outcome.get("score_source"), "deterministic", f"{task_id} evaluation route")
            task_provenance.append(
                SelectorTaskProvenance(
                    task_id=task_id,
                    subject=str(spec["subject"]),
                    row_index=int(spec["row_index"]),
                    anchor_answer=str(spec["anchor"]),
                    selected_answer=str(spec["selected"]),
                    structural_answer=str(proposal["structural_challenger"]),
                    native_answer=str(proposal["native_group_answer"]),
                    native_v4_answer=str(native_members["v4"]),
                    native_v5_answer=str(native_members["v5"]),
                    parallel_answer=str(proposal["parallel_group_answer"]),
                    route=str(proposal["authoritative_evaluation_route"]),
                    reason=str(primary["reason"]),
                    base_row_sha256=str(decision["base_row_sha256"]),
                    output_row_sha256=str(decision["output_row_sha256"]),
                    proposal_row_sha256=str(decision["selector_proposal_row_sha256"]),
                    source_row_sha256=str(decision["source_row_sha256"]),
                )
            )

        # A later answer-contract repair is deliberately represented as a
        # post-score null/integrity result, not as a ninth milestone and not as
        # a 241 claim. It changed one solver row but no correctness verdict.
        repair_rule = _read_json(pinned_paths[REPAIR_RULE_FREEZE])
        _expect(
            repair_rule.get("schema_version"),
            "maxim-9b-answer-contract-repair-rule-freeze-v1.1",
            "repair rule schema",
        )
        _expect(
            repair_rule.get("status"),
            "development_rules_frozen_before_candidate_build_not_blind_not_preregistered",
            "repair rule status",
        )
        repair_chronology = repair_rule.get("chronology") or {}
        for key, expected in (
            ("post_score_motivated", True),
            ("historical_residual_outcomes_known_before_design", True),
            ("frozen_before_candidate_materialization", True),
            ("candidate_output_absent_at_freeze", True),
            ("blind_claim", False),
            ("preregistered_claim", False),
        ):
            _expect(repair_chronology.get(key), expected, f"repair chronology {key}")
        _expect(repair_rule.get("runtime_outcome_access"), False, "repair runtime outcome access")
        _expect(
            repair_rule.get("inputs", {})
            .get("v1_2_primary_240_solver", {})
            .get("sha256"),
            PINNED_SHA256[PRIMARY_SOLVER],
            "repair base selector solver",
        )

        repair_freeze = _read_json(pinned_paths[REPAIR_OUTPUT_FREEZE])
        _expect(
            repair_freeze.get("schema_version"),
            "maxim-9b-answer-contract-repair-output-freeze-v1.1",
            "repair output schema",
        )
        _expect(
            repair_freeze.get("status"),
            "development_candidate_frozen_unscored_not_evaluated",
            "repair output freeze status",
        )
        _expect(repair_freeze.get("runtime_outcome_access"), False, "repair output runtime outcome access")
        _expect(
            repair_freeze.get("rule_freeze_sha256"),
            PINNED_SHA256[REPAIR_RULE_FREEZE],
            "repair rule/output binding",
        )
        repair_variant = (repair_freeze.get("base_variants") or {}).get(
            "on_v1_2_primary_240", {}
        )
        _expect(
            repair_variant.get("base_solver", {}).get("sha256"),
            PINNED_SHA256[PRIMARY_SOLVER],
            "repair frozen base solver",
        )
        repair_arm = (repair_variant.get("arms") or {}).get(
            "exploratory_explicit_key_scalar", {}
        )
        repair_artifacts = repair_arm.get("artifacts") or {}
        _expect(
            repair_artifacts.get("candidate_solver", {}).get("sha256"),
            PINNED_SHA256[REPAIR_SOLVER],
            "repair candidate solver pin",
        )
        _expect(
            repair_artifacts.get("decisions", {}).get("sha256"),
            PINNED_SHA256[REPAIR_DECISIONS],
            "repair decisions pin",
        )

        repair_solver_rows = _index(_read_jsonl(pinned_paths[REPAIR_SOLVER]), "repair solver")
        repair_decisions = _index(_read_jsonl(pinned_paths[REPAIR_DECISIONS]), "repair decisions")
        _expect(len(repair_solver_rows), 274, "repair solver rows")
        _expect(set(repair_solver_rows), set(selector_outcomes), "repair solver task set")
        _expect(set(repair_decisions), set(selector_outcomes), "repair decision task set")
        _expect(
            {str(row.get("model") or "") for row in repair_solver_rows.values()},
            {MODEL},
            "repair model closure",
        )
        changed_repair_rows = {
            task_id
            for task_id, row in repair_decisions.items()
            if row.get("action") == "repair_from_raw_response"
        }
        _expect(changed_repair_rows, {"val_0223"}, "repair changed task")
        _expect(
            sum(
                row.get("action") == "preserve_base_exact_bytes"
                for row in repair_decisions.values()
            ),
            273,
            "repair byte-exact passthrough rows",
        )
        repair_decision = repair_decisions["val_0223"]
        _expect(repair_decision.get("row_index"), 221, "repair val_0223 row index")
        _expect(
            repair_decision.get("reason"),
            "unique_explicit_key_scalar_candidate_exploratory",
            "repair val_0223 reason",
        )
        _expect(
            repair_decision.get("parser_status"),
            "unique_explicit_key_scalar_candidate",
            "repair val_0223 parser status",
        )
        _expect(repair_decision.get("runtime_outcome_access"), False, "repair decision outcome access")
        _expect(repair_decision.get("protected_by_source_union"), False, "repair source protection")
        _expect(
            repair_decision.get("base_row_sha256"),
            "21cf70cbaa68ed0e2b65311e11094dfbacfdac56819c32d324d2981290184096",
            "repair val_0223 base row",
        )
        _expect(
            repair_decision.get("output_row_sha256"),
            "894932605d20d915cdb5f261e8f2d356ed180adcb9e3589035dd5ae787ec8752",
            "repair val_0223 output row",
        )

        repair_score = _read_json(pinned_paths[REPAIR_SCORE])
        _expect(repair_score.get("schema_version"), "maxim-full274-score-v1", "repair score schema")
        _expect(
            repair_score.get("label"),
            "maxim_9b_v1_2_primary_plus_answer_contract_repair_v1_1",
            "repair score label",
        )
        _expect(repair_score.get("models"), [MODEL], "repair score model closure")
        repair_overall = repair_score.get("overall") or {}
        _expect(
            (repair_overall.get("new_correct"), repair_overall.get("n")),
            (240, 274),
            "repair null score",
        )
        _expect(repair_overall.get("new_accuracy"), 0.875912, "repair null accuracy")
        _expect(
            repair_score.get("provenance", {})
            .get("solver_results", {})
            .get("sha256"),
            PINNED_SHA256[REPAIR_SOLVER],
            "repair score/solver binding",
        )
        repair_outcomes = _index(repair_score.get("task_outcomes") or [], "repair outcomes")
        _expect(set(repair_outcomes), set(selector_outcomes), "repair outcome task set")
        correctness_changes = {
            task_id
            for task_id in selector_outcomes
            if bool(selector_outcomes[task_id].get("new_correct"))
            != bool(repair_outcomes[task_id].get("new_correct"))
        }
        _expect(correctness_changes, set(), "post-score repair correctness delta")
        _expect(
            repair_outcomes["val_0223"].get("new_correct"),
            True,
            "repair val_0223 stays correct",
        )

        verified_files = tuple(
            dict.fromkeys(
                [*pinned_paths.values(), *completion_files]
                + [self._path(spec[2]) for spec in MILESTONE_SPECS]
            )
        )
        return SelectorWaveSummary(
            model=MODEL,
            milestones=milestones,
            correct=240,
            rows=274,
            math_correct=109,
            math_rows=139,
            history_correct=10,
            history_rows=10,
            deterministic_correct=158,
            deterministic_rows=177,
            image_correct=82,
            image_rows=97,
            fixes=2,
            regressions=0,
            passthrough_rows=272,
            replacement_rows=2,
            source_preserved_rows=156,
            image_preserved_rows=97,
            tasks=tuple(task_provenance),
            comparison_sha256=PINNED_SHA256[COMPARISON],
            completion_sha256=PINNED_SHA256[COMPLETION],
            score_sha256=PINNED_SHA256[SELECTOR_SCORE],
            composition_sha256=PINNED_SHA256[COMPOSITION],
            solver_sha256=PINNED_SHA256[PRIMARY_SOLVER],
            repair_rule_sha256=PINNED_SHA256[REPAIR_RULE_FREEZE],
            repair_output_freeze_sha256=PINNED_SHA256[REPAIR_OUTPUT_FREEZE],
            repair_solver_sha256=PINNED_SHA256[REPAIR_SOLVER],
            repair_score_sha256=PINNED_SHA256[REPAIR_SCORE],
            repair_task_id="val_0223",
            repair_base_row_sha256=str(repair_decision["base_row_sha256"]),
            repair_output_row_sha256=str(repair_decision["output_row_sha256"]),
            verified_files=verified_files,
        )

    def validation_report(self) -> dict[str, Any]:
        summary = self.load()
        return {
            "status": "ok",
            "label": "Baseline Selector v1.2 · audited development layer",
            "model": summary.model,
            "canonical_milestones": [
                {
                    "id": milestone.milestone_id,
                    "correct": milestone.correct,
                    "rows": milestone.rows,
                    "accuracy": round(milestone.accuracy, 6),
                }
                for milestone in summary.milestones
            ],
            "selector_layer": {
                "correct": summary.correct,
                "rows": summary.rows,
                "accuracy": round(summary.accuracy, 6),
                "math": [summary.math_correct, summary.math_rows],
                "history": [summary.history_correct, summary.history_rows],
                "deterministic": [summary.deterministic_correct, summary.deterministic_rows],
                "image_judge": [summary.image_correct, summary.image_rows],
                "fixes_vs_source_v7": summary.fixes,
                "regressions_vs_source_v7": summary.regressions,
                "supported_correct_count": 240,
                "unsupported_claim": "241/274 (not observed)",
            },
            "preservation": {
                "byte_exact_passthrough": summary.passthrough_rows,
                "source_rows_unchanged": summary.source_preserved_rows,
                "image_rows_unchanged": summary.image_preserved_rows,
            },
            "changed_tasks": [
                {
                    "task_id": task.task_id,
                    "subject": task.subject,
                    "answer_change": f"{task.anchor_answer}->{task.selected_answer}",
                    "route": task.route,
                    "selector_reason": task.reason,
                }
                for task in summary.tasks
            ],
            "integrity": {
                "completion_sha256": summary.completion_sha256,
                "solver_sha256": summary.solver_sha256,
                "verified_files": len(summary.verified_files),
            },
            "post_score_repair": {
                "status": "audited null result",
                "changed_task": summary.repair_task_id,
                "correct_before": 240,
                "correct_after": 240,
                "score_sha256": summary.repair_score_sha256,
                "candidate_solver_sha256": summary.repair_solver_sha256,
                "blind": False,
                "preregistered": False,
            },
            "caveat": "known development benchmark; frozen one-shot multi-arm wave; not a blind holdout",
            "claim_boundary": "selector evidence is not source lookup, QA accuracy, or hidden chain-of-thought",
        }
