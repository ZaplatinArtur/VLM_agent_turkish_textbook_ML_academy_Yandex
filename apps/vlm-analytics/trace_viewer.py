from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vlm_trace_viewer import (
    ArtifactError,
    HoldoutIntegrityError,
    NineBV7ArtifactAdapter,
    ReplayAggregateError,
    SelectorWaveAdapter,
    V7ArtifactAdapter,
    build_active_selector_dataset,
    empty_milestone_schema,
    intermediate_timeline_schema,
    load_frozen_9b_comparison,
    load_holdout80_summary,
    unloaded_replay_report,
)


DEFAULT_NINE_B_COMPARISON = Path(
    "reports/maxim_9b_source_replay_v1_20260809/comparison.json"
)


def resolve_nine_b_comparison_path(
    explicit: Path | None,
    artifact_root: Path,
) -> Path | None:
    """Resolve the only implicit 9B manifest; validation still happens downstream."""

    if explicit is not None:
        return explicit.expanduser().resolve()
    candidate = (artifact_root / DEFAULT_NINE_B_COMPARISON).resolve()
    return candidate if candidate.is_file() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline visual trace explorer for the frozen V7 VLM artifacts"
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Project root containing reports/maxim_official_exact_source_v2_20260805",
    )
    parser.add_argument(
        "--nine-b-comparison",
        type=Path,
        help=(
            "Explicit seven-milestone 9B comparison manifest. When omitted, the "
            "single frozen comparison under the discovered artifact root is used "
            "if present. No score is labelled 9B-only without full hash validation."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=("auto", "nine-b-v7", "archived-27b-v7"),
        default="auto",
        help=(
            "auto selects the validated 9B V7 trace when the explicit or canonical "
            "comparison exists; otherwise it opens the marked archived 27B reference"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Join and validate all artifacts, print a JSON report, then exit",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Render the window into a PNG and exit (useful for smoke/visual QA)",
    )
    parser.add_argument(
        "--screenshot-tab",
        type=int,
        default=1,
        help=(
            "0 = trace explorer, 1 = Holdout80 source evidence, 2 = V7 metrics, "
            "3 = honest 9B milestone schema, 4 = audited selector v1.2"
        ),
    )
    parser.add_argument(
        "--task",
        default="val_0163",
        help="Task selected at startup; defaults to a certified 9B row with a local image",
    )
    parser.add_argument(
        "--detail-tab",
        type=int,
        default=0,
        help="Task detail tab: 0 reasoning, 1 route, 2 evidence, 3 comparison, 4 JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_only and hasattr(sys.stdout, "reconfigure"):
        # Windows installations may default to a legacy code page which turns
        # the UTF-8 report label into replacement characters.
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
    try:
        holdout80 = load_holdout80_summary()
        archived_adapter = V7ArtifactAdapter(args.artifact_root)
        comparison_path = resolve_nine_b_comparison_path(
            args.nine_b_comparison,
            archived_adapter.root,
        )
        nine_b_comparison = (
            load_frozen_9b_comparison(comparison_path)
            if comparison_path
            else None
        )
        selector_summary = (
            SelectorWaveAdapter(archived_adapter.root).load()
            if nine_b_comparison
            else None
        )
        if args.validate_only:
            nine_b_trace = (
                NineBV7ArtifactAdapter(
                    nine_b_comparison,
                    display_asset_root=archived_adapter.root,
                ).load()
                if nine_b_comparison
                else None
            )
            active_selector_trace = (
                build_active_selector_dataset(nine_b_trace, selector_summary)
                if nine_b_trace is not None and selector_summary is not None
                else None
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "holdout80_source_evidence": holdout80.validation_report(),
                        "archived_27b_v7_reference": archived_adapter.validation_report(),
                        "nine_b_comparison": (
                            nine_b_comparison.validation_report()
                            if nine_b_comparison
                            else {
                                **unloaded_replay_report(),
                                "primary_milestones": list(empty_milestone_schema()),
                                "intermediate_timeline": list(
                                    intermediate_timeline_schema()
                                ),
                            }
                        ),
                        "nine_b_v7_trace": (
                            {
                                "status": "ok",
                                "role": "canonical seven-step lineage endpoint; not active headline",
                                "rows": nine_b_trace.summary.rows,
                                "correct": nine_b_trace.summary.correct,
                                "accuracy": nine_b_trace.summary.accuracy,
                                "pipeline": nine_b_trace.summary.pipeline_provenance,
                                "local_question_images": sum(
                                    task.question_image is not None
                                    for task in nine_b_trace.tasks
                                ),
                            }
                            if nine_b_trace
                            else unloaded_replay_report()
                        ),
                        "active_all_9b_analytics": (
                            {
                                "status": "ok",
                                "label": "Baseline Selector v1.2 · active audited development result",
                                "correct": active_selector_trace.summary.correct,
                                "rows": active_selector_trace.summary.rows,
                                "accuracy": active_selector_trace.summary.accuracy,
                                "math": [
                                    selector_summary.math_correct,
                                    selector_summary.math_rows,
                                ],
                                "history": [
                                    selector_summary.history_correct,
                                    selector_summary.history_rows,
                                ],
                                "deterministic": [
                                    selector_summary.deterministic_correct,
                                    selector_summary.deterministic_rows,
                                ],
                                "image_judge": [
                                    selector_summary.image_correct,
                                    selector_summary.image_rows,
                                ],
                                "task_overlay_correct": sum(
                                    task.correct for task in active_selector_trace.tasks
                                ),
                            }
                            if active_selector_trace is not None
                            else unloaded_replay_report()
                        ),
                        "nine_b_selector_v1_2": (
                            SelectorWaveAdapter(archived_adapter.root).validation_report()
                            if selector_summary
                            else unloaded_replay_report()
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.dataset == "nine-b-v7" and nine_b_comparison is None:
            raise ReplayAggregateError(
                "--dataset nine-b-v7 requires --nine-b-comparison with the complete hash chain"
            )
        use_nine_b = args.dataset == "nine-b-v7" or (
            args.dataset == "auto" and nine_b_comparison is not None
        )
        if use_nine_b:
            dataset = NineBV7ArtifactAdapter(
                nine_b_comparison,  # type: ignore[arg-type]
                display_asset_root=archived_adapter.root,
            ).load()
            active_dataset = "nine-b-v7"
        else:
            dataset = archived_adapter.load()
            active_dataset = "archived-27b-v7"
        qa_reference_summary = (
            dataset.summary
            if active_dataset == "archived-27b-v7"
            else archived_adapter.load().summary
        )
    except (ArtifactError, HoldoutIntegrityError, ReplayAggregateError) as exc:
        print(f"artifact error: {exc}", file=sys.stderr)
        return 2

    from PySide6.QtCore import QPoint, QTimer, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QApplication

    from vlm_trace_viewer.ui import TraceViewerWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("VLM Trace")
    window = TraceViewerWindow(
        dataset,
        holdout80,
        nine_b_comparison,
        selector_summary,
        active_dataset=active_dataset,
        qa_reference_summary=qa_reference_summary,
    )
    window.explorer.select_task_id(args.task)
    window.explorer.detail.tabs.setCurrentIndex(max(0, min(args.detail_tab, 4)))
    window.tabs.setCurrentIndex(max(0, min(args.screenshot_tab, 4)))
    window.resize(1900, 1080)
    if args.screenshot:
        # Keep the requested logical geometry even on a smaller/high-DPI desktop.
        # QWidget.grab() returns device pixels (often 3800 px on a 200% display),
        # so the screenshot path renders into an explicit 1900x1080 image instead.
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.show()

    if args.screenshot:
        output = args.screenshot.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        def capture() -> None:
            canvas = QImage(1900, 1080, QImage.Format.Format_ARGB32)
            canvas.fill(Qt.GlobalColor.transparent)
            painter = QPainter(canvas)
            window.render(painter, QPoint())
            painter.end()
            if not canvas.save(str(output)):
                print(f"cannot save screenshot: {output}", file=sys.stderr)
                app.exit(3)
                return
            print(output)
            window.close()
            app.quit()

        QTimer.singleShot(900, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
