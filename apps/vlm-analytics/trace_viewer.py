from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vlm_trace_viewer import (
    ArtifactError,
    HoldoutIntegrityError,
    V7ArtifactAdapter,
    load_holdout80_summary,
)


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
        help="0 = trace explorer, 1 = Holdout80 source evidence, 2 = V7 metrics",
    )
    parser.add_argument(
        "--task",
        default="val_0196",
        help="Task selected at startup; defaults to the V7 direct source gain",
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
        adapter = V7ArtifactAdapter(args.artifact_root)
        if args.validate_only:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "holdout80_source_evidence": holdout80.validation_report(),
                        "v7_development_replay": adapter.validation_report(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        dataset = adapter.load()
    except (ArtifactError, HoldoutIntegrityError) as exc:
        print(f"artifact error: {exc}", file=sys.stderr)
        return 2

    from PySide6.QtCore import QPoint, QTimer, Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtWidgets import QApplication

    from vlm_trace_viewer.ui import TraceViewerWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("VLM Trace")
    window = TraceViewerWindow(dataset, holdout80)
    window.explorer.select_task_id(args.task)
    window.explorer.detail.tabs.setCurrentIndex(max(0, min(args.detail_tab, 4)))
    window.tabs.setCurrentIndex(max(0, min(args.screenshot_tab, 2)))
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
