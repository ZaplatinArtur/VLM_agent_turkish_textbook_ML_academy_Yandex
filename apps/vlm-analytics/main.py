from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vlm_analytics.config import DATASET_VERSION, application_dir, default_database_path
from vlm_analytics.database import Database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM Analytics")
    parser.add_argument(
        "--db",
        type=Path,
        default=default_database_path(),
        help="Путь к SQLite базе",
    )
    parser.add_argument(
        "--sync-once",
        action="store_true",
        help="Скачать текущие результаты по SSH, импортировать и выйти",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Сохранить скриншот главного окна и выйти",
    )
    parser.add_argument(
        "--screenshot-tab",
        type=int,
        default=0,
        help="Индекс вкладки для тестового скриншота",
    )
    parser.add_argument("--import-run-key", help="Импортировать локальный прогон")
    parser.add_argument("--display-name")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--judge", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-version")
    parser.add_argument(
        "--paired-summary",
        action="store_true",
        help="Вывести парные сравнения последних прогонов с b0_no_tools",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    database = Database(args.db.resolve())
    database.initialize()

    if args.import_run_key:
        missing = [
            name
            for name, value in (
                ("--display-name", args.display_name),
                ("--raw", args.raw),
                ("--judge", args.judge),
                ("--manifest", args.manifest),
            )
            if not value
        ]
        if missing:
            raise SystemExit(f"Для импорта нужны: {', '.join(missing)}")
        from vlm_analytics.importer import import_run

        result = import_run(
            database,
            run_key=args.import_run_key,
            display_name=args.display_name,
            raw_path=args.raw,
            judge_path=args.judge,
            manifest_path=args.manifest,
            dataset_version=args.dataset_version or DATASET_VERSION,
        )
        print(result.message)
        return 0

    if args.paired_summary:
        from vlm_analytics.analytics import AnalyticsService

        print(
            json.dumps(
                AnalyticsService(database).paired_comparisons(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.sync_once:
        from vlm_analytics.sync import SyncManager

        manager = SyncManager(database, application_dir() / ".sync_cache")
        results = manager.sync_all(progress=print)
        for result in results:
            print(result.message)
        return 0

    if args.screenshot:
        from PySide6.QtWidgets import QApplication

        from vlm_analytics.ui import MainWindow

        app = QApplication.instance() or QApplication(sys.argv)
        window = MainWindow(database)
        window.tabs.setCurrentIndex(
            max(0, min(args.screenshot_tab, window.tabs.count() - 1))
        )
        window.resize(1600, 960)
        window.show()
        app.processEvents()
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(args.screenshot)):
            raise RuntimeError(f"Не удалось сохранить {args.screenshot}")
        window.close()
        return 0

    from vlm_analytics.ui import run_gui

    return run_gui(database)


if __name__ == "__main__":
    raise SystemExit(main())
