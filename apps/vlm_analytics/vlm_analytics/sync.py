from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import (
    DEFAULT_KEY_PATH,
    DEFAULT_SERVER,
    DEFAULT_USER,
    REMOTE_BUNDLES,
    REMOTE_MANIFEST,
)
from .database import Database
from .importer import ImportResult, import_run


@dataclass
class SyncSettings:
    server: str = DEFAULT_SERVER
    user: str = DEFAULT_USER
    key_path: str = DEFAULT_KEY_PATH


class SyncError(RuntimeError):
    pass


class SyncManager:
    def __init__(self, database: Database, cache_dir: Path):
        self.database = database
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def settings(self) -> SyncSettings:
        return SyncSettings(
            server=self.database.get_setting("server", DEFAULT_SERVER),
            user=self.database.get_setting("user", DEFAULT_USER),
            key_path=self.database.get_setting("key_path", DEFAULT_KEY_PATH),
        )

    @staticmethod
    def _binary(name: str) -> str:
        binary = shutil.which(name)
        if not binary:
            raise SyncError(
                f"Команда {name} не найдена. Установите Windows OpenSSH Client."
            )
        return binary

    def _scp(self, remote_path: str, local_path: Path, settings: SyncSettings) -> None:
        key_path = str(Path(settings.key_path).expanduser())
        command = [
            self._binary("scp"),
            "-q",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-i",
            key_path,
            f"{settings.user}@{settings.server}:{remote_path}",
            str(local_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            ),
            timeout=180,
        )
        if completed.returncode:
            error = completed.stderr.strip() or completed.stdout.strip()
            raise SyncError(f"Не удалось скачать {remote_path}: {error}")

    def sync_all(
        self, progress: Callable[[str], None] | None = None
    ) -> list[ImportResult]:
        settings = self.settings()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_dir = self.cache_dir / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)

        def notify(message: str) -> None:
            if progress:
                progress(message)

        manifest_path = session_dir / "validation_manifest.jsonl"
        notify("Скачиваю манифест…")
        self._scp(REMOTE_MANIFEST, manifest_path, settings)
        imported: list[ImportResult] = []
        for index, bundle in enumerate(REMOTE_BUNDLES, 1):
            notify(f"{index}/4: {bundle.display_name} — скачиваю результаты…")
            raw_path = session_dir / f"{bundle.key}_raw.jsonl"
            judge_path = session_dir / f"{bundle.key}_judge.jsonl"
            self._scp(bundle.raw_path, raw_path, settings)
            self._scp(bundle.judge_path, judge_path, settings)
            notify(f"{index}/4: {bundle.display_name} — сохраняю в БД…")
            imported.append(
                import_run(
                    self.database,
                    run_key=bundle.key,
                    display_name=bundle.display_name,
                    raw_path=raw_path,
                    judge_path=judge_path,
                    manifest_path=manifest_path,
                    raw_source=bundle.raw_path,
                    judge_source=bundle.judge_path,
                    observed_at=datetime.now(timezone.utc)
                    .astimezone()
                    .isoformat(timespec="seconds"),
                )
            )
        self.database.set_setting(
            "last_sync",
            datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        )
        notify("Синхронизация завершена.")
        return imported
