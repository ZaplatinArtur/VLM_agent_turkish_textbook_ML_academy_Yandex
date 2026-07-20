from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import mimetypes
import os
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .adjudication import AdjudicationStore, build_adjudication_context


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} of {path} is not a JSON object")
            records.append(value)
    return records


class AnnotationStore:
    STATUSES = {"draft", "complete", "skipped"}
    MODES = {"pointwise", "pairwise"}
    WINNERS = {"A", "B", "tie", "unjudgeable"}
    LABEL_BY_SCORE = {
        0: "incorrect",
        1: "partially_correct",
        2: "partially_correct",
        3: "mostly_correct",
        4: "fully_correct",
    }

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        if path.exists():
            for record in load_jsonl(path):
                key = str(record.get("annotation_id") or record.get("task_id") or "")
                if key:
                    self._records[key] = record

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._records.values()]

    def upsert(self, annotation: dict[str, Any]) -> dict[str, Any]:
        task_id = str(annotation.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("task_id is required")
        annotation_id = str(annotation.get("annotation_id") or task_id).strip()
        record = dict(annotation)
        record["annotation_id"] = annotation_id
        record["task_id"] = task_id
        self._validate(record)
        record["updated_at"] = datetime.now(UTC).isoformat()
        with self._lock:
            self._records[annotation_id] = record
            self._write_atomic()
        return dict(record)

    @classmethod
    def _validate(cls, record: dict[str, Any]) -> None:
        status = record.get("status", "draft")
        mode = record.get("mode", "pointwise")
        if status not in cls.STATUSES:
            raise ValueError(f"invalid annotation status: {status}")
        if mode not in cls.MODES:
            raise ValueError(f"invalid annotation mode: {mode}")

        score = record.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, int) or score not in range(5)):
            raise ValueError("score must be an integer from 0 to 4 or null")
        winner = record.get("winner")
        if winner is not None and winner not in cls.WINNERS:
            raise ValueError(f"invalid pairwise winner: {winner}")

        if mode == "pointwise":
            if winner is not None:
                raise ValueError("pointwise annotations must not have a winner")
            if status == "complete" and score is None:
                raise ValueError("complete pointwise annotations require a score")
            expected_label = cls.LABEL_BY_SCORE.get(score)
            label = record.get("label")
            if label not in (None, expected_label):
                raise ValueError(f"label {label!r} is inconsistent with score {score!r}")
            strict_correct = record.get("strict_correct")
            expected_strict = score == 4 if score is not None else None
            if strict_correct not in (None, expected_strict):
                raise ValueError("strict_correct is inconsistent with score")
        else:
            if score is not None or record.get("label") is not None:
                raise ValueError("pairwise annotations must not have a pointwise score or label")
            if status == "complete" and winner is None:
                raise ValueError("complete pairwise annotations require a winner")

        for field_name in ("final_answer_correct", "reasoning_correct", "complete"):
            value = record.get(field_name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field_name} must be a boolean or null")
        reference_issue = record.get("reference_quality_issue")
        if reference_issue is not None and not isinstance(reference_issue, bool):
            raise ValueError("reference_quality_issue must be a boolean")
        confidence = record.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise ValueError("confidence must be numeric")
            if not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
        error_types = record.get("error_types")
        if error_types is not None and (
            not isinstance(error_types, list) or not all(isinstance(value, str) for value in error_types)
        ):
            raise ValueError("error_types must be an array of strings")
        for field_name in ("rationale", "annotator"):
            value = record.get(field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string")

    def _write_atomic(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for key in sorted(self._records):
                handle.write(json.dumps(self._records[key], ensure_ascii=False) + "\n")
        os.replace(temporary, self.path)

    def export_csv(self) -> bytes:
        records = self.list()
        fields = [
            "annotation_id",
            "task_id",
            "setup",
            "pair_id",
            "mode",
            "subject",
            "grade",
            "answer_type",
            "status",
            "score",
            "label",
            "final_answer_correct",
            "reasoning_correct",
            "complete",
            "confidence",
            "reference_quality_issue",
            "error_types",
            "rationale",
            "winner",
            "candidate_a_setup",
            "candidate_b_setup",
            "side_swapped",
            "mirrored",
            "annotator",
            "updated_at",
        ]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            normalized = dict(record)
            if isinstance(normalized.get("error_types"), list):
                normalized["error_types"] = ";".join(normalized["error_types"])
            writer.writerow(normalized)
        return ("\ufeff" + output.getvalue()).encode("utf-8")


class GoldStore:
    STATUSES = {"draft", "verified", "skipped"}
    QUALITIES = {"unknown", "clear", "ambiguous", "incorrect", "unreadable"}

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        if path.exists():
            for record in load_jsonl(path):
                task_id = str(record.get("task_id") or "").strip()
                if task_id:
                    self._records[task_id] = record

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._records.values()]

    def upsert(self, value: dict[str, Any]) -> dict[str, Any]:
        task_id = str(value.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("task_id is required")
        status = str(value.get("status") or "draft")
        quality = str(value.get("quality") or "unknown")
        if status not in self.STATUSES:
            raise ValueError(f"invalid gold status: {status}")
        if quality not in self.QUALITIES:
            raise ValueError(f"invalid gold quality: {quality}")
        if status == "verified" and quality == "unknown":
            raise ValueError("verified gold requires an explicit quality")
        record = dict(value)
        record["task_id"] = task_id
        record["status"] = status
        record["quality"] = quality
        for field in ("acceptable_answers", "subanswers"):
            current = record.get(field)
            if current is None:
                record[field] = []
            elif not isinstance(current, list):
                raise ValueError(f"{field} must be an array")
            else:
                record[field] = [str(item).strip() for item in current if str(item).strip()]
        for field in ("transcription", "notes", "annotator"):
            current = record.get(field)
            if current is not None and not isinstance(current, str):
                raise ValueError(f"{field} must be a string")
        if (
            status == "verified"
            and quality in {"clear", "ambiguous"}
            and not str(record.get("transcription") or "").strip()
            and not record["subanswers"]
        ):
            raise ValueError("verified readable gold requires a transcription or subanswers")
        record["updated_at"] = datetime.now(UTC).isoformat()
        with self._lock:
            self._records[task_id] = record
            self._write_atomic()
        return dict(record)

    def _write_atomic(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for key in sorted(self._records):
                handle.write(json.dumps(self._records[key], ensure_ascii=False) + "\n")
        os.replace(temporary, self.path)

    def export_csv(self) -> bytes:
        fields = [
            "task_id", "status", "quality", "transcription", "acceptable_answers",
            "subanswers", "notes", "annotator", "updated_at",
        ]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for value in self.list():
            record = dict(value)
            for field in ("acceptable_answers", "subanswers"):
                if isinstance(record.get(field), list):
                    record[field] = ";".join(record[field])
            writer.writerow(record)
        return ("\ufeff" + output.getvalue()).encode("utf-8")


class ImageCache:
    _ALLOWED_HOST_SUFFIXES = (
        "yadi.sk",
        "disk.yandex.com",
        "disk.yandex.ru",
        "yandex.net",
        "yandex-team.ru",
        "odevjet.com",
    )

    def __init__(self, directory: Path, *, max_bytes: int = 25_000_000) -> None:
        self.directory = directory
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        directory.mkdir(parents=True, exist_ok=True)

    def _validate(self, url: str) -> urllib.parse.ParseResult:
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").casefold()
        if parsed.scheme != "https" or not host:
            raise ValueError("only HTTPS image URLs are supported")
        if not any(host == suffix or host.endswith("." + suffix) for suffix in self._ALLOWED_HOST_SUFFIXES):
            raise ValueError(f"image host is not allowlisted: {host}")
        return parsed

    def _download_url(self, public_url: str, parsed: urllib.parse.ParseResult) -> str:
        host = (parsed.hostname or "").casefold()
        if host in {"yadi.sk", "disk.yandex.com", "disk.yandex.ru"}:
            query = urllib.parse.urlencode({"public_key": public_url})
            api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?{query}"
            request = urllib.request.Request(api_url, headers={"User-Agent": "vlm-judge/0.1"})
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.load(response)
            href = payload.get("href")
            if not href:
                raise ValueError("Yandex API did not return a download URL")
            return str(href)
        return public_url

    def get(self, public_url: str) -> tuple[bytes, str]:
        with self._lock:
            return self._get_locked(public_url)

    def _get_locked(self, public_url: str) -> tuple[bytes, str]:
        parsed = self._validate(public_url)
        key = hashlib.sha256(public_url.encode("utf-8")).hexdigest()
        data_path = self.directory / f"{key}.bin"
        meta_path = self.directory / f"{key}.json"
        if data_path.exists() and meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            return data_path.read_bytes(), str(metadata.get("content_type") or "image/jpeg")

        download_url = self._download_url(public_url, parsed)
        request = urllib.request.Request(download_url, headers={"User-Agent": "vlm-judge/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = response.headers.get_content_type()
            if not content_type.startswith("image/"):
                raise ValueError(f"remote resource is not an image: {content_type}")
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > self.max_bytes:
                raise ValueError("remote image exceeds size limit")
            data = response.read(self.max_bytes + 1)
        if len(data) > self.max_bytes:
            raise ValueError("remote image exceeds size limit")
        temporary = data_path.with_suffix(".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, data_path)
        meta_path.write_text(
            json.dumps({"url": public_url, "content_type": content_type}, ensure_ascii=False),
            encoding="utf-8",
        )
        return data, content_type


def _handler_factory(
    tasks: list[dict[str, Any]],
    dataset_name: str,
    annotations: AnnotationStore,
    gold: GoldStore,
    judge_results: list[dict[str, Any]],
    adjudications: AdjudicationStore,
    image_cache: ImageCache,
    static_root: Path,
    *,
    low_confidence_threshold: float,
    agreement_sample_rate: float,
    agreement_sample_seed: str,
):
    class Handler(BaseHTTPRequestHandler):
        server_version = "VLMJudgeUI/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, value: Any, status: int = HTTPStatus.OK) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _send_bytes(self, payload: bytes, content_type: str, *, filename: str | None = None) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/health":
                context = build_adjudication_context(
                    tasks,
                    judge_results,
                    annotations.list(),
                    adjudications.list(),
                    low_confidence_threshold=low_confidence_threshold,
                    agreement_sample_rate=agreement_sample_rate,
                    agreement_sample_seed=agreement_sample_seed,
                )
                self._send_json(
                    {
                        "ok": True,
                        "tasks": len(tasks),
                        "annotations": len(annotations.list()),
                        "gold": len(gold.list()),
                        "judge_results": len(judge_results),
                        "adjudications": len(adjudications.list()),
                        "adjudication_queue": context["stats"]["queue_items"],
                    }
                )
                return
            if parsed.path == "/api/tasks":
                self._send_json({"dataset": dataset_name, "tasks": tasks})
                return
            if parsed.path == "/api/annotations":
                self._send_json({"annotations": annotations.list()})
                return
            if parsed.path == "/api/gold":
                self._send_json({"gold": gold.list()})
                return
            if parsed.path == "/api/adjudication-context":
                self._send_json(
                    build_adjudication_context(
                        tasks,
                        judge_results,
                        annotations.list(),
                        adjudications.list(),
                        low_confidence_threshold=low_confidence_threshold,
                        agreement_sample_rate=agreement_sample_rate,
                        agreement_sample_seed=agreement_sample_seed,
                    )
                )
                return
            if parsed.path == "/api/adjudications":
                self._send_json({"adjudications": adjudications.list()})
                return
            if parsed.path == "/api/export.csv":
                self._send_bytes(annotations.export_csv(), "text/csv; charset=utf-8", filename="annotations.csv")
                return
            if parsed.path == "/api/export.jsonl":
                payload = "".join(
                    json.dumps(record, ensure_ascii=False) + "\n"
                    for record in annotations.list()
                ).encode("utf-8")
                self._send_bytes(payload, "application/x-ndjson; charset=utf-8", filename="annotations.jsonl")
                return
            if parsed.path == "/api/export-gold.csv":
                self._send_bytes(gold.export_csv(), "text/csv; charset=utf-8", filename="gold_transcriptions.csv")
                return
            if parsed.path == "/api/export-gold.jsonl":
                payload = "".join(
                    json.dumps(record, ensure_ascii=False) + "\n" for record in gold.list()
                ).encode("utf-8")
                self._send_bytes(payload, "application/x-ndjson; charset=utf-8", filename="gold_transcriptions.jsonl")
                return
            if parsed.path == "/api/export-adjudications.csv":
                self._send_bytes(
                    adjudications.export_csv(),
                    "text/csv; charset=utf-8",
                    filename="adjudications.csv",
                )
                return
            if parsed.path == "/api/export-adjudications.jsonl":
                payload = "".join(
                    json.dumps(record, ensure_ascii=False) + "\n" for record in adjudications.list()
                ).encode("utf-8")
                self._send_bytes(
                    payload,
                    "application/x-ndjson; charset=utf-8",
                    filename="adjudications.jsonl",
                )
                return
            if parsed.path == "/api/image":
                url = urllib.parse.parse_qs(parsed.query).get("url", [""])[0]
                try:
                    data, content_type = image_cache.get(url)
                except Exception as exc:
                    self._send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_GATEWAY)
                    return
                self._send_bytes(data, content_type)
                return
            self._serve_static(parsed.path)

        def _serve_static(self, request_path: str) -> None:
            relative = urllib.parse.unquote(request_path).lstrip("/") or "index.html"
            candidate = (static_root / relative).resolve()
            if not candidate.is_relative_to(static_root.resolve()) or not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = candidate.read_bytes()
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self._send_bytes(payload, content_type)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path not in {"/api/annotations", "/api/gold", "/api/adjudications"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if not 0 < content_length <= 2_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict):
                    raise ValueError("request must be a JSON object")
                if parsed.path == "/api/annotations":
                    saved = annotations.upsert(payload)
                elif parsed.path == "/api/gold":
                    saved = gold.upsert(payload)
                else:
                    saved = adjudications.upsert(payload)
            except Exception as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"saved": saved})

    return Handler


def serve_ui(
    dataset_path: Path,
    annotations_path: Path,
    gold_path: Path,
    *,
    judge_results_path: Path | None = None,
    adjudications_path: Path | None = None,
    low_confidence_threshold: float = 0.75,
    agreement_sample_rate: float = 0.10,
    agreement_sample_seed: str = "adjudication-control-v1",
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    tasks = load_jsonl(dataset_path)
    if not tasks:
        raise ValueError("dataset is empty")
    static_root = Path(__file__).with_name("ui")
    if not (static_root / "index.html").exists():
        raise FileNotFoundError("UI assets are missing")
    store = AnnotationStore(annotations_path)
    gold_store = GoldStore(gold_path)
    judge_results = load_jsonl(judge_results_path) if judge_results_path is not None else []
    adjudication_store = AdjudicationStore(
        adjudications_path or annotations_path.with_name("adjudications.jsonl")
    )
    image_cache = ImageCache(annotations_path.parent / "image_cache")
    handler = _handler_factory(
        tasks,
        dataset_path.name,
        store,
        gold_store,
        judge_results,
        adjudication_store,
        image_cache,
        static_root,
        low_confidence_threshold=low_confidence_threshold,
        agreement_sample_rate=agreement_sample_rate,
        agreement_sample_seed=agreement_sample_seed,
    )
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_port}/"
    print(f"VLM Judge UI: {url}")
    print(f"Dataset: {dataset_path}")
    print(f"Annotations: {annotations_path}")
    print(f"Gold transcriptions: {gold_path}")
    if judge_results_path is not None:
        print(f"Judge results: {judge_results_path}")
        print(f"Adjudications: {adjudication_store.path}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vlm-judge-ui")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--annotations", default="artifacts/annotations/annotations.jsonl")
    parser.add_argument("--gold", default="artifacts/annotations/gold_transcriptions.jsonl")
    parser.add_argument("--judge-results")
    parser.add_argument("--adjudications", default="artifacts/annotations/adjudications.jsonl")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--agreement-sample-rate", type=float, default=0.10)
    parser.add_argument("--agreement-sample-seed", default="adjudication-control-v1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args()
    serve_ui(
        Path(args.dataset),
        Path(args.annotations),
        Path(args.gold),
        judge_results_path=Path(args.judge_results) if args.judge_results else None,
        adjudications_path=Path(args.adjudications),
        low_confidence_threshold=args.low_confidence_threshold,
        agreement_sample_rate=args.agreement_sample_rate,
        agreement_sample_seed=args.agreement_sample_seed,
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
