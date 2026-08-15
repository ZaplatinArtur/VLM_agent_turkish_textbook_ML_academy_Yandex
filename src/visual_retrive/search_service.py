"""HTTP-сервис визуального поиска: индекс один на карту, воркеров много.

Транспорт inprocess держит свою копию ColQwen в каждом воркере — это нормально
при раскладке «воркер на карту». Когда воркеров больше, поднимается этот
сервис, и модель с индексом остаются в одном экземпляре.

Картинки страниц отдаются base64, а не путями: агент может стоять на другой
машине, а корпус у него может быть меньше индексного.

    python -m visual_retrive.search_service --port 8780
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .paths import PROJECT_ROOT, VISUAL_RETRIVE_DIR

DEFAULT_PORT = 8780
DEFAULT_IMAGE_MAX_SIDE = 1024
DEFAULT_IMAGE_QUALITY = 80


def data_root() -> Path:
    """Тот же корень, от которого пути страниц пишет индексатор."""
    return VISUAL_RETRIVE_DIR if (VISUAL_RETRIVE_DIR / "books").is_dir() else PROJECT_ROOT


class VisualSearchApp:
    """Логика без HTTP — чтобы проверять её без сервера и без карты."""

    def __init__(
        self,
        *,
        index_dir: str | None = None,
        root: Path | None = None,
        image_max_side: int = DEFAULT_IMAGE_MAX_SIDE,
        image_quality: int = DEFAULT_IMAGE_QUALITY,
        include_images: bool = True,
        index: Any | None = None,
    ) -> None:
        if image_max_side < 1:
            raise ValueError("image_max_side must be positive")
        self.index_dir = index_dir
        self.root = (root or data_root()).resolve()
        self.image_max_side = image_max_side
        self.image_quality = image_quality
        self.include_images = include_images
        self._index = index
        # Модель и индекс общие на процесс, а torch к параллельному входу не
        # готов: сериализуем так же, как кросс-энкодер в текстовом ретриве.
        self._search_lock = threading.Lock()

    @property
    def index(self) -> Any:
        if self._index is None:
            from .service import get_visual_index

            self._index = get_visual_index(self.index_dir, load_model=True)
        return self._index

    def _encode_image(self, relative: str) -> str | None:
        if not self.include_images or not relative:
            return None
        path = (self.root / relative).resolve()
        # Путь приходит из индекса, но выходить за корень ему всё равно нельзя.
        if self.root not in path.parents or not path.is_file():
            return None
        try:
            from PIL import Image

            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((self.image_max_side, self.image_max_side))
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=self.image_quality)
        except Exception:
            # Битая страница не должна ронять весь запрос: агент обойдётся
            # текстом, а флаг has_page_image покажет, что картинки нет.
            return None
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = " ".join(str(payload.get("query") or "").split())
        if len(query) < 2:
            raise ValueError("query is too short")
        top_k = int(payload.get("top_k") or 5)
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        subject = payload.get("subject") or None
        grade = payload.get("grade")
        grade = None if grade in ("", None) else grade

        started = time.perf_counter()
        with self._search_lock:
            hits = self.index.search(query, top_k=top_k, subject=subject, grade=grade)
        rows = []
        for rank, hit in enumerate(hits, 1):
            row = dict(hit)
            row["rank"] = rank
            encoded = self._encode_image(str(row.get("page_image") or ""))
            if encoded:
                row["page_image_b64"] = encoded
            rows.append(row)
        return {
            "query": query,
            "filters": {"subject": subject, "grade": grade},
            "latency_ms": round((time.perf_counter() - started) * 1_000, 3),
            "returned": len(rows),
            "hits": rows,
        }

    def health(self) -> dict[str, Any]:
        """Готовность проверяется загрузкой индекса, а не ответом «ok»."""
        index = self.index
        return {
            "status": "ok",
            "pages": len(getattr(index, "pages", []) or []),
            "index_dir": self.index_dir or "",
            "images": self.include_images,
        }


def _handler(app: VisualSearchApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                return self._send(404, {"error": "not found"})
            try:
                self._send(200, app.health())
            except Exception as exc:
                self._send(503, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:
            if self.path != "/api/search":
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                request = json.loads(self.rfile.read(length))
            except Exception as exc:
                return self._send(400, {"error": f"bad request: {exc}"})
            try:
                self._send(200, app.search(request))
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *_: Any) -> None:
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--image-max-side", type=int, default=DEFAULT_IMAGE_MAX_SIDE)
    parser.add_argument("--image-quality", type=int, default=DEFAULT_IMAGE_QUALITY)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    app = VisualSearchApp(
        index_dir=args.index_dir,
        image_max_side=args.image_max_side,
        image_quality=args.image_quality,
        include_images=not args.no_images,
    )
    # Грузим до первого запроса: пусть падает на старте, а не посреди прогона.
    report = app.health()
    print(f"visual index loaded: {report['pages']} pages", flush=True)
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), _handler(app)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
