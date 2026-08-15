from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder


class App:
    def __init__(self, model_name: str, data_root: Path, batch_size: int):
        self.root = data_root.resolve()
        self.batch_size = batch_size
        self.model = CrossEncoder(
            model_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "sdpa"},
        )

    def resolve_image(self, relative: str) -> str:
        path = (self.root / relative).resolve()
        if self.root not in path.parents or not path.is_file():
            raise ValueError("candidate image is missing or outside data root")
        return str(path)

    def rerank(self, payload: dict) -> dict:
        query = " ".join(str(payload.get("query") or "").split())
        if len(query) < 3:
            raise ValueError("query is too short")
        candidates = payload.get("candidates") or []
        if not 1 <= len(candidates) <= 50:
            raise ValueError("candidates must contain 1..50 items")
        top_k = max(1, min(int(payload.get("top_k") or 5), len(candidates)))
        docs = [self.resolve_image(str(row["page_image"])) for row in candidates]
        prompt = "Retrieve Turkish textbook page images relevant to the student's query."
        started = time.perf_counter()
        scores = self.model.predict(
            [(query, image) for image in docs],
            prompt=prompt,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        rows = [
            {"page_id": str(candidate["page_id"]), "score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        rows.sort(key=lambda row: (-row["score"], row["page_id"]))
        return {"results": rows[:top_k], "latency_ms": round((time.perf_counter()-started)*1000, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-Reranker-2B")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8782)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args(); app = App(args.model, args.data_root, args.batch_size)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: dict):
            body=json.dumps(payload,ensure_ascii=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            self.send_json(200,{"status":"ok"}) if self.path=="/health" else self.send_json(404,{"error":"not found"})
        def do_POST(self):
            if self.path!="/rerank": return self.send_json(404,{"error":"not found"})
            try:
                length=int(self.headers.get("Content-Length") or 0)
                self.send_json(200,app.rerank(json.loads(self.rfile.read(length))))
            except Exception as exc: self.send_json(400,{"error":f"{type(exc).__name__}: {exc}"})
        def log_message(self, *_): pass
    ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()


if __name__=="__main__": main()
