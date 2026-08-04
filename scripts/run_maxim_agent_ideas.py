"""Run Maksim's answer-blind agent ideas on the shared 274-task benchmark.

This experimental runner intentionally uses only the Python standard library so
it can run next to an OpenAI-compatible vLLM endpoint without changing the
project environment.  It never serializes reference/gold fields into prompts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MODEL = "Qwen/Qwen3.5-9B"
SYSTEM_PROMPT = (
    "Sen Türkçe okul sorularını çözen dikkatli bir uzmansın. Görseldeki "
    "soruyu doğrudan incele. Kaynak cevap, cevap anahtarı veya gold bilgi "
    "verilmemiştir. Yalnızca sorunun kendi kanıtlarını kullan. Çoktan seçmeli "
    "soruda final_answer yalnızca seçenek harfi olsun; sayısal veya açık uçlu "
    "soruda gerçek cevabı yaz. Gerekçeyi kısa tut: uzun deneme listeleri üretme; "
    "matematikte önce denklem, bölünebilirlik ve sınır kullan. İstenen JSON "
    "şemasından çıkma ve final_answer alanını gerekçeden önce yaz."
)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_decomposition": {"type": "boolean"},
        "task_type": {"type": "string", "maxLength": 80},
        "critical_evidence": {
            "type": "array",
            "items": {"type": "string", "maxLength": 320},
            "maxItems": 6,
        },
        "subtasks": {
            "type": "array",
            "items": {"type": "string", "maxLength": 320},
            "minItems": 1,
            "maxItems": 6,
        },
    },
    "required": [
        "needs_decomposition",
        "task_type",
        "critical_evidence",
        "subtasks",
    ],
    "additionalProperties": False,
}

SOLVE_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string", "maxLength": 100},
        "reasoning": {"type": "string", "maxLength": 1200},
        "solution_steps": {"type": "string", "maxLength": 1600},
    },
    "required": ["final_answer", "reasoning", "solution_steps"],
    "additionalProperties": False,
}

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string", "maxLength": 100},
        "reasoning": {"type": "string", "maxLength": 900},
    },
    "required": ["final_answer", "reasoning"],
    "additionalProperties": False,
}

SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_index": {"type": "integer", "minimum": 1, "maximum": 8},
        "final_answer": {"type": "string", "maxLength": 100},
        "reasoning": {"type": "string", "maxLength": 1200},
        "solution_steps": {"type": "string", "maxLength": 1600},
    },
    "required": [
        "selected_index",
        "final_answer",
        "reasoning",
        "solution_steps",
    ],
    "additionalProperties": False,
}

PARALLEL_ROUTES = (
    (
        "literal_direct",
        "Soruyu kelimesi kelimesine ve görseldeki tüm seçenekleri okuyarak "
        "doğrudan çöz. Varsayım ekleme.",
    ),
    (
        "formal_structure",
        "Sorunun mantıksal, matematiksel veya dilbilgisel yapısını biçimsel "
        "olarak çıkar; gerekli işlemleri yap ve sonucu seçeneklerle eşleştir.",
    ),
    (
        "option_elimination",
        "Her seçeneği bağımsız olarak sınayıp yanlış olanları ele. Açık uçluysa "
        "en olası alternatif sonuçları karşılaştır.",
    ),
    (
        "visual_evidence",
        "Önce görseldeki kritik metin, sayı, tablo, grafik ve sembolleri yeniden "
        "kontrol et; sonra yalnızca görülen kanıta dayanarak çöz.",
    ),
    (
        "domain_expert",
        "Sorunun ders alanında uzman öğretmen gibi çöz; ilgili kural veya "
        "kavramı belirle ve kısa bir doğrulama yap.",
    ),
    (
        "counterexample_check",
        "İlk akla gelen cevabı çürütmeye çalış. İşaret, istisna, olumsuzluk, "
        "birim ve kapsam hatalarını kontrol ettikten sonra cevap ver.",
    ),
    (
        "independent_rederive",
        "Önceki hiçbir çözüme erişimin yokmuş gibi sıfırdan ikinci bir yöntemle "
        "çöz ve sonucu tekrar hesapla.",
    ),
    (
        "concise_verifier",
        "En kısa güvenilir çözümü üret. Son cevabı vermeden önce soru kökü, "
        "istenen çıktı biçimi ve seçenek harfini son kez doğrula.",
    ),
)


class CallFailure(RuntimeError):
    pass


class EndpointPool:
    def __init__(self, base_urls: list[str], *, model: str, timeout_s: float) -> None:
        if not base_urls:
            raise ValueError("at least one --base-url is required")
        self.base_urls = [value.rstrip("/") for value in base_urls]
        self.model = model
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._index = 0

    def _next(self) -> str:
        with self._lock:
            value = self.base_urls[self._index % len(self.base_urls)]
            self._index += 1
        return value

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int,
        temperature: float,
        seed: int,
        retries: int = 1,
    ) -> dict[str, Any]:
        failures: list[str] = []
        for attempt in range(retries + 1):
            base_url = self._next()
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "top_p": 0.95,
                "max_tokens": max_tokens,
                "seed": seed + attempt * 1009,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
            request = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer EMPTY",
                },
                method="POST",
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    body = response.read().decode("utf-8")
                envelope = json.loads(body)
                choice = envelope["choices"][0]
                content = choice["message"]["content"]
                if isinstance(content, list):
                    content = "\n".join(
                        str(block.get("text") or "")
                        for block in content
                        if isinstance(block, dict)
                    )
                usage = envelope.get("usage") or {}
                recovered_partial = False
                parse_error = None
                try:
                    parsed = _parse_json_object(str(content))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    parsed = _recover_partial_object(str(content), schema_name)
                    if parsed is None:
                        completion_tokens = int(usage.get("completion_tokens") or 0)
                        tail = str(content)[-240:].replace("\n", "\\n")
                        raise ValueError(
                            "invalid structured response "
                            f"finish_reason={choice.get('finish_reason')!r} "
                            f"completion_tokens={completion_tokens} tail={tail!r}"
                        ) from exc
                    recovered_partial = True
                    parse_error = f"{type(exc).__name__}: {exc}"
                return {
                    "parsed": parsed,
                    "raw": str(content),
                    "endpoint": base_url,
                    "finish_reason": choice.get("finish_reason"),
                    "attempt": attempt + 1,
                    "latency_s": round(time.perf_counter() - started, 3),
                    "input_tokens": int(usage.get("prompt_tokens") or 0),
                    "output_tokens": int(usage.get("completion_tokens") or 0),
                    "recovered_partial": recovered_partial,
                    "parse_error": parse_error,
                }
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                failures.append(f"{base_url}:{type(exc).__name__}:{exc}")
                if attempt >= retries:
                    break
                time.sleep(0.25 * (attempt + 1))
        raise CallFailure(" | ".join(failures))


def _parse_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response is not a JSON object")
    return parsed


def _json_field(value: str, key: str) -> Any:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*', value)
    if not match:
        raise ValueError(key)
    parsed, _ = json.JSONDecoder().raw_decode(value[match.end() :])
    return parsed


def _recover_partial_object(value: str, schema_name: str) -> dict[str, Any] | None:
    """Recover only fields fully emitted before a length-truncated JSON string."""
    if schema_name == "decomposition_solution" or schema_name.startswith("candidate_"):
        try:
            final_answer = _json_field(value, "final_answer")
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return {
            "final_answer": str(final_answer),
            "reasoning": "[response truncated after final_answer]",
            "solution_steps": "",
        }
    if schema_name == "parallel_selector":
        try:
            selected_index = int(_json_field(value, "selected_index"))
            final_answer = _json_field(value, "final_answer")
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not 1 <= selected_index <= 8:
            return None
        return {
            "selected_index": selected_index,
            "final_answer": str(final_answer),
            "reasoning": "[response truncated after selected answer]",
            "solution_steps": "",
        }
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _task_view(task: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "task_id",
        "subject",
        "grade",
        "question",
        "question_images",
        "answer_type",
    }
    view = {key: task.get(key) for key in allowed}
    serialized = json.dumps(view, ensure_ascii=False).casefold()
    for forbidden in ("reference_answer", "reference_solution", "gold_answer"):
        if forbidden in serialized:
            raise ValueError(f"forbidden field leaked into task view: {forbidden}")
    return view


def _image_blocks(
    task: dict[str, Any], *, image_root: Path, image_url_root: str
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for image in task.get("question_images") or []:
        if not isinstance(image, dict):
            continue
        data = str(image.get("data") or "")
        name = Path(data).name
        if not name:
            continue
        local_path = (image_root / name).resolve()
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        url = f"{image_url_root.rstrip('/')}/{name}"
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    if not blocks:
        raise ValueError(f"task {task.get('task_id')}: no usable question image")
    return blocks


def _task_prompt(task: dict[str, Any]) -> str:
    question = str(task.get("question") or "").strip()
    if question.casefold() in {"(soru görselde)", "(question in image)"}:
        question = ""
    return (
        f"Ders: {task.get('subject') or 'unknown'}\n"
        f"Sınıf: {task.get('grade') if task.get('grade') is not None else 'unknown'}\n"
        f"Cevap türü: {task.get('answer_type') or 'unknown'}\n"
        f"Ek soru metni: {question or '[yalnızca görsel]'}"
    )


def _messages(
    task: dict[str, Any],
    instruction: str,
    *,
    image_root: Path,
    image_url_root: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"{_task_prompt(task)}\n\n{instruction}",
        }
    ]
    content.extend(
        _image_blocks(task, image_root=image_root, image_url_root=image_url_root)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _usage(calls: list[dict[str, Any]], latency_s: float) -> dict[str, Any]:
    return {
        "input_tokens": sum(int(call.get("input_tokens") or 0) for call in calls),
        "output_tokens": sum(int(call.get("output_tokens") or 0) for call in calls),
        "latency_s": round(latency_s, 3),
    }


def _base_result(
    task: dict[str, Any],
    *,
    condition: str,
    final: dict[str, Any] | None,
    calls: list[dict[str, Any]],
    started: float,
    generation: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    parsed = final or {}
    return {
        "task_id": str(task["task_id"]),
        "condition": condition,
        "model": MODEL,
        "prompt_version": condition,
        "final_answer": str(parsed.get("final_answer") or "") or None,
        "solution_steps": str(parsed.get("solution_steps") or "") or None,
        "reasoning": str(parsed.get("reasoning") or "") or None,
        "forced_answer": False,
        "raw_response": json.dumps(parsed, ensure_ascii=False) if parsed else None,
        "generation": {
            "temperature": 0.0,
            "top_p": 0.95,
            "max_tokens": 1024,
            "structured_mode": "response_format",
            "enable_thinking": False,
            "gold_access": False,
            "call_count": len(calls),
            "retry_calls": sum(max(0, int(call.get("attempt") or 1) - 1) for call in calls),
            **generation,
        },
        "tool_calls": [],
        "usage": _usage(calls, time.perf_counter() - started),
        "error": error,
    }


def run_decompose(
    task: dict[str, Any],
    *,
    pool: EndpointPool,
    image_root: Path,
    image_url_root: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    calls: list[dict[str, Any]] = []
    try:
        plan_call = pool.complete(
            messages=_messages(
                task,
                (
                    "Henüz nihai cevap verme. Soruyu bağımsız ve denetlenebilir "
                    "alt görevlere ayır. Kolay bir soruysa tek alt görev yeterlidir. "
                    "Görselden okunması gereken kritik kanıtları ayrıca listele."
                ),
                image_root=image_root,
                image_url_root=image_url_root,
            ),
            schema_name="decomposition_plan",
            schema=PLAN_SCHEMA,
            max_tokens=768,
            temperature=0.0,
            seed=101,
        )
        calls.append(plan_call)
        plan = plan_call["parsed"]
        solve_instruction = (
            "Aşağıdaki cevap-kör planı uygula. Her alt görevi çöz, sonuçları "
            "birleştir ve cevabı soru köküyle yeniden doğrula. Plan hatalıysa "
            "görsele göre düzelt.\nPLAN:\n"
            + json.dumps(plan, ensure_ascii=False)
        )
        solve_call = pool.complete(
            messages=_messages(
                task,
                solve_instruction,
                image_root=image_root,
                image_url_root=image_url_root,
            ),
            schema_name="decomposition_solution",
            schema=SOLVE_SCHEMA,
            max_tokens=2048,
            temperature=0.0,
            seed=202,
        )
        calls.append(solve_call)
        return _base_result(
            task,
            condition="maxim_decompose_v1",
            final=solve_call["parsed"],
            calls=calls,
            started=started,
            generation={
                "idea": "complex_task_decomposition",
                "max_tokens_per_call": [768, 2048],
                "plan": plan,
                "call_traces": [_compact_call(call) for call in calls],
            },
            error=None,
        )
    except Exception as exc:
        return _base_result(
            task,
            condition="maxim_decompose_v1",
            final=None,
            calls=calls,
            started=started,
            generation={"idea": "complex_task_decomposition"},
            error=f"{type(exc).__name__}: {exc}",
        )


def _parallel_candidate(
    task: dict[str, Any],
    *,
    route_index: int,
    route: tuple[str, str],
    pool: EndpointPool,
    image_root: Path,
    image_url_root: str,
) -> dict[str, Any]:
    name, instruction = route
    call = pool.complete(
        messages=_messages(
            task,
            instruction,
            image_root=image_root,
            image_url_root=image_url_root,
        ),
        schema_name=f"candidate_{route_index}",
        schema=CANDIDATE_SCHEMA,
        max_tokens=1536,
        temperature=0.35,
        seed=1000 + route_index * 97,
    )
    call["route"] = name
    call["route_index"] = route_index
    return call


def run_parallel8(
    task: dict[str, Any],
    *,
    pool: EndpointPool,
    image_root: Path,
    image_url_root: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    calls: list[dict[str, Any]] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    _parallel_candidate,
                    task,
                    route_index=index,
                    route=route,
                    pool=pool,
                    image_root=image_root,
                    image_url_root=image_url_root,
                )
                for index, route in enumerate(PARALLEL_ROUTES, 1)
            ]
            candidates = [future.result() for future in futures]
        candidates.sort(key=lambda value: int(value["route_index"]))
        calls.extend(candidates)
        selector_rows = []
        for candidate in candidates:
            parsed = candidate["parsed"]
            selector_rows.append(
                {
                    "index": candidate["route_index"],
                    "route": candidate["route"],
                    "final_answer": str(parsed.get("final_answer") or "")[:300],
                    "reasoning": str(parsed.get("reasoning") or "")[:1800],
                }
            )
        selector_instruction = (
            "Aşağıda aynı soru için birbirinden bağımsız sekiz aday çözüm var. "
            "Gold veya cevap anahtarı yoktur. Görseli kendin yeniden kontrol et; "
            "çoğunluk oyu kullanmak zorunda değilsin. Kanıt, hesap, olumsuz soru "
            "kökü ve seçenek eşlemesini denetle. En iyi adayı seç veya gerekirse "
            "doğru cevabı kendin düzelt. selected_index 1-8 olmalıdır.\nADAYLAR:\n"
            + json.dumps(selector_rows, ensure_ascii=False)
        )
        selector = pool.complete(
            messages=_messages(
                task,
                selector_instruction,
                image_root=image_root,
                image_url_root=image_url_root,
            ),
            schema_name="parallel_selector",
            schema=SELECT_SCHEMA,
            max_tokens=2048,
            temperature=0.0,
            seed=9090,
        )
        selector["route"] = "independent_selector"
        calls.append(selector)
        return _base_result(
            task,
            condition="maxim_parallel8_judge_v1",
            final=selector["parsed"],
            calls=calls,
            started=started,
            generation={
                "idea": "eight_parallel_reasonings_with_judge",
                "max_tokens_per_call": [1536] * 8 + [2048],
                "candidate_count": 8,
                "selected_index": selector["parsed"].get("selected_index"),
                "candidate_traces": [
                    {
                        "index": candidate["route_index"],
                        "route": candidate["route"],
                        "final_answer": candidate["parsed"].get("final_answer"),
                        "reasoning": candidate["parsed"].get("reasoning"),
                        **_compact_call(candidate),
                    }
                    for candidate in candidates
                ],
                "selector_trace": _compact_call(selector),
            },
            error=None,
        )
    except Exception as exc:
        return _base_result(
            task,
            condition="maxim_parallel8_judge_v1",
            final=None,
            calls=calls,
            started=started,
            generation={
                "idea": "eight_parallel_reasonings_with_judge",
                "candidate_count": len(calls),
            },
            error=f"{type(exc).__name__}: {exc}",
        )


def _compact_call(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": call.get("endpoint"),
        "finish_reason": call.get("finish_reason"),
        "attempt": call.get("attempt"),
        "latency_s": call.get("latency_s"),
        "input_tokens": call.get("input_tokens"),
        "output_tokens": call.get("output_tokens"),
        "recovered_partial": bool(call.get("recovered_partial")),
        "parse_error": call.get("parse_error"),
    }


def _canonicalize_output(
    output: Path, tasks: list[dict[str, Any]], rows: dict[str, dict[str, Any]]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as sink:
        for task in tasks:
            task_id = str(task["task_id"])
            if task_id in rows:
                sink.write(json.dumps(rows[task_id], ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("decompose", "parallel8"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-url-root", default="file:///images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--task-concurrency", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not 1 <= args.task_concurrency <= 64:
        raise SystemExit("--task-concurrency must be in [1, 64]")
    tasks = [_task_view(task) for task in _load_jsonl(args.input)]
    if args.task_id:
        selected = set(args.task_id)
        tasks = [task for task in tasks if task.get("task_id") in selected]
        missing = sorted(selected - {str(task.get("task_id")) for task in tasks})
        if missing:
            raise SystemExit(f"unknown task IDs: {missing}")
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if len({str(task["task_id"]) for task in tasks}) != len(tasks):
        raise SystemExit("duplicate task IDs")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "tasks": len(tasks),
                    "task_ids": [task["task_id"] for task in tasks[:10]],
                    "gold_access": False,
                    "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        for row in _load_jsonl(args.output):
            task_id = str(row.get("task_id") or "")
            if task_id and (not args.retry_errors or not row.get("error")):
                existing[task_id] = row

    pool = EndpointPool(args.base_url, model=args.model, timeout_s=args.timeout_s)
    output_rows = dict(existing)
    write_lock = threading.Lock()
    pending = [task for task in tasks if str(task["task_id"]) not in existing]
    runner = run_decompose if args.mode == "decompose" else run_parallel8

    def execute(task: dict[str, Any]) -> dict[str, Any]:
        return runner(
            task,
            pool=pool,
            image_root=args.image_root,
            image_url_root=args.image_url_root,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.task_concurrency
    ) as executor:
        future_to_task = {executor.submit(execute, task): task for task in pending}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                row = future.result()
            except Exception as exc:  # defensive task-level fail-closed record
                row = _base_result(
                    task,
                    condition=(
                        "maxim_decompose_v1"
                        if args.mode == "decompose"
                        else "maxim_parallel8_judge_v1"
                    ),
                    final=None,
                    calls=[],
                    started=time.perf_counter(),
                    generation={"idea": args.mode},
                    error=f"{type(exc).__name__}: {exc}",
                )
            with write_lock:
                output_rows[str(task["task_id"])] = row
                _canonicalize_output(args.output, tasks, output_rows)
            completed += 1
            print(
                f"[{completed}/{len(pending)}] {task['task_id']} "
                f"answer={row.get('final_answer')!r} error={row.get('error')!r}",
                flush=True,
            )

    _canonicalize_output(args.output, tasks, output_rows)
    errors = sum(bool(row.get("error")) for row in output_rows.values())
    print(
        json.dumps(
            {
                "mode": args.mode,
                "rows": len(output_rows),
                "errors": errors,
                "output": str(args.output),
                "gold_access": False,
            },
            ensure_ascii=False,
        )
    )
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
