from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .prompts import build_judge_request
from .schema import EvaluationItem


def request_id(item: EvaluationItem, prompt_version: str = "judge-v2") -> str:
    request = build_judge_request(item)
    canonical = json.dumps(
        {
            "prompt_version": prompt_version,
            "system_prompt": request.system_prompt,
            "user_prompt": request.user_prompt,
            "image_urls": list(request.image_urls),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prepare_request_records(
    input_path: Path,
    output_path: Path,
    *,
    prompt_version: str = "judge-v2",
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = EvaluationItem.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid item on line {line_number}: {exc}") from exc
            request = build_judge_request(item)
            destination.write(
                json.dumps(
                    {
                        "request_id": request_id(item, prompt_version),
                        "prompt_version": prompt_version,
                        "task_id": item.task_id,
                        "setup": item.setup,
                        "subject": item.subject,
                        "system_prompt": request.system_prompt,
                        "user_prompt": request.user_prompt,
                        "image_urls": list(request.image_urls),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count
