from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from .prompts import JudgeRequest


@dataclass(frozen=True, slots=True)
class BackendResponse:
    text: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


class JudgeBackend(Protocol):
    name: str
    model: str

    def complete(self, request: JudgeRequest) -> BackendResponse:
        """Return the raw model response for one judge request."""


class ReplayBackend:
    """Deterministic backend used for tests and replaying captured responses."""

    name = "replay"

    def __init__(self, responses: list[str], *, model: str = "replay-model") -> None:
        self.model = model
        self._responses = iter(responses)
        self.requests: list[JudgeRequest] = []

    def complete(self, request: JudgeRequest) -> BackendResponse:
        self.requests.append(request)
        try:
            response = next(self._responses)
        except StopIteration as exc:
            raise RuntimeError("replay backend has no responses left") from exc
        return BackendResponse(response, self.model)


class OpenAICompatibleBackend:
    """Multimodal chat-completions adapter for OpenRouter or local vLLM."""

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 900,
        seed: int | None = 20260714,
        use_response_format: bool = True,
        enable_thinking: bool | None = None,
        provider: Literal["vllm", "openrouter"] = "vllm",
        image_mode: str = "url",
        image_cache_dir: Path | None = None,
    ) -> None:
        if image_mode not in {"url", "data_url"}:
            raise ValueError("image_mode must be 'url' or 'data_url'")
        if timeout <= 0 or max_tokens < 1:
            raise ValueError("timeout and max_tokens must be positive")
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.use_response_format = use_response_format
        self.enable_thinking = enable_thinking
        self.provider = provider
        self.image_mode = image_mode
        self.image_cache_dir = image_cache_dir or Path("artifacts/cache/judge_images")
        self._image_cache = None
        if image_mode == "data_url":
            from .ui_server import ImageCache

            self._image_cache = ImageCache(self.image_cache_dir)

    def _image_reference(self, url: str) -> str:
        if self.image_mode == "url":
            return url
        if url.startswith("data:"):
            return url
        local_path = Path(url)
        if local_path.is_file():
            content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
            if not content_type.startswith("image/"):
                raise ValueError(f"local attachment is not an image: {local_path}")
            encoded = base64.b64encode(local_path.read_bytes()).decode("ascii")
            return f"data:{content_type};base64,{encoded}"
        if self._image_cache is None:
            raise RuntimeError("image cache is not initialized")
        data, content_type = self._image_cache.get(url)
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{content_type};base64,{encoded}"

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                    parts.append(str(part.get("text") or ""))
            if parts:
                return "".join(parts)
        raise ValueError("chat response does not contain textual message content")

    def complete(self, request: JudgeRequest) -> BackendResponse:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": request.user_prompt}]
        labels = request.image_labels if len(request.image_labels) == len(request.image_urls) else tuple(
            f"attached image {index}" for index in range(1, len(request.image_urls) + 1)
        )
        for label, url in zip(labels, request.image_urls):
            user_content.append({"type": "text", "text": f"{label}:"})
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_reference(url)},
                }
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.use_response_format:
            payload["response_format"] = {"type": "json_object"}
        if self.enable_thinking is not None:
            if self.provider == "openrouter":
                payload["reasoning"] = (
                    {"enabled": True}
                    if self.enable_thinking
                    else {"effort": "none"}
                )
            else:
                payload["chat_template_kwargs"] = {
                    "enable_thinking": self.enable_thinking
                }
        headers = {"Content-Type": "application/json", "User-Agent": "vlm-judge/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                response_payload = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"judge endpoint returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"judge endpoint request failed: {exc.reason}") from exc
        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("invalid chat-completions response shape") from exc
        return BackendResponse(
            self._message_text(content),
            self.model,
            metadata={
                "response_id": response_payload.get("id"),
                "served_model": response_payload.get("model"),
                "created": response_payload.get("created"),
                "usage": response_payload.get("usage"),
                "finish_reason": (response_payload.get("choices") or [{}])[0].get("finish_reason"),
            },
        )
