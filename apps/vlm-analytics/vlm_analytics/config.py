from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


APP_NAME = "VLM Analytics"
DATASET_VERSION = "validation_v2_274_hybrid"


def application_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def default_database_path() -> Path:
    return application_dir() / "vlm_analytics.db"


@dataclass(frozen=True)
class RemoteBundle:
    key: str
    display_name: str
    raw_path: str
    judge_path: str


DEFAULT_SERVER = os.environ.get("VLM_ANALYTICS_SERVER", "")
DEFAULT_USER = os.environ.get("VLM_ANALYTICS_USER", "")
DEFAULT_KEY_PATH = str(Path.home() / ".ssh" / "id_rsa")
V2_274_WORK = os.environ.get(
    "VLM_ANALYTICS_REMOTE_ROOT", "/path/to/v2_274/app"
).rstrip("/")
REMOTE_MANIFEST = f"{V2_274_WORK}/validation_manifest.jsonl"

REMOTE_BUNDLES = (
    RemoteBundle(
        "b0_no_tools",
        "Без тулов",
        f"{V2_274_WORK}/b0_no_tools_raw.jsonl",
        f"{V2_274_WORK}/b0_no_tools_judge.jsonl",
    ),
    RemoteBundle(
        "web_search",
        "Веб",
        f"{V2_274_WORK}/web_search_raw.jsonl",
        f"{V2_274_WORK}/web_search_judge.jsonl",
    ),
    RemoteBundle(
        "agent_rag",
        "RAG",
        f"{V2_274_WORK}/agent_rag_raw.jsonl",
        f"{V2_274_WORK}/agent_rag_judge.jsonl",
    ),
    RemoteBundle(
        "agent_rag_thinking",
        "RAG thinking",
        f"{V2_274_WORK}/agent_rag_thinking_raw.jsonl",
        f"{V2_274_WORK}/agent_rag_thinking_judge.jsonl",
    ),
)


MODE_ORDER = {
    "b0_no_tools": 0,
    "web_search": 1,
    "agent_rag": 2,
    "agent_rag_routed": 3,
    "agent_rag_thinking": 4,
    "agent_rag_hybrid_chunks": 5,
    "agent_rag_hybrid_chunks_thinking": 6,
}

MODE_COLORS = {
    "b0_no_tools": "#8797a5",
    "web_search": "#3ddc97",
    "agent_rag": "#ffb454",
    "agent_rag_routed": "#22c55e",
    "agent_rag_thinking": "#a78bfa",
    "agent_rag_hybrid_chunks": "#38bdf8",
    "agent_rag_hybrid_chunks_thinking": "#f472b6",
}
