"""Проверка готовности визуального ретрива до того, как занята карта.

На этом проекте прогоны уже отрабатывали вхолостую: не задан HF_HOME — модель
уходила искаться в интернет; от упавшей попытки оставался замок сборки,
роняющий следующие; конфигурация бралась из умолчаний в файлах вместо
переменных запуска. Час карты стоит дороже двух секунд этой проверки.

    python -m mla_baseline.visual_preflight

Ненулевой выход — запускать нельзя. Каждая строка говорит, что именно чинить.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

TOOL_POLICY_PROFILES = ("v2_cot_text_rag_v1",)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def check_backend_selected() -> Check:
    backend = _env("MLA_RETRIEVAL_BACKEND") or "text"
    if backend == "visual":
        return Check("бэкенд", True, "MLA_RETRIEVAL_BACKEND=visual")
    return Check(
        "бэкенд",
        False,
        f"MLA_RETRIEVAL_BACKEND={backend!r}: визуальный ретрив не включён, "
        "задайте MLA_RETRIEVAL_BACKEND=visual",
    )


def check_dependency() -> Check:
    from importlib.util import find_spec

    if find_spec("colpali_engine") is None:
        return Check(
            "зависимости",
            False,
            "colpali-engine не установлен: pip install -e '.[visual]'",
        )
    return Check("зависимости", True, "colpali-engine на месте")


def check_threshold() -> Check:
    raw = _env("MLA_VISUAL_MIN_SCORE")
    if not raw:
        return Check(
            "порог",
            False,
            "MLA_VISUAL_MIN_SCORE не задан. Шкала MaxSim своя у каждого индекса, "
            "дефолтный порог пропустил бы любую выдачу. "
            "Снимите: python -m retrieve.evaluation.calibrate_visual",
        )
    try:
        value = float(raw)
    except ValueError:
        return Check("порог", False, f"MLA_VISUAL_MIN_SCORE={raw!r} не число")
    return Check("порог", True, f"MLA_VISUAL_MIN_SCORE={value}")


def check_index(index_dir: str | None = None) -> Check:
    configured = index_dir or _env("MLA_VISUAL_INDEX_DIR")
    if not configured:
        return Check(
            "индекс",
            False,
            "MLA_VISUAL_INDEX_DIR не задан и умолчание не проверено",
        )
    root = Path(configured).expanduser()
    required = ("meta.json", "pages.jsonl", "offsets.npy",
                "tokens.f16.npy", "pooled.f16.npy")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        return Check("индекс", False, f"в {root} не хватает: {', '.join(missing)}")
    return Check("индекс", True, str(root))


def check_adapter(index_dir: str | None = None) -> Check:
    """Индекс помнит, каким адаптером собран: расхождение даёт тихий мусор."""
    configured = index_dir or _env("MLA_VISUAL_INDEX_DIR")
    if not configured:
        return Check("адаптер", False, "проверить нечего: индекс не задан")
    meta_path = Path(configured).expanduser() / "meta.json"
    if not meta_path.is_file():
        return Check("адаптер", False, f"нет {meta_path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return Check("адаптер", False, f"meta.json не читается: {exc}")
    adapter = str(meta.get("adapter") or "")
    if not adapter:
        return Check("адаптер", False, "в meta.json не записан adapter")
    if not Path(adapter).expanduser().is_dir():
        return Check(
            "адаптер",
            False,
            f"индекс собран адаптером {adapter}, которого нет на диске",
        )
    return Check("адаптер", True, adapter)


def check_prompt_profile() -> Check:
    """Без политики для текстовых задач агент тулу просто не вызовет."""
    profile = _env("MLA_PROMPT_VERSION")
    if profile in TOOL_POLICY_PROFILES:
        return Check("промпт", True, f"MLA_PROMPT_VERSION={profile}")
    return Check(
        "промпт",
        False,
        f"MLA_PROMPT_VERSION={profile or 'v2_cot (дефолт)'}: политика тулы "
        "написана вокруг визуальных задач и на текстовых не срабатывает. "
        f"Задайте один из {', '.join(TOOL_POLICY_PROFILES)}",
        fatal=False,
    )


def check_live_query(probe: str = "üçgen alanı nasıl bulunur") -> Check:
    """Единственная проверка, которая реально дёргает ретрив."""
    try:
        from .tools.visual_search import visual_client_from_env

        client = visual_client_from_env()
        result = client.search(probe, top_k=3)
    except Exception as exc:
        return Check("пробный запрос", False, f"{type(exc).__name__}: {exc}")
    label = (result.get("relevance") or {}).get("label")
    returned = result.get("retrieved", 0)
    return Check(
        "пробный запрос",
        returned > 0,
        f"найдено {returned}, вердикт {label}, {result.get('latency_ms')} мс"
        if returned
        else "ретрив ответил, но выдача пуста — проверьте индекс и фильтры",
    )


CHECKS: tuple[tuple[str, Callable[[], Check]], ...] = (
    ("backend", check_backend_selected),
    ("dependency", check_dependency),
    ("threshold", check_threshold),
    ("index", check_index),
    ("adapter", check_adapter),
    ("prompt", check_prompt_profile),
)


def run_checks(*, live: bool = True) -> list[Check]:
    results = [factory() for _, factory in CHECKS]
    # Пробный запрос имеет смысл только если конфигурация уже сошлась.
    if live and all(check.ok for check in results if check.fatal):
        results.append(check_live_query())
    return results


def report(checks: list[Check]) -> int:
    for check in checks:
        if check.ok:
            mark = "OK  "
        else:
            mark = "FAIL" if check.fatal else "WARN"
        print(f"[{mark}] {check.name:16} {check.detail}")
    broken = [check for check in checks if not check.ok and check.fatal]
    if broken:
        print(f"\nзапускать нельзя: не прошло {len(broken)}", file=sys.stderr)
        return 1
    print("\nготово к запуску")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-live", action="store_true",
                       help="не дёргать ретрив, проверить только конфигурацию")
    args = parser.parse_args()
    return report(run_checks(live=not args.no_live))


if __name__ == "__main__":
    raise SystemExit(main())
