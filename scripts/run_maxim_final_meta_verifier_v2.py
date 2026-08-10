#!/usr/bin/env python3
"""Run the separate preregistered 10-candidate meta-verifier v2.

The implementation reuses the immutable v1 execution engine with the isolated
v2 preparation contract.  It does not load the private identity key and keeps
the same independent-verification and content-exact Router fallback policy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - deployment failure
        raise RuntimeError(f"cannot load required module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = Path(__file__).resolve().parent
preparation = _load(
    "maxim_final_meta_prepare_v2_runtime",
    SCRIPT_DIR / "prepare_maxim_final_meta_verifier_v2.py",
)
implementation = _load(
    "maxim_final_meta_run_v2_base",
    SCRIPT_DIR / "run_maxim_final_meta_verifier_v1.py",
)

implementation.preparation = preparation
implementation.SCHEMA_VERSION = "maxim-final-meta-verifier-runner-v2"
implementation.CONDITION = "maxim_final_gold_blind_meta_verifier_v2"
implementation.PROMPT_VERSION = "original-first-anonymous-10way-verification-v2"

SCHEMA_VERSION = implementation.SCHEMA_VERSION
CONDITION = implementation.CONDITION
PROMPT_VERSION = implementation.PROMPT_VERSION
SYSTEM_PROMPT = implementation.SYSTEM_PROMPT
MetaVerifierError = implementation.MetaVerifierError
verdict_schema = implementation.verdict_schema
build_messages = implementation.build_messages
validate_verdict = implementation.validate_verdict
validate_queue = implementation.validate_queue
run_one = implementation.run_one
apply_frozen_policy = implementation.apply_frozen_policy
compose_solver_row = implementation.compose_solver_row
run_queue = implementation.run_queue
build_parser = implementation.build_parser


def main(argv: Sequence[str] | None = None) -> int:
    implementation.__doc__ = __doc__
    return implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
