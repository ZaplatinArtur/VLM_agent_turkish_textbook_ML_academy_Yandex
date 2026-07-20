"""Evaluation utilities for the VLM homework-agent benchmark."""

from .metrics import DeterministicResult, deterministic_match
from .schema import BenchmarkTask, EvaluationItem, JudgeVerdict

__all__ = [
    "DeterministicResult",
    "BenchmarkTask",
    "EvaluationItem",
    "JudgeVerdict",
    "deterministic_match",
]
