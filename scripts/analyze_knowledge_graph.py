from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieve.graph import KnowledgeGraph


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze educational graph coverage.")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph = KnowledgeGraph.load(args.graph_dir)
    exercise_ids = {
        node.chunk_id
        for node in graph.nodes.values()
        if str(node.metadata.get("unit_kind")) == "exercise"
    }
    anchors_by_relation: dict[str, set[str]] = defaultdict(set)
    safe_solution_anchors: set[str] = set()
    methods: Counter[str] = Counter()
    confidence_by_relation: dict[str, list[float]] = defaultdict(list)
    for edge in graph.edges.values():
        relation = edge.relation.value
        anchors_by_relation[relation].add(edge.source_id)
        methods[f"{relation}:{edge.method}"] += 1
        confidence_by_relation[relation].append(edge.confidence)
        if relation == "solution_of" and edge.confidence >= 0.8:
            safe_solution_anchors.add(edge.source_id)

    report = {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "exercises": len(exercise_ids),
        "relation_counts": graph.relation_counts(),
        "linked_exercise_anchors": {
            relation: len(anchors & exercise_ids)
            for relation, anchors in anchors_by_relation.items()
        },
        "exercise_coverage": {
            relation: round(len(anchors & exercise_ids) / len(exercise_ids), 4)
            if exercise_ids
            else 0.0
            for relation, anchors in anchors_by_relation.items()
        },
        "agent_safe_solutions": {
            "linked_exercises": len(safe_solution_anchors & exercise_ids),
            "coverage": round(
                len(safe_solution_anchors & exercise_ids) / len(exercise_ids),
                4,
            )
            if exercise_ids
            else 0.0,
            "minimum_confidence": 0.8,
        },
        "edge_methods": dict(methods.most_common()),
        "confidence": {
            relation: {
                "mean": round(statistics.fmean(values), 4),
                "median": round(statistics.median(values), 4),
            }
            for relation, values in confidence_by_relation.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
