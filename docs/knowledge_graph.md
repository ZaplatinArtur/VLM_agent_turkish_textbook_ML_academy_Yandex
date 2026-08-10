# Educational knowledge graph

The graph keeps similarity search and textbook relations separate:

1. dense + BM25 + RRF ranks individual theory, exercise and worked-example
   nodes;
2. the reranker selects task-shaped candidates;
3. `GraphExpansionRanker` expands a similar exercise through typed edges;
4. the agent receives a compact bundle with provenance:
   `THEORY -> WORKED EXAMPLE -> SIMILAR EXERCISE -> SOLUTION`.

Persisted relation types:

- `theory_for`: nearby or same-section theory supporting an exercise/example;
- `worked_example_for`: a preceding worked example in the same local section;
- `solution_of`: an explicit, task-number or nearest-exercise solution link.

`similar_to` is intentionally computed at query time instead of materializing
an all-pairs task graph. This keeps the graph small and lets improved
embedders/rerankers change similarity without rebuilding textbook relations.

## Build

```bash
PYTHONPATH=src python scripts/build_knowledge_graph.py \
  --input-dir artifacts/hybrid_chunks_v3 \
  --output-dir artifacts/knowledge_graph_v1 \
  --report reports/knowledge_graph_v1.json
```

## Use

```bash
export MLA_KNOWLEDGE_GRAPH_DIR=artifacts/knowledge_graph_v1
export MLA_INDEX_DIR=data/knowledge_graph_v1/index
PYTHONPATH=src python -m retrieve.build_index \
  --sample-query "dikdörtgen alan hesaplama"
```

When `MLA_KNOWLEDGE_GRAPH_DIR` is set, the normal `textbook_retrieve` and
`AgentRag` paths use graph-expanded evidence automatically. Every returned hit
contains `graph_anchor_id`, `graph_paths`, relation confidence/method and the
original source chunk IDs.

## Current full-corpus build

- 200 books;
- 151,576 nodes;
- 137,253 searchable nodes;
- 114,594 typed edges;
- 54,000 exercises linked to theory (90.19%);
- 9,729 linked to a worked example (16.25%);
- 1,950 linked to at least one solution (3.26%).
- 1,627 expose a high-confidence solution to the agent (2.72%); inferred
  nearest-exercise links stay in the graph for audit but are not injected.

Solution coverage is currently source-limited: the hybrid parser recognizes
only 3,052 solution units and 143 answer-key units. Raising this coverage
requires parsing answer images/pages and repairing unit labels, not adding more
nearest-neighbor edges.
