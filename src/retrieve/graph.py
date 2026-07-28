from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from schemas.retrieve import RetrievedChunk


class RelationType(str, Enum):
    THEORY_FOR = "theory_for"
    WORKED_EXAMPLE_FOR = "worked_example_for"
    SOLUTION_OF = "solution_of"


class KnowledgeEdge(BaseModel):
    edge_id: str
    source_id: str
    target_id: str
    relation: RelationType
    confidence: float = Field(ge=0.0, le=1.0)
    method: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def _clean(value: Any) -> str:
    return "\n".join(
        line.strip()
        for line in str(value or "").replace("\r", "\n").splitlines()
        if line.strip()
    )


def _kind(chunk: RetrievedChunk) -> str:
    return str(chunk.metadata.get("unit_kind") or "other")


def _book(chunk: RetrievedChunk) -> str:
    return str(
        chunk.metadata.get("textbook")
        or chunk.metadata.get("book_id")
        or chunk.chunk_id.split(":", 1)[0]
    )


def _page(chunk: RetrievedChunk) -> int:
    value = chunk.metadata.get("source_page", chunk.metadata.get("page"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _unit_index(chunk: RetrievedChunk) -> int:
    try:
        return int(chunk.metadata.get("unit_index", 0))
    except (TypeError, ValueError):
        return 0


def _edge_id(
    source_id: str,
    target_id: str,
    relation: RelationType,
    method: str,
) -> str:
    digest = hashlib.sha256(
        f"{source_id}\0{target_id}\0{relation.value}\0{method}".encode("utf-8")
    ).hexdigest()
    return f"edge_{digest[:24]}"


def _card_id(anchor_id: str, edge_ids: Iterable[str]) -> str:
    digest = hashlib.sha256(anchor_id.encode("utf-8"))
    for edge_id in sorted(edge_ids):
        digest.update(b"\0")
        digest.update(edge_id.encode("utf-8"))
    return f"graph_{digest.hexdigest()[:24]}"


def _useful_theory(chunk: RetrievedChunk) -> bool:
    text = _clean(chunk.text)
    if len(text) < 70:
        return False
    plain = text.casefold()
    boilerplate = (
        "isbn",
        "her hakkı saklıdır",
        "her hakki saklidir",
        "yayın basım dağıtım",
        "yayin basim dagitim",
        "ders kitabı olarak kabul edilmiştir",
        "ders kitabi olarak kabul edilmistir",
    )
    return not any(marker in plain for marker in boilerplate)


class KnowledgeGraph:
    def __init__(
        self,
        nodes: Iterable[RetrievedChunk] = (),
        edges: Iterable[KnowledgeEdge] = (),
    ) -> None:
        self.nodes: dict[str, RetrievedChunk] = {
            node.chunk_id: node for node in nodes
        }
        self.edges: dict[str, KnowledgeEdge] = {}
        self._outgoing: dict[str, list[KnowledgeEdge]] = defaultdict(list)
        for edge in edges:
            self.add_edge(edge)

    def add_node(self, node: RetrievedChunk) -> None:
        self.nodes[node.chunk_id] = node

    def add_edge(self, edge: KnowledgeEdge) -> None:
        if edge.edge_id in self.edges:
            return
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(
                f"edge references unknown node: {edge.source_id} -> {edge.target_id}"
            )
        self.edges[edge.edge_id] = edge
        self._outgoing[edge.source_id].append(edge)

    def outgoing(
        self,
        node_id: str,
        relation: RelationType | None = None,
    ) -> list[KnowledgeEdge]:
        edges = self._outgoing.get(node_id, [])
        if relation is not None:
            edges = [edge for edge in edges if edge.relation is relation]
        return sorted(
            edges,
            key=lambda edge: (-edge.confidence, edge.target_id, edge.edge_id),
        )

    def searchable_nodes(self) -> list[RetrievedChunk]:
        searchable = {"exercise", "theory", "worked_example"}
        output: list[RetrievedChunk] = []
        for node in self.nodes.values():
            if _kind(node) not in searchable or not _clean(node.text):
                continue
            relations = {edge.relation for edge in self._outgoing.get(node.chunk_id, [])}
            metadata = dict(node.metadata)
            metadata.update(
                {
                    "knowledge_graph_node": True,
                    "retrieval_text": _clean(
                        node.metadata.get("retrieval_text") or node.text
                    ),
                    "has_theory": RelationType.THEORY_FOR in relations,
                    "has_example": RelationType.WORKED_EXAMPLE_FOR in relations,
                    "has_solution": RelationType.SOLUTION_OF in relations,
                }
            )
            output.append(node.model_copy(update={"metadata": metadata}))
        return output

    def relation_counts(self) -> dict[str, int]:
        return dict(Counter(edge.relation.value for edge in self.edges.values()))

    def bundle(
        self,
        anchor_id: str,
        *,
        score: float | None = None,
        theory_chars: int = 1_100,
        example_chars: int = 900,
        solution_chars: int = 1_300,
        max_chars: int = 3_800,
        min_solution_confidence: float = 0.8,
        include_solutions: bool = True,
        anchor_first: bool = True,
    ) -> RetrievedChunk:
        anchor = self.nodes[anchor_id]
        anchor_kind = _kind(anchor)
        expansions: list[tuple[str, RetrievedChunk, KnowledgeEdge | None]] = []

        if anchor_kind in {"exercise", "worked_example"}:
            for edge in self.outgoing(anchor_id, RelationType.THEORY_FOR)[:2]:
                expansions.append(("THEORY", self.nodes[edge.target_id], edge))
        if anchor_kind == "exercise":
            for edge in self.outgoing(
                anchor_id, RelationType.WORKED_EXAMPLE_FOR
            )[:1]:
                expansions.append(
                    ("WORKED EXAMPLE", self.nodes[edge.target_id], edge)
                )

        anchor_label = {
            "exercise": "SIMILAR EXERCISE",
            "worked_example": "WORKED EXAMPLE",
            "theory": "THEORY",
        }.get(anchor_kind, anchor_kind.replace("_", " ").upper())
        anchor_entry = (anchor_label, anchor, None)
        selected = (
            [anchor_entry, *expansions]
            if anchor_first
            else [*expansions, anchor_entry]
        )

        if include_solutions and anchor_kind in {"exercise", "worked_example"}:
            for edge in self.outgoing(anchor_id, RelationType.SOLUTION_OF)[:2]:
                if edge.confidence < min_solution_confidence:
                    continue
                selected.append(("SOLUTION", self.nodes[edge.target_id], edge))

        limits = {
            "THEORY": theory_chars,
            "WORKED EXAMPLE": example_chars,
            "SOLUTION": solution_chars,
            "SIMILAR EXERCISE": 1_800,
        }
        parts: list[str] = []
        graph_paths: list[dict[str, Any]] = []
        source_ids: list[str] = []
        images = []
        seen_images: set[str] = set()
        remaining = max_chars
        used_labels: set[str] = set()
        used_chars_by_label: dict[str, int] = defaultdict(int)

        for label, node, edge in selected:
            if remaining <= 0:
                break
            text = _clean(node.text)
            if not text:
                continue
            label_budget = limits.get(label, 1_800)
            label_remaining = max(
                0,
                label_budget - used_chars_by_label[label],
            )
            clipped = text[: min(label_remaining, remaining)]
            if not clipped:
                continue
            parts.append(f"[{label}]\n{clipped}")
            remaining -= len(clipped)
            used_chars_by_label[label] += len(clipped)
            used_labels.add(label)
            source_ids.append(node.chunk_id)
            if edge is not None:
                graph_paths.append(
                    {
                        "edge_id": edge.edge_id,
                        "relation": edge.relation.value,
                        "target_id": edge.target_id,
                        "confidence": edge.confidence,
                        "method": edge.method,
                    }
                )
            for image in node.images:
                key = image.model_dump_json()
                if key not in seen_images and len(images) < 3:
                    seen_images.add(key)
                    images.append(image)

        metadata = dict(anchor.metadata)
        metadata.update(
            {
                "knowledge_graph": True,
                "graph_anchor_id": anchor_id,
                "graph_anchor_kind": anchor_kind,
                "graph_paths": graph_paths,
                "source_chunk_ids": source_ids,
                "retrieval_text": _clean(
                    anchor.metadata.get("retrieval_text") or anchor.text
                ),
                "knowledge_kind": anchor_kind,
                "unit_kind": "knowledge_card",
                "has_theory": "THEORY" in used_labels
                and anchor_kind != "theory",
                "has_example": "WORKED EXAMPLE" in used_labels
                and anchor_kind != "worked_example",
                "has_solution": "SOLUTION" in used_labels,
                "card_chars": sum(len(part) for part in parts),
                "bundle_anchor_first": anchor_first,
                "bundle_includes_solutions": include_solutions,
            }
        )
        edge_ids = [path["edge_id"] for path in graph_paths]
        return RetrievedChunk(
            chunk_id=_card_id(anchor_id, edge_ids),
            text="\n\n".join(parts),
            images=images,
            score=float(anchor.score if score is None else score),
            metadata=metadata,
        )

    def save(self, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        node_path = directory / "nodes.jsonl"
        edge_path = directory / "edges.jsonl"
        manifest_path = directory / "manifest.json"
        node_tmp = node_path.with_suffix(".jsonl.tmp")
        edge_tmp = edge_path.with_suffix(".jsonl.tmp")

        with node_tmp.open("w", encoding="utf-8", newline="\n") as output:
            for node in sorted(self.nodes.values(), key=lambda item: item.chunk_id):
                output.write(node.model_dump_json() + "\n")
        with edge_tmp.open("w", encoding="utf-8", newline="\n") as output:
            for edge in sorted(self.edges.values(), key=lambda item: item.edge_id):
                output.write(edge.model_dump_json() + "\n")
        node_tmp.replace(node_path)
        edge_tmp.replace(edge_path)

        manifest = {
            "schema_version": 1,
            "nodes": len(self.nodes),
            "searchable_nodes": len(self.searchable_nodes()),
            "edges": len(self.edges),
            "node_kinds": dict(
                Counter(_kind(node) for node in self.nodes.values()).most_common()
            ),
            "relation_counts": self.relation_counts(),
            "files": {
                "nodes": node_path.name,
                "edges": edge_path.name,
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest

    @classmethod
    def load(cls, directory: Path) -> "KnowledgeGraph":
        node_path = directory / "nodes.jsonl"
        edge_path = directory / "edges.jsonl"
        with node_path.open(encoding="utf-8") as source:
            nodes = [
                RetrievedChunk.model_validate_json(line)
                for line in source
                if line.strip()
            ]
        graph = cls(nodes=nodes)
        with edge_path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    graph.add_edge(KnowledgeEdge.model_validate_json(line))
        return graph


class KnowledgeGraphBuilder:
    def __init__(
        self,
        *,
        context_page_radius: int = 2,
        max_theory_edges: int = 2,
        max_example_edges: int = 1,
    ) -> None:
        self.context_page_radius = context_page_radius
        self.max_theory_edges = max_theory_edges
        self.max_example_edges = max_example_edges

    @staticmethod
    def _edge(
        source: RetrievedChunk,
        target: RetrievedChunk,
        relation: RelationType,
        confidence: float,
        method: str,
    ) -> KnowledgeEdge:
        return KnowledgeEdge(
            edge_id=_edge_id(
                source.chunk_id,
                target.chunk_id,
                relation,
                method,
            ),
            source_id=source.chunk_id,
            target_id=target.chunk_id,
            relation=relation,
            confidence=confidence,
            method=method,
            metadata={
                "source_page": _page(source),
                "target_page": _page(target),
            },
        )

    def _previous(
        self,
        ordered: list[RetrievedChunk],
        position: int,
        *,
        kinds: set[str],
        limit: int,
    ) -> list[RetrievedChunk]:
        source = ordered[position]
        source_page = _page(source)
        section = _clean(source.metadata.get("section_title"))
        candidates: list[tuple[int, int, RetrievedChunk]] = []
        for candidate in ordered[:position]:
            if _kind(candidate) not in kinds:
                continue
            if _kind(candidate) == "theory" and not _useful_theory(candidate):
                continue
            candidate_page = _page(candidate)
            distance = (
                source_page - candidate_page
                if source_page >= 0 and candidate_page >= 0
                else 0
            )
            if distance < 0 or distance > self.context_page_radius:
                continue
            candidate_section = _clean(candidate.metadata.get("section_title"))
            same_section = int(bool(section and section == candidate_section))
            candidates.append((same_section, -distance, candidate))
        candidates.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                -_unit_index(item[2]),
                item[2].chunk_id,
            )
        )
        return [candidate for _, _, candidate in candidates[:limit]]

    def build(self, chunks: Iterable[RetrievedChunk]) -> KnowledgeGraph:
        nodes = [chunk for chunk in chunks if _clean(chunk.text)]
        graph = KnowledgeGraph(nodes=nodes)
        by_book: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for node in nodes:
            by_book[_book(node)].append(node)

        for book_nodes in by_book.values():
            ordered = sorted(
                book_nodes,
                key=lambda node: (_page(node), _unit_index(node), node.chunk_id),
            )
            positions = {node.chunk_id: index for index, node in enumerate(ordered)}
            exercises = [
                node for node in ordered if _kind(node) == "exercise"
            ]
            exercises_by_number: dict[str, list[RetrievedChunk]] = defaultdict(list)
            for exercise in exercises:
                task_number = _clean(exercise.metadata.get("task_number"))
                if task_number:
                    exercises_by_number[task_number].append(exercise)

            for solution in ordered:
                if _kind(solution) not in {"solution", "answer_key"}:
                    continue
                anchor: RetrievedChunk | None = None
                method = ""
                confidence = 0.0
                explicit = _clean(solution.metadata.get("exercise_id"))
                if explicit and explicit in graph.nodes:
                    candidate = graph.nodes[explicit]
                    if _kind(candidate) in {"exercise", "worked_example"}:
                        anchor = candidate
                        method = "explicit_exercise_id"
                        confidence = 1.0

                task_number = _clean(solution.metadata.get("task_number"))
                if anchor is None and task_number:
                    candidates = exercises_by_number.get(task_number, [])
                    preceding = [
                        candidate
                        for candidate in candidates
                        if 0 <= _page(solution) - _page(candidate) <= 3
                    ]
                    if preceding:
                        anchor = max(
                            preceding,
                            key=lambda candidate: (
                                _page(candidate),
                                _unit_index(candidate),
                            ),
                        )
                        method = "task_number"
                        confidence = 0.9

                if anchor is None:
                    solution_position = positions[solution.chunk_id]
                    preceding = [
                        candidate
                        for candidate in exercises
                        if positions[candidate.chunk_id] < solution_position
                        and (
                            _page(solution) < 0
                            or _page(candidate) < 0
                            or 0 <= _page(solution) - _page(candidate) <= 1
                        )
                    ]
                    if preceding:
                        anchor = max(
                            preceding,
                            key=lambda candidate: positions[candidate.chunk_id],
                        )
                        method = "nearest_exercise"
                        confidence = 0.58

                if anchor is not None:
                    graph.add_edge(
                        self._edge(
                            anchor,
                            solution,
                            RelationType.SOLUTION_OF,
                            confidence,
                            method,
                        )
                    )

            for position, source in enumerate(ordered):
                source_kind = _kind(source)
                if source_kind not in {"exercise", "worked_example"}:
                    continue
                theories = self._previous(
                    ordered,
                    position,
                    kinds={"theory"},
                    limit=self.max_theory_edges,
                )
                for theory in theories:
                    same_section = bool(
                        _clean(source.metadata.get("section_title"))
                        and _clean(source.metadata.get("section_title"))
                        == _clean(theory.metadata.get("section_title"))
                    )
                    graph.add_edge(
                        self._edge(
                            source,
                            theory,
                            RelationType.THEORY_FOR,
                            0.86 if same_section else 0.68,
                            "same_section" if same_section else "nearby_theory",
                        )
                    )
                if source_kind == "exercise":
                    examples = self._previous(
                        ordered,
                        position,
                        kinds={"worked_example"},
                        limit=self.max_example_edges,
                    )
                    for example in examples:
                        graph.add_edge(
                            self._edge(
                                source,
                                example,
                                RelationType.WORKED_EXAMPLE_FOR,
                                0.72,
                                "nearby_worked_example",
                            )
                        )
        return graph
