from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset, Sampler


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


class DSU:
    def __init__(self): self.parent = {}
    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[b] = a


def reviewed_page_groups(path: Path) -> list[list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("groups", payload) if isinstance(payload, dict) else payload
    if isinstance(groups, dict):
        groups = list(groups.values())
    out = []
    for group in groups:
        if isinstance(group, dict):
            group = group.get("page_ids") or group.get("pages") or group.get("positive_page_ids") or []
        pages = [str(x) for x in group]
        if pages: out.append(pages)
    return out


@dataclass
class Record:
    page_id: str
    group_id: str
    subject: str
    grade: int | None
    image: Path
    queries: list[str]


def build_records(pairs: Path, groups: Path, data_root: Path) -> list[Record]:
    rows = read_jsonl(pairs)
    dsu = DSU()
    for pages in reviewed_page_groups(groups):
        for page in pages[1:]: dsu.union(pages[0], page)

    # Exact duplicate queries are also known positives, even if a reviewed
    # relevance file omitted the edge.
    query_pages = defaultdict(set)
    for row in rows:
        query = normalize_query(str(row.get("query") or ""))
        page = str(row.get("positive_page_id") or "")
        subject = str(row.get("subject") or "unknown")
        if query and page: query_pages[(subject, query)].add(page)
    for pages in query_pages.values():
        pages = sorted(pages)
        for page in pages[1:]: dsu.union(pages[0], page)

    pages: dict[str, dict] = {}
    for row in rows:
        query = str(row.get("query") or "").strip()
        page_id = str(row.get("positive_page_id") or "")
        image = data_root / str(row.get("positive_image") or "")
        if len(query) < 3 or not page_id or not image.is_file(): continue
        rec = pages.setdefault(page_id, {
            "page_id": page_id, "subject": str(row.get("subject") or "unknown"),
            "grade": row.get("grade"), "image": image, "queries": [],
        })
        if query not in rec["queries"]: rec["queries"].append(query)

    records = []
    for page_id, row in pages.items():
        root = dsu.find(page_id)
        group_hash = hashlib.sha1(f"{row['subject']}:{root}".encode()).hexdigest()[:16]
        records.append(Record(group_id=f"rel:{group_hash}", **row))
    return records


def split_by_group(records: list[Record], pages_per_subject: int, seed: int):
    """Exactly N validation pages per subject; whole relevance groups stay held out."""
    by_subject = defaultdict(lambda: defaultdict(list))
    for record in records: by_subject[record.subject][record.group_id].append(record)
    train, validation, stats = [], [], {}
    for subject, grouped in sorted(by_subject.items()):
        group_ids = sorted(grouped)
        random.Random(f"{seed}:{subject}").shuffle(group_ids)
        chosen, page_count = [], 0
        for group_id in group_ids:
            if page_count >= pages_per_subject: break
            chosen.append(group_id); page_count += len(grouped[group_id])
        held_out = set(chosen)
        candidates = [r for gid in chosen for r in grouped[gid]]
        subject_val = candidates[:pages_per_subject]
        validation.extend(subject_val)
        train.extend(r for gid, values in grouped.items() if gid not in held_out for r in values)
        stats[subject] = {
            "total_pages": sum(map(len, grouped.values())),
            "train_pages": sum(len(v) for k, v in grouped.items() if k not in held_out),
            "validation_pages": len(subject_val),
            "held_out_group_pages": page_count,
            "excluded_boundary_pages": max(0, page_count-len(subject_val)),
        }
    return train, validation, stats


class PageDataset(Dataset):
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, index): return self.records[index]


class SubjectGlobalBatchSampler(Sampler[list[int]]):
    """Global same-subject batches, packed by positive group before DDP splitting."""
    def __init__(self, records, global_batch_size: int, seed=17, drop_last=True):
        self.records, self.batch_size = records, global_batch_size
        self.seed, self.drop_last, self.epoch = seed, drop_last, 0
        self.by_subject = defaultdict(lambda: defaultdict(list))
        for i, record in enumerate(records):
            self.by_subject[record.subject][record.group_id].append(i)
    def set_epoch(self, epoch): self.epoch = epoch
    def _batches(self):
        rng = random.Random(self.seed+self.epoch); batches = []
        for grouped in self.by_subject.values():
            chunks = []
            for indices in grouped.values():
                values = indices.copy(); rng.shuffle(values); chunks.append(values)
            rng.shuffle(chunks); flat = [i for chunk in chunks for i in chunk]
            for start in range(0, len(flat), self.batch_size):
                batch = flat[start:start+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last: batches.append(batch)
        rng.shuffle(batches)
        return batches
    def __iter__(self): yield from self._batches()
    def __len__(self):
        if self.drop_last: return sum(sum(map(len,g.values()))//self.batch_size for g in self.by_subject.values())
        return sum((sum(map(len,g.values()))+self.batch_size-1)//self.batch_size for g in self.by_subject.values())
