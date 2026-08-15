from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset, Sampler


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_relevance_groups(path: Path) -> dict[str, str]:
    """Map each page to a stable equivalence-group id."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    page_to_group: dict[str, str] = {}
    for number, pages in enumerate(payload.get("groups", [])):
        group_id = f"reviewed:{number}"
        for page_id in pages:
            if page_id in page_to_group:
                raise ValueError(f"page occurs in two relevance groups: {page_id}")
            page_to_group[str(page_id)] = group_id
    return page_to_group


@dataclass
class Record:
    page_id: str
    group_id: str
    subject: str
    grade: int | None
    image: Path
    queries: list[str]


def build_records(pairs: Path, groups: Path, data_root: Path) -> list[Record]:
    page_to_group = load_relevance_groups(groups)
    pages: dict[str, dict] = {}
    for row in read_jsonl(pairs):
        query = str(row.get("query") or "").strip()
        page_id = str(row.get("positive_page_id") or "")
        image = data_root / str(row.get("positive_image") or "")
        if not query or not page_id or not image.is_file():
            continue
        rec = pages.setdefault(page_id, {
            "page_id": page_id,
            "group_id": page_to_group.get(page_id, f"page:{page_id}"),
            "subject": str(row.get("subject") or "unknown"),
            "grade": row.get("grade"), "image": image, "queries": [],
        })
        if query not in rec["queries"]:
            rec["queries"].append(query)
    return [Record(**row) for row in pages.values()]


def split_by_group(records: list[Record], pages_per_subject: int, seed: int):
    """Hold out whole equivalence groups; never leak sibling positives to train."""
    by_subject: dict[str, dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_subject[record.subject][record.group_id].append(record)
    train, validation, stats = [], [], {}
    for subject, grouped in sorted(by_subject.items()):
        group_ids = sorted(grouped)
        random.Random(f"{seed}:{subject}").shuffle(group_ids)
        chosen, count = [], 0
        for group_id in group_ids:
            if count >= pages_per_subject:
                break
            chosen.append(group_id)
            count += len(grouped[group_id])
        chosen_set = set(chosen)
        subject_val = [r for group_id in chosen for r in grouped[group_id]][:pages_per_subject]
        validation.extend(subject_val)
        val_ids = {r.page_id for r in subject_val}
        # If a relevance group crosses the 120-page boundary, keep all of it out of train.
        train.extend(r for group_id, rows in grouped.items() if group_id not in chosen_set for r in rows)
        stats[subject] = {"train_pages": sum(len(v) for k, v in grouped.items() if k not in chosen_set),
                          "validation_pages": len(subject_val), "held_out_group_pages": count,
                          "excluded_boundary_pages": count - len(val_ids)}
    return train, validation, stats


class PageDataset(Dataset):
    def __init__(self, records: list[Record], seed: int = 17):
        self.records, self.seed, self.epoch = records, seed, 0
    def __len__(self): return len(self.records)
    def __getitem__(self, index): return self.records[index]


class SubjectBatchSampler(Sampler[list[int]]):
    """Every in-batch negative comes from the same school subject."""
    def __init__(self, records: list[Record], batch_size: int, seed: int = 17, drop_last: bool = True):
        self.records, self.batch_size, self.seed, self.drop_last, self.epoch = records, batch_size, seed, drop_last, 0
        self.by_subject = defaultdict(list)
        for i, record in enumerate(records): self.by_subject[record.subject].append(i)
    def set_epoch(self, epoch): self.epoch = epoch
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for indices in self.by_subject.values():
            indices = indices.copy(); rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last: batches.append(batch)
        rng.shuffle(batches)
        yield from batches
    def __len__(self):
        if self.drop_last: return sum(len(v)//self.batch_size for v in self.by_subject.values())
        return sum((len(v)+self.batch_size-1)//self.batch_size for v in self.by_subject.values())

