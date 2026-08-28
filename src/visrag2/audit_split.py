import argparse
from pathlib import Path

from .data import build_records, split_by_group


p = argparse.ArgumentParser()
p.add_argument("--pairs", type=Path, required=True)
p.add_argument("--groups", type=Path, required=True)
p.add_argument("--data-root", type=Path, required=True)
p.add_argument("--pages-per-subject", type=int, default=120)
p.add_argument("--seed", type=int, default=17)
a = p.parse_args()
records = build_records(a.pairs, a.groups, a.data_root)
train, validation, stats = split_by_group(records, a.pages_per_subject, a.seed)
assert not ({record.group_id for record in train} & {record.group_id for record in validation})
assert all(values["validation_pages"] == a.pages_per_subject for values in stats.values())
print("records", len(records), "train", len(train), "validation", len(validation))
for subject, values in stats.items(): print(subject, values)
