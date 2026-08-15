from pathlib import Path

from .data import build_records, split_by_group


records = build_records(
    Path("catalog/train_queries_grades_1_12_blocks.cleaned.jsonl"),
    Path("catalog/visrag_relevance_groups_v3_reviewed.json"),
    Path("."),
)
train, validation, stats = split_by_group(records, 120, 17)
train_groups = {record.group_id for record in train}
validation_groups = {record.group_id for record in validation}
assert not train_groups.intersection(validation_groups)
print("records", len(records), "train", len(train), "validation", len(validation))
for subject, values in stats.items():
    print(subject, values)
