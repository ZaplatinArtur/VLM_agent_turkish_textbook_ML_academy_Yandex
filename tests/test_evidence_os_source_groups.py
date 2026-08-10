from __future__ import annotations

from collections import defaultdict

from evidence_os.source_groups import (
    SourceGroupIndex,
    assign_group_folds,
    normalize_source_family,
)


def test_nosw_query_parameter_does_not_split_one_source_document() -> None:
    without_nosw = (
        "https://docs.yandex.ru/docs/view?lang=tr&name=Matematik.pdf"
        "&url=https%3A%2F%2Fexample.edu%2Fbooks%2Fmath-6.pdf"
    )
    with_nosw_zero = without_nosw + "&nosw=0"
    with_nosw_one = without_nosw + "&nosw=1"

    assert normalize_source_family(with_nosw_zero) == normalize_source_family(
        with_nosw_one
    )
    assert normalize_source_family(with_nosw_zero) == normalize_source_family(
        without_nosw
    )


def test_nosw_coalescing_also_applies_to_regular_urls() -> None:
    first = "https://www.example.edu/textbooks/math.pdf?chapter=3&nosw=0"
    second = "http://example.edu/textbooks/math.pdf?nosw=1&chapter=3"

    assert normalize_source_family(first) == normalize_source_family(second)


def test_source_group_folds_are_family_disjoint_and_deterministic() -> None:
    records: list[dict[str, str]] = []
    task_ids: list[str] = []
    for family_number, family_size in enumerate((4, 3, 3, 2, 2, 1), start=1):
        for item_number in range(family_size):
            task_id = f"task-{family_number}-{item_number}"
            task_ids.append(task_id)
            nosw = item_number % 2
            records.append(
                {
                    "task_id": task_id,
                    "source": (
                        f"https://example.edu/books/book-{family_number}.pdf"
                        f"?nosw={nosw}"
                    ),
                }
            )

    index = SourceGroupIndex.from_records(records)
    first = assign_group_folds(task_ids, index, n_folds=3, seed="frozen-v1")
    second = assign_group_folds(
        tuple(reversed(task_ids)), index, n_folds=3, seed="frozen-v1"
    )
    assert dict(first) == dict(second)

    folds_by_family: dict[str, set[int]] = defaultdict(set)
    for task_id in task_ids:
        folds_by_family[index.family_for(task_id)].add(first[task_id])
    assert all(len(folds) == 1 for folds in folds_by_family.values())

    for held_out_fold in range(3):
        train_families = {
            index.family_for(task_id)
            for task_id in task_ids
            if first[task_id] != held_out_fold
        }
        test_families = {
            index.family_for(task_id)
            for task_id in task_ids
            if first[task_id] == held_out_fold
        }
        assert train_families.isdisjoint(test_families)
        assert test_families
