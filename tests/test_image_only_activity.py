from __future__ import annotations

import copy

import pytest

from evidence_os.image_only_activity import (
    ImageOnlyActivityError,
    OBSERVATION_KIND,
    project_image_only_activity_observation,
)


def _row() -> dict[str, object]:
    return {
        "schema_version": "maxim-paddleocr-vl16-task-parse-v1",
        "task_id": "alignment-only-id",
        "parser": {
            "pipeline_version": "v1.6",
            "layout_model": "PP-DocLayoutV3",
            "recognition_model": "PaddleOCR-VL-1.6-0.9B",
            "recognition_backend": "vllm-server",
            "max_new_tokens": 1024,
            "gold_access": False,
        },
        "images": [
            {
                "image_index": 0,
                "image_basename": "task.png",
                "image_sha256": "a" * 64,
                "width": 1000,
                "height": 1400,
                "input_decode": {"kind": "path"},
                "parsing_res_list": [
                    {
                        "block_label": "image",
                        "block_content": '<div><img src="crop.jpg" /></div>',
                        "block_bbox": [20, 0, 980, 1400],
                        "block_id": 0,
                        "block_order": None,
                        "group_id": 0,
                        "block_polygon_points": [
                            [20.0, 0.0],
                            [980.0, 0.0],
                            [980.0, 1400.0],
                            [20.0, 1400.0],
                        ],
                    }
                ],
            }
        ],
    }


def test_projects_one_full_page_image_without_task_id_as_a_feature() -> None:
    first = project_image_only_activity_observation(_row())
    changed = _row()
    changed["task_id"] = "different-alignment-id"
    second = project_image_only_activity_observation(changed)

    assert first.task_id != second.task_id
    assert first.parser_projection_sha256 == second.parser_projection_sha256
    assert first.block_area_coverage == pytest.approx(0.96)
    assert OBSERVATION_KIND in (
        "single_full_page_image_block_without_text_v1",
    )


def test_rejects_any_textual_parser_block() -> None:
    value = _row()
    image = value["images"][0]  # type: ignore[index]
    image["parsing_res_list"].append(  # type: ignore[index, union-attr]
        {
            "block_label": "text",
            "block_content": "answer-like text",
            "block_bbox": [10, 10, 300, 50],
        }
    )

    with pytest.raises(ImageOnlyActivityError, match="exactly one image block"):
        project_image_only_activity_observation(value)


def test_rejects_a_small_embedded_image_block() -> None:
    value = _row()
    image = value["images"][0]  # type: ignore[index]
    block = image["parsing_res_list"][0]  # type: ignore[index]
    block["block_bbox"] = [200, 200, 800, 1000]  # type: ignore[index]

    with pytest.raises(ImageOnlyActivityError, match="does not cover"):
        project_image_only_activity_observation(value)


def test_rejects_forbidden_outcome_metadata_at_any_depth() -> None:
    value = copy.deepcopy(_row())
    image = value["images"][0]  # type: ignore[index]
    image["outcome"] = True  # type: ignore[index]

    with pytest.raises(ImageOnlyActivityError, match="forbidden"):
        project_image_only_activity_observation(value)
