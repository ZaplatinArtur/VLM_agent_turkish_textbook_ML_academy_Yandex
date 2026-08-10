import json

from retrieve.chunking.refinement import RefinementDecision, _extract_json


def test_extracts_fenced_json() -> None:
    payload = _extract_json(
        """```json
{"decisions":[{"index":0,"kind":"exercise","confidence":0.9,"reason":"question"}]}
```"""
    )

    assert payload["decisions"][0]["kind"] == "exercise"


def test_extracts_json_surrounded_by_text() -> None:
    source = 'prefix {"decisions": []} suffix'

    assert json.dumps(_extract_json(source)) == '{"decisions": []}'


def test_optional_qwen_explanation_fields_have_safe_defaults() -> None:
    decision = RefinementDecision.model_validate(
        {"index": 0, "kind": "exercise"}
    )

    assert decision.confidence == 0.5
    assert decision.reason == "not provided"
