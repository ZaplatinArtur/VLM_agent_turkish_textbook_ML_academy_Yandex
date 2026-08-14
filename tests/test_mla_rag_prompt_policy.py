from mla_baseline.prompts import PROMPTS, RAG_TOOL_POLICY_V1, RAG_TOOL_POLICY_V2_TEXT


def test_text_rag_policy_is_versioned_without_mutating_v1() -> None:
    assert PROMPTS["v2_cot"]["rag_tool_policy"] == RAG_TOOL_POLICY_V1
    assert PROMPTS["v2_cot_text_rag_v1"]["rag_tool_policy"] == RAG_TOOL_POLICY_V2_TEXT


def test_text_rag_policy_explicitly_encourages_textbook_lookup() -> None:
    policy = str(PROMPTS["v2_cot_text_rag_v1"]["rag_tool_policy"])
    assert "Metin sorularında da aracı kullan" in policy
    assert "formül, tanım, kural" in policy
    assert PROMPTS["v2_cot_text_rag_v1"]["agent_tool_note"] == policy
