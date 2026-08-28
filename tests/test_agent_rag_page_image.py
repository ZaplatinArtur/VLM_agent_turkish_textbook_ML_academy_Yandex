import base64
import json

from mla_baseline.config import Settings
from mla_baseline.solvers.agent_rag import AgentRag

JPEG = b"\xff\xd8\xff\xe0" + b"page-bytes"


class ImageClient:
    """Визуальный бэкенд: отдаёт байты страницы по page_id."""

    def __init__(self, images=None):
        self.images = images if images is not None else {"p1": JPEG, "p2": JPEG}

    def get_page_image(self, page_id):
        return self.images.get(page_id)


class TextClient:
    """Текстовый бэкенд картинок не отдаёт вовсе."""


def tool_output(*, is_useful=True, hits=(("p1", 18.4),)) -> str:
    return json.dumps(
        {
            "relevance": {"label": "confident" if is_useful else "weak",
                          "is_useful": is_useful},
            "hits": [
                {"chunk_id": pid, "page_id": pid, "score": score, "text": "çözüm"}
                for pid, score in hits
            ],
        },
        ensure_ascii=False,
    )


def solver(top_n: int, client) -> AgentRag:
    agent = AgentRag.__new__(AgentRag)
    agent.settings = Settings(visual_image_top_n=top_n)
    agent.search_client = client
    return agent


def test_disabled_by_default():
    # Дефолт 0: режим text не отличается от прежнего ни на одно сообщение.
    assert solver(0, ImageClient())._page_image_message(tool_output(), 0) is None


def test_attaches_page_image_when_enabled():
    message = solver(1, ImageClient())._page_image_message(tool_output(), 0)
    assert message is not None
    blocks = message.content
    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    encoded = blocks[1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == JPEG


def test_label_separates_the_page_from_the_question():
    message = solver(1, ImageClient())._page_image_message(tool_output(), 0)
    text = message.content[0]["text"]
    assert "soru değildir" in text


def test_weak_output_gets_no_image():
    # Гейт уже спрятал выдачу — картинку тем более показывать нельзя.
    message = solver(1, ImageClient())._page_image_message(
        tool_output(is_useful=False), 0
    )
    assert message is None


def test_cap_counts_per_task_not_per_call():
    agent = solver(1, ImageClient())
    assert agent._page_image_message(tool_output(), 0) is not None
    assert agent._page_image_message(tool_output(), 1) is None


def test_text_backend_never_attaches():
    assert solver(1, TextClient())._page_image_message(tool_output(), 0) is None


def test_missing_bytes_fall_through_to_the_next_hit():
    client = ImageClient({"p2": JPEG})
    message = solver(1, client)._page_image_message(
        tool_output(hits=(("p1", 18.4), ("p2", 15.0))), 0
    )
    assert message is not None


def test_no_bytes_at_all_means_no_message():
    message = solver(1, ImageClient({}))._page_image_message(tool_output(), 0)
    assert message is None


def test_broken_tool_output_is_survivable():
    assert solver(1, ImageClient())._page_image_message("not json", 0) is None
