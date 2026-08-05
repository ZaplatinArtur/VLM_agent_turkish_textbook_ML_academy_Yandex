from visual_retrive.page_parse import parse_answer_page


def test_parse_text_and_image_answer() -> None:
    html = """
    <article>
      <div class="solution-container">
        <div class="image-viewer">
          <img src="https://www.odevjet.com/download/cevaplar/demo-book/20.webp" />
        </div>
        <div class="text-solution-content">
          <p>Merhaba</p><p>Çözüm: 1/4</p>
        </div>
      </div>
      <img src="https://www.odevjet.com/download/sayfalar/demo-book/20.webp" />
    </article>
    """
    parsed = parse_answer_page(html)
    assert parsed["has_solution"] is True
    assert parsed["answer_kinds"] == ["text", "image"]
    assert "1/4" in parsed["answer_text"]
    assert parsed["answer_image_urls"] == [
        "https://www.odevjet.com/download/cevaplar/demo-book/20.webp"
    ]
    assert parsed["page_image_urls"] == [
        "https://www.odevjet.com/download/sayfalar/demo-book/20.webp"
    ]


def test_parse_empty_answer_marker() -> None:
    html = """
    <div class="no-solution">Bu sayfada henüz çözüm bulunmamaktadır.</div>
    <div class="text-solution-content"></div>
    """
    parsed = parse_answer_page(html)
    assert parsed["has_solution"] is False
    assert parsed["no_solution"] is True
    assert parsed["answer_text"] == ""
    assert parsed["answer_kinds"] == []
