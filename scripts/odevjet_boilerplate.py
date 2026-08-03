# -*- coding: utf-8 -*-
"""Фильтр шаблонных строк ÖdevJet — общий для скрейпера и чистки корпуса.

Списки построены по фактическим данным собранного корпуса: перечисленные
строки встречаются ровно по одному разу на КАЖДОЙ из 42 553 страниц (виджет
реакций — на 15 783), то есть это элементы интерфейса, а не учебный текст.
Совпадение строгое, по всей строке: подстрокой резать нельзя, потому что
«Çözüm» и «Sonuç» встречаются внутри настоящих решений.

Разбор ошибок (reports/tool_errors_analysis.md): 16% выдач ретрива верхним
чанком отдавали именно этот мусор — BM25 ловил на нём лексические совпадения.
"""

import re

# строки-подстроки: дисклеймеры и подсказки загрузки, встречаются с вариациями
BOILERPLATE_LINE_MARKERS = (
    "anlık görüntüleyici", "henüz görsel eklenmemiş",
    "onaylanmış öğrenci çözümü yok", "ilk çözümü sen paylaş",
    "kendi çözümünü paylaş", "jpg, png veya webp",
    "yalnızca kontrol ve öğrenme amacıyla", "velilerimiz de bu içerikleri",
    "fotoğrafını yükle", "çerez", "gizlilik politikası",
)

# строки целиком: навигация, кнопки, виджет реакций, авторская подпись сайта
BOILERPLATE_EXACT_LINES = frozenset(s.casefold() for s in (
    "Ders Kitabı", "Çözüm", "Öğrenci Çözümleri", "Çözümü Değerlendir",
    "Bildir", "Ödevi tamamladım", "Adresi kopyala", "Yazdır", "Paylaş",
    "Tamamlayan sayısı yükleniyor…", "Tamamlayan sayısı yükleniyor...",
    "Beğendim", "Bayıldım", "Güldüm", "Muhteşem", "Şaşırdım", "Sinirlendim",
    "Üzüldüm", "Mehmet Ali Taş",
    "Bu sayfada henüz çözüm bulunmamaktadır.",
    "×", "🔍",
))

# счётчик подписчиков: «1 takipçi», «12 takipçi»
BOILERPLATE_PATTERNS = (re.compile(r"^\d+\s+takipçi$", re.I),)


def is_boilerplate(line: str) -> bool:
    """Строка — элемент интерфейса ÖdevJet, а не учебный текст."""
    low = line.casefold()
    if low in BOILERPLATE_EXACT_LINES:
        return True
    if any(mark in low for mark in BOILERPLATE_LINE_MARKERS):
        return True
    return any(p.match(line) for p in BOILERPLATE_PATTERNS)


def clean_text(text: str) -> str:
    """Убирает шаблонные строки, схлопывая пробелы внутри строк."""
    out = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if line and not is_boilerplate(line):
            out.append(line)
    return "\n".join(out)
