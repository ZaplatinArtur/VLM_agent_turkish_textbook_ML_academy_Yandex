from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .source_wave import SourceWaveSummary


def _badge(text: str, kind: str = "neutral") -> QLabel:
    names = {
        "good": "BadgeGood",
        "bad": "BadgeBad",
        "info": "BadgeInfo",
        "warn": "BadgeWarn",
        "neutral": "BadgeNeutral",
    }
    label = QLabel(text)
    label.setObjectName(names.get(kind, "BadgeNeutral"))
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return label


def _short_hash(value: str) -> str:
    return f"{value[:12]}…{value[-8:]}"


class SourceMetricCard(QFrame):
    def __init__(self, title: str, value: str, hint: str, accent: str):
        super().__init__()
        self.setObjectName("MetricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 9, 14, 9)
        layout.setSpacing(1)
        title_label = QLabel(title.upper())
        title_label.setObjectName("MetricLabel")
        value_label = QLabel(value)
        value_label.setObjectName("MetricValue")
        value_label.setStyleSheet(f"color: {accent};")
        hint_label = QLabel(hint)
        hint_label.setObjectName("MetricHint")
        hint_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(hint_label)


class LineageCard(QFrame):
    def __init__(
        self,
        index: str,
        title: str,
        score: str,
        detail: str,
        *,
        active: bool = False,
    ):
        super().__init__()
        self.setObjectName("NoticeCard")
        if active:
            self.setStyleSheet(
                "QFrame#NoticeCard { background: #12312d; border: 2px solid #4be1c3; "
                "border-radius: 12px; }"
            )
        layout = QGridLayout(self)
        layout.setContentsMargins(13, 10, 13, 10)
        layout.setHorizontalSpacing(10)
        number = QLabel(index)
        number.setObjectName("TimelineNumber")
        number.setAlignment(Qt.AlignmentFlag.AlignCenter)
        number.setFixedSize(34, 34)
        name = QLabel(title)
        name.setStyleSheet("color: #eef7fc; font-size: 10pt; font-weight: 700;")
        value = QLabel(score)
        value.setStyleSheet(
            "color: #4be1c3; font-size: 19pt; font-weight: 800;"
            if active
            else "color: #9db8cc; font-size: 19pt; font-weight: 800;"
        )
        note = QLabel(detail)
        note.setObjectName("Tiny")
        note.setWordWrap(True)
        layout.addWidget(number, 0, 0, 2, 1)
        layout.addWidget(name, 0, 1)
        layout.addWidget(value, 0, 2, 2, 1)
        layout.addWidget(note, 1, 1)
        layout.setColumnStretch(1, 1)


class SourceExpansionWavePage(QWidget):
    def __init__(self, summary: SourceWaveSummary):
        super().__init__()
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(14, 13, 14, 13)
        root.setSpacing(10)

        heading = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("All-9B · official source expansion wave")
        title.setObjectName("SectionTitle")
        subtitle = QLabel(
            "активный development headline прочитан из frozen official16: десять arms "
            "стартовали одновременно, а официальный результат отделён от research-веток"
        )
        subtitle.setObjectName("Subtle")
        subtitle.setWordWrap(True)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        heading.addLayout(title_box, 1)
        heading.addWidget(_badge("QWEN3.5-9B ONLY", "info"))
        heading.addWidget(_badge("POST-SCORE AUDIT · PASS", "good"))
        heading.addWidget(_badge("DEV · KNOWN BENCHMARK", "warn"))
        root.addLayout(heading)

        metrics = QHBoxLayout()
        metrics.setSpacing(9)
        metrics.addWidget(
            SourceMetricCard(
                "Official16 · active",
                f"{summary.accuracy:.4%}",
                f"{summary.correct}/{summary.rows} · exact {summary.correct / summary.rows:.10f}",
                "#4be1c3",
            )
        )
        metrics.addWidget(
            SourceMetricCard(
                "Math",
                f"{summary.math_correct}/{summary.math_rows}",
                f"{summary.math_correct / summary.math_rows:.2%} · +8 correct vs 240",
                "#71aef5",
            )
        )
        metrics.addWidget(
            SourceMetricCard(
                "English",
                f"{summary.english_correct}/{summary.english_rows}",
                "100% · +1 correct vs 240",
                "#b99cff",
            )
        )
        metrics.addWidget(
            SourceMetricCard(
                "Deterministic",
                f"{summary.deterministic_correct}/{summary.deterministic_rows}",
                f"{summary.deterministic_correct / summary.deterministic_rows:.2%} · unchanged",
                "#f0ad62",
            )
        )
        metrics.addWidget(
            SourceMetricCard(
                "Image judge",
                f"{summary.image_correct}/{summary.image_rows}",
                f"{summary.image_correct / summary.image_rows:.2%} · +9",
                "#e8899c",
            )
        )
        root.addLayout(metrics)

        body = QHBoxLayout()
        body.setSpacing(10)
        left = QVBoxLayout()
        left.setSpacing(9)

        lineage = QFrame()
        lineage.setObjectName("Panel")
        lineage_layout = QVBoxLayout(lineage)
        lineage_layout.setContentsMargins(12, 10, 12, 10)
        lineage_header = QHBoxLayout()
        lineage_title = QLabel("Честная линия результата")
        lineage_title.setStyleSheet("color: #eef7fc; font-size: 11pt; font-weight: 700;")
        lineage_header.addWidget(lineage_title)
        lineage_header.addStretch(1)
        lineage_header.addWidget(_badge("238 И 240 СОХРАНЕНЫ КАК LINEAGE", "info"))
        lineage_layout.addLayout(lineage_header)
        lineage_cards = QHBoxLayout()
        lineage_cards.setSpacing(8)
        lineage_cards.addWidget(
            LineageCard("07", "Source V7", "238/274", "каноническая семиступенчатая точка")
        )
        lineage_cards.addWidget(
            LineageCard("08", "Selector v1.2", "240/274", "+2 / 0 · прежний audited headline")
        )
        lineage_cards.addWidget(
            LineageCard(
                "09",
                "Official16",
                "249/274",
                "+9 / 0 · новый активный headline",
                active=True,
            )
        )
        lineage_layout.addLayout(lineage_cards)
        left.addWidget(lineage)

        fixes = QFrame()
        fixes.setObjectName("Panel")
        fixes_layout = QVBoxLayout(fixes)
        fixes_layout.setContentsMargins(12, 9, 12, 9)
        fixes_header = QHBoxLayout()
        fixes_title = QLabel("Девять проверенных исправлений против audited 240")
        fixes_title.setStyleSheet("color: #eef7fc; font-size: 11pt; font-weight: 700;")
        fixes_header.addWidget(fixes_title)
        fixes_header.addStretch(1)
        fixes_header.addWidget(_badge("+9 FIXES · 0 REGRESSIONS", "good"))
        fixes_layout.addLayout(fixes_header)
        fixes_grid = QGridLayout()
        fixes_grid.setHorizontalSpacing(7)
        fixes_grid.setVerticalSpacing(6)
        for index, task_id in enumerate(summary.fix_task_ids):
            pill = QFrame()
            pill.setObjectName("NoticeCard")
            pill_layout = QHBoxLayout(pill)
            pill_layout.setContentsMargins(9, 6, 9, 6)
            task = QLabel(task_id)
            task.setStyleSheet("color: #dffaf3; font-weight: 750; font-family: 'Cascadia Mono';")
            subject = QLabel("Math" if task_id != "val_0182" else "English")
            subject.setObjectName("Tiny")
            pill_layout.addWidget(task)
            pill_layout.addStretch(1)
            pill_layout.addWidget(subject)
            fixes_grid.addWidget(pill, index // 3, index % 3)
        fixes_layout.addLayout(fixes_grid)
        left.addWidget(fixes)

        sources = QFrame()
        sources.setObjectName("NoticeCard")
        sources_layout = QGridLayout(sources)
        sources_layout.setContentsMargins(12, 9, 12, 9)
        sources_title = QLabel("Официальные source-компоненты official16")
        sources_title.setObjectName("MetricLabel")
        sources_layout.addWidget(sources_title, 0, 0, 1, 3)
        components = (
            ("Math12 · 5", "визуально верные графики, формулы и построение"),
            ("MEB7 · 6", "точная локализация страницы и crop официального PDF"),
            ("English10 · 5", "официальный источник и source-adjudicated judge"),
        )
        for column, (name, detail) in enumerate(components):
            card = QFrame()
            card.setObjectName("ChronologyCard")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(10, 7, 10, 7)
            label = QLabel(name)
            label.setStyleSheet("color: #8fdcca; font-weight: 750;")
            note = QLabel(detail)
            note.setObjectName("Tiny")
            note.setWordWrap(True)
            layout.addWidget(label)
            layout.addWidget(note)
            sources_layout.addWidget(card, 1, column)
        left.addWidget(sources)
        left.addStretch(1)
        body.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(9)
        integrity = QFrame()
        integrity.setObjectName("Panel")
        integrity_layout = QGridLayout(integrity)
        integrity_layout.setContentsMargins(12, 9, 12, 9)
        integrity_layout.setHorizontalSpacing(12)
        integrity_title = QLabel("Fail-closed provenance chain")
        integrity_title.setStyleSheet("color: #eef7fc; font-size: 11pt; font-weight: 700;")
        integrity_layout.addWidget(integrity_title, 0, 0, 1, 2)
        hashes = (
            ("freeze", summary.freeze_sha256),
            ("audit amendment", summary.amendment_sha256),
            ("wave completion", summary.completion_sha256),
            ("official16 metrics", summary.official_metrics_sha256),
            ("official solver", summary.official_solver_sha256),
            ("official image judge", summary.official_image_judge_sha256),
        )
        for row, (name, value) in enumerate(hashes, 1):
            label = QLabel(name)
            label.setObjectName("Tiny")
            digest = QLabel(_short_hash(value))
            digest.setObjectName("Tiny")
            digest.setStyleSheet("font-family: 'Cascadia Mono'; color: #9db9ce;")
            digest.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            digest.setToolTip(value)
            integrity_layout.addWidget(label, row, 0)
            integrity_layout.addWidget(digest, row, 1)
        integrity_layout.setColumnStretch(1, 1)
        right.addWidget(integrity)

        research = QFrame()
        research.setObjectName("NoticeCard")
        research.setStyleSheet(
            "QFrame#NoticeCard { background: #2a2118; border: 2px solid #8c6330; "
            "border-radius: 12px; }"
        )
        research_layout = QVBoxLayout(research)
        research_layout.setContentsMargins(13, 10, 13, 10)
        research_header = QHBoxLayout()
        research_title = QLabel("Отдельный research-блок")
        research_title.setStyleSheet("color: #ffd68a; font-size: 11pt; font-weight: 750;")
        research_header.addWidget(research_title)
        research_header.addStretch(1)
        research_header.addWidget(_badge("НЕ HEADLINE", "warn"))
        research_score = QLabel(
            f"research_all36 · {summary.research_all36.correct}/{summary.research_all36.rows} "
            f"= {summary.research_all36.accuracy:.4%}"
        )
        research_score.setStyleSheet("color: #ffd68a; font-size: 16pt; font-weight: 800;")
        research_note = QLabel(
            "research_evaluation_only · источники BS/Fenomen не входят в официальный MEB-контур · "
            "лицензии не проверены · production=false. 251 никогда не используется как active score."
        )
        research_note.setObjectName("Subtle")
        research_note.setWordWrap(True)
        research_layout.addLayout(research_header)
        research_layout.addWidget(research_score)
        research_layout.addWidget(research_note)
        right.addWidget(research)

        boundary = QFrame()
        boundary.setObjectName("NoticeCard")
        boundary_layout = QVBoxLayout(boundary)
        boundary_layout.setContentsMargins(13, 10, 13, 10)
        boundary_title = QLabel("Граница утверждения")
        boundary_title.setObjectName("MetricLabel")
        boundary_text = QLabel(
            "Единственный активный headline: 249/274 = 0.908759 (точное отношение 249/274). "
            "Это известный development benchmark, frozen simultaneous wave и независимый post-score audit; "
            "не blind holdout и не production accuracy."
        )
        boundary_text.setObjectName("Subtle")
        boundary_text.setWordWrap(True)
        boundary_layout.addWidget(boundary_title)
        boundary_layout.addWidget(boundary_text)
        right.addWidget(boundary)
        right.addStretch(1)
        body.addLayout(right, 2)

        root.addLayout(body, 1)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
