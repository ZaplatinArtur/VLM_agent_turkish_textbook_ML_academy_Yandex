from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
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

from .selector_wave import Milestone, SelectorTaskProvenance, SelectorWaveSummary


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


class SelectorMetricCard(QFrame):
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


class CanonicalLadderCanvas(QWidget):
    """The seven pinned source-replay milestones; the selector is not drawn here."""

    def __init__(self, milestones: tuple[Milestone, ...]):
        super().__init__()
        self.milestones = milestones
        self.setMinimumHeight(238)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0d1924"))
        if not self.milestones:
            return

        margin = 18.0
        gap = 10.0
        node_width = max(
            105.0,
            (self.width() - margin * 2 - gap * (len(self.milestones) - 1))
            / len(self.milestones),
        )
        top, node_height = 33.0, 140.0
        centers: list[QPointF] = []
        for index in range(len(self.milestones)):
            x = margin + index * (node_width + gap)
            centers.append(QPointF(x + node_width / 2, top + node_height / 2))

        painter.setPen(QPen(QColor("#31526a"), 2.0))
        for left, right in zip(centers, centers[1:]):
            start = QPointF(left.x() + node_width / 2, left.y())
            end = QPointF(right.x() - node_width / 2, right.y())
            painter.drawLine(start, end)
            painter.setBrush(QColor("#4b7895"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(end, 3.2, 3.2)
            painter.setPen(QPen(QColor("#31526a"), 2.0))

        previous = 0
        for index, milestone in enumerate(self.milestones):
            x = margin + index * (node_width + gap)
            rect = QRectF(x, top, node_width, node_height)
            source_stage = index >= 3
            final_stage = index == len(self.milestones) - 1
            fill = QColor("#112234" if not source_stage else "#102b29")
            border = QColor("#315978" if not source_stage else "#277563")
            accent = QColor("#84c3ff" if not source_stage else "#4be1c3")
            if final_stage:
                fill = QColor("#153832")
                border = QColor("#4be1c3")
            painter.setPen(QPen(border, 2.1 if final_stage else 1.2))
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 11, 11)

            painter.setPen(accent)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(
                QRectF(rect.left() + 10, rect.top() + 8, rect.width() - 20, 19),
                f"{index + 1:02d}",
            )
            painter.setPen(QColor("#eaf4fb"))
            label_font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
            painter.setFont(label_font)
            label = QFontMetrics(label_font).elidedText(
                milestone.label,
                Qt.TextElideMode.ElideRight,
                max(40, int(rect.width() - 20)),
            )
            painter.drawText(
                QRectF(rect.left() + 10, rect.top() + 31, rect.width() - 20, 34),
                Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                label,
            )
            painter.setPen(accent)
            painter.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
            painter.drawText(
                QRectF(rect.left() + 10, rect.top() + 67, rect.width() - 20, 29),
                f"{milestone.correct}/{milestone.rows}",
            )
            painter.setPen(QColor("#91a5b6"))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(
                QRectF(rect.left() + 10, rect.top() + 100, rect.width() - 20, 18),
                f"{milestone.accuracy:.2%}",
            )
            delta = milestone.correct - previous if index else 0
            painter.setPen(QColor("#4be1c3" if delta > 0 else "#70869a"))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.DemiBold))
            painter.drawText(
                QRectF(rect.left() + 10, rect.bottom() - 22, rect.width() - 20, 14),
                "НАЧАЛО" if index == 0 else f"+{delta} к этапу {index:02d}",
            )
            previous = milestone.correct

        painter.setPen(QColor("#6e8497"))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(
            QRectF(margin, top + node_height + 17, self.width() - 2 * margin, 31),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            "Хронология читается слева направо. Source V7 = 238 — последняя точка канонической лестницы; слой selector v1.2 вынесен отдельно.",
        )


class SelectorLayerCard(QFrame):
    def __init__(self, summary: SelectorWaveSummary):
        super().__init__()
        self.setObjectName("SelectorLayerCard")
        self.setStyleSheet(
            "QFrame#SelectorLayerCard { background: #102c2a; border: 2px solid #3bd4b4; "
            "border-radius: 13px; }"
        )
        layout = QGridLayout(self)
        layout.setContentsMargins(16, 13, 16, 13)
        layout.setHorizontalSpacing(17)
        layout.setVerticalSpacing(5)

        index = QLabel("08")
        index.setStyleSheet("color: #4be1c3; font-size: 26pt; font-weight: 800;")
        title = QLabel("Baseline Selector v1.2")
        title.setObjectName("SectionTitle")
        subtitle = QLabel(
            "отдельный development layer поверх Source V7 · не переписывает каноническую семёрку"
        )
        subtitle.setObjectName("Subtle")
        subtitle.setWordWrap(True)
        score = QLabel(f"{summary.milestones[-1].correct}  →  {summary.correct}")
        score.setStyleSheet("color: #ffffff; font-size: 25pt; font-weight: 800;")
        delta = QLabel(f"+{summary.fixes} fixes · {summary.regressions} regressions")
        delta.setObjectName("Success")
        contract = QLabel(
            "три заранее определённые группы согласились по двум deterministic-задачам; "
            "272/274 строк прошли побайтово, без замены"
        )
        contract.setObjectName("Subtle")
        contract.setWordWrap(True)
        badges = QHBoxLayout()
        badges.addWidget(_badge("QWEN3.5-9B ONLY", "info"))
        badges.addWidget(_badge("240 · NOT 241", "warn"))
        badges.addWidget(_badge("ONE-SHOT ARM", "neutral"))
        badges.addStretch(1)

        layout.addWidget(index, 0, 0, 2, 1, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(title, 0, 1)
        layout.addWidget(subtitle, 1, 1)
        layout.addWidget(score, 0, 2, 2, 1, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(delta, 0, 3)
        layout.addWidget(contract, 1, 3)
        layout.addLayout(badges, 2, 1, 1, 3)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(3, 3)


class ProvenanceCard(QFrame):
    def __init__(self, task: SelectorTaskProvenance):
        super().__init__()
        self.setObjectName("Panel")
        self.setMinimumHeight(166)
        self.setMaximumHeight(178)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 7, 11, 7)
        layout.setSpacing(1)

        header = QHBoxLayout()
        title = QLabel(f"{task.task_id} · {task.subject}")
        title.setStyleSheet("color: #f4fbff; font-size: 11pt; font-weight: 700;")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(_badge("WRONG → CORRECT", "good"))
        layout.addLayout(header)

        answer = QLabel(f"{task.anchor_answer}   →   {task.selected_answer}")
        answer.setStyleSheet("color: #4be1c3; font-size: 18pt; font-weight: 800;")
        layout.addWidget(answer)

        votes = QGridLayout()
        votes.setContentsMargins(0, 0, 0, 0)
        votes.setVerticalSpacing(0)
        rows = (
            ("structural", task.structural_answer),
            ("native group", f"{task.native_answer}  ·  v4={task.native_v4_answer}, v5={task.native_v5_answer}"),
            ("parallel group", task.parallel_answer),
        )
        for row_index, (name, value) in enumerate(rows):
            label = QLabel(name.upper())
            label.setObjectName("MetricLabel")
            vote = QLabel(value)
            vote.setStyleSheet("color: #b9f5e7; font-weight: 700;")
            votes.addWidget(label, row_index, 0)
            votes.addWidget(vote, row_index, 1)
        votes.setColumnStretch(1, 1)
        layout.addLayout(votes)

        route = QLabel(
            f"route: {task.route} · rule: all_three_preregistered_groups_agree · row {task.row_index}"
        )
        route.setObjectName("Tiny")
        layout.addWidget(route)

        hashes = QLabel(
            f"base {_short_hash(task.base_row_sha256)}  →  output {_short_hash(task.output_row_sha256)}\n"
            f"proposal {_short_hash(task.proposal_row_sha256)}  ·  source {_short_hash(task.source_row_sha256)}"
        )
        hashes.setObjectName("Tiny")
        hashes.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hashes.setToolTip(
            f"base {task.base_row_sha256}\noutput {task.output_row_sha256}\n"
            f"proposal {task.proposal_row_sha256}\nsource {task.source_row_sha256}"
        )
        layout.addWidget(hashes)


class SelectorWavePage(QWidget):
    def __init__(
        self,
        summary: SelectorWaveSummary,
        qa_correct: int | None = None,
        qa_rows: int | None = None,
    ):
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
        title = QLabel("All-9B · audited selector wave")
        title.setObjectName("SectionTitle")
        subtitle = QLabel(
            "история результатов и отдельный frozen development-слой выбора ответа · данные читаются только из проверенных артефактов"
        )
        subtitle.setObjectName("Subtle")
        subtitle.setWordWrap(True)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        heading.addLayout(title_box, 1)
        if qa_correct is not None and qa_rows:
            heading.addWidget(
                _badge(
                    f"QA TRACE V7 SEPARATE · {qa_correct}/{qa_rows} = {qa_correct / qa_rows:.6f}",
                    "info",
                )
            )
        heading.addWidget(_badge("DEV · KNOWN BENCHMARK", "warn"))
        heading.addWidget(_badge("HASH-BOUND", "good"))
        root.addLayout(heading)

        metrics = QHBoxLayout()
        metrics.setSpacing(9)
        metrics.addWidget(
            SelectorMetricCard(
                "Selector v1.2",
                f"{summary.accuracy:.2%}",
                f"{summary.correct}/{summary.rows} · audited; 241 не заявлен",
                "#4be1c3",
            )
        )
        metrics.addWidget(
            SelectorMetricCard(
                "Math",
                f"{summary.math_correct}/{summary.math_rows}",
                f"{summary.math_correct / summary.math_rows:.2%}",
                "#71aef5",
            )
        )
        metrics.addWidget(
            SelectorMetricCard(
                "Evaluation split",
                f"{summary.deterministic_correct}/{summary.deterministic_rows}",
                f"deterministic · image {summary.image_correct}/{summary.image_rows}",
                "#b99cff",
            )
        )
        metrics.addWidget(
            SelectorMetricCard(
                "Preserved",
                f"{summary.passthrough_rows}/{summary.rows}",
                f"byte-exact · source {summary.source_preserved_rows} · image {summary.image_preserved_rows}",
                "#f0ad62",
            )
        )
        root.addLayout(metrics)

        body = QHBoxLayout()
        body.setSpacing(10)

        left = QVBoxLayout()
        left.setSpacing(9)
        ladder_panel = QFrame()
        ladder_panel.setObjectName("Panel")
        ladder_layout = QVBoxLayout(ladder_panel)
        ladder_layout.setContentsMargins(11, 9, 11, 8)
        ladder_header = QHBoxLayout()
        ladder_title = QLabel("Каноническая лестница · 7 milestones")
        ladder_title.setStyleSheet("color: #eef7fc; font-size: 11pt; font-weight: 700;")
        ladder_header.addWidget(ladder_title)
        ladder_header.addStretch(1)
        ladder_header.addWidget(_badge("SOURCE V7 ENDS AT 238", "info"))
        ladder_layout.addLayout(ladder_header)
        ladder_layout.addWidget(CanonicalLadderCanvas(summary.milestones), 1)
        left.addWidget(ladder_panel, 3)
        left.addWidget(SelectorLayerCard(summary), 0)

        chronology = QFrame()
        chronology.setObjectName("NoticeCard")
        chronology_layout = QVBoxLayout(chronology)
        chronology_layout.setContentsMargins(13, 10, 13, 10)
        chronology_title = QLabel("Что именно добавил selector")
        chronology_title.setObjectName("MetricLabel")
        chronology_text = QLabel(
            "Source V7 сначала был полностью завершён и оценён как 238/274. Затем один frozen one-shot wave "
            "собрал независимые структурную, native и parallel группы. Композитор заменил ответ только при "
            "единогласии всех трёх групп: две замены, два исправления, ни одного отката. Это выбор между "
            "сохранёнными кандидатами — не новый source lookup и не отдельная метрика reasoning/QA."
        )
        chronology_text.setObjectName("Subtle")
        chronology_text.setWordWrap(True)
        chronology_layout.addWidget(chronology_title)
        chronology_layout.addWidget(chronology_text)
        left.addWidget(chronology)
        body.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(8)
        provenance_title = QLabel("Две реально изменённые задачи · provenance")
        provenance_title.setStyleSheet("color: #eef7fc; font-size: 11pt; font-weight: 700;")
        right.addWidget(provenance_title)
        for task in summary.tasks:
            right.addWidget(ProvenanceCard(task))

        integrity = QFrame()
        integrity.setObjectName("NoticeCard")
        integrity.setMaximumHeight(132)
        integrity_layout = QGridLayout(integrity)
        integrity_layout.setContentsMargins(11, 7, 11, 7)
        integrity_layout.setHorizontalSpacing(12)
        integrity_layout.setVerticalSpacing(1)
        integrity_title = QLabel("Integrity chain")
        integrity_title.setObjectName("MetricLabel")
        integrity_layout.addWidget(integrity_title, 0, 0, 1, 2)
        hashes = (
            ("completion", summary.completion_sha256),
            ("primary score", summary.score_sha256),
            ("solver", summary.solver_sha256),
            ("composition", summary.composition_sha256),
            ("repair freeze", summary.repair_output_freeze_sha256),
            ("repair score", summary.repair_score_sha256),
        )
        for row, (name, value) in enumerate(hashes, 1):
            label = QLabel(name)
            label.setObjectName("Tiny")
            hash_label = QLabel(_short_hash(value))
            hash_label.setObjectName("Tiny")
            hash_label.setStyleSheet("font-family: 'Cascadia Mono'; color: #9db9ce;")
            hash_label.setToolTip(value)
            hash_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            integrity_layout.addWidget(label, row, 0)
            integrity_layout.addWidget(hash_label, row, 1)
        integrity_layout.setColumnStretch(1, 1)
        right.addWidget(integrity)

        repair = QFrame()
        repair.setObjectName("NoticeCard")
        repair_layout = QVBoxLayout(repair)
        repair_layout.setContentsMargins(13, 9, 13, 9)
        repair_header = QHBoxLayout()
        repair_title = QLabel("Post-score answer-contract repair v1.1")
        repair_title.setObjectName("MetricLabel")
        repair_header.addWidget(repair_title)
        repair_header.addStretch(1)
        repair_header.addWidget(_badge("NULL · 240 → 240", "neutral"))
        repair_text = QLabel(
            f"{summary.repair_task_id}: одна content-замена после анализа известных ошибок; "
            "correctness не изменился. это post-score, non-blind и non-preregistered integrity result, "
            "а не новый milestone."
        )
        repair_text.setObjectName("Subtle")
        repair_text.setWordWrap(True)
        repair_layout.addLayout(repair_header)
        repair_layout.addWidget(repair_text)
        right.addWidget(repair)

        caveat = QFrame()
        caveat.setObjectName("NoticeCard")
        caveat_layout = QVBoxLayout(caveat)
        caveat_layout.setContentsMargins(13, 10, 13, 10)
        caveat_title = QLabel("Граница утверждения")
        caveat_title.setObjectName("MetricLabel")
        caveat_text = QLabel(
            "известный development benchmark · один frozen multi-arm запуск · не blind holdout · "
            "source 156 и image 97 сохранены побайтово · только Qwen3.5-9B · repair проверен отдельно и дал null · "
            "единственный подтверждённый headline здесь: 240/274 = 0.875912"
        )
        caveat_text.setObjectName("Subtle")
        caveat_text.setWordWrap(True)
        caveat_layout.addWidget(caveat_title)
        caveat_layout.addWidget(caveat_text)
        right.addWidget(caveat)
        right.addStretch(1)
        body.addLayout(right, 2)

        root.addLayout(body, 1)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
