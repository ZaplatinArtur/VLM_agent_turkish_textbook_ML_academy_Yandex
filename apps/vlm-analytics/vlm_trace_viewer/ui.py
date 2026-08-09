from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QRect, QRectF, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .holdout80 import Holdout80Summary, load_holdout80_summary
from .model import PipelineStage, RunSummary, TaskTrace, TraceDataset
from .replay_aggregate import (
    FrozenReplayComparison,
    empty_milestone_schema,
    intermediate_timeline_schema,
)
from .selector_ui import SelectorWavePage
from .selector_wave import SelectorWaveSummary, build_active_selector_dataset
from .style import TRACE_STYLESHEET


def _clear_layout(layout: QVBoxLayout | QGridLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child is not None:
            _clear_layout(child)  # type: ignore[arg-type]


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 60:
        return f"{value:.1f} с"
    minutes, seconds = divmod(value, 60)
    return f"{int(minutes)}м {seconds:02.0f}с"


def _format_tokens(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value / 1000:.1f}k" if value >= 1000 else str(value)


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


class MetricCard(QFrame):
    def __init__(self, title: str, value: str, hint: str = "", accent: str = ""):
        super().__init__()
        self.setObjectName("MetricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        self.title_label = QLabel(title.upper())
        self.title_label.setObjectName("MetricLabel")
        value_label = QLabel(value)
        value_label.setObjectName("MetricValue")
        if accent:
            value_label.setStyleSheet(f"color: {accent};")
        hint_label = QLabel(hint)
        hint_label.setObjectName("MetricHint")
        hint_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        layout.addWidget(value_label)
        layout.addWidget(hint_label)


class AnswerCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("AnswerCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(2)
        self.title_label = QLabel(title.upper())
        self.title_label.setObjectName("MetricLabel")
        # Answers such as fill-blank maps can be several hundred characters long.
        # A QLabel silently clipped them inside the three-column comparison view;
        # keep the card compact but make the complete frozen answer scrollable.
        self.value = QTextBrowser()
        self.value.setPlainText("—")
        self.value.setObjectName("AnswerValue")
        self.value.setOpenExternalLinks(False)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value, 1)


class QuestionCanvas(QFrame):
    """Paint a real local question image, or reconstruct its saved OCR layout."""

    def __init__(self):
        super().__init__()
        self.setObjectName("Panel")
        self.setMinimumSize(390, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.task: TaskTrace | None = None
        self.pixmap = QPixmap()
        self.active_region = -1
        self.show_regions = True

    def set_task(self, task: TaskTrace) -> None:
        self.task = task
        self.active_region = 0 if task.attention_regions else -1
        self.pixmap = QPixmap(str(task.question_image)) if task.question_image else QPixmap()
        self.update()

    def set_active_region(self, index: int) -> None:
        self.active_region = index
        self.update()

    def set_regions_visible(self, visible: bool) -> None:
        self.show_regions = visible
        self.update()

    def _target_rect(self, canvas: QRectF, source_width: float, source_height: float) -> QRectF:
        ratio = min(canvas.width() / source_width, canvas.height() / source_height)
        width, height = source_width * ratio, source_height * ratio
        return QRectF(
            canvas.center().x() - width / 2,
            canvas.center().y() - height / 2,
            width,
            height,
        )

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect().adjusted(1, 1, -1, -1), QColor("#0b1621"))
        if not self.task:
            painter.setPen(QColor("#71879a"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Выберите задачу")
            return

        header = QRectF(18, 12, self.width() - 36, 50)
        painter.setPen(QColor("#eef7ff"))
        font = QFont("Segoe UI", 11, QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.drawText(QRectF(header.left(), header.top(), header.width(), 24), "Вход задачи")
        provenance = self.task.raw.get("provenance") or {}
        source_label = (
            str(provenance.get("question_image_origin") or "локальный исходник")
            if not self.pixmap.isNull()
            else "OCR-реконструкция · исходный файл не сохранён в bundle"
        )
        painter.setPen(QColor("#6f879b"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(
            QRectF(header.left(), header.top() + 25, header.width(), 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            source_label,
        )

        canvas = QRectF(18, 66, self.width() - 36, self.height() - 112)
        regions = self.task.attention_regions
        if not self.pixmap.isNull():
            source_width, source_height = self.pixmap.width(), self.pixmap.height()
            target = self._target_rect(canvas, source_width, source_height)
            painter.drawPixmap(target.toRect(), self.pixmap, self.pixmap.rect())
        else:
            if regions:
                source_width = regions[0].image_width
                source_height = regions[0].image_height
            else:
                source_width, source_height = 800, 1100
            target = self._target_rect(canvas, source_width, source_height)
            painter.setPen(QPen(QColor("#dbe4ea"), 1))
            painter.setBrush(QColor("#f6f4ed"))
            painter.drawRoundedRect(target, 4, 4)
            for index, region in enumerate(regions):
                x1, y1, x2, y2 = region.bbox
                rect = QRectF(
                    target.left() + (x1 / source_width) * target.width(),
                    target.top() + (y1 / source_height) * target.height(),
                    max(2.0, ((x2 - x1) / source_width) * target.width()),
                    max(2.0, ((y2 - y1) / source_height) * target.height()),
                )
                painter.setPen(QColor("#27343c"))
                point_size = max(5, min(9, int(rect.height() / 5)))
                painter.setFont(QFont("Segoe UI", point_size))
                block_text = region.text
                if "<img" in block_text.casefold():
                    match = re.search(r"src=[\"']([^\"']+)", block_text, flags=re.IGNORECASE)
                    block_text = f"IMAGE BLOCK\n{match.group(1) if match else 'embedded image'}"
                painter.drawText(
                    rect.adjusted(2, 1, -2, -1),
                    Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignTop,
                    block_text,
                )

        if self.show_regions and regions:
            for index, region in enumerate(regions):
                x1, y1, x2, y2 = region.bbox
                rect = QRectF(
                    target.left() + (x1 / source_width) * target.width(),
                    target.top() + (y1 / source_height) * target.height(),
                    max(2.0, ((x2 - x1) / source_width) * target.width()),
                    max(2.0, ((y2 - y1) / source_height) * target.height()),
                )
                if index == self.active_region:
                    painter.setPen(QPen(QColor("#25e0bd"), 2.4))
                    painter.setBrush(QColor(37, 224, 189, 42))
                else:
                    painter.setPen(QPen(QColor(80, 153, 196, 110), 0.8))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(rect, 3, 3)

        footer = QRectF(18, self.height() - 40, self.width() - 36, 32)
        painter.setPen(QColor("#6f879b"))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(
            footer,
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            "OCR-подсветка: эвристическое совпадение шага и блока; это не neural attention",
        )


class PipelineCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.stages: tuple[PipelineStage, ...] = ()
        self.setMinimumHeight(360)

    def set_stages(self, stages: tuple[PipelineStage, ...]) -> None:
        self.stages = stages
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0b1621"))
        if not self.stages:
            return
        margin, gap = 22.0, 18.0
        node_w = max(150.0, (self.width() - 2 * margin - 3 * gap) / 4)
        node_h = 108.0
        y_top, y_bottom = 38.0, 210.0
        positions: list[QRectF] = []
        for index in range(4):
            positions.append(QRectF(margin + index * (node_w + gap), y_top, node_w, node_h))
        for index in range(3):
            positions.append(QRectF(margin + (2 - index) * (node_w + gap), y_bottom, node_w, node_h))

        pen = QPen(QColor("#38566d"), 2)
        painter.setPen(pen)
        for index in range(len(positions) - 1):
            first, second = positions[index], positions[index + 1]
            if index == 3:
                start = QPointF(first.center().x(), first.bottom())
                end = QPointF(second.center().x(), second.top())
            else:
                start = QPointF(first.right(), first.center().y())
                end = QPointF(second.left(), second.center().y())
            painter.drawLine(start, end)
            painter.setBrush(QColor("#38566d"))
            painter.drawEllipse(end, 3.2, 3.2)

        colors = {
            "pass": ("#12362f", "#2c997d", "#6cf0c0"),
            "active": ("#17334f", "#367cc0", "#8fc7ff"),
            "skipped": ("#18232d", "#34495a", "#879cad"),
            "fail": ("#391b24", "#783848", "#ff98a4"),
            "neutral": ("#16232f", "#31485c", "#a7bacb"),
        }
        for index, (stage, rect) in enumerate(zip(self.stages, positions)):
            fill, border, accent = colors.get(stage.state, colors["neutral"])
            painter.setPen(QPen(QColor(border), 1.4))
            painter.setBrush(QColor(fill))
            painter.drawRoundedRect(rect, 11, 11)
            painter.setPen(QColor(accent))
            painter.setBrush(QColor(accent))
            painter.drawEllipse(QPointF(rect.left() + 18, rect.top() + 20), 5, 5)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.drawText(
                QRectF(rect.left() + 30, rect.top() + 9, rect.width() - 40, 20),
                Qt.AlignmentFlag.AlignVCenter,
                f"{index + 1:02d}",
            )
            painter.setPen(QColor("#f4f9fd"))
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
            painter.drawText(
                rect.adjusted(14, 32, -12, -48),
                Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignTop,
                stage.title,
            )
            painter.setPen(QColor("#8da2b5"))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(
                rect.adjusted(14, 66, -12, -8),
                Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignTop,
                stage.subtitle,
            )


class SubjectChart(QWidget):
    def __init__(self, summary: RunSummary):
        super().__init__()
        self.summary = summary
        self.setMinimumHeight(520)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0d1924"))
        rows = sorted(
            self.summary.by_subject.items(),
            key=lambda item: (-int(item[1].get("n") or 0), item[0]),
        )
        # Keep the sample count in its own column. Long subject names used to
        # paint through `n=...` on the 1080p metrics screen.
        left = min(285.0, max(205.0, self.width() * 0.28))
        top, row_h = 42.0, 35.0
        bar_w = max(120.0, self.width() - left - 68.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.setPen(QColor("#eaf3fa"))
        painter.drawText(QRectF(18, 8, self.width() - 36, 24), "Accuracy по предметам")
        for index, (subject, metrics) in enumerate(rows):
            y = top + index * row_h
            accuracy = float(metrics.get("new_accuracy") or 0.0)
            n = int(metrics.get("n") or 0)
            subject_font = QFont("Segoe UI", 8)
            painter.setFont(subject_font)
            painter.setPen(QColor("#9db0c0"))
            subject_width = max(60, int(left - 88))
            subject_label = QFontMetrics(subject_font).elidedText(
                subject,
                Qt.TextElideMode.ElideRight,
                subject_width,
            )
            painter.drawText(
                QRectF(18, y, subject_width, 22),
                Qt.AlignmentFlag.AlignVCenter,
                subject_label,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#162738"))
            painter.drawRoundedRect(QRectF(left, y + 4, bar_w, 14), 7, 7)
            color = QColor("#4be1c3") if accuracy >= 0.88 else QColor("#71aef5")
            if accuracy < 0.82:
                color = QColor("#f0ad62")
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(left, y + 4, bar_w * accuracy, 14), 7, 7)
            painter.setPen(QColor("#dce8f1"))
            painter.drawText(
                QRectF(left + bar_w + 8, y, 55, 22),
                Qt.AlignmentFlag.AlignVCenter,
                f"{accuracy:.1%}",
            )
            painter.setPen(QColor("#657b8e"))
            painter.drawText(
                QRectF(left - 48, y, 42, 22),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"n={n}",
            )


class LatencyChart(QWidget):
    def __init__(self, summary: RunSummary):
        super().__init__()
        self.summary = summary
        self.setMinimumHeight(210)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0d1924"))
        values = [
            ("p50", self.summary.latency_median_s, "#4be1c3"),
            ("p95", self.summary.latency_p95_s, "#71aef5"),
            ("max", self.summary.latency_max_s, "#f0ad62"),
        ]
        max_value = max((value or 0.0) for _, value, _ in values) or 1.0
        painter.setPen(QColor("#eef7fc"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(18, 10, self.width() - 36, 24),
            "Recorded inherited-anchor latency · not E2E",
        )
        for index, (label, value, color) in enumerate(values):
            y = 52 + index * 45
            painter.setPen(QColor("#8fa4b7"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(QRectF(20, y, 42, 24), Qt.AlignmentFlag.AlignVCenter, label)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#162738"))
            painter.drawRoundedRect(QRectF(66, y + 5, self.width() - 154, 14), 7, 7)
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(
                QRectF(66, y + 5, (self.width() - 154) * ((value or 0.0) / max_value), 14),
                7,
                7,
            )
            painter.setPen(QColor("#dce8f1"))
            painter.drawText(
                QRectF(self.width() - 80, y, 64, 24),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                _format_seconds(value),
            )


class SourceFirstProjectionChart(QWidget):
    """Artifact-replay projection; deliberately not presented as online speedup."""

    def __init__(self, summary: RunSummary):
        super().__init__()
        self.summary = summary
        self.setMinimumHeight(210)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0d1924"))
        painter.setPen(QColor("#eef7fc"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(18, 10, self.width() - 36, 24),
            "Source-first · artifact replay",
        )

        values = [
            ("shortcuts", self.summary.source_shortcut_rate, "#4be1c3"),
            (
                "anchor latency",
                self.summary.avoidable_recorded_latency_fraction,
                "#71aef5",
            ),
            (
                "anchor input",
                self.summary.avoidable_input_tokens_fraction,
                "#b99cff",
            ),
        ]
        bar_left = min(126.0, max(92.0, self.width() * 0.36))
        bar_width = max(48.0, self.width() - bar_left - 62.0)
        for index, (label, value, color) in enumerate(values):
            y = 45 + index * 35
            painter.setPen(QColor("#8fa4b7"))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(
                QRectF(18, y, bar_left - 24, 22),
                Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#162738"))
            painter.drawRoundedRect(QRectF(bar_left, y + 4, bar_width, 13), 6, 6)
            if value is not None:
                painter.setBrush(QColor(color))
                painter.drawRoundedRect(
                    QRectF(bar_left, y + 4, bar_width * max(0.0, min(value, 1.0)), 13),
                    6,
                    6,
                )
            painter.setPen(QColor("#dce8f1"))
            painter.drawText(
                QRectF(bar_left + bar_width + 6, y, 48, 22),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                "—" if value is None else f"{value:.1%}",
            )

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.setPen(QColor("#4be1c3"))
        painter.drawText(
            QRectF(18, 151, self.width() - 36, 18),
            f"{self.summary.answer_equivalent_shortcuts}/"
            f"{self.summary.source_shortcuts} shortcut-ответов эквивалентны V7",
        )
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#f0ad62"))
        caveat = "inherited anchor usage · не online/E2E speedup"
        if not self.summary.speed_source_lookup_cost_included:
            caveat += " · lookup cost исключён"
        painter.drawText(QRectF(18, 174, self.width() - 36, 20), caveat)


class HoldoutComparisonChart(QWidget):
    """Raw, corrected-inclusive and valid-task views on one fixed scale."""

    def __init__(self, summary: Holdout80Summary):
        super().__init__()
        self.summary = summary
        self.setMinimumHeight(250)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0d1924"))
        painter.setPen(QColor("#f4f9fd"))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(20, 14, self.width() - 40, 26),
            "Raw и protocol erratum — две отдельные проекции",
        )
        painter.setPen(QColor("#8194a7"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(
            QRectF(20, 41, self.width() - 40, 22),
            "официальная перепроверка не перезаписывает frozen raw",
        )

        rows = (
            ("RAW · FROZEN", self.summary.raw, "#f0ad62"),
            ("ERRATUM · 80", self.summary.erratum_inclusive, "#b99cff"),
            ("VALID · 79", self.summary.valid, "#4be1c3"),
        )
        label_width = min(178.0, max(128.0, self.width() * 0.25))
        bar_left = label_width + 22.0
        bar_width = max(150.0, self.width() - bar_left - 112.0)
        for index, (label, score, color) in enumerate(rows):
            y = 82.0 + index * 51.0
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QColor("#9db0c0"))
            painter.drawText(
                QRectF(20, y - 3, label_width - 10, 25),
                Qt.AlignmentFlag.AlignVCenter,
                label,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#162738"))
            painter.drawRoundedRect(QRectF(bar_left, y + 2, bar_width, 16), 8, 8)
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(
                QRectF(bar_left, y + 2, bar_width * score.accuracy, 16), 8, 8
            )
            painter.setPen(QColor("#eef7fc"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
            painter.drawText(
                QRectF(bar_left + bar_width + 10, y - 4, 88, 26),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{score.accuracy:.2%}",
            )
            painter.setPen(QColor("#71869a"))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(
                QRectF(bar_left, y + 22, bar_width, 19),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"{score.correct}/{score.total} · {score.label}",
            )


def _chronology_card(step: dict[str, Any]) -> QFrame:
    card = QFrame()
    card.setObjectName("ChronologyCard")
    layout = QHBoxLayout(card)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(9)
    number = QLabel(f"{int(step['step']):02d}")
    number.setObjectName("TimelineNumber")
    number.setAlignment(Qt.AlignmentFlag.AlignCenter)
    number.setFixedSize(34, 34)
    text_box = QVBoxLayout()
    text_box.setSpacing(1)
    title = QLabel(str(step["title"]))
    title.setObjectName("TimelineTitle")
    detail = QLabel(str(step["detail"]))
    detail.setObjectName("Tiny")
    detail.setWordWrap(True)
    text_box.addWidget(title)
    text_box.addWidget(detail)
    layout.addWidget(number)
    layout.addLayout(text_box, 1)
    return card


class Holdout80Page(QWidget):
    """Public aggregate only; deliberately contains no private holdout rows."""

    def __init__(self, summary: Holdout80Summary):
        super().__init__()
        self.summary = summary
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        heading = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Holdout80 · source evidence без подмены метрики")
        title.setObjectName("SectionTitle")
        subtitle = QLabel(
            "80 новых задач из уже доступных учебников: task-disjoint, но не book-disjoint. "
            "Здесь измеряется поиск и привязка официального источника — не качество ответа "
            "модели и не математическое reasoning."
        )
        subtitle.setObjectName("Subtle")
        subtitle.setWordWrap(True)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        heading.addLayout(title_box, 1)
        heading.addWidget(_badge("SEALED BEFORE GOLD", "good"))
        heading.addWidget(_badge("AUDIT · PASS", "info"))
        heading.addWidget(_badge("SOURCE ≠ QA", "warn"))
        root.addLayout(heading)

        cards = QHBoxLayout()
        cards.addWidget(
            MetricCard(
                "Raw protocol",
                f"{summary.raw.accuracy:.2%}",
                f"{summary.raw.correct}/{summary.raw.total} · immutable",
                "#f0ad62",
            )
        )
        cards.addWidget(
            MetricCard(
                "Erratum inclusive",
                f"{summary.erratum_inclusive.accuracy:.2%}",
                f"{summary.erratum_inclusive.correct}/{summary.erratum_inclusive.total} · denominator retained",
                "#b99cff",
            )
        )
        cards.addWidget(
            MetricCard(
                "Valid tasks",
                f"{summary.valid.accuracy:.2%}",
                f"{summary.valid.correct}/{summary.valid.total} · one invalid excluded",
                "#4be1c3",
            )
        )
        v7_correct = int(summary.v7_reference["correct"])
        v7_total = int(summary.v7_reference["total"])
        cards.addWidget(
            MetricCard(
                "V7 QA · separate",
                f"{v7_correct / v7_total:.2%}",
                f"{v7_correct}/{v7_total} · development replay",
                "#71aef5",
            )
        )
        root.addLayout(cards)

        upper = QSplitter(Qt.Orientation.Horizontal)
        chart_panel = QFrame()
        chart_panel.setObjectName("Panel")
        chart_layout = QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(4, 4, 4, 4)
        chart_layout.addWidget(HoldoutComparisonChart(summary))
        upper.addWidget(chart_panel)

        subject_panel = QFrame()
        subject_panel.setObjectName("Panel")
        subject_layout = QVBoxLayout(subject_panel)
        subject_layout.setContentsMargins(14, 12, 14, 12)
        subject_title = QLabel("Предметные срезы")
        subject_title.setObjectName("SectionTitle")
        subject_layout.addWidget(subject_title)
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        for column, label_text in enumerate(("ПРЕДМЕТ", "RAW", "VALID", "ЧТО ИЗМЕРЯЕМ")):
            label = QLabel(label_text)
            label.setObjectName("MetricLabel")
            grid.addWidget(label, 0, column)
        for row_index, row in enumerate(summary.subjects, start=1):
            name = QLabel(row.subject)
            name.setObjectName("TimelineTitle")
            raw = QLabel(f"{row.raw_correct}/{row.raw_total}")
            raw.setObjectName("Accent" if row.raw_correct == row.raw_total else "Danger")
            valid = QLabel(f"{row.valid_correct}/{row.valid_total}")
            valid.setObjectName("Success")
            measurement = QLabel(row.measurement)
            measurement.setObjectName("Tiny")
            measurement.setWordWrap(True)
            grid.addWidget(name, row_index, 0)
            grid.addWidget(raw, row_index, 1)
            grid.addWidget(valid, row_index, 2)
            grid.addWidget(measurement, row_index, 3)
        grid.setColumnStretch(3, 1)
        subject_layout.addLayout(grid)
        mcq = summary.mcq
        mcq_note = QLabel(
            f"MCQ: raw {mcq['raw_correct']}/{mcq['raw_total']} = "
            f"{mcq['raw_correct'] / mcq['raw_total']:.2%}  →  "
            f"official-key erratum {mcq['erratum_correct']}/{mcq['erratum_total']} = "
            f"{mcq['erratum_correct'] / mcq['erratum_total']:.2%}  →  "
            f"valid {mcq['valid_correct']}/{mcq['valid_total']} = 100%"
        )
        mcq_note.setObjectName("Subtle")
        mcq_note.setWordWrap(True)
        subject_layout.addWidget(mcq_note)
        scope_note = QLabel(
            "Math 20/20 — точность activity binding. Biology/Physics — exact-choice "
            "lookup по официальному ключу. Эти числа нельзя складывать с QA как одну "
            "reasoning-метрику."
        )
        scope_note.setObjectName("HoldoutCaveat")
        scope_note.setWordWrap(True)
        subject_layout.addWidget(scope_note)
        subject_layout.addStretch(1)
        upper.addWidget(subject_panel)
        upper.setSizes([950, 850])
        root.addWidget(upper)

        lower = QSplitter(Qt.Orientation.Horizontal)
        chronology_panel = QFrame()
        chronology_panel.setObjectName("Panel")
        chronology_layout = QVBoxLayout(chronology_panel)
        chronology_layout.setContentsMargins(12, 10, 12, 10)
        chronology_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        chronology_title = QLabel("Хронология целостности")
        chronology_title.setObjectName("SectionTitle")
        chronology_layout.addWidget(chronology_title)
        chronology_grid = QGridLayout()
        chronology_grid.setSpacing(7)
        for index, step in enumerate(summary.chronology):
            chronology_grid.addWidget(_chronology_card(step), index // 2, index % 2)
        chronology_layout.addLayout(chronology_grid)
        lower.addWidget(chronology_panel)

        audit_panel = QFrame()
        audit_panel.setObjectName("Panel")
        audit_layout = QVBoxLayout(audit_panel)
        audit_layout.setContentsMargins(14, 10, 14, 10)
        audit_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        audit_heading = QHBoxLayout()
        audit_title = QLabel("Errata и проверяемый след")
        audit_title.setObjectName("SectionTitle")
        audit_heading.addWidget(audit_title, 1)
        audit_heading.addWidget(_badge("RAW PRESERVED", "warn"))
        audit_layout.addLayout(audit_heading)
        for item in summary.errata:
            row = QFrame()
            row.setObjectName("ErratumCard")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 7, 10, 7)
            count = QLabel(str(item.affected_rows))
            count.setObjectName("ErratumCount")
            count.setAlignment(Qt.AlignmentFlag.AlignCenter)
            count.setFixedWidth(38)
            text_box = QVBoxLayout()
            text_box.setSpacing(1)
            finding = QLabel(item.finding)
            finding.setObjectName("TimelineTitle")
            finding.setWordWrap(True)
            treatment = QLabel(item.treatment)
            treatment.setObjectName("Tiny")
            treatment.setWordWrap(True)
            text_box.addWidget(finding)
            text_box.addWidget(treatment)
            row_layout.addWidget(count)
            row_layout.addLayout(text_box, 1)
            audit_layout.addWidget(row)

        hashes = summary.integrity
        hash_label = QLabel(
            "manifest  " + hashes["selection_manifest_sha256"][:16] + "…   "
            "gold  " + hashes["sealed_gold_sha256"][:16] + "…\n"
            "MCQ output  " + hashes["mcq_prediction_sha256"][:16] + "…   "
            "Math seal  " + hashes["math_output_seal_sha256"][:16] + "…\n"
            "public summary projection  " + summary.projection_sha256
        )
        hash_label.setObjectName("HashTrace")
        hash_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hash_label.setWordWrap(True)
        hash_label.setMinimumHeight(60)
        audit_layout.addWidget(hash_label)
        audit_scope = QLabel(
            "PASS: aggregate arithmetic, official-PDF key alignment и классификация errata. "
            "Приватные строки и ответы в приложение не встроены."
        )
        audit_scope.setObjectName("Success")
        audit_scope.setWordWrap(True)
        audit_layout.addWidget(audit_scope)
        lower.addWidget(audit_panel)
        lower.setSizes([1040, 760])
        root.addWidget(lower, 1)


class NineBMilestonesPage(QWidget):
    """Seven honest 9B comparison slots; empty until the full freeze chain exists."""

    def __init__(self, comparison: FrozenReplayComparison | None):
        super().__init__()
        self.comparison = comparison
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        heading = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("9B replay · семь честных milestones")
        title.setObjectName("SectionTitle")
        subtitle = QLabel(
            "Каждая цифра появится только после проверки exact aggregate hash, "
            "одного task set и Qwen3.5-9B model closure. Сила provenance показывается "
            "отдельно: legacy controls не выдаются за preregistered replay. Старые 27B "
            "stage scores сюда не импортируются."
        )
        subtitle.setObjectName("Subtle")
        subtitle.setWordWrap(True)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        heading.addLayout(title_box, 1)
        heading.addWidget(
            _badge(
                "7× HASH CLOSURE · PASS" if comparison else "NO FROZEN 9B AGGREGATES",
                "good" if comparison else "warn",
            )
        )
        root.addLayout(heading)

        panel = QFrame()
        panel.setObjectName("Panel")
        grid = QGridLayout(panel)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        loaded = (
            {item.milestone_id: item for item in comparison.milestones}
            if comparison
            else {}
        )
        for index, spec in enumerate(empty_milestone_schema(), start=1):
            milestone_id = spec["milestone_id"]
            result = loaded.get(milestone_id)
            row = QFrame()
            row.setObjectName("NoticeCard")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(11, 7, 11, 7)
            number = QLabel(f"{index:02d}")
            number.setObjectName("TimelineNumber")
            number.setAlignment(Qt.AlignmentFlag.AlignCenter)
            number.setFixedSize(34, 34)
            copy = QVBoxLayout()
            label = QLabel(spec["label"])
            label.setObjectName("TimelineTitle")
            detail_parts = [f"pipeline: {spec['pipeline']}"]
            if milestone_id == "query_active_crop_v2_9b":
                detail_parts.append("preregistered final 9B anchor")
            if result:
                detail_parts.append(result.provenance_status)
                if result.source_union["size"]:
                    detail_parts.append(
                        "source origin "
                        f"{result.source_union['replacements']} replacements / "
                        f"{result.source_union['confirmations']} confirmations"
                    )
                detail_parts.append(
                    "evaluator "
                    f"det {result.evaluator['deterministic_rows']} / "
                    f"image {result.evaluator['image_rows']}"
                )
                if "source_adjudicated_image_rows" in result.evaluator:
                    detail_parts.append(
                        "final image verdicts "
                        f"source-adjudicated {result.evaluator['source_adjudicated_image_rows']} / "
                        f"original 9B {result.evaluator['original_9b_judge_rows']}"
                    )
                active_delta = next(
                    (
                        item
                        for item in result.comparisons
                        if item["baseline_milestone_id"] == "query_active_crop_v2_9b"
                    ),
                    None,
                )
                if active_delta:
                    detail_parts.append(
                        f"vs ActiveCrop +{active_delta['fixes']} / "
                        f"−{active_delta['regressions']}"
                    )
            detail = QLabel(" · ".join(detail_parts))
            detail.setObjectName("Tiny")
            detail.setWordWrap(True)
            copy.addWidget(label)
            copy.addWidget(detail)
            value = QLabel(
                f"{result.correct}/{result.rows} · {result.accuracy:.4f}"
                if result
                else "—  awaiting frozen aggregate + pins"
            )
            value.setObjectName("Success" if result else "Subtle")
            progress = QProgressBar()
            progress.setFixedSize(310, 12)
            progress.setTextVisible(False)
            progress.setRange(0, result.rows if result else 1)
            progress.setValue(result.correct if result else 0)
            progress.setToolTip(
                f"{result.correct}/{result.rows} = {result.accuracy:.4%}"
                if result
                else "awaiting validated aggregate"
            )
            fill = "#4be1c3" if result and index >= 4 else "#71aef5"
            progress.setStyleSheet(
                "QProgressBar { background: #142536; border: 1px solid #244159; "
                "border-radius: 5px; } "
                f"QProgressBar::chunk {{ background: {fill}; border-radius: 4px; }}"
            )
            row_layout.addWidget(number)
            row_layout.addLayout(copy, 1)
            row_layout.addWidget(progress)
            row_layout.addWidget(value)
            grid.addWidget(row, index - 1, 0)
        root.addWidget(panel, 1)

        timeline = QFrame()
        timeline.setObjectName("NoticeCard")
        timeline_layout = QVBoxLayout(timeline)
        timeline_layout.setContentsMargins(12, 8, 12, 8)
        timeline_title = QLabel("Промежуточная provenance timeline")
        timeline_title.setObjectName("MetricLabel")
        timeline_layout.addWidget(timeline_title)
        intermediate = QLabel(
            "  →  ".join(item["label"] for item in intermediate_timeline_schema())
        )
        intermediate.setObjectName("Subtle")
        intermediate.setWordWrap(True)
        timeline_layout.addWidget(intermediate)
        timeline_note = QLabel(
            "Source V2/V4/V5 сохраняются для объяснения эволюции, но не подменяют "
            "семь основных comparison points. До frozen 9B aggregate здесь нет score."
        )
        timeline_note.setObjectName("Tiny")
        timeline_note.setWordWrap(True)
        timeline_layout.addWidget(timeline_note)
        root.addWidget(timeline)


class TaskDetail(QWidget):
    def __init__(self):
        super().__init__()
        self.task: TaskTrace | None = None
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(1150)
        self.play_timer.timeout.connect(self._advance_step)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        header = QFrame()
        header.setObjectName("Panel")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(15, 11, 15, 11)
        top = QHBoxLayout()
        self.title = QLabel("—")
        self.title.setObjectName("TaskTitle")
        self.subject_badge = _badge("—", "info")
        self.correct_badge = _badge("—")
        top.addWidget(self.title)
        top.addWidget(self.subject_badge)
        top.addStretch(1)
        top.addWidget(self.correct_badge)
        self.meta = QLabel("—")
        self.meta.setObjectName("Subtle")
        self.meta.setWordWrap(True)
        self.meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header_layout.addLayout(top)
        header_layout.addWidget(self.meta)
        root.addWidget(header)

        content = QSplitter(Qt.Orientation.Horizontal)
        question_panel = QWidget()
        question_layout = QVBoxLayout(question_panel)
        question_layout.setContentsMargins(0, 0, 0, 0)
        question_layout.setSpacing(6)
        self.canvas = QuestionCanvas()
        self.regions_checkbox = QCheckBox("показывать OCR bbox")
        self.regions_checkbox.setChecked(True)
        self.regions_checkbox.toggled.connect(self.canvas.set_regions_visible)
        question_layout.addWidget(self.canvas, 1)
        question_layout.addWidget(self.regions_checkbox)
        content.addWidget(question_panel)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_reasoning_tab(), "Рассуждение")
        self.tabs.addTab(self._build_route_tab(), "Маршрут")
        self.tabs.addTab(self._build_evidence_tab(), "Доказательство")
        self.tabs.addTab(self._build_compare_tab(), "Сравнение")
        self.tabs.addTab(self._build_raw_tab(), "JSON")
        content.addWidget(self.tabs)
        content.setStretchFactor(0, 5)
        content.setStretchFactor(1, 6)
        content.setSizes([620, 760])
        root.addWidget(content, 1)

    def _build_reasoning_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        notice = QFrame()
        notice.setObjectName("NoticeCard")
        notice_layout = QVBoxLayout(notice)
        notice_layout.setContentsMargins(11, 8, 11, 8)
        title = QLabel("ЗАПИСАННЫЙ TRACE")
        title.setObjectName("MetricLabel")
        self.reasoning_notice = QLabel(
            "показываются сохранённые solution_steps и reasoning. это объяснение ответа, "
            "а не доступ к скрытому chain-of-thought модели."
        )
        self.reasoning_notice.setObjectName("Subtle")
        self.reasoning_notice.setWordWrap(True)
        notice_layout.addWidget(title)
        notice_layout.addWidget(self.reasoning_notice)
        layout.addWidget(notice)

        controls = QHBoxLayout()
        self.play_button = QPushButton("▶ воспроизвести")
        self.play_button.setObjectName("Primary")
        self.play_button.clicked.connect(self._toggle_play)
        self.step_counter = QLabel("шаг —/—")
        self.step_counter.setObjectName("Subtle")
        controls.addWidget(self.play_button)
        controls.addWidget(self.step_counter)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.step_list = QListWidget()
        self.step_list.setObjectName("StepList")
        self.step_list.currentRowChanged.connect(self._step_selected)
        layout.addWidget(self.step_list, 3)
        raw_label = QLabel("полное сохранённое reasoning")
        raw_label.setObjectName("MetricLabel")
        self.reasoning_text = QTextBrowser()
        self.reasoning_text.setOpenExternalLinks(False)
        self.reasoning_text.setMinimumHeight(145)
        layout.addWidget(raw_label)
        layout.addWidget(self.reasoning_text, 2)
        return page

    def _build_route_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        title = QLabel("Как ответ прошёл через V7")
        title.setObjectName("SectionTitle")
        subtitle = QLabel(
            "зелёный — проверено; синий — composer вмешался; серый — ветка fail-closed и не меняла anchor."
        )
        subtitle.setObjectName("Subtle")
        subtitle.setWordWrap(True)
        self.pipeline = PipelineCanvas()
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.pipeline, 1)
        return page

    def _build_evidence_tab(self) -> QWidget:
        outer = QWidget()
        layout = QVBoxLayout(outer)
        layout.setContentsMargins(12, 12, 12, 12)
        self.evidence_status = QLabel("—")
        self.evidence_status.setObjectName("SectionTitle")
        self.evidence_summary = QLabel("—")
        self.evidence_summary.setObjectName("Subtle")
        self.evidence_summary.setWordWrap(True)
        layout.addWidget(self.evidence_status)
        layout.addWidget(self.evidence_summary)

        card = QFrame()
        card.setObjectName("EvidenceCard")
        self.evidence_grid = QGridLayout(card)
        self.evidence_grid.setContentsMargins(13, 12, 13, 12)
        self.evidence_grid.setHorizontalSpacing(18)
        self.evidence_grid.setVerticalSpacing(8)
        layout.addWidget(card)

        checks_title = QLabel("Детерминированные проверки")
        checks_title.setObjectName("MetricLabel")
        layout.addWidget(checks_title)
        checks_scroll = QScrollArea()
        checks_scroll.setWidgetResizable(True)
        checks_scroll.setFrameShape(QFrame.Shape.NoFrame)
        checks_widget = QWidget()
        self.checks_layout = QVBoxLayout(checks_widget)
        self.checks_layout.setContentsMargins(0, 0, 0, 0)
        self.checks_layout.setSpacing(5)
        checks_scroll.setWidget(checks_widget)
        layout.addWidget(checks_scroll, 1)
        button_row = QHBoxLayout()
        self.source_button = QPushButton("Открыть официальный источник")
        self.source_button.clicked.connect(self._open_source)
        self.copy_trace_button = QPushButton("Копировать fingerprint")
        self.copy_trace_button.setObjectName("Ghost")
        self.copy_trace_button.clicked.connect(self._copy_trace)
        button_row.addWidget(self.source_button)
        button_row.addWidget(self.copy_trace_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        return outer

    def _build_compare_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        cards = QHBoxLayout()
        self.anchor_card = AnswerCard("Pre-V7 composite anchor")
        self.challenger_card = AnswerCard("Deterministic source challenger")
        self.final_card = AnswerCard("Final · origin pending")
        cards.addWidget(self.anchor_card)
        cards.addWidget(self.challenger_card)
        cards.addWidget(self.final_card)
        layout.addLayout(cards)
        self.composer_explanation = QLabel("—")
        self.composer_explanation.setWordWrap(True)
        self.composer_explanation.setObjectName("Subtle")
        layout.addWidget(self.composer_explanation)
        label = QLabel("Анонимные кандидаты reasoning-ансамбля")
        label.setObjectName("MetricLabel")
        self.candidate_combo = QComboBox()
        self.candidate_combo.currentIndexChanged.connect(self._candidate_changed)
        self.candidate_text = QTextBrowser()
        layout.addWidget(label)
        layout.addWidget(self.candidate_combo)
        layout.addWidget(self.candidate_text, 1)
        return page

    def _build_raw_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        label = QLabel("Сырые joined-артефакты этой задачи")
        label.setObjectName("SectionTitle")
        self.raw_text = QTextBrowser()
        self.raw_text.setFont(QFont("Cascadia Mono", 8))
        layout.addWidget(label)
        layout.addWidget(self.raw_text, 1)
        return page

    def set_task(self, task: TaskTrace) -> None:
        self.task = task
        self.play_timer.stop()
        self.play_button.setText("▶ воспроизвести")
        self.title.setText(task.task_id)
        self.subject_badge.setText(task.subject)
        self.correct_badge.setText("CORRECT" if task.correct else "INCORRECT")
        self.correct_badge.setObjectName("BadgeGood" if task.correct else "BadgeBad")
        self.correct_badge.style().unpolish(self.correct_badge)
        self.correct_badge.style().polish(self.correct_badge)
        self.meta.setText(
            f"BASE ROW MODEL · {task.base_row_model or 'metadata absent'}    |    "
            f"FINAL ORIGIN · {task.final_origin}\n"
            f"RECORDED INHERITED-ANCHOR USAGE · {_format_seconds(task.latency_s)} · "
            f"in {_format_tokens(task.input_tokens)} / out {_format_tokens(task.output_tokens)} "
            "· NOT E2E"
        )
        self.canvas.set_task(task)
        self.pipeline.set_stages(task.pipeline)
        self._fill_reasoning(task)
        self._fill_evidence(task)
        self._fill_comparison(task)
        self.raw_text.setPlainText(json.dumps(task.raw, ensure_ascii=False, indent=2))

    def _fill_reasoning(self, task: TaskTrace) -> None:
        selector = task.raw.get("selector_v1_2")
        if isinstance(selector, dict):
            self.reasoning_notice.setText(
                f"Источник reasoning: {task.reasoning_origin}; это неизменённый Source V7 trace. "
                "Baseline Selector v1.2 выбрал сохранённый кандидат после этого trace по "
                "hash-bound согласию трёх групп. Для selector не показывается и не "
                "выдумывается отдельный chain-of-thought."
            )
        else:
            self.reasoning_notice.setText(
                f"Источник reasoning: {task.reasoning_origin}. Финальный ответ имеет отдельное "
                f"происхождение: {task.final_origin}. Показанный trace не является скрытым "
                "chain-of-thought и не доказывает, что base model породила source replacement."
            )
        self.step_list.clear()
        steps = task.solution_steps or ("у сохранённого ответа нет отдельных solution_steps",)
        for index, step in enumerate(steps, start=1):
            item = QListWidgetItem(f"{index:02d}   {step}")
            item.setToolTip(step)
            self.step_list.addItem(item)
        self.reasoning_text.setPlainText(task.reasoning or "reasoning не сохранён")
        self.step_list.setCurrentRow(0)

    def _fill_evidence(self, task: TaskTrace) -> None:
        source = task.source
        if source.accepted:
            self.evidence_status.setText("Сильный source certificate принят")
            self.evidence_status.setStyleSheet("color: #67e8b2;")
            self.evidence_summary.setText(
                "ответ связан одновременно с входной задачей, конкретной страницей PDF и областью ключа. "
                "composer может использовать это доказательство."
            )
        else:
            self.evidence_status.setText("Source certificate отсутствует")
            self.evidence_status.setStyleSheet("color: #94a8b9;")
            self.evidence_summary.setText(
                "ветка источников не предъявила достаточное доказательство. V7 сработал fail-closed "
                "и сохранил reasoning anchor."
            )

        _clear_layout(self.evidence_grid)
        fields = (
            ("document", source.document_name or "—"),
            ("record", source.record_id or "—"),
            ("question", str(source.question_number) if source.question_number is not None else "numberless"),
            ("matched page", str(source.matched_page or "—")),
            ("key page", str(source.key_page or "—")),
            ("key bbox", str(list(source.key_bbox)) if source.key_bbox else "—"),
            ("page coverage", f"{source.page_coverage:.3f}" if source.page_coverage is not None else "—"),
            ("page margin", f"{source.page_margin:.3f}" if source.page_margin is not None else "—"),
            ("verifier", source.verifier or "—"),
            ("trace", (source.trace_fingerprint[:18] + "…") if source.trace_fingerprint else "—"),
        )
        for row, (key, value) in enumerate(fields):
            key_label = QLabel(key.upper())
            key_label.setObjectName("MetricLabel")
            value_label = QLabel(value)
            value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value_label.setWordWrap(True)
            self.evidence_grid.addWidget(key_label, row, 0)
            self.evidence_grid.addWidget(value_label, row, 1)

        _clear_layout(self.checks_layout)
        for name, passed in source.checks:
            row = QFrame()
            row.setObjectName("NoticeCard")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(9, 5, 9, 5)
            icon = QLabel("✓" if passed else "×")
            icon.setObjectName("Success" if passed else "Danger")
            text = QLabel(name.replace("_", " "))
            text.setWordWrap(True)
            row_layout.addWidget(icon)
            row_layout.addWidget(text, 1)
            self.checks_layout.addWidget(row)
        if not source.checks:
            empty = QLabel("для этой задачи source-проверки не запускались")
            empty.setObjectName("Subtle")
            self.checks_layout.addWidget(empty)
        self.checks_layout.addStretch(1)
        self.source_button.setEnabled(bool(source.public_locator))
        self._reset_source_button_label()
        self.copy_trace_button.setEnabled(bool(source.trace_fingerprint))

    def _fill_comparison(self, task: TaskTrace) -> None:
        selector = task.raw.get("selector_v1_2")
        if isinstance(selector, dict):
            self.anchor_card.title_label.setText("SOURCE V7 INPUT TO SELECTOR")
            self.challenger_card.title_label.setText("UNANIMOUS SELECTOR CANDIDATE")
            self.anchor_card.value.setText(
                str(selector.get("source_v7_final_answer") or "—")
            )
            self.challenger_card.value.setText(task.final_answer or "—")
        else:
            self.anchor_card.title_label.setText("PRE-V7 COMPOSITE ANCHOR")
            self.challenger_card.title_label.setText("DETERMINISTIC SOURCE CHALLENGER")
            self.anchor_card.value.setText(task.anchor_answer or "—")
            self.challenger_card.value.setText(task.challenger_answer or "ABSTAIN")
        self.final_card.title_label.setText(f"FINAL · {task.final_origin}".upper())
        self.final_card.value.setText(task.final_answer or "—")
        if isinstance(selector, dict):
            self.composer_explanation.setText(
                "selector заменил Source V7 answer только после единогласия structural, "
                "native и parallel groups; это выбор между сохранёнными кандидатами, "
                "не новый source lookup и не новое reasoning. "
                f"route = {selector.get('route')}; reason = {selector.get('reason')}"
            )
        elif task.decision_action == "replace_anchor":
            self.composer_explanation.setText(
                "финал создан deterministic source layer, а не моделью из поля base row: "
                "challenger прошёл строгую привязку к официальному PDF, странице и ключу. "
                "reason = " + task.decision_reason
            )
        elif task.has_certificate:
            self.composer_explanation.setText(
                "сертификат подтвердил тот же ответ, поэтому байты anchor были сохранены. "
                "reason = " + task.decision_reason
            )
        else:
            self.composer_explanation.setText(
                "доказательства для замены не было; fail-closed composer оставил anchor. "
                "reason = " + task.decision_reason
            )
        self.candidate_combo.blockSignals(True)
        self.candidate_combo.clear()
        for candidate in task.candidates:
            self.candidate_combo.addItem(
                f"{candidate.get('candidate_id')} · answer {candidate.get('final_answer')}"
            )
        self.candidate_combo.blockSignals(False)
        self._candidate_changed(0)

    def _candidate_changed(self, index: int) -> None:
        if not self.task or index < 0 or index >= len(self.task.candidates):
            self.candidate_text.setPlainText("кандидаты не сохранены")
            return
        candidate = self.task.candidates[index]
        evidence = "\n".join(f"• {line}" for line in candidate.get("evidence") or [])
        self.candidate_text.setPlainText(
            f"answer: {candidate.get('final_answer')}\n\n"
            f"reasoning:\n{candidate.get('reasoning') or '—'}\n\n"
            f"evidence:\n{evidence or '—'}"
        )

    def _step_selected(self, index: int) -> None:
        if not self.task:
            return
        count = self.step_list.count()
        self.step_counter.setText(f"шаг {index + 1}/{count}" if index >= 0 else f"шаг —/{count}")
        regions = self.task.attention_regions
        if not regions:
            self.canvas.set_active_region(-1)
            return
        text = self.step_list.item(index).text() if index >= 0 else ""
        tokens = {
            token.casefold()
            for token in re.findall(r"[^\W\d_]{4,}", text, flags=re.UNICODE)
        }
        best_index, best_score = index % len(regions), 0
        for region_index, region in enumerate(regions):
            region_tokens = {
                token.casefold()
                for token in re.findall(r"[^\W\d_]{4,}", region.text, flags=re.UNICODE)
            }
            score = len(tokens & region_tokens)
            if score > best_score:
                best_index, best_score = region_index, score
        self.canvas.set_active_region(best_index)

    def _toggle_play(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("▶ воспроизвести")
        else:
            if self.step_list.currentRow() >= self.step_list.count() - 1:
                self.step_list.setCurrentRow(0)
            self.play_timer.start()
            self.play_button.setText("Ⅱ пауза")

    def _advance_step(self) -> None:
        next_row = self.step_list.currentRow() + 1
        if next_row >= self.step_list.count():
            self.play_timer.stop()
            self.play_button.setText("▶ воспроизвести")
            return
        self.step_list.setCurrentRow(next_row)

    def _open_source(self) -> None:
        if not self.task or not self.task.source.public_locator:
            return
        locator = self.task.source.public_locator
        if locator.startswith(("https://", "http://")):
            QDesktopServices.openUrl(QUrl(locator))
        else:
            QGuiApplication.clipboard().setText(locator)
            self.source_button.setText("locator скопирован")
            QTimer.singleShot(1600, self._reset_source_button_label)

    def _reset_source_button_label(self) -> None:
        locator = self.task.source.public_locator if self.task else ""
        if locator.startswith(("https://", "http://")):
            self.source_button.setText("Открыть официальный источник")
        elif locator:
            self.source_button.setText("Копировать locator источника")
        else:
            self.source_button.setText("Источник недоступен")

    def _copy_trace(self) -> None:
        if self.task:
            QGuiApplication.clipboard().setText(self.task.source.trace_fingerprint)


class TraceExplorer(QWidget):
    def __init__(
        self,
        dataset: TraceDataset,
        selector_summary: SelectorWaveSummary | None = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.filtered_tasks: list[TaskTrace] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        metrics = QHBoxLayout()
        if selector_summary is not None:
            metrics.addWidget(
                MetricCard(
                    "Selector v1.2 · active",
                    f"{selector_summary.accuracy:.4%}",
                    f"{selector_summary.correct}/{selector_summary.rows} · audited dev",
                    "#4be1c3",
                )
            )
            metrics.addWidget(
                MetricCard(
                    "Math",
                    f"{selector_summary.math_correct / selector_summary.math_rows:.2%}",
                    f"{selector_summary.math_correct}/{selector_summary.math_rows}",
                    "#71aef5",
                )
            )
            metrics.addWidget(
                MetricCard(
                    "History",
                    f"{selector_summary.history_correct / selector_summary.history_rows:.2%}",
                    f"{selector_summary.history_correct}/{selector_summary.history_rows}",
                    "#b99cff",
                )
            )
            metrics.addWidget(
                MetricCard(
                    "Deterministic split",
                    f"{selector_summary.deterministic_correct}/{selector_summary.deterministic_rows}",
                    f"image {selector_summary.image_correct}/{selector_summary.image_rows}",
                    "#f0ad62",
                )
            )
        else:
            metrics.addWidget(
                MetricCard(
                    "Source V7 lineage accuracy",
                    f"{dataset.summary.accuracy:.2%}",
                    f"{dataset.summary.correct}/{dataset.summary.rows} · {dataset.summary.pipeline_provenance}",
                    "#4be1c3",
                )
            )
            metrics.addWidget(
                MetricCard(
                    "Math",
                    f"{dataset.summary.math_accuracy:.2%}",
                    f"{dataset.summary.math_correct}/{dataset.summary.math_rows}",
                    "#71aef5",
                )
            )
            metrics.addWidget(
                MetricCard(
                    "Source certificates",
                    str(dataset.summary.source_certificates),
                    "strong · input/page/key bound",
                    "#b99cff",
                )
            )
            metrics.addWidget(
                MetricCard(
                    "Inherited anchor p50",
                    _format_seconds(dataset.summary.latency_median_s),
                    f"recorded, not E2E · p95 {_format_seconds(dataset.summary.latency_p95_s)}",
                    "#f0ad62",
                )
            )
        root.addLayout(metrics)

        if selector_summary is not None:
            scope = QLabel(
                "ACTIVE VIEW · frozen selector overlay: +2 fixes / 0 regressions over "
                "Source V7. Only val_0089 and val_0251 changed; their reasoning panel "
                "remains the recorded Source V7 trace, not selector chain-of-thought."
            )
            scope.setObjectName("Subtle")
            scope.setWordWrap(True)
            root.addWidget(scope)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        sidebar = QFrame()
        sidebar.setObjectName("Panel")
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(12, 12, 12, 12)
        title = QLabel("Задачи")
        title.setObjectName("SectionTitle")
        self.search = QLineEdit()
        self.search.setPlaceholderText("task id, предмет, ответ, источник…")
        self.subject = QComboBox()
        self.subject.addItem("Все предметы")
        self.subject.addItems(sorted({task.subject for task in dataset.tasks}))
        self.correctness = QComboBox()
        self.correctness.addItems(("Все результаты", "Только correct", "Только incorrect"))
        self.certificate = QComboBox()
        self.certificate.addItems(("Любой source status", "Есть certificate", "Нет certificate"))
        self.action = QComboBox()
        self.action.addItems(
            (
                "Любое действие",
                "Source replacement",
                "Selector replacement",
                "Anchor kept",
            )
        )
        filters = QGridLayout()
        filters.setContentsMargins(0, 0, 0, 0)
        filters.addWidget(self.subject, 0, 0, 1, 2)
        filters.addWidget(self.correctness, 1, 0)
        filters.addWidget(self.certificate, 1, 1)
        filters.addWidget(self.action, 2, 0, 1, 2)
        self.count_label = QLabel("—")
        self.count_label.setObjectName("Subtle")
        self.task_list = QListWidget()
        self.task_list.setObjectName("TaskList")
        self.task_list.currentRowChanged.connect(self._select_task)
        side.addWidget(title)
        side.addWidget(self.search)
        side.addLayout(filters)
        side.addWidget(self.count_label)
        side.addWidget(self.task_list, 1)
        sidebar.setMinimumWidth(330)
        sidebar.setMaximumWidth(470)
        splitter.addWidget(sidebar)

        self.detail = TaskDetail()
        splitter.addWidget(self.detail)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([365, 1450])
        root.addWidget(splitter, 1)

        for widget in (self.search,):
            widget.textChanged.connect(self._apply_filters)
        for widget in (self.subject, self.correctness, self.certificate, self.action):
            widget.currentIndexChanged.connect(self._apply_filters)
        self._apply_filters()

    def _apply_filters(self) -> None:
        query = self.search.text().strip().casefold()
        subject = self.subject.currentText()
        correctness = self.correctness.currentIndex()
        certificate = self.certificate.currentIndex()
        action = self.action.currentIndex()
        selected_id = None
        item = self.task_list.currentItem()
        if item:
            selected_id = item.data(Qt.ItemDataRole.UserRole)

        filtered: list[TaskTrace] = []
        for task in self.dataset.tasks:
            haystack = " ".join(
                (
                    task.task_id,
                    task.subject,
                    task.final_answer,
                    task.source.document_name,
                    task.source.record_id,
                )
            ).casefold()
            if query and query not in haystack:
                continue
            if subject != "Все предметы" and task.subject != subject:
                continue
            if correctness == 1 and not task.correct:
                continue
            if correctness == 2 and task.correct:
                continue
            if certificate == 1 and not task.has_certificate:
                continue
            if certificate == 2 and task.has_certificate:
                continue
            if action == 1 and task.decision_action != "replace_anchor":
                continue
            if action == 2 and task.decision_action != "selector_replace":
                continue
            if action == 3 and task.decision_action in {
                "replace_anchor",
                "selector_replace",
            }:
                continue
            filtered.append(task)

        self.filtered_tasks = filtered
        self.task_list.blockSignals(True)
        self.task_list.clear()
        selected_row = 0
        for row, task in enumerate(filtered):
            state = "✓" if task.correct else "×"
            cert = (
                "SELECTOR"
                if task.decision_action == "selector_replace"
                else "CERT" if task.has_certificate else "ANCHOR"
            )
            answer = task.final_answer.replace("\n", " ")
            if len(answer) > 44:
                answer = answer[:41] + "…"
            item = QListWidgetItem(
                f"{state}  {task.task_id}   ·   {task.subject}\n     {cert}   →   {answer}"
            )
            item.setData(Qt.ItemDataRole.UserRole, task.task_id)
            item.setToolTip(
                f"{task.task_id}\n{task.subject}\nanswer: {task.final_answer}\n"
                f"source: {task.source.record_id or 'none'}"
            )
            item.setSizeHint(item.sizeHint().expandedTo(QRect(0, 0, 0, 58).size()))
            item.setForeground(QColor("#dce9f3" if task.correct else "#ff9ba7"))
            self.task_list.addItem(item)
            if task.task_id == selected_id:
                selected_row = row
        self.task_list.blockSignals(False)
        self.count_label.setText(f"показано {len(filtered)} из {len(self.dataset.tasks)}")
        if filtered:
            self.task_list.setCurrentRow(min(selected_row, len(filtered) - 1))

    def _select_task(self, row: int) -> None:
        if 0 <= row < len(self.filtered_tasks):
            self.detail.set_task(self.filtered_tasks[row])

    def select_task_id(self, task_id: str) -> bool:
        if self.search.text():
            self.search.clear()
        for row, task in enumerate(self.filtered_tasks):
            if task.task_id == task_id:
                self.task_list.setCurrentRow(row)
                return True
        return False


class MetricsPage(QWidget):
    def __init__(
        self,
        dataset: TraceDataset,
        selector_summary: SelectorWaveSummary | None = None,
    ):
        super().__init__()
        summary = dataset.summary
        is_nine_b = summary.pipeline_provenance.startswith("9B ")
        is_selector = is_nine_b and selector_summary is not None
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)
        heading = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel(
            (
                "Baseline Selector v1.2 · active audited all-9B analytics"
                if is_selector
                else "9B Source V7 · canonical lineage replay"
            )
            if is_nine_b
            else "V7 reference · META-27B anchor + deterministic source layers"
        )
        title.setObjectName("SectionTitle")
        subtitle = QLabel(
            (
                (
                    f"{selector_summary.correct}/{selector_summary.rows} — frozen one-shot "
                    "development selector поверх полностью завершённого Source V7 "
                    "238/274: +2 исправления, 0 откатов. Canonical lineage не переписана."
                )
                if is_selector
                else (
                    f"{summary.correct}/{summary.rows} — отдельный Qwen3.5-9B replay с exact "
                    "benchmark/model/hash closure. Evaluator split и deterministic source "
                    "origins не свёрнуты в model gain."
                )
            )
            if is_nine_b
            else (
                f"{summary.correct}/{summary.rows} — archived/reference development replay. "
                "Base-row model metadata и final origin разделены: source replacements не "
                "приписываются 27B. Это не unseen holdout и не production accuracy."
            )
        )
        subtitle.setObjectName("Subtle")
        subtitle.setWordWrap(True)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        heading.addLayout(title_box, 1)
        heading.addWidget(
            _badge(
                (
                    "SELECTOR v1.2 · ACTIVE 240/274"
                    if is_selector
                    else "9B · PROFILE-BOUND REPLAY"
                )
                if is_nine_b
                else "ARCHIVED 27B REFERENCE",
                "good" if is_nine_b else "warn",
            )
        )
        root.addLayout(heading)

        cards = QHBoxLayout()
        if is_selector:
            cards.addWidget(
                MetricCard(
                    "Overall · active",
                    f"{selector_summary.accuracy:.4%}",
                    f"{selector_summary.correct}/{selector_summary.rows}",
                    "#4be1c3",
                )
            )
            cards.addWidget(
                MetricCard(
                    "Math",
                    f"{selector_summary.math_correct / selector_summary.math_rows:.2%}",
                    f"{selector_summary.math_correct}/{selector_summary.math_rows}",
                    "#71aef5",
                )
            )
            cards.addWidget(
                MetricCard(
                    "History",
                    f"{selector_summary.history_correct / selector_summary.history_rows:.2%}",
                    f"{selector_summary.history_correct}/{selector_summary.history_rows}",
                    "#b99cff",
                )
            )
            cards.addWidget(
                MetricCard(
                    "Deterministic",
                    f"{selector_summary.deterministic_correct}/{selector_summary.deterministic_rows}",
                    f"{selector_summary.deterministic_correct / selector_summary.deterministic_rows:.2%}",
                    "#f0ad62",
                )
            )
            cards.addWidget(
                MetricCard(
                    "Image judge",
                    f"{selector_summary.image_correct}/{selector_summary.image_rows}",
                    f"{selector_summary.image_correct / selector_summary.image_rows:.2%}",
                    "#e8899c",
                )
            )
        else:
            cards.addWidget(MetricCard("Overall", f"{summary.accuracy:.2%}", f"{summary.correct}/{summary.rows}", "#4be1c3"))
            cards.addWidget(MetricCard("Math", f"{summary.math_accuracy:.2%}", f"{summary.math_correct}/{summary.math_rows}", "#71aef5"))
            cards.addWidget(
                MetricCard(
                    "vs Active Crop" if is_nine_b else "vs page-RAG",
                    f"+{(summary.accuracy-summary.baseline_accuracy):.1%}",
                    f"baseline {summary.baseline_accuracy:.2%}",
                    "#b99cff",
                )
            )
            delta_v6 = summary.direct_gain_vs_v6 + summary.evaluator_corrections_vs_v6
            cards.addWidget(
                MetricCard(
                    "Source origins" if is_nine_b else "vs V6",
                    (
                        f"{summary.answer_overrides} / "
                        f"{summary.source_certificates - summary.answer_overrides}"
                        if is_nine_b
                        else f"+{delta_v6}"
                    ),
                    (
                        "replacements / confirmations"
                        if is_nine_b
                        else f"+{summary.direct_gain_vs_v6} answer · +{summary.evaluator_corrections_vs_v6} eval"
                    ),
                    "#f0ad62",
                )
            )
        root.addLayout(cards)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        subject_panel = QFrame()
        subject_panel.setObjectName("Panel")
        subject_panel.setMinimumWidth(720)
        subject_layout = QVBoxLayout(subject_panel)
        subject_layout.setContentsMargins(4, 4, 4, 4)
        subject_layout.addWidget(SubjectChart(summary))
        splitter.addWidget(subject_panel)

        right = QWidget()
        right.setMinimumWidth(680)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        run_profile = QSplitter(Qt.Orientation.Horizontal)
        latency_panel = QFrame()
        latency_panel.setObjectName("Panel")
        latency_layout = QVBoxLayout(latency_panel)
        latency_layout.setContentsMargins(4, 4, 4, 4)
        latency_layout.addWidget(LatencyChart(summary))
        run_profile.addWidget(latency_panel)

        speed_panel = QFrame()
        speed_panel.setObjectName("Panel")
        speed_layout = QVBoxLayout(speed_panel)
        speed_layout.setContentsMargins(4, 4, 4, 4)
        speed_layout.addWidget(SourceFirstProjectionChart(summary))
        run_profile.addWidget(speed_panel)
        run_profile.setSizes([340, 340])
        right_layout.addWidget(run_profile)

        interpretation = QFrame()
        interpretation.setObjectName("NoticeCard")
        info = QVBoxLayout(interpretation)
        info.setContentsMargins(14, 12, 14, 12)
        label = QLabel(
            (
                "Что изменил selector и что он не доказывает"
                if is_selector
                else "Evaluator/origin split 9B V7"
            )
            if is_nine_b
            else "Provenance: что реально изменилось между V6 и V7"
        )
        label.setObjectName("SectionTitle")
        text = QLabel(
            (
                (
                    "• active aggregate: 240/274 = 87.5912%; Math 109/139, "
                    "History 10/10, deterministic 158/177, image 82/97.\n"
                    "• selector заменил только val_0089 A→D и val_0251 A→B; "
                    "остальные 272 строки сохранены byte-exact.\n"
                    "• Source V7 = 238/274 остаётся завершением канонической "
                    "семиступенчатой lineage, а не активным headline.\n"
                    "• это известный development benchmark и один frozen one-shot arm; "
                    "selector provenance не является новым reasoning или source lookup."
                )
                if is_selector
                else (
                    f"• {summary.source_certificates} строк входят в проверенный source union; "
                    f"{summary.answer_overrides} финалов имеют deterministic source origin.\n"
                    f"• cumulative image-verdict split: source-adjudicated "
                    f"{summary.source_adjudicated_image_rows} / original ActiveCrop 9B "
                    f"{summary.original_9b_judge_rows}. Original означает byte-identical с исходным "
                    "9B judge, а не только копию immediate base.\n"
                    "• deterministic evaluator и image-judge остаются разными ветками; их "
                    "вклад не называется чистым model gain.\n"
                    "• latency/tokens — inherited Active Crop anchor usage, не полный E2E."
                )
            )
            if is_nine_b
            else (
                f"• {summary.direct_gain_vs_v6} задача получила новый правильный solver-answer из официального источника.\n"
                f"• {summary.evaluator_corrections_vs_v6} неизменённых правильных ответа получили исправленный verdict.\n"
                f"• {summary.source_certificates} сильных сертификатов доступны в trace; layered composer заменил anchor "
                f"в {summary.answer_overrides} задачах.\n"
                "• latency и tokens относятся к записанной inherited-anchor строке; lookup, "
                "certificate, composer и полный E2E wall clock в них не входят.\n"
                "• весь набор уже изучался при разработке, поэтому экран явно маркирован как development replay."
            )
        )
        text.setObjectName("Subtle")
        text.setWordWrap(True)
        info.addWidget(label)
        info.addWidget(text)
        right_layout.addWidget(interpretation)

        limitations = QFrame()
        limitations.setObjectName("NoticeCard")
        lim_layout = QVBoxLayout(limitations)
        lim_layout.setContentsMargins(14, 12, 14, 12)
        lim_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        lim_title = QLabel(
            "Ограничения replay"
            if is_nine_b
            else "Ограничения из V7_POST_SCORE_RESULT"
        )
        lim_title.setObjectName("MetricLabel")
        lim_layout.addWidget(lim_title)
        caveats = QLabel("\n".join("• " + item for item in summary.limitations))
        caveats.setObjectName("Subtle")
        caveats.setWordWrap(True)
        caveats.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        lim_layout.addWidget(caveats)
        right_layout.addWidget(limitations)
        right_layout.addStretch(1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1040, 820])
        root.addWidget(splitter, 1)


class TraceViewerWindow(QMainWindow):
    def __init__(
        self,
        dataset: TraceDataset,
        holdout80: Holdout80Summary | None = None,
        nine_b_comparison: FrozenReplayComparison | None = None,
        selector_summary: SelectorWaveSummary | None = None,
        *,
        active_dataset: str = "archived-27b-v7",
        qa_reference_summary: RunSummary | None = None,
    ):
        super().__init__()
        self.source_v7_dataset = dataset
        self.active_selector = (
            active_dataset == "nine-b-v7" and selector_summary is not None
        )
        if self.active_selector:
            dataset = build_active_selector_dataset(dataset, selector_summary)
        self.dataset = dataset
        self.holdout80 = holdout80 or load_holdout80_summary()
        self.nine_b_comparison = nine_b_comparison
        self.selector_summary = selector_summary
        self.active_dataset = active_dataset
        self.setWindowTitle(
            (
                "VLM Trace · 9B Baseline Selector v1.2 Evidence OS"
                if self.active_selector
                else "VLM Trace · 9B V7 Evidence OS"
            )
            if active_dataset == "nine-b-v7"
            else "VLM Trace · archived 27B V7 reference"
        )
        self.resize(1860, 1050)
        self.setMinimumSize(1280, 760)
        self.setStyleSheet(TRACE_STYLESHEET)
        self._build_toolbar()
        self.tabs = QTabWidget()
        self.explorer = TraceExplorer(
            dataset,
            selector_summary if self.active_selector else None,
        )
        self.tabs.addTab(
            self.explorer,
            (
                "9B selector · tasks"
                if self.active_selector
                else "9B Source V7 · trace"
            )
            if active_dataset == "nine-b-v7"
            else "Archived 27B · trace",
        )
        self.tabs.addTab(Holdout80Page(self.holdout80), "Holdout80 · source evidence")
        self.tabs.addTab(
            MetricsPage(
                dataset,
                selector_summary if self.active_selector else None,
            ),
            (
                "9B selector · analytics"
                if self.active_selector
                else "9B Source V7 · metrics"
            )
            if active_dataset == "nine-b-v7"
            else "Archived 27B · metrics",
        )
        self.tabs.addTab(
            NineBMilestonesPage(nine_b_comparison),
            "9B · seven milestones",
        )
        if selector_summary is not None:
            self.tabs.addTab(
                SelectorWavePage(
                    selector_summary,
                    qa_correct=(
                        qa_reference_summary.correct
                        if qa_reference_summary is not None
                        else None
                    ),
                    qa_rows=(
                        qa_reference_summary.rows
                        if qa_reference_summary is not None
                        else None
                    ),
                ),
                "9B · selector v1.2",
            )
        self.setCentralWidget(self.tabs)
        status = QStatusBar()
        active_view = (
            "nine-b-selector-v1.2"
            if self.active_selector
            else (
                "nine-b-source-v7-lineage"
                if active_dataset == "nine-b-v7"
                else active_dataset
            )
        )
        status.showMessage(
            f"offline · active view: {active_view} · {dataset.summary.rows} joined tasks · "
            f"{len(dataset.source_files)} hash-bound provenance files"
        )
        self.setStatusBar(status)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        brand = QLabel("VLM")
        brand.setObjectName("Brand")
        accent = QLabel("TRACE")
        accent.setObjectName("BrandAccent")
        toolbar.addWidget(brand)
        toolbar.addWidget(accent)
        toolbar.addSeparator()
        descriptor = QLabel("Evidence OS · reasoning / retrieval / certificates")
        descriptor.setObjectName("Subtle")
        toolbar.addWidget(descriptor)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addWidget(_badge("OFFLINE", "good"))
        toolbar.addWidget(_badge("H80 · SEALED", "good"))
        toolbar.addWidget(
            _badge(
                (
                    "9B SELECTOR v1.2 · ACTIVE"
                    if self.active_selector
                    else "9B SOURCE V7 · LINEAGE"
                )
                if self.active_dataset == "nine-b-v7"
                else "27B · ARCHIVED REFERENCE",
                "good" if self.active_dataset == "nine-b-v7" else "warn",
            )
        )
        toolbar.addWidget(
            _badge(
                f"{self.dataset.summary.correct}/{self.dataset.summary.rows}",
                "info",
            )
        )
        if self.nine_b_comparison is None:
            toolbar.addWidget(_badge("9B · AWAITING PINS", "warn"))
        if self.selector_summary is not None and not self.active_selector:
            toolbar.addWidget(
                _badge(
                    f"SELECTOR · {self.selector_summary.correct}/{self.selector_summary.rows}",
                    "good",
                )
            )


def run_gui(dataset: TraceDataset, argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("VLM Trace")
    window = TraceViewerWindow(dataset)
    window.show()
    return app.exec()
