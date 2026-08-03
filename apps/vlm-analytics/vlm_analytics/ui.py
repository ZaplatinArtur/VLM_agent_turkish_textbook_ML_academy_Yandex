from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .analytics import AnalyticsService
from .chunking_importer import import_chunking_experiment
from .config import (
    APP_NAME,
    DEFAULT_KEY_PATH,
    DEFAULT_SERVER,
    DEFAULT_USER,
    MODE_COLORS,
    MODE_ORDER,
    application_dir,
)
from .database import Database
from .importer import import_run
from .style import APP_STYLESHEET
from .sync import SyncManager


class ChartCanvas(FigureCanvasQTAgg):
    def __init__(self, width: float = 6, height: float = 4):
        self.figure = Figure(figsize=(width, height), tight_layout=True)
        self.figure.patch.set_facecolor("#151e29")
        super().__init__(self.figure)
        self.setMinimumHeight(260)

    def axis(self):
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor("#151e29")
        axis.tick_params(colors="#aebbc8")
        axis.xaxis.label.set_color("#aebbc8")
        axis.yaxis.label.set_color("#aebbc8")
        axis.title.set_color("#edf4fb")
        for spine in axis.spines.values():
            spine.set_color("#334255")
        axis.grid(axis="y", color="#2b3948", alpha=0.55, linewidth=0.7)
        return axis


class MetricCard(QFrame):
    def __init__(self, title: str, value: str = "—", subtitle: str = ""):
        super().__init__()
        self.setObjectName("MetricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 13)
        self.value_label = QLabel(value)
        self.value_label.setObjectName("MetricValue")
        self.title_label = QLabel(title)
        self.title_label.setObjectName("MetricLabel")
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("Muted")
        self.subtitle_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.subtitle_label)

    def set_value(self, value: str, subtitle: str = "") -> None:
        self.value_label.setText(value)
        self.subtitle_label.setText(subtitle)


def make_table(columns: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(columns)
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def fill_table(
    table: QTableWidget,
    rows: list[list[Any]],
    *,
    user_data: list[Any] | None = None,
    colorizer: Callable[[int, int, Any], QColor | None] | None = None,
) -> None:
    table.setSortingEnabled(False)
    table.setRowCount(len(rows))
    for row_index, values in enumerate(rows):
        for column_index, value in enumerate(values):
            text = "" if value is None else str(value)
            item = QTableWidgetItem(text)
            if column_index == 0 and user_data:
                item.setData(Qt.ItemDataRole.UserRole, user_data[row_index])
            color = colorizer(row_index, column_index, value) if colorizer else None
            if color:
                item.setBackground(color)
            table.setItem(row_index, column_index, item)
    table.resizeColumnsToContents()
    table.setSortingEnabled(True)


class SyncWorker(QThread):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, manager: SyncManager):
        super().__init__()
        self.manager = manager

    def run(self) -> None:
        try:
            results = self.manager.sync_all(self.progress.emit)
            self.completed.emit(results)
        except Exception:
            self.failed.emit(traceback.format_exc())


class IssueDialog(QDialog):
    def __init__(self, issue: dict[str, Any], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Карточка факапа")
        self.resize(540, 430)
        layout = QFormLayout(self)
        self.component = QLabel(str(issue.get("component") or ""))
        self.title = QLabel(str(issue.get("title") or ""))
        self.title.setWordWrap(True)
        self.status = QComboBox()
        self.status.addItems(("Новый", "В работе", "Нужна проверка", "Исправлен"))
        self.status.setCurrentText(str(issue.get("status") or "Новый"))
        self.severity = QComboBox()
        self.severity.addItems(("Критический", "Высокий", "Средний", "Низкий"))
        self.severity.setCurrentText(str(issue.get("severity") or "Средний"))
        self.owner = QComboBox()
        self.owner.addItems(
            (
                "Агент",
                "Ретрив",
                "Джадж",
                "Агент + ретрив",
                "Комбинированный: джадж + данные",
                "Комбинированный",
            )
        )
        self.owner.setCurrentText(str(issue.get("owner") or "Комбинированный"))
        self.notes = QPlainTextEdit(str(issue.get("notes") or ""))
        layout.addRow("Компонент", self.component)
        layout.addRow("Проблема", self.title)
        layout.addRow("Статус", self.status)
        layout.addRow("Приоритет", self.severity)
        layout.addRow("Ответственный", self.owner)
        layout.addRow("Комментарий", self.notes)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self) -> dict[str, str]:
        return {
            "status": self.status.currentText(),
            "severity": self.severity.currentText(),
            "owner": self.owner.currentText(),
            "notes": self.notes.toPlainText().strip(),
        }


class ManualIssueDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Добавить факап")
        self.resize(520, 390)
        layout = QFormLayout(self)
        self.component = QComboBox()
        self.component.addItems(("Judge", "Agent", "Retrieval", "Data", "Integration"))
        self.title = QLineEdit()
        self.severity = QComboBox()
        self.severity.addItems(("Критический", "Высокий", "Средний", "Низкий"))
        self.severity.setCurrentText("Средний")
        self.owner = QComboBox()
        self.owner.addItems(
            (
                "Агент",
                "Ретрив",
                "Джадж",
                "Агент + ретрив",
                "Комбинированный: джадж + данные",
                "Комбинированный",
            )
        )
        self.notes = QPlainTextEdit()
        layout.addRow("Компонент", self.component)
        layout.addRow("Краткое название", self.title)
        layout.addRow("Приоритет", self.severity)
        layout.addRow("Ответственный", self.owner)
        layout.addRow("Описание", self.notes)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self) -> dict[str, str]:
        return {
            "component": self.component.currentText(),
            "title": self.title.text().strip(),
            "severity": self.severity.currentText(),
            "owner": self.owner.currentText(),
            "notes": self.notes.toPlainText().strip(),
        }


class MainWindow(QMainWindow):
    def __init__(self, database: Database):
        super().__init__()
        self.database = database
        self.analytics = AnalyticsService(database)
        self.sync_manager = SyncManager(database, application_dir() / "sync_cache")
        self.sync_worker: SyncWorker | None = None
        self.task_cache: list[dict[str, Any]] = []
        self.issue_cache: list[dict[str, Any]] = []

        self.setWindowTitle(APP_NAME)
        self.resize(1580, 940)
        self.setMinimumSize(1180, 720)
        self.setStyleSheet(APP_STYLESHEET)
        self._create_toolbar()
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._create_overview_tab()
        self._create_subjects_tab()
        self._create_runs_tab()
        self._create_tasks_tab()
        self._create_judge_tab()
        self._create_agent_tab()
        self._create_retrieval_tab()
        self._create_chunking_tab()
        self._create_issues_tab()
        self._create_dynamics_tab()
        self._create_settings_tab()
        self.setStatusBar(QStatusBar())
        self.auto_timer = QTimer(self)
        self.auto_timer.timeout.connect(self.start_sync)
        self.refresh_all()
        self._update_auto_timer()

    def _create_toolbar(self) -> None:
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        title = QLabel("VLM Analytics")
        title.setObjectName("AppTitle")
        toolbar.addWidget(title)
        spacer = QWidget()
        spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        toolbar.addWidget(spacer)
        self.toolbar_status = QLabel("Локальная аналитика")
        self.toolbar_status.setObjectName("Muted")
        toolbar.addWidget(self.toolbar_status)
        refresh = QPushButton("Обновить")
        refresh.clicked.connect(self.refresh_all)
        toolbar.addWidget(refresh)
        sync = QPushButton("Синхронизировать")
        sync.setObjectName("Primary")
        sync.clicked.connect(self.start_sync)
        toolbar.addWidget(sync)

    @staticmethod
    def _page() -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        return page, layout

    @staticmethod
    def _title(text: str, subtitle: str = "") -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel(text)
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        if subtitle:
            description = QLabel(subtitle)
            description.setObjectName("Muted")
            description.setWordWrap(True)
            layout.addWidget(description)
        return wrapper

    def _create_overview_tab(self) -> None:
        page, layout = self._page()
        layout.addWidget(
            self._title(
                "Общая картина",
                "Последний снимок каждого режима на одном каноническом датасете.",
            )
        )
        cards = QHBoxLayout()
        self.card_best = MetricCard("Лучший режим")
        self.card_tasks = MetricCard("Задач в срезе")
        self.card_runs = MetricCard("Снимков в истории")
        self.card_issues = MetricCard("Активных факапов")
        for card in (
            self.card_best,
            self.card_tasks,
            self.card_runs,
            self.card_issues,
        ):
            cards.addWidget(card)
        layout.addLayout(cards)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.overview_table = make_table(
            [
                "Режим",
                "Верно",
                "Accuracy",
                "Forced",
                "Поиск",
                "Tool calls",
                "Judge errors",
                "Время, с",
            ]
        )
        self.overview_chart = ChartCanvas(7, 4)
        splitter.addWidget(self.overview_table)
        splitter.addWidget(self.overview_chart)
        splitter.setSizes([780, 680])
        layout.addWidget(splitter, 1)
        self.tabs.addTab(page, "Обзор")

    def _create_subjects_tab(self) -> None:
        page, layout = self._page()
        layout.addWidget(
            self._title(
                "Аналитика по предметам",
                "Точность каждого режима и размер предметного среза.",
            )
        )
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.subject_table = make_table(["Предмет", "N"])
        self.subject_chart = ChartCanvas(10, 5)
        splitter.addWidget(self.subject_table)
        splitter.addWidget(self.subject_chart)
        splitter.setSizes([390, 430])
        layout.addWidget(splitter, 1)
        self.tabs.addTab(page, "Предметы")

    def _create_runs_tab(self) -> None:
        page, layout = self._page()
        controls = QHBoxLayout()
        controls.addWidget(
            self._title(
                "История прогонов",
                "Каждое изменение файлов создаёт новый неизменяемый снимок.",
            ),
            1,
        )
        button = QPushButton("Синхронизировать сейчас")
        button.setObjectName("Primary")
        button.clicked.connect(self.start_sync)
        controls.addWidget(button)
        layout.addLayout(controls)
        self.runs_table = make_table(
            [
                "ID",
                "Режим",
                "Дата импорта",
                "Набор",
                "Модель",
                "Prompt",
                "Задач",
                "Верно",
                "Accuracy",
                "Judge errors",
            ]
        )
        layout.addWidget(self.runs_table)
        self.tabs.addTab(page, "Прогоны")

    def _create_tasks_tab(self) -> None:
        page, layout = self._page()
        layout.addWidget(
            self._title(
                "Задачи",
                "Полные ответы, reasoning, вердикты и следы инструментов.",
            )
        )
        filters = QHBoxLayout()
        self.task_mode = QComboBox()
        self.task_subject = QComboBox()
        self.task_search = QLineEdit()
        self.task_search.setPlaceholderText("task_id, предмет или ответ")
        self.task_problems = QCheckBox("Только проблемные")
        for widget in (
            QLabel("Режим"),
            self.task_mode,
            QLabel("Предмет"),
            self.task_subject,
            self.task_search,
            self.task_problems,
        ):
            filters.addWidget(widget)
        filters.setStretch(5, 1)
        layout.addLayout(filters)
        self.task_mode.currentIndexChanged.connect(self.refresh_tasks)
        self.task_subject.currentIndexChanged.connect(self.refresh_tasks)
        self.task_search.textChanged.connect(self.refresh_tasks)
        self.task_problems.toggled.connect(self.refresh_tasks)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.tasks_table = make_table(
            [
                "Task",
                "Режим",
                "Предмет",
                "Ответ",
                "Score",
                "Forced",
                "Calls",
                "Input",
                "Output",
                "Время",
            ]
        )
        self.tasks_table.itemSelectionChanged.connect(self.show_task_detail)
        detail_box = QWidget()
        detail_layout = QVBoxLayout(detail_box)
        detail_actions = QHBoxLayout()
        self.open_question = QPushButton("Открыть/скопировать вопрос")
        self.open_reference = QPushButton("Открыть/скопировать эталон")
        self.open_question.clicked.connect(
            lambda: self._open_current_link("question_image_url")
        )
        self.open_reference.clicked.connect(
            lambda: self._open_current_link("reference_image_url")
        )
        detail_actions.addWidget(self.open_question)
        detail_actions.addWidget(self.open_reference)
        detail_actions.addStretch()
        self.task_detail = QPlainTextEdit()
        self.task_detail.setReadOnly(True)
        detail_layout.addLayout(detail_actions)
        detail_layout.addWidget(self.task_detail)
        splitter.addWidget(self.tasks_table)
        splitter.addWidget(detail_box)
        splitter.setSizes([470, 320])
        layout.addWidget(splitter, 1)
        self.tabs.addTab(page, "Задачи")

    def _create_judge_tab(self) -> None:
        page, layout = self._page()
        layout.addWidget(
            self._title(
                "Judge",
                "Ошибки оценки, reference issues и расхождения бинарного score.",
            )
        )
        vertical = QSplitter(Qt.Orientation.Vertical)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.judge_table = make_table(
            [
                "Режим",
                "Fully",
                "Partial",
                "Incorrect",
                "Judge errors",
                "Reference issues",
                "Final≠Strict",
            ]
        )
        self.judge_chart = ChartCanvas(7, 4)
        splitter.addWidget(self.judge_table)
        splitter.addWidget(self.judge_chart)
        splitter.setSizes([760, 700])
        vertical.addWidget(splitter)

        audit_box = QGroupBox("Ручная проверка judge")
        audit_layout = QVBoxLayout(audit_box)
        audit_cards = QHBoxLayout()
        self.audit_accuracy = MetricCard("Согласие с ручной разметкой")
        self.audit_precision = MetricCard("Precision оценки «верно»")
        self.audit_recall = MetricCard("Recall оценки «верно»")
        self.audit_reasoning = MetricCard("Согласие по reasoning")
        self.audit_kappa = MetricCard("Cohen's kappa")
        self.audit_reference = MetricCard("Поиск плохих эталонов")
        for card in (
            self.audit_accuracy,
            self.audit_precision,
            self.audit_recall,
            self.audit_reasoning,
            self.audit_kappa,
            self.audit_reference,
        ):
            audit_cards.addWidget(card)
        audit_layout.addLayout(audit_cards)
        self.audit_caption = QLabel()
        self.audit_caption.setObjectName("Muted")
        self.audit_caption.setWordWrap(True)
        audit_layout.addWidget(self.audit_caption)
        audit_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.audit_mode_table = make_table(
            ["Режим", "Проверено", "Согласие", "False positive", "False negative"]
        )
        self.audit_errors_table = make_table(
            [
                "Тип ошибки",
                "Режим",
                "Task",
                "Предмет",
                "Judge answer",
                "Ручной answer",
                "Judge reasoning",
                "Ручной reasoning",
                "Что произошло",
            ]
        )
        audit_splitter.addWidget(self.audit_mode_table)
        audit_splitter.addWidget(self.audit_errors_table)
        audit_splitter.setSizes([570, 920])
        audit_layout.addWidget(audit_splitter, 1)
        vertical.addWidget(audit_box)
        vertical.setSizes([380, 460])
        layout.addWidget(vertical, 1)
        self.tabs.addTab(page, "Judge")

    def _create_agent_tab(self) -> None:
        page, layout = self._page()
        layout.addWidget(
            self._title(
                "Agent",
                "Формат ответа, forced-final, reasoning и стоимость генерации.",
            )
        )
        self.agent_table = make_table(
            [
                "Режим",
                "Reasoning",
                "Forced",
                "Accuracy",
                "Input tokens",
                "Output tokens",
                "Время, с",
            ]
        )
        self.agent_chart = ChartCanvas(10, 4)
        layout.addWidget(self.agent_table)
        layout.addWidget(self.agent_chart, 1)
        self.tabs.addTab(page, "Agent")

    def _create_retrieval_tab(self) -> None:
        page, layout = self._page()
        layout.addWidget(
            self._title(
                "Retrieval и веб-поиск",
                "Частота вызовов, пустые результаты, ошибки и объём контекста.",
            )
        )
        self.retrieval_table = make_table(
            [
                "Режим",
                "Задач с поиском",
                "Search rate",
                "Calls",
                "Уник. запросов",
                "Ошибок",
                "Без результатов",
                "Chunks",
                "Accuracy",
            ]
        )
        self.retrieval_chart = ChartCanvas(10, 4)
        layout.addWidget(self.retrieval_table)
        layout.addWidget(self.retrieval_chart, 1)
        self.tabs.addTab(page, "Retrieval")

    def _create_chunking_tab(self) -> None:
        page, layout = self._page()
        header = QHBoxLayout()
        header.addWidget(
            self._title(
                "Чанкинг учебников",
                "Сравнение страничных чанков с task-aware нарезкой. "
                "Hit@K здесь измеряет локализацию нужного задания, "
                "а не конечную accuracy агента.",
            ),
            1,
        )
        import_button = QPushButton("Импортировать отчёты")
        import_button.setObjectName("Primary")
        import_button.clicked.connect(self.import_chunking_reports)
        header.addWidget(import_button)
        layout.addLayout(header)
        cards = QHBoxLayout()
        self.chunk_hit1_card = MetricCard("Hit@1")
        self.chunk_hit5_card = MetricCard("Hit@5")
        self.chunk_mrr_card = MetricCard("MRR@5")
        self.chunk_corpus_card = MetricCard("Размер корпуса")
        for card in (
            self.chunk_hit1_card,
            self.chunk_hit5_card,
            self.chunk_mrr_card,
            self.chunk_corpus_card,
        ):
            cards.addWidget(card)
        layout.addLayout(cards)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.chunking_table = make_table(
            ["Группа", "Метрика", "Было", "Стало", "Δ", "N", "Комментарий"]
        )
        self.chunking_chart = ChartCanvas(7, 4)
        splitter.addWidget(self.chunking_table)
        splitter.addWidget(self.chunking_chart)
        splitter.setSizes([880, 620])
        layout.addWidget(splitter, 1)

        self.chunking_details = make_table(["Показатель", "Значение"])
        self.chunking_details.setMaximumHeight(190)
        layout.addWidget(self.chunking_details)
        self.tabs.addTab(page, "Чанкинг")

    def import_chunking_reports(self, *_args) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Папка с отчётами чанкинга",
            str(Path.home()),
        )
        if not selected:
            return
        directory = Path(selected)

        def latest(pattern: str) -> Path:
            matches = sorted(
                directory.glob(pattern),
                key=lambda path: (path.stat().st_mtime, path.name),
                reverse=True,
            )
            if not matches:
                raise FileNotFoundError(f"Не найден файл {pattern}")
            return matches[0]

        try:
            experiment_id = import_chunking_experiment(
                self.database,
                corpus_report=latest("hybrid_chunking_all_*.json"),
                localization_report=latest("chunk_localization_*.json"),
                refinement_report=latest("hybrid_qwen_refine_*.json"),
                audit_report=latest("hybrid_qwen_holdout_*_repaired.json"),
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Не удалось импортировать",
                str(exc),
            )
            return
        self.refresh_chunking()
        self.statusBar().showMessage(
            f"Эксперимент чанкинга #{experiment_id} импортирован",
            5000,
        )

    def _create_issues_tab(self) -> None:
        page, layout = self._page()
        header = QHBoxLayout()
        header.addWidget(
            self._title(
                "Факапы",
                "Автоматические и ручные проблемы, отслеживаемые между прогонами.",
            ),
            1,
        )
        add = QPushButton("Добавить вручную")
        add.clicked.connect(self.add_manual_issue)
        edit = QPushButton("Открыть карточку")
        edit.setObjectName("Primary")
        edit.clicked.connect(self.edit_issue)
        header.addWidget(add)
        header.addWidget(edit)
        layout.addLayout(header)
        filters = QHBoxLayout()
        self.issue_component = QComboBox()
        self.issue_component.addItems(
            ("Все", "Judge", "Agent", "Retrieval", "Data", "Integration")
        )
        self.issue_owner = QComboBox()
        self.issue_owner.addItems(
            (
                "Все",
                "Агент",
                "Ретрив",
                "Джадж",
                "Комбинированные",
            )
        )
        self.issue_status = QComboBox()
        self.issue_status.addItems(
            ("Все", "Новый", "В работе", "Нужна проверка", "Исправлен")
        )
        filters.addWidget(QLabel("Компонент"))
        filters.addWidget(self.issue_component)
        filters.addWidget(QLabel("Ответственный"))
        filters.addWidget(self.issue_owner)
        filters.addWidget(QLabel("Статус"))
        filters.addWidget(self.issue_status)
        filters.addStretch()
        layout.addLayout(filters)
        self.issue_component.currentIndexChanged.connect(self.refresh_issues)
        self.issue_owner.currentIndexChanged.connect(self.refresh_issues)
        self.issue_status.currentIndexChanged.connect(self.refresh_issues)
        self.issues_table = make_table(
            [
                "Компонент",
                "Приоритет",
                "Проблема",
                "Статус",
                "Ответственный",
                "Task",
                "Последний режим",
                "Первый раз",
                "Последний раз",
                "Повторов",
            ]
        )
        self.issues_table.doubleClicked.connect(self.edit_issue)
        layout.addWidget(self.issues_table)
        self.tabs.addTab(page, "Факапы")

    def _create_dynamics_tab(self) -> None:
        page, layout = self._page()
        layout.addWidget(
            self._title(
                "Динамика",
                "Рост или падение метрик между импортированными запусками.",
            )
        )
        filters = QHBoxLayout()
        self.dynamic_metric = QComboBox()
        self.dynamic_metric.addItem("Accuracy, %", "accuracy")
        self.dynamic_metric.addItem("Forced answer, %", "forced_rate")
        self.dynamic_metric.addItem("Reasoning, %", "reasoning_rate")
        self.dynamic_metric.addItem("Средняя задержка, с", "avg_latency")
        self.dynamic_metric.addItem("Средние input tokens", "avg_input_tokens")
        self.dynamic_metric.addItem("Средние output tokens", "avg_output_tokens")
        self.dynamic_metric.addItem("Judge errors, %", "judge_error_rate")
        self.dynamic_subject = QComboBox()
        filters.addWidget(QLabel("Метрика"))
        filters.addWidget(self.dynamic_metric)
        filters.addWidget(QLabel("Предмет"))
        filters.addWidget(self.dynamic_subject)
        filters.addStretch()
        layout.addLayout(filters)
        self.dynamic_metric.currentIndexChanged.connect(self.refresh_dynamics)
        self.dynamic_subject.currentIndexChanged.connect(self.refresh_dynamics)
        self.dynamic_chart = ChartCanvas(12, 5)
        self.dynamic_table = make_table(
            ["Дата", "Режим", "Значение", "N", "Δ к прошлому"]
        )
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.dynamic_chart)
        splitter.addWidget(self.dynamic_table)
        splitter.setSizes([500, 280])
        layout.addWidget(splitter, 1)
        self.tabs.addTab(page, "Динамика")

    def _create_settings_tab(self) -> None:
        page, layout = self._page()
        layout.addWidget(
            self._title(
                "Настройки и синхронизация",
                "SSH использует установленный Windows OpenSSH и существующий ключ.",
            )
        )
        group = QGroupBox("Подключение")
        form = QFormLayout(group)
        self.setting_server = QLineEdit(
            self.database.get_setting("server", DEFAULT_SERVER)
        )
        self.setting_user = QLineEdit(
            self.database.get_setting("user", DEFAULT_USER)
        )
        self.setting_key = QLineEdit(
            self.database.get_setting("key_path", DEFAULT_KEY_PATH)
        )
        form.addRow("Сервер", self.setting_server)
        form.addRow("Пользователь", self.setting_user)
        form.addRow("SSH-ключ", self.setting_key)
        layout.addWidget(group)
        actions = QHBoxLayout()
        save = QPushButton("Сохранить настройки")
        save.clicked.connect(self.save_settings)
        sync = QPushButton("Синхронизировать по SSH")
        sync.setObjectName("Primary")
        sync.clicked.connect(self.start_sync)
        local_import = QPushButton("Импортировать локальный прогон")
        local_import.clicked.connect(self.import_local_run)
        actions.addWidget(save)
        actions.addWidget(sync)
        actions.addWidget(local_import)
        actions.addStretch()
        layout.addLayout(actions)
        auto_group = QGroupBox("Автоматическая синхронизация")
        auto_layout = QHBoxLayout(auto_group)
        self.auto_sync = QCheckBox("Включить")
        self.auto_sync.setChecked(
            self.database.get_setting("auto_sync", "0") == "1"
        )
        self.auto_minutes = QSpinBox()
        self.auto_minutes.setRange(5, 1440)
        self.auto_minutes.setValue(
            int(self.database.get_setting("auto_sync_minutes", "15"))
        )
        auto_layout.addWidget(self.auto_sync)
        auto_layout.addWidget(QLabel("Интервал, минут"))
        auto_layout.addWidget(self.auto_minutes)
        auto_layout.addStretch()
        self.auto_sync.toggled.connect(self._save_auto_sync)
        self.auto_minutes.valueChanged.connect(self._save_auto_sync)
        layout.addWidget(auto_group)
        self.sync_progress = QProgressBar()
        self.sync_progress.setRange(0, 1)
        self.sync_progress.setValue(0)
        self.sync_log = QPlainTextEdit()
        self.sync_log.setReadOnly(True)
        self.sync_log.setMaximumBlockCount(500)
        layout.addWidget(self.sync_progress)
        layout.addWidget(self.sync_log, 1)
        footer = QLabel(f"База: {self.database.path}")
        footer.setObjectName("Muted")
        footer.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(footer)
        self.tabs.addTab(page, "Настройки")

    def refresh_all(self) -> None:
        self.refresh_filters()
        self.refresh_overview()
        self.refresh_subjects()
        self.refresh_runs()
        self.refresh_tasks()
        self.refresh_judge()
        self.refresh_agent()
        self.refresh_retrieval()
        self.refresh_chunking()
        self.refresh_issues()
        self.refresh_dynamics()
        last_sync = self.database.get_setting("last_sync", "ещё не было")
        self.toolbar_status.setText(f"Последняя синхронизация: {last_sync}")
        self.statusBar().showMessage("Данные обновлены", 3000)

    def refresh_filters(self) -> None:
        modes = self.analytics.mode_summaries()
        subjects, _, _ = self.analytics.subject_matrix()
        current_mode = self.task_mode.currentData() if self.task_mode.count() else None
        self.task_mode.blockSignals(True)
        self.task_mode.clear()
        self.task_mode.addItem("Все последние режимы", None)
        for mode in modes:
            self.task_mode.addItem(mode.display_name, mode.run_id)
        if current_mode:
            index = self.task_mode.findData(current_mode)
            self.task_mode.setCurrentIndex(max(0, index))
        self.task_mode.blockSignals(False)
        for combo in (self.task_subject, self.dynamic_subject):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Все предметы")
            combo.addItems(subjects)
            index = combo.findText(current)
            combo.setCurrentIndex(max(0, index))
            combo.blockSignals(False)

    def refresh_overview(self) -> None:
        summaries = self.analytics.mode_summaries()
        history_count = int(self.database.scalar("SELECT COUNT(*) FROM runs") or 0)
        active_issues = int(
            self.database.scalar(
                "SELECT COUNT(*) FROM issues WHERE status <> 'Исправлен'"
            )
            or 0
        )
        if summaries:
            best = max(summaries, key=lambda item: item.accuracy)
            self.card_best.set_value(
                f"{best.accuracy:.1f}%", best.display_name
            )
            self.card_tasks.set_value(
                str(max(mode.total for mode in summaries)),
                "в последнем каноническом срезе",
            )
        else:
            self.card_best.set_value("—", "нет данных")
            self.card_tasks.set_value("0")
        self.card_runs.set_value(str(history_count), "импортированных снимков")
        self.card_issues.set_value(str(active_issues), "не закрыто")
        rows = [
            [
                item.display_name,
                f"{item.correct}/{item.total}",
                f"{item.accuracy:.1f}%",
                f"{item.forced} ({item.forced_rate:.1f}%)",
                f"{item.search_tasks} ({item.search_rate:.1f}%)",
                item.tool_calls,
                item.judge_errors,
                f"{item.avg_latency_s:.1f}",
            ]
            for item in summaries
        ]
        fill_table(self.overview_table, rows)
        axis = self.overview_chart.axis()
        if summaries:
            names = [item.display_name for item in summaries]
            values = [item.accuracy for item in summaries]
            colors = [MODE_COLORS.get(item.run_key, "#7e8b99") for item in summaries]
            bars = axis.bar(names, values, color=colors, width=0.62)
            axis.set_ylim(0, 100)
            axis.set_ylabel("Accuracy, %")
            axis.set_title("Последняя точность по режимам")
            axis.tick_params(axis="x", rotation=12)
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.5,
                    f"{value:.1f}%",
                    ha="center",
                    color="#eef4f8",
                    fontsize=10,
                )
        else:
            axis.text(
                0.5,
                0.5,
                "Синхронизируйте данные",
                ha="center",
                va="center",
                color="#91a0af",
                transform=axis.transAxes,
            )
        self.overview_chart.draw_idle()

    def refresh_subjects(self) -> None:
        subjects, modes, values = self.analytics.subject_matrix()
        columns = ["Предмет", "N"] + [mode.display_name for mode in modes]
        self.subject_table.setColumnCount(len(columns))
        self.subject_table.setHorizontalHeaderLabels(columns)
        rows: list[list[Any]] = []
        matrix: list[list[float]] = []
        for subject in subjects:
            n = max(
                (
                    values[subject].get(mode.run_key, {}).get("total", 0)
                    for mode in modes
                ),
                default=0,
            )
            row: list[Any] = [subject, n]
            matrix_row: list[float] = []
            for mode in modes:
                cell = values[subject].get(mode.run_key, {})
                accuracy = float(cell.get("accuracy", 0))
                correct = int(cell.get("correct", 0))
                total = int(cell.get("total", 0))
                row.append(f"{correct}/{total} ({accuracy:.1f}%)")
                matrix_row.append(accuracy)
            rows.append(row)
            matrix.append(matrix_row)
        fill_table(self.subject_table, rows)
        axis = self.subject_chart.axis()
        if matrix and modes:
            image = axis.imshow(
                matrix,
                cmap="RdYlGn",
                vmin=0,
                vmax=100,
                aspect="auto",
            )
            axis.set_xticks(range(len(modes)), [mode.display_name for mode in modes])
            axis.set_yticks(range(len(subjects)), subjects)
            axis.tick_params(axis="x", rotation=15)
            for y, row in enumerate(matrix):
                for x, value in enumerate(row):
                    axis.text(
                        x,
                        y,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        color="#10151c" if 35 < value < 85 else "#f7fbff",
                        fontsize=8,
                        fontweight="bold",
                    )
            colorbar = self.subject_chart.figure.colorbar(
                image, ax=axis, fraction=0.025, pad=0.02
            )
            colorbar.ax.tick_params(colors="#aebbc8")
            axis.set_title("Accuracy по предметам")
        self.subject_chart.draw_idle()

    def refresh_runs(self) -> None:
        runs = self.database.run_history()
        rows = []
        for run in runs:
            total = int(run["record_count"] or 0)
            correct = int(run["correct"] or 0)
            accuracy = 100 * correct / total if total else 0
            rows.append(
                [
                    run["id"],
                    run["display_name"],
                    run["imported_at"],
                    run["dataset_version"],
                    run["model"],
                    run["prompt_version"],
                    total,
                    correct,
                    f"{accuracy:.1f}%",
                    run["judge_errors"],
                ]
            )
        fill_table(self.runs_table, rows)

    def refresh_tasks(self, *_args) -> None:
        if not hasattr(self, "tasks_table"):
            return
        self.task_cache = self.analytics.task_rows(
            run_id=self.task_mode.currentData(),
            subject=self.task_subject.currentText(),
            search=self.task_search.text().strip(),
            only_problems=self.task_problems.isChecked(),
        )
        rows = [
            [
                item["task_id"],
                item["display_name"],
                item["subject"],
                item["final_answer"],
                (
                    "1"
                    if item["strict_correct"] == 1
                    else "0"
                    if item["strict_correct"] == 0
                    else "ERR"
                ),
                "да" if item["forced_answer"] else "",
                item["tool_call_count"],
                item["input_tokens"],
                item["output_tokens"],
                (
                    f"{float(item['latency_s']):.1f}"
                    if item["latency_s"] is not None
                    else ""
                ),
            ]
            for item in self.task_cache
        ]

        def colorizer(_: int, column: int, value: Any) -> QColor | None:
            if column == 4:
                if value == "1":
                    return QColor("#1f5a45")
                if value == "0":
                    return QColor("#65343d")
                return QColor("#765f2e")
            return None

        fill_table(
            self.tasks_table,
            rows,
            user_data=list(range(len(rows))),
            colorizer=colorizer,
        )
        self.task_detail.clear()

    def current_task(self) -> dict[str, Any] | None:
        row = self.tasks_table.currentRow()
        if row < 0:
            return None
        item = self.tasks_table.item(row, 0)
        if item is None:
            return None
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None or int(index) >= len(self.task_cache):
            return None
        return self.task_cache[int(index)]

    def show_task_detail(self) -> None:
        task = self.current_task()
        if not task:
            return
        tool_rows = self.database.rows(
            """
            SELECT tool, query, returned_count, latency_ms, error, result_preview
            FROM tool_calls
            WHERE task_result_id = ?
            ORDER BY call_index
            """,
            (task["id"],),
        )
        blocks = [
            f"TASK: {task['task_id']}",
            f"РЕЖИМ: {task['display_name']}",
            f"ПРЕДМЕТ: {task['subject']}",
            f"ТИП ЭТАЛОНА: {task['reference_kind']}",
            f"FINAL ANSWER:\n{task['final_answer'] or '[пусто]'}",
            f"SOLUTION STEPS:\n{task['solution_steps'] or '[пусто]'}",
            f"REASONING:\n{task['reasoning'] or '[пусто]'}",
            (
                "JUDGE:\n"
                f"strict={task['strict_correct']} "
                f"final={task['final_answer_correct']} "
                f"reasoning={task['reasoning_correct']} "
                f"label={task['judge_label']} confidence={task['judge_confidence']}\n"
                f"{task['judge_rationale'] or ''}\n"
                f"error={task['judge_error'] or ''}"
            ),
        ]
        if tool_rows:
            tool_text = []
            for index, call in enumerate(tool_rows, 1):
                tool_text.append(
                    f"{index}. {call['tool']} query={call['query']}\n"
                    f"returned={call['returned_count']} latency={call['latency_ms']} "
                    f"error={call['error'] or ''}\n"
                    f"{call['result_preview'] or ''}"
                )
            blocks.append("TOOL CALLS:\n" + "\n\n".join(tool_text))
        blocks.append(f"RAW RESPONSE:\n{task['raw_response'] or '[пусто]'}")
        self.task_detail.setPlainText("\n\n".join(blocks))

    def _open_current_link(self, field: str) -> None:
        task = self.current_task()
        if not task or not task.get(field):
            QMessageBox.information(self, APP_NAME, "Ссылка отсутствует.")
            return
        value = str(task[field])
        path = Path(value)
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            return
        QGuiApplication.clipboard().setText(value)
        QMessageBox.information(
            self,
            APP_NAME,
            "Путь не доступен локально и скопирован в буфер обмена.",
        )

    def refresh_judge(self) -> None:
        stats = self.analytics.judge_stats()
        rows = []
        for item in stats:
            labels = item["labels"]
            rows.append(
                [
                    item["display_name"],
                    labels.get("fully_correct", 0),
                    labels.get("partially_correct", 0)
                    + labels.get("mostly_correct", 0),
                    labels.get("incorrect", 0),
                    item["judge_errors"],
                    item["reference_issues"],
                    item["final_strict_disagreements"],
                ]
            )
        fill_table(self.judge_table, rows)
        axis = self.judge_chart.axis()
        if stats:
            names = [item["display_name"] for item in stats]
            full = [item["labels"].get("fully_correct", 0) for item in stats]
            partial = [
                item["labels"].get("partially_correct", 0)
                + item["labels"].get("mostly_correct", 0)
                for item in stats
            ]
            incorrect = [item["labels"].get("incorrect", 0) for item in stats]
            axis.bar(names, full, label="Fully correct", color="#3ddc97")
            axis.bar(
                names,
                partial,
                bottom=full,
                label="Partial",
                color="#ffb454",
            )
            bottoms = [a + b for a, b in zip(full, partial)]
            axis.bar(
                names,
                incorrect,
                bottom=bottoms,
                label="Incorrect",
                color="#ef6475",
            )
            axis.set_title("Распределение вердиктов")
            axis.set_ylabel("Задач")
            axis.tick_params(axis="x", rotation=12)
            legend = axis.legend(facecolor="#151e29", labelcolor="#dce6f0")
            legend.get_frame().set_edgecolor("#334255")
        self.judge_chart.draw_idle()

        audit = self.analytics.latest_judge_audit()
        if not audit:
            self.audit_caption.setText("Ручная разметка пока не импортирована.")
            fill_table(self.audit_mode_table, [])
            fill_table(self.audit_errors_table, [])
            return
        self.audit_accuracy.set_value(
            f"{audit['accuracy']:.1f}%",
            f"{audit['agreed']}/{audit['evaluable']} оцениваемых ответов",
        )
        self.audit_precision.set_value(
            f"{audit['precision']:.1f}%",
            f"FP: {audit['fp']}",
        )
        self.audit_recall.set_value(
            f"{audit['recall']:.1f}%",
            f"FN: {audit['fn']}",
        )
        self.audit_reasoning.set_value(
            f"{audit['reasoning_accuracy']:.1f}%",
            (
                f"{audit['reasoning_agreed']}/"
                f"{audit['reasoning_evaluable']} оценённых reasoning"
            ),
        )
        self.audit_kappa.set_value(
            f"{audit['kappa']:.2f}",
            "согласие с поправкой на случайность",
        )
        self.audit_reference.set_value(
            f"{audit['reference_recall']:.1f}%",
            (
                f"найдено {audit['ref_detected']}/{audit['ref_positive']}; "
                f"лишних флагов {audit['ref_false_positive']}"
            ),
        )
        self.audit_caption.setText(
            f"{audit['name']}. {audit['methodology']} "
            f"Неоценимых из-за сломанного задания: "
            f"{audit['total'] - audit['evaluable']}."
        )
        mode_names = {
            "b0_no_tools": "Без тулов",
            "web_search": "Веб",
            "agent_rag": "RAG",
            "agent_rag_thinking": "RAG thinking",
        }
        mode_rows = [
            [
                mode_names.get(str(item["label"]), item["label"]),
                item["evaluable"],
                f"{item['accuracy']:.1f}%",
                item["fp"],
                item["fn"],
            ]
            for item in self.analytics.judge_audit_breakdown("run_key")
        ]
        fill_table(self.audit_mode_table, mode_rows)
        error_rows = [
            [
                item["error_category"],
                item["display_name"] or mode_names.get(
                    str(item["run_key"]), item["run_key"]
                ),
                item["task_id"],
                item["subject"],
                item["judge_answer_correct"],
                item["manual_answer_correct"],
                item["judge_reasoning_correct"],
                item["manual_reasoning_correct"],
                item["manual_note"],
            ]
            for item in self.analytics.judge_audit_errors()
        ]
        fill_table(self.audit_errors_table, error_rows)

    def refresh_agent(self) -> None:
        summaries = self.analytics.mode_summaries()
        rows = [
            [
                item.display_name,
                f"{item.reasoning}/{item.total} ({item.reasoning_rate:.1f}%)",
                f"{item.forced}/{item.total} ({item.forced_rate:.1f}%)",
                f"{item.accuracy:.1f}%",
                f"{item.avg_input_tokens:.0f}",
                f"{item.avg_output_tokens:.0f}",
                f"{item.avg_latency_s:.1f}",
            ]
            for item in summaries
        ]
        fill_table(self.agent_table, rows)
        axis = self.agent_chart.axis()
        if summaries:
            x = list(range(len(summaries)))
            width = 0.24
            axis.bar(
                [value - width for value in x],
                [item.accuracy for item in summaries],
                width,
                label="Accuracy",
                color="#3ddc97",
            )
            axis.bar(
                x,
                [item.forced_rate for item in summaries],
                width,
                label="Forced",
                color="#ef6475",
            )
            axis.bar(
                [value + width for value in x],
                [item.reasoning_rate for item in summaries],
                width,
                label="Reasoning",
                color="#8b9cff",
            )
            axis.set_xticks(x, [item.display_name for item in summaries])
            axis.set_ylim(0, 105)
            axis.set_ylabel("% задач")
            axis.set_title("Качество и стабильность ответа")
            axis.tick_params(axis="x", rotation=12)
            legend = axis.legend(facecolor="#151e29", labelcolor="#dce6f0")
            legend.get_frame().set_edgecolor("#334255")
        self.agent_chart.draw_idle()

    def refresh_retrieval(self) -> None:
        stats = self.analytics.tool_stats()
        rows = [
            [
                item["display_name"],
                f"{item['search_tasks']}/{item['tasks']}",
                f"{item['search_rate']:.1f}%",
                item["calls"],
                item["unique_queries"],
                item["errors"],
                item["no_result"],
                item["returned_chunks"],
                f"{item['accuracy']:.1f}%",
            ]
            for item in stats
        ]
        fill_table(self.retrieval_table, rows)
        axis = self.retrieval_chart.axis()
        searchable = [item for item in stats if item["calls"]]
        if searchable:
            names = [item["display_name"] for item in searchable]
            x = list(range(len(searchable)))
            width = 0.35
            axis.bar(
                [value - width / 2 for value in x],
                [item["search_rate"] for item in searchable],
                width,
                color="#50b7d8",
                label="Задачи с поиском, %",
            )
            error_rates = [
                100 * item["errors"] / item["calls"] if item["calls"] else 0
                for item in searchable
            ]
            axis.bar(
                [value + width / 2 for value in x],
                error_rates,
                width,
                color="#ef6475",
                label="Ошибки calls, %",
            )
            axis.set_xticks(x, names)
            axis.set_ylim(0, 105)
            axis.set_title("Использование и ошибки поиска")
            axis.tick_params(axis="x", rotation=12)
            legend = axis.legend(facecolor="#151e29", labelcolor="#dce6f0")
            legend.get_frame().set_edgecolor("#334255")
        self.retrieval_chart.draw_idle()

    def refresh_chunking(self) -> None:
        experiment = self.analytics.latest_retrieval_experiment()
        cards = (
            self.chunk_hit1_card,
            self.chunk_hit5_card,
            self.chunk_mrr_card,
            self.chunk_corpus_card,
        )
        if not experiment:
            for card in cards:
                card.set_value("—", "Нет импортированных экспериментов")
            fill_table(self.chunking_table, [])
            fill_table(self.chunking_details, [])
            self.chunking_chart.axis().text(
                0.5,
                0.5,
                "Импортируйте отчёт чанкинга",
                ha="center",
                va="center",
                color="#91a0af",
                transform=self.chunking_chart.figure.axes[0].transAxes,
            )
            self.chunking_chart.draw_idle()
            return

        metrics = {
            str(item["metric_key"]): item for item in experiment["metrics"]
        }

        def percent_card(card: MetricCard, key: str) -> None:
            item = metrics[key]
            before = float(item["baseline_value"] or 0)
            after = float(item["candidate_value"] or 0)
            card.set_value(
                f"{before:.1f}% → {after:.1f}%",
                f"{after - before:+.1f} п.п. · N={item['sample_size']}",
            )

        percent_card(self.chunk_hit1_card, "hit_at_1")
        percent_card(self.chunk_hit5_card, "hit_at_5")
        percent_card(self.chunk_mrr_card, "mrr_at_5")
        chunks = metrics["chunks"]
        self.chunk_corpus_card.set_value(
            f"{int(chunks['baseline_value']):,} → "
            f"{int(chunks['candidate_value']):,}".replace(",", " "),
            experiment["dataset"],
        )

        def format_value(value: Any, unit: str) -> str:
            if value is None:
                return "—"
            number = float(value)
            if unit == "%":
                return f"{number:.1f}%"
            if unit == "шт.":
                return f"{int(number):,}".replace(",", " ")
            if unit == "с":
                return f"{number:.1f} с"
            return f"{number:.3g}{unit}"

        rows = []
        for item in experiment["metrics"]:
            before = item["baseline_value"]
            after = item["candidate_value"]
            unit = str(item["unit"] or "")
            delta = "—"
            if before is not None and after is not None:
                difference = float(after) - float(before)
                if unit == "%":
                    delta = f"{difference:+.1f} п.п."
                elif unit == "шт.":
                    delta = f"{int(difference):+,}".replace(",", " ") + " шт."
                else:
                    delta = f"{difference:+.2g} {unit}".rstrip()
            rows.append(
                [
                    item["category"],
                    item["label"],
                    format_value(before, unit),
                    format_value(after, unit),
                    delta,
                    item["sample_size"],
                    item["note"],
                ]
            )
        fill_table(self.chunking_table, rows)

        localization = [
            metrics[key] for key in ("hit_at_1", "hit_at_5", "mrr_at_5")
        ]
        axis = self.chunking_chart.axis()
        x = list(range(len(localization)))
        width = 0.34
        axis.bar(
            [value - width / 2 for value in x],
            [float(item["baseline_value"]) for item in localization],
            width,
            color="#8797a5",
            label="Страница",
        )
        axis.bar(
            [value + width / 2 for value in x],
            [float(item["candidate_value"]) for item in localization],
            width,
            color="#3ddc97",
            label="Hybrid",
        )
        axis.set_xticks(x, [str(item["label"]) for item in localization])
        axis.set_ylim(0, 105)
        axis.set_ylabel("%")
        axis.set_title("Локализация задания: было → стало")
        legend = axis.legend(facecolor="#151e29", labelcolor="#dce6f0")
        legend.get_frame().set_edgecolor("#334255")
        self.chunking_chart.draw_idle()

        metadata = experiment.get("metadata", {})
        reports = metadata.get("reports", {})
        corpus = reports.get("hybrid_chunking_all_200_v3.json", {})
        refinement = reports.get("hybrid_qwen_refine_500_v3.json", {})
        audit = reports.get("hybrid_qwen_holdout_100_v3_repaired.json", {})
        kinds = corpus.get("unit_kinds", {})
        detail_rows = [
            ["Эксперимент", experiment["name"]],
            ["Дата", experiment["created_at"]],
            ["Учебники / страницы", f"{corpus.get('books', 0)} / {corpus.get('pages', 0):,}".replace(",", " ")],
            ["Theory / exercises / solutions", f"{kinds.get('theory', 0):,} / {kinds.get('exercise', 0):,} / {kinds.get('solution', 0):,}".replace(",", " ")],
            ["Qwen refinement", f"{refinement.get('refined_units', 0)} блоков, изменено {refinement.get('changed_units', 0)}"],
            ["Независимый аудит", f"{audit.get('successful_pages', 0)}/{audit.get('sampled_pages', 0)} страниц, agreement {100 * float(audit.get('agreement_rate', 0)):.1f}%"],
            ["Ограничение", experiment.get("notes") or ""],
        ]
        fill_table(self.chunking_details, detail_rows)

    def refresh_issues(self, *_args) -> None:
        if not hasattr(self, "issues_table"):
            return
        self.issue_cache = self.analytics.issue_rows(
            component=self.issue_component.currentText(),
            owner=self.issue_owner.currentText(),
            status=self.issue_status.currentText(),
        )
        rows = [
            [
                item["component"],
                item["severity"],
                item["title"],
                item["status"],
                item["owner"],
                item["task_id"],
                item["latest_run"],
                item["first_seen_at"],
                item["last_seen_at"],
                item["occurrence_count"],
            ]
            for item in self.issue_cache
        ]

        def colorizer(_: int, column: int, value: Any) -> QColor | None:
            if column != 1:
                return None
            return {
                "Критический": QColor("#773945"),
                "Высокий": QColor("#6b4d2d"),
                "Средний": QColor("#3f4d67"),
                "Низкий": QColor("#254c43"),
            }.get(str(value))

        fill_table(
            self.issues_table,
            rows,
            user_data=list(range(len(rows))),
            colorizer=colorizer,
        )

    def edit_issue(self, *_args) -> None:
        row = self.issues_table.currentRow()
        if row < 0:
            QMessageBox.information(self, APP_NAME, "Выберите факап в таблице.")
            return
        item = self.issues_table.item(row, 0)
        index = item.data(Qt.ItemDataRole.UserRole) if item else None
        if index is None:
            return
        issue = self.issue_cache[int(index)]
        dialog = IssueDialog(issue, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.analytics.update_issue(int(issue["id"]), **dialog.values())
            self.refresh_issues()
            self.refresh_overview()

    def add_manual_issue(self) -> None:
        dialog = ManualIssueDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        if not values["title"]:
            QMessageBox.warning(self, APP_NAME, "Введите название проблемы.")
            return
        self.analytics.add_manual_issue(**values)
        self.refresh_issues()
        self.refresh_overview()

    def refresh_dynamics(self, *_args) -> None:
        if not hasattr(self, "dynamic_chart"):
            return
        metric = self.dynamic_metric.currentData() or "accuracy"
        subject = self.dynamic_subject.currentText()
        records = self.analytics.run_metrics(metric, subject=subject)
        axis = self.dynamic_chart.axis()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(str(record["run_key"]), []).append(record)
        table_rows = []
        for run_key in sorted(grouped, key=lambda key: MODE_ORDER.get(key, 99)):
            points = grouped[run_key]
            previous: float | None = None
            x = list(range(len(points)))
            y = [float(point["value"] or 0) for point in points]
            label = str(points[0]["display_name"])
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                color=MODE_COLORS.get(run_key, "#91a0af"),
                label=label,
            )
            for point, value in zip(points, y):
                delta = "" if previous is None else f"{value - previous:+.1f}"
                table_rows.append(
                    [
                        point["imported_at"],
                        point["display_name"],
                        f"{value:.1f}",
                        point["total"],
                        delta,
                    ]
                )
                previous = value
        axis.set_title(
            f"{self.dynamic_metric.currentText()} — {subject or 'Все предметы'}"
        )
        axis.set_xlabel("Порядковый номер снимка внутри режима")
        if grouped:
            legend = axis.legend(facecolor="#151e29", labelcolor="#dce6f0")
            legend.get_frame().set_edgecolor("#334255")
        self.dynamic_chart.draw_idle()
        fill_table(self.dynamic_table, table_rows)

    def save_settings(self) -> None:
        self.database.set_setting("server", self.setting_server.text().strip())
        self.database.set_setting("user", self.setting_user.text().strip())
        self.database.set_setting("key_path", self.setting_key.text().strip())
        self.sync_log.appendPlainText("Настройки сохранены.")

    def _save_auto_sync(self, *_args) -> None:
        if not hasattr(self, "auto_sync"):
            return
        self.database.set_setting("auto_sync", int(self.auto_sync.isChecked()))
        self.database.set_setting(
            "auto_sync_minutes", self.auto_minutes.value()
        )
        self._update_auto_timer()

    def _update_auto_timer(self) -> None:
        if not hasattr(self, "auto_timer") or not hasattr(self, "auto_sync"):
            return
        self.auto_timer.stop()
        if self.auto_sync.isChecked():
            self.auto_timer.start(self.auto_minutes.value() * 60 * 1000)

    def start_sync(self) -> None:
        if self.sync_worker and self.sync_worker.isRunning():
            self.statusBar().showMessage("Синхронизация уже выполняется.", 3000)
            return
        self.save_settings()
        self.sync_progress.setRange(0, 0)
        self.sync_log.appendPlainText("Начинаю SSH-синхронизацию…")
        self.toolbar_status.setText("Синхронизация…")
        self.sync_worker = SyncWorker(self.sync_manager)
        self.sync_worker.progress.connect(self._sync_progress)
        self.sync_worker.completed.connect(self._sync_completed)
        self.sync_worker.failed.connect(self._sync_failed)
        self.sync_worker.start()

    def _sync_progress(self, message: str) -> None:
        self.sync_log.appendPlainText(message)
        self.statusBar().showMessage(message)

    def _sync_completed(self, results: object) -> None:
        self.sync_progress.setRange(0, 1)
        self.sync_progress.setValue(1)
        imported = list(results)
        new_count = sum(1 for item in imported if item.imported)
        self.sync_log.appendPlainText(
            f"Готово: новых снимков {new_count}, проверено {len(imported)}."
        )
        self.refresh_all()

    def _sync_failed(self, trace: str) -> None:
        self.sync_progress.setRange(0, 1)
        self.sync_progress.setValue(0)
        self.sync_log.appendPlainText(trace)
        self.toolbar_status.setText("Ошибка синхронизации")
        QMessageBox.critical(
            self,
            APP_NAME,
            "Синхронизация завершилась ошибкой. Подробности во вкладке Настройки.",
        )

    def import_local_run(self) -> None:
        manifest, _ = QFileDialog.getOpenFileName(
            self, "Выберите manifest JSONL", "", "JSONL (*.jsonl)"
        )
        if not manifest:
            return
        raw, _ = QFileDialog.getOpenFileName(
            self, "Выберите результаты агента", "", "JSONL (*.jsonl)"
        )
        if not raw:
            return
        judge, _ = QFileDialog.getOpenFileName(
            self, "Выберите результаты джаджа", "", "JSONL (*.jsonl)"
        )
        if not judge:
            return
        modes = {
            "Без тулов": "b0_no_tools",
            "Веб": "web_search",
            "RAG": "agent_rag",
            "RAG thinking": "agent_rag_thinking",
        }
        display_name, ok = QInputDialog.getItem(
            self,
            "Режим",
            "Выберите режим",
            list(modes),
            editable=False,
        )
        if not ok:
            return
        try:
            result = import_run(
                self.database,
                run_key=modes[display_name],
                display_name=display_name,
                raw_path=Path(raw),
                judge_path=Path(judge),
                manifest_path=Path(manifest),
            )
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, str(exc))
            return
        QMessageBox.information(self, APP_NAME, result.message)
        self.refresh_all()


def run_gui(database: Database) -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName(APP_NAME)
    application.setOrganizationName("VLM Team")
    window = MainWindow(database)
    window.show()
    return application.exec()
