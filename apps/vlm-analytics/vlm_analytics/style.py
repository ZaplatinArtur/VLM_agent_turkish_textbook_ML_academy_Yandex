APP_STYLESHEET = """
QMainWindow, QWidget {
    background: #0f141b;
    color: #e7edf4;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QToolBar {
    background: #121a24;
    border: none;
    border-bottom: 1px solid #263241;
    spacing: 8px;
    padding: 7px;
}
QToolBar QLabel#AppTitle {
    font-size: 16pt;
    font-weight: 700;
    color: #f5f8fb;
}
QTabWidget::pane {
    border: 1px solid #263241;
    background: #0f141b;
}
QTabBar::tab {
    background: #151e29;
    color: #9eabb9;
    padding: 10px 16px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected {
    background: #223043;
    color: #ffffff;
}
QTabBar::tab:hover {
    background: #1d2937;
}
QFrame#MetricCard, QGroupBox {
    background: #151e29;
    border: 1px solid #263241;
    border-radius: 9px;
}
QGroupBox {
    margin-top: 12px;
    padding: 13px 8px 8px 8px;
    font-weight: 600;
    color: #dce6f0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QLabel#MetricValue {
    font-size: 22pt;
    font-weight: 700;
    color: #ffffff;
}
QLabel#MetricLabel {
    color: #91a0af;
    font-size: 9pt;
}
QLabel#SectionTitle {
    font-size: 14pt;
    font-weight: 650;
    color: #ffffff;
}
QLabel#Muted {
    color: #8492a2;
}
QPushButton {
    background: #253449;
    color: #edf4fb;
    border: 1px solid #344860;
    border-radius: 6px;
    padding: 7px 12px;
}
QPushButton:hover {
    background: #30445e;
}
QPushButton:pressed {
    background: #1c2939;
}
QPushButton#Primary {
    background: #17865f;
    border-color: #22a876;
}
QPushButton#Primary:hover {
    background: #1f9a70;
}
QPushButton#Danger {
    background: #763642;
    border-color: #9c4b5a;
}
QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QTextEdit {
    background: #101821;
    border: 1px solid #2b3a4c;
    border-radius: 6px;
    padding: 6px;
    selection-background-color: #2f765f;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QTableWidget, QTableView {
    background: #111923;
    alternate-background-color: #151f2b;
    border: 1px solid #263241;
    border-radius: 7px;
    gridline-color: #23303e;
    selection-background-color: #285b4b;
    selection-color: #ffffff;
}
QHeaderView::section {
    background: #1b2735;
    color: #bdc9d5;
    padding: 7px;
    border: none;
    border-right: 1px solid #2a394a;
    border-bottom: 1px solid #2a394a;
    font-weight: 600;
}
QScrollBar:vertical {
    width: 11px;
    background: #101720;
}
QScrollBar::handle:vertical {
    background: #344457;
    border-radius: 5px;
    min-height: 24px;
}
QStatusBar {
    background: #121a24;
    color: #91a0af;
    border-top: 1px solid #263241;
}
QProgressBar {
    border: 1px solid #2c3c4e;
    border-radius: 5px;
    background: #101720;
    text-align: center;
}
QProgressBar::chunk {
    background: #27a879;
    border-radius: 4px;
}
QSplitter::handle {
    background: #263241;
}
"""
