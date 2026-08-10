TRACE_STYLESHEET = r"""
QMainWindow, QWidget {
    background: #08111a;
    color: #eaf2fb;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QToolBar {
    background: #0b1621;
    border: none;
    border-bottom: 1px solid #203244;
    padding: 9px 12px;
    spacing: 10px;
}
QLabel#Brand {
    color: #ffffff;
    font-size: 17pt;
    font-weight: 750;
    letter-spacing: 1px;
}
QLabel#BrandAccent { color: #4be1c3; font-size: 17pt; font-weight: 750; }
QLabel#SectionTitle { color: #f7fbff; font-size: 15pt; font-weight: 700; }
QLabel#Subtle { color: #8194a7; }
QLabel#Tiny { color: #708498; font-size: 8.5pt; }
QLabel#Success { color: #67e8b2; font-weight: 650; }
QLabel#Danger { color: #ff8290; font-weight: 650; }
QLabel#Accent { color: #70b7ff; font-weight: 650; }
QFrame#Panel, QFrame#MetricCard, QFrame#AnswerCard, QFrame#EvidenceCard,
QFrame#CompareCard, QFrame#NoticeCard, QFrame#ChronologyCard,
QFrame#ErratumCard {
    background: #0d1924;
    border: 1px solid #203244;
    border-radius: 12px;
}
QFrame#MetricCard { min-height: 78px; }
QFrame#NoticeCard { background: #101b25; border-color: #2d4256; }
QFrame#ChronologyCard { background: #0f1d29; border-color: #1d3448; border-radius: 9px; }
QFrame#ErratumCard { background: #111d28; border-color: #314358; border-radius: 9px; }
QFrame#AnswerCard { background: #0f1d29; }
QLabel#MetricLabel { color: #7f93a7; font-size: 8.5pt; font-weight: 650; }
QLabel#MetricValue { color: #ffffff; font-size: 22pt; font-weight: 750; }
QLabel#MetricHint { color: #6f8498; font-size: 8pt; }
QLabel#TimelineNumber {
    color: #4be1c3;
    background: #12372e;
    border: 1px solid #205e4d;
    border-radius: 17px;
    font-size: 9pt;
    font-weight: 750;
}
QLabel#TimelineTitle { color: #eaf3fa; font-size: 9pt; font-weight: 650; }
QLabel#ErratumCount { color: #ffd68a; font-size: 18pt; font-weight: 750; }
QLabel#HashTrace {
    color: #71869a;
    background: #09141e;
    border: 1px solid #1c3042;
    border-radius: 7px;
    padding: 7px;
    font-family: "Cascadia Mono";
    font-size: 7.5pt;
}
QLabel#HoldoutCaveat {
    color: #ffd68a;
    background: #2b2215;
    border: 1px solid #5a4624;
    border-radius: 8px;
    padding: 8px;
    font-size: 8.5pt;
}
QLabel#TaskTitle { color: #ffffff; font-size: 16pt; font-weight: 720; }
QTextBrowser#AnswerValue {
    color: #4be1c3;
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
    font-size: 15pt;
    font-weight: 700;
}
QLabel#BadgeGood, QLabel#BadgeBad, QLabel#BadgeInfo, QLabel#BadgeWarn,
QLabel#BadgeNeutral {
    border-radius: 9px;
    padding: 3px 8px;
    font-size: 8.5pt;
    font-weight: 700;
}
QLabel#BadgeGood { color: #7df2c0; background: #12372e; border: 1px solid #205e4d; }
QLabel#BadgeBad { color: #ff9ba7; background: #3a1b25; border: 1px solid #66303e; }
QLabel#BadgeInfo { color: #9fcbff; background: #142d47; border: 1px solid #28537c; }
QLabel#BadgeWarn { color: #ffd68a; background: #3a2b16; border: 1px solid #66502a; }
QLabel#BadgeNeutral { color: #aebdcb; background: #172431; border: 1px solid #2b4053; }
QLineEdit, QComboBox, QPlainTextEdit, QTextBrowser, QListWidget {
    background: #09141e;
    color: #e9f1f8;
    border: 1px solid #203549;
    border-radius: 8px;
    padding: 7px;
    selection-background-color: #176b5a;
}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTextBrowser:focus,
QListWidget:focus { border-color: #3aa98f; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background: #0d1924; selection-background-color: #1c5c50; }
QPushButton {
    background: #17283a;
    color: #e9f3fb;
    border: 1px solid #294058;
    border-radius: 8px;
    padding: 7px 12px;
    font-weight: 600;
}
QPushButton:hover { background: #20384e; border-color: #3d5d7b; }
QPushButton:pressed { background: #122233; }
QPushButton#Primary { background: #16745f; border-color: #249c81; color: #ffffff; }
QPushButton#Primary:hover { background: #1b8a71; }
QPushButton#Ghost { background: transparent; }
QPushButton:disabled { color: #506274; background: #101b25; border-color: #1a2a39; }
QTabWidget::pane { border: 1px solid #203244; border-radius: 10px; background: #0b1621; }
QTabBar::tab {
    background: transparent;
    color: #7f93a7;
    padding: 10px 14px;
    margin: 0 2px;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected { color: #f5fbff; border-bottom-color: #4be1c3; }
QTabBar::tab:hover { color: #cde2f2; }
QListWidget { outline: none; padding: 4px; }
QListWidget::item { border: none; margin: 2px; padding: 0; }
QListWidget::item:selected { background: transparent; }
QListWidget#TaskList::item { border-radius: 9px; padding: 8px; border: 1px solid transparent; }
QListWidget#TaskList::item:hover { background: #102333; border-color: #203d54; }
QListWidget#TaskList::item:selected { background: #13362f; border-color: #257a66; }
QListWidget#StepList::item { border-radius: 8px; padding: 9px; margin: 3px; background: #0e1b27; }
QListWidget#StepList::item:selected { background: #153b34; border-left: 3px solid #4be1c3; }
QSplitter::handle { background: #172839; width: 2px; height: 2px; }
QScrollBar:vertical { width: 10px; background: #09131d; }
QScrollBar::handle:vertical { background: #2b4257; border-radius: 5px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: #3b5872; }
QScrollBar:horizontal { height: 9px; background: #09131d; }
QScrollBar::handle:horizontal { background: #2b4257; border-radius: 4px; min-width: 28px; }
QCheckBox { color: #a9bac9; spacing: 7px; }
QCheckBox::indicator { width: 15px; height: 15px; }
QStatusBar { background: #0b1621; color: #778b9e; border-top: 1px solid #203244; }
QToolTip { background: #142433; color: #ffffff; border: 1px solid #34516a; padding: 5px; }
"""
