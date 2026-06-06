"""Shared application stylesheet.

A single compact look (matched to nifti_viewer.py: 11px text, 3px radii, tight
padding) used by the programmatic forms and applied to the remaining `.ui`-based
dialogs (SelFiles, Options) so the whole app is visually consistent.

Apply with ``widget.setStyleSheet(APP_QSS)`` — it cascades to all children.
"""

ACCENT     = "#00c8ff"
LABEL_BLUE = "#166eff"


APP_QSS = f"""
QLabel {{ font-size: 11px; }}

QPushButton {{
    border: 1px solid palette(mid);
    border-radius: 3px;
    padding: 3px 10px;
    min-height: 20px;
    font-size: 11px;
}}
QPushButton:hover {{
    border-color: {ACCENT};
    color: {ACCENT};
}}
QPushButton:pressed {{ background: palette(midlight); }}
QPushButton:disabled {{ color: palette(mid); }}

QTabWidget::pane {{
    border: 1px solid palette(mid);
    border-radius: 4px;
    top: -1px;
}}
QTabBar::tab {{
    padding: 5px 12px;
    margin-right: 2px;
    border: 1px solid palette(mid);
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-size: 11px;
}}
QTabBar::tab:selected {{ color: {ACCENT}; font-weight: bold; }}
QTabBar::tab:hover:!selected {{ color: {ACCENT}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    border: 1px solid palette(mid);
    border-radius: 3px;
    padding: 0px 4px;
    min-height: 18px;
    font-size: 11px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid palette(text);
    margin-right: 6px;
}}
QComboBox QLineEdit {{
    border: none;
    background: transparent;
    padding: 0px;
    font-size: 11px;
}}
QComboBox QAbstractItemView {{ font-size: 11px; }}

QCheckBox {{ spacing: 5px; font-size: 11px; }}
QRadioButton {{ spacing: 5px; font-size: 11px; }}

QGroupBox {{
    border: 1px solid palette(mid);
    border-radius: 4px;
    margin-top: 8px;
    font-size: 11px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
}}

QTableView, QTableWidget {{
    border: 1px solid palette(mid);
    border-radius: 3px;
    font-size: 11px;
}}
QHeaderView::section {{
    padding: 2px 4px;
    border: none;
    font-size: 11px;
}}
"""
