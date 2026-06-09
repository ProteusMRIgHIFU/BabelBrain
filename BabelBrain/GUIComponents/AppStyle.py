"""Shared application stylesheet.

A single compact look (matched to nifti_viewer.py: 11px text, 3px radii, tight
padding) used by the programmatic forms and applied to the remaining `.ui`-based
dialogs (SelFiles, Options) so the whole app is visually consistent.

A couple of colours are palette-aware (resolved at build time from the active
palette), so call the builder when applying:

    widget.setStyleSheet(app_qss(widget))   # cascades to all children

`palette_is_dark`, `selected_tab_color` and `button_border_color` are exported
so the other form stylesheets stay consistent.
"""

import platform

from PySide6.QtCore import QSize
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory, QAbstractSpinBox

_IS_WINDOWS = platform.system() == "Windows"

ACCENT     = "#00c8ff"      # cyan — hover / focus border (both themes)
LABEL_BLUE = "#166eff"
TAB_SELECTED = "#0d47a1"    # dark blue — selected tab text in LIGHT mode


def palette_is_dark(widget=None):
    """True when the active palette is dark (Window lightness < 128)."""
    pal = widget.palette() if widget is not None else None
    if pal is None:
        app = QApplication.instance()
        pal = app.palette() if app is not None else None
    if pal is None:
        return False
    return pal.color(QPalette.Window).lightness() < 128


def selected_tab_color(widget=None):
    """Dark blue on light themes; cyan accent on dark themes (better contrast)."""
    return ACCENT if palette_is_dark(widget) else TAB_SELECTED


def button_border_color(widget=None):
    """`palette(mid)` reads well on light themes but is nearly invisible on dark
    ones, so fall back to the text colour there."""
    return "palette(text)" if palette_is_dark(widget) else "palette(mid)"


def scrollbar_handle_color(widget=None):
    """Scrollbar knob: medium gray on light themes, light gray on dark themes
    (palette(mid) is too dark to spot against a dark track)."""
    return "#9a9a9a" if palette_is_dark(widget) else "palette(mid)"


def disabled_text_color(widget=None):
    """Disabled control text: dim but still legible. palette(mid) is nearly
    invisible on dark themes, so use a mid gray there."""
    return "#808080" if palette_is_dark(widget) else "palette(mid)"


def disabled_input_qss(widget=None):
    """Dark mode only: grey out the text of disabled spin boxes / combos / line
    edits so they visibly read as disabled (matching disabled buttons). Light
    mode already conveys the disabled state natively, so emit nothing there."""
    if not palette_is_dark(widget):
        return ""
    c = disabled_text_color(widget)
    return (
        "QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, "
        f"QComboBox:disabled {{ color: {c}; }}"
    )


def scrollbar_track_color(widget=None):
    """Scrollbar groove. palette(base) barely separates from the window on dark
    themes, so use the slightly lighter palette(mid) there to define the track."""
    return "palette(mid)" if palette_is_dark(widget) else "palette(base)"


_fusion_style = None


def apply_native_spinbox_style(root):
    """Windows only: render every spin box under `root` with the Fusion style.

    The windowsvista style draws stylesheet'd spin boxes with big side-by-side
    buttons that overlap the text and size inconsistently; Fusion draws small,
    vertically stacked, palette-aware arrows with uniform sizing (like macOS).
    Our border/padding QSS still applies on top. No-op on macOS/Linux.

    One shared style instance is kept alive (QWidget.setStyle does not take
    ownership) and reused for all spin boxes.
    """
    if not _IS_WINDOWS:
        return
    global _fusion_style
    if _fusion_style is None:
        _fusion_style = QStyleFactory.create("Fusion")
    if _fusion_style is None:
        return
    for sb in root.findChildren(QAbstractSpinBox):
        sb.setStyle(_fusion_style)


# Compact, flat style for matplotlib's NavigationToolbar2QT (a QToolBar). The
# default toolbar is tall (large icons + padded buttons) and unstyled; this
# shrinks the icons and flattens the buttons so it blends with the app.
_NAV_TOOLBAR_QSS = f"""
QToolBar {{ border: none; padding: 0px; spacing: 1px; background: transparent; }}
QToolButton {{ border: none; padding: 2px; margin: 0px; }}
QToolButton:hover {{ background: palette(midlight); border-radius: 3px; }}
QToolButton:checked {{ background: palette(midlight); border-radius: 3px; }}
QToolBar QLabel {{ font-size: 11px; }}
"""


def style_nav_toolbar(toolbar, icon_px=16, max_height=26):
    """Make a matplotlib NavigationToolbar2QT compact and flat. Returns the
    toolbar so it can be used inline."""
    toolbar.setIconSize(QSize(icon_px, icon_px))
    if max_height:
        toolbar.setMaximumHeight(max_height)
    toolbar.setStyleSheet(_NAV_TOOLBAR_QSS)
    return toolbar


def app_qss(widget=None):
    """Build the shared dialog stylesheet, resolving the palette-aware colours
    (selected-tab text, button border, scrollbar knob, disabled text) from
    `widget`'s active palette."""
    _border = button_border_color(widget)
    _tabsel = selected_tab_color(widget)
    _handle = scrollbar_handle_color(widget)
    _disabled = disabled_text_color(widget)
    _track = scrollbar_track_color(widget)
    _dis_inputs = disabled_input_qss(widget)
    return f"""
QLabel {{ font-size: 11px; }}

QPushButton {{
    border: 1px solid {_border};
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
QPushButton:disabled {{ color: {_disabled}; }}

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
QTabBar::tab:selected {{ color: {_tabsel}; font-weight: bold; }}
QTabBar::tab:hover:!selected {{ color: {ACCENT}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    border: 1px solid {_border};
    border-radius: 3px;
    padding: 0px 4px;
    min-height: 18px;
    font-size: 11px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
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

QScrollBar:horizontal {{ background: {_track}; height: 14px; border-radius: 7px; margin: 0; }}
QScrollBar:vertical {{ background: {_track}; width: 14px; border-radius: 7px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {_handle}; border-radius: 6px; min-width: 20px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {_handle}; border-radius: 6px; min-height: 20px; margin: 2px; }}
QScrollBar::handle:horizontal:hover, QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
{_dis_inputs}
"""
