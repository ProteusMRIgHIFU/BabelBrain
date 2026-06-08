"""Shared base widget for transducer panels (Step 2 of BabelBrain).

Provides:
  * The common layout shape: left controls column (ending with the
    HideMarks / ShowWaterResults checkboxes) · plot (expanding) over an
    IsppaScrollBars host.
  * Visual styling consistent with MainForm.py (rounded corners, cyan
    accent on hover / focus, OS-palette backgrounds).
  * Helper methods so each subclass can describe its left-panel controls
    in a compact, declarative style.

Subclasses must override `_build_left_panel()` to return a QWidget holding
the transducer-specific controls (typically built via `_make_left_panel`).
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QLabel,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSizePolicy,
)


# ── Accent palette (kept in sync with MainForm.py / nifti_viewer.py) ──────
from GUIComponents.AppStyle import (
    button_border_color, scrollbar_handle_color, disabled_text_color,
    scrollbar_track_color, apply_native_spinbox_style,
)
ACCENT     = "#00c8ff"
LABEL_BLUE = "#166eff"


# Compact look matched to nifti_viewer.py: 11px text, 3px radii, tight padding.
# Built from the active palette so the button border stays visible on dark
# themes (palette(mid) is nearly invisible there).
def _panel_qss(widget=None):
    _border = button_border_color(widget)
    _handle = scrollbar_handle_color(widget)
    _disabled = disabled_text_color(widget)
    _track = scrollbar_track_color(widget)
    return f"""
QLabel {{ font-size: 11px; }}

QPushButton {{
    border: 1px solid {_border};
    border-radius: 3px;
    padding: 3px 8px;
    min-height: 20px;
    font-size: 11px;
}}
QPushButton:hover {{
    border-color: {ACCENT};
    color: {ACCENT};
}}
QPushButton:pressed {{ background: palette(midlight); }}
QPushButton:disabled {{ color: {_disabled}; }}

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

QCheckBox {{ spacing: 5px; font-size: 11px; }}

QScrollBar:horizontal {{ background: {_track}; height: 14px; border-radius: 7px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {_handle}; border-radius: 6px; min-width: 20px; margin: 2px; }}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

QFrame#panelLeftFrame {{ border: none; }}
"""


# ── Public helpers ─────────────────────────────────────────────────────────

def make_dspin(name, value=0.0, minimum=0.0, maximum=100.0,
               decimals=1, step=0.1, width=90):
    sb = QDoubleSpinBox()
    sb.setObjectName(name)
    sb.setDecimals(decimals)
    sb.setMinimum(minimum)
    sb.setMaximum(maximum)
    sb.setSingleStep(step)
    sb.setValue(value)
    sb.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    sb.setMinimumWidth(width)
    return sb


def make_combo(name, items=None, width=110):
    cb = QComboBox()
    cb.setObjectName(name)
    if items:
        for it in items:
            cb.addItem(it)
    cb.setMinimumWidth(width)
    return cb


def make_button(name, text, bold=False, min_height=None):
    btn = QPushButton(text)
    btn.setObjectName(name)
    if bold:
        f = btn.font()
        f.setBold(True)
        btn.setFont(f)
    if min_height:
        btn.setMinimumHeight(min_height)
    return btn


def make_label(text="", name=None, bold=False, color=None, align=None):
    lbl = QLabel(text)
    if name:
        lbl.setObjectName(name)
    if bold:
        f = lbl.font()
        f.setBold(True)
        lbl.setFont(f)
    if color:
        lbl.setStyleSheet(f"color: {color};")
    if align is not None:
        lbl.setAlignment(align)
    return lbl


def form_row(label_widget_or_text, *widgets, spacing=6):
    """Build a horizontal row: [label] [stretch] [widgets…].

    `label_widget_or_text` may be a QLabel or a plain string (turned into a
    QLabel for you).
    """
    row = QHBoxLayout()
    row.setSpacing(spacing)
    if isinstance(label_widget_or_text, str):
        row.addWidget(QLabel(label_widget_or_text))
    else:
        row.addWidget(label_widget_or_text)
    row.addStretch(1)
    for w in widgets:
        row.addWidget(w)
    return row


# ── Base class ────────────────────────────────────────────────────────────

class TxPanelBase(QWidget):
    """Common scaffolding for a Step-2 transducer panel.

    Subclass and implement `_build_left_panel()`. The base class wires up:
        * `self.AcField_plot1`        — plot host (expanding)
        * `self.IsppaScrollBars`      — scrollbar strip below the plot
        * `self.HideMarkscheckBox`    — bottom row, left-aligned
        * `self.ShowWaterResultscheckBox` — bottom row, right-aligned

    Controls are top-aligned in the left column. The checkbox row and the
    scrollbar strip share the bottom grid row (below the plot), so they stay
    vertically aligned.
    """

    LEFT_PANEL_WIDTH = 340

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Widget")
        self.setStyleSheet(_panel_qss(self))
        self._build()
        apply_native_spinbox_style(self)  # Windows: compact stacked spin arrows

    # ── Construction ──────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # A 2×2 grid keeps the checkbox row (col 0) and the scrollbar strip
        # (col 1) on the SAME bottom row, so they stay vertically aligned
        # regardless of how tall the plot grows.
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        # Row 0: controls fill the column (geometry rows top, action buttons +
        # FLHM row pushed to the bottom) · expanding plot.
        left = self._build_left_panel()
        self._anchor_actions_to_bottom(left)
        grid.addWidget(left, 0, 0)
        grid.addWidget(self._build_plot_host(), 0, 1)

        # Row 1: checkboxes under FLHM · scrollbar strip under the plot.
        grid.addLayout(self._build_checkbox_row(), 1, 0, alignment=Qt.AlignTop)
        grid.addWidget(self._build_scrollbar_host(), 1, 1)

        grid.setRowStretch(0, 1)
        grid.setColumnStretch(1, 1)
        root.addLayout(grid, stretch=1)

    # ── Hooks ─────────────────────────────────────────────────────────────
    def _build_left_panel(self):
        raise NotImplementedError("Subclasses must implement _build_left_panel()")

    def _anchor_actions_to_bottom(self, left_frame):
        """Push the action buttons (Calculate Fields / Mechanical Adjustments)
        and the 'Distance target to FLHM' row to the bottom of the left column:
        drop the subclass's trailing stretch and insert one before the first
        action button instead. Relies on the shared `_build_left_panel()` shape
        (geometry rows · CalculateAcField · … · FLHM row · trailing stretch)."""
        lay = left_frame.layout()
        btn = getattr(self, "CalculateAcField", None)
        if lay is None or btn is None:
            return
        # Drop the trailing stretch the subclass added (nothing pads below).
        last = lay.itemAt(lay.count() - 1)
        if last is not None and last.spacerItem() is not None:
            lay.takeAt(lay.count() - 1)
        # Insert a stretch immediately before the first action button so the
        # gap opens up between the geometry controls and the buttons.
        for i in range(lay.count()):
            if lay.itemAt(i).widget() is btn:
                lay.insertStretch(i, 1)
                break

    # ── Common pieces ────────────────────────────────────────────────────
    def _make_left_frame(self):
        frame = QFrame()
        frame.setObjectName("panelLeftFrame")
        frame.setFixedWidth(self.LEFT_PANEL_WIDTH)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        return frame, layout

    def _build_plot_host(self):
        self.AcField_plot1 = QWidget()
        self.AcField_plot1.setObjectName("AcField_plot1")
        self.AcField_plot1.setMinimumSize(500, 280)
        self.AcField_plot1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return self.AcField_plot1

    def _build_scrollbar_host(self):
        self.IsppaScrollBars = QWidget()
        self.IsppaScrollBars.setObjectName("IsppaScrollBars")
        self.IsppaScrollBars.setMinimumSize(400, 61)
        self.IsppaScrollBars.setMaximumHeight(80)
        self.IsppaScrollBars.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return self.IsppaScrollBars

    def _build_checkbox_row(self):
        """HideMarks (left) / ShowWaterResults (right) sit in the bottom-left
        grid cell, on the same grid row as the scrollbar strip, so the two line
        up vertically below the plot."""
        self.HideMarkscheckBox = QCheckBox("Hide marks")
        self.HideMarkscheckBox.setObjectName("HideMarkscheckBox")
        self.HideMarkscheckBox.setEnabled(False)

        self.ShowWaterResultscheckBox = QCheckBox("Show water only")
        self.ShowWaterResultscheckBox.setObjectName("ShowWaterResultscheckBox")
        self.ShowWaterResultscheckBox.setEnabled(False)

        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.HideMarkscheckBox)
        row.addStretch(1)
        row.addWidget(self.ShowWaterResultscheckBox)
        return row
