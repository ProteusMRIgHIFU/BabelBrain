"""Shared base widget for transducer panels (Step 2 of BabelBrain).

Provides:
  * The common layout shape: left controls column · plot (expanding) over an
    IsppaScrollBars host · bottom row with HideMarks / ShowWaterResults.
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
    QSizePolicy,
)


# ── Accent palette (kept in sync with MainForm.py / nifti_viewer.py) ──────
ACCENT     = "#00c8ff"
LABEL_BLUE = "#166eff"


_PANEL_QSS = f"""
QPushButton {{
    border: 1px solid palette(mid);
    border-radius: 5px;
    padding: 5px 12px;
    min-height: 22px;
}}
QPushButton:hover {{
    border-color: {ACCENT};
    color: {ACCENT};
}}
QPushButton:pressed {{ background: palette(midlight); }}
QPushButton:disabled {{ color: palette(mid); }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    border: 1px solid palette(mid);
    border-radius: 4px;
    padding: 2px 6px;
    min-height: 20px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}

QCheckBox {{ spacing: 6px; }}

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


def form_row(label_widget_or_text, *widgets, spacing=8):
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
        * `self.HideMarkscheckBox`    — bottom-left checkbox
        * `self.ShowWaterResultscheckBox` — bottom-right checkbox
    """

    LEFT_PANEL_WIDTH = 340

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Widget")
        self.setStyleSheet(_PANEL_QSS)
        self._build()

    # ── Construction ──────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(self._build_left_panel())
        top.addLayout(self._build_plot_column(), stretch=1)
        root.addLayout(top, stretch=1)

        root.addLayout(self._build_bottom_row())

    # ── Hooks ─────────────────────────────────────────────────────────────
    def _build_left_panel(self):
        raise NotImplementedError("Subclasses must implement _build_left_panel()")

    # ── Common pieces ────────────────────────────────────────────────────
    def _make_left_frame(self):
        frame = QFrame()
        frame.setObjectName("panelLeftFrame")
        frame.setFixedWidth(self.LEFT_PANEL_WIDTH)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)
        return frame, layout

    def _build_plot_column(self):
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        self.AcField_plot1 = QWidget()
        self.AcField_plot1.setObjectName("AcField_plot1")
        self.AcField_plot1.setMinimumSize(500, 280)
        self.AcField_plot1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        col.addWidget(self.AcField_plot1, stretch=1)

        self.IsppaScrollBars = QWidget()
        self.IsppaScrollBars.setObjectName("IsppaScrollBars")
        self.IsppaScrollBars.setMinimumSize(400, 61)
        self.IsppaScrollBars.setMaximumHeight(80)
        self.IsppaScrollBars.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        col.addWidget(self.IsppaScrollBars)
        return col

    def _build_bottom_row(self):
        row = QHBoxLayout()
        row.setSpacing(10)

        self.HideMarkscheckBox = QCheckBox("Hide marks")
        self.HideMarkscheckBox.setObjectName("HideMarkscheckBox")
        self.HideMarkscheckBox.setEnabled(False)
        row.addWidget(self.HideMarkscheckBox)

        row.addStretch(1)

        self.ShowWaterResultscheckBox = QCheckBox("Show\nwater only")
        self.ShowWaterResultscheckBox.setObjectName("ShowWaterResultscheckBox")
        self.ShowWaterResultscheckBox.setEnabled(False)
        row.addWidget(self.ShowWaterResultscheckBox)
        return row
