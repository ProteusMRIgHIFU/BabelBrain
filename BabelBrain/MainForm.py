"""Programmatically constructed main form for BabelBrain.
  * ZTE tab exposes a QVBoxLayout (LayRange) discoverable via
    `ZTE.findChildren(QVBoxLayout)[0]` — used to insert the dynamic
    QLabeledDoubleRangeSlider at runtime.
  * ZTE tab has a QLabel(objectName="RangeLabel").
  * CT  tab has a QLabel(objectName="HULabel") and a QDoubleSpinBox
    (HUThresholdSpinBox).
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QTabWidget,
    QTextBrowser,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QScrollBar,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSizePolicy,
)

from GUIComponents.AppStyle import (
    selected_tab_color, button_border_color,
    scrollbar_handle_color, disabled_text_color,
)


# ── Accent palette (mirrors GUIComponents/nifti_viewer.py) ─────────────────
ACCENT     = "#00c8ff"      # cyan — hover / focus border
LABEL_BLUE = "#166eff"      # blue used by the original form for value labels
TEXT_LBL   = "#ffda6b"      # yellow accent (reserved for future use)
# Selected-tab text colour is palette-aware via AppStyle.selected_tab_color().


# Form-level stylesheet. Only sets shape (rounded corners, padding, accent
# highlight) — never sets a hard-coded background colour, so the OS palette
# still drives surfaces. A couple of colours are palette-aware (button border,
# selected-tab text), so this is built from the active palette at runtime.
# Compact look matched to nifti_viewer.py: 11px text, 3px radii, tight padding.
def _form_qss(widget=None):
    _border = button_border_color(widget)
    _tabsel = selected_tab_color(widget)
    _handle = scrollbar_handle_color(widget)
    _disabled = disabled_text_color(widget)
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
QPushButton:pressed {{
    background: palette(midlight);
}}
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
QTabBar::tab::disabled {{
    width: 0; height: 0; margin: 0; padding: 0; border: none;
}}

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
/* A stylesheet-rendered combo with a customized ::drop-down draws no arrow
   unless ::down-arrow is given — draw one with a CSS triangle (no image). */
QComboBox::down-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid palette(text);
    margin-right: 6px;
}}
/* Editable combos (e.g. USMaskkHzDropDown) render via an internal QLineEdit,
   and the popup list is a separate view — neither inherits the combo font
   unless targeted explicitly. */
QComboBox QLineEdit {{
    border: none;
    background: transparent;
    padding: 0px;
    font-size: 11px;
}}
QComboBox QAbstractItemView {{ font-size: 11px; }}

QTextBrowser {{
    border: 1px solid palette(mid);
    border-radius: 3px;
    font-size: 11px;
}}

QCheckBox {{ spacing: 5px; font-size: 11px; }}

QScrollBar:horizontal {{
    background: palette(base);
    height: 14px;
    border-radius: 7px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {_handle};
    border-radius: 6px;
    min-width: 20px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{ background: {ACCENT}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""


def _bold(widget, point_delta=0):
    f = widget.font()
    f.setBold(True)
    if point_delta:
        f.setPointSize(f.pointSize() + point_delta)
    widget.setFont(f)
    return widget


def _header_label(text):
    """Section header — bold."""
    return _bold(QLabel(text))


def _value_label(name, color=LABEL_BLUE, point_delta=2, min_width=110):
    """Dynamic-value label (Target ID, Tx system, profile name…)."""
    lbl = QLabel("")
    lbl.setObjectName(name)
    # Slightly larger + bold so dynamic values stand out from the 11px base.
    # Set via stylesheet (not setFont) so it wins over the QLabel rule in QSS.
    lbl.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: bold;")
    lbl.setMinimumWidth(min_width)
    return lbl


class BabelBrainMainForm(QWidget):
    """Top-level form for BabelBrain — programmatically constructed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Widget")  # preserves original root name
        self.setStyleSheet(_form_qss(self))
        self._build()

    # ── Construction ──────────────────────────────────────────────────────
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        root.addLayout(self._build_top_bar())
        root.addWidget(self._build_tab_widget(), stretch=1)
        root.addWidget(self._build_log_header())
        root.addWidget(self._build_log_view())

        self.setMinimumSize(1200, 720)

    # Top bar -----------------------------------------------------------------
    def _build_top_bar(self):
        lay = QHBoxLayout()
        lay.setSpacing(8)

        lay.addWidget(_header_label("Target"))
        self.IDLabel = _value_label("IDLabel", min_width=110)
        lay.addWidget(self.IDLabel)

        lay.addSpacing(20)
        lay.addWidget(_header_label("TUS System"))
        self.TXLabel = _value_label("TXLabel", min_width=110)
        lay.addWidget(self.TXLabel)

        lay.addSpacing(20)
        lay.addWidget(_header_label("Thermal profile"))
        self.ThermalProfileLabel = _value_label("ThermalProfileLabel",
                                                min_width=200)
        lay.addWidget(self.ThermalProfileLabel)

        lay.addStretch(1)

        self.vtkVisualizationqPushButton = QPushButton("VTK visualization")
        self.vtkVisualizationqPushButton.setObjectName("vtkVisualizationqPushButton")
        lay.addWidget(self.vtkVisualizationqPushButton)

        self.AdvancedOptions = QPushButton("Advanced Options")
        self.AdvancedOptions.setObjectName("AdvancedOptions")
        lay.addWidget(self.AdvancedOptions)

        return lay

    # Tab widget --------------------------------------------------------------
    def _build_tab_widget(self):
        self.tabWidget = QTabWidget()
        self.tabWidget.setObjectName("tabWidget")
        self.tabWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabWidget.setMinimumSize(1180, 520)
        # macOS elides tab labels to "…" by default; show them in full.
        self.tabWidget.tabBar().setElideMode(Qt.ElideNone)
        # Step 1 is owned by this form; Step 2 / Step 3 are added at runtime
        # by BabelBrain.load_ui via tabWidget.addTab(...).
        self.tabWidget.addTab(self._build_step1_tab(), "Step 1 - Calculate Mask")
        return self.tabWidget

    # Step 1 ------------------------------------------------------------------
    def _build_step1_tab(self):
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        # ── Left control panel ─────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(260)
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 4, 0, 0)
        # Spread the input rows out (~2–3× a label's height) and nudge the whole
        # block down from the top so the column isn't mostly empty for CT/ZTE.
        left_l.setSpacing(18)
        left_l.addStretch(1)

        # US Frequency row
        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel("US Frequency (kHz)"))
        freq_row.addStretch(1)
        self.USMaskkHzDropDown = QComboBox()
        self.USMaskkHzDropDown.setObjectName("USMaskkHzDropDown")
        self.USMaskkHzDropDown.addItem("500")
        self.USMaskkHzDropDown.setMinimumWidth(100)
        freq_row.addWidget(self.USMaskkHzDropDown)
        left_l.addLayout(freq_row)

        # PPW row
        ppw_row = QHBoxLayout()
        ppw_row.addWidget(QLabel("PPW"))
        ppw_row.addStretch(1)
        self.USPPWSpinBox = QDoubleSpinBox()
        self.USPPWSpinBox.setObjectName("USPPWSpinBox")
        self.USPPWSpinBox.setDecimals(0)
        self.USPPWSpinBox.setMinimum(6)
        self.USPPWSpinBox.setMaximum(12)
        self.USPPWSpinBox.setSingleStep(1)
        self.USPPWSpinBox.setValue(6)
        self.USPPWSpinBox.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.USPPWSpinBox.setMinimumWidth(90)
        ppw_row.addWidget(self.USPPWSpinBox)
        left_l.addLayout(ppw_row)

        # CT / ZTE nested tabs
        self.CTZTETabs = QTabWidget()
        self.CTZTETabs.setObjectName("CTZTETabs")
        self.CTZTETabs.setMinimumSize(240, 160)
        self.CTZTETabs.tabBar().setElideMode(Qt.ElideNone)
        self.CTZTETabs.addTab(self._build_zte_tab(), "ZTE")
        self.CTZTETabs.addTab(self._build_ct_tab(),  "CT")
        self.CTZTETabs.setCurrentIndex(1)
        left_l.addWidget(self.CTZTETabs)

        # Larger bottom stretch keeps the block in the upper portion (a little
        # lower than the top) while the action button stays pinned at the bottom.
        left_l.addStretch(2)

        # Calculate planning mask button
        self.CalculatePlanningMask = QPushButton("Calculate planning\nmask")
        self.CalculatePlanningMask.setObjectName("CalculatePlanningMask")
        self.CalculatePlanningMask.setMinimumHeight(64)
        _bold(self.CalculatePlanningMask)
        left_l.addWidget(self.CalculatePlanningMask)

        grid.addWidget(left, 0, 0)

        # ── Plot area ───────────────────────────────────────────────────
        self.USMask = QWidget()
        self.USMask.setObjectName("USMask")
        self.USMask.setMinimumSize(620, 380)
        self.USMask.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        grid.addWidget(self.USMask, 0, 1)

        # ── Bottom row ──────────────────────────────────────────────────
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        self.HideMarkscheckBox = QCheckBox("Hide marks")
        self.HideMarkscheckBox.setObjectName("HideMarkscheckBox")
        self.HideMarkscheckBox.setEnabled(False)
        bottom.addWidget(self.HideMarkscheckBox)
        bottom.addStretch(1)

        bottom.addWidget(QLabel("T1W transparency"))
        self.TransparencyScrollBar = QScrollBar(Qt.Horizontal)
        self.TransparencyScrollBar.setObjectName("TransparencyScrollBar")
        self.TransparencyScrollBar.setMaximum(100)
        self.TransparencyScrollBar.setValue(50)
        self.TransparencyScrollBar.setEnabled(False)
        self.TransparencyScrollBar.setMinimumWidth(180)
        self.TransparencyScrollBar.setMaximumWidth(240)
        bottom.addWidget(self.TransparencyScrollBar)

        grid.addLayout(bottom, 1, 0, 1, 2)

        # Column stretch: plot takes all extra horizontal space
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 0)

        return page

    # CT / ZTE inner tabs -----------------------------------------------------
    def _build_zte_tab(self):
        """ZTE tab — contains the range label + the QVBoxLayout into which
        BabelBrain.load_ui inserts the QLabeledDoubleRangeSlider at runtime
        (located via `ZTE.findChildren(QVBoxLayout)[0]`)."""
        page = QWidget()
        page_l = QVBoxLayout(page)
        page_l.setObjectName("LayRange")
        page_l.setContentsMargins(8, 8, 8, 8)
        page_l.setSpacing(8)

        range_lbl = QLabel("Normalized ZTE Range")
        range_lbl.setObjectName("RangeLabel")
        range_lbl.setAlignment(Qt.AlignCenter)
        page_l.addWidget(range_lbl)
        page_l.addStretch(1)
        return page

    def _build_ct_tab(self):
        page = QWidget()
        page_l = QVBoxLayout(page)
        page_l.setContentsMargins(8, 8, 8, 8)
        page_l.setSpacing(8)

        hu_lbl = QLabel("HU threshold")
        hu_lbl.setObjectName("HULabel")
        hu_lbl.setAlignment(Qt.AlignCenter)
        page_l.addWidget(hu_lbl)

        self.HUThresholdSpinBox = QDoubleSpinBox()
        self.HUThresholdSpinBox.setObjectName("HUThresholdSpinBox")
        self.HUThresholdSpinBox.setDecimals(0)
        self.HUThresholdSpinBox.setMaximum(5000)
        self.HUThresholdSpinBox.setSingleStep(10)
        self.HUThresholdSpinBox.setValue(300)
        self.HUThresholdSpinBox.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        page_l.addWidget(self.HUThresholdSpinBox)
        page_l.addStretch(1)
        return page

    # Log section -------------------------------------------------------------
    def _build_log_header(self):
        return _bold(QLabel("Terminal output"))

    def _build_log_view(self):
        self.outputTerminal = QTextBrowser()
        self.outputTerminal.setObjectName("outputTerminal")
        self.outputTerminal.setReadOnly(True)
        self.outputTerminal.setMinimumHeight(120)
        self.outputTerminal.setSizePolicy(QSizePolicy.Expanding,
                                          QSizePolicy.Preferred)
        return self.outputTerminal
