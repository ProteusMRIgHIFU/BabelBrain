"""Programmatic Step-3 thermal panel form.

 Structurally different from the Step-2
transducer panels (different control set, two-row bottom strip with Loc
buttons), so this class extends QWidget directly rather than TxPanelBase.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QLabel,
    QPushButton,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QScrollBar,
    QTableWidget,
    QHeaderView,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QSizePolicy,
)

from GUIComponents.TxPanelBase import (
    LABEL_BLUE,
    make_dspin,
    make_combo,
    make_button,
    make_label,
)
from GUIComponents.AppStyle import (
    button_border_color, scrollbar_handle_color, disabled_text_color,
    scrollbar_track_color, apply_native_spinbox_style, disabled_input_qss,
)


# Compact look matched to nifti_viewer.py: 11px text, 3px radii, tight padding.
# Built from the active palette so the button border stays visible on dark
# themes (palette(mid) is nearly invisible there).
def _thermal_qss(widget=None):
    _border = button_border_color(widget)
    _handle = scrollbar_handle_color(widget)
    _disabled = disabled_text_color(widget)
    _track = scrollbar_track_color(widget)
    _dis_inputs = disabled_input_qss(widget)
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
    border-color: #00c8ff;
    color: #00c8ff;
}}
QPushButton:pressed {{ background: palette(midlight); }}
QPushButton:disabled {{ color: {_disabled}; }}
/* Two-line labels: a stylesheet min-height overrides setMinimumHeight(), so the
   extra room is set here. 30px clears two 13px lines while staying compact. */
QPushButton#ExportSummary, QPushButton#ExportMaps {{ min-height: 30px; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    border: 1px solid {_border};
    border-radius: 3px;
    padding: 0px 4px;
    min-height: 18px;
    font-size: 11px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid #00c8ff;
}}

QCheckBox {{ spacing: 5px; font-size: 11px; }}
QFrame#panelLeftFrame, QFrame#panelBottomFrame {{ border: none; }}
QTableWidget {{
    background: transparent;
    border: 1px solid palette(mid);
    border-radius: 3px;
    font-size: 11px;
}}
QTableWidget::item {{ background: transparent; }}

QScrollBar:horizontal {{ background: {_track}; height: 14px; border-radius: 7px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {_handle}; border-radius: 6px; min-width: 20px; margin: 2px; }}
QScrollBar::handle:horizontal:hover {{ background: #00c8ff; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
{_dis_inputs}
"""


class ThermalForm(QWidget):
    """Step-3 thermal-simulation panel."""

    LEFT_PANEL_WIDTH = 380

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Babel_Thermal")
        self.setStyleSheet(_thermal_qss(self))
        self._build()
        apply_native_spinbox_style(self)  # Windows: compact stacked spin arrows

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Top: left control panel + plot
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self._build_left_panel())
        top.addWidget(self._build_plot(), stretch=1)
        root.addLayout(top, stretch=1)

        # Bottom area (two rows of controls)
        root.addWidget(self._build_bottom_panel())

    # Left -------------------------------------------------------------------
    def _build_left_panel(self):
        frame = QFrame()
        frame.setObjectName("panelLeftFrame")
        frame.setFixedWidth(self.LEFT_PANEL_WIDTH)

        lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 4, 0, 0)
        # Tight row spacing leaves the results table as much height as possible.
        lay.setSpacing(2)

        # Top action buttons (Calculate Thermal / Update Profile)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.CalculateThermal = make_button(
            "CalculateThermal", "Calculate Thermal Fields",
            bold=True, min_height=40)
        self.CalculateThermal.setStyleSheet("color: #e03030;")
        actions.addWidget(self.CalculateThermal, stretch=1)

        self.SelectProfile = make_button(
            "SelectProfile", "Update Profile and Calculate",
            bold=True, min_height=40)
        self.SelectProfile.setStyleSheet("color: #2db52d;")
        actions.addWidget(self.SelectProfile, stretch=1)
        lay.addLayout(actions)

        # Combination timing + duration/DC/PRF triplet
        comb_row = QHBoxLayout()
        comb_row.setSpacing(8)
        comb_row.addWidget(QLabel("Combination timing"))
        comb_row.addStretch(1)
        self.SelCombinationDropDown = make_combo(
            "SelCombinationDropDown", items=None, width=140)
        comb_row.addWidget(self.SelCombinationDropDown)
        lay.addLayout(comb_row)

        triplet_lbl = QLabel("[Duration, DC, PRF]")
        triplet_lbl.setAlignment(Qt.AlignRight)
        lay.addWidget(triplet_lbl)

        # Isppa
        isppa_row = QHBoxLayout()
        isppa_row.addWidget(QLabel("Isppa (W/cm²)"))
        isppa_row.addStretch(1)
        self.IsppaSpinBox = make_dspin(
            "IsppaSpinBox", value=5.0, minimum=0.1, maximum=60.0,
            decimals=2, step=0.1, width=110)
        isppa_row.addWidget(self.IsppaSpinBox)
        lay.addLayout(isppa_row)

        # Isppa in Water
        isppaw_row = QHBoxLayout()
        isppaw_row.addWidget(QLabel("Isppa in Water (W/cm²)"))
        isppaw_row.addStretch(1)
        self.IsppaWaterSpinBox = make_dspin(
            "IsppaWaterSpinBox", value=5.0, minimum=0.1, maximum=1000.0,
            decimals=2, step=0.1, width=110)
        isppaw_row.addWidget(self.IsppaWaterSpinBox)
        lay.addLayout(isppaw_row)

        # Babel_Thermal.py relies on (it calls `setItem(0..10, 0/1, …)`
        # without first growing the table).
        self.tableWidget = QTableWidget(11, 2)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setShowGrid(False)
        _hdr = self.tableWidget.horizontalHeader()
        _hdr.setVisible(False)
        # Fit the columns to the viewport (no horizontal scrollbar): the label
        # column sizes to its contents, the value column fills the remainder.
        _hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        _hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setMinimumHeight(150)
        self.tableWidget.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding)
        lay.addWidget(self.tableWidget, stretch=1)

        return frame

    # Plot -------------------------------------------------------------------
    def _build_plot(self):
        self.AcField_plot1 = QWidget()
        self.AcField_plot1.setObjectName("AcField_plot1")
        self.AcField_plot1.setMinimumSize(600, 360)
        self.AcField_plot1.setSizePolicy(QSizePolicy.Expanding,
                                         QSizePolicy.Expanding)
        return self.AcField_plot1

    # Bottom -----------------------------------------------------------------
    def _build_bottom_panel(self):
        frame = QFrame()
        frame.setObjectName("panelBottomFrame")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        outer = QVBoxLayout(frame)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # ── Controls ────────────────────────────────────────────────────────
        # Height comes from the QSS #ExportSummary/#ExportMaps rule (two-line
        # labels); the value below is a no-op floor kept for clarity.
        self.ExportSummary = make_button(
            "ExportSummary", "Export summary\n(CSV)", min_height=40)
        self.ExportSummary.setEnabled(False)

        self.ExportMaps = make_button(
            "ExportMaps", "Export maps\n(.nii.gz)", min_height=40)
        self.ExportMaps.setEnabled(False)

        self.label_22 = make_label("Show", name="label_22")

        self.DisplayDropDown = make_combo(
            "DisplayDropDown", items=["Maps", "Profiles"])

        self.IsppaScrollBar = QScrollBar(Qt.Horizontal)
        self.IsppaScrollBar.setObjectName("IsppaScrollBar")
        self.IsppaScrollBar.setEnabled(False)
        self.IsppaScrollBar.setMinimumWidth(220)
        self.IsppaScrollBar.setSizePolicy(QSizePolicy.Expanding,
                                          QSizePolicy.Fixed)

        self.SliceLabel = make_label("-", name="SliceLabel",
                                     bold=True, color=LABEL_BLUE)
        self.SliceLabel.setMinimumWidth(80)

        self.HideMarkscheckBox = QCheckBox("Hide marks")
        self.HideMarkscheckBox.setObjectName("HideMarkscheckBox")
        self.HideMarkscheckBox.setEnabled(False)

        self.LocMTB = make_button("LocMTB", "Max. Temp. Brain")
        self.LocMTS = make_button("LocMTS", "Max. Temp. Skin")
        self.LocMTC = make_button("LocMTC", "Max. Temp. Skull")

        # Both bottom rows share the top row's column split: a fixed-width left
        # zone sits under the control column, the rest aligns under the plot.
        # Row 1: export/show/display under the column · scrollbar (starting at
        #        the plot's left edge) + slice/hide to its right under the plot.
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        left1 = QWidget()
        left1.setFixedWidth(self.LEFT_PANEL_WIDTH)
        left1_l = QHBoxLayout(left1)
        left1_l.setContentsMargins(0, 0, 0, 0)
        left1_l.setSpacing(8)
        left1_l.addWidget(self.ExportSummary)
        left1_l.addWidget(self.ExportMaps)
        left1_l.addWidget(self.label_22)
        left1_l.addWidget(self.DisplayDropDown)
        left1_l.addStretch(3)
        row1.addWidget(left1)

        row1.addWidget(self.IsppaScrollBar, stretch=1)
        row1.addWidget(self.SliceLabel)
        row1.addWidget(self.HideMarkscheckBox)
        outer.addLayout(row1)

        # Row 2: Max-Temp buttons centred under the plot (left zone is an empty
        # spacer matching the control column so the centring excludes it).
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        left2 = QWidget()
        left2.setFixedWidth(self.LEFT_PANEL_WIDTH)
        row2.addWidget(left2)

        row2.addStretch(1)
        row2.addWidget(self.LocMTB)
        row2.addWidget(self.LocMTS)
        row2.addWidget(self.LocMTC)
        row2.addStretch(1)
        outer.addLayout(row2)

        return frame
