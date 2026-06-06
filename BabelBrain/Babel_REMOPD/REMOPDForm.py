"""Programmatic Step-2 form for the REMOPD transducer.
"""

from GUIComponents.TxPanelBase import (
    TxPanelBase,
    LABEL_BLUE,
    make_dspin,
    make_combo,
    make_button,
    make_label,
    form_row,
)
from PySide6.QtWidgets import QCheckBox


class REMOPDForm(TxPanelBase):
    def _build_left_panel(self):
        frame, lay = self._make_left_frame()

        # Tx element set
        self.MultifocusLabel = make_label("Tx Elements Set",
                                          name="MultifocusLabel")
        self.SelTxSetDropDown = make_combo(
            "SelTxSetDropDown", items=["Total", "Sector1", "Sector2"])
        lay.addLayout(form_row(self.MultifocusLabel, self.SelTxSetDropDown))

        # Refocusing checkbox
        self.RefocusingcheckBox = QCheckBox()
        self.RefocusingcheckBox.setObjectName("RefocusingcheckBox")
        lay.addLayout(form_row("Refocusing", self.RefocusingcheckBox))

        # Steering
        self.XSteeringSpinBox = make_dspin(
            "XSteeringSpinBox", minimum=-5.0, maximum=5.0)
        lay.addLayout(form_row("XSteering  (mm)", self.XSteeringSpinBox))

        self.YSteeringSpinBox = make_dspin(
            "YSteeringSpinBox", minimum=-5.0, maximum=5.0)
        lay.addLayout(form_row("YSteering  (mm)", self.YSteeringSpinBox))

        self.ZSteeringSpinBox = make_dspin(
            "ZSteeringSpinBox", minimum=-5.0, maximum=5.0)
        lay.addLayout(form_row("ZSteering  (mm)", self.ZSteeringSpinBox))

        self.ZRotationSpinBox = make_dspin(
            "ZRotationSpinBox", minimum=-180.0, maximum=180.0,
            decimals=1, step=5.0)
        lay.addLayout(form_row("Z Rotation (degrees)", self.ZRotationSpinBox))

        lay.addSpacing(6)

        # Device → target display
        self.DistanceSkinLabel = make_label(
            "0.0", name="DistanceSkinLabel", bold=True, color=LABEL_BLUE)
        self.DistanceSkinLabel.setMinimumWidth(60)
        lay.addLayout(form_row(
            make_label("Distance device\nto target (mm) :"),
            self.DistanceSkinLabel))

        lay.addSpacing(6)

        # Mechanical adjustments
        self.XMechanicSpinBox = make_dspin(
            "XMechanicSpinBox", minimum=-10.0, maximum=10.0)
        lay.addLayout(form_row("Mechanical adj. X (mm)", self.XMechanicSpinBox))

        self.YMechanicSpinBox = make_dspin(
            "YMechanicSpinBox", minimum=-10.0, maximum=10.0)
        lay.addLayout(form_row("Mechanical adj. Y (mm)", self.YMechanicSpinBox))

        self.SkinDistanceSpinBox = make_dspin(
            "SkinDistanceSpinBox", minimum=-90.0, maximum=90.0)
        lay.addLayout(form_row(
            make_label("Distance Tx outplane\nto skin (mm)"),
            self.SkinDistanceSpinBox))

        self.MaxDepthSpinBox = make_dspin(
            "MaxDepthSpinBox", value=40.0, minimum=20.0, maximum=100.0,
            decimals=1, step=1.0)
        lay.addLayout(form_row(
            make_label("Max. depth beyond\ntarget (mm)"),
            self.MaxDepthSpinBox))

        lay.addSpacing(8)

        self.CalculateAcField = make_button(
            "CalculateAcField", "Calculate Fields",
            bold=True, min_height=40)
        lay.addWidget(self.CalculateAcField)

        self.CalculateMechAdj = make_button(
            "CalculateMechAdj", "Calculate Mechanical Adjustments",
            bold=True, min_height=40)
        lay.addWidget(self.CalculateMechAdj)

        self.DistanceTargetLabel = make_label(
            "-", name="DistanceTargetLabel", bold=True, color=LABEL_BLUE)
        lay.addLayout(form_row(
            make_label("Distance target to FLHM\ncenter [X, Y, Z] (mm):"),
            self.DistanceTargetLabel))

        self.LabelTissueRemoved = make_label(
            "Tissue layers\nwill be removed!", name="LabelTissueRemoved")
        self.LabelTissueRemoved.setStyleSheet("color: #e03030;")
        lay.addWidget(self.LabelTissueRemoved)

        lay.addStretch(1)
        return frame
