"""Programmatic Step-2 form for the H246 transducer.


"""

from GUIComponents.TxPanelBase import (
    TxPanelBase,
    LABEL_BLUE,
    make_dspin,
    make_button,
    make_label,
    form_row,
)


class H246Form(TxPanelBase):
    def _build_left_panel(self):
        frame, lay = self._make_left_frame()

        # Z distance / TPO display
        self.TPODistanceSpinBox = make_dspin(
            "TPODistanceSpinBox", minimum=-5.0, maximum=5.0)
        lay.addLayout(form_row("Z Distance (mm)", self.TPODistanceSpinBox))

        self.TPORangeLabel = make_label(
            "[0.0 - 0.0]", name="TPORangeLabel", bold=True, color=LABEL_BLUE)
        self.TPORangeLabel.setMinimumWidth(100)
        lay.addLayout(form_row("Range Z distance (mm)", self.TPORangeLabel))

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
            "XMechanicSpinBox", minimum=-5.0, maximum=5.0)
        lay.addLayout(form_row("Mechanical adj. X (mm)", self.XMechanicSpinBox))

        self.YMechanicSpinBox = make_dspin(
            "YMechanicSpinBox", minimum=-5.0, maximum=5.0)
        lay.addLayout(form_row("Mechanical adj. Y (mm)", self.YMechanicSpinBox))

        self.SkinDistanceSpinBox = make_dspin(
            "SkinDistanceSpinBox", minimum=-35.0, maximum=0.0)
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
            "Tissue layers will be removed!", name="LabelTissueRemoved")
        self.LabelTissueRemoved.setStyleSheet("color: #e03030;")
        lay.addWidget(self.LabelTissueRemoved)

        lay.addStretch(1)
        return frame
