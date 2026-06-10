"""Programmatic Step-2 form for the REMOPD transducer.

The common bottom half is built by TxPanelBase._build_mech_and_actions().
"""

from GUIComponents.TxPanelBase import (
    TxPanelBase,
    LABEL_BLUE,
    make_dspin,
    make_combo,
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

        self._build_mech_and_actions(
            lay, xy_mech=(-10.0, 10.0), skin_distance=(-90.0, 90.0),
            tissue_warning="Tissue layers\nwill be removed!")
        return frame
