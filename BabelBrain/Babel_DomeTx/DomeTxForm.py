"""Programmatic Step-2 form for the DomeTx phased-array transducer.

 All widget access goes through
_BabelBasePhasedArray which checks `hasattr` for optional widgets like
DistanceConeToFocusSpinBox — that is intentionally omitted here. The common
bottom half is built by TxPanelBase._build_mech_and_actions().
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


class DomeTxForm(TxPanelBase):
    def _build_left_panel(self):
        frame, lay = self._make_left_frame()

        # Multifocus combination
        self.MultifocusLabel = make_label("Multi-focus", name="MultifocusLabel")
        self.SelCombinationDropDown = make_combo(
            "SelCombinationDropDown", items=["ALL"])
        lay.addLayout(form_row(self.MultifocusLabel, self.SelCombinationDropDown))

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

        # Skin → target display
        self.DistanceSkinLabel = make_label(
            "0.0", name="DistanceSkinLabel", bold=True, color=LABEL_BLUE)
        self.DistanceSkinLabel.setMinimumWidth(60)
        lay.addLayout(form_row(
            make_label("Distance skin\nto target (mm) :"),
            self.DistanceSkinLabel))

        lay.addSpacing(6)

        self._build_mech_and_actions(
            lay, xy_mech=(-10.0, 10.0), z_mechanic=(-90.0, 90.0),
            tissue_warning=None)
        return frame
