"""Programmatic Step-2 forms for single-element transducers.

Both forms share the same layout scaffold (TxPanelBase) and differ only in
their left-panel controls (focal length / diameter vs. transducer-model
dropdown). The common bottom half (mechanical adjustments, action buttons,
FLHM readout, warning) is built by TxPanelBase._build_mech_and_actions().
"""

from GUIComponents.TxPanelBase import (
    TxPanelBase,
    LABEL_BLUE,
    make_dspin,
    make_combo,
    make_label,
    form_row,
)


class SingleTxForm(TxPanelBase):
    """Form for the Single transducer family."""

    def _build_left_panel(self):
        frame, lay = self._make_left_frame()

        # Tx geometry
        self.FocalLengthSpinBox = make_dspin(
            "FocalLengthSpinBox", value=50.0, minimum=20.0, maximum=150.0,
            decimals=1, step=0.1)
        self.FocalLengthLabel = make_label("Focal length (mm)",
                                           name="FocalLengthLabel")
        lay.addLayout(form_row(self.FocalLengthLabel, self.FocalLengthSpinBox))

        self.DiameterSpinBox = make_dspin(
            "DiameterSpinBox", value=50.0, minimum=20.0, maximum=150.0,
            decimals=1, step=0.1)
        self.DiameterLabel = make_label("Diameter (mm)", name="DiameterLabel")
        lay.addLayout(form_row(self.DiameterLabel, self.DiameterSpinBox))

        # Computed outplane-to-focus distance
        self.DistanceOutplaneLabel = make_label(
            "0.0", name="DistanceOutplaneLabel", bold=True, color=LABEL_BLUE)
        self.DistanceOutplaneLabel.setMinimumWidth(60)
        lay.addLayout(form_row(
            make_label("Distance from Tx's\noutplane to focus"),
            self.DistanceOutplaneLabel))

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
            lay, xy_mech=(-5.0, 5.0), skin_distance=(-5.0, 5.0))
        return frame


class BSonixForm(TxPanelBase):
    """Form for the BSonix variant (formerly formBx.ui).

    Differs from SingleTx by replacing focal length / diameter inputs with a
    transducer-model dropdown (35/55/65/80 mm).
    """

    def _build_left_panel(self):
        frame, lay = self._make_left_frame()

        # Transducer model selector
        self.TransducerModelcomboBox = make_combo(
            "TransducerModelcomboBox",
            items=["35mm", "55mm", "65mm", "80mm"])
        lay.addLayout(form_row("Transducer model", self.TransducerModelcomboBox))

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
            lay, xy_mech=(-5.0, 5.0), skin_distance=(-25.0, 5.0))
        return frame
