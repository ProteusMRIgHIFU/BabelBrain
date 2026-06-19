"""Programmatic Step-2 form for the RingTx transducer family.

nherited by R15287 and R15473 which only
tweak label text after construction (`self.Widget.labelTPORange.setText(...)`
etc.) — those attribute names are preserved. The common bottom half is built by
TxPanelBase._build_mech_and_actions().
"""

from GUIComponents.TxPanelBase import (
    TxPanelBase,
    LABEL_BLUE,
    make_dspin,
    make_label,
    form_row,
)


class RingTxForm(TxPanelBase):
    def _build_left_panel(self):
        frame, lay = self._make_left_frame()

        # Z distance / TPO
        self.labelTPODistance = make_label("Z Steering (mm)",
                                           name="labelTPODistance")
        self.TPODistanceSpinBox = make_dspin(
            "TPODistanceSpinBox", minimum=-5.0, maximum=5.0)
        lay.addLayout(form_row(self.labelTPODistance, self.TPODistanceSpinBox))

        self.labelTPORange = make_label("Range Z distance (mm)",
                                        name="labelTPORange")
        self.TPORangeLabel = make_label(
            "[0.0 - 0.0]", name="TPORangeLabel", bold=True, color=LABEL_BLUE)
        self.TPORangeLabel.setMinimumWidth(100)
        lay.addLayout(form_row(self.labelTPORange, self.TPORangeLabel))

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
            lay, xy_mech=(-5.0, 5.0), skin_distance=(-50.0, 50.0))
        return frame
