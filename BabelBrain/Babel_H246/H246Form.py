"""Programmatic Step-2 form for the H246 transducer.

The common bottom half is built by TxPanelBase._build_mech_and_actions().
"""

from GUIComponents.TxPanelBase import (
    TxPanelBase,
    LABEL_BLUE,
    make_dspin,
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

        self._build_mech_and_actions(
            lay, xy_mech=(-5.0, 5.0), skin_distance=(-35.0, 0.0))
        return frame
