from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QButtonGroup,
    QPushButton,
    QFrame,
)


# Telemetry levels — keep stable; values are persisted in user config.
TELEMETRY_OFF = 0           # Default. Nothing is sent.
TELEMETRY_BASIC = 1         # App start/stop + errors only.
TELEMETRY_STEP_TIMINGS = 2  # Basic + per-step (Steps 1/2/3) execution times.
TELEMETRY_DETAILED = 3      # Step timings + granular timings of demanding sections.

LEVEL_LABELS = {
    TELEMETRY_OFF: "No telemetry (default)",
    TELEMETRY_BASIC: "Basic — notify that the app ran, and report errors",
    TELEMETRY_STEP_TIMINGS: "Basic + execution times of the 3 main simulation steps",
    TELEMETRY_DETAILED: "Basic + step timings + granular timings of the most demanding sections",
}


class TelemetryConsentDialog(QDialog):
    """Opt-in consent dialog for anonymous telemetry collection.

    Shown only on first launch (when no telemetry level is configured yet).
    Users can change the setting later via Advanced Options. The default
    selection is `TELEMETRY_OFF` so closing the dialog without picking a
    different option leaves telemetry disabled.
    """

    def __init__(self, parent=None, current_level=TELEMETRY_OFF):
        super(TelemetryConsentDialog, self).__init__(parent)
        self._level = current_level
        self._initUI(current_level)

    def _initUI(self, current_level):
        self.setWindowTitle("Help Improve BabelBrain")
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(10)

        title = QLabel("Help us improve BabelBrain")
        title_font = title.font()
        title_font.setPointSize(title_font.pointSize() + 2)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        intro = QLabel(
            "BabelBrain is developed with academic funding. To help us demonstrate "
            "real-world use to our funding agencies, we would like to invite you to "
            "share a small amount of anonymous usage information."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        disclaimer = QLabel(
            "<b>Your privacy:</b> data collection is <b>opt-in</b> and is "
            "<b>disabled by default</b>. No personal data, no file contents, no "
            "patient information and <b>no IP address</b> are collected or stored. "
            "Only anonymous execution times and error occurrences are reported, "
            "tagged with a random install identifier that cannot be tied back to you. "
            "You can change or revoke this choice at any time from "
            "<b>Advanced Options</b>."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setTextFormat(Qt.RichText)
        layout.addWidget(disclaimer)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        prompt = QLabel("Please choose the level of information you are comfortable sharing:")
        prompt.setWordWrap(True)
        layout.addWidget(prompt)

        self._group = QButtonGroup(self)
        self._buttons = {}
        for level in (TELEMETRY_OFF, TELEMETRY_BASIC, TELEMETRY_STEP_TIMINGS, TELEMETRY_DETAILED):
            rb = QRadioButton(LEVEL_LABELS[level])
            rb.setStyleSheet("QRadioButton { padding: 2px 0px; }")
            self._group.addButton(rb, level)
            self._buttons[level] = rb
            layout.addWidget(rb)

        if current_level not in self._buttons:
            current_level = TELEMETRY_OFF
        self._buttons[current_level].setChecked(True)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self._continue_btn = QPushButton("Continue")
        self._continue_btn.setDefault(True)
        button_row.addWidget(self._continue_btn)
        layout.addLayout(button_row)

        self._continue_btn.clicked.connect(self._on_accept)

    def _on_accept(self):
        self._level = self._group.checkedId()
        if self._level == -1:
            self._level = TELEMETRY_OFF
        self.accept()

    def selected_level(self):
        """Return the telemetry level selected by the user."""
        return self._level


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dlg = TelemetryConsentDialog()
    dlg.exec()
    print("Selected level:", dlg.selected_level())
