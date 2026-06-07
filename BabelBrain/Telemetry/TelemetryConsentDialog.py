from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
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
TELEMETRY_BASIC = 1        
TELEMETRY_STEP_TIMINGS = 2  
TELEMETRY_DETAILED = 3      
TELEMETRY_HIGHLY_DETAILED = 4      

LEVEL_LABELS = {
    TELEMETRY_OFF: "L0: No telemetry (default)",
    TELEMETRY_BASIC: "L1: Basic — notify that the app ran, CPU, OS, main memory, GPU model and errors",
    TELEMETRY_STEP_TIMINGS: "L2: L1  + execution times of the 3 main simulation steps",
    TELEMETRY_DETAILED: "L3: L2 + Frequency, PPW, domain size and granular timings of the most demanding sections",
    TELEMETRY_HIGHLY_DETAILED: "L4: L3 + Tx model, total duration (with no details of timing)",
}


class TelemetrySettingsWidget(QWidget):
    """Reusable widget with the telemetry consent text and level selector.

    Embedded in both the first-launch consent dialog and the Advanced Options
    dialog (Telemetry tab). Use `selected_level()` / `set_level()` to read or
    update the current choice.
    """

    def __init__(self, parent=None, current_level=TELEMETRY_OFF, show_title=True):
        super(TelemetrySettingsWidget, self).__init__(parent)
        self._build(current_level, show_title)

    def _build(self, current_level, show_title):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        if show_title:
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
            "The collected data is particularly useful to understand the performance "
            "of the tool.Depending on the level of data collection, this can include "
            "from basic hardware information, anonymous execution times, to the "
            "selected device and total duration (no details of timing are sent). This "
            "data is tagged with a random install identifier that cannot be tied back "
            "to you. "
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
        for level in (TELEMETRY_OFF, TELEMETRY_BASIC, TELEMETRY_STEP_TIMINGS, TELEMETRY_DETAILED,TELEMETRY_HIGHLY_DETAILED):
            rb = QRadioButton(LEVEL_LABELS[level])
            rb.setStyleSheet("QRadioButton { padding: 2px 0px; }")
            self._group.addButton(rb, level)
            self._buttons[level] = rb
            layout.addWidget(rb)

        layout.addStretch(1)

        self.set_level(current_level)

    def selected_level(self):
        """Return the telemetry level currently selected in the widget."""
        level = self._group.checkedId()
        return TELEMETRY_OFF if level == -1 else level

    def set_level(self, level):
        """Programmatically select a telemetry level."""
        if level not in self._buttons:
            level = TELEMETRY_OFF
        self._buttons[level].setChecked(True)


class TelemetryConsentDialog(QDialog):
    """Opt-in consent dialog shown only on first launch.

    Once a telemetry level has been recorded in the user configuration this
    dialog is no longer shown; users can change the setting later from the
    Advanced Options "Telemetry" tab.
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

        Header=QLabel("ONE TIME REQUEST- promise! :)")
        title_font = Header.font()
        title_font.setPointSize(title_font.pointSize() + 2)
        title_font.setBold(True)
        Header.setFont(title_font)

        layout.addWidget(Header)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        self._settings = TelemetrySettingsWidget(self, current_level=current_level)
        layout.addWidget(self._settings)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self._continue_btn = QPushButton("Continue")
        self._continue_btn.setDefault(True)
        button_row.addWidget(self._continue_btn)
        layout.addLayout(button_row)

        self._continue_btn.clicked.connect(self._on_accept)

    def _on_accept(self):
        self._level = self._settings.selected_level()
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
