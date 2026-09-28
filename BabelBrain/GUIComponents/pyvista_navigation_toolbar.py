from pathlib import Path

from matplotlib import get_data_path
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QToolBar,
)


class PyVistaNavigationToolbar(QToolBar):
    """Matplotlib-style navigation toolbar for a PyVista plotter."""

    def __init__(self, plotter, home_callback=None, parent=None):
        super().__init__(parent)

        self.plotter = plotter
        self.home_callback = home_callback
        self._history = []
        self._history_index = -1
        self._restoring_history = False

        self.home_action = QAction(
            self._matplotlib_icon("home.png"), "Home", self
        )
        self.home_action.setToolTip("Reset original view")
        self.home_action.triggered.connect(self.home)
        self.addAction(self.home_action)

        self.back_action = QAction(
            self._matplotlib_icon("back.png"), "Back", self
        )
        self.back_action.setToolTip("Back to previous view")
        self.back_action.triggered.connect(self.back)
        self.addAction(self.back_action)

        self.forward_action = QAction(
            self._matplotlib_icon("forward.png"), "Forward", self
        )
        self.forward_action.setToolTip("Forward to next view")
        self.forward_action.triggered.connect(self.forward)
        self.addAction(self.forward_action)

        self.addSeparator()

        self.pan_action = QAction(
            self._matplotlib_icon("move.png"), "Pan", self
        )
        self.pan_action.setCheckable(True)
        self.pan_action.setToolTip("Pan with left mouse button")
        self.pan_action.triggered.connect(self.toggle_pan)
        self.addAction(self.pan_action)

        self.zoom_action = QAction(
            self._matplotlib_icon("zoom_to_rect.png"), "Zoom", self
        )
        self.zoom_action.setCheckable(True)
        self.zoom_action.setToolTip("Zoom to rectangle")
        self.zoom_action.triggered.connect(self.toggle_zoom)
        self.addAction(self.zoom_action)

        self.addSeparator()

        self.save_action = QAction(
            self._matplotlib_icon("filesave.png"), "Save", self
        )
        self.save_action.setToolTip("Save figure")
        self.save_action.triggered.connect(self.save)
        self.addAction(self.save_action)

        self.setMovable(False)
        self.setFloatable(False)

        self._set_default_interaction()
        self.push_current_view()
        self._history_observer = self.plotter.iren.add_observer(
            "EndInteractionEvent",
            self._interaction_finished,
        )
        self._update_buttons()

    def _matplotlib_icon(self, name):
        """Load an icon the same way Matplotlib's NavigationToolbar2QT does."""
        icon_dir = Path(get_data_path()) / "images"
        path_regular = icon_dir / name
        path_large = path_regular.with_name(
            path_regular.name.replace(".png", "_large.png")
        )
        filename = str(path_large if path_large.exists() else path_regular)

        pixmap = QPixmap(filename)
        pixmap.setDevicePixelRatio(self.devicePixelRatioF() or 1)

        if self.palette().color(self.backgroundRole()).value() < 128:
            icon_color = self.palette().color(self.foregroundRole())
            mask = pixmap.createMaskFromColor(
                QColor("black"),
                Qt.MaskMode.MaskOutColor,
            )
            pixmap.fill(icon_color)
            pixmap.setMask(mask)

        return QIcon(pixmap)

    def _set_default_interaction(self):
        self.plotter.enable_custom_trackball_style(
            left="rotate",
            middle="pan",
            right="dolly",
        )
        self.plotter.interactor.setCursor(
            Qt.CursorShape.ArrowCursor
        )

    def _interaction_finished(self, *_):
        if not self._restoring_history:
            self.push_current_view()

    def _get_camera_position(self):
        return tuple(
            tuple(float(value) for value in vector)
            for vector in self.plotter.camera_position
        )

    @staticmethod
    def _camera_positions_equal(a, b, tolerance=1e-10):
        if a is None or b is None:
            return False

        return all(
            abs(value_a - value_b) <= tolerance
            for vector_a, vector_b in zip(a, b)
            for value_a, value_b in zip(vector_a, vector_b)
        )

    def push_current_view(self):
        camera_position = self._get_camera_position()

        if (
            self._history_index >= 0
            and self._camera_positions_equal(
                camera_position,
                self._history[self._history_index],
            )
        ):
            return

        if self._history_index < len(self._history) - 1:
            self._history = self._history[:self._history_index + 1]

        self._history.append(camera_position)
        self._history_index = len(self._history) - 1
        self._update_buttons()

    def _restore_view(self, camera_position):
        self._restoring_history = True

        try:
            self.plotter.camera_position = camera_position
            self.plotter.render()
        finally:
            self._restoring_history = False

    def home(self):
        if self.home_callback is not None:
            self.home_callback()
        else:
            self.plotter.reset_camera()
            self.plotter.render()
            self.push_current_view()

    def back(self):
        if self._history_index <= 0:
            return

        self._history_index -= 1
        self._restore_view(self._history[self._history_index])
        self._update_buttons()

    def forward(self):
        if self._history_index >= len(self._history) - 1:
            return

        self._history_index += 1
        self._restore_view(self._history[self._history_index])
        self._update_buttons()

    def toggle_pan(self, checked):
        if checked:
            self.zoom_action.blockSignals(True)
            self.zoom_action.setChecked(False)
            self.zoom_action.blockSignals(False)

            self.plotter.enable_custom_trackball_style(
                left="pan",
                middle="pan",
                right="dolly",
            )
            self.plotter.interactor.setCursor(
                Qt.CursorShape.SizeAllCursor
            )
        else:
            self._set_default_interaction()

    def toggle_zoom(self, checked):
        if checked:
            self.pan_action.blockSignals(True)
            self.pan_action.setChecked(False)
            self.pan_action.blockSignals(False)

            self.plotter.enable_zoom_style()
            self.plotter.interactor.setCursor(
                Qt.CursorShape.CrossCursor
            )
        else:
            self._set_default_interaction()

    def save(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Transducer Geometry Image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)",
        )

        if not file_path:
            return

        self.plotter.screenshot(file_path)

    def _update_buttons(self):
        self.back_action.setEnabled(self._history_index > 0)
        self.forward_action.setEnabled(
            0 <= self._history_index < len(self._history) - 1
        )