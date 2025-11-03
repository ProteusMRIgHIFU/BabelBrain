
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton,
    QSizePolicy, QApplication, QTabWidget, QDialog, QDialogButtonBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QPoint
import fitz  # PyMuPDF
import os


# ---------------------------------------------------------------------
# Zoomable image widget
# ---------------------------------------------------------------------
class ZoomableImageLabel(QLabel):
    def __init__(self, pixmap: QPixmap,defaulRatio):
        super().__init__()
        self._pixmap_original = pixmap
        self._scale = 1.0
        self._panning = False
        self._last_mouse_pos = QPoint()
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._defaulRatio=defaulRatio
        self.set_scale(defaulRatio)

    def _apply_scale(self):
        if not self._pixmap_original.isNull():
            scaled_pixmap = self._pixmap_original.scaled(
                self._pixmap_original.size() * self._scale,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)

    def reset_scale(self):
        self.set_scale(self._defaulRatio)

    def set_scale(self, scale):
        self._scale = max(0.05, min(3.0, scale))
        self._apply_scale()

    def fit_to_width(self, width):
        if not self._pixmap_original.isNull():
            scale = width / self._pixmap_original.width()
            self._scale = scale
            self._defaulRatio=scale
            self._apply_scale()

    # def wheelEvent(self, event):
    #     delta = event.angleDelta().y()
    #     zoom_factor = 1.05 if delta > 0 else 0.95
    #     self.set_scale(self._scale * zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._panning = True
            self._last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._panning:
            diff = event.pos() - self._last_mouse_pos
            self.parent().horizontalScrollBar().setValue(
                self.parent().horizontalScrollBar().value() - diff.x()
            )
            self.parent().verticalScrollBar().setValue(
                self.parent().verticalScrollBar().value() - diff.y()
            )
            self._last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)


# ---------------------------------------------------------------------
# Image panel with toolbar
# ---------------------------------------------------------------------
class ZoomableImagePanel(QWidget):
    def __init__(self, pixmap: QPixmap,defaulRatio):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar_layout = QHBoxLayout()
        btn_zoom_in = QPushButton("+")
        btn_zoom_out = QPushButton("−")
        btn_reset = QPushButton("⟳")
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(btn_zoom_in)
        toolbar_layout.addWidget(btn_zoom_out)
        toolbar_layout.addWidget(btn_reset)
        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)

        # Scrollable image
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = ZoomableImageLabel(pixmap,defaulRatio)
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

        # Connect toolbar buttons
        btn_zoom_in.clicked.connect(lambda: self.image_label.set_scale(self.image_label._scale * 1.25))
        btn_zoom_out.clicked.connect(lambda: self.image_label.set_scale(self.image_label._scale * 0.8))
        btn_reset.clicked.connect(lambda: self.image_label.reset_scale())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.doResize()

    def doResize(self):
        viewport_width = self.scroll_area.viewport().width()
        self.image_label.fit_to_width(viewport_width)




# ---------------------------------------------------------------------
# Tab widget for multiple images / PDFs
# ---------------------------------------------------------------------
class PlotViewerTabs(QWidget):
    """Tab-based viewer supporting PNG and PDF (via PyMuPDF)."""
    def __init__(self, file_paths, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        for path in file_paths:
            for page_index, pixmap in self._load_file_as_pixmaps(path):
                panel = ZoomableImagePanel(pixmap,0.2)
                tab_name = f"{os.path.basename(path)}".split('.')[0].split('Plots-')[1]
                if page_index > 0:
                    tab_name += f" (p{page_index+1})"
                self.tab_widget.addTab(panel, tab_name)
                

    def _load_file_as_pixmaps(self, path):
        """Yield (page_index, QPixmap) pairs for image or PDF."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            doc = fitz.open(path)
            for i, page in enumerate(doc):
                zoom = 4.0  
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
                yield i, QPixmap.fromImage(qimg)
        else:
            yield 0, QPixmap(path)


# ---------------------------------------------------------------------
# Dialog wrapper
# ---------------------------------------------------------------------
class PlotViewerCalibration(QDialog):
    """Dialog wrapping the tabbed plot viewer with Accept/Cancel."""
    def __init__(self, file_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review Calibration Results")
        self.resize(1000, 800)

        layout = QVBoxLayout(self)

        # Viewer widget
        self.viewer = PlotViewerTabs(file_paths)
        layout.addWidget(self.viewer)

        # Button box (Accept/Cancel)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_button = buttons.button(QDialogButtonBox.Ok)
        ok_button.setText("Confirm and accept calibration")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def showEvent(self, event):
        super().showEvent(event)
        # After the dialog is visible, adjust all current image panels
        for i in range(self.viewer.tab_widget.count()):
            self.viewer.tab_widget.setCurrentIndex(i)
            panel = self.viewer.tab_widget.widget(i)
            panel.resizeEvent(None)
            # panel.image_label.reset_scale()
        self.viewer.tab_widget.setCurrentIndex(0)


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    image_files = ["/Users/spichardo/Library/CloudStorage/OneDrive-UniversityofCalgary/GDrive/BABELBRAIN Tx information/Radboud/Imasonic_15287_1002_cal/Plots-AcProfiles.pdf",
                   "/Users/spichardo/Library/CloudStorage/OneDrive-UniversityofCalgary/GDrive/BABELBRAIN Tx information/Radboud/Imasonic_15287_1002_cal/Plots-Acplanes.pdf",
                   '/Users/spichardo/Library/CloudStorage/OneDrive-UniversityofCalgary/GDrive/BABELBRAIN Tx information/Radboud/Imasonic_15287_1002_cal/Plots-weight.pdf']

    dialog = PlotViewerCalibration(image_files)

    if dialog.exec() == QDialog.Accepted:
        print("User accepted the results.")
    else:
        print("User canceled.")

    sys.exit(0)
