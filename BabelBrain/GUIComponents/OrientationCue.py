"""Visual cue for the transducer steering-axes convention.

Phased-array vendors do not agree on the sign of the steering axes, so
`_BabelBaseTx.FlipSteeringX` / `.FlipSteeringY` tell BabelBrain whether the
device's +X / +Y steering directions run opposite to the on-screen convention
(+X to the right, +Y to the top).  Those flips are easy to miss, hence this
small axes glyph shown in the top bar of the main window.

The glyph is drawn with QPainter (palette-aware, crisp on HiDPI), but a PNG
drop-in is supported as well: place files named

    OrientationXRightYTop.png     OrientationXRightYBottom.png
    OrientationXLeftYTop.png      OrientationXLeftYBottom.png

in `GUIComponents/OrientationIcons/` and they take precedence over the drawn
version (useful if a vendor-specific artwork is preferred later on).
"""

import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QSize, QPointF
from PySide6.QtGui import QPainter, QPen, QPolygonF, QPixmap, QFont, QColor
from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout, QSizePolicy

from GUIComponents.AppStyle import palette_is_dark

# Same cyan accent used by the main form / nifti viewer.
ACCENT = "#00c8ff"
# Second accent so the two axes are told apart at a glance.
ACCENT_Y = "#ff9f43"


def _icon_dir():
    """Folder holding the optional PNG overrides (dev tree and frozen app)."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / 'GUIComponents' / 'OrientationIcons'
    return Path(__file__).parent / 'OrientationIcons'


def orientation_name(bFlipX, bFlipY):
    """Canonical name of an orientation, e.g. 'XRightYTop'."""
    return 'X%sY%s' % ('Left' if bFlipX else 'Right',
                       'Bottom' if bFlipY else 'Top')


def orientation_text(bFlipX, bFlipY):
    """Compact description of the convention in effect, arrows only."""
    return '+X %s, +Y %s' % ('←' if bFlipX else '→',
                             '↓' if bFlipY else '↑')


class OrientationGlyph(QWidget):
    """Small axes cross showing where +X and +Y point for the active device."""

    _SIZE = QSize(58, 38)

    def __init__(self, parent=None, bFlipX=False, bFlipY=False):
        super().__init__(parent)
        self._bFlipX = bFlipX
        self._bFlipY = bFlipY
        self.setFixedSize(self._SIZE)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    # ── API ───────────────────────────────────────────────────────────────
    def SetOrientation(self, bFlipX, bFlipY):
        if (bool(bFlipX), bool(bFlipY)) == (self._bFlipX, self._bFlipY):
            return
        self._bFlipX = bool(bFlipX)
        self._bFlipY = bool(bFlipY)
        self.update()

    # ── Painting ──────────────────────────────────────────────────────────
    def _pixmap(self):
        """PNG override for the current orientation, or None."""
        fname = _icon_dir() / ('Orientation%s.png' %
                              orientation_name(self._bFlipX, self._bFlipY))
        if not os.path.isfile(fname):
            return None
        pix = QPixmap(str(fname))
        return None if pix.isNull() else pix

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)

        pix = self._pixmap()
        if pix is not None:
            scaled = pix.scaled(self.size(), Qt.KeepAspectRatio,
                                Qt.SmoothTransformation)
            p.drawPixmap((self.width() - scaled.width()) // 2,
                         (self.height() - scaled.height()) // 2, scaled)
            return

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        # Both axes get the same arm length so neither reads as "longer"; the
        # limit is whichever side has less room once the +X / +Y captions at
        # the arrow tips are accounted for.
        arm = min(w / 2.0 - 15.0, h / 2.0 - 9.0)
        armx = army = arm

        # Axis directions in widget coordinates (y grows downwards on screen,
        # so "+Y to the top" is a negative dy).
        dx = -1.0 if self._bFlipX else 1.0
        dy = 1.0 if self._bFlipY else -1.0

        cX = QColor(ACCENT)
        cY = QColor(ACCENT_Y)
        faint = QColor(cX)
        faint.setAlpha(70)

        # Faint full cross so the origin reads as an origin, not an elbow.
        p.setPen(QPen(faint, 1, Qt.DotLine))
        p.drawLine(QPointF(cx - armx, cy), QPointF(cx + armx, cy))
        p.drawLine(QPointF(cx, cy - army), QPointF(cx, cy + army))

        self._draw_axis(p, cX, QPointF(cx, cy),
                        QPointF(cx + dx * armx, cy))
        self._draw_axis(p, cY, QPointF(cx, cy),
                        QPointF(cx, cy + dy * army))

        # Captions, tucked just past each arrow tip.
        f = QFont(self.font())
        f.setPointSizeF(max(7.0, f.pointSizeF() - 2.0))
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()

        p.setPen(QPen(cX))
        tx = cx + dx * armx + (2.0 if dx > 0 else -2.0 - fm.horizontalAdvance('+X'))
        p.drawText(QPointF(tx, cy + fm.capHeight() / 2.0), '+X')

        p.setPen(QPen(cY))
        ty = cy + dy * army + (fm.capHeight() + 2.0 if dy > 0 else -3.0)
        p.drawText(QPointF(cx - fm.horizontalAdvance('+Y') / 2.0, ty), '+Y')

        # Origin dot, drawn last so it sits on top of both axes.
        p.setPen(Qt.NoPen)
        p.setBrush(QColor('#dddddd') if palette_is_dark(self) else QColor('#333333'))
        p.drawEllipse(QPointF(cx, cy), 1.8, 1.8)

    @staticmethod
    def _draw_axis(p, color, origin, tip):
        """Solid shaft from `origin` to `tip` capped with a filled arrow head."""
        p.setPen(QPen(color, 1.6, Qt.SolidLine, Qt.RoundCap))
        p.setBrush(Qt.NoBrush)
        p.drawLine(origin, tip)

        vx, vy = tip.x() - origin.x(), tip.y() - origin.y()
        norm = (vx * vx + vy * vy) ** 0.5
        if norm == 0:
            return
        vx, vy = vx / norm, vy / norm
        # Perpendicular to the shaft, for the two base corners of the head.
        px, py = -vy, vx
        head, half = 6.0, 3.2
        base = QPointF(tip.x() - vx * head, tip.y() - vy * head)
        tri = QPolygonF([tip,
                         QPointF(base.x() + px * half, base.y() + py * half),
                         QPointF(base.x() - px * half, base.y() - py * half)])
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.drawPolygon(tri)


class OrientationCue(QWidget):
    """`OrientationGlyph` plus its caption — sits in the main form top bar."""

    def __init__(self, parent=None, bFlipX=False, bFlipY=False):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._title = QLabel('Tx. orientation')
        f = self._title.font()
        f.setBold(True)
        self._title.setFont(f)
        lay.addWidget(self._title)

        self.Glyph = OrientationGlyph(self, bFlipX=bFlipX, bFlipY=bFlipY)
        lay.addWidget(self.Glyph)

        self.SetOrientation(bFlipX, bFlipY)

    def SetOrientation(self, bFlipX, bFlipY, DeviceName=None):
        bFlipX, bFlipY = bool(bFlipX), bool(bFlipY)
        self.Glyph.SetOrientation(bFlipX, bFlipY)
        tip = ('Steering axes of %s as seen in the Step 2/3 views:\n%s' %
               (DeviceName if DeviceName else 'the selected device',
                orientation_text(bFlipX, bFlipY)))
        if bFlipX or bFlipY:
            tip += ('\nThis device flips %s relative to the default '
                    'convention (+X right, +Y top).' %
                    (' and '.join([n for n, f in (('X', bFlipX), ('Y', bFlipY)) if f])))
        self.setToolTip(tip)
