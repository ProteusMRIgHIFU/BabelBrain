"""
nifti_viewer.py
===============
PySide6 + VTK NIfTI viewer — multi-volume with overlay transparency.

Features
--------
  • Load any number of NIfTI files.  The first file sets the reference
    geometry (affine orientation, camera planes, slice count).
  • Each subsequent file is an overlay rendered on top, with its own
    opacity slider (0–100 %) and colour-map selector (Grey / Hot / Cool /
    Green / Red).
  • Individual eye-icon toggle to show/hide each overlay.
  • Both AFFINE mode (native voxel axes of file 1) and MEDICAL mode
    (RAS world axes: Axial / Coronal / Sagittal) are supported.
  • Neurogical flip (L↔R) available in medical mode.
  • Crosshair lines synchronised across all three views on scroll.

Architecture
------------
  VolumeRecord  – bundles (vtk_idx, vtk_xform, vtk_property) for one file.
  SliceViewport – owns a vtkRenderer with one vtkImageSlice per loaded volume,
                  all sharing the same camera (driven by volume 0 geometry).
  NiftiViewer   – manages the list of VolumeRecords and coordinates viewports.
  LayerPanel    – sidebar widget; one LayerRow per volume with opacity/cmap/eye.

Camera / geometry
-----------------
  Always derived from volume 0's affine (affine mode) or from the RAS
  bounding box of volume 0 (medical mode).  Overlay volumes are rendered
  with their own affine UserTransform so VTK cuts them at the same world-
  space plane as the base volume — even if they have different voxel grids.

Requirements
------------
    pip install PySide6 vtk nibabel numpy

Usage
-----
    python nifti_viewer.py [file1.nii.gz [file2.nii.gz ...]]
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette, QDoubleValidator
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider,
    QSizePolicy, QFileDialog, QPushButton, QFrame, QSplitter,
    QButtonGroup, QRadioButton, QCheckBox, QGroupBox,
    QScrollArea, QComboBox, QToolButton, QLineEdit,
)

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util import numpy_support

try:
    import nibabel as nib
except ImportError:
    nib = None


# ── Palette ────────────────────────────────────────────────────────────────
BG_DARK   = "#1a1a1e"
BG_PANEL  = "#222228"
BG_VP     = "#0d0d0f"
BG_LAYER  = "#28282f"
ACCENT    = "#00c8ff"
COLORS    = ["#ff6b6b", "#6bffb8", "#ffda6b"]
TEXT      = "#d4d4d8"
TEXT_DIM  = "#71717a"
TEXT_LBL  = "#ffda6b"

# Per-volume accent colours for the layer panel rows
VOL_COLORS = ["#ffffff", "#ff9f43", "#48dbfb", "#ff6b81",
              "#a29bfe", "#00d2d3", "#ffd32a", "#0abde3"]

AFFINE_NAMES  = ["Slice j ", "Slice i", "Slice k"]
MEDICAL_NAMES = ["Axial", "Coronal", "Sagittal"]

CMAPS = {
    "TissueLabel": "tissue_label",
    "Grey":  None,          # black→white
    "Hot":   "hot",
    "Cool":  "cool",
    "Green": "green",
    "Red":   "red",
    "Jet":   "jet",         # blue→cyan→green→yellow→red
}

# ---------------------------------------------------------------------------
# Tissue colour table  (matches existing BabelBrain colour definitions)
# mask integer value → (R, G, B, A) in 0-255
# ---------------------------------------------------------------------------
_TISSUE_RGBA: dict[int, tuple[int, int, int, int]] = {
    0: (  0,   0,   0,   0),   # background  – fully transparent
    1: (  0,  77, 255, 255),   # scalp
    2: (  0, 128, 255, 255),   # cortical bone
    3: ( 21, 255, 225, 255),   # trabecular bone
    4: (124, 255, 121, 255),   # brain (non-segmented)
    5: (255, 255,   0, 220),   # focal-point voxel  (bright yellow)
    6: (255, 148,   0, 255),   # white matter
    7: (255,  29,   0, 255),   # grey matter
    8: (127,   0,   0, 255),   # CSF
}

_N_LABELS = max(_TISSUE_RGBA) + 1

def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _np4(m: np.ndarray) -> vtk.vtkMatrix4x4:
    mat = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            mat.SetElement(r, c, float(m[r, c]))
    return mat


def _build_tissue_lut() -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(_N_LABELS)
    lut.SetTableRange(0, _N_LABELS - 1)
    for v in range(_N_LABELS):
        r, g, b, a = _TISSUE_RGBA.get(v, (0, 0, 0, 0))
        lut.SetTableValue(v, r / 255.0, g / 255.0, b / 255.0, a / 255.0)
    lut.Build()
    return lut

def _make_lut(
    name: str | None,
    lo: float,
    hi: float,
    cutoff: float | None = None,
) -> vtk.vtkLookupTable:
    """
    Build a vtkLookupTable for the given colour map.
    If cutoff is not None, all scalar values < cutoff are mapped to alpha=0
    (fully transparent), effectively masking them out.  Values >= cutoff use
    the normal colour map with alpha=1.
    """

    if name == 'tissue_label':
        return _build_tissue_lut()

    N = 1024   # enough entries for smooth gradients and a sharp cutoff edge
    lut = vtk.vtkLookupTable()
    lut.SetRange(lo, hi)
    lut.SetNumberOfTableValues(N)

    if name == "jet":
        # Jet is piecewise-linear RGB — cannot be expressed via VTK's HSV ranges.
        # Interpolate the canonical Matplotlib Jet control points directly.
        jet_cps = [        # (t,   R,    G,    B)
            (0.000, 0.000, 0.000, 0.500),
            (0.125, 0.000, 0.000, 1.000),
            (0.375, 0.000, 1.000, 1.000),
            (0.625, 1.000, 1.000, 0.000),
            (0.875, 1.000, 0.000, 0.000),
            (1.000, 0.500, 0.000, 0.000),
        ]
        # Build() allocates the internal table; we overwrite every entry below.
        lut.Build()
        for i in range(N):
            t = i / (N - 1)
            for k in range(len(jet_cps) - 1):
                t0, r0, g0, b0 = jet_cps[k]
                t1, r1, g1, b1 = jet_cps[k + 1]
                if t0 <= t <= t1:
                    f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                    lut.SetTableValue(i,
                                      r0 + f * (r1 - r0),
                                      g0 + f * (g1 - g0),
                                      b0 + f * (b1 - b0),
                                      1.0)
                    break
    else:
        if name is None or name == "grey":
            lut.SetSaturationRange(0, 0)
            lut.SetHueRange(0, 0)
            lut.SetValueRange(0, 1)
        elif name == "hot":
            lut.SetHueRange(0.0, 0.1667)
            lut.SetSaturationRange(1, 0)
            lut.SetValueRange(0.5, 1)
        elif name == "cool":
            lut.SetHueRange(0.5, 0.833)
            lut.SetSaturationRange(1, 1)
            lut.SetValueRange(1, 1)
        elif name == "green":
            lut.SetHueRange(0.333, 0.333)
            lut.SetSaturationRange(0, 1)
            lut.SetValueRange(0, 1)
        elif name == "red":
            lut.SetHueRange(0.0, 0.0)
            lut.SetSaturationRange(0, 1)
            lut.SetValueRange(0, 1)
        else:
            lut.SetSaturationRange(0, 0)
            lut.SetHueRange(0, 0)
            lut.SetValueRange(0, 1)
        lut.Build()

    # Apply cutoff: zero the alpha of every entry whose scalar < cutoff.
    if cutoff is not None and hi > lo:
        span = hi - lo
        for i in range(N):
            scalar = lo + (i / (N - 1)) * span
            if scalar < cutoff:
                r, g, b, _ = lut.GetTableValue(i)
                lut.SetTableValue(i, r, g, b, 0.0)

    return lut



# ── PlaneGeometry ──────────────────────────────────────────────────────────
@dataclass
class PlaneGeometry:
    normal:  np.ndarray
    right:   np.ndarray
    up:      np.ndarray
    centre:  np.ndarray
    step:    float
    n:       int
    ps:      float        # max(plane_w, plane_h) / 2  — default parallel scale
    plane_w: float = 0.   # physical width of the image plane (right direction)
    plane_h: float = 0.   # physical height of the image plane (up direction)
    flip_lr: bool  = False


# ── VolumeRecord ───────────────────────────────────────────────────────────
@dataclass
class VolumeRecord:
    """Everything VTK needs to render one loaded NIfTI file."""
    name:      str
    vtk_idx:   vtk.vtkImageData       # data in index/physical space
    vtk_xform: vtk.vtkTransform       # VTK physical → world (RAS)
    lo:        float                  # data min (fixed, from file)
    hi:        float                  # data max (fixed, from file)
    wl_window: float = 0.             # current display window width (hi-lo at load)
    wl_level:  float = 0.             # current display window centre
    opacity:   float = 1.0
    visible:   bool  = True
    cmap:      str   = "Grey"         # key into CMAPS
    cutoff:    float | None = None    # values < cutoff rendered transparent
    id:        str   = ''             # ID, used to searc/replace in case of need

    def __post_init__(self):
        if self.wl_window == 0.:
            self.wl_window = self.hi - self.lo or 1.
        if self.wl_level == 0.:
            self.wl_level = (self.hi + self.lo) / 2.


# ── Affine / geometry helpers ──────────────────────────────────────────────

def _decompose(affine: np.ndarray):
    M  = affine[:3, :3]
    sp = np.linalg.norm(M, axis=0)
    return sp, M / sp[np.newaxis, :]


def _ras_bbox(affine: np.ndarray, shape: tuple):
    corners = np.array([
        (affine @ np.array([i, j, k, 1.]))[:3]
        for i in (0, shape[0]-1)
        for j in (0, shape[1]-1)
        for k in (0, shape[2]-1)
    ])
    return corners.min(axis=0), corners.max(axis=0)


def _make_vtk_transform(affine: np.ndarray, spacing: np.ndarray) -> vtk.vtkTransform:
    S_inv = np.diag([1./spacing[0], 1./spacing[1], 1./spacing[2], 1.])
    xf = vtk.vtkTransform()
    xf.SetMatrix(_np4(affine @ S_inv))
    return xf


def numpy_to_vtk_index(data: np.ndarray, spacing: np.ndarray) -> vtk.vtkImageData:
    img = vtk.vtkImageData()
    img.SetDimensions(int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
    img.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
    img.SetOrigin(0., 0., 0.)
    flat = data.ravel(order="F")
    arr  = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    img.GetPointData().SetScalars(arr)
    return img


def load_volume_record(ni_path: object, inname='',use_percentile=False,id='') -> tuple[VolumeRecord, tuple, tuple, str]:
    """Load a NIfTI file and return a VolumeRecord + metadata."""
    if nib is None:
        raise ImportError("nibabel is required: pip install nibabel")
    if type(ni_path) is str:
        img  = nib.load(ni_path)
        name      = os.path.basename(ni_path)
    else:
        assert(type(ni_path)  in [nib.nifti1.Nifti1Image,nib.nifti2.Nifti2Image] )
        img = ni_path
        name      = inname
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D, got shape {data.shape}")

    affine = img.affine.astype(np.float64)
    sp, _  = _decompose(affine)
    if use_percentile:
        perc=np.percentile(data.flatten(),[0.5,99.5])
        lo=perc[0]
        hi=perc[1]
    else: 
        lo, hi = float(data.min()), float(data.max())

    rec = VolumeRecord(
        name      = name,
        vtk_idx   = numpy_to_vtk_index(data, sp),
        vtk_xform = _make_vtk_transform(affine, sp),
        lo        = lo,
        hi        = hi,
        id        = id
    )
    zooms = img.header.get_zooms()[:3]
    code  = "".join(nib.aff2axcodes(affine))
    return rec, data.shape, tuple(float(z) for z in zooms), code, affine


# ── Plane geometry factories ───────────────────────────────────────────────

def affine_plane_geoms(affine: np.ndarray, shape: tuple) -> list[PlaneGeometry]:
    sp, R = _decompose(affine)
    di, dj, dk = R[:,0], R[:,1], R[:,2]
    si, sj, sk = sp
    ni, nj, nk = shape
    mid = np.array([(ni-1)/2., (nj-1)/2., (nk-1)/2., 1.])
    ctr = (affine @ mid)[:3]
    def ps(a, b): return max(a, b) / 2.
    return [
        PlaneGeometry(normal=dj, right=di, up=dk,  centre=ctr.copy(), step=sj, n=nj,
                      ps=ps(ni*si, nk*sk), plane_w=float(ni*si), plane_h=float(nk*sk),flip_lr=True),
        PlaneGeometry(normal=di, right=dj, up=dk,  centre=ctr.copy(), step=si, n=ni,
                      ps=ps(nj*sj, nk*sk), plane_w=float(nj*sj), plane_h=float(nk*sk)),
        PlaneGeometry(normal=dk, right=di, up=-dj,  centre=ctr.copy(), step=sk, n=nk,
                      ps=ps(ni*si, nj*sj), plane_w=float(ni*si), plane_h=float(nj*sj),flip_lr=True),
    ]


def medical_plane_geoms(ras_min, ras_max, iso, neurogical) -> list[PlaneGeometry]:
    """
    Three PlaneGeometry objects for the three RAS-axis planes.

    Neurogical L/R flip is achieved by setting flip_lr=True, which makes
    _init_camera place the camera on the opposite side of the focal plane
    (position = focal - normal*DIST instead of + normal*DIST).  VTK then
    computes screen-right = up × viewPlaneNormal where viewPlaneNormal = -normal,
    giving -(up × normal) — effectively mirroring left and right while keeping
    up unchanged.
    """
    ctr  = (ras_min + ras_max) / 2.
    flip = neurogical

    ax_x  = np.array([1.,  0., 0.])   # right (always +X in world; flip is in camera)
    ax_y  = np.array([0.,  1., 0.])
    ax_z  = np.array([0.,  0., 1.])
    sa_r  = np.array([0., -1., 0.])   # sagittal screen-right = -Y (ant on screen-right)

    def ns(axis): return max(2, int(np.ceil((ras_max[axis]-ras_min[axis])/iso))+1)
    def ps(a, b): return max(a, b) / 2.
    ex = ras_max[0]-ras_min[0]; ey = ras_max[1]-ras_min[1]; ez = ras_max[2]-ras_min[2]

    return [
        PlaneGeometry(normal=ax_z,  right=ax_x, up=ax_y, centre=ctr.copy(), step=iso, n=ns(2),
                      ps=ps(ex,ey), plane_w=float(ex), plane_h=float(ey), flip_lr=not flip),
        PlaneGeometry(normal=ax_y,  right=ax_x, up=ax_z, centre=ctr.copy(), step=iso, n=ns(1),
                      ps=ps(ex,ez), plane_w=float(ex), plane_h=float(ez), flip_lr=flip),
        PlaneGeometry(normal=ax_x,  right=sa_r, up=ax_z, centre=ctr.copy(), step=iso, n=ns(0),
                      ps=ps(ey,ez), plane_w=float(ey), plane_h=float(ez), flip_lr=True),
    ]


# ── Line actor helpers ─────────────────────────────────────────────────────

def _make_line_actor(color_hex):
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(0, 0, 0); pts.InsertNextPoint(1, 0, 0)
    poly = vtk.vtkPolyData(); poly.SetPoints(pts)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(2); cells.InsertCellPoint(0); cells.InsertCellPoint(1)
    poly.SetLines(cells)
    m = vtk.vtkPolyDataMapper(); m.SetInputData(poly)
    a = vtk.vtkActor(); a.SetMapper(m)
    a.GetProperty().SetColor(*_hex_rgb(color_hex))
    a.GetProperty().SetLineWidth(1.2)
    a.VisibilityOff()
    return a


def _set_line(actor, p1, p2):
    poly = actor.GetMapper().GetInput()
    pts  = poly.GetPoints()
    pts.SetPoint(0, *p1.tolist()); pts.SetPoint(1, *p2.tolist())
    pts.Modified(); poly.Modified()


def _make_wl_style(on_wl_drag, on_wl_end, on_scroll) -> vtk.vtkInteractorStyle:
    """
    Custom interactor style for medical image viewing.

    Uses vtkInteractorStyleUser as base (no built-in image interactions)
    so we have complete control and VTK never touches any actor property.

    Interactions
    ------------
    Right button drag         : window/level — on_wl_drag(dw_norm, dl_norm)
    Shift + Right button drag : zoom  (vertical drag — up = zoom in)
    Left button drag          : pan
    Middle button drag        : zoom
    Mouse wheel               : scroll slices — on_scroll(±1)
    Ctrl + mouse wheel        : zoom ±10 %
    """
    style = vtk.vtkInteractorStyleUser()

    _s = {"rmb": False, "lmb": False, "mmb": False, "lx": 0, "ly": 0}

    def _pos(obj):
        return obj.GetInteractor().GetEventPosition()

    def _pan(obj, dx, dy):
        iren  = obj.GetInteractor()
        rw    = iren.GetRenderWindow()
        ren   = rw.GetRenderers().GetFirstRenderer()
        cam   = ren.GetActiveCamera()
        fp    = np.array(cam.GetFocalPoint())
        pos   = np.array(cam.GetPosition())
        h     = rw.GetSize()[1] or 1
        wpp   = 2.0 * cam.GetParallelScale() / h
        up    = np.array(cam.GetViewUp())
        vpn   = np.array(cam.GetViewPlaneNormal())
        right = np.cross(up, vpn)
        right /= (np.linalg.norm(right) + 1e-9)
        up    /= (np.linalg.norm(up)    + 1e-9)
        delta  = (-dx * right - dy * up) * wpp
        cam.SetFocalPoint(*(fp  + delta).tolist())
        cam.SetPosition( *(pos + delta).tolist())
        rw.Render()

    def _zoom(obj, factor):
        iren = obj.GetInteractor()
        rw   = iren.GetRenderWindow()
        ren  = rw.GetRenderers().GetFirstRenderer()
        cam  = ren.GetActiveCamera()
        cam.SetParallelScale(max(0.1, cam.GetParallelScale() * factor))
        rw.Render()

    def _on_lmb_press(obj, event):
        _s["lmb"] = True; _s["lx"], _s["ly"] = _pos(obj)

    def _on_lmb_release(obj, event):
        _s["lmb"] = False

    def _on_rmb_press(obj, event):
        _s["rmb"] = True; _s["lx"], _s["ly"] = _pos(obj)

    def _on_rmb_release(obj, event):
        _s["rmb"] = False; on_wl_end()

    def _on_mmb_press(obj, event):
        _s["mmb"] = True; _s["lx"], _s["ly"] = _pos(obj)

    def _on_mmb_release(obj, event):
        _s["mmb"] = False

    def _on_move(obj, event):
        x, y = _pos(obj)
        dx = x - _s["lx"]; dy = y - _s["ly"]
        _s["lx"], _s["ly"] = x, y
        if _s["rmb"]:
            if obj.GetInteractor().GetShiftKey():
                # Shift + RMB drag → zoom  (vertical drag: up = zoom in)
                _zoom(obj, 1.0 - dy * 0.01)
            else:
                # Plain RMB drag → window / level
                w, h = obj.GetInteractor().GetRenderWindow().GetSize()
                on_wl_drag(dx / (w or 1), dy / (h or 1))
        elif _s["lmb"]:
            _pan(obj, dx, dy)
        elif _s["mmb"]:
            _zoom(obj, 1.0 - dy * 0.01)

    def _on_wheel_fwd(obj, event):
        iren = obj.GetInteractor()
        if iren.GetControlKey():
            _zoom(obj, 0.9)
        else:
            on_scroll(-1)   # scroll toward lower-index slice

    def _on_wheel_bwd(obj, event):
        iren = obj.GetInteractor()
        if iren.GetControlKey():
            _zoom(obj, 1.1)
        else:
            on_scroll(+1)

    style.AddObserver("LeftButtonPressEvent",     _on_lmb_press)
    style.AddObserver("LeftButtonReleaseEvent",   _on_lmb_release)
    style.AddObserver("RightButtonPressEvent",    _on_rmb_press)
    style.AddObserver("RightButtonReleaseEvent",  _on_rmb_release)
    style.AddObserver("MiddleButtonPressEvent",   _on_mmb_press)
    style.AddObserver("MiddleButtonReleaseEvent", _on_mmb_release)
    style.AddObserver("MouseMoveEvent",           _on_move)
    style.AddObserver("MouseWheelForwardEvent",   _on_wheel_fwd)
    style.AddObserver("MouseWheelBackwardEvent",  _on_wheel_bwd)
    return style


# ── SliceViewport ──────────────────────────────────────────────────────────

class SliceViewport(QFrame):
    """
    One orthogonal view.  Maintains a list of (vtkImageSlice, vtkImageProperty)
    pairs, one per VolumeRecord.  All share the same vtkRenderer and camera.

    The base volume (index 0) drives the camera geometry.
    Overlays are stacked on top in insertion order with their own properties.
    """

    slice_changed = Signal(int, int)

    def __init__(self, plane_idx: int, parent=None):
        super().__init__(parent)
        self.plane_idx = plane_idx
        self._pg: PlaneGeometry | None = None
        self._current_slice = 0
        # Per-volume actor/property pairs
        self._layers: list[tuple[vtk.vtkImageSlice, vtk.vtkImageProperty]] = []
        self._build_ui()
        self._build_pipeline()

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        c = COLORS[self.plane_idx]
        self.setStyleSheet(
            f"SliceViewport {{ border:2px solid {c}; border-radius:4px; background:{BG_VP}; }}")
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

        hdr = QWidget(); hdr.setFixedHeight(28)
        hdr.setStyleSheet(f"background:{BG_PANEL}; border-bottom:1px solid {c};")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(8,0,8,0)
        self._lbl_title = QLabel("—")
        self._lbl_title.setStyleSheet(f"color:{c}; font-size:11px; font-weight:bold; letter-spacing:2px;")
        self._lbl_slice = QLabel("—")
        self._lbl_slice.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        hl.addWidget(self._lbl_title); hl.addStretch(); hl.addWidget(self._lbl_slice)
        lay.addWidget(hdr)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self.vtk_widget)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ height:4px; background:#333340; }}
            QSlider::handle:horizontal {{
                background:{c}; width:14px; height:14px; margin:-5px 0; border-radius:7px; }}
            QSlider::sub-page:horizontal {{ background:{c}; }}""")
        self._slider.setFixedHeight(24)
        self._slider.valueChanged.connect(self._on_slider)
        lay.addWidget(self._slider)
        # self.setFixedWidth(300)

    def set_title(self, t):
        self._lbl_title.setText(t.upper())

    # ── VTK pipeline ──────────────────────────────────────────────────────
    def _build_pipeline(self):
        rw = self.vtk_widget.GetRenderWindow()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.05, 0.06)
        rw.AddRenderer(self.renderer)

        self._cross_h = _make_line_actor(ACCENT)
        self._cross_v = _make_line_actor(ACCENT)
        self.renderer.AddActor(self._cross_h)
        self.renderer.AddActor(self._cross_v)

        # Orientation label actors (L/R/A/P/S/I) — only shown in medical mode
        self._orient_actors: list[vtk.vtkTextActor] = []
        self._build_orient_actors()

        self.renderer.GetActiveCamera().ParallelProjectionOn()

        # Placeholder callbacks — replaced by set_wl_callbacks()
        self._wl_drag_cb = lambda dw, dl: None
        self._wl_end_cb  = lambda: None

        def _on_scroll(delta):
            """delta = ±1; moves slider by one step."""
            new_val = self._slider.value() + delta
            new_val = max(self._slider.minimum(), min(self._slider.maximum(), new_val))
            self._slider.setValue(new_val)   # triggers _on_slider → _move_to_slice

        style = _make_wl_style(
            on_wl_drag=lambda dw, dl: self._wl_drag_cb(dw, dl),
            on_wl_end=lambda: self._wl_end_cb(),
            on_scroll=_on_scroll,
        )
        rw.GetInteractor().SetInteractorStyle(style)

    def set_wl_callbacks(self, on_drag, on_end) -> None:
        """Called by NiftiViewer to route WL drag to the selected volume."""
        self._wl_drag_cb = on_drag
        self._wl_end_cb  = on_end

    # ── Orientation labels ─────────────────────────────────────────────────

    def _build_orient_actors(self) -> None:
        """
        Create four vtkTextActor objects (left, right, top, bottom) positioned
        in normalised viewport coordinates so they always hug the edges regardless
        of pan/zoom.  They start invisible; set_orientation_labels() shows them.
        """
        FONT_SIZE = 14
        BOLD      = True
        COLOR     = (1.0, 1.0, 0.6)   # warm yellow — readable on dark background

        # Positions in normalised viewport coords and horizontal justification.
        # Justification integers are stable across VTK versions:
        #   0 = left,  1 = centred,  2 = right  (vtkTextProperty constants)
        configs = [
            (0.02,  0.50, 0),   # screen-left  label  (left-justified)
            (0.98,  0.50, 2),   # screen-right label  (right-justified)
            (0.50,  0.97, 1),   # screen-top   label  (centred)
            (0.50,  0.03, 1),   # screen-bottom label (centred)
        ]

        self._orient_actors = []
        for nx, ny, halign in configs:
            actor = vtk.vtkTextActor()
            actor.SetInput("")
            actor.GetTextProperty().SetFontSize(FONT_SIZE)
            actor.GetTextProperty().SetBold(BOLD)
            actor.GetTextProperty().SetColor(*COLOR)
            actor.GetTextProperty().SetShadow(True)
            actor.GetTextProperty().SetJustification(halign)
            actor.GetTextProperty().SetVerticalJustificationToCentered()
            actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            actor.SetPosition(nx, ny)
            actor.VisibilityOff()
            self.renderer.AddActor2D(actor)
            self._orient_actors.append(actor)

    def set_orientation_labels(self, left: str, right: str,
                                top: str,  bottom: str) -> None:
        """Show anatomical labels at the four edges of the viewport."""
        texts = [left, right, top, bottom]
        for actor, text in zip(self._orient_actors, texts):
            actor.SetInput(text)
            actor.VisibilityOn()
        self.vtk_widget.GetRenderWindow().Render()

    def clear_orientation_labels(self) -> None:
        """Hide all orientation labels (used in affine mode)."""
        for actor in self._orient_actors:
            actor.VisibilityOff()
        self.vtk_widget.GetRenderWindow().Render()

    def reset_camera(self) -> None:
        """
        Restore the default camera (pan, zoom) and slider position.
        Does not touch image data or WL — those are reset by NiftiViewer.
        """
        if self._pg is None:
            return
        mid = self._pg.n // 2
        self._slider.blockSignals(True)
        self._slider.setValue(mid)
        self._slider.blockSignals(False)
        self._current_slice = mid
        self._lbl_slice.setText(f"{mid+1} / {self._pg.n}")
        self._init_camera(mid)
        self.vtk_widget.GetRenderWindow().Render()

    def _make_slice_actor(self) -> tuple[vtk.vtkImageSlice, vtk.vtkImageProperty]:
        mapper = vtk.vtkImageResliceMapper()
        mapper.SliceFacesCameraOn()
        mapper.SliceAtFocalPointOn()
        mapper.BorderOff()
        mapper.SetSlabThickness(0)

        prop = vtk.vtkImageProperty()
        prop.SetInterpolationTypeToLinear()

        actor = vtk.vtkImageSlice()
        actor.SetMapper(mapper)
        actor.SetProperty(prop)
        return actor, prop

    # ── Public API ────────────────────────────────────────────────────────

    def configure_base(self, rec: VolumeRecord, pg: PlaneGeometry) -> None:
        """
        Set up the base (first) volume and camera geometry.
        Clears any previously loaded volumes.
        """
        # Remove old actors
        for actor, _ in self._layers:
            self.renderer.RemoveActor(actor)
        self._layers.clear()

        self._pg = pg

        # Create actor for base volume
        actor, prop = self._make_slice_actor()
        actor.GetMapper().SetInputData(rec.vtk_idx)
        actor.SetUserTransform(rec.vtk_xform)
        self._apply_volume_property(prop, rec)
        self.renderer.AddActor(actor)
        self._layers.append((actor, prop))

        # Slider
        mid = pg.n // 2
        self._slider.blockSignals(True)
        self._slider.setMinimum(0)
        self._slider.setMaximum(pg.n - 1)
        self._slider.setValue(mid)
        self._slider.blockSignals(False)
        self._current_slice = mid
        self._lbl_slice.setText(f"{mid+1} / {pg.n}")

        # Re-add crosshairs above image actors
        self.renderer.RemoveActor(self._cross_h)
        self.renderer.RemoveActor(self._cross_v)
        self.renderer.AddActor(self._cross_h)
        self.renderer.AddActor(self._cross_v)

        self._init_camera(mid)
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

    def add_overlay(self, rec: VolumeRecord) -> int:
        """Add an overlay volume on top.  Returns the layer index."""
        actor, prop = self._make_slice_actor()
        actor.GetMapper().SetInputData(rec.vtk_idx)
        actor.SetUserTransform(rec.vtk_xform)
        self._apply_volume_property(prop, rec)

        # Insert before crosshairs (which are the last two actors)
        self.renderer.RemoveActor(self._cross_h)
        self.renderer.RemoveActor(self._cross_v)
        self.renderer.AddActor(actor)
        self.renderer.AddActor(self._cross_h)
        self.renderer.AddActor(self._cross_v)

        self._layers.append((actor, prop))
        self.vtk_widget.GetRenderWindow().Render()
        return len(self._layers) - 1

    def remove_overlay(self, layer_idx: int) -> None:
        """Remove an overlay by layer index (0 = base, cannot be removed here)."""
        if layer_idx == 0 or layer_idx >= len(self._layers):
            return
        actor, _ = self._layers.pop(layer_idx)
        self.renderer.RemoveActor(actor)
        self.vtk_widget.GetRenderWindow().Render()

    def update_layer_property(self, layer_idx: int, rec: VolumeRecord) -> None:
        """Update opacity / visibility / cmap / wl for a layer."""
        if layer_idx >= len(self._layers):
            return
        actor, prop = self._layers[layer_idx]
        self._apply_volume_property(prop, rec)
        actor.SetVisibility(rec.visible)
        self.vtk_widget.GetRenderWindow().Render()

    def set_wl(self, layer_idx: int, window: float, level: float) -> None:
        """Fast path: update only window/level for a layer (called during drag)."""
        if layer_idx >= len(self._layers):
            return
        _, prop = self._layers[layer_idx]
        prop.SetColorWindow(window)
        prop.SetColorLevel(level)
        self.vtk_widget.GetRenderWindow().Render()

    def set_slice(self, index: int) -> None:
        if self._pg is None:
            return
        index = max(0, min(index, self._pg.n - 1))
        self._slider.blockSignals(True)
        self._slider.setValue(index)
        self._slider.blockSignals(False)
        self._move_to_slice(index)

    def current_slice(self) -> int:
        return self._current_slice

    def world_position(self) -> np.ndarray:
        return self._focal_for(self._current_slice)

    def set_crosshair(self, world_pt: np.ndarray, half_len: float) -> None:
        if self._pg is None:
            return
        _set_line(self._cross_h, world_pt - self._pg.right * half_len, world_pt + self._pg.right * half_len)
        _set_line(self._cross_v, world_pt - self._pg.up    * half_len, world_pt + self._pg.up    * half_len)
        self._cross_h.VisibilityOn(); self._cross_v.VisibilityOn()
        self.vtk_widget.GetRenderWindow().Render()

    def set_crosshair_visible(self, visible: bool) -> None:
        """Show or hide both crosshair lines without moving them."""
        if visible:
            self._cross_h.VisibilityOn()
            self._cross_v.VisibilityOn()
        else:
            self._cross_h.VisibilityOff()
            self._cross_v.VisibilityOff()
        self.vtk_widget.GetRenderWindow().Render()
        

    # ── Internals ─────────────────────────────────────────────────────────

    def _apply_volume_property(self, prop: vtk.vtkImageProperty, rec: VolumeRecord):
        prop.SetOpacity(rec.opacity if rec.visible else 0.0)
        prop.SetColorWindow(rec.wl_window)
        prop.SetColorLevel(rec.wl_level)
        lut = _make_lut(CMAPS.get(rec.cmap), rec.lo, rec.hi, rec.cutoff)
        lut.SetAlphaRange(1.0, 1.0)   # per-entry alpha from SetTableValue is honoured
        prop.SetLookupTable(lut)
        # UseLookupTableScalarRange keeps window/level separate from LUT range
        prop.UseLookupTableScalarRangeOff()

    def _focal_for(self, index: int) -> np.ndarray:
        mid = self._pg.n // 2
        return self._pg.centre + (index - mid) * self._pg.step * self._pg.normal

    def _init_camera(self, index: int) -> None:
        focal  = self._focal_for(index)
        DIST   = 2000.
        side   = -1. if self._pg.flip_lr else 1.
        cam    = self.renderer.GetActiveCamera()
        cam.ParallelProjectionOn()
        cam.SetFocalPoint(*focal.tolist())
        cam.SetPosition(*(focal + self._pg.normal * DIST * side).tolist())
        cam.SetViewUp(*self._pg.up.tolist())
        cam.SetParallelScale(self._fit_parallel_scale())
        cam.SetClippingRange(1., DIST * 2.)

    def _fit_parallel_scale(self) -> float:
        """
        Compute the parallel scale that fits the entire image plane in the
        current viewport without cropping either axis.

        VTK parallel_scale = half the world-space height shown in the viewport.
        The viewport shows:
          world_height = 2 * ps
          world_width  = 2 * ps * (vp_w / vp_h)

        For the image to fit without cropping:
          ps >= plane_h / 2                      (fits vertically)
          ps >= plane_w * vp_h / (2 * vp_w)     (fits horizontally)

        Taking the max satisfies both.
        """
        if self._pg is None:
            return 1.0
        pw = self._pg.plane_w or self._pg.ps * 2.
        ph = self._pg.plane_h or self._pg.ps * 2.
        vp_w = self.vtk_widget.width()
        vp_h = self.vtk_widget.height()
        if vp_w <= 0 or vp_h <= 0:
            return self._pg.ps
        ps_v = ph / 2.0                       # fits vertically
        ps_h = pw * vp_h / (2.0 * vp_w)      # fits horizontally
        return max(ps_v, ps_h)

    def resizeEvent(self, event) -> None:     # type: ignore[override]
        """Refit the parallel scale whenever the viewport is resized."""
        super().resizeEvent(event)
        if self._pg is not None:
            ps = self._fit_parallel_scale()
            self.renderer.GetActiveCamera().SetParallelScale(ps)
            self.vtk_widget.GetRenderWindow().Render()

    def _move_to_slice(self, index: int) -> None:
        if self._pg is None:
            return
        cam = self.renderer.GetActiveCamera()
        # Compute how far to move along the slice normal.
        # We project the current focal point onto the normal to find its current
        # normal-axis position, then compute the delta to the target slice's
        # normal-axis position.  This leaves the lateral (panned) offset
        # completely untouched — only the depth along the normal changes.
        normal = self._pg.normal
        target_normal_pos = (
            self._pg.centre
            + (index - self._pg.n // 2) * self._pg.step * normal
        )
        old_fp = np.array(cam.GetFocalPoint())
        current_depth = np.dot(old_fp,  normal)
        target_depth  = np.dot(target_normal_pos, normal)
        delta = (target_depth - current_depth) * normal

        cam.SetFocalPoint(*(old_fp + delta).tolist())
        cam.SetPosition(*(np.array(cam.GetPosition()) + delta).tolist())
        self._current_slice = index
        self._lbl_slice.setText(f"{index+1} / {self._pg.n}")
        self.vtk_widget.GetRenderWindow().Render()

    def _on_slider(self, value: int) -> None:
        self._move_to_slice(value)
        self.slice_changed.emit(self.plane_idx, value)


# ── ElidedLabel ────────────────────────────────────────────────────────────

class ElidedLabel(QLabel):
    """
    A QLabel that elides (truncates with '…') its text when the available
    width is too narrow to show it in full, rather than expanding the parent
    layout.  The full text is always available as the tooltip.

    sizeHint() returns a compact width (just the minimum) so the label never
    forces the containing layout to grow.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full_text = text
        self.setToolTip(text)
        # Tell Qt this widget accepts any width — it won't push for more room
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)

    def setText(self, text: str) -> None:        # type: ignore[override]
        self._full_text = text
        self.setToolTip(text)
        super().setText(text)

    def paintEvent(self, event) -> None:
        from PySide6.QtGui import QPainter
        painter = QPainter(self)
        metrics = painter.fontMetrics()
        elided  = metrics.elidedText(
            self._full_text, Qt.TextElideMode.ElideRight, self.width())
        painter.drawText(self.rect(), self.alignment() or Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided)


# ── ColourBar ──────────────────────────────────────────────────────────────

def _cmap_rgb(name: str | None, t: float) -> tuple[int, int, int]:
    """
    Sample a colourmap at normalised position t ∈ [0, 1].
    Returns (R, G, B) in 0-255 range.  Mirrors the logic in _make_lut exactly
    so the bar is always consistent with what VTK renders.
    """
    def lerp(a, b, f): return a + f * (b - a)

    if name is None or name == "grey":
        v = int(t * 255)
        return (v, v, v)

    elif name == "hot":
        # hue 0→0.1667, sat 1→0, val 0.5→1
        h = t * 0.1667
        s = 1.0 - t
        v = 0.5 + t * 0.5
        return _hsv_to_rgb255(h, s, v)

    elif name == "cool":
        # hue 0.5→0.833, sat=1, val=1
        h = 0.5 + t * 0.333
        return _hsv_to_rgb255(h, 1.0, 1.0)

    elif name == "green":
        return _hsv_to_rgb255(0.333, t, t)

    elif name == "red":
        return _hsv_to_rgb255(0.0, t, t)

    elif name == "jet":
        jet_cps = [
            (0.000, 0.000, 0.000, 0.500),
            (0.125, 0.000, 0.000, 1.000),
            (0.375, 0.000, 1.000, 1.000),
            (0.625, 1.000, 1.000, 0.000),
            (0.875, 1.000, 0.000, 0.000),
            (1.000, 0.500, 0.000, 0.000),
        ]
        for k in range(len(jet_cps) - 1):
            t0, r0, g0, b0 = jet_cps[k]
            t1, r1, g1, b1 = jet_cps[k + 1]
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0) if t1 > t0 else 0.
                return (int(lerp(r0, r1, f) * 255),
                        int(lerp(g0, g1, f) * 255),
                        int(lerp(b0, b1, f) * 255))
        return (128, 0, 0)

    return (int(t * 255),) * 3


def _hsv_to_rgb255(h, s, v) -> tuple[int, int, int]:
    """HSV (0-1 each) → (R, G, B) 0-255."""
    if s == 0:
        val = int(v * 255)
        return (val, val, val)
    h6 = (h % 1.0) * 6.0
    i  = int(h6)
    f  = h6 - i
    p  = v * (1 - s)
    q  = v * (1 - s * f)
    t_ = v * (1 - s * (1 - f))
    rgb = [(v, t_, p), (q, v, p), (p, v, t_),
           (p, q, v), (t_, p, v), (v, p, q)][i % 6]
    return tuple(int(c * 255) for c in rgb)


class ColourBar(QWidget):
    """
    A vertical colourbar showing the window/level mapping for the currently
    selected volume.

    The bar spans [level - window/2, level + window/2] (the visible data
    range).  Five evenly-spaced tick marks with numeric labels are drawn on
    the right side.  The colourmap exactly mirrors the LUT used by VTK.

    Call update_bar(rec) whenever the selected volume or its WL/cmap changes.
    """

    BAR_W   = 24    # width of the gradient strip in pixels
    TICK_W  = 6     # tick mark length
    PAD_L   = 6     # left padding
    PAD_R   = 52    # right padding (room for labels)
    PAD_T   = 12    # top / bottom padding
    PAD_B   = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cmap:   str | None = None
        self._lo:     float = 0.
        self._hi:     float = 1.
        self._window: float = 1.
        self._level:  float = 0.5
        self._cutoff: float | None = None

        total_w = self.PAD_L + self.BAR_W + self.TICK_W + self.PAD_R
        self.setFixedWidth(total_w)
        self.setMinimumHeight(120)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setStyleSheet(f"background:{BG_DARK};")

    def update_bar(self, rec: "VolumeRecord") -> None:
        self._cmap   = CMAPS.get(rec.cmap)
        self._lo     = rec.lo
        self._hi     = rec.hi
        self._window = rec.wl_window
        self._level  = rec.wl_level
        self._cutoff = rec.cutoff
        self.update()   # trigger repaint

    def render_to_image(self, width: int, height: int) -> "QImage":
        """
        Render the colourbar into a QImage of exactly (width, height) pixels
        with a white background.  Used by grab_screenshot so the bar matches
        the VTK viewport height precisely, regardless of on-screen widget size.
        """
        from PySide6.QtGui import QImage, QPainter, QColor
        img = QImage(width, height, QImage.Format.Format_RGB32)
        img.fill(QColor("white"))
        painter = QPainter(img)
        self._paint_into(painter, width, height, bg_color=QColor("white"),
                         label_color=QColor("#222222"), tick_color=QColor("#555566"),
                         border_color=QColor("#888899"),fontsize=16)
        painter.end()
        return img

    def paintEvent(self, event) -> None:  # type: ignore[override]
        from PySide6.QtGui import QPainter, QColor
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self._paint_into(painter, self.width(), self.height(),
                         bg_color=QColor(BG_DARK),
                         label_color=QColor(TEXT_LBL),
                         tick_color=QColor("#888899"),
                         border_color=QColor("#444455"))
        painter.end()

    def _paint_into(self, painter, w: int, h: int,
                    bg_color, label_color, tick_color, border_color,
                    fontsize = 10) -> None:
        from PySide6.QtGui import QColor, QFont

        bar_x = self.PAD_L
        bar_h = h - self.PAD_T - self.PAD_B
        bar_y = self.PAD_T

        if bar_h < 4:
            return

        half_w    = self._window / 2.0
        disp_lo   = self._level - half_w
        disp_hi   = self._level + half_w
        data_span = self._hi - self._lo or 1.0

        # ── gradient strip ───────────────────────────────────────────────────
        for py in range(bar_h):
            t_bar  = 1.0 - py / max(bar_h - 1, 1)
            scalar = disp_lo + t_bar * (disp_hi - disp_lo)
            t_lut  = max(0.0, min(1.0, (scalar - self._lo) / data_span))

            if self._cutoff is not None and scalar < self._cutoff:
                painter.setPen(bg_color)
            else:
                r, g, b = _cmap_rgb(self._cmap, t_lut)
                painter.setPen(QColor(r, g, b))

            painter.drawLine(bar_x, bar_y + py,
                             bar_x + self.BAR_W - 1, bar_y + py)

        # ── border ───────────────────────────────────────────────────────────
        painter.setPen(border_color)
        painter.drawRect(bar_x, bar_y, self.BAR_W - 1, bar_h - 1)

        # ── ticks and labels ─────────────────────────────────────────────────
        tick_x  = bar_x + self.BAR_W
        label_x = tick_x + self.TICK_W + 2

        font = QFont()
        font.setPixelSize(fontsize)
        painter.setFont(font)

        for i in range(5):
            t_tick = i / 4
            scalar = disp_lo + t_tick * (disp_hi - disp_lo)
            py     = bar_y + bar_h - 1 - int(t_tick * (bar_h - 1))

            painter.setPen(tick_color)
            painter.drawLine(tick_x, py, tick_x + self.TICK_W, py)

            painter.setPen(label_color)
            painter.drawText(label_x, py - 8,
                             w - label_x - 2, 18,
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             f"{scalar:.4g}")


# ── LayerRow ───────────────────────────────────────────────────────────────

class LayerRow(QWidget):
    """
    One row in the layer panel representing a single loaded volume.
    Signals:
      opacity_changed    (vol_idx, float 0..1)
      cmap_changed       (vol_idx, str)
      visibility_changed (vol_idx, bool)
      remove_requested   (vol_idx)
      wl_select          (vol_idx)  — user clicked the WL target button
    """
    opacity_changed    = Signal(int, float)
    cmap_changed       = Signal(int, str)
    visibility_changed = Signal(int, bool)
    remove_requested   = Signal(int)
    wl_select          = Signal(int)
    cutoff_changed     = Signal(int, object)   # (vol_idx, float | None)

    def __init__(self, vol_idx: int, rec: VolumeRecord,parent=None,tissue_label=False):
        super().__init__(parent)
        self._vol_idx = vol_idx
        self._is_base = (vol_idx == 0)
        self._tissue_label=tissue_label
        self._id=rec.id
        self._build_ui(rec)

    def set_wl_active(self, active: bool) -> None:
        """Highlight this row's WL button as the current WL target."""
        color = VOL_COLORS[self._vol_idx % len(VOL_COLORS)]
        if active:
            self._wl_btn.setStyleSheet(f"""
                QToolButton {{ border:none; background:{color}; color:#0d0d0f;
                               border-radius:3px; font-size:10px; font-weight:bold; }}
            """)
        else:
            self._wl_btn.setStyleSheet(f"""
                QToolButton {{ border:1px solid #444455; background:transparent;
                               color:{TEXT_DIM}; border-radius:3px; font-size:10px; }}
                QToolButton:hover {{ border-color:{color}; color:{color}; }}
            """)

    def update_wl_readout(self, window: float, level: float) -> None:
        self._wl_lbl.setText(f"W {window:.0f}  L {level:.0f}")

    def _build_ui(self, rec: VolumeRecord):
        color = VOL_COLORS[self._vol_idx % len(VOL_COLORS)]
        self.setStyleSheet(f"""
            LayerRow {{
                background:{BG_LAYER};
                border-left: 3px solid {color};
                border-radius: 3px;
                margin: 2px 0;
            }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)

        # ── Header row: eye | WL-target | name | [×] ──────────────────
        hrow = QHBoxLayout(); hrow.setSpacing(5)

        # Eye / visibility toggle
        self._eye_btn = QToolButton()
        self._eye_btn.setCheckable(True)
        self._eye_btn.setChecked(True)
        self._eye_btn.setFixedSize(22, 22)
        self._eye_btn.setStyleSheet(f"""
            QToolButton {{ border:none; background:transparent; font-size:14px; }}
        """)
        self._eye_btn.setText("👁")

        def _on_eye_toggled(checked: bool) -> None:
            self._eye_btn.setText("👁" if checked else "🚫")
            self.visibility_changed.emit(self._vol_idx, checked)

        self._eye_btn.toggled.connect(_on_eye_toggled)
        hrow.addWidget(self._eye_btn)

        # WL target button — clicking makes this the active WL volume
        self._wl_btn = QToolButton()
        self._wl_btn.setText("W/L")
        self._wl_btn.setFixedSize(30, 22)
        self._wl_btn.clicked.connect(lambda: self.wl_select.emit(self._vol_idx))
        hrow.addWidget(self._wl_btn)
        if self._tissue_label:
            self._wl_btn.setVisible(False)

        lbl = ElidedLabel(rec.name)
        lbl.setStyleSheet(f"color:{color}; font-size:11px; font-weight:bold;")
        lbl.setToolTip(rec.name)
        hrow.addWidget(lbl, 1)

        # Cutoff threshold input ─────────────────────────────────────────
        co_lbl = QLabel("≥")
        co_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        hrow.addWidget(co_lbl)
        if self._tissue_label:
            co_lbl.setVisible(False)


        self._cutoff_edit = QLineEdit()
        self._cutoff_edit.setPlaceholderText("cutoff")
        self._cutoff_edit.setFixedWidth(56)
        self._cutoff_edit.setFixedHeight(22)
        self._cutoff_edit.setValidator(QDoubleValidator())
        self._cutoff_edit.setStyleSheet(f"""
            QLineEdit {{
                background:#1a1a22; color:{color};
                border:1px solid #444455; border-radius:3px;
                padding:0 4px; font-size:11px; font-family:monospace;
            }}
            QLineEdit:focus {{ border-color:{color}; }}
        """)
        self._cutoff_edit.setToolTip(
            "Mask values below this threshold (leave empty to disable)")

        self._cutoff_edit.editingFinished.connect(self._on_cutoff_changed)
        hrow.addWidget(self._cutoff_edit)
        if self._tissue_label:
            self._cutoff_edit.setVisible(False)

        if not self._is_base:
            rm_btn = QToolButton()
            rm_btn.setText("✕")
            rm_btn.setFixedSize(20, 20)
            rm_btn.setStyleSheet(f"""
                QToolButton {{ border:none; background:transparent;
                               color:{TEXT_DIM}; font-size:11px; }}
                QToolButton:hover {{ color:#ff6b6b; }}
            """)
            rm_btn.clicked.connect(lambda: self.remove_requested.emit(self._vol_idx))
            hrow.addWidget(rm_btn)

        lay.addLayout(hrow)

        # ── W/L readout ────────────────────────────────────────────────
        self._wl_lbl = QLabel(
            f"W {rec.wl_window:.0f}  L {rec.wl_level:.0f}")
        self._wl_lbl.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:10px; font-family:monospace;")
        lay.addWidget(self._wl_lbl)

        # Style the WL button initially (inactive)
        self.set_wl_active(False)

        if self._tissue_label:
            self._wl_lbl.setVisible(False)

        # ── Opacity row (overlays only) ────────────────────────────────
        # if not self._is_base:
        orow = QHBoxLayout(); orow.setSpacing(6)
        olk = QLabel("Opacity")
        olk.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        orow.addWidget(olk)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(100)
        self._opacity_slider.setFixedHeight(16)
        self._opacity_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ height:3px; background:#333340; }}
            QSlider::handle:horizontal {{
                background:{color}; width:12px; height:12px;
                margin:-5px 0; border-radius:6px; }}
            QSlider::sub-page:horizontal {{ background:{color}; }}
        """)
        self._opacity_slider.valueChanged.connect(
            lambda v: self.opacity_changed.emit(self._vol_idx, v / 100.0))
        orow.addWidget(self._opacity_slider, 1)

        self._pct_lbl = QLabel("100%")
        self._pct_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        self._pct_lbl.setFixedWidth(32)
        self._opacity_slider.valueChanged.connect(
            lambda v: self._pct_lbl.setText(f"{v}%"))
        orow.addWidget(self._pct_lbl)
        lay.addLayout(orow)

        # ── Colourmap row ──────────────────────────────────────────────
        crow = QHBoxLayout(); crow.setSpacing(6)
        clbl = QLabel("Colourmap")
        clbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        crow.addWidget(clbl)

        if self._tissue_label:
            clbl.setVisible(False)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(list(CMAPS.keys())[1:])
        self._cmap_combo.setCurrentText("Grey")
        self._cmap_combo.setFixedHeight(22)
        self._cmap_combo.setStyleSheet(f"""
            QComboBox {{
                background:#1e1e25; color:{TEXT}; border:1px solid #444455;
                border-radius:3px; padding:1px 6px; font-size:11px;
            }}
            QComboBox::drop-down {{ border:none; }}
            QComboBox QAbstractItemView {{
                background:#1e1e25; color:{TEXT}; selection-background-color:#333345;
            }}
        """)
        self._cmap_combo.currentTextChanged.connect(
            lambda t: self.cmap_changed.emit(self._vol_idx, t))
        crow.addWidget(self._cmap_combo, 1)
        if self._tissue_label:
            self._cmap_combo.setVisible(False)
        lay.addLayout(crow)

    def _on_cutoff_changed(self):
        txt = self._cutoff_edit.text().strip()
        if txt == "":
            self.cutoff_changed.emit(self._vol_idx, None)
        else:
            try:
                self.cutoff_changed.emit(self._vol_idx, float(txt))
            except ValueError:
                pass

# ── LayerPanel ─────────────────────────────────────────────────────────────

class LayerPanel(QWidget):
    """
    Scrollable sidebar listing all loaded volumes as LayerRows.
    """
    opacity_changed    = Signal(int, float)
    cmap_changed       = Signal(int, str)
    visibility_changed = Signal(int, bool)
    remove_requested   = Signal(int)
    wl_select_changed  = Signal(int)    # (vol_idx) — WL target changed
    cutoff_changed     = Signal(int, object)   # (vol_idx, float | None)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(240)
        self.setMaximumWidth(240)
        self.setStyleSheet(f"background:{BG_PANEL};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        hdr = QLabel("  LAYERS")
        hdr.setFixedHeight(32)
        hdr.setStyleSheet(
            f"background:{BG_PANEL}; color:{TEXT_DIM}; font-size:10px; "
            f"font-weight:bold; letter-spacing:2px; "
            f"border-bottom:1px solid #333340;")
        outer.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        outer.addWidget(scroll, 1)

        self._container = QWidget()
        self._container.setStyleSheet(f"background:{BG_PANEL};")
        self._vlay = QVBoxLayout(self._container)
        self._vlay.setContentsMargins(6, 6, 6, 6)
        self._vlay.setSpacing(4)
        self._vlay.addStretch(1)
        scroll.setWidget(self._container)

        self._rows: list[LayerRow] = []
        self._active_wl: int = 0   # which row has WL active

    def add_row(self, vol_idx: int, rec: VolumeRecord,tissue_label=False) -> None:
        row = LayerRow(vol_idx, rec, self._container,tissue_label)
        row.opacity_changed.connect(self.opacity_changed)
        row.cmap_changed.connect(self.cmap_changed)
        row.visibility_changed.connect(self.visibility_changed)
        row.remove_requested.connect(self.remove_requested)
        row.wl_select.connect(self._on_row_wl_select)
        row.cutoff_changed.connect(self.cutoff_changed)
        self._vlay.insertWidget(self._vlay.count() - 1, row)
        self._rows.append(row)
        # Auto-select WL on the newest row
        self._set_active_wl(vol_idx)

    def remove_row(self, vol_idx: int) -> None:
        for row in self._rows:
            if row._vol_idx == vol_idx:
                self._vlay.removeWidget(row)
                row.deleteLater()
                self._rows.remove(row)
                break
        for i, row in enumerate(self._rows):
            row._vol_idx = i
        # Fall back to base if active row was removed
        if self._active_wl >= len(self._rows):
            self._set_active_wl(0)

    def update_wl_readout(self, vol_idx: int, window: float, level: float) -> None:
        for row in self._rows:
            if row._vol_idx == vol_idx:
                row.update_wl_readout(window, level)
                break

    def _on_row_wl_select(self, vol_idx: int) -> None:
        self._set_active_wl(vol_idx)
        self.wl_select_changed.emit(vol_idx)

    def _set_active_wl(self, vol_idx: int) -> None:
        self._active_wl = vol_idx
        for row in self._rows:
            row.set_wl_active(row._vol_idx == vol_idx)
        self.wl_select_changed.emit(vol_idx)


# ── NiftiViewer ────────────────────────────────────────────────────────────

class NiftiViewer(QWidget):
    """
    Three-view viewer that can display multiple NIfTI volumes simultaneously.

    Volume 0 (base) drives camera geometry in both AFFINE and MEDICAL modes.
    Overlay volumes (indices 1+) are rendered with their own affine transforms
    so that different-grid overlays are still correctly co-registered.

    Window/level dragging (right-mouse-button drag in any viewport) is routed
    to the currently *selected* volume (highlighted in the layer panel).
    """

    MODE_AFFINE  = "affine"
    MODE_MEDICAL = "medical"

    # Emitted after WL changes so the layer panel can update its readout
    wl_updated = Signal(int, float, float)   # (vol_idx, window, level)

    notify_load_base = Signal()

    # Emitted whenever the crosshair moves to a new RAS position
    ras_changed = Signal(float, float, float)   # (x, y, z)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._volumes:       list[VolumeRecord] = []
        self._base_affine:   np.ndarray | None = None
        self._base_shape:    tuple | None = None
        self._ras_min:       np.ndarray | None = None
        self._ras_max:       np.ndarray | None = None
        self._mode           = self.MODE_AFFINE
        self._neurogical   = False
        self._geoms:         list[PlaneGeometry] = []
        self._half_len:      float = 100.
        self._selected_vol:  int = 0
        self._crosshair_visible: bool = True
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"background:{BG_DARK};")
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Outer splitter: viewports | layer panel
        outer_spl = QSplitter(Qt.Orientation.Horizontal)
        outer_spl.setStyleSheet("QSplitter::handle{background:#333340;width:2px;}")
        outer_spl.setChildrenCollapsible(False)

        # Inner splitter — three viewports side by side
        # Added directly to outer_spl (no intermediate wrapper widget) so there
        # is no dead space between the last viewport and the layer panel.
        spl = QSplitter(Qt.Orientation.Horizontal)
        spl.setStyleSheet("""
            QSplitter { margin: 6px 4px 6px 8px; }
            QSplitter::handle { background:#333340; width:4px; }
        """)
        spl.setChildrenCollapsible(False)
        self._vps: list[SliceViewport] = []
        for i in range(3):
            vp = SliceViewport(i, self)
            vp.setMinimumWidth(180)
            vp.slice_changed.connect(self._on_slice_changed)
            vp.set_wl_callbacks(self._on_wl_drag, self._on_wl_end)
            self._vps.append(vp)
            spl.addWidget(vp)
        outer_spl.addWidget(spl)

        # Colourbar — narrow strip between viewports and layer panel
        self._colorbar = ColourBar(self)
        outer_spl.addWidget(self._colorbar)

        # Layer panel
        self._layer_panel = LayerPanel(self)
        self._layer_panel.opacity_changed.connect(self._on_opacity_changed)
        self._layer_panel.cmap_changed.connect(self._on_cmap_changed)
        self._layer_panel.visibility_changed.connect(self._on_visibility_changed)
        self._layer_panel.remove_requested.connect(self._on_remove_requested)
        self._layer_panel.wl_select_changed.connect(self._on_wl_select_changed)
        self._layer_panel.cutoff_changed.connect(self._on_cutoff_changed)
        outer_spl.addWidget(self._layer_panel)

        # Viewports expand, colorbar and layer panel stay fixed
        outer_spl.setStretchFactor(0, 1)
        outer_spl.setStretchFactor(1, 0)
        outer_spl.setStretchFactor(2, 0)

        root.addWidget(outer_spl)

    # ── Public API ────────────────────────────────────────────────────────

    def load_base(self, ni_path: object, target:np.ndarray, name='',tissue_label=False) -> tuple:
        """Load the first (base) volume.  Clears all existing volumes."""
        rec, shape, zooms, code, affine = load_volume_record(ni_path,inname=name)
        self._volumes.clear()
        self._volumes.append(rec)
        self._base_affine = affine
        self._base_shape  = shape
        self._ras_min, self._ras_max = _ras_bbox(affine, shape)
        self._half_len = np.linalg.norm(self._ras_max - self._ras_min) * 0.65

        # Rebuild layer panel
        for row in list(self._layer_panel._rows):
            self._layer_panel._vlay.removeWidget(row)
            row.deleteLater()
        self._layer_panel._rows.clear()
        self._layer_panel.add_row(0, rec,tissue_label=tissue_label)

        self._selected_vol = 0
        self._target=target
        self._refresh()
        self.notify_load_base.emit()
        return shape, zooms, code

    def add_overlay(self, ni_path: object, name='',use_percentile=False,id='') -> tuple:
        """Add an overlay volume.  Requires at least one base volume loaded."""
        if not self._volumes:
            raise RuntimeError("Load a base volume first.")
        rec, shape, zooms, code, _ = load_volume_record(ni_path,inname=name,use_percentile=use_percentile,id=id)
        self._volumes.append(rec)
        vol_idx = len(self._volumes) - 1
        self._layer_panel.add_row(vol_idx, rec)

        for vp in self._vps:
            vp.add_overlay(rec)

        return shape, zooms, code

    def set_mode(self, mode: str, neurogical: bool = False) -> None:
        self._mode = mode
        self._neurogical = neurogical
        if self._volumes:
            self._refresh()

    def set_neurological(self, enabled: bool) -> None:
        if self._neurogical == enabled:
            return
        self._neurogical = enabled
        if self._mode == self.MODE_MEDICAL and self._volumes:
            self._refresh()

    def set_crosshair_visible(self, visible: bool) -> None:
        """Show or hide crosshairs in all three viewports."""
        self._crosshair_visible = visible
        for vp in self._vps:
            vp.set_crosshair_visible(visible)

    def reset_view(self) -> None:
        """
        Restore the default view for all volumes:
          - Sliders back to mid-slice
          - Camera pan and zoom reset to initial state
          - Window/level reset to the original data range for each volume
          - Layer panel WL readouts updated

        Everything else (opacity, colourmap, cutoff, visibility) is left
        unchanged — only the interactive adjustments are undone.
        """
        if not self._volumes:
            return

        # 1. Reset window/level on every volume to its initial values
        for rec in self._volumes:
            rec.wl_window = rec.hi - rec.lo or 1.
            rec.wl_level  = (rec.hi + rec.lo) / 2.

        # 2. Push reset WL to all viewports and update layer panel readouts
        for vol_idx, rec in enumerate(self._volumes):
            for vp in self._vps:
                vp.set_wl(vol_idx, rec.wl_window, rec.wl_level)
            self.wl_updated.emit(vol_idx, rec.wl_window, rec.wl_level)

        # 3. Reset camera (pan + zoom) and slider on each viewport
        for vp in self._vps:
            vp.reset_camera()

        # 4. Redraw crosshairs at the restored mid-slice position
        if self._mode == self.MODE_AFFINE:
            self._vps[0]._slider.setValue(self._target[1])
            self._vps[1]._slider.setValue(self._target[0])
            self._vps[2]._slider.setValue(self._target[2])
        else:
            target_med=self.affine_to_medical_coords(tuple(self._target))
            self._vps[0]._slider.setValue(int(target_med[0]))
            self._vps[1]._slider.setValue(int(target_med[1]))
            self._vps[2]._slider.setValue(int(target_med[2]))

        # 5. Refresh colorbar with reset WL
        if self._selected_vol < len(self._volumes):
            self._colorbar.update_bar(self._volumes[self._selected_vol])


    def navigate_to_ras(self, x: float, y: float, z: float) -> None:
        """
        Move all three viewports so the crosshair sits at RAS world position (x, y, z).

        In AFFINE mode, the RAS point is converted to voxel indices using the
        inverse affine of the base volume, then each slider is set to the
        corresponding slice index.

        In MEDICAL mode, the RAS point is projected onto the three canonical
        RAS-axis grids (axial=Z, coronal=Y, sagittal=X).

        Silently does nothing if no base volume is loaded or the point is outside
        the volume bounding box.
        """
        if self._base_affine is None or not self._geoms:
            return

        world = np.array([x, y, z])

        if self._mode == self.MODE_AFFINE:
            # RAS world → voxel index via inverse affine
            affine_inv = np.linalg.inv(self._base_affine)
            vox = (affine_inv @ np.append(world, 1.))[:3]

            sp, R = _decompose(self._base_affine)

            # VP0: normal=dj (Slice j) → scroll along j-axis → vox[1]
            # VP1: normal=di (Slice i) → scroll along i-axis → vox[0]
            # VP2: normal=dk (Slice k) → scroll along k-axis → vox[2]
            indices = [
                int(np.clip(round(vox[1]), 0, self._geoms[0].n - 1)),
                int(np.clip(round(vox[0]), 0, self._geoms[1].n - 1)),
                int(np.clip(round(vox[2]), 0, self._geoms[2].n - 1)),
            ]
        else:
            # RAS world → medical grid indices
            sp, _ = _decompose(self._base_affine)
            iso   = float(np.min(sp))
            indices = []
            for g in self._geoms:
                raw = np.dot(world - self._ras_min, g.normal) / iso
                indices.append(int(np.clip(round(raw), 0, g.n - 1)))

        for vp, idx in zip(self._vps, indices):
            vp.set_slice(idx)

        self._update_crosshairs()

    def get_current_ras(self) -> tuple[float, float, float]:
        """
        Return the RAS world coordinates of the current crosshair intersection.
        Returns (0, 0, 0) if no base volume is loaded.
        """
        if not self._geoms:
            return (0., 0., 0.)
        pts     = [vp.world_position() for vp in self._vps]
        normals = [g.normal for g in self._geoms]
        A = np.vstack(normals)
        b = np.array([np.dot(normals[i], pts[i]) for i in range(3)])
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = self._geoms[0].centre.copy()
        return (float(w[0]), float(w[1]), float(w[2]))

    def affine_to_medical_coords(
        self,
        ijk: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """
        Convert a voxel coordinate in the base volume's native (affine) index
        space to the equivalent slider indices in medical (RAS) display mode.

        Parameters
        ----------
        ijk : (i, j, k)
            Voxel indices in the base volume's index space (can be fractional).

        Returns
        -------
        (i_axial, i_coronal, i_sagittal) : float tuple
            Corresponding slider indices for the three medical-mode viewports:
              i_axial    — Axial viewport    (scroll along RAS-Z)
              i_coronal  — Coronal viewport  (scroll along RAS-Y)
              i_sagittal — Sagittal viewport (scroll along RAS-X)

            Values are continuous (fractional) so you can round to int for
            the nearest slice.  They are clipped to [0, n-1] of each plane.

        Raises
        ------
        RuntimeError
            If no base volume has been loaded yet.

        Notes
        -----
        The medical grid is defined as:
          origin  = ras_min  (minimum-coordinate corner of the RAS bounding box)
          spacing = iso      (minimum voxel size of the base affine)
          axes    = world X, Y, Z

        So for any world point W:
          i_axial    = (W[2] - ras_min[2]) / iso
          i_coronal  = (W[1] - ras_min[1]) / iso
          i_sagittal = (W[0] - ras_min[0]) / iso
        """
        if self._base_affine is None or self._ras_min is None:
            raise RuntimeError("No base volume loaded.")

        # Step 1 — affine voxel index → RAS world position
        i, j, k = ijk
        world = (self._base_affine @ np.array([i, j, k, 1.0]))[:3]

        # Step 2 — RAS world → medical grid indices
        sp, _ = _decompose(self._base_affine)
        iso   = float(np.min(sp))

        geoms = medical_plane_geoms(
            self._ras_min, self._ras_max, iso, neurogical=False)

        def _clamp(val, pg):
            return float(np.clip(val, 0, pg.n - 1))

        # Each PlaneGeometry's scroll axis = its normal.  The scroll index is
        # the projection of (world - ras_min) onto the normal, divided by iso.
        # For axial    (normal=[0,0,1]): index = (world[2] - ras_min[2]) / iso
        # For coronal  (normal=[0,1,0]): index = (world[1] - ras_min[1]) / iso
        # For sagittal (normal=[1,0,0]): index = (world[0] - ras_min[0]) / iso
        i_ax  = _clamp(np.dot(world - self._ras_min, geoms[0].normal) / iso, geoms[0])
        i_co  = _clamp(np.dot(world - self._ras_min, geoms[1].normal) / iso, geoms[1])
        i_sa  = _clamp(np.dot(world - self._ras_min, geoms[2].normal) / iso, geoms[2])

        return (i_ax, i_co, i_sa)

    def grab_screenshot(self, path: str) -> None:
        """
        Capture all three viewports plus the colourbar and save side-by-side as PNG.

        Each viewport is captured via vtkWindowToImageFilter at its native
        render-window resolution.  The colourbar is captured via QWidget.grab().
        All strips are stitched horizontally using QPainter on a QImage.
        """
        from vtkmodules.vtkIOImage import vtkPNGWriter
        from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
        from PySide6.QtGui import QImage, QPainter
        import tempfile, os as _os

        strips: list[QImage] = []
        tmp_files: list[str] = []

        # Capture each viewport via VTK (back buffer, no UI chrome)
        for vp in self._vps:
            rw = vp.vtk_widget.GetRenderWindow()
            rw.Render()

            wti = vtkWindowToImageFilter()
            wti.SetInput(rw)
            wti.SetScale(1)
            wti.ReadFrontBufferOff()
            wti.Update()

            fd, tmp = tempfile.mkstemp(suffix=".png")
            _os.close(fd)
            tmp_files.append(tmp)

            writer = vtkPNGWriter()
            writer.SetFileName(tmp)
            writer.SetInputConnection(wti.GetOutputPort())
            writer.Write()

            strips.append(QImage(tmp))

        # Capture the colourbar at exactly the same height as the VTK strips,
        # rendered on a white background into a fresh QImage (no widget sizing).
        vtk_h = strips[0].height() if strips else self._colorbar.height()
        cb_img = self._colorbar.render_to_image(self._colorbar.width(), vtk_h)
        strips.append(cb_img)

        # Stitch all strips side-by-side on a white background
        total_w = sum(img.width()  for img in strips)
        max_h   = max(img.height() for img in strips)
        combined = QImage(total_w, max_h, QImage.Format.Format_RGB32)
        combined.fill(QColor("white"))

        painter = QPainter(combined)
        x = 0
        for img in strips:
            # Centre each strip vertically if heights differ
            y_off = (max_h - img.height()) // 2
            painter.drawImage(x, y_off, img)
            x += img.width()
        painter.end()

        combined.save(path, "PNG")

        for tmp in tmp_files:
            try:
                _os.unlink(tmp)
            except OSError:
                pass

    # ── Window / level drag ───────────────────────────────────────────────

    def _on_wl_select_changed(self, vol_idx: int) -> None:
        self._selected_vol = vol_idx
        if vol_idx < len(self._volumes):
            self._colorbar.update_bar(self._volumes[vol_idx])

    def _on_wl_drag(self, dw_norm: float, dl_norm: float) -> None:
        """
        dw_norm, dl_norm are drag distances normalised to [-1..+1] by viewport size.
        We scale them by the volume's full data range — identical to vtkInteractorStyleImage:
          window += dw_norm * range
          level  -= dl_norm * range   (VTK: up = decrease level, down = increase)
        This gives a natural feel: dragging the full width changes window by one full range.
        """
        idx = self._selected_vol
        if idx >= len(self._volumes):
            return
        rec   = self._volumes[idx]
        scale = rec.hi - rec.lo or 1.0
        rec.wl_window = max(1.0, rec.wl_window + dw_norm * scale)
        rec.wl_level  = rec.wl_level  - dl_norm * scale
        for vp in self._vps:
            vp.set_wl(idx, rec.wl_window, rec.wl_level)
        self._colorbar.update_bar(rec)
        self.wl_updated.emit(idx, rec.wl_window, rec.wl_level)

    def _on_wl_end(self) -> None:
        pass   # could trigger a full property refresh if needed

    # ── Layer signal handlers ─────────────────────────────────────────────

    def _on_opacity_changed(self, vol_idx: int, opacity: float) -> None:
        if vol_idx >= len(self._volumes):
            return
        self._volumes[vol_idx].opacity = opacity
        for vp in self._vps:
            vp.update_layer_property(vol_idx, self._volumes[vol_idx])

    def _on_cmap_changed(self, vol_idx: int, cmap: str) -> None:
        if vol_idx >= len(self._volumes):
            return
        self._volumes[vol_idx].cmap = cmap
        for vp in self._vps:
            vp.update_layer_property(vol_idx, self._volumes[vol_idx])
        if vol_idx == self._selected_vol:
            self._colorbar.update_bar(self._volumes[vol_idx])

    def _on_cutoff_changed(self, vol_idx: int, cutoff) -> None:
        if vol_idx >= len(self._volumes):
            return
        self._volumes[vol_idx].cutoff = cutoff   # float or None
        for vp in self._vps:
            vp.update_layer_property(vol_idx, self._volumes[vol_idx])
        if vol_idx == self._selected_vol:
            self._colorbar.update_bar(self._volumes[vol_idx])

    def _on_visibility_changed(self, vol_idx: int, visible: bool) -> None:
        if vol_idx >= len(self._volumes):
            return
        self._volumes[vol_idx].visible = visible
        for vp in self._vps:
            vp.update_layer_property(vol_idx, self._volumes[vol_idx])

    def _on_remove_requested(self, vol_idx: int) -> None:
        if vol_idx == 0 or vol_idx >= len(self._volumes):
            return
        self._volumes.pop(vol_idx)
        for vp in self._vps:
            vp.remove_overlay(vol_idx)
        self._layer_panel.remove_row(vol_idx)

    # ── Internal refresh ──────────────────────────────────────────────────

    def _refresh(self):
        """Rebuild camera geometry and reconfigure all viewports."""
        if not self._volumes:
            return

        if self._mode == self.MODE_AFFINE:
            self._geoms = affine_plane_geoms(self._base_affine, self._base_shape)
        else:
            sp, _ = _decompose(self._base_affine)
            iso   = float(np.min(sp))
            self._geoms = medical_plane_geoms(
                self._ras_min, self._ras_max, iso, self._neurogical)

        names = AFFINE_NAMES if self._mode == self.MODE_AFFINE else MEDICAL_NAMES

        for i, vp in enumerate(self._vps):
            vp.set_title(names[i])
            # Reconfigure base volume (resets camera + slider)
            vp.configure_base(self._volumes[0], self._geoms[i])
            # Re-add all overlays
            for rec in self._volumes[1:]:
                vp.add_overlay(rec)

        # Orientation labels: shown only in medical mode
        if self._mode == self.MODE_MEDICAL:
            self._apply_orientation_labels()
        else:
            for vp in self._vps:
                vp.clear_orientation_labels()

        if self._mode == self.MODE_AFFINE:
            self._vps[0]._slider.setValue(self._target[1])
            self._vps[1]._slider.setValue(self._target[0])
            self._vps[2]._slider.setValue(self._target[2])
        else:
            target_med=self.affine_to_medical_coords(tuple(self._target))
            self._vps[0]._slider.setValue(int(target_med[0]))
            self._vps[1]._slider.setValue(int(target_med[1]))
            self._vps[2]._slider.setValue(int(target_med[2]))

    # ── Crosshair sync ────────────────────────────────────────────────────

    def _apply_orientation_labels(self) -> None:
        """
        Apply anatomical edge labels to the three medical-mode viewports.

        In neurological convention (neurogical=False):
          Screen-right = +RAS-X = patient Right → R on screen-right, L on screen-left

        In neurogical convention (neurogical=True):
          Camera is placed on the opposite side, so screen-right = -RAS-X = patient Left
          → L and R labels swap

        Axial    (looking down +Z):
          right axis on screen = ±RAS-X,  up axis = +RAS-Y (Posterior at top)
          left/right = R or L,  top = P,  bottom = A

        Coronal  (looking along +Y, from anterior):
          right axis = ±RAS-X,  up axis = +RAS-Z (Superior at top)
          left/right = R or L,  top = S,  bottom = I

        Sagittal (looking along +X, from patient right):
          right axis = -RAS-Y (pg.right = [0,-1,0], so screen-right = Posterior)
          up axis = +RAS-Z
          left = A (anterior),  right = P (posterior),  top = S,  bottom = I
          The sagittal left/right do NOT flip with neurogical (scrolling axis changes
          but the A/P assignment remains the same because right=−Y is independent of L/R)
        """
        if self._neurogical:
            # Screen-right = patient Left (camera flipped)
            ax_left, ax_right = "L", "R"
            co_left, co_right = "L", "R"
        else:
            ax_left, ax_right = "R", "L"
            co_left, co_right = "R", "L"

        label_sets = [
            # (left,     right,     top,  bottom)
            (ax_left,  ax_right,  "A",  "P"),   # Axial
            (co_left,  co_right,  "S",  "I"),   # Coronal
            ("A",      "P",       "S",  "I"),   # Sagittal
        ]

        for vp, (lft, rgt, top, bot) in zip(self._vps, label_sets):
            vp.set_orientation_labels(lft, rgt, top, bot)

    def _update_crosshairs(self) -> None:
        """Compute the crosshair world point and draw it in all viewports."""
        if not self._geoms:
            return
        pts     = [vp.world_position() for vp in self._vps]
        normals = [g.normal for g in self._geoms]
        A = np.vstack(normals)
        b = np.array([np.dot(normals[i], pts[i]) for i in range(3)])
        try:
            world_pt = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            world_pt = self._geoms[0].centre.copy()
        if self._crosshair_visible:
            for vp in self._vps:
                vp.set_crosshair(world_pt, self._half_len)
        self.ras_changed.emit(float(world_pt[0]), float(world_pt[1]), float(world_pt[2]))

    def _on_slice_changed(self, _plane_idx: int, _slice_idx: int) -> None:
        self._update_crosshairs()


# ── Main window ────────────────────────────────────────────────────────────

TB_STYLE = f"""
QWidget#NiftiToolbar {{
    background:{BG_PANEL};
    border-bottom:1px solid #333340;
}}
QPushButton {{
    background:#2d2d38; color:{TEXT};
    border:1px solid #444455; border-radius:4px;
    padding:4px 12px; font-size:12px;
}}
QPushButton:hover {{ background:#3d3d50; border-color:{ACCENT}; color:{ACCENT}; }}
QPushButton:disabled {{ color:#555566; border-color:#333340; }}
QRadioButton, QCheckBox {{ color:{TEXT}; font-size:12px; }}
QGroupBox {{
    color:{TEXT_DIM}; font-size:11px;
    border:1px solid #444455; border-radius:4px;
    margin-top:6px; padding:2px 8px;
}}
QGroupBox::title {{ subcontrol-origin:margin; left:6px; }}
"""

# Thin separator for the toolbar (replaces QToolBar separator)
def _tb_sep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setFixedWidth(1)
    sep.setStyleSheet("background:#333340;")
    return sep


class NiftiViewerWindow(QWidget):
    """
    Self-contained NIfTI viewer widget.  Derives from QWidget so it can be
    embedded directly in any parent layout of an existing application:

        viewer_widget = NiftiViewerWindow(parent=some_parent)
        some_layout.addWidget(viewer_widget)

    """

    closed = Signal()  # custom signal emitted on close

    def __init__(self, parent=None, stand_alone=False):
        super().__init__(parent)
        self.setStyleSheet(f"NiftiViewerWindow {{ background:{BG_DARK}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar ────────────────────────────────────────────────────
        tb = QWidget(objectName="NiftiToolbar")
        tb.setFixedHeight(44)
        tb.setStyleSheet(TB_STYLE)
        tb_lay = QHBoxLayout(tb)
        tb_lay.setContentsMargins(8, 4, 8, 4)
        tb_lay.setSpacing(6)

        if stand_alone:
            self._btn_open = QPushButton("  Open NIfTI…")
            self._btn_open.clicked.connect(self._open_base)
            tb_lay.addWidget(self._btn_open)

        self._btn_overlay = QPushButton("  Add overlay…")
        self._btn_overlay.setEnabled(False)
        self._btn_overlay.clicked.connect(self._add_overlay)
        tb_lay.addWidget(self._btn_overlay)

        tb_lay.addWidget(_tb_sep())

        mode_box = QGroupBox("Display mode")
        ml = QHBoxLayout(mode_box); ml.setContentsMargins(6,2,6,2); ml.setSpacing(12)
        self._rb_affine  = QRadioButton("Affine (native axes)")
        self._rb_medical = QRadioButton("Medical (RAS)")
        self._rb_affine.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self._rb_affine,  0)
        grp.addButton(self._rb_medical, 1)
        grp.idClicked.connect(self._on_mode)
        ml.addWidget(self._rb_affine); ml.addWidget(self._rb_medical)
        tb_lay.addWidget(mode_box)

        tb_lay.addWidget(_tb_sep())

        conv_box = QGroupBox("Convention")
        cl = QHBoxLayout(conv_box); cl.setContentsMargins(6,2,6,2)
        self._cb_radio = QCheckBox("Neurological (flip L↔R)")
        self._cb_radio.setEnabled(False)
        self._cb_radio.stateChanged.connect(
            lambda s: self.viewer.set_neurological(bool(s)))
        cl.addWidget(self._cb_radio)
        tb_lay.addWidget(conv_box)

        tb_lay.addWidget(_tb_sep())

        view_box = QGroupBox("View")
        vl = QHBoxLayout(view_box); vl.setContentsMargins(6,2,6,2); vl.setSpacing(10)
        self._cb_crosshair = QCheckBox("Crosshairs")
        self._cb_crosshair.setChecked(True)
        self._cb_crosshair.stateChanged.connect(
            lambda s: self.viewer.set_crosshair_visible(bool(s)))
        vl.addWidget(self._cb_crosshair)
        tb_lay.addWidget(view_box)

        tb_lay.addWidget(_tb_sep())

        self._btn_screenshot = QPushButton("  📷 Screenshot…")
        self._btn_screenshot.setEnabled(False)
        self._btn_screenshot.setToolTip("Save all three views side-by-side as a PNG")
        self._btn_screenshot.clicked.connect(self._take_screenshot)
        tb_lay.addWidget(self._btn_screenshot)

        self._btn_reset = QPushButton("  ↺ Reset view")
        self._btn_reset.setEnabled(False)
        self._btn_reset.setToolTip(
            "Restore default zoom, pan, slice positions and window/level for all volumes")
        self._btn_reset.clicked.connect(self._reset_view)
        tb_lay.addWidget(self._btn_reset)

        tb_lay.addWidget(_tb_sep())

        # ── RAS coordinate navigator ───────────────────────────────────
        ras_box = QGroupBox("RAS (mm)")
        ras_box.setToolTip(
            "Current crosshair position in RAS world coordinates.\n"
            "Type a value and press Enter (or click away) to navigate there.")
        rl = QHBoxLayout(ras_box)
        rl.setContentsMargins(6, 1, 6, 1)
        rl.setSpacing(4)

        _coord_edit_style = f"""
            QLineEdit {{
                background:#1a1a22; color:{ACCENT};
                border:1px solid #444455; border-radius:3px;
                padding:0 3px; font-size:11px; font-family:monospace;
                min-width:52px; max-width:64px;
            }}
            QLineEdit:focus {{ border-color:{ACCENT}; }}
        """
        _lbl_style = f"color:{TEXT_DIM}; font-size:10px;"

        self._ras_edits: list[QLineEdit] = []
        for axis in ("X", "Y", "Z"):
            lbl = QLabel(axis)
            lbl.setStyleSheet(_lbl_style)
            rl.addWidget(lbl)
            edit = QLineEdit("0.0")
            edit.setValidator(QDoubleValidator(-9999., 9999., 2))
            edit.setStyleSheet(_coord_edit_style)
            edit.setToolTip(f"RAS {axis} coordinate in mm")
            edit.returnPressed.connect(self._on_ras_entered)
            edit.editingFinished.connect(self._on_ras_entered)
            rl.addWidget(edit)
            self._ras_edits.append(edit)

        tb_lay.addWidget(ras_box)

        tb_lay.addStretch(1)
        root.addWidget(tb)

        # ── Viewer ─────────────────────────────────────────────────────
        self.viewer = NiftiViewer(self)
        self.viewer.wl_updated.connect(self.viewer._layer_panel.update_wl_readout)
        root.addWidget(self.viewer, 1)
        self.viewer.notify_load_base.connect(self.loaded_base)
        # Update the RAS boxes whenever the crosshair moves (slider scroll, etc.)
        self.viewer.ras_changed.connect(self._on_ras_changed)
        # Guard: editingFinished fires twice sometimes; track last-committed value
        self._ras_committing = False

    def loaded_base(self):
        self._btn_overlay.setEnabled(True)
        self._btn_screenshot.setEnabled(True)
        self._btn_reset.setEnabled(True)

    # ── RAS coordinate handlers ────────────────────────────────────────

    def _on_ras_changed(self, x: float, y: float, z: float) -> None:
        """Update the RAS text boxes when the crosshair moves (read-only update)."""
        if self._ras_committing:
            return   # avoid feedback loop while we're the ones navigating
        for edit, val in zip(self._ras_edits, (x, y, z)):
            edit.blockSignals(True)
            edit.setText(f"{val:.2f}")
            edit.blockSignals(False)

    def _on_ras_entered(self) -> None:
        """Navigate to the typed RAS coordinate when Enter is pressed or focus is lost."""
        if self._ras_committing:
            return
        try:
            vals = [float(e.text()) for e in self._ras_edits]
        except ValueError:
            return
        self._ras_committing = True
        self.viewer.navigate_to_ras(*vals)
        # Clear the flag after a zero-delay timer so that any pending
        # editingFinished events in the same Qt event-loop cycle are ignored,
        # but the ras_changed signal (which fires synchronously from
        # navigate_to_ras → _update_crosshairs) is no longer blocked.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: setattr(self, '_ras_committing', False))

    # ── Helpers ────────────────────────────────────────────────────────

    def _on_mode(self, btn_id: int):
        mode = NiftiViewer.MODE_AFFINE if btn_id == 0 else NiftiViewer.MODE_MEDICAL
        self._cb_radio.setEnabled(btn_id == 1)
        self.viewer.set_mode(mode, self._cb_radio.isChecked())

    def _open_base(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Base NIfTI", "",
            "NIfTI Files (*.nii *.nii.gz);;All Files (*)")
        if not path:
            return
        try:
            focal_voxel = (np.array(nib.load(path).get_fdata().shape)/2).astype(int)
            self.viewer.load_base(path,focal_voxel,os.path.basename(path))
            self._btn_overlay.setEnabled(True)
            self._btn_screenshot.setEnabled(True)
        except Exception as exc:
            raise

    def _add_overlay(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Add Overlay NIfTI", "",
            "NIfTI Files (*.nii *.nii.gz);;All Files (*)")
        if not path:
            return
        try:
            self.viewer.add_overlay(path,name=os.path.basename(path))
        except Exception as exc:
            raise

    def _reset_view(self):
        self.viewer.reset_view()

    def _take_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot.png",
            "PNG Images (*.png);;All Files (*)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            self.viewer.grab_screenshot(path)
        except Exception as exc:
            raise

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)  # let the default close logic run


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("NIfTI Viewer")

    win = NiftiViewerWindow(stand_alone=True)
    win.setWindowTitle("NIfTI Viewer — Multi-Volume")
    win.resize(1580, 620)
    win.show()

    # First positional arg = base, rest = overlays
    args = sys.argv[1:]
    if args:
        try:
            focal_voxel = (np.array(nib.load(args[0]).get_fdata().shape)/2).astype(int)
            win.viewer.load_base(args[0],focal_voxel,os.path.basename(args[0]))
            for path in args[1:]:
                win.viewer.add_overlay(path,name=os.path.basename(path))
        except Exception as exc:
            raise

    sys.exit(app.exec())


if __name__ == "__main__":
    main()