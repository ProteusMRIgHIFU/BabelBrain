"""The subset of VTK that BabelBrain uses, imported module by module.

``import vtk`` imports ``vtkmodules.all``, i.e. every one of the ~150 compiled
VTK modules, and PyInstaller then bundles all of them together with their
native dependencies (VTK-m/viskores, IOSS, CGNS, NetCDF, FFmpeg, ...). Use this
module as a drop-in replacement so existing ``vtk.vtkXxx`` code keeps working::

    from Utils import vtk_api as vtk

When a new VTK class is needed, add it here from its ``vtkmodules`` module
(the class documentation names the module, e.g. "Module: vtkFiltersCore").
"""

# Side-effect imports. ``import vtk`` used to register these implementations
# implicitly; without them render windows, mappers, text and interactor
# styles fall back to abstract base classes that render nothing.
import vtkmodules.vtkInteractionStyle  # noqa: F401
import vtkmodules.vtkRenderingFreeType  # noqa: F401
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

from vtkmodules.util import numpy_support  # noqa: F401
from vtkmodules.vtkCommonCore import (
    VTK_FLOAT,
    VTK_UNSIGNED_CHAR,
    vtkFloatArray,
    vtkLookupTable,
    vtkPoints,
)
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import (
    vtkSmoothPolyDataFilter,
    vtkWindowedSincPolyDataFilter,
)
from vtkmodules.vtkFiltersSources import vtkLineSource, vtkSphereSource
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleImage,
    vtkInteractorStyleTrackballCamera,
    vtkInteractorStyleUser,
)
from vtkmodules.vtkInteractionWidgets import (
    vtkOrientationMarkerWidget,
    vtkScalarBarWidget,
)
from vtkmodules.vtkIOGeometry import vtkSTLWriter
from vtkmodules.vtkIOImage import vtkNIFTIImageReader, vtkPNGWriter
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkScalarBarActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkCellPicker,
    vtkImageProperty,
    vtkImageSlice,
    vtkImageSliceMapper,
    vtkInteractorStyle,
    vtkPolyDataMapper,
    vtkPropPicker,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkTextActor,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkRenderingImage import vtkImageResliceMapper
