import sys
import nibabel as nib
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QRadioButton,QCheckBox,
    QToolBar, QFileDialog, QSlider, QLabel, QComboBox, QHBoxLayout,QPushButton,QSizePolicy
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
import numpy as np
from vtk.util import numpy_support

class GiftiViewer(QWidget):
    def __init__(self, gifti_files,selectedFunc=0,shared_camera=None, parent=None, callbackSync=None):
        super().__init__(parent)

        # --- Qt Layout ---
        layout = QVBoxLayout()
        self.setLayout(layout)

        # VTK Widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)

        self.titleLabel = QComboBox(self)
        self.titleLabel.addItems([g[4] for g in gifti_files])
        self.titleLabel.setCurrentIndex(selectedFunc)
        self.titleLabel.currentIndexChanged.connect(self.select_function)

        # Apply stylesheet
        self.titleLabel.setStyleSheet("""
            QComboBox {
                font-size: 14px;       /* Change font size */
                color: green;            /* Change text color */
            }
        """)
        layout.addWidget(self.titleLabel,alignment=Qt.AlignCenter)
        layout.addWidget(self.vtkWidget)

        self.valueLabel = QLabel("Value: N/A")
        layout.addWidget(self.valueLabel)

        self.renderer = vtk.vtkRenderer()

        self.selectedFunc=selectedFunc

        self.Entries=[]

        self.func_data = []
        self.faces = []

        self.currentHeatmapVisibility = True

        for g in gifti_files:
            entry={}

        # --- Load GIFTI File ---
            gii = nib.load(g[0])
            coords = gii.darrays[0].data  # vertex coordinates (Nx3)
            faces = gii.darrays[1].data   # triangles (Mx3)

            entry['coords'] = coords
            coordsOrig = coords.copy()
            entry['coordsOrig'] = coordsOrig
            entry['faces'] = faces
            entry['title'] = g[4] if len(g) > 4 else ""


            func = nib.load(g[1])
            func_data = func.darrays[0].data

            entry['func_data'] = func_data

            thresh = nib.load(g[2])
            thresh_data = thresh.darrays[0].data
            coords_outside_mask = coords.copy()
            coords_outside_mask[thresh_data==1, :] = np.nan  # remove vertices below threshold
            coords[thresh_data==0, :] = np.nan  # remove vertices below threshold

            scalars = func_data
            # --- Convert to VTK PolyData ---
            
            for n,c in enumerate([coords, coords_outside_mask,coordsOrig]):
                points = vtk.vtkPoints()
                for x, y, z in c:
                    points.InsertNextPoint(x, y, z)

                polys = vtk.vtkCellArray()
                for tri in faces:
                    polys.InsertNextCell(3)
                    polys.InsertCellPoint(int(tri[0]))
                    polys.InsertCellPoint(int(tri[1]))
                    polys.InsertCellPoint(int(tri[2]))

                polydata = vtk.vtkPolyData()
                polydata.SetPoints(points)
                polydata.SetPolys(polys)

                # --- Mapper + Actor ---
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
            
                if  n in [0,2]:
                    vtk_scalars = vtk.vtkFloatArray()
                    vtk_scalars.SetName("Heatmap")
                    for val in scalars:
                        vtk_scalars.InsertNextValue(float(val))

                    polydata.GetPointData().SetScalars(vtk_scalars)
                    if g[3] is None:
                        mapper.SetScalarRange(vtk_scalars.GetRange())
                    else:
                        mapper.SetScalarRange(g[3])
                    self.renderer.AddActor(actor)
                    if n==0:
                        entry['mapperMasked'] = mapper
                        entry['actorHeatMapMasked'] = actor
                    else:
                        entry['mapperUnmasked'] = mapper
                        entry['actorHeatMapUnmasked'] = actor
                else:
                    # If no scalars or the rest of the scalp, just give the actor a solid color
                    actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # light gray
                    self.renderer.AddActor(actor)
                    entry['actorSkinMasked'] = actor

                actor.SetVisibility(False)

            self.Entries.append(entry)


        # --- Renderer ---
        self.select_function(self.selectedFunc)
        self.renderer.SetBackground(0.1, 0.1, 0.1)

        if shared_camera:
            self.renderer.SetActiveCamera(shared_camera)

        # --- Render Window ---
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()

        # --- Better Interaction (like 3D Slicer) ---
        style = vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        def zoom_callback(obj, event):
            camera = self.renderer.GetActiveCamera()
            if event == "MouseWheelForwardEvent":
                camera.Dolly(1.05)  # zoom in
            elif event == "MouseWheelBackwardEvent":
                camera.Dolly(0.95)  # zoom out
            self.renderer.ResetCameraClippingRange()
            if callbackSync:
                callbackSync(self, event)
            else:
                self.vtkWidget.GetRenderWindow().Render()

        self.interactor.AddObserver("MouseWheelForwardEvent", zoom_callback)
        self.interactor.AddObserver("MouseWheelBackwardEvent", zoom_callback)

        def keypress_callback(obj, event):
            key = obj.GetKeySym()
            camera = self.renderer.GetActiveCamera()
            if key == "plus" or key == "equal":  # "+" key
                camera.Dolly(1.1)
            elif key == "minus":
                camera.Dolly(0.9)
            self.renderer.ResetCameraClippingRange()
            if callbackSync:
                callbackSync(self, event)
            else:
                self.vtkWidget.GetRenderWindow().Render()

        self.interactor.AddObserver("KeyPressEvent", keypress_callback)

        self.interactor.Initialize()
        self.interactor.Start()

        # --- Picker ---
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.001)

        # Selection marker
        self.sphere_source = vtk.vtkSphereSource()
        self.sphere_source.SetRadius(2.0)
        self.sphere_mapper = vtk.vtkPolyDataMapper()
        self.sphere_mapper.SetInputConnection(self.sphere_source.GetOutputPort())
        self.sphere_actor = vtk.vtkActor()
        self.sphere_actor.SetMapper(self.sphere_mapper)
        self.sphere_actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # red sphere
        self.renderer.AddActor(self.sphere_actor)
        self.sphere_actor.SetVisibility(False)

        # Selection mode toggle
        self.selection_mode = False

        # Listen for clicks
        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_click)

        # Callback for broadcasting selection
        self.selection_callback = None

        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(50, 50, 50)   # size of arrows
        axes.AxisLabelsOff()               # show X/Y/Z labels
        axes.SetCylinderRadius(0.1)
        axes.SetShaftTypeToCylinder()

        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(axes)
        self.orientation_widget.SetInteractor(self.interactor)
        self.orientation_widget.SetViewport(0.0, 0.0, 0.2, 0.2)  # bottom-left corner
        self.orientation_widget.SetEnabled(1)
        self.orientation_widget.InteractiveOff()

        self.set_colormap('jet')

        # --- Scalar bar ---
        scalar_bar = vtk.vtkScalarBarActor()
        self.renderer.AddActor(scalar_bar)
        scalar_bar.SetLookupTable(self.current_ActorsEntry['mapperMasked'].GetLookupTable())
        scalar_bar.SetNumberOfLabels(5)

        # Place on right side of the render window
        scalar_bar_widget = vtk.vtkScalarBarWidget()
        scalar_bar_widget.SetInteractor(self.interactor)
        scalar_bar_widget.SetScalarBarActor(scalar_bar)
        scalar_bar_widget.On()   # ena

    def select_function(self,selection):
        #we first hide the current function's actors
        self.current_ActorsEntry['actorHeatMapMasked'].SetVisibility(False)
        self.current_ActorsEntry['actorHeatMapUnmasked'].SetVisibility(False)
        self.current_ActorsEntry['actorSkinMasked'].SetVisibility(False)
        self.selectedFunc=selection
        self.set_heatmap_visibility(self.currentHeatmapVisibility) #this will honor the current selection


    @property
    def current_ActorsEntry(self):
        return self.Entries[self.selectedFunc]

    def on_left_click(self, obj, event):
        if not self.selection_mode:
            return  # normal camera mode

        x, y = self.interactor.GetEventPosition()
        if self.picker.Pick(x, y, 0, self.renderer):
            cell_id = self.picker.GetCellId()
            if cell_id >= 0 and cell_id < len(self.current_ActorsEntry['faces']):
                # Get triangle vertices
                tri = self.current_ActorsEntry['faces'][cell_id]
                vtx_coords = self.current_ActorsEntry['coordsOrig'][tri]
                if np.any(np.isnan(vtx_coords)):
                    return  # skip if any vertex is NaN
                
                # Place sphere at picked point
                pick_pos = self.picker.GetPickPosition()
                if self.selection_callback:
                    # Notify main window about selection
                    self.selection_callback(tri[0], cell_id,pick_pos)

        return
    
    def highlight_triangle(self, cell_id,pick_pos):
        """Highlight a triangle by ID and place sphere at pick position."""

        tri = self.current_ActorsEntry['faces'][cell_id]
        values = self.current_ActorsEntry['func_data'][tri]
        value = np.mean(values)
        if np.isnan(value):
            return  # skip if value is NaN
        self.valueLabel.setText(f"Value: {value:.2f}")

        self.sphere_source.SetCenter(*pick_pos)
        self.sphere_actor.SetVisibility(True)
        self.vtkWidget.GetRenderWindow().Render()

    def set_selection_mode(self, enabled: bool):
        self.selection_mode = enabled

    def reset_camera(self):
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def save_screenshot(self):
        print('save screenshot')
        # filename, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "", "PNG Files (*.png)")
        # if filename:
        #     w2i = vtk.vtkWindowToImageFilter()
        #     w2i.SetInput(self.vtkWidget.GetRenderWindow())
        #     w2i.Update()

        #     writer = vtk.vtkPNGWriter()
        #     writer.SetFileName(filename)
        #     writer.SetInputConnection(w2i.GetOutputPort())
        #     writer.Write()

    def set_colormap(self, name):
        
        from matplotlib.pyplot import get_cmap

        cmap = get_cmap(name)  # get all colors
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        for i in range(256):
            val = cmap(i)
            lut.SetTableValue(i, val[0], val[1], val[2], 1.0)

        self.current_ActorsEntry['mapperMasked'].SetLookupTable(lut)
        self.current_ActorsEntry['mapperUnmasked'].SetLookupTable(lut)

        self.vtkWidget.GetRenderWindow().Render()

    def set_heatmap_visibility(self, visible):
        self.currentHeatmapVisibility = visible
        self.current_ActorsEntry['actorHeatMapMasked'].SetVisibility(visible)
        self.current_ActorsEntry['actorSkinMasked'].SetVisibility(visible)
        self.current_ActorsEntry['actorHeatMapUnmasked'].SetVisibility(not visible)
        self.vtkWidget.GetRenderWindow().Render()

class MultiGiftiViewerWidget(QWidget):
    def __init__(self, gifti_files, MaxViews=4, parent=None, callBackAfterGenTrajectory=None):
        super().__init__(parent)
        self.viewers = []
        self.MaxViews = MaxViews
        self.callBackAfterGenTrajectory = callBackAfterGenTrajectory

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # Toolbar
        self.toolbar = QToolBar("Controls", self)
        layout.addWidget(self.toolbar)

        # Horizontal layout for the viewers
        viewers_layout = QHBoxLayout()
        layout.addLayout(viewers_layout)

        # Shared camera
        shared_camera = vtk.vtkCamera()

        # --- Synchronize Rendering ---
        def sync_cameras(caller=None, event=None):
            for v in self.viewers:
                v.vtkWidget.GetRenderWindow().Render()

        # Create viewers
        for n in range(self.MaxViews):
            v = GiftiViewer(gifti_files,
                            shared_camera=shared_camera,
                            callbackSync=sync_cameras,
                            parent=self,
                            selectedFunc=n)
            viewers_layout.addWidget(v)
            self.viewers.append(v)

        # Attach observer to all interactors
        for v in self.viewers:
            v.interactor.AddObserver("InteractionEvent", sync_cameras)

        # Toolbar buttons
        self.selection_button = QCheckBox("Selection Mode")
        self.selection_button.toggled.connect(self.toggle_selection)
        self.toolbar.addWidget(self.selection_button)

        self.heatmap_checkbox = QCheckBox("Show masked")
        self.heatmap_checkbox.setChecked(True)  # default ON
        self.heatmap_checkbox.toggled.connect(self.toggle_heatmap)
        self.toolbar.addWidget(self.heatmap_checkbox)

        axial_action = QAction("Top", self)
        axial_action.triggered.connect(lambda: self.set_preset_view("top"))
        self.toolbar.addAction(axial_action)

        coronal_action = QAction("Front", self)
        coronal_action.triggered.connect(lambda: self.set_preset_view("front"))
        self.toolbar.addAction(coronal_action)

        sagittal_action = QAction("Lateral", self)
        sagittal_action.triggered.connect(lambda: self.set_preset_view("lateral"))
        self.toolbar.addAction(sagittal_action)

        oblique_action = QAction("3D", self)
        oblique_action.triggered.connect(lambda: self.set_preset_view("oblique"))
        self.toolbar.addAction(oblique_action)

        screenshot_action = QAction("Screenshot", self)
        for v in self.viewers:
            screenshot_action.triggered.connect(v.save_screenshot)
        self.toolbar.addAction(screenshot_action)

        # Hook up selection synchronization
        for v in self.viewers:
            v.selection_callback = self.broadcast_selection

        # Initial camera reset and preset
        if self.viewers:
            self.viewers[0].renderer.ResetCamera()
            for v in self.viewers:
                v.vtkWidget.GetRenderWindow().Render()

        self.set_preset_view("oblique")

        self.select_vortex = None

        button = QPushButton("Generate Trajectory", self)
        button.clicked.connect(self.GenerateTrajectory)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # only as wide as needed
        button.setStyleSheet("padding: 8px 36px;")
        button.setStyleSheet("""
            QPushButton {
                font-size: 16px;       /* Change font size */
                color: gray;     /* Make text bold */
            }
        """)
        button.setEnabled(False)  
        self.generateTrajectoryPushButton = button
        layout.addWidget(button, alignment=Qt.AlignHCenter) 

    def GenerateTrajectory(self):
        print("Generating trajectory... for vortex id",self.select_vortex)
        if self.callBackAfterGenTrajectory:
            self.callBackAfterGenTrajectory(self.select_vortex )

    # --------------------------
    # Methods from MainWindow
    # --------------------------
    def toggle_selection(self, checked):
        for v in self.viewers:
            v.set_selection_mode(checked)

    def toggle_heatmap(self, checked):
        for v in self.viewers:
            v.set_heatmap_visibility(checked)

    def broadcast_selection(self, vertex, cell_id,pick_pos):
        """Called when one viewer selects a triangle."""
        self.select_vortex = vertex
        for v in self.viewers:
            v.highlight_triangle(cell_id, pick_pos)
        #once a valid triangle is selected, enable the button
        self.generateTrajectoryPushButton.setEnabled(True)
        self.generateTrajectoryPushButton.setStyleSheet("""
            QPushButton {
                font-size: 16px;       /* Change font size */
                color: blue;            /* Change text color */
                font-weight: bold;     /* Make text bold */
            }
        """)

    def set_preset_view(self, preset):
        if not self.viewers:
            return

        camera = self.viewers[0].renderer.GetActiveCamera()
        self.viewers[0].renderer.ResetCamera()

        bounds = self.viewers[0].current_ActorsEntry['actorHeatMapMasked'].GetBounds()
        center = [(bounds[0] + bounds[1]) / 2,
                  (bounds[2] + bounds[3]) / 2,
                  (bounds[4] + bounds[5]) / 2]

        camera.SetFocalPoint(center)

        if preset == "top":
            camera.SetPosition(center[0], center[1], center[2] +800)
            camera.SetViewUp(0, 1, 0)
        elif preset == "front":
            camera.SetPosition(center[0], center[1] + 800, center[2])
            camera.SetViewUp(0, 0, 1)
        elif preset == "lateral":
            camera.SetPosition(center[0] + 800, center[1], center[2])
            camera.SetViewUp(0, 0, 1)
        elif preset == "oblique":
            camera.SetPosition(center[0] + 400,
                               center[1] + 400,
                               center[2] + 400)
            camera.SetViewUp(0, 0, 1)

        # Update all viewers
        for v in self.viewers:
            v.vtkWidget.GetRenderWindow().Render()


class FinalResultViewer(QWidget):
    def __init__(self, gifti_files, parent=None, callbackSync=None):
        super().__init__(parent)

        # --- Qt Layout ---
        layout = QVBoxLayout()
        self.setLayout(layout)

        # VTK Widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)

        layout.addWidget(self.vtkWidget)


        self.renderer = vtk.vtkRenderer()

        self.Entries=[]

        self.func_data = []
        self.faces = []

        self.currentHeatmapVisibility = True

        # --- Load GIFTI File ---
        gii = nib.load(gifti_files[0])
        coordsHead = gii.darrays[0].data  # vertex coordinates (Nx3)
        facesHead = gii.darrays[1].data   # triangles (Mx3)

        gii = nib.load(gifti_files[1])
        coordsTx = gii.darrays[0].data  # vertex coordinates (Nx3)
        facesTx = gii.darrays[1].data   # triangles (Mx3)

        objects=[[coordsHead,facesHead],
                    [coordsTx,facesTx]]

        # --- Convert to VTK PolyData ---

        for n,e in enumerate(objects):
            coords, faces = e
            points = vtk.vtkPoints()
            for x, y, z in coords:
                points.InsertNextPoint(x, y, z)

            polys = vtk.vtkCellArray()
            for tri in faces:
                polys.InsertNextCell(3)
                polys.InsertCellPoint(int(tri[0]))
                polys.InsertCellPoint(int(tri[1]))
                polys.InsertCellPoint(int(tri[2]))

            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(polys)

            # --- Mapper + Actor ---
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
        
        
            # If no scalars or the rest of the scalp, just give the actor a solid color
            if n==0:
                actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # light gray
                self.ActorHead = actor
            else:
                actor.GetProperty().SetColor(0.4, 0.4, 1.0)  # blue
                self.ActorTx = actor
            self.renderer.AddActor(actor)

            actor.SetVisibility(True)

        # --- Renderer ---
        self.renderer.SetBackground(0.1, 0.1, 0.1)

     
        # --- Render Window ---
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()

        # --- Better Interaction (like 3D Slicer) ---
        style = vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        def zoom_callback(obj, event):
            camera = self.renderer.GetActiveCamera()
            if event == "MouseWheelForwardEvent":
                camera.Dolly(1.05)  # zoom in
            elif event == "MouseWheelBackwardEvent":
                camera.Dolly(0.95)  # zoom out
            self.renderer.ResetCameraClippingRange()
            if callbackSync:
                callbackSync(self, event)
            else:
                self.vtkWidget.GetRenderWindow().Render()

        self.interactor.AddObserver("MouseWheelForwardEvent", zoom_callback)
        self.interactor.AddObserver("MouseWheelBackwardEvent", zoom_callback)

        def keypress_callback(obj, event):
            key = obj.GetKeySym()
            camera = self.renderer.GetActiveCamera()
            if key == "plus" or key == "equal":  # "+" key
                camera.Dolly(1.1)
            elif key == "minus":
                camera.Dolly(0.9)
            self.renderer.ResetCameraClippingRange()
            if callbackSync:
                callbackSync(self, event)
            else:
                self.vtkWidget.GetRenderWindow().Render()

        self.interactor.AddObserver("KeyPressEvent", keypress_callback)

        self.interactor.Initialize()
        self.interactor.Start()
        self.reset_camera()




    def reset_camera(self):
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

        camera = self.renderer.GetActiveCamera()

        bounds = self.ActorHead.GetBounds()
        center = [(bounds[0] + bounds[1]) / 2,
                  (bounds[2] + bounds[3]) / 2,
                  (bounds[4] + bounds[5]) / 2]


        camera.SetPosition(center[0] + 400,
                            center[1] + 400,
                            center[2] + 400)
        camera.SetViewUp(0, 0, 1)


class OrthoSliceViewer(QWidget):
    """
    Three linked orthogonal views (Axial/Coronal/Sagittal) for a NIfTI volume.
    - Click in any view to move the crosshair: only the in-plane indices change.
    - Crosshairs update in all views.
    - Uses vtkNIFTIImageReader to preserve orientation (origin/spacing/direction).
    """
    def __init__(self, nifti_path: str, size, parent=None):
        super().__init__(parent)
        

        # ----- Read NIfTI (respects qform/sform) -----
        self.reader = vtk.vtkNIFTIImageReader()
        self.reader.SetFileName(nifti_path)
        self.reader.Update()
        self.image = self.reader.GetOutput()

        # Convenience: extent/origin/spacing/direction
        self.extent = self.image.GetExtent()        # (iMin,iMax, jMin,jMax, kMin,kMax)
        self.origin = np.array(self.image.GetOrigin(), dtype=float)
        self.spacing = np.array(self.image.GetSpacing(), dtype=float)

        # direction is a 3x3
        dir_mat = self.image.GetDirectionMatrix()
        self.direction = np.array([[dir_mat.GetElement(r, c) for c in range(3)] for r in range(3)], dtype=float)
        self.dir_inv = np.linalg.inv(self.direction)

        # Crosshair starts at volume center (indices)
        self.crosshair = [
            (self.extent[0] + self.extent[1]) // 2,  # i
            (self.extent[2] + self.extent[3]) // 2,  # j
            (self.extent[4] + self.extent[5]) // 2,  # k
        ]

        # ----- Build UI layout -----
        layout = QHBoxLayout(self)
        self.setLayout(layout)

        # Keep per-view data
        self.views = []  # list of dicts: {name, widget, renderer, mapper, actor, lines, orientation}

        # Create the three orthogonal views
        self._add_view(layout, name="Axial",    orientation="Z")  # k fixed, (i,j) vary
        self._add_view(layout, name="Coronal",  orientation="Y")  # j fixed, (i,k) vary
        self._add_view(layout, name="Sagittal", orientation="X")  # i fixed, (j,k) vary

        # Initial draw
        self._update_all()

    # -----------------------------
    # Helpers: IJK <-> World
    # -----------------------------
    def ijk_to_world(self, i, j, k):
        """Convert voxel indices (i,j,k) to world coordinates using origin/spacing/direction."""
        ijk_mm = np.array([i * self.spacing[0], j * self.spacing[1], k * self.spacing[2]], dtype=float)
        return (self.direction @ ijk_mm) + self.origin

    def world_to_ijk(self, x, y, z):
        """Convert world to voxel indices (floating)."""
        xyz = np.array([x, y, z], dtype=float)
        ijk_mm = self.dir_inv @ (xyz - self.origin)
        ijk = ijk_mm / self.spacing
        return ijk  # float indices

    # -----------------------------
    # View construction
    # -----------------------------
    def _add_view(self, parent_layout, name, orientation):
        # Mapper oriented to X/Y/Z
        mapper = vtk.vtkImageSliceMapper()
        mapper.SetInputConnection(self.reader.GetOutputPort())
        if orientation == "Z":
            mapper.SetOrientationToZ()
            slice_index = self.crosshair[2]
        elif orientation == "Y":
            mapper.SetOrientationToY()
            slice_index = self.crosshair[1]
        elif orientation == "X":
            mapper.SetOrientationToX()
            slice_index = self.crosshair[0]
        else:
            raise ValueError("orientation must be 'X','Y', or 'Z'.")

        mapper.SetSliceNumber(int(slice_index))

        # Image actor + contrast defaults
        actor = vtk.vtkImageSlice()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        # Window/level from data range for a reasonable default
        lo, hi = self.image.GetScalarRange()
        prop.SetColorWindow(max(hi - lo, 1e-3))
        prop.SetColorLevel((hi + lo) * 0.5)

        # Renderer
        ren = vtk.vtkRenderer()
        ren.AddViewProp(actor)
        ren.SetBackground(0, 0, 0)

        # Crosshair (two lines)
        line_color = (1.0, 0.0, 0.0)
        lines = []
        for _ in range(2):
            line = vtk.vtkLineSource()
            mapper_line = vtk.vtkPolyDataMapper()
            mapper_line.SetInputConnection(line.GetOutputPort())
            actor_line = vtk.vtkActor()
            actor_line.SetMapper(mapper_line)
            actor_line.GetProperty().SetColor(*line_color)
            actor_line.GetProperty().SetLineWidth(1.5)
            ren.AddActor(actor_line)
            lines.append((line, actor_line))

        # Widget + interactor
        w = QVTKRenderWindowInteractor()
        w.GetRenderWindow().AddRenderer(ren)
        iren = w.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleImage()
        iren.SetInteractorStyle(style)

        # Picker (pick only this slice actor to get correct world coords)
        picker = vtk.vtkPropPicker()
        picker.PickFromListOn()
        picker.AddPickList(actor)

        # Click handler: update in-plane indices only
        def on_left_click(obj, evt, view_name=name, orient=orientation, pick=picker, renderer=ren):
            x, y = iren.GetEventPosition()
            if pick.Pick(x, y, 0, renderer):
                wx, wy, wz = pick.GetPickPosition()
                fi, fj, fk = self.world_to_ijk(wx, wy, wz)
                # Clamp to extent & round only the in-plane axes
                i, j, k = self.crosshair
                if orient == "Z":      # axial: (i,j) from click, keep k
                    i = int(np.clip(round(fi), self.extent[0], self.extent[1]))
                    j = int(np.clip(round(fj), self.extent[2], self.extent[3]))
                elif orient == "Y":    # coronal: (i,k) from click, keep j
                    i = int(np.clip(round(fi), self.extent[0], self.extent[1]))
                    k = int(np.clip(round(fk), self.extent[4], self.extent[5]))
                elif orient == "X":    # sagittal: (j,k) from click, keep i
                    j = int(np.clip(round(fj), self.extent[2], self.extent[3]))
                    k = int(np.clip(round(fk), self.extent[4], self.extent[5]))
                self.crosshair = [i, j, k]
                self._update_all()
            # obj.OnLeftButtonDown()  # preserve default pan/zoom behavior

        iren.AddObserver("LeftButtonPressEvent", on_left_click)
        iren.Initialize()

        if orientation == "Z":
            # View from superior (looking down -Z)
            ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
            ren.GetActiveCamera().SetPosition(0, 0, 1)
            ren.GetActiveCamera().SetViewUp(0, 1, 0)

        elif orientation == "Y":
            # View from front (looking along -Y)
            ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
            ren.GetActiveCamera().SetPosition(0, -1, 0)
            ren.GetActiveCamera().SetViewUp(0, 0, 1)

        elif orientation == "X":
            # View from left (looking along -X)
            ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
            ren.GetActiveCamera().SetPosition(-1, 0, 0)
            ren.GetActiveCamera().SetViewUp(0, 0, 1)

        ren.ResetCamera()

        mainQ = QWidget()
        tlayout = QVBoxLayout()
        mainQ.setLayout(tlayout)
        secondQ = QWidget()
        slayout = QHBoxLayout()
        secondQ.setLayout(slayout)

        if orientation == "Z":
            al=QLabel("A")
            al.setStyleSheet("""
            QLabel {
                font-size: 18px;       /* Change font size */
                color: green;            /* Change text color */
                font-weight: bold;       /* Change font weight */
            }
            """)
            tlayout.addWidget(al,alignment=Qt.AlignCenter)
            ll=QLabel("L")
            ll.setStyleSheet("""
            QLabel {
                font-size: 18px;       /* Change font size */
                color: blue;            /* Change text color */
                font-weight: bold;       /* Change font weight */
            }
            """)
            slayout.addWidget(ll)
        elif orientation == "Y":
            sl=QLabel("S")
            sl.setStyleSheet("""
            QLabel {
                font-size: 18px;       /* Change font size */
                color: red;            /* Change text color */
                font-weight: bold;       /* Change font weight */
            }
            """)
            tlayout.addWidget(sl,alignment=Qt.AlignCenter)
            ll=QLabel("L")
            ll.setStyleSheet("""
            QLabel {
                font-size: 18px;       /* Change font size */
                color: blue;            /* Change text color */
                font-weight: bold;       /* Change font weight */
            }
            """)
            slayout.addWidget(ll)
        else:
            sl=QLabel("S")
            sl.setStyleSheet("""
            QLabel {
                font-size: 18px;       /* Change font size */
                color: red;            /* Change text color */
                font-weight: bold;       /* Change font weight */
            }
            """)
            tlayout.addWidget(sl,alignment=Qt.AlignCenter)
            al=QLabel("A")
            al.setStyleSheet("""
            QLabel {
                font-size: 18px;       /* Change font size */
                color: green;            /* Change text color */
                font-weight: bold;       /* Change font weight */
            }
            """)
            slayout.addWidget(al)

        tlayout.addWidget(w)
        slayout.addWidget(mainQ)
        parent_layout.addWidget(secondQ)
        self.views.append(dict(
            name=name, orientation=orientation,
            widget=w, renderer=ren, mapper=mapper, actor=actor, lines=lines
        ))

        # w.GetRenderWindow().Render()

        # def add_orientation_labels(vtk_widget,orientation):
        # # Helper to create a text actor
        #     def make_label(text, x, y):
        #         actor = vtk.vtkTextActor()
        #         actor.SetInput(text)
        #         actor.GetTextProperty().SetFontSize(24)
        #         actor.GetTextProperty().SetColor(1, 1, 1)  # white text
        #         actor.GetTextProperty().SetBold(True)
        #         actor.SetDisplayPosition(x, y)
        #         return actor

        #     # Get render window size
        #     iren.Initialize()
        #     size = vtk_widget.GetRenderWindow().GetSize()
        #     dpr = vtk_widget.devicePixelRatioF()
        #     w = int(vtk_widget.width()*dpr)
        #     h = int(vtk_widget.height()*dpr)
        #     # w = int(size[0] )
        #     # h = int(size[1] )


        #     if orientation == "Z":
        #         # Axial: L/R (X), P/A (Y)
        #         ren.AddActor2D(make_label("L", int(w/100*5), h//2))
        #         ren.AddActor2D(make_label("R", w-int(w/10)*4, h//2))
        #         ren.AddActor2D(make_label("P", w//2, 10))
        #         ren.AddActor2D(make_label("A", w//2, h-10))

        #     elif orientation == "Y":
        #         # Coronal: L/R (X), S/I (Z)
        #         ren.AddActor2D(make_label("L", 10, h//2))
        #         ren.AddActor2D(make_label("R", w-100, h//2))
        #         ren.AddActor2D(make_label("S", w//2, h-10))
        #         ren.AddActor2D(make_label("I", w//2, 10))

        #     elif orientation == "X":
        #         # Sagittal: P/A (Y), S/I (Z)
        #         ren.AddActor2D(make_label("P", 10, h//2))
        #         ren.AddActor2D(make_label("A", w-10, h//2))
        #         ren.AddActor2D(make_label("S", w//2, h-10))
        #         ren.AddActor2D(make_label("I", w//2, 10))

        #     vtk_widget.GetRenderWindow().Render()
        # add_orientation_labels(w, orientation)


    # -----------------------------
    # Update all view slices + crosshairs
    # -----------------------------
    def _update_all(self):
        i, j, k = self.crosshair
        ei0, ei1, ej0, ej1, ek0, ek1 = self.extent

        for view in self.views:
            orient = view["orientation"]
            mapper = view["mapper"]
            lines = view["lines"]

            # Set the slice number appropriate for this view
            if orient == "Z":
                mapper.SetSliceNumber(int(k))
            elif orient == "Y":
                mapper.SetSliceNumber(int(j))
            elif orient == "X":
                mapper.SetSliceNumber(int(i))

            # Update crosshair lines (world coords via ijk_to_world)
            # Each view plane has two in-plane axes; draw lines across full extent.

            if orient == "Z":
                # Plane: k fixed; in-plane axes: i, j
                p1 = self.ijk_to_world(ei0, j,  k)
                p2 = self.ijk_to_world(ei1, j,  k)
                p3 = self.ijk_to_world(i,  ej0, k)
                p4 = self.ijk_to_world(i,  ej1, k)

            elif orient == "Y":
                # Plane: j fixed; in-plane axes: i, k
                p1 = self.ijk_to_world(ei0, j,  k)
                p2 = self.ijk_to_world(ei1, j,  k)
                p3 = self.ijk_to_world(i,  j,  ek0)
                p4 = self.ijk_to_world(i,  j,  ek1)

            elif orient == "X":
                # Plane: i fixed; in-plane axes: j, k
                p1 = self.ijk_to_world(i,  ej0, k)
                p2 = self.ijk_to_world(i,  ej1, k)
                p3 = self.ijk_to_world(i,  j,  ek0)
                p4 = self.ijk_to_world(i,  j,  ek1)

            # Apply to the two lines
            lineA, _ = lines[0]
            lineB, _ = lines[1]
            lineA.SetPoint1(*p1); lineA.SetPoint2(*p2)
            lineB.SetPoint1(*p3); lineB.SetPoint2(*p4)

            # Render this view
            view["widget"].GetRenderWindow().Render()


if __name__ == "__main__":

    
    app = QApplication(sys.argv)

    # Replace with path to your GIFTI file
    # gifti_files = []
    # gifti_files.append(('/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/skin.surf.gii',
    #                     '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/distances_skin.func.gii',
    #                     '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/distances_skin_thresholded.func.gii',
    #                     [0,100],
    #                     'Distance'))
    # gifti_files.append(('/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/skin.surf.gii',
    #                     '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/target_intersection_skin.func.gii',
    #                     '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/distances_skin_thresholded.func.gii',
    #                     [0,100],
    #                     'Target Intersection'))
    # gifti_files.append(('/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/skin.surf.gii',
    #                     '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/angles_skin.func.gii',
    #                     '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/distances_skin_thresholded.func.gii',
    #                     [0,20],
    #                     'Angle'))
    # gifti_files.append(('/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/skin.surf.gii',
    #                     '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/skin_skull_angles_skin.func.gii',
    #                     '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/distances_skin_thresholded.func.gii',
    #                     [0,20],
    #                     'Skin-Skull Angle'))

    # widget = MultiGiftiViewerWidget(gifti_files,MaxViews=4)
    # widget.resize(1600, 600)
    # widget.show()

    # sliceviewer=OrthoSliceViewer('/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/T1.nii.gz',[1600,600])
    # sliceviewer.resize(1600, 600)
    # sliceviewer.show()
    gifti_files = ['/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/skin.surf.gii',
                   '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/Q/transducer_Q_PlanTUSMask_Q.surf.gii']
    VW=FinalResultViewer(gifti_files)
    VW.resize(600, 600)
    VW.show()
    sys.exit(app.exec())
