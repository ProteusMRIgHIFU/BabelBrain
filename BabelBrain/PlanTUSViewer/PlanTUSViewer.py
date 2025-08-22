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

class GiftiViewer(QWidget):
    def __init__(self, gifti_files,selectedFunc=0,shared_camera=None, parent=None, callbackSync=None):
        super().__init__(parent)

        # --- Qt Layout ---
        layout = QVBoxLayout()
        self.setLayout(layout)

        # VTK Widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)

        self.titleLabel = QLabel("")

        # Apply stylesheet
        self.titleLabel.setStyleSheet("""
            QLabel {
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
        self.titleLabel.setText(self.current_ActorsEntry['title'])
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
                    self.selection_callback(cell_id, pick_pos)

        return
    
    def highlight_triangle(self, cell_id, pick_pos=None):
        """Highlight a triangle by ID and place sphere at pick position."""
        if cell_id < 0 or cell_id >= len(self.current_ActorsEntry['faces']):
            return

        tri = self.current_ActorsEntry['faces'][cell_id]
        vtx_coords = self.current_ActorsEntry['coordsOrig'][tri]
        if np.any(np.isnan(vtx_coords)):
            return  # skip if any vertex is NaN
        values=self.current_ActorsEntry['func_data'][tri]
        self.valueLabel.setText(f"Value: {np.mean(values):.2f}")

        if pick_pos is None:
            # Use centroid if no explicit pick position
            pick_pos = vtx_coords.mean(axis=0)

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
    def __init__(self, gifti_files, MaxViews=4, parent=None):
        super().__init__(parent)
        self.viewers = []
        self.MaxViews = MaxViews

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

        axial_action = QAction("Axial", self)
        axial_action.triggered.connect(lambda: self.set_preset_view("axial"))
        self.toolbar.addAction(axial_action)

        coronal_action = QAction("Coronal", self)
        coronal_action.triggered.connect(lambda: self.set_preset_view("coronal"))
        self.toolbar.addAction(coronal_action)

        sagittal_action = QAction("Sagittal", self)
        sagittal_action.triggered.connect(lambda: self.set_preset_view("sagittal"))
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

        self.select_cell_id = None

        button = QPushButton("Generate Trajectory", self)
        button.clicked.connect(self.GenerateTrajectory)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # only as wide as needed
        button.setStyleSheet("padding: 8px 36px;")
        button.setStyleSheet("""
            QPushButton {
                font-size: 16px;       /* Change font size */
                color: blue;            /* Change text color */
                font-weight: bold;     /* Make text bold */
            }
        """)
        layout.addWidget(button, alignment=Qt.AlignHCenter) 

    def GenerateTrajectory(self):
        print("Generating trajectory...")

    # --------------------------
    # Methods from MainWindow
    # --------------------------
    def toggle_selection(self, checked):
        for v in self.viewers:
            v.set_selection_mode(checked)

    def toggle_heatmap(self, checked):
        for v in self.viewers:
            v.set_heatmap_visibility(checked)

    def broadcast_selection(self, cell_id, pick_pos):
        """Called when one viewer selects a triangle."""
        self.select_cell_id = cell_id
        for v in self.viewers:
            v.highlight_triangle(cell_id, pick_pos)

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

        if preset == "axial":
            camera.SetPosition(center[0], center[1], center[2] + 600)
            camera.SetViewUp(0, 1, 0)
        elif preset == "coronal":
            camera.SetPosition(center[0], center[1] + 600, center[2])
            camera.SetViewUp(0, 0, 1)
        elif preset == "sagittal":
            camera.SetPosition(center[0] + 600, center[1], center[2])
            camera.SetViewUp(0, 0, 1)
        elif preset == "oblique":
            camera.SetPosition(center[0] + 400,
                               center[1] + 400,
                               center[2] + 400)
            camera.SetViewUp(0, 0, 1)

        # Update all viewers
        for v in self.viewers:
            v.vtkWidget.GetRenderWindow().Render()

if __name__ == "__main__":

    
    app = QApplication(sys.argv)

    # Replace with path to your GIFTI file
    gifti_files = []
    gifti_files.append(('/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/skin.surf.gii',
                        '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/distances_skin.func.gii',
                        '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/distances_skin_thresholded.func.gii',
                        [0,100],
                        'Distance'))
    gifti_files.append(('/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/skin.surf.gii',
                        '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/target_intersection_skin.func.gii',
                        '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/distances_skin_thresholded.func.gii',
                        [0,100],
                        'Target Intersection'))
    gifti_files.append(('/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/skin.surf.gii',
                        '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/angles_skin.func.gii',
                        '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/distances_skin_thresholded.func.gii',
                        [0,20],
                        'Angle'))
    gifti_files.append(('/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/skin.surf.gii',
                        '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/skin_skull_angles_skin.func.gii',
                        '/Users/spichardo/Documents/TempForSim/SDR_0p55/m2m_SDR_0p55/PlanTUS/Q_PlanTUSMask/distances_skin_thresholded.func.gii',
                        [0,20],
                        'Skin-Skull Angle'))

    widget = MultiGiftiViewerWidget(gifti_files,MaxViews=4)
    widget.resize(1600, 600)
    widget.show()

    sys.exit(app.exec())
