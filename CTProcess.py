import SimpleITK as sitk
import numpy as np
import pydicom as dicom
import os
# import matplotlib.pyplot as plt
from glob import glob
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy

# from skimage import morphology
# from skimage import measure
# from skimage import feature
# from skimage.transform import resize
# from skimage.filters import threshold_otsu, threshold_local, median, gaussian, wiener,sobel, hessian, prewitt

from sklearn.cluster import KMeans

# from plotly import __version__
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import plotly.figure_factory as FF
# from plotly.graph_objs import *

from mayavi import mlab
from tvtk.api import tvtk
from trimesh import Trimesh
# from scipy import stats
# import skimage

# init_notebook_mode(connected=True) 

def GetSTL(CTFile):
    # sitk image reader
    imgMain = sitk.ReadImage(CTFile)
    imgOriginal = imgMain
    # threshold the image with upper and lower threshold values
    imgThreshold = sitk.DoubleThreshold(
            imgOriginal, 350, 350, 350, 5000, 1, 0)
    # convert the images into numpy array from sitk images
    imgArray = sitk.GetArrayFromImage(imgThreshold)
    # imgMainArray = sitk.GetArrayFromImage(imgMain)
    # imgOriginalArray = sitk.GetArrayFromImage(imgOriginal)
    # Get spacing origin and the direction of the images
    # spacing = np.asarray(imgOriginal.GetSpacing())
    # origin = np.asarray(imgOriginal.GetOrigin())
    # direction = np.asarray(imgOriginal.GetDirection())
    # print(spacing)

    # Swap the axes of the image stack to match with the spacing, origin and direction axes
    # imgArray = np.swapaxes(imgArray, 0, 2)
    edges = imgArray
    # Perform morpology in the image
    edges=scipy.ndimage.binary_erosion(imgArray,iterations=2)*1.0
    edges=scipy.ndimage.binary_dilation(imgArray,iterations=2)*1.0
    # imgTo3D = edges
    # SkullRing, points, faces,normals,result,compute_normals = ObtainSkullSurfaceAndRing(imgTo3D,spacing, origin)
    # boneMesh = Trimesh(vertices=points,faces=faces)
    # nP=(affine[:3,:3]@boneMesh.vertices.T).T
    # nP[:,0]+=affine[0,3]
    # nP[:,1]+=affine[1,3]
    # nP[:,2]+=affine[2,3]
    # boneMesh.vertices=nP
    return edges
    
    # 
# function to construct the surface of the skull using contours
def ObtainSkullSurfaceAndRing(MaterialMap, spacing, origin):
    SkullRegion=MaterialMap!=0
    SkullRing=np.logical_xor(scipy.ndimage.binary_dilation(SkullRegion),scipy.ndimage.binary_erosion(SkullRegion))
    
    # print ("0");
    
    data=SkullRegion.copy()
#     perfform dilation
    data=scipy.ndimage.binary_dilation(data,iterations=1)*1.0
    
    # print ("1");
    

#     Convert the image into a scalar field
    src = mlab.pipeline.scalar_field(data)
#     Set the origin and the spacing
    src.spacing = spacing
    src.update_image_data = True
    src.origin=origin

    
    # print ("2");
    
    srcOrig = mlab.pipeline.scalar_field((SkullRing)*1.0)
    srcOrig.spacing = spacing
    srcOrig.update_image_data = True
    srcOrig.origin=origin

    # print ("3");

#   Add median filter to the pipeline to remove any noise
    median_filter = tvtk.ImageMedian3D()
    try:
        median_filter.set_kernel_size(3, 3, 3)
    except AttributeError:
        median_filter.kernel_size = [3, 3, 3]
    
    # print ("4");
    
    median = mlab.pipeline.user_defined(src, filter=median_filter)

#     Difussion filter to remove the noise
    diffuse_filter = tvtk.ImageAnisotropicDiffusion3D(
                                        diffusion_factor=0.5,
                                        diffusion_threshold=1,
                                        number_of_iterations=1)

    # print ("5");
    
    diffuse = mlab.pipeline.user_defined(median, filter=diffuse_filter)

    # print ("6");
# create contour of the image
    contour = mlab.pipeline.contour(diffuse, )

    # print ("7");
    
    contour.filter.contours = [1, ]
# Aply decimation
    dec = mlab.pipeline.decimate_pro(contour)
    dec.filter.feature_angle = 90.
    dec.filter.target_reduction = 0.6

#     Apply smoothing filter
    smooth_ = tvtk.SmoothPolyDataFilter(
                        number_of_iterations=100,
                        relaxation_factor=0.1,
                        feature_angle=90,
                        feature_edge_smoothing=False,
                        boundary_smoothing=False,
                        convergence=0.,
                    )
    smooth = mlab.pipeline.user_defined(dec, filter=smooth_)
    
    # print ("8");
    
    # Get the largest connected region
    connect_ = tvtk.PolyDataConnectivityFilter(extraction_mode=4)
    connect = mlab.pipeline.user_defined(smooth, filter=connect_)
    
    # print ("9");
    
    # Compute normals for shading the surface
    compute_normals = mlab.pipeline.poly_data_normals(connect)
    compute_normals.filter.feature_angle = 80.

# get vertices and faces
    result=compute_normals.get_output_dataset()
    normals= np.array(result.point_data.normals)
    faces=result.polys.get_data().to_array().reshape((result.polys.number_of_cells,4))[:,1:4]

    points=np.array(result.points)
    points[:,0]-=1
    points[:,1]-=1
    points[:,2]-=1
    return SkullRing, points, faces,normals,result,compute_normals