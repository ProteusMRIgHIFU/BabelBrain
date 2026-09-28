import os

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.io import loadmat

def generate_annular_curved_surface(length_step, focal_length, outer_diameter, inner_diameter=0):
    """
    One sentence summary of what the method does.
    
    Origin is at focal spot
    -z -> towards tx surface
    +z -> away from tx surface
    
    Args:
        param1 (type): Description of param1.
        param2 (type): Description of param2.
    
    Returns:
        return_type: Description of what is returned.
    
    Raises:
        ValueError: When and why this is raised.
        TypeError:  When and why this is raised.
    
    Sets:
        self.attr (type): Description of attribute assigned.
    
    Notes:
        - Any gotchas, assumptions, or non-obvious behaviour.
        - Reference to related methods if helpful.
    """
    
    tx = {}
    inner_radius = inner_diameter / 2
    outer_radius = outer_diameter / 2
    inner_beta = np.arcsin(inner_radius / focal_length)
    outer_beta = np.arcsin(outer_radius / focal_length)

    beta_elem_center = (outer_beta + inner_beta) / 2
    
    # xyz coordinates for center of ring face
    element_center = [
        [
            np.sin(beta_elem_center) * focal_length,    # x
            0.0,                                        # y
            -np.cos(beta_elem_center) * focal_length,   # z
        ]
    ]

    beta_difference = outer_beta - inner_beta

    arc_length_element = beta_difference * focal_length

    num_steps = np.ceil(arc_length_element / length_step)
    beta_step = beta_difference / num_steps

    print(f"Angle Start: {inner_beta + beta_step / 2} rads\nAngle End:{inner_beta + beta_step * (1 / 2 + num_steps)}rads\nAngle step:{beta_step}rads")
    
    betas = np.arange(inner_beta + beta_step / 2, inner_beta + beta_step * (1 / 2 + num_steps), beta_step)

    # Idea to to break annular curved surface into grid of quad elements
    centres = np.zeros((0, 3)) # Holds xyz coordinates for each quad element center
    normals = np.zeros((0, 3)) # Holds normal vector for each quad element
    ds = np.zeros((0, 1))   # Hold surface area size of each quad element
    vertices_display = np.zeros((0, 3)) # Hold corner/vertices xyz coordinates of each quad element
    face_display = np.zeros((0, 4), np.int64) # Finds the indices for 4 vertices in vertices_display that make up a face 

    index = 0
    for beta_index in range(len(betas)):
        circumference = np.sin(betas[beta_index]) * focal_length * 2 * np.pi

        num_alphas = np.ceil(circumference / length_step)
        alpha_step = 2 * np.pi / num_alphas

        alphas = np.arange(alpha_step / 2, alpha_step * (1 / 2 + num_alphas), alpha_step)

        centres = np.vstack((centres, np.zeros((len(alphas), 3))))
        normals = np.vstack((normals, np.zeros((len(alphas), 3))))
        ds = np.vstack((ds, np.zeros((len(alphas), 1))))

        vertices_display = np.vstack((vertices_display, np.zeros((len(alphas) * 4, 3))))
        face_display = np.vstack((face_display, np.zeros((len(alphas), 4), np.int64)))

        z_centre = -np.cos(betas[beta_index]) * focal_length
        radius_centre = np.sin(betas[beta_index]) * focal_length

        beta_1 = betas[beta_index] - beta_step / 2
        beta_2 = betas[beta_index] + beta_step / 2
        if beta_index == 0 and inner_diameter == 0.0:
            radius_centre_1 = 0
        else:
            radius_centre_1 = np.sin(beta_1) * focal_length

        radius_centre_2 = np.sin(beta_2) * focal_length

        z_centre_1 = -np.cos(beta_1) * focal_length
        z_centre_2 = -np.cos(beta_2) * focal_length

        centres[index:, 0] = radius_centre * np.cos(alphas)
        centres[index:, 1] = radius_centre * np.sin(alphas)
        centres[index:, 2] = z_centre

        alphas_1 = alphas - alpha_step / 2
        alphas_2 = alphas + alpha_step / 2
        
        # Small Surface Area
        ds[index:, 0] = focal_length**2 * (np.cos(beta_1) - np.cos(beta_2)) * (alphas_2 - alphas_1) 
        
        # Normals
        normals[index:, :] = centres[index:, :] / np.repeat(
            np.linalg.norm(centres[index:, :], axis=1).reshape((len(alphas), 1)),
            3,
            axis=1,
        )
        
        # Top Inner
        vertices_display[index * 4 :: 4, 0] = radius_centre_1 * np.cos(alphas_1)
        vertices_display[index * 4 :: 4, 1] = radius_centre_1 * np.sin(alphas_1)
        vertices_display[index * 4 :: 4, 2] = z_centre_1

        # Top Outer
        vertices_display[index * 4 + 1 :: 4, 0] = radius_centre_1 * np.cos(alphas_2)
        vertices_display[index * 4 + 1 :: 4, 1] = radius_centre_1 * np.sin(alphas_2)
        vertices_display[index * 4 + 1 :: 4, 2] = z_centre_1

        # Bottom Inner
        vertices_display[index * 4 + 2 :: 4, 0] = radius_centre_2 * np.cos(alphas_1)
        vertices_display[index * 4 + 2 :: 4, 1] = radius_centre_2 * np.sin(alphas_1)
        vertices_display[index * 4 + 2 :: 4, 2] = z_centre_2

        # Bottom Outer
        vertices_display[index * 4 + 3 :: 4, 0] = radius_centre_2 * np.cos(alphas_2)
        vertices_display[index * 4 + 3 :: 4, 1] = radius_centre_2 * np.sin(alphas_2)
        vertices_display[index * 4 + 3 :: 4, 2] = z_centre_2

        # Top Inner -> Top Outer -> Bottom Outerer -> Bottom Innerer
        face_display[index:, 0] = (index + np.arange(len(alphas))) * 4
        face_display[index:, 1] = (index + np.arange(len(alphas))) * 4 + 1
        face_display[index:, 2] = (index + np.arange(len(alphas))) * 4 + 3
        face_display[index:, 3] = (index + np.arange(len(alphas))) * 4 + 2
        index += len(alphas)

    tx["center"] = centres
    tx["ds"] = ds
    tx["normal"] = normals
    tx["VertDisplay"] = vertices_display
    tx["FaceDisplay"] = face_display
    tx["Beta1"] = np.array([[inner_beta]])
    tx["Beta2"] = np.array([[outer_beta]])
    tx["elemcenter"] = np.array(element_center)
    tx["elemdims"] = np.array([[len(ds)]])
    return tx

def generate_curved_surface(length_step, outer_diameter, focal_length):
    curved_tx = generate_annular_curved_surface(length_step,focal_length,outer_diameter,inner_diameter=0)
    
    return curved_tx
    
# def generate_focused_tx(frequency, focal_length, tx_diameter, sos=1500, ppw_surface=4):
#     wavelength = sos / frequency
#     lstep = wavelength / ppw_surface
#     tx = generate_curved_surface(lstep, tx_diameter, focal_length)
#     return tx

def generate_curved_element(frequency, focal_length, tx_diameter, sos=1500, ppw_surface=4):
    wavelength = sos / frequency
    lstep = wavelength / ppw_surface
    tx = generate_curved_surface(lstep, tx_diameter, focal_length)
    return tx

def generate_annular_array_tx(frequency, focal_length, inner_diameters, outer_diameters, sos=1500, ppw_surface=8):
    print(f"Generating Tx with frequency {frequency}, focal length {focal_length}, inner diameters {inner_diameters} and outer diameters {outer_diameters}")
    wavelength = sos / frequency
    lstep = wavelength / ppw_surface

    is_first_ring = True
    for inner_diameter, outer_diameter in zip(inner_diameters, outer_diameters):
        tx_ring = generate_annular_curved_surface(lstep, focal_length, outer_diameter, inner_diameter)
        if is_first_ring:
            annular_array_tx = tx_ring
            annular_array_tx["RingFaceDisplay"] = [annular_array_tx.pop("FaceDisplay")]
            annular_array_tx["RingVertDisplay"] = [annular_array_tx.pop("VertDisplay")]
            is_first_ring = False
        else:
            for key in tx_ring.keys():
                if key in ["FaceDisplay", "VertDisplay"]:
                    annular_array_tx["Ring" + key].append(tx_ring[key])
                else:
                    annular_array_tx[key] = np.vstack((annular_array_tx[key], tx_ring[key]))

    return annular_array_tx

def generate_square_element(tx_elem_length, deadspace, frequency, sos=1500, ppw=12.0):
    """
    One sentence summary of what the method does.
    
    Origin is at outplane
    -z -> towards tx surface
    +z -> away from tx surface
    
    Args:
        param1 (type): Description of param1.
        param2 (type): Description of param2.
    
    Returns:
        return_type: Description of what is returned.
    
    Raises:
        ValueError: When and why this is raised.
        TypeError:  When and why this is raised.
    
    Sets:
        self.attr (type): Description of attribute assigned.
    
    Notes:
        - Any gotchas, assumptions, or non-obvious behaviour.
        - Reference to related methods if helpful.
    """
    tx = {}

    sim_step = sos / frequency / ppw
    half_step = sim_step / 2.0
    num_lat_steps = int(np.round(tx_elem_length / sim_step))
    lat_step = tx_elem_length / num_lat_steps

    centers_x = np.arange(num_lat_steps) * lat_step
    centers_x -= np.mean(centers_x)

    centres = np.zeros((num_lat_steps**2, 3))
    normals = np.zeros((centres.shape[0], 3))
    normals[:, 2] = 1 # +z direction
    ds = np.ones((centres.shape[0], 1)) * lat_step**2
    vertices_display = np.zeros((centres.shape[0] * 4, 3))
    face_display = np.arange(centres.shape[0] * 4, dtype=int).reshape((centres.shape[0], 4))

    element_centers_xx, element_centers_yy = np.meshgrid(centers_x, centers_x)
    centres[:, 0] = element_centers_xx.flatten()
    centres[:, 1] = element_centers_yy.flatten()
    centres[:, 2] = deadspace

    # Top Left Corner
    vertices_display[0::4, 0] = centres[:, 0] - half_step
    vertices_display[0::4, 1] = centres[:, 1] - half_step

    # Top Right Corner
    vertices_display[1::4, 0] = centres[:, 0] + half_step
    vertices_display[1::4, 1] = centres[:, 1] - half_step

    # Bottom Right Corner
    vertices_display[2::4, 0] = centres[:, 0] + half_step
    vertices_display[2::4, 1] = centres[:, 1] + half_step

    # Bottom Left Corner
    vertices_display[3::4, 0] = centres[:, 0] - half_step
    vertices_display[3::4, 1] = centres[:, 1] + half_step

    vertices_display[:, 2] = deadspace

    tx["center"] = centres
    tx["ds"] = ds
    tx["normal"] = normals
    tx["VertDisplay"] = vertices_display
    tx["FaceDisplay"] = face_display

    return tx

def generate_flat_array_2d_tx(element_coords, num_elements, tx_elem_width, deadspace, frequency, rotation_z=0.0, validate_elements=True, ppw_surface=12.0, sos=1500):
    '''
    Creates an individual square tx element and performs matrix operations to copy/move to every tx element location
    in a 2D array
    '''

    # Individual tx element
    tx_element = generate_square_element(tx_elem_width, deadspace, frequency, sos, ppw_surface)
    
    # Validate element coords
    if validate_elements:
        check_normal_distance_tolerance(element_coords,num_elements,tx_elem_width)

    # Initialization
    flat_2d_array_tx = {}
    flat_2d_array_tx["center"] = np.zeros((0, 3))
    flat_2d_array_tx["elemcenter"] = np.zeros((num_elements, 3))
    flat_2d_array_tx["ds"] = np.zeros((0, 1))
    flat_2d_array_tx["normal"] = np.zeros((0, 3))
    flat_2d_array_tx["elemdims"] = tx_element["ds"].size
    flat_2d_array_tx["NumberElems"] = flat_2d_array_tx["elemcenter"].shape[0]
    flat_2d_array_tx["VertDisplay"] = np.zeros((0, 3))
    flat_2d_array_tx["FaceDisplay"] = np.zeros((0, 4), np.int64)

    # Rotation matrix for z axis
    if isinstance(rotation_z, np.ndarray) and rotation_z.ndim == 2:
        rotation_matrix_z = rotation_z
    else:
        rotation_z = np.deg2rad(rotation_z)
        rotation_matrix_z = np.array([
            [-np.cos(rotation_z), np.sin(rotation_z), 0],
            [-np.sin(rotation_z), -np.cos(rotation_z), 0],
            [0, 0, 1],
        ])
    
    for n in range(element_coords.shape[0]):

        # Get centre coordinates for each ds surface in current tx element
        current_element_center = tx_element["center"] + element_coords[n, :]
        current_element_center = (rotation_matrix_z @ current_element_center.T).T # Rotation around z axis

        # Center of whole tx element
        flat_2d_array_tx["elemcenter"][n, :] = np.mean(current_element_center, axis=0)

        # Normal vector stays the same as individual element
        normal = tx_element["normal"].copy()

        # Get coordinates for each vertex/corner of each ds surface in current tx element
        vertices_display = tx_element["VertDisplay"] + element_coords[n, :]
        vertices_display = (rotation_matrix_z @ vertices_display.T).T   # Rotation around z axis
        
        prev_face_length = flat_2d_array_tx["VertDisplay"].shape[0]

        # Add current tx element values to overall tx
        flat_2d_array_tx["center"] = np.vstack((flat_2d_array_tx["center"], current_element_center))
        flat_2d_array_tx["ds"] = np.vstack((flat_2d_array_tx["ds"], tx_element["ds"]))
        flat_2d_array_tx["normal"] = np.vstack((flat_2d_array_tx["normal"], normal))
        flat_2d_array_tx["VertDisplay"] = np.vstack((flat_2d_array_tx["VertDisplay"], vertices_display))
        flat_2d_array_tx["FaceDisplay"] = np.vstack((flat_2d_array_tx["FaceDisplay"], tx_element["FaceDisplay"] + prev_face_length))
        
    print('Aperture dimensions (x,y) =',flat_2d_array_tx['VertDisplay'][:,0].max()-flat_2d_array_tx['VertDisplay'][:,0].min(),
                                        flat_2d_array_tx['VertDisplay'][:,1].max()-flat_2d_array_tx['VertDisplay'][:,1].min())
    flat_2d_array_tx['Aperture'] = np.max([flat_2d_array_tx['VertDisplay'][:,0].max()-flat_2d_array_tx['VertDisplay'][:,0].min(),
                                        flat_2d_array_tx['VertDisplay'][:,1].max()-flat_2d_array_tx['VertDisplay'][:,1].min()]);
    return flat_2d_array_tx

def generate_flat_annular_array_tx(frequency, aperture, focal_length, inner_diameters, outer_diameters, sos=1500, ppw_surface=8, is_original_dimensions=False, enlargement_factor=None):
    
    flat_annular_array_tx = generate_annular_array_tx(frequency, focal_length, inner_diameters, outer_diameters, sos, ppw_surface)

    flat_annular_array_tx['Aperture'] = aperture
    flat_annular_array_tx['NumberElems'] = len(inner_diameters)
    flat_annular_array_tx['center'][:,2] = 0
    flat_annular_array_tx['elemcenter'][:,2] = 0
    
    for n in range(len(flat_annular_array_tx['RingVertDisplay'])):
        flat_annular_array_tx['RingVertDisplay'][n][:,2]=0
            
    return flat_annular_array_tx

def generate_focused_array_tx(element_coords, num_elements, frequency, focal_length, element_diameter, validate_elements=True, sos=1500, rotation_z=0.0, coordinate_sys="spherical",show_plot=False,ppw_surface=8):
    
    # Individual tx element
    tx_element = generate_curved_element(frequency,focal_length,element_diameter,sos,ppw_surface)
    
    # Convert degrees to radians
    if coordinate_sys == "spherical":
        element_coords[:,1] = np.deg2rad(element_coords[:,1]) # theta
        element_coords[:,2] = np.deg2rad(element_coords[:,2]) # phi
        
    # Validate element coords
    if validate_elements:
        check_angular_distance_tolerance(element_coords,num_elements,focal_length,element_diameter,coordinate_sys)
    
    new_element_coords = element_coords.copy()
    
    # Convert cartesian coordinates to spherical
    if coordinate_sys == "cartesian":
        # Reverse z coorindates
        # new_element_coords[:,2] = focal_length-new_element_coords[:,2]
        # new_element_coords[:,2] -= focal_length 
        
        new_element_coords[:,2] *= -1 # reverse z coordinates sign
        cart_coords = new_element_coords.copy()
        new_element_coords = cart_to_spherical(cart_coords)
    
    # Perform specified z rotation
    new_element_coords[:,2] += np.deg2rad(rotation_z)
    
    # Initialization
    focused_array_tx = {}
    focused_array_tx["center"] = np.zeros((0, 3))
    focused_array_tx["elemcenter"] = np.zeros((num_elements, 3))
    focused_array_tx["ds"] = np.zeros((0, 1))
    focused_array_tx["normal"] = np.zeros((0, 3))
    focused_array_tx["elemdims"] = tx_element["ds"].size
    focused_array_tx["NumberElems"] = focused_array_tx["elemcenter"].shape[0]
    focused_array_tx["VertDisplay"] = np.zeros((0, 3))
    focused_array_tx["FaceDisplay"] = np.zeros((0, 4), np.int64)
    
    # Loop through each theta value
    thetas = new_element_coords[:,1]
    phis = new_element_coords[:,2]
    
    for n in range(len(thetas)):
        theta = thetas[n]
        phi = phis[n]
        
        prev_face_length = focused_array_tx['VertDisplay'].shape[0]
        
        # Rotate element to face origin
        rotation_matrix_y = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        rotation_matrix_z = np.array([[-np.cos(phi),np.sin(phi),0],[-np.sin(phi),-np.cos(phi),0],[0,0,1]])
        rotation_matrix = rotation_matrix_z@rotation_matrix_y

        center = (rotation_matrix@tx_element['center'].T).T
        focused_array_tx['elemcenter'][n,:] = center[0,:] # the very first subelement is at the center
        normal = (rotation_matrix@tx_element['normal'].T).T
        vertices_display = (rotation_matrix@tx_element['VertDisplay'].T).T

        # Add rotated element to overall transducer
        focused_array_tx['center'] = np.vstack((focused_array_tx['center'],center))
        focused_array_tx['ds'] = np.vstack((focused_array_tx['ds'],tx_element['ds']))
        focused_array_tx['normal'] = np.vstack((focused_array_tx['normal'],normal))
        focused_array_tx['VertDisplay'] = np.vstack((focused_array_tx['VertDisplay'],vertices_display))
        focused_array_tx['FaceDisplay']= np.vstack((focused_array_tx['FaceDisplay'],tx_element['FaceDisplay']+prev_face_length))
        
    # Shift origin back to tx back
    focused_array_tx['VertDisplay'][:,2] += focal_length
    focused_array_tx['center'][:,2] += focal_length
    focused_array_tx['elemcenter'][:,2] += focal_length

    if show_plot:
        plot_elements(focused_array_tx['center'])
        
    focused_array_tx['FocalLength'] = focal_length
    focused_array_tx["Aperture"] = np.max([
        focused_array_tx["VertDisplay"][:, 0].max() - focused_array_tx["VertDisplay"][:, 0].min(),
        focused_array_tx["VertDisplay"][:, 1].max() - focused_array_tx["VertDisplay"][:, 1].min(),
    ])
    print('Aperture dimensions (x,y) =',focused_array_tx['VertDisplay'][:,0].max()-focused_array_tx['VertDisplay'][:,0].min(),
                                        focused_array_tx['VertDisplay'][:,1].max()-focused_array_tx['VertDisplay'][:,1].min())
    
    return focused_array_tx
    
def check_angular_distance_tolerance(element_coords,num_elements,focal_length,distance_tolerance=0,coordinate_sys="spherical"):
    min_inter_element_distance = distance_tolerance / focal_length # Units: radians
    print('*****\nMinimal angular distance\n*****', min_inter_element_distance)
    
    # Calculate the closest Tx element distance to each other
    min_distances=np.zeros(num_elements)
    for n in range(num_elements):
        selected_element = element_coords[n,:].reshape((1,3))
        
        # All other tx elements
        rest_indices = np.hstack((np.arange(0,n),np.arange(n+1,num_elements)))
        rest_tx = element_coords[rest_indices,:]
        
        # Calculate angular distances
        if coordinate_sys == "spherical":
            angular_distances = angular_distance_spherical(selected_element[0,1],selected_element[0,2],rest_tx[:,1],rest_tx[:,2])
        elif coordinate_sys == "cartesian":
            angular_distances = angular_distance_cartesian(selected_element,rest_tx,focal_length)
        else:
            raise ValueError(f"coordinate_sys is not valid value ({coordinate_sys})")
        
        min_distances[n] = angular_distances.min()
        print('Closest element distance',n,min_distances[n]) # just for nicer printing
        
        if min_distances[n] < min_inter_element_distance:
            print(f' ******** overlap of elem {n} with {rest_indices[np.argmin(angular_distances)]}')
    
    if not np.all(min_distances >= min_inter_element_distance):
        raise ValueError("There are some tx elements spaced closer than the minimum allowable distance")

def check_normal_distance_tolerance(element_coords,num_elements,distance_tolerance=0):
    min_inter_element_distance = distance_tolerance # Units: m
    print('*****\nMinimal flat distance\n*****', min_inter_element_distance)
    
    # Calculate the closest Tx element distance to each other in a flat plane
    min_distances=np.zeros(num_elements)
    for n in range(num_elements):
        selected_element = element_coords[n,:].reshape((1,3))
        
        # All other tx elements
        rest_indices = np.hstack((np.arange(0,n),np.arange(n+1,num_elements)))
        rest_tx = element_coords[rest_indices,:]
        
        # Calculate distances, assuming elements possess the same z coordinate
        euc_distances = euclidean_distance(selected_element,rest_tx)
        
        min_distances[n] = euc_distances.min()
        print('Closest element distance',n,min_distances[n]) # just for nicer printing
        
        if min_distances[n] < min_inter_element_distance:
            print(f' ******** overlap of elem {n} with {rest_indices[np.argmin(euc_distances)]}')
    
    if not np.all(min_distances >= min_inter_element_distance):
        raise ValueError("There are some tx elements spaced closer than the minimum allowable distance")
    
def angular_distance_cartesian(p1, p2, focal_length):
    
    # Calculate the Euclidean distances 
    euclidean_distances = np.linalg.norm(p2 - p1, axis=-1)   # axis=-1 needed for broadcasting case
    normalized_distances = euclidean_distances / focal_length
    
    # Calculate angular distance
    angular_distance = 2 * np.arcsin(np.clip(normalized_distances / 2.0, -1.0, 1.0))
    
    return angular_distance
    
def angular_distance_spherical(theta1, phi1, theta2, phi2):
    """
    theta = polar angle (from z-axis)
    phi   = azimuthal angle (x-y plane)
    """
    dtheta = theta2 - theta1
    dphi   = phi2 - phi1
    a = np.sin(dtheta / 2)**2 + np.sin(theta1) * np.sin(theta2) * np.sin(dphi / 2)**2
    return 2 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))

def euclidean_distance(p1,p2):
    euclidean_distances = np.linalg.norm(p2 - p1, axis=-1)   # axis=-1 needed for broadcasting case
    
    return euclidean_distances

def cart_to_spherical(xyz):
    """
    Convert Cartesian coordinates to spherical coordinates (physics convention).
    
    Convention: theta = polar angle from +z axis [0, pi]
                phi    = azimuthal angle in x-y plane [-pi, pi]
    
    Parameters
    ----------
    xyz : array_like, shape (N, 3)
        Cartesian coordinates as columns [x, y, z].
    
    Returns
    -------
    rtp : ndarray, shape (N, 3)
        Columns [r, theta, phi]:
        r     - radial distance
        theta - polar angle (radians), measured from +z axis
        phi   - azimuthal angle (radians), measured from +x axis in x-y plane
    """
    xyz = np.asarray(xyz, dtype=float)
    # x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
    # r = np.sqrt(x**2 + y**2 + z**2)
    # theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # clip guards r=0 or fp roundoff
    # phi = np.arctan2(y, x)
    
    # return np.column_stack((r, theta, phi))
    
    r_xy = np.linalg.norm(xyz[:,:2],axis=1)
    r = np.linalg.norm(xyz,axis=1)
    
    thetas = np.arcsin(r_xy/r)
    phis = np.arctan2(xyz[:,1],xyz[:,0])
    
    return np.column_stack((r, thetas, phis))


def spherical_to_cart(rtp):
    """
    Convert spherical coordinates to Cartesian coordinates (physics convention).
    
    Convention: theta = polar angle from +z axis
                phi    = azimuthal angle in x-y plane
    
    Parameters
    ----------
    rtp : array_like, shape (N, 3)
        Spherical coordinates as columns [r, theta, phi] (theta, phi in radians).
    
    Returns
    -------
    xyz : ndarray, shape (N, 3)
        Cartesian coordinates as columns [x, y, z].
    """
    rtp = np.asarray(rtp, dtype=float)
    r, theta, phi = rtp[:, 0], rtp[:, 1], rtp[:, 2]
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.column_stack((x, y, z))

def plot_elements(element_locations,element_step=1,show_origin=True,reverse_z_dir=True):
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(element_locations[::element_step,0],element_locations[::element_step,1],element_locations[::element_step,2], marker='o')
    if show_origin:
        ax.scatter(0, 0, 0, marker='x', color='red', s=100, label='Origin')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.axis('equal')
    
    # Reverse Z direction
    if reverse_z_dir:
        ax.invert_zaxis()   
    
    plt.show()