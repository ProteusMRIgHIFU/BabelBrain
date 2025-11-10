#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import nibabel as nib

def ras_to_voxel_idx(affine, ras_xyz):
    # Convert RAS (mm) to voxel index using inverse affine, round to nearest voxel
    vox = np.linalg.inv(affine).dot(np.append(ras_xyz, 1.0))[:3]
    return np.rint(vox).astype(int)

def in_bounds(idx, shape3d):
    return np.all(idx >= 0) and np.all(idx < np.array(shape3d))

def default_output_path(in_path, provided: None):
    if provided:
        return provided
    if in_path.endswith(".nii.gz"):
        stem = in_path[:-7]
    elif in_path.endswith(".nii"):
        stem = in_path[:-4]
    else:
        stem = in_path
    return f"{stem}_mask.nii.gz"

def ellipsoid_mask(shape, radii, center=None):
    """
    Create an ellipsoidal mask in a 3D array.

    Parameters
    ----------
    shape : tuple of int
        Dimensions of the 3D array (I, J, K).
    radii : tuple of float
        Radii along each axis (ri, rj, rk).
    center : tuple of float, optional
        Center of the ellipsoid (zi, yj, xk). If None, defaults to the center of the volume.

    Returns
    -------
    mask : np.ndarray of bool
        Boolean mask with True inside the ellipsoid.
    """
    I, J, K = shape
    ri, rj, rk = radii

    if center is None:
        center = (I / 2, J / 2, K / 2)

    # Coordinate grid
    z, y, x = np.ogrid[:I, :J, :K]

    # Normalized distance for ellipsoid
    dist = ((z - center[0]) / ri) ** 2 + ((y - center[1]) / rj) ** 2 + ((x - center[2]) / rk) ** 2

    # Inside ellipsoid if sum of squares <= 1
    mask = dist <= 1
    return mask.astype(np.float32)

def create_target_mask(in_path, ras, out_path, raddi=(1,1,1)):
    '''
    Create a binary mask NIfTI file with a single voxel set to 1 at the specified RAS coordinate.
    '''
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Error: input file not found: {in_path}")
    img = nib.load(in_path)
    affine = img.affine
    shape = img.shape
    if len(shape) < 3:
        raise ValueError("Error: input NIfTI must be at least 3D.")
    shape3d = shape[:3]


    idx = ras_to_voxel_idx(affine, ras)

    if not in_bounds(idx, shape3d):
        raise ValueError(f"Error: computed voxel index {tuple(idx)} is out of bounds for volume shape {shape3d}.")

    mask = ellipsoid_mask(shape3d, raddi, center=idx)

    # Prepare header: copy from input, but ensure correct datatype and 3D shape
    header = img.header.copy()
    header.set_data_dtype(np.float32)
    header.set_data_shape(mask.shape)

    out_img = nib.Nifti1Image(mask, affine, header=header)

    # Preserve sform/qform codes if set
    sform, scode = img.get_sform(coded=True)
    qform, qcode = img.get_qform(coded=True)
    if scode > 0:
        out_img.set_sform(affine, code=scode)
    if qcode > 0:
        out_img.set_qform(affine, code=qcode)

    
    nib.save(out_img, out_path)
    print(f"Saved mask to {out_path} (voxel index {tuple(idx)})")

def main():
    parser = argparse.ArgumentParser(
        description="Create a single-voxel mask NIfTI at a given RAS coordinate."
    )
    parser.add_argument("input_nifti", help="Path to input NIfTI file")
    parser.add_argument("ras_x", type=float, help="RAS X (mm)")
    parser.add_argument("ras_y", type=float, help="RAS Y (mm)")
    parser.add_argument("ras_z", type=float, help="RAS Z (mm)")
    parser.add_argument("-o", "--output", help="Output NIfTI path (default: <input>_mask.nii.gz)")
    args = parser.parse_args()

    in_path = args.input_nifti
    
    ras = np.array([args.ras_x, args.ras_y, args.ras_z], dtype=float)

    out_path = default_output_path(in_path, args.output)

    create_target_mask(in_path, ras, out_path)

if __name__ == "__main__":
    main()