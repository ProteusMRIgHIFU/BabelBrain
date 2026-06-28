"""
merge_nifti.py
--------------
Combine two NIfTI volumes into a single image that is the smallest possible
3D bounding box (in world/RAS space) that contains all voxels from both inputs.

Requirements:
    pip install nibabel numpy scipy

Usage:
    python merge_nifti.py img1.nii.gz img2.nii.gz merged.nii.gz

Strategy
--------
1. Pick one image's affine as the output reference frame (default: image 1).
   You can also pass --use-diagonal-affine to build a clean axis-aligned
   output affine from the voxel size of image 1.
2. Map every corner of both volumes into world (mm) space.
3. Compute the tight bounding box in the *output* voxel space that encloses
   all corners → gives the minimum output shape.
4. Resample both volumes into that bounding box with scipy ndimage.
5. Merge by taking the maximum value at each voxel (change MERGE_FUNC
   to np.nansum, np.nanmean, etc. if you prefer a different blend).
"""

import argparse
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def volume_corners_world(img):
    """Return the 8 corners of a 3-D volume in world (mm) space."""
    shape = np.array(img.shape[:3])  # ignore 4th dim if present
    # 8 corners in voxel coordinates (0-based, at edge of last voxel)
    corners_vox = np.array(
        [[i, j, k]
         for i in [0, shape[0] - 1]
         for j in [0, shape[1] - 1]
         for k in [0, shape[2] - 1]], dtype=float
    )
    # Apply the affine to convert to world coords  (N×3)
    corners_world = nib.affines.apply_affine(img.affine, corners_vox)
    return corners_world


def build_output_affine(ref_img, use_diagonal=False):
    """
    Build the affine for the output image.

    ref_img       : nibabel image whose orientation / spacing we adopt.
    use_diagonal  : if True, strip shear/rotation and use a pure
                    diagonal (axis-aligned) affine.  Useful when you
                    want a clean RAS+ output regardless of the source
                    orientation.
    """
    if use_diagonal:
        voxel_sizes = ref_img.header.get_zooms()[:3]
        # Preserve the sign (L/R flip) of the original affine diagonal
        signs = np.sign(np.diag(ref_img.affine)[:3])
        signs[signs == 0] = 1
        aff = np.diag([*signs * voxel_sizes, 1]).astype(float)
        # Translation will be set later
        return aff
    else:
        # Use the full affine (including any oblique orientation)
        return ref_img.affine.copy().astype(float)


def world_to_vox(affine, world_pts):
    """Map an (N×3) array of world coords back to voxel coords."""
    inv_aff = np.linalg.inv(affine)
    return nib.affines.apply_affine(inv_aff, world_pts)


def resample_img_into_target(src_img, target_affine, target_shape,
                              order=1, cval=0.0):
    """
    Resample src_img into the space defined by target_affine / target_shape.

    Uses scipy.ndimage.affine_transform which maps *output* voxel coords
    back to *input* voxel coords.

    order : interpolation order (0=nearest, 1=linear, 3=cubic)
    cval  : fill value for regions outside the source volume
    """
    # Matrix that maps output voxels → world → source voxels
    src_inv_affine = np.linalg.inv(src_img.affine)
    out2src = src_inv_affine @ target_affine          # 4×4

    # scipy affine_transform wants (matrix, offset) in voxel space
    matrix = out2src[:3, :3]
    offset = out2src[:3,  3]

    src_data = np.asarray(src_img.dataobj, dtype=np.float64)
    if src_data.ndim == 4:
        # Handle 4-D by resampling each volume
        n_vols = src_data.shape[3]
        out = np.zeros((*target_shape, n_vols), dtype=np.float64)
        for v in range(n_vols):
            out[..., v] = affine_transform(
                src_data[..., v], matrix, offset=offset,
                output_shape=target_shape, order=order, cval=cval,
                mode="constant"
            )
        return out
    else:
        return affine_transform(
            src_data, matrix, offset=offset,
            output_shape=target_shape, order=order, cval=cval,
            mode="constant"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Core merge logic
# ──────────────────────────────────────────────────────────────────────────────

def merge_niftis(img1, img2,
                 use_diagonal_affine=False,
                 interp_order=1,
                 merge_func=np.fmax):           # element-wise max, NaN-safe
    """
    Return a new nibabel Nifti1Image that is the smallest bounding box
    (in the reference frame of img1) containing all data from img1 and img2.
    """
    # ── 1. Choose output affine direction/spacing (no translation yet) ──────
    out_affine = build_output_affine(img1, use_diagonal=use_diagonal_affine)

    # ── 2. Find bounding box in output voxel space ──────────────────────────
    # Gather corners of both images in world space
    corners_world = np.vstack([
        volume_corners_world(img1),
        volume_corners_world(img2),
    ])

    # Map to output voxel space (using only rotation/scaling, no translation)
    # We'll set translation so that voxel (0,0,0) maps to the min corner.
    inv_rot = np.linalg.inv(out_affine[:3, :3])
    corners_vox_tmp = (inv_rot @ corners_world.T).T   # N×3

    min_vox = np.floor(corners_vox_tmp.min(axis=0)).astype(int)
    max_vox = np.ceil( corners_vox_tmp.max(axis=0)).astype(int)

    out_shape = tuple((max_vox - min_vox + 1).tolist())
    print(f"Output shape (voxels): {out_shape}")

    # ── 3. Set translation so voxel (0,0,0) = min corner in world space ─────
    # world = R @ vox_tmp  →  vox_tmp = R⁻¹ @ world
    # We want vox_out = vox_tmp - min_vox, so:
    #   world = R @ (vox_out + min_vox) = R @ vox_out + R @ min_vox
    out_affine[:3, 3] = out_affine[:3, :3] @ min_vox

    print("Output affine:\n", out_affine)

    # ── 4. Resample both images into the output grid ─────────────────────────
    print("Resampling image 1 …")
    data1 = resample_img_into_target(img1, out_affine, out_shape,
                                     order=interp_order)
    print("Resampling image 2 …")
    data2 = resample_img_into_target(img2, out_affine, out_shape,
                                     order=interp_order)

    # ── 5. Merge ──────────────────────────────────────────────────────────────
    print("Merging …")
    merged = merge_func(data1, data2)

    # ── 6. Build output NIfTI ─────────────────────────────────────────────────
    out_img = nib.Nifti1Image(merged.astype(np.float32), out_affine)
    # Copy header fields (pixdim, units, etc.) from the reference image
    out_img.header.set_zooms(img1.header.get_zooms()[:3])
    out_img.header.set_xyzt_units(
        xyz=img1.header.get_xyzt_units()[0],
        t=img1.header.get_xyzt_units()[1],
    )

    return out_img


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

MERGE_MODES = {
    "max":  np.fmax,                                                              # element-wise max (ignores NaN)
    "sum":  lambda a, b: np.nansum( np.stack([a, b], axis=0), axis=0),           # sum
    "mean": lambda a, b: np.nanmean(np.stack([a, b], axis=0), axis=0),           # mean
    "min":  np.fmin,                                                              # element-wise min
}


def main():
    parser = argparse.ArgumentParser(
        description="Merge two NIfTI volumes into the smallest bounding box "
                    "that contains both."
    )
    parser.add_argument("input1",  help="First NIfTI file (.nii / .nii.gz)")
    parser.add_argument("input2",  help="Second NIfTI file (.nii / .nii.gz)")
    parser.add_argument("output",  help="Output NIfTI file (.nii / .nii.gz)")
    parser.add_argument(
        "--interp", type=int, default=1, choices=[0, 1, 3],
        help="Interpolation order: 0=nearest, 1=linear (default), 3=cubic"
    )
    parser.add_argument(
        "--merge", default="sum", choices=list(MERGE_MODES),
        help="How to combine overlapping voxels (default: sum)"
    )
    parser.add_argument(
        "--use-diagonal-affine", action="store_true",
        help="Strip rotation from output affine → axis-aligned output"
    )
    args = parser.parse_args()

    print(f"Loading {args.input1} …")
    img1 = nib.load(args.input1)
    print(f"  shape: {img1.shape}  zooms: {img1.header.get_zooms()[:3]}")

    print(f"Loading {args.input2} …")
    img2 = nib.load(args.input2)
    print(f"  shape: {img2.shape}  zooms: {img2.header.get_zooms()[:3]}")

    # Basic sanity check: similar voxel sizes
    z1 = np.array(img1.header.get_zooms()[:3])
    z2 = np.array(img2.header.get_zooms()[:3])
    if not np.allclose(z1, z2, rtol=0.05):
        print(
            f"WARNING: voxel sizes differ > 5 %: {z1} vs {z2}. "
            "Results may not be as expected.", file=sys.stderr
        )

    merged = merge_niftis(
        img1, img2,
        use_diagonal_affine=args.use_diagonal_affine,
        interp_order=args.interp,
        merge_func=MERGE_MODES[args.merge],
    )

    print(f"Saving to {args.output} …")
    nib.save(merged, args.output)
    print("Done.")
    print(f"  Output shape : {merged.shape}")
    print(f"  Output affine:\n{merged.affine}")


if __name__ == "__main__":
    main()