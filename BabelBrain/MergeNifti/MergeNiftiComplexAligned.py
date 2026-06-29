"""
merge_nifti_complex.py
----------------------
Combine two complex NIfTI datasets — each represented as an amplitude/phase
pair — into a single summed complex field, saving the result as separate
amplitude and phase NIfTI files.

Given:
    Z1 = A * exp(i * B)     (file_A = amplitude, file_B = phase)
    Z2 = C * exp(i * D)     (file_C = amplitude, file_D = phase)

Outputs:
    |Z1 + Z2|               → amplitude output
    angle(Z1 + Z2)          → phase output  (radians, in [-π, π])

The output grid can be aligned to any of the three canonical anatomical
planes in world (RAS) space, regardless of the input orientations:

    --orientation axial      → voxel i=R, j=A, k=S  (standard RAS)
    --orientation coronal    → voxel i=R, j=S, k=A  (rows=R, slices=A)
    --orientation sagittal   → voxel i=A, j=S, k=R  (rows=A, slices=R)

In every case the bounding box is the smallest integer-voxel grid (in that
orientation) that encloses all data from both input pairs.

Requirements:
    pip install nibabel numpy scipy

Usage:
    python merge_nifti_complex.py \\
        amp1.nii.gz phase1.nii.gz \\
        amp2.nii.gz phase2.nii.gz \\
        out_amp.nii.gz out_phase.nii.gz \\
        [--orientation axial|coronal|sagittal] \\
        [--interp 0|1|3]
"""

import argparse
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


# ──────────────────────────────────────────────────────────────────────────────
# Canonical orientation rotation matrices  (pure rotation, no scaling)
#
# Each matrix R defines how voxel axes (i, j, k) map to world RAS axes.
# Column 0 = direction of increasing i  (first voxel index)
# Column 1 = direction of increasing j
# Column 2 = direction of increasing k
#
#   RAS unit vectors:  R=(1,0,0)  A=(0,1,0)  S=(0,0,1)
#
#   Axial    : i→R, j→A, k→S   — standard neurological axial view
#   Coronal  : i→R, j→S, k→A   — rows run left-right, slices front-back
#   Sagittal : i→A, j→S, k→R   — rows run front-back, slices left-right
# ──────────────────────────────────────────────────────────────────────────────

ORIENTATION_ROTATION = {
    #                  i-col        j-col        k-col
    "axial":    np.array([[ 1,  0,  0],   # i → R
                           [ 0,  1,  0],   # j → A
                           [ 0,  0,  1]],  # k → S
                          dtype=float).T,  # columns = voxel-axis directions

    "coronal":  np.array([[ 1,  0,  0],   # i → R
                           [ 0,  0,  1],   # j → S
                           [ 0,  1,  0]],  # k → A
                          dtype=float).T,

    "sagittal": np.array([[ 0,  1,  0],   # i → A
                           [ 0,  0,  1],   # j → S
                           [ 1,  0,  0]],  # k → R
                          dtype=float).T,
}


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def volume_corners_world(img):
    """Return the 8 corners of a 3-D volume in world (mm) space."""
    shape = np.array(img.shape[:3])
    corners_vox = np.array(
        [[i, j, k]
         for i in [0, shape[0] - 1]
         for j in [0, shape[1] - 1]
         for k in [0, shape[2] - 1]], dtype=float
    )
    return nib.affines.apply_affine(img.affine, corners_vox)


def build_canonical_affine(ref_img, orientation="axial"):
    """
    Build a pure canonical-orientation affine (rotation + isotropic voxel
    size from ref_img, no translation yet).

    The rotation matrix is one of the three standard anatomical planes.
    Voxel size is taken as the mean of the reference image's zooms so the
    output is isotropic (you can change this to per-axis zooms if desired).
    """
    zooms = np.array(ref_img.header.get_zooms()[:3], dtype=float)
    # Use per-axis voxel sizes mapped to the new axis order
    # For simplicity we keep the mean zoom so output is isotropic.
    # To preserve anisotropy swap the zoom assignment below.
    vox_size = float(np.mean(zooms))

    R = ORIENTATION_ROTATION[orientation]        # 3×3 pure rotation
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = R * vox_size               # scale columns by voxel size
    # Translation set later once min corner is known
    return affine


def compute_bounding_affine(imgs, orientation="axial", ref_img=None):
    """
    Return (out_affine, out_shape): the tightest bounding box in the chosen
    canonical orientation that encloses all volumes in `imgs`.

    imgs     : list of nibabel images whose extents must be covered
    ref_img  : image whose voxel size is used; defaults to imgs[0]
    """
    if ref_img is None:
        ref_img = imgs[0]

    out_affine = build_canonical_affine(ref_img, orientation=orientation)

    # Gather all corners in world space
    all_corners = np.vstack([volume_corners_world(img) for img in imgs])

    # Project into the (unshifted) output voxel space
    inv_rot = np.linalg.inv(out_affine[:3, :3])
    corners_vox_tmp = (inv_rot @ all_corners.T).T   # N×3

    min_vox = np.floor(corners_vox_tmp.min(axis=0)).astype(int)
    max_vox = np.ceil( corners_vox_tmp.max(axis=0)).astype(int)

    out_shape = tuple((max_vox - min_vox + 1).tolist())

    # Set translation: voxel (0,0,0) maps to the min-corner world point
    out_affine[:3, 3] = out_affine[:3, :3] @ min_vox

    return out_affine, out_shape


def resample_into_target(src_img, target_affine, target_shape,
                          order=1, cval=0.0):
    """
    Resample src_img into the space defined by target_affine / target_shape.
    Uses backward mapping (output voxel → source voxel) via scipy.
    Supports 3-D and 4-D volumes.
    """
    src_inv_affine = np.linalg.inv(src_img.affine)
    out2src = src_inv_affine @ target_affine
    matrix = out2src[:3, :3]
    offset = out2src[:3,  3]

    src_data = np.asarray(src_img.dataobj, dtype=np.float64)

    if src_data.ndim == 4:
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


def save_nifti(data, affine, ref_header, path):
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    # Infer voxel size from the affine column norms (may be isotropic now)
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    img.header.set_zooms(zooms)
    img.header.set_xyzt_units(
        xyz=ref_header.get_xyzt_units()[0],
        t=ref_header.get_xyzt_units()[1],
    )
    nib.save(img, path)


# ──────────────────────────────────────────────────────────────────────────────
# Core merge logic
# ──────────────────────────────────────────────────────────────────────────────

def merge_complex(amp1_img, pha1_img,
                  amp2_img, pha2_img,
                  orientation="axial",
                  interp_order=1):
    """
    Resample Z1 and Z2 into a canonical-orientation bounding grid and return:
        (out_affine, abs(Z1+Z2), angle(Z1+Z2))

    Complex resampling strategy
    ---------------------------
    Phase fields have ±π wrap-around discontinuities that break interpolation.
    We convert to Cartesian (real + imag) *before* resampling, interpolate
    the smooth components, then reconstruct the complex number afterwards.
    """
    # ── 1. Compute bounding box in the chosen canonical orientation ───────────
    print(f"Orientation: {orientation.upper()}")
    print("Computing bounding box …")
    out_affine, out_shape = compute_bounding_affine(
        [amp1_img, amp2_img],
        orientation=orientation,
        ref_img=amp1_img,
    )
    print(f"  Output shape : {out_shape}")
    print(f"  Output affine:\n{out_affine}")

    # ── 2. Convert each pair to real/imag, resample, reconstruct complex ──────
    def resample_pair_as_complex(amp_img, pha_img, label):
        print(f"Converting pair {label} to real/imag …")
        amp  = np.asarray(amp_img.dataobj, dtype=np.float64)
        pha  = np.asarray(pha_img.dataobj, dtype=np.float64)
        real = amp * np.cos(pha)
        imag = amp * np.sin(pha)

        real_img = nib.Nifti1Image(real, amp_img.affine)
        imag_img = nib.Nifti1Image(imag, amp_img.affine)

        print(f"Resampling pair {label} real component …")
        real_r = resample_into_target(real_img, out_affine, out_shape,
                                      order=interp_order)
        print(f"Resampling pair {label} imag component …")
        imag_r = resample_into_target(imag_img, out_affine, out_shape,
                                      order=interp_order)
        return real_r + 1j * imag_r

    Z1 = resample_pair_as_complex(amp1_img, pha1_img, "1")
    Z2 = resample_pair_as_complex(amp2_img, pha2_img, "2")

    # ── 3. Sum and extract amplitude / phase ──────────────────────────────────
    print("Summing complex fields …")
    Z_sum = Z1 + Z2

    out_amp   = np.abs(Z_sum)
    out_phase = np.angle(Z_sum)   # radians in [-π, π]

    return out_affine, out_amp, out_phase


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine two complex NIfTI datasets (each as amplitude+phase pair) "
            "into |Z1+Z2| and angle(Z1+Z2) output files, with the result "
            "aligned to a canonical anatomical orientation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Orientation conventions (RAS world space):
  axial     : i→R, j→A, k→S  — standard neurological axial
  coronal   : i→R, j→S, k→A  — rows L/R, slices anterior/posterior
  sagittal  : i→A, j→S, k→R  — rows front/back, slices left/right

Examples:
  # Axial output (default)
  python merge_nifti_complex.py A.nii B.nii C.nii D.nii amp_out.nii phase_out.nii

  # Coronal output with cubic interpolation
  python merge_nifti_complex.py A.nii B.nii C.nii D.nii amp_out.nii phase_out.nii \\
      --orientation coronal --interp 3
        """
    )
    parser.add_argument("amp1",      help="Amplitude of Z1  (file A)")
    parser.add_argument("phase1",    help="Phase of Z1      (file B, radians)")
    parser.add_argument("amp2",      help="Amplitude of Z2  (file C)")
    parser.add_argument("phase2",    help="Phase of Z2      (file D, radians)")
    parser.add_argument("out_amp",   help="Output amplitude  |Z1+Z2|")
    parser.add_argument("out_phase", help="Output phase      angle(Z1+Z2)  [radians]")
    parser.add_argument(
        "--orientation", default="coronal",
        choices=list(ORIENTATION_ROTATION),
        help="Canonical orientation of the output grid (default: coronal)"
    )
    parser.add_argument(
        "--interp", type=int, default=1, choices=[0, 1, 3],
        help="Interpolation order: 0=nearest, 1=linear (default), 3=cubic"
    )
    args = parser.parse_args()

    def load(path, label):
        print(f"Loading {label}: {path} …")
        img = nib.load(path)
        print(f"  shape: {img.shape}   zooms: {img.header.get_zooms()[:3]}")
        return img

    amp1_img = load(args.amp1,   "amp1  (A)")
    pha1_img = load(args.phase1, "phase1(B)")
    amp2_img = load(args.amp2,   "amp2  (C)")
    pha2_img = load(args.phase2, "phase2(D)")

    # Sanity: each amplitude/phase pair must share shape
    for img_a, img_b, pair in [
        (amp1_img, pha1_img, "A/B"),
        (amp2_img, pha2_img, "C/D"),
    ]:
        if img_a.shape[:3] != img_b.shape[:3]:
            print(f"ERROR: pair {pair} shapes differ: "
                  f"{img_a.shape} vs {img_b.shape}", file=sys.stderr)
            sys.exit(1)
        if not np.allclose(img_a.affine, img_b.affine, atol=1e-5):
            print(f"WARNING: pair {pair} affines differ — using amplitude "
                  "affine as reference for that pair.", file=sys.stderr)

    # Warn if voxel sizes differ across pairs
    z1 = np.array(amp1_img.header.get_zooms()[:3])
    z2 = np.array(amp2_img.header.get_zooms()[:3])
    if not np.allclose(z1, z2, rtol=0.05):
        print(f"WARNING: voxel sizes differ > 5 %: {z1} vs {z2}",
              file=sys.stderr)

    out_affine, out_amp, out_phase = merge_complex(
        amp1_img, pha1_img,
        amp2_img, pha2_img,
        orientation=args.orientation,
        interp_order=args.interp,
    )

    print(f"Saving amplitude  → {args.out_amp} …")
    save_nifti(out_amp,   out_affine, amp1_img.header, args.out_amp)

    print(f"Saving phase      → {args.out_phase} …")
    save_nifti(out_phase, out_affine, amp1_img.header, args.out_phase)

    print("Done.")
    print(f"  Output shape     : {out_amp.shape}")
    print(f"  Amplitude range  : [{out_amp.min():.4g}, {out_amp.max():.4g}]")
    print(f"  Phase range (rad): [{out_phase.min():.4g}, {out_phase.max():.4g}]")


if __name__ == "__main__":
    main()