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

The bounding-box / resampling logic is identical to merge_nifti.py:
the smallest possible output volume in the reference frame of file_A/B
is computed, both complex fields are resampled into it, then summed.

Requirements:
    pip install nibabel numpy scipy

Usage:
    python merge_nifti_complex.py \\
        amp1.nii.gz phase1.nii.gz \\
        amp2.nii.gz phase2.nii.gz \\
        out_amp.nii.gz out_phase.nii.gz

Optional flags:
    --interp 0/1/3          Nearest / linear (default) / cubic interpolation
    --use-diagonal-affine   Strip oblique rotation → axis-aligned output
"""

import argparse
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers  (shared with merge_nifti.py)
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


def build_output_affine(ref_img, use_diagonal=False):
    if use_diagonal:
        voxel_sizes = ref_img.header.get_zooms()[:3]
        signs = np.sign(np.diag(ref_img.affine)[:3])
        signs[signs == 0] = 1
        aff = np.diag([*signs * voxel_sizes, 1]).astype(float)
        return aff
    else:
        return ref_img.affine.copy().astype(float)


def compute_bounding_affine(ref_img, other_img, use_diagonal=False):
    """
    Return (out_affine, out_shape): the tightest output grid (in the
    reference frame of ref_img) that encloses both volumes.
    """
    out_affine = build_output_affine(ref_img, use_diagonal=use_diagonal)

    corners_world = np.vstack([
        volume_corners_world(ref_img),
        volume_corners_world(other_img),
    ])

    inv_rot = np.linalg.inv(out_affine[:3, :3])
    corners_vox_tmp = (inv_rot @ corners_world.T).T

    min_vox = np.floor(corners_vox_tmp.min(axis=0)).astype(int)
    max_vox = np.ceil( corners_vox_tmp.max(axis=0)).astype(int)

    out_shape = tuple((max_vox - min_vox + 1).tolist())
    out_affine[:3, 3] = out_affine[:3, :3] @ min_vox

    return out_affine, out_shape


def resample_into_target(src_img, target_affine, target_shape,
                          order=1, cval=0.0):
    """
    Resample src_img into the space defined by target_affine / target_shape.
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
    img.header.set_zooms(ref_header.get_zooms()[:3])
    img.header.set_xyzt_units(
        xyz=ref_header.get_xyzt_units()[0],
        t=ref_header.get_xyzt_units()[1],
    )
    nib.save(img, path)


# ──────────────────────────────────────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────────────────────────────────────

def merge_complex(amp1_img, pha1_img,
                  amp2_img, pha2_img,
                  use_diagonal_affine=False,
                  interp_order=1):
    """
    Resample Z1 and Z2 into a common bounding grid and return:
        (out_affine, abs(Z1+Z2), angle(Z1+Z2))

    Amplitude and phase of each pair share the same affine/shape, so we
    only need to resample two datasets (one per pair) — the bounding box
    is computed from the amplitude images as proxies for the pair geometry.

    Complex resampling strategy
    ---------------------------
    Resampling a phase field directly causes wrap-around artefacts at ±π
    discontinuities.  Instead we convert each pair to real + imaginary
    parts *before* resampling, interpolate those smooth fields, then
    reconstruct the complex number.  This is the standard approach used
    in MRI phase processing.
    """
    # ── 1. Compute common output grid ────────────────────────────────────────
    print("Computing bounding box …")
    out_affine, out_shape = compute_bounding_affine(
        amp1_img, amp2_img, use_diagonal=use_diagonal_affine
    )
    print(f"  Output shape : {out_shape}")
    print(f"  Output affine:\n{out_affine}")

    # ── 2. Convert each pair to real/imag, then resample ─────────────────────
    def resample_pair_as_complex(amp_img, pha_img, label):
        print(f"Converting pair {label} to real/imag …")
        amp = np.asarray(amp_img.dataobj, dtype=np.float64)
        pha = np.asarray(pha_img.dataobj, dtype=np.float64)
        real = amp * np.cos(pha)
        imag = amp * np.sin(pha)

        # Wrap in temporary nibabel images so we can reuse resample_into_target
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
            "into |Z1+Z2| and angle(Z1+Z2) output files."
        )
    )
    parser.add_argument("amp1",       help="Amplitude of Z1  (file A)")
    parser.add_argument("phase1",     help="Phase of Z1      (file B, radians)")
    parser.add_argument("amp2",       help="Amplitude of Z2  (file C)")
    parser.add_argument("phase2",     help="Phase of Z2      (file D, radians)")
    parser.add_argument("out_amp",    help="Output amplitude  |Z1+Z2|")
    parser.add_argument("out_phase",  help="Output phase      angle(Z1+Z2)  [radians]")
    parser.add_argument(
        "--interp", type=int, default=1, choices=[0, 1, 3],
        help="Interpolation order: 0=nearest, 1=linear (default), 3=cubic"
    )
    parser.add_argument(
        "--use-diagonal-affine", action="store_true",
        help="Strip rotation from output affine → axis-aligned output"
    )
    args = parser.parse_args()

    def load(path, label):
        print(f"Loading {label}: {path} …")
        img = nib.load(path)
        print(f"  shape: {img.shape}   zooms: {img.header.get_zooms()[:3]}")
        return img

    amp1_img  = load(args.amp1,   "amp1  (A)")
    pha1_img  = load(args.phase1, "phase1(B)")
    amp2_img  = load(args.amp2,   "amp2  (C)")
    pha2_img  = load(args.phase2, "phase2(D)")

    # Sanity: pairs must share affine & shape
    for (img_a, img_b, pair) in [
        (amp1_img, pha1_img, "A/B"),
        (amp2_img, pha2_img, "C/D"),
    ]:
        if img_a.shape[:3] != img_b.shape[:3]:
            print(f"ERROR: pair {pair} shapes differ: "
                  f"{img_a.shape} vs {img_b.shape}", file=sys.stderr)
            sys.exit(1)
        if not np.allclose(img_a.affine, img_b.affine, atol=1e-5):
            print(f"WARNING: pair {pair} affines differ — using amplitude "
                  f"affine as reference for that pair.", file=sys.stderr)

    # Warn if voxel sizes differ across pairs
    z1 = np.array(amp1_img.header.get_zooms()[:3])
    z2 = np.array(amp2_img.header.get_zooms()[:3])
    if not np.allclose(z1, z2, rtol=0.05):
        print(f"WARNING: voxel sizes differ > 5%: {z1} vs {z2}",
              file=sys.stderr)

    out_affine, out_amp, out_phase = merge_complex(
        amp1_img, pha1_img,
        amp2_img, pha2_img,
        use_diagonal_affine=args.use_diagonal_affine,
        interp_order=args.interp,
    )

    print(f"Saving amplitude  → {args.out_amp} …")
    save_nifti(out_amp.astype(np.float32),   out_affine, amp1_img.header, args.out_amp)

    print(f"Saving phase      → {args.out_phase} …")
    save_nifti(out_phase.astype(np.float32), out_affine, amp1_img.header, args.out_phase)

    print("Done.")
    print(f"  Output shape : {out_amp.shape}")
    print(f"  Amplitude range  : [{out_amp.min():.4g}, {out_amp.max():.4g}]")
    print(f"  Phase range (rad): [{out_phase.min():.4g}, {out_phase.max():.4g}]")


if __name__ == "__main__":
    main()