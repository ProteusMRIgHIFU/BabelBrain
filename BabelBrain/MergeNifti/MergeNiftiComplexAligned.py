"""
merge_nifti_complex.py
----------------------
Combine N complex NIfTI datasets — each represented as an amplitude/phase
pair — into a single summed complex field, saving the result as separate
amplitude and phase NIfTI files.

Given N pairs:
    Zn = amp_n * exp(i * phase_n)

Outputs:
    |Z1 + Z2 + ... + ZN|       → amplitude output
    angle(Z1 + Z2 + ... + ZN)  → phase output  (radians, in [-π, π])

All inputs and outputs are specified via a YAML config file.

Requirements:
    pip install nibabel numpy scipy pyyaml

Usage:
    python merge_nifti_complex.py config.yaml

--- YAML format -----------------------------------------------------------------

orientation: axial          # axial | coronal | sagittal  (default: axial)
interp: 1                   # 0=nearest, 1=linear (default), 3=cubic

pairs:
  - amp:   path/to/amp1.nii.gz
    phase: path/to/phase1.nii.gz
  - amp:   path/to/amp2.nii.gz
    phase: path/to/phase2.nii.gz
  # ... as many pairs as needed

output:
  amp:   path/to/out_amplitude.nii.gz
  phase: path/to/out_phase.nii.gz

---------------------------------------------------------------------------------
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
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
    "axial":    np.array([[ 1,  0,  0],
                           [ 0,  1,  0],
                           [ 0,  0,  1]], dtype=float).T,

    "coronal":  np.array([[ 1,  0,  0],
                           [ 0,  0,  1],
                           [ 0,  1,  0]], dtype=float).T,

    "sagittal": np.array([[ 0,  1,  0],
                           [ 0,  0,  1],
                           [ 1,  0,  0]], dtype=float).T,
}


# ──────────────────────────────────────────────────────────────────────────────
# YAML config loading
# ──────────────────────────────────────────────────────────────────────────────

def load_config(yaml_path):
    """
    Load and validate the YAML config file.
    Returns a dict with keys: orientation, interp, pairs, output.
    All file paths are resolved relative to the YAML file's directory.
    """
    yaml_path = Path(yaml_path).resolve()
    base_dir  = yaml_path.parent

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # ── Defaults ──────────────────────────────────────────────────────────────
    cfg.setdefault("orientation", "axial")
    cfg.setdefault("interp", 1)

    # ── Validate orientation ──────────────────────────────────────────────────
    if cfg["orientation"] not in ORIENTATION_ROTATION:
        print(f"ERROR: orientation must be one of "
              f"{list(ORIENTATION_ROTATION)}, got '{cfg['orientation']}'",
              file=sys.stderr)
        sys.exit(1)

    # ── Validate interp ───────────────────────────────────────────────────────
    if cfg["interp"] not in (0, 1, 3):
        print(f"ERROR: interp must be 0, 1, or 3, got {cfg['interp']}",
              file=sys.stderr)
        sys.exit(1)

    # ── Validate pairs ────────────────────────────────────────────────────────
    if "pairs" not in cfg or not isinstance(cfg["pairs"], list) or len(cfg["pairs"]) < 2:
        print("ERROR: 'pairs' must be a list of at least 2 amp/phase entries.",
              file=sys.stderr)
        sys.exit(1)

    resolved_pairs = []
    for i, pair in enumerate(cfg["pairs"]):
        for key in ("amp", "phase"):
            if key not in pair:
                print(f"ERROR: pair {i+1} is missing '{key}' key.", file=sys.stderr)
                sys.exit(1)
        resolved_pairs.append({
            "amp":   _resolve(pair["amp"],   base_dir, f"pair {i+1} amp"),
            "phase": _resolve(pair["phase"], base_dir, f"pair {i+1} phase"),
        })
    cfg["pairs"] = resolved_pairs

    # ── Validate output ───────────────────────────────────────────────────────
    if "output" not in cfg:
        print("ERROR: 'output' section missing from config.", file=sys.stderr)
        sys.exit(1)
    for key in ("amp", "phase"):
        if key not in cfg["output"]:
            print(f"ERROR: 'output.{key}' missing from config.", file=sys.stderr)
            sys.exit(1)
        # Resolve output paths relative to YAML location too
        cfg["output"][key] = base_dir / cfg["output"][key]

    return cfg


def _resolve(path_str, base_dir, label):
    """Resolve a path relative to base_dir; abort if file not found."""
    p = Path(path_str)
    if not p.is_absolute():
        p = base_dir / p
    if not p.exists():
        print(f"ERROR: {label} file not found: {p}", file=sys.stderr)
        sys.exit(1)
    return p


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
    Build a pure canonical-orientation affine (rotation + mean voxel size
    from ref_img, no translation yet).
    """
    vox_size = float(np.mean(ref_img.header.get_zooms()[:3]))
    R = ORIENTATION_ROTATION[orientation]
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = R * vox_size
    return affine


def compute_bounding_affine(amp_imgs, orientation="axial"):
    """
    Return (out_affine, out_shape): the tightest bounding box in the chosen
    canonical orientation that encloses all volumes in amp_imgs.

    amp_imgs : list of nibabel images (one per pair) used for extent only.
               The voxel size is taken from amp_imgs[0].
    """
    out_affine = build_canonical_affine(amp_imgs[0], orientation=orientation)

    all_corners = np.vstack([volume_corners_world(img) for img in amp_imgs])

    inv_rot = np.linalg.inv(out_affine[:3, :3])
    corners_vox_tmp = (inv_rot @ all_corners.T).T

    min_vox = np.floor(corners_vox_tmp.min(axis=0)).astype(int)
    max_vox = np.ceil( corners_vox_tmp.max(axis=0)).astype(int)

    out_shape  = tuple((max_vox - min_vox + 1).tolist())
    out_affine[:3, 3] = out_affine[:3, :3] @ min_vox

    return out_affine, out_shape


def resample_into_target(src_img, target_affine, target_shape,
                          order=1, cval=0.0):
    """
    Resample src_img into the space defined by target_affine / target_shape.
    Uses backward mapping via scipy.ndimage.affine_transform.
    Supports 3-D and 4-D volumes.
    """
    out2src = np.linalg.inv(src_img.affine) @ target_affine
    matrix  = out2src[:3, :3]
    offset  = out2src[:3,  3]

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
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    img.header.set_zooms(zooms)
    img.header.set_xyzt_units(
        xyz=ref_header.get_xyzt_units()[0],
        t=ref_header.get_xyzt_units()[1],
    )
    nib.save(img, str(path))


# ──────────────────────────────────────────────────────────────────────────────
# Core merge logic
# ──────────────────────────────────────────────────────────────────────────────

def merge_complex(pairs, orientation="axial", interp_order=1):
    """
    Resample all (amp, phase) pairs into a canonical-orientation bounding grid,
    sum the complex fields, and return:
        (out_affine, abs(Z_sum), angle(Z_sum))

    Complex resampling strategy
    ---------------------------
    Phase fields have ±π wrap-around discontinuities that break interpolation.
    We convert to Cartesian (real + imag) *before* resampling, interpolate
    the smooth components, then reconstruct the complex number afterwards.
    """
    n = len(pairs)

    # ── 1. Compute bounding box over all amplitude images ────────────────────
    amp_imgs = [p["amp_img"] for p in pairs]

    print(f"Orientation  : {orientation.upper()}")
    print(f"Pairs        : {n}")
    print("Computing bounding box …")
    out_affine, out_shape = compute_bounding_affine(amp_imgs, orientation)
    print(f"  Output shape : {out_shape}")
    print(f"  Output affine:\n{out_affine}")

    # ── 2. Accumulate complex sum ─────────────────────────────────────────────
    Z_sum = np.zeros(out_shape, dtype=complex)

    for idx, pair in enumerate(pairs, start=1):
        amp_img = pair["amp_img"]
        pha_img = pair["pha_img"]

        print(f"\n[{idx}/{n}] Converting to real/imag …")
        amp  = np.asarray(amp_img.dataobj, dtype=np.float64)
        pha  = np.asarray(pha_img.dataobj, dtype=np.float64)
        real = amp * np.cos(pha)
        imag = amp * np.sin(pha)

        real_nii = nib.Nifti1Image(real, amp_img.affine)
        imag_nii = nib.Nifti1Image(imag, amp_img.affine)

        print(f"[{idx}/{n}] Resampling real component …")
        real_r = resample_into_target(real_nii, out_affine, out_shape,
                                      order=interp_order)
        print(f"[{idx}/{n}] Resampling imag component …")
        imag_r = resample_into_target(imag_nii, out_affine, out_shape,
                                      order=interp_order)

        Z_sum += real_r + 1j * imag_r

    # ── 3. Extract amplitude / phase ──────────────────────────────────────────
    print("\nExtracting amplitude and phase …")
    out_amp   = np.abs(Z_sum)
    out_phase = np.angle(Z_sum)   # radians in [-π, π]

    return out_affine, out_amp, out_phase


def do_complex_merge(cfg):
    orientation  = cfg["orientation"]
    interp_order = cfg["interp"]
    out_amp_path = cfg["output"]["amp"]
    out_pha_path = cfg["output"].get("phase") #if we do not get phase, we do not save it

    # ── Load all pairs ────────────────────────────────────────────────────────
    loaded_pairs = []
    ref_header   = None

    for i, pair in enumerate(cfg["pairs"], start=1):
        print(f"Loading pair {i}: amp={pair['amp'].name}  "
              f"phase={pair['phase'].name}")

        amp_img = nib.load(pair["amp"])
        pha_img = nib.load(pair["phase"])

        print(f"  amp   shape: {amp_img.shape}  "
              f"zooms: {amp_img.header.get_zooms()[:3]}")
        print(f"  phase shape: {pha_img.shape}  "
              f"zooms: {pha_img.header.get_zooms()[:3]}")

        # Validate pair consistency
        if amp_img.shape[:3] != pha_img.shape[:3]:
            print(f"ERROR: pair {i} amp/phase shapes differ: "
                  f"{amp_img.shape} vs {pha_img.shape}", file=sys.stderr)
            sys.exit(1)
        if not np.allclose(amp_img.affine, pha_img.affine, atol=1e-5):
            print(f"WARNING: pair {i} amp/phase affines differ — "
                  "using amplitude affine as reference.", file=sys.stderr)

        if ref_header is None:
            ref_header = amp_img.header

        # Cross-pair voxel size check
        if i > 1:
            z0 = np.array(loaded_pairs[0]["amp_img"].header.get_zooms()[:3])
            zi = np.array(amp_img.header.get_zooms()[:3])
            if not np.allclose(z0, zi, rtol=0.05):
                print(f"WARNING: pair {i} voxel sizes {zi} differ > 5% "
                      f"from pair 1 {z0}.", file=sys.stderr)

        loaded_pairs.append({"amp_img": amp_img, "pha_img": pha_img})

    # ── Merge ─────────────────────────────────────────────────────────────────
    print()
    out_affine, out_amp, out_phase = merge_complex(
        loaded_pairs,
        orientation=orientation,
        interp_order=interp_order,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_amp_path.parent.mkdir(parents=True, exist_ok=True)
    if out_pha_path:
        out_pha_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving amplitude  → {out_amp_path} …")
    save_nifti(out_amp,   out_affine, ref_header, out_amp_path)

    if out_pha_path:
        print(f"Saving phase      → {out_pha_path} …")
        save_nifti(out_phase, out_affine, ref_header, out_pha_path)

    print("\nDone.")
    print(f"  Output shape     : {out_amp.shape}")
    print(f"  Amplitude range  : [{out_amp.min():.4g}, {out_amp.max():.4g}]")
    print(f"  Phase range (rad): [{out_phase.min():.4g}, {out_phase.max():.4g}]")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge N complex NIfTI amplitude/phase pairs into |ΣZn| and "
            "angle(ΣZn), aligned to a canonical anatomical orientation. "
            "All inputs and outputs are specified in a YAML config file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example YAML (config.yaml):
──────────────────────────────────────────────
orientation: axial       # axial | coronal | sagittal
interp: 1                # 0=nearest  1=linear  3=cubic

pairs:
  - amp:   inputs/amp1.nii.gz
    phase: inputs/phase1.nii.gz
  - amp:   inputs/amp2.nii.gz
    phase: inputs/phase2.nii.gz
  - amp:   inputs/amp3.nii.gz
    phase: inputs/phase3.nii.gz

output:
  amp:   results/out_amplitude.nii.gz
  phase: results/out_phase.nii.gz
──────────────────────────────────────────────
Paths are resolved relative to the YAML file's directory.
        """
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    do_complex_merge(cfg)

if __name__ == "__main__":
    main()
