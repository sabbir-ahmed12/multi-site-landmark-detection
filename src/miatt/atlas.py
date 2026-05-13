"""Approach 5 — Multi-atlas affine registration landmark detection.

Strategy
--------
1. Select N labeled siteA subjects as atlases.  SiteA subjects are already in
   ACPC space (AC at physical origin), making them the cleanest reference set.
2. Preprocess atlas and subject T1s identically: cast to float32, intensity
   normalise (clip+rescale to [0,1]), resample to 1 mm isotropic.  No reorientation
   — all images retain their original physical space so that TransformPoint works
   consistently with the fcsv coordinate convention used elsewhere in this codebase.
3. Register each preprocessed atlas (fixed) to the preprocessed subject (moving)
   with affine + Mattes MI.  SimpleITK Execute convention: the returned transform
   maps fixed-space (atlas) coords → moving-space (subject) coords, so applying it
   directly to atlas landmark RAS coords gives predicted subject RAS coords.
4. Fuse N atlas predictions by taking the per-landmark, per-axis median.

Evaluation
----------
Predicted scanner-space landmarks are ACPC-aligned via the predicted AC/PC/LE/RE
and compared to the true ACPC-space landmarks, matching the evaluation protocol
of Approaches 1–4 in pipeline.py.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from miatt.io import iter_subjects, load_fcsv
from miatt.preprocessing import normalize_intensity, resample_to_isotropic

if TYPE_CHECKING:
    import SimpleITK as sitk

ATLAS_SITE = "siteA"
N_ATLASES  = 5
ATLAS_SEED = 7   # deterministic — independent of the eval/train seeds used elsewhere


def _sitk():
    import SimpleITK  # noqa: PLC0415
    return SimpleITK


# ---------------------------------------------------------------------------
# Atlas selection
# ---------------------------------------------------------------------------

def select_atlases(
    data_root: Path,
    n: int = N_ATLASES,
    seed: int = ATLAS_SEED,
    skip_first_n: int = 0,
) -> list[tuple[Path, Path]]:
    """Return n (subject_dir, fcsv_path) pairs from siteA labeled subjects.

    Args:
        data_root:    Root of MIATTFINALEXAMDATA.
        n:            Number of atlases.
        seed:         RNG seed for reproducible selection.
        skip_first_n: Number of subjects to skip from the front of the sorted
                      list (pass the eval holdout size to prevent data leakage
                      when evaluating on siteA itself).
    """
    candidates = [
        (sd, fp)
        for sd, fp in iter_subjects(data_root, ATLAS_SITE, labeled=True)
        if fp is not None
    ]
    candidates = candidates[skip_first_n:]
    return random.Random(seed).sample(candidates, min(n, len(candidates)))


# ---------------------------------------------------------------------------
# Preprocessing for registration
# ---------------------------------------------------------------------------

def prep_for_registration(image: "sitk.Image") -> "sitk.Image":
    """Cast → float32, normalise intensities, resample to 1 mm isotropic.

    Does NOT reorient — preserving the original physical coordinate frame keeps
    TransformPoint consistent with the RAS-as-physical convention used throughout
    this codebase (see registration.py, acpc.py).
    """
    sitk = _sitk()
    img = sitk.Cast(image, sitk.sitkFloat32)
    img = normalize_intensity(img)
    img = resample_to_isotropic(img)
    return img


# ---------------------------------------------------------------------------
# Affine registration
# ---------------------------------------------------------------------------

def register_affine(
    fixed: "sitk.Image",
    moving: "sitk.Image",
    n_iterations: int = 200,
) -> "sitk.Transform":
    """Affine (12 DOF) registration of *moving* to *fixed* using Mattes MI.

    Multi-resolution pyramid: 4×, 2×, 1× with matching Gaussian smoothing.
    Initialised with moment-based centred alignment to handle large initial
    translations between sites.

    Returns a transform that maps *fixed*-space coords to *moving*-space coords
    (SimpleITK Execute(fixed, moving) convention).  Applied to atlas landmark
    coords this gives predicted landmark coords in subject space.
    """
    sitk = _sitk()

    init = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.15)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=n_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(init, inPlace=False)

    return R.Execute(fixed, moving)


# ---------------------------------------------------------------------------
# Landmark transfer
# ---------------------------------------------------------------------------

def transfer_landmarks(
    transform: "sitk.Transform",
    atlas_landmarks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map atlas RAS landmark coords to subject RAS coords via *transform*.

    SimpleITK physical space is LPS, so we convert RAS→LPS before calling
    TransformPoint and convert the output LPS→RAS before returning.
    fcsv files store RAS coords: x_LPS = -x_RAS, y_LPS = -y_RAS, z_LPS = z_RAS.
    """
    result = {}
    for name, ras in atlas_landmarks.items():
        lps_in = (-ras[0], -ras[1], float(ras[2]))      # RAS → LPS
        lps_out = transform.TransformPoint(lps_in)
        result[name] = np.array([-lps_out[0], -lps_out[1], lps_out[2]])  # LPS → RAS
    return result


# ---------------------------------------------------------------------------
# Multi-atlas prediction
# ---------------------------------------------------------------------------

def predict_landmarks_atlas(
    subject_prep: "sitk.Image",
    atlas_images: list["sitk.Image"],
    atlas_landmarks: list[dict[str, np.ndarray]],
    n_iterations: int = 200,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Register each preprocessed atlas to *subject_prep*, fuse by median.

    Args:
        subject_prep:    Subject T1 preprocessed via prep_for_registration().
        atlas_images:    Preprocessed atlas T1s (same preprocessing).
        atlas_landmarks: Corresponding atlas landmark dicts (RAS mm from fcsv).
        n_iterations:    Gradient descent iterations per resolution level.
        verbose:         Print per-atlas progress.

    Returns:
        Predicted landmark dict in subject physical (RAS) space.
    """
    all_preds: dict[str, list[np.ndarray]] = {}

    for i, (atlas_img, atlas_lm) in enumerate(zip(atlas_images, atlas_landmarks)):
        if verbose:
            print(f"      atlas {i + 1}/{len(atlas_images)}", end=" ", flush=True)
        tx = register_affine(atlas_img, subject_prep, n_iterations=n_iterations)
        for name, coord in transfer_landmarks(tx, atlas_lm).items():
            all_preds.setdefault(name, []).append(coord)
        if verbose:
            print("done", flush=True)

    return {
        name: np.median(np.stack(coords), axis=0)
        for name, coords in all_preds.items()
    }


# ---------------------------------------------------------------------------
# Post-processing corrections
# ---------------------------------------------------------------------------

def enforce_eye_symmetry(
    landmarks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Enforce bilateral symmetry for LE/RE in ACPC space.

    In ACPC RAS space the mid-sagittal plane lies at x=0, so LE and RE should
    have equal and opposite x-coordinates and the same z-coordinate (same
    axial level).  Predicted landmarks often violate this due to registration
    error.

    Modifies only x (lateral) and z (axial) of LE and RE; y (A/P) is left
    unchanged as it is not constrained by bilateral symmetry.

    Args:
        landmarks: Predicted landmark dict in ACPC RAS space.

    Returns:
        Copy of landmarks with LE/RE x and z corrected.
    """
    out = {k: v.copy() for k, v in landmarks.items()}
    if "LE" not in out or "RE" not in out:
        return out
    le, re = out["LE"], out["RE"]
    # Enforce equal lateral distance from mid-sagittal (x=0)
    x_half = (abs(float(le[0])) + abs(float(re[0]))) / 2
    # LE is the subject's left eye → negative x in RAS
    if le[0] <= re[0]:
        out["LE"][0] = -x_half
        out["RE"][0] =  x_half
    else:
        out["LE"][0] =  x_half
        out["RE"][0] = -x_half
    # Enforce same superior/inferior level
    out["LE"][2] = out["RE"][2] = (float(le[2]) + float(re[2])) / 2
    return out


# ---------------------------------------------------------------------------
# QC visualisation
# ---------------------------------------------------------------------------

def visualize_predictions(
    t1_image: "sitk.Image",
    predicted: dict[str, np.ndarray],
    true: dict[str, np.ndarray] | None = None,
    landmarks_to_show: list[str] | None = None,
) -> "object":
    """Return a matplotlib Figure showing T1 slices at AC with landmark overlays.

    Plots axial / coronal / sagittal slices at the predicted AC position.
    Predicted landmarks are shown as red circles; true landmarks (if provided)
    as green crosses.  Only landmarks visible in the centre slice ±10 voxels
    are plotted per view.

    Args:
        t1_image:          Raw (unpreprocessed) subject T1 image.
        predicted:         Predicted landmark dict (RAS mm).
        true:              Ground-truth landmark dict (RAS mm), optional.
        landmarks_to_show: Subset of landmark names.  Defaults to AC/PC/LE/RE.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    sitk = _sitk()
    if landmarks_to_show is None:
        landmarks_to_show = ["AC", "PC", "LE", "RE"]

    arr = sitk.GetArrayFromImage(t1_image).astype(np.float32)  # (Z, Y, X)

    def ras_to_vox(ras):
        idx = t1_image.TransformPhysicalPointToIndex(ras.tolist())
        return idx[2], idx[1], idx[0]   # (iz, iy, ix) → array shape (Z,Y,X)

    ac_pred = predicted.get("AC", np.zeros(3))
    iz0, iy0, ix0 = ras_to_vox(ac_pred)
    iz0 = int(np.clip(iz0, 0, arr.shape[0] - 1))
    iy0 = int(np.clip(iy0, 0, arr.shape[1] - 1))
    ix0 = int(np.clip(ix0, 0, arr.shape[2] - 1))

    views = [
        ("Axial",    arr[iz0, :, :],   1, 2),   # slice at z=iz0, axes: y→row, x→col
        ("Coronal",  arr[:, iy0, :],   0, 2),   # slice at y=iy0, axes: z→row, x→col
        ("Sagittal", arr[:, :, ix0],   0, 1),   # slice at x=ix0, axes: z→row, y→col
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#111")
    fig.suptitle("Atlas predictions (red=pred, green=GT)", color="white", fontsize=10)

    SLAB = 10   # voxels — only plot landmarks within this distance of the slice

    for ax, (title, sl, row_ax, col_ax) in zip(axes, views):
        ax.imshow(sl, cmap="gray", origin="lower",
                  vmin=np.percentile(sl, 1), vmax=np.percentile(sl, 99))
        ax.set_title(title, color="white", fontsize=9)
        ax.axis("off")

        slice_coord = [ix0, iy0, iz0]   # current slice position in x,y,z voxels

        for name in landmarks_to_show:
            if name not in predicted:
                continue
            iz_l, iy_l, ix_l = ras_to_vox(predicted[name])
            vox_all = [ix_l, iy_l, iz_l]
            # Only draw if within slab of the slice plane
            slice_axis = 3 - row_ax - col_ax    # the axis perpendicular to this view
            # For axial: slice_axis would be 2 (z), for coronal 1 (y), sagittal 0 (x)
            # vox_all[0]=x, vox_all[1]=y, vox_all[2]=z
            perp_axes = {(1,2): 0, (0,2): 1, (0,1): 2}
            p_ax = perp_axes.get((min(row_ax,col_ax), max(row_ax,col_ax)), 0)
            if abs(vox_all[p_ax] - slice_coord[p_ax]) > SLAB:
                continue
            r = vox_all[row_ax]
            c = vox_all[col_ax]
            ax.plot(c, r, "ro", markersize=6, markeredgewidth=1.5,
                    markeredgecolor="white", alpha=0.85)
            ax.annotate(name, (c, r), color="red", fontsize=6,
                        xytext=(3, 3), textcoords="offset points")

            if true and name in true:
                iz_t, iy_t, ix_t = ras_to_vox(true[name])
                vox_t = [ix_t, iy_t, iz_t]
                if abs(vox_t[p_ax] - slice_coord[p_ax]) <= SLAB:
                    rt = vox_t[row_ax]
                    ct = vox_t[col_ax]
                    ax.plot(ct, rt, "g+", markersize=8, markeredgewidth=2, alpha=0.85)

    plt.tight_layout()
    return fig
