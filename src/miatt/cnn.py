"""Approach 4 — 3D CNN direct coordinate regression.

Architecture
------------
Simple 5-block 3D CNN encoder (stride-2 convolutions) + adaptive average
pool + 2-layer fully-connected head that outputs 51×3 ACPC coordinates.
Input: single-channel ACPC-resampled T1 on the fixed 2 mm grid (1×96×101×101).

Training strategy
-----------------
All 6 sites are pooled: the ACPC transform normalises coordinate systems so
a single cross-site model is feasible.  Per-subject ACPC T1 volumes are
cached as .npz files to avoid repeated resampling during training.  The
80/20 deterministic split (first 20% held out for evaluation, same as
Approaches 1-3) is respected per site so results are directly comparable.

Loss: smooth L1 (Huber δ=1) on raw mm coordinates.
Augmentation: mild intensity jitter (scale ±10%, shift ±5%) and a coin-flip
left-right reflection (x-axis in ACPC space) with the corresponding landmark
x-coordinate negation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


# Ordered list of all 51 landmark labels (alphabetical, matches training targets)
LANDMARK_LABELS: list[str] = sorted([
    "AC", "BPons", "CM", "LE", "PC", "RE", "RP", "RP_front", "SMV", "VN4",
    "callosum_left", "callosum_right", "dens_axis", "genu",
    "l_caud_head", "l_corp", "l_front_pole", "l_inner_corpus",
    "l_lat_ext", "l_occ_pole", "l_prim_ext", "l_sup_ext", "l_temp_pole",
    "lat_left", "lat_right", "lat_ven_left", "lat_ven_right",
    "left_cereb", "left_lateral_inner_ear",
    "m_ax_inf", "m_ax_sup", "mid_basel", "mid_lat",
    "mid_prim_inf", "mid_prim_sup", "mid_sup", "optic_chiasm",
    "r_caud_head", "r_corp", "r_front_pole", "r_inner_corpus",
    "r_lat_ext", "r_occ_pole", "r_prim_ext", "r_sup_ext", "r_temp_pole",
    "right_lateral_inner_ear", "rostrum", "rostrum_front",
    "top_left", "top_right",
])
N_LANDMARKS = len(LANDMARK_LABELS)  # 51

# Fixed ACPC grid constants (must match registration.py)
_TEMPLATE_SPACING = (2.0, 2.0, 2.0)
_TEMPLATE_SIZE = (101, 101, 96)   # (nx, ny, nz)
_TEMPLATE_ORIGIN = (-100.0, -120.0, -90.0)

# Coordinate normalisation: divide raw mm by this constant before training so
# that all targets live in roughly [-1, +1].  The ACPC grid spans ±100-120 mm
# in each axis, so 120 mm gives coordinates in [-1, 1].
COORD_SCALE: float = 120.0


def _sitk():
    import SimpleITK  # noqa: PLC0415
    return SimpleITK


def _torch():
    import torch  # noqa: PLC0415
    return torch


# ---------------------------------------------------------------------------
# Cache building
# ---------------------------------------------------------------------------

def build_cnn_cache(
    data_root: Path,
    site: str,
    cache_dir: Path,
    labeled: bool = True,
) -> list[Path]:
    """Resample T1s and save ACPC volumes + landmark arrays to .npz cache files.

    For labeled subjects, also stores the ACPC landmark coordinates as a
    (N_LANDMARKS, 3) float32 array in 'landmarks' key.  Unlabeled subjects
    only store 'volume'.

    Returns list of cache file paths (one per subject).
    """
    from miatt.acpc import compute_acpc_transform, transform_landmarks  # noqa: PLC0415
    from miatt.io import iter_subjects, load_fcsv  # noqa: PLC0415
    from miatt.preprocessing import normalize_intensity  # noqa: PLC0415
    from miatt.registration import resample_to_acpc_space  # noqa: PLC0415

    sitk = _sitk()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for subject_dir, fcsv_path in iter_subjects(data_root, site, labeled=labeled):
        tag = f"{site}_{'unlabeled_' if not labeled else ''}{subject_dir.name}"
        npz_path = cache_dir / f"{tag}.npz"

        if npz_path.exists():
            paths.append(npz_path)
            continue

        t1_path = subject_dir / f"t1_{site}.nii.gz"
        if not t1_path.exists():
            continue

        t1 = sitk.ReadImage(str(t1_path))
        t1_norm = normalize_intensity(sitk.Cast(t1, sitk.sitkFloat32))

        if labeled and fcsv_path is not None:
            lm = load_fcsv(fcsv_path)
            try:
                T = compute_acpc_transform(
                    lm["AC"], lm["PC"], lm["LE"], lm["RE"]
                )
            except (KeyError, ValueError):
                continue
            acpc_vol = resample_to_acpc_space(t1_norm, T)
            acpc_lm = transform_landmarks(T, lm)
            # Build ordered coordinate array
            coords = np.zeros((N_LANDMARKS, 3), dtype=np.float32)
            for i, label in enumerate(LANDMARK_LABELS):
                if label in acpc_lm:
                    coords[i] = acpc_lm[label].astype(np.float32)
            arr = sitk.GetArrayFromImage(acpc_vol).astype(np.float32)  # (nz, ny, nx)
            np.savez_compressed(npz_path, volume=arr, landmarks=coords)
        else:
            # For unlabeled subjects we still try to resample using registration
            # transform.  If not available, skip.
            continue

        paths.append(npz_path)

    return paths


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _make_dataset(cache_files: list[Path]):
    """Return a torch Dataset that loads from .npz cache files."""
    import torch  # noqa: PLC0415
    from torch.utils.data import Dataset  # noqa: PLC0415

    class _Ds(Dataset):
        def __init__(self, files: list[Path], augment: bool = False) -> None:
            self.files = files
            self.augment = augment

        def __len__(self) -> int:
            return len(self.files)

        def __getitem__(self, idx: int):
            data = np.load(self.files[idx])
            vol = data["volume"].copy()  # (nz, ny, nx)
            coords = data["landmarks"].copy()  # (51, 3)

            if self.augment:
                # Intensity jitter
                scale = np.random.uniform(0.9, 1.1)
                shift = np.random.uniform(-0.05, 0.05)
                vol = vol * scale + shift
                vol = np.clip(vol, 0.0, 1.2)

                # Left-right flip (x-axis in ACPC space)
                if np.random.rand() < 0.5:
                    vol = vol[:, :, ::-1].copy()
                    # ACPC grid: x goes from -100 to +100 mm
                    # After flip, physical x → -x
                    coords[:, 0] = -coords[:, 0]

            # Normalise coordinates to [-1, 1] for stable training
            coords_norm = (coords / COORD_SCALE).astype(np.float32)

            # Shape: (1, nz, ny, nx)
            vol_t = torch.from_numpy(vol[np.newaxis]).float()
            coords_t = torch.from_numpy(coords_norm).float()
            return vol_t, coords_t

    return _Ds(cache_files)


def _make_aug_dataset(cache_files: list[Path]):
    import torch  # noqa: PLC0415
    from torch.utils.data import Dataset  # noqa: PLC0415

    class _Ds(Dataset):
        def __init__(self, files: list[Path]) -> None:
            self.files = files

        def __len__(self) -> int:
            return len(self.files)

        def __getitem__(self, idx: int):
            data = np.load(self.files[idx])
            vol = data["volume"].copy()
            coords = data["landmarks"].copy()

            scale = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(-0.05, 0.05)
            vol = np.clip(vol * scale + shift, 0.0, 1.2)

            if np.random.rand() < 0.5:
                vol = vol[:, :, ::-1].copy()
                coords[:, 0] = -coords[:, 0]

            # Normalise coordinates to [-1, 1] for stable training
            coords_norm = (coords / COORD_SCALE).astype(np.float32)

            import torch  # noqa: PLC0415
            return torch.from_numpy(vol[np.newaxis]).float(), torch.from_numpy(coords_norm).float()

    return _Ds(cache_files)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> "nn.Module":
    """5-block 3D CNN encoder → global avg pool → FC → 51×3 coordinates."""
    import torch.nn as nn  # noqa: PLC0415

    def _block(c_in: int, c_out: int, stride: int = 2) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_out, c_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(c_out),
            nn.ReLU(inplace=True),
        )

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Sequential(
                _block(1, 24),     # 96×101×101 → 48×51×51
                _block(24, 48),    # → 24×26×26
                _block(48, 96),    # → 12×13×13
                _block(96, 192),   # → 6×7×7
                _block(192, 384),  # → 3×4×4
            )
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.head = nn.Sequential(
                nn.Linear(384, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, N_LANDMARKS * 3),
                nn.Tanh(),  # outputs in (-1, 1) — matches normalised coordinate range
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.enc(x)
            x = self.pool(x).flatten(1)
            x = self.head(x)
            return x.view(-1, N_LANDMARKS, 3)

    return _Net()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_cnn(
    train_files: list[Path],
    val_files: list[Path],
    model_path: Path,
    n_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 3e-4,
    device: str = "cuda",
) -> dict[str, list[float]]:
    """Train the 3D CNN and save the best checkpoint.

    Returns dict with 'train_loss' and 'val_loss' per epoch.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    train_ds = _make_aug_dataset(train_files)
    val_ds = _make_dataset(val_files)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model = build_model().to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.SmoothL1Loss(beta=1.0)

    best_val = float("inf")
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for vols, coords in train_loader:
            vols, coords = vols.to(dev), coords.to(dev)
            optimizer.zero_grad()
            pred = model(vols)
            loss = criterion(pred, coords)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for vols, coords in val_loader:
                vols, coords = vols.to(dev), coords.to(dev)
                pred = model(vols)
                val_losses.append(criterion(pred, coords).item())

        t_loss = float(np.mean(train_losses))
        v_loss = float(np.mean(val_losses))
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  epoch {epoch:3d}/{n_epochs}  "
                f"train={t_loss:.4f}  val={v_loss:.4f}  "
                f"lr={lr_now:.2e}  best_val={best_val:.4f}"
            )

    return history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_cnn(
    model_path: Path,
    acpc_volume: np.ndarray,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Run inference and return {label: xyz_mm} predictions in ACPC space.

    Args:
        model_path: Path to saved model state dict.
        acpc_volume: (nz, ny, nx) float32 ACPC T1 volume.
        device: torch device string.

    Returns:
        Dict mapping landmark label → (3,) ACPC coordinate array.
    """
    import torch  # noqa: PLC0415

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = build_model().to(dev)
    model.load_state_dict(torch.load(model_path, map_location=dev, weights_only=True))
    model.eval()

    vol_t = torch.from_numpy(acpc_volume[np.newaxis, np.newaxis]).float().to(dev)
    with torch.no_grad():
        pred = model(vol_t).squeeze(0).cpu().numpy()  # (51, 3) in normalised space

    # Denormalise back to mm
    pred_mm = pred * COORD_SCALE
    return {label: pred_mm[i].astype(float) for i, label in enumerate(LANDMARK_LABELS)}
