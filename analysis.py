"""
Pilot RSA analysis

This script implements a compact, pilot representational similarity analysis (RSA)
pipeline applied to time-resolved EEG from the THINGS-EEG dataset (Subject 09 by
default). 

Key steps contained here:
 - load preprocessed EEG epochs and image metadata
 - average across repetitions to obtain stimulus patterns
 - compute neural RDMs over time (correlation-distance)
 - compute model RDMs (ResNet-50 PCA features)
 - compute RSA timecourses (Spearman correlation between vectorized RDMs)

"""

import os
from typing import Tuple, Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import spearmanr


# -------------------------
# Paths (defaults)
# -------------------------
EEG_TRAIN_PATH = r"..\sub-09\sub-09\preprocessed_eeg_training.npy"
EEG_TEST_PATH = r"..\sub-09\sub-09\preprocessed_eeg_test.npy"
IMG_META_PATH = r"image_metadata.npy"
IMAGE_BASE = r"..\training_images\training_images"
RESNET_PCA_PATH = r"..\resnet50\resnet50\pretrained-True\layers-single\pca_feature_maps_training.npy"

# Figures directory (sibling to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGS_DIR = os.path.join(SCRIPT_DIR, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)


def save_figure(name: str):
    """Save current matplotlib figure to the figs directory as PNG and close it."""
    path = os.path.join(FIGS_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print("Saved figure:", path)
    plt.close()


# -------------------------
# Utilities: RDMs and vectorization
# -------------------------

def upper_tri_vec(R: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(R.shape[0], k=1)
    return R[iu]


def corr_dist_rdm(P: np.ndarray) -> np.ndarray:
    """Compute correlation-distance RDM for rows of P.

    P: (N, D) array where each row is a pattern for one stimulus.
    Returns: (N, N) RDM with zeros on the diagonal.
    """
    P = P - P.mean(axis=1, keepdims=True)
    P = P / (P.std(axis=1, keepdims=True) + 1e-8)
    C = (P @ P.T) / (P.shape[1] - 1)
    C = np.clip(C, -1.0, 1.0)
    R = 1.0 - C
    np.fill_diagonal(R, 0.0)
    return R


# -------------------------
# Data loading and preprocessing
# -------------------------

def load_eeg(eeg_train_path: str = EEG_TRAIN_PATH, eeg_test_path: str = EEG_TEST_PATH) -> Tuple[Dict, Dict]:
    train = np.load(eeg_train_path, allow_pickle=True).item()
    test = np.load(eeg_test_path, allow_pickle=True).item()
    return train, test


def select_images_and_patterns(X_train: np.ndarray, N: int = 200, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select N images and compute average patterns per stimulus.

    Returns:
      sel: (N,) selected indices
      X_sub: (N, reps, ch, time)
      patterns: (N, ch, time)
    """
    rng = np.random.RandomState(seed)
    n_total = X_train.shape[0]
    sel = rng.choice(n_total, size=N, replace=False)
    sel = np.sort(sel)
    X_sub = X_train[sel]
    patterns = X_sub.mean(axis=1)
    return sel, X_sub, patterns


def compute_eeg_rdm_vecs(patterns: np.ndarray) -> np.ndarray:
    """Compute neural RDM upper-triangular vectors for every timepoint.

    patterns: (N, ch, time)
    Returns: eeg_rdm_vecs: (time, rdm_len)
    """
    N, n_ch, n_t = patterns.shape
    rdm_len = N * (N - 1) // 2
    eeg_rdm_vecs = np.zeros((n_t, rdm_len), dtype=np.float32)
    for t in range(n_t):
        P_t = patterns[:, :, t]
        R_t = corr_dist_rdm(P_t)
        eeg_rdm_vecs[t] = upper_tri_vec(R_t)
    return eeg_rdm_vecs


# -------------------------
# Model RDMs
# -------------------------

def build_pixel_model_vec(sel: Sequence[int], img_meta: dict, image_base: str = IMAGE_BASE, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    img_vecs = []
    for idx in sel:
        concept_folder = img_meta["train_img_concepts"][idx]
        fname = img_meta["train_img_files"][idx]
        img_path = os.path.join(image_base, concept_folder, fname)
        img = Image.open(img_path).convert("L")
        img = img.resize(size)
        v = np.asarray(img, dtype=np.float32).reshape(-1)
        img_vecs.append(v)
    img_vecs = np.stack(img_vecs, axis=0)
    model_rdm = corr_dist_rdm(img_vecs)
    return upper_tri_vec(model_rdm)


def load_resnet_pca(pca_path: str = RESNET_PCA_PATH) -> dict:
    obj = np.load(pca_path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        obj = obj.item()
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in PCA file, got {type(obj)}")
    return obj


# -------------------------
# RSA computations
# -------------------------

def rsa_timecourse(eeg_rdm_vecs: np.ndarray, model_vec: np.ndarray) -> np.ndarray:
    n_t = eeg_rdm_vecs.shape[0]
    rsa_t = np.zeros(n_t, dtype=np.float32)
    for t in range(n_t):
        rsa_t[t] = spearmanr(eeg_rdm_vecs[t], model_vec).correlation
    return rsa_t


def rsa_by_layers(obj: dict, sel: np.ndarray, eeg_rdm_vecs: np.ndarray) -> Dict[str, np.ndarray]:
    rsa_by_layer = {}
    for lk in obj.keys():
        feats = obj[lk]
        if not (isinstance(feats, np.ndarray) and feats.ndim == 2) or lk =="fc":
            continue
        feats_sub = feats[sel]
        model_vec = upper_tri_vec(corr_dist_rdm(feats_sub))
        rsa_t = rsa_timecourse(eeg_rdm_vecs, model_vec)
        rsa_by_layer[lk] = rsa_t
    return rsa_by_layer


# Main Flow


def main(
    eeg_train_path: str = EEG_TRAIN_PATH,
    eeg_test_path: str = EEG_TEST_PATH,
    img_meta_path: str = IMG_META_PATH,
    image_base: str = IMAGE_BASE,
    resnet_pca_path: str = RESNET_PCA_PATH,
    N: int = 200,
    seed: int = 0,
):
    # Load EEG
    eeg_train, eeg_test = load_eeg(eeg_train_path, eeg_test_path)
    X_train = eeg_train["preprocessed_eeg_data"]
    ch_names = eeg_train["ch_names"]
    times = eeg_train["times"]

    # Select images and compute patterns
    sel, X_sub, patterns = select_images_and_patterns(X_train, N=N, seed=seed)
    print("Selected indices:", sel[:10], "...")
    print("X_sub shape:", X_sub.shape)
    print("patterns shape:", patterns.shape)

    # Compute EEG RDM vectors
    eeg_rdm_vecs = compute_eeg_rdm_vecs(patterns)
    print("eeg_rdm_vecs shape:", eeg_rdm_vecs.shape)

    # ResNet PCA features
    obj = load_resnet_pca(resnet_pca_path)

    # Sanity check: ensure alignment
    first_key = next((k for k, v in obj.items() if isinstance(v, np.ndarray) and v.ndim == 2), None)
    if first_key is None:
        raise RuntimeError("No suitable feature arrays found in ResNet PCA object")

    n_feat_images = obj[first_key].shape[0]
    print("Feature rows:", n_feat_images, "| EEG train images:", X_train.shape[0])
    assert n_feat_images == X_train.shape[0], "Feature rows must match EEG training images"

    rsa_layers = rsa_by_layers(obj, sel, eeg_rdm_vecs)
    for lk, rsa_t in rsa_layers.items():
        print(f"{lk}: max RSA={np.nanmax(rsa_t):.4f} at t={times[np.nanargmax(rsa_t)]:.3f}s")

    plt.figure(figsize=(9, 4))
    for lk, rsa_t in rsa_layers.items():
        plt.plot(times, rsa_t, label=lk)
    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("RSA (Spearman ρ)")
    plt.title("EEG–ResNet RSA over time (training images, layer-wise)")
    plt.legend(fontsize=8)
    save_figure('EEG-ResNet_RSA')


if __name__ == "__main__":
    main(N=500, seed=42)
