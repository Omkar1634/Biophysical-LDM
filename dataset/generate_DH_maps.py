"""
generate_DH_maps.py
Generates Height (H) and Deformation/Vector field (D) maps
from existing normal maps in FFHQ-UV dataset.

Saves results as Arrow dataset — same format as original dataset.
Uses same IDs (ori_id, sh_id) from the original normal maps.

Usage:
    python generate_DH_maps.py --test        # test 1 sample first
    python generate_DH_maps.py               # full 30K
    python generate_DH_maps.py --start 5000  # resume from idx 5000
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from datasets import load_from_disk, Dataset, Features, Value
from datasets import Image as HFImage
from tqdm import tqdm


# ── Config ─────────────────────────────────────────────────────
DATASET_PATH = r"D:\Github\PhD Code\FFHQ-UV\dataset"
OUTPUT_PATH  = r"D:\Github\PhD Code\FFHQ-UV\dataset_DH"
RESOLUTION   = 1024
BATCH_SIZE   = 500    # save to Arrow every N samples


# ── Step 1 — Decode normal map → Nx, Ny, Nz ───────────────────
def decode_normal_map(normal_img):
    """
    Convert RGB-encoded tangent space normal map
    to actual normal vectors in [-1, 1] range.
    OpenGL convention: R=X(right), G=Y(up), B=Z(forward)
    """
    arr = np.array(normal_img, dtype=np.float32)

    Nx = (arr[:, :, 0] / 255.0) * 2.0 - 1.0   # R → X
    Ny = (arr[:, :, 1] / 255.0) * 2.0 - 1.0   # G → Y
    Nz = (arr[:, :, 2] / 255.0) * 2.0 - 1.0   # B → Z

    # Normalize to unit vectors
    length = np.sqrt(Nx**2 + Ny**2 + Nz**2) + 1e-8
    Nx /= length
    Ny /= length
    Nz /= length

    return Nx, Ny, Nz


# ── Step 2 — Height map from Nz + masked Frankot-Chellappa ─────
def compute_height_map(Nx, Ny, Nz):
    """
    Generate height map from normal map for UV-unwrapped face textures.

    Strategy:
      1. Mask background regions (Nz ≈ 0 → not face)
      2. Use masked Frankot-Chellappa for face pixels only
      3. Fall back to Nz-based height where integration is unreliable

    Nx, Ny, Nz : normal components in [-1, 1]
    Returns H  : height map as uint8 RGB PIL Image
    """
    eps = 1e-8

    # ── Background mask ────────────────────────────────────────
    magnitude = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    mask      = magnitude > 0.1       # True = face pixel

    # ── Masked gradients — zero out background ─────────────────
    p = np.where(mask, -Nx / (np.abs(Nz) + eps), 0.0)
    q = np.where(mask, -Ny / (np.abs(Nz) + eps), 0.0)
    p = np.clip(p, -5, 5)
    q = np.clip(q, -5, 5)

    H_sz, W_sz = p.shape

    # ── Frankot-Chellappa in frequency domain ──────────────────
    fy           = np.fft.fftfreq(H_sz).reshape(-1, 1)
    fx           = np.fft.fftfreq(W_sz).reshape(1, -1)
    P            = np.fft.fft2(p)
    Q            = np.fft.fft2(q)
    denom        = fx**2 + fy**2
    denom[0, 0]  = 1.0
    Z_freq       = (fx * P + fy * Q) / denom
    Z_freq[0, 0] = 0.0
    Z            = np.real(np.fft.ifft2(Z_freq))

    # ── Apply mask — zero background ───────────────────────────
    Z[~mask] = 0.0

    # ── Normalize face region only ─────────────────────────────
    if mask.sum() > 0:
        face_vals = Z[mask]
        Z[mask]   = (face_vals - face_vals.min()) / \
                    (face_vals.max() - face_vals.min() + eps)

    Z = np.clip(Z * 255, 0, 255).astype(np.uint8)

    # ── Blend with Nz-based height for stability ───────────────
    # Nz directly encodes forward-facing depth — good complement
    Nz_height         = np.zeros_like(Nz)
    Nz_height[mask]   = (Nz[mask] + 1.0) / 2.0
    Nz_height         = (Nz_height * 255).astype(np.uint8)

    # 70% Frankot-Chellappa + 30% Nz → stable result
    Z_blended = (0.7 * Z + 0.3 * Nz_height).astype(np.uint8)

    Z_rgb = np.stack([Z_blended, Z_blended, Z_blended], axis=-1)
    return Image.fromarray(Z_rgb)


# ── Step 3 — Vector field (D) from normal gradients ────────────
def compute_vector_field(Nx, Ny, Nz):
    """
    Compute 2D deformation/vector field from normal map.
    Substitute for paper's D (deformation) map.
    Encodes how surface normal direction changes across UV space.

    Returns D : RGB PIL Image encoding (dx, dy, magnitude)
    """
    # Spatial gradients of normal components
    dNx_dx = np.gradient(Nx, axis=1)
    dNx_dy = np.gradient(Nx, axis=0)
    dNy_dx = np.gradient(Ny, axis=1)
    dNy_dy = np.gradient(Ny, axis=0)

    # Vector field components
    dx        = dNx_dx + dNy_dy    # horizontal flow
    dy        = dNx_dy - dNy_dx    # vertical flow
    magnitude = np.sqrt(dx**2 + dy**2)

    def norm_to_uint8(ch):
        ch = ch - ch.min()
        ch = ch / (ch.max() + 1e-8)
        return (ch * 255).astype(np.uint8)

    R = norm_to_uint8(dx)
    G = norm_to_uint8(dy)
    B = norm_to_uint8(magnitude)

    D_rgb = np.stack([R, G, B], axis=-1)
    return Image.fromarray(D_rgb)


# ── Step 4 — Process and save as Arrow ─────────────────────────
def process_dataset(start_idx=0, max_samples=None):
    print(f"Loading source dataset from {DATASET_PATH}...")
    source = load_from_disk(DATASET_PATH)["train"]

    n = len(source) if max_samples is None else min(max_samples, len(source))
    print(f"Processing {n} samples (starting from idx {start_idx})...")

    # ── Check for existing partial output ──────────────────────
    existing_dataset = None
    if os.path.exists(OUTPUT_PATH):
        try:
            existing_dataset = load_from_disk(OUTPUT_PATH)
            print(f"Found existing output: {len(existing_dataset)} samples already done")
            start_idx = max(start_idx, len(existing_dataset))
            print(f"Resuming from idx {start_idx}")
        except Exception:
            print("No valid existing output found — starting fresh")

    # ── Process in batches and save ────────────────────────────
    batch_records = []
    success = 0
    failed  = 0

    for idx in tqdm(range(start_idx, n), desc="Generating H & D maps"):
        try:
            sample = source[idx]

            # ── Get IDs from original sample — same as normal map ──
            ori_id = sample["ori_id"]   # original FFHQ face ID
            sh_id  = sample["sh_id"]    # spherical harmonics ID

            # ── Load normal map ────────────────────────────────────
            normal_img = sample["normal_path"]
            if not isinstance(normal_img, Image.Image):
                normal_img = Image.fromarray(normal_img)
            normal_img = normal_img.convert("RGB")

            if normal_img.size != (RESOLUTION, RESOLUTION):
                normal_img = normal_img.resize(
                    (RESOLUTION, RESOLUTION), Image.BILINEAR)

            # ── Generate H and D maps ──────────────────────────────
            Nx, Ny, Nz = decode_normal_map(normal_img)
            H_img      = compute_height_map(Nx, Ny, Nz)       # PIL RGB
            D_img      = compute_vector_field(Nx, Ny, Nz)   # PIL RGB

            # ── Build record — same ID fields as source ────────────
            record = {
                "ori_id":     ori_id,       # same as source
                "sh_id":      sh_id,        # same as source
                "height_map": H_img,        # PIL Image → HF Image
                "deform_map": D_img,        # PIL Image → HF Image
            }
            batch_records.append(record)
            success += 1

            # ── Save batch to Arrow ────────────────────────────────
            if len(batch_records) >= BATCH_SIZE:
                _save_batch(batch_records, existing_dataset)
                existing_dataset = load_from_disk(OUTPUT_PATH)
                batch_records    = []
                print(f"\n💾 Saved batch | Total: {len(existing_dataset)} samples")

        except Exception as e:
            print(f"\nFailed idx {idx} (ori_id={sample.get('ori_id','?')}): {e}")
            failed += 1
            continue

    # ── Save remaining records ─────────────────────────────────
    if batch_records:
        _save_batch(batch_records, existing_dataset)
        existing_dataset = load_from_disk(OUTPUT_PATH)

    print(f"\n✅ Done!")
    print(f"   Success  : {success}")
    print(f"   Failed   : {failed}")
    print(f"   Total    : {len(existing_dataset)} samples in Arrow")
    print(f"   Saved to : {OUTPUT_PATH}")


def _save_batch(records, existing_dataset):
    """Save a batch of records — append to existing or create new."""
    new_ds = Dataset.from_list(
        records,
        features = Features({
            "ori_id":     Value("string"),
            "sh_id":      Value("string"),
            "height_map": HFImage(),
            "deform_map": HFImage(),
        })
    )

    if existing_dataset is not None:
        from datasets import concatenate_datasets
        combined = concatenate_datasets([existing_dataset, new_ds])
    else:
        combined = new_ds

    combined.save_to_disk(OUTPUT_PATH)


# ── Quick visual test — 1 sample ───────────────────────────────
def test_single(idx=0):
    import matplotlib.pyplot as plt

    print(f"Testing on sample {idx}...")
    source     = load_from_disk(DATASET_PATH)["train"]
    sample     = source[idx]

    normal_img = sample["normal_path"]
    albedo_img = sample["albedo_path"]

    if not isinstance(normal_img, Image.Image):
        normal_img = Image.fromarray(normal_img)
    if not isinstance(albedo_img, Image.Image):
        albedo_img = Image.fromarray(albedo_img)

    Nx, Ny, Nz = decode_normal_map(normal_img.convert("RGB"))
    H_img      = compute_height_map(Nx, Ny, Nz)
    D_img      = compute_vector_field(Nx, Ny, Nz)

    H_arr = np.array(H_img)
    D_arr = np.array(D_img)

    # Visual check
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(albedo_img)
    axes[0].set_title(f"Albedo | ori_id={sample['ori_id']}")
    axes[1].imshow(normal_img)
    axes[1].set_title("Normal (source)")
    axes[2].imshow(H_arr, cmap="gray")
    axes[2].set_title("H — Height (generated)")
    axes[3].imshow(D_arr)
    axes[3].set_title("D — Vector field (generated)")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("DH_test.png", dpi=150)
    print(f"\nori_id : {sample['ori_id']}")
    print(f"sh_id  : {sample['sh_id']}")
    print(f"H map  : shape={H_arr.shape} min={H_arr.min()} max={H_arr.max()} mean={H_arr.mean():.1f}")
    print(f"D map  : shape={D_arr.shape} min={D_arr.min()} max={D_arr.max()} mean={D_arr.mean():.1f}")
    print(f"\n✅ Test saved → DH_test.png")

    # Also test Arrow save with 1 record
    print("\nTesting Arrow save...")
    test_record = [{
        "ori_id":     sample["ori_id"],
        "sh_id":      sample["sh_id"],
        "height_map": H_img,
        "deform_map": D_img,
    }]
    test_ds = Dataset.from_list(
        test_record,
        features = Features({
            "ori_id":     Value("string"),
            "sh_id":      Value("string"),
            "height_map": HFImage(),
            "deform_map": HFImage(),
        })
    )
    test_out = r"D:\Github\PhD Code\FFHQ-UV\dataset_DH_test"
    test_ds.save_to_disk(test_out)
    print(f"✅ Arrow test saved → {test_out}")

    # Reload and verify
    reloaded = load_from_disk(test_out)
    print(f"✅ Reloaded: {len(reloaded)} sample")
    print(f"   Columns : {reloaded.column_names}")
    r = reloaded[0]
    print(f"   ori_id  : {r['ori_id']}")
    print(f"   sh_id   : {r['sh_id']}")
    h_reload = np.array(r["height_map"])
    d_reload = np.array(r["deform_map"])
    print(f"   H shape : {h_reload.shape}")
    print(f"   D shape : {d_reload.shape}")
    print(f"\n✅ Full test passed — Arrow format working!")


# ── Entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  action="store_true",
                        help="Test on 1 sample before full run")
    parser.add_argument("--start", type=int, default=0,
                        help="Resume from this index")
    parser.add_argument("--max",   type=int, default=None,
                        help="Max samples (useful for partial runs)")
    args = parser.parse_args()

    if args.test:
        test_single(0)
    else:
        process_dataset(
            start_idx   = args.start,
            max_samples = args.max,
        )