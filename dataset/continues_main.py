import os
import re
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import json


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
FOLDER_PATH = r"D:\Github\PhD Code\FFHQ-UV\dataset\data"
CHECKPOINT  = r"D:\Github\PhD Code\FFHQ-UV\Pretrain_Model\BioSkinAO.pt"
OUTPUT_DIR  = r"D:\Github\PhD Code\FFHQ-UV\output_maps"


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
class BioSkinEncoder(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        out_size = state_dict['module.fc_enc_out.weight'].shape[0]
        self.fc_enc_in  = nn.Linear(3,       70)
        self.fc_enc     = nn.Linear(70,      70)
        self.fc_enc_out = nn.Linear(70, out_size)
        enc_sd = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
            if "fc_enc" in k
        }
        self.load_state_dict(enc_sd)

    def forward(self, x):
        x = torch.relu(self.fc_enc_in(x))
        x = torch.relu(self.fc_enc(x))
        return self.fc_enc_out(x)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def compute_mode(arr):
    """Histogram-based mode for continuous pixel distributions."""
    hist, edges = np.histogram(arr.flatten(), bins=100)
    peak = np.argmax(hist)
    return float((edges[peak] + edges[peak + 1]) / 2.0)


def normalize_map(param_map):
    """
    Normalize parameter map to [0, 1] using min-max normalization.
    Handles the case where min == max (avoid division by zero).
    """
    min_val = np.min(param_map)
    max_val = np.max(param_map)
    
    if min_val == max_val:
        # All values are the same, return 0.5 (middle of range)
        return np.full_like(param_map, 0.5, dtype=np.float32)
    
    normalized = (param_map - min_val) / (max_val - min_val)
    return normalized.astype(np.float32)


def process_one(img_path, encoder):
    """
    Load albedo → run BioSkinAO pixel-wise → return parameter maps.
    Returns:
        - Parameter maps at original resolution (H, W)
        - Aggregated statistics (mode, std)
    """
    img    = Image.open(img_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0   # (H, W, 3)
    H, W, _ = img_np.shape

    pixels = torch.from_numpy(img_np.reshape(-1, 3))
    with torch.no_grad():
        raw = encoder(pixels).numpy()                    # (H*W, 6)

    # Reshape to spatial maps
    maps = raw.reshape(H, W, -1)                         # (H, W, 6)

    mel  = maps[:, :, 0]   # melanin
    hemo = maps[:, :, 1]   # hemoglobin
    thk  = maps[:, :, 2]   # epidermal thickness
    eur  = maps[:, :, 3]   # eumelanin ratio
    oxy  = maps[:, :, 4]   # blood oxygenation

    # Compute aggregated statistics (before normalization)
    stats = {
        'melanin_mode':    round(compute_mode(mel),   6),
        'melanin_std':     round(float(np.std(mel)),  6),
        'hemoglobin_mode': round(compute_mode(hemo),  6),
        'hemoglobin_std':  round(float(np.std(hemo)), 6),
        'eumelanin_ratio': round(compute_mode(eur),   6),
        'oxygenation':     round(compute_mode(oxy),   6),
        'epidermal_thick': round(compute_mode(thk),   6),
    }

    # Normalize each map to [0, 1]
    mel_norm  = normalize_map(mel)
    hemo_norm = normalize_map(hemo)
    thk_norm  = normalize_map(thk)
    eur_norm  = normalize_map(eur)
    oxy_norm  = normalize_map(oxy)

    return {
        'melanin': (mel_norm, stats['melanin_mode'], stats['melanin_std']),
        'hemoglobin': (hemo_norm, stats['hemoglobin_mode'], stats['hemoglobin_std']),
        'epidermal_thick': (thk_norm, stats['epidermal_thick'], 0.0),
        'eumelanin_ratio': (eur_norm, stats['eumelanin_ratio'], 0.0),
        'oxygenation': (oxy_norm, stats['oxygenation'], 0.0),
    }, stats


def save_param_map_with_stats(param_map, mode, std, output_path):
    """
    Save parameter map with statistics as extra rows.
    
    Structure:
        - Rows 0:(H-1) → normalized parameter map (H, W)
        - Row H → mode values (broadcasted across width)
        - Row H+1 → std values (broadcasted across width)
    
    Final shape: (H+2, W)
    """
    H, W = param_map.shape
    
    # Create extended array with 2 extra rows
    extended = np.zeros((H + 2, W), dtype=np.float32)
    
    # Row 0:H → normalized map
    extended[0:H, :] = param_map
    
    # Row H → mode (broadcasted across width)
    extended[H, :] = mode
    
    # Row H+1 → std (broadcasted across width)
    extended[H+1, :] = std
    
    np.save(output_path, extended)


def get_albedo_paths(folder):
    """
    Collect all albedo.png paths.
    Folder structure: data/1/albedo.png, data/2/albedo.png ...
    Sort numerically by folder number — not alphabetically.
    """
    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    paths = []

    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(supported) and 'albedo' in f.lower():
                paths.append(os.path.join(root, f))

    # Sort by the numeric folder name — 1, 2, 3 ... not 1, 10, 100
    def numeric_sort_key(path):
        parent = os.path.basename(os.path.dirname(path))
        digits = re.findall(r'\d+', parent)
        return int(digits[0]) if digits else 0

    return sorted(paths, key=numeric_sort_key)


def get_id(path):
    """
    Extract ID from parent folder name.
    data/1/albedo.png   → 1
    data/42/albedo.png  → 42
    """
    parent = os.path.basename(os.path.dirname(path))
    digits = re.findall(r'\d+', parent)
    return int(digits[0]) if digits else parent


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():

    # ── Create output directory ───────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # ── Load encoder ──────────────────────────────────────────
    print("Loading BioSkinAO encoder...")
    ckpt    = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    encoder = BioSkinEncoder(ckpt)
    encoder.eval()
    encoder.requires_grad_(False)
    print("Encoder ready.\n")

    # ── Collect paths ─────────────────────────────────────────
    paths = get_albedo_paths(FOLDER_PATH)
    print(f"Found {len(paths)} albedo images")

    if not paths:
        print("No albedo images found. Check folder path.")
        return

    # Preview first 5 paths so you can confirm structure
    print("First 5 paths found:")
    for p in paths[:5]:
        print(f"  ID={get_id(p):>6}  →  {p}")
    print()

    failed  = []
    written = 0
    metadata_records = []

    # ── Process ───────────────────────────────────────────────
    for path in tqdm(paths,
                     desc  = "BioSkin extraction (UV maps)",
                     unit  = "img",
                     colour= "green"):
        try:
            sample_id = get_id(path)
            param_maps, stats = process_one(path, encoder)
            
            # Create per-sample output directory
            sample_output_dir = os.path.join(OUTPUT_DIR, str(sample_id))
            os.makedirs(sample_output_dir, exist_ok=True)
            
            # Save each parameter map with statistics in last 2 rows
            for param_name, (normalized_map, mode, std) in param_maps.items():
                output_path = os.path.join(sample_output_dir, f"{param_name}.npy")
                save_param_map_with_stats(normalized_map, mode, std, output_path)
            
            # Store metadata for this sample
            metadata_records.append({
                'id': sample_id,
                'folder': sample_output_dir,
                'image_shape': list(normalized_map.shape),
                'stored_shape': [normalized_map.shape[0] + 2, normalized_map.shape[1]],
                'stats': stats
            })
            
            written += 1

        except Exception as e:
            fname = os.path.basename(path)
            failed.append((fname, str(e)))

    # ── Save master metadata JSON ─────────────────────────────
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata_records, f, indent=2)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Processed       : {written} images")
    print(f"  Failed          : {len(failed)} images")
    print(f"  Output dir      : {OUTPUT_DIR}")
    print(f"  Metadata saved  : {metadata_path}")
    print(f"{'='*70}")

    if failed:
        print(f"\nFailed files ({len(failed)}):")
        for fname, err in failed:
            print(f"  {fname:<30} {err}")

    # ── Preview first sample ──────────────────────────────────
    print("\nPreview (first sample):")
    if metadata_records:
        first_sample = metadata_records[0]
        print(f"  Sample ID      : {first_sample['id']}")
        print(f"  Image shape    : {first_sample['image_shape']}")
        print(f"  Stored shape   : {first_sample['stored_shape']} (map + 2 stat rows)")
        print(f"  Output dir     : {first_sample['folder']}")
        print(f"  Parameters     :")
        for param, value in first_sample['stats'].items():
            print(f"    {param:<20} : {value}")


if __name__ == '__main__':
    main()
