import os
import re
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm
import csv


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
FOLDER_PATH = r"D:\Github\PhD Code\FFHQ-UV\dataset\data"
CHECKPOINT  = r"D:\Github\PhD Code\FFHQ-UV\Pretrain_Model\BioSkinAO.pt"
OUTPUT_CSV  = r"D:\Github\PhD Code\FFHQ-UV\skin_params.csv"


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


def process_one(img_path, encoder):
    """
    Load albedo → run BioSkinAO pixel-wise → return raw params.
    NO np.exp() — BioSkinAO outputs are already in linear space.
    """
    img    = Image.open(img_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0   # (H, W, 3)
    H, W, _ = img_np.shape

    pixels = torch.from_numpy(img_np.reshape(-1, 3))
    with torch.no_grad():
        raw = encoder(pixels).numpy()                    # (H*W, 6)

    # Reshape to spatial maps — NO exp() transform
    maps = raw.reshape(H, W, -1)                         # (H, W, 6)

    mel  = maps[:, :, 0]   # melanin
    hemo = maps[:, :, 1]   # hemoglobin
    thk  = maps[:, :, 2]   # epidermal thickness
    eur  = maps[:, :, 3]   # eumelanin ratio
    oxy  = maps[:, :, 4]   # blood oxygenation

    return {
        'melanin_mode':    round(compute_mode(mel),   6),
        'melanin_std':     round(float(np.std(mel)),  6),
        'hemoglobin_mode': round(compute_mode(hemo),  6),
        'hemoglobin_std':  round(float(np.std(hemo)), 6),
        'eumelanin_ratio': round(compute_mode(eur),   6),
        'oxygenation':     round(compute_mode(oxy),   6),
        'epidermal_thick': round(compute_mode(thk),   6),
    }


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

    # ── CSV setup ─────────────────────────────────────────────
    fieldnames = [
        'id',
        'melanin_mode',
        'melanin_std',
        'hemoglobin_mode',
        'hemoglobin_std',
        'eumelanin_ratio',
        'oxygenation',
        'epidermal_thick',
    ]

    failed  = []
    written = 0

    # ── Process ───────────────────────────────────────────────
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for path in tqdm(paths,
                         desc  = "BioSkin extraction",
                         unit  = "img",
                         colour= "green"):
            try:
                params       = process_one(path, encoder)
                params['id'] = get_id(path)
                writer.writerow({col: params[col] for col in fieldnames})
                written += 1

            except Exception as e:
                fname = os.path.basename(path)
                failed.append((fname, str(e)))

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Processed : {written} images")
    print(f"  Failed    : {len(failed)} images")
    print(f"  Saved to  : {OUTPUT_CSV}")
    print(f"{'='*55}")

    if failed:
        print(f"\nFailed files ({len(failed)}):")
        for fname, err in failed:
            print(f"  {fname:<30} {err}")

    # ── Preview first 5 rows of CSV ───────────────────────────
    print("\nPreview (first 5 rows):")
    with open(OUTPUT_CSV, 'r') as f:
        lines = f.readlines()
    for line in lines[:6]:
        print(" ", line.strip())


if __name__ == '__main__':
    main()