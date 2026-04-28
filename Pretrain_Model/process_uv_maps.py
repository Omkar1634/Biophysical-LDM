import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import os
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# STEP 1 — Rebuild BioSkinAO encoder (6 outputs, not 5)
# Use BioSkinAO.pt — this is the correct one for the paper
# ─────────────────────────────────────────────────────────────
class BioSkinEncoder(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        # Detect output size from checkpoint automatically
        out_size = state_dict['module.fc_enc_out.weight'].shape[0]
        # BioSkin   → 5 outputs
        # BioSkinAO → 6 outputs (includes oxygenation) ← use this one

        self.fc_enc_in  = nn.Linear(3,        70)
        self.fc_enc     = nn.Linear(70,        70)
        self.fc_enc_out = nn.Linear(70,  out_size)

        # Strip 'module.' prefix added by DataParallel training
        enc_sd = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
            if "fc_enc" in k
        }
        self.load_state_dict(enc_sd)

    def forward(self, x):
        x = torch.relu(self.fc_enc_in(x))
        x = torch.relu(self.fc_enc(x))
        return self.fc_enc_out(x)   # raw log-space outputs


# ─────────────────────────────────────────────────────────────
# STEP 2 — Inverse transform: log-space → physical values
# BioSkin outputs are in log space — must invert before use
# This is described in the BioSkin paper (Aliaga et al. 2023)
# ─────────────────────────────────────────────────────────────
def to_physical(param_maps):
    """
    Convert raw encoder outputs (log space) to physical skin params.
    BioSkinAO outputs 6 channels:
      0: melanin concentration       (nM)   → exp(x) then normalise
      1: hemoglobin concentration    (nB)   → exp(x) then normalise
      2: epidermal thickness         (nT)
      3: eumelanin/pheomelanin ratio (nE)
      4: blood oxygenation           (nO)   ← extra in AO variant
      5: (additional chromophore)
    """
    physical = np.exp(param_maps)   # invert log transform
    return physical


# ─────────────────────────────────────────────────────────────
# STEP 3 — Compute the 9 condition vector components
# melanin:    mode + std  (2 values) — captures even vs patchy
# hemoglobin: mode + std  (2 values) — captures uniform vs rosacea
# thickness:  mode only   (1 value)  — spatially uniform
# eu ratio:   mode only   (1 value)  — spatially uniform
# oxy:        mode only   (1 value)  — systemic, no variation
# age:        from metadata
# gender:     from metadata
# ─────────────────────────────────────────────────────────────
def compute_mode(arr):
    """Histogram-based mode for continuous distributions."""
    hist, bin_edges = np.histogram(arr.flatten(), bins=100)
    peak = np.argmax(hist)
    return float((bin_edges[peak] + bin_edges[peak + 1]) / 2.0)


def extract_condition_vector(param_maps_physical):
    """
    param_maps_physical: (H, W, 6)  — physical space after exp()

    Returns dict with all 7 BioSkin-derived values.
    Age and gender are added separately from metadata.
    """
    melanin_map    = param_maps_physical[:, :, 0]
    hemoglobin_map = param_maps_physical[:, :, 1]
    thickness_map  = param_maps_physical[:, :, 2]
    euratio_map    = param_maps_physical[:, :, 3]
    oxy_map        = param_maps_physical[:, :, 4]

    return {
        # melanin: mode + std (2 values)
        'melanin_mode':      compute_mode(melanin_map),
        'melanin_std':       float(np.std(melanin_map)),

        # hemoglobin: mode + std (2 values)
        'hemoglobin_mode':   compute_mode(hemoglobin_map),
        'hemoglobin_std':    float(np.std(hemoglobin_map)),

        # mode only for these three
        'eumelanin_ratio':   compute_mode(euratio_map),
        'oxygenation':       compute_mode(oxy_map),
        'epidermal_thick':   compute_mode(thickness_map),

        # age and gender added from metadata → total = 9
    }


# ─────────────────────────────────────────────────────────────
# STEP 4 — Process one image end to end
# ─────────────────────────────────────────────────────────────
def process_one_albedo(img_path, encoder):
    # Load and normalise
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0   # (H, W, 3)
    H, W, _ = img_np.shape

    # Run encoder pixel-wise
    pixels = torch.from_numpy(img_np.reshape(-1, 3))
    with torch.no_grad():
        raw_params = encoder(pixels).numpy()             # (H*W, 6)

    # Reshape to spatial maps
    param_maps_raw = raw_params.reshape(H, W, -1)        # (H, W, 6)

    # Convert from log space to physical values
    param_maps_physical = to_physical(param_maps_raw)    # (H, W, 6)

    # Extract 7 condition vector components
    c_vec = extract_condition_vector(param_maps_physical)

    return c_vec, param_maps_physical


# ─────────────────────────────────────────────────────────────
# STEP 5 — Process entire dataset and save labels
# ─────────────────────────────────────────────────────────────
def process_dataset(folder_path, encoder,
                    output_json='skin_params.json'):

    supported = ('.jpg', '.jpeg', '.png', '.bmp')
    paths = [
        os.path.join(r, f)
        for r, _, files in os.walk(folder_path)
        for f in files
        if f.lower().endswith(supported)
    ]
    print(f"Found {len(paths)} images")

    results = {}
    for path in tqdm(paths, desc="Extracting skin params"):
        fname = os.path.basename(path)
        try:
            c_vec, _ = process_one_albedo(path, encoder)
            # Add placeholders for age/gender
            # Fill these in from your dataset metadata
            c_vec['age']    = None
            c_vec['gender'] = None
            results[fname] = c_vec
        except Exception as e:
            print(f"  Failed: {fname} — {e}")

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} entries → {output_json}")
    return results


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Load BioSkinAO — 6 outputs, includes oxygenation
    ckpt = torch.load(
        r"D:\Github\PhD Code\Biophysical-LDM\Pretrain_Model\BioSkinAO.pt",
        map_location="cpu",
        weights_only=False   # suppresses the FutureWarning you saw
    )

    encoder = BioSkinEncoder(ckpt)
    encoder.eval()
    encoder.requires_grad_(False)   # fully frozen

    # Process UV Maps from the Final UV Maps folder
    process_dataset(
        folder_path  = r"D:\Github\PhD Code\Biophysical-LDM\dataset\Final UV Maps",
        encoder      = encoder,
        output_json  = r"D:\Github\PhD Code\Biophysical-LDM\Pretrain_Model\skin_params_final_uv.json"
    )
