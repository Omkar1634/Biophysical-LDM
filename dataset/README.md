# FFHQ-UV BioSkin Parameter Extraction

This project extracts skin biophysical parameters from FFHQ face albedo images using a pretrained BioSkinAO encoder.

---

## Scripts Overview

### 1. **main.py** — Aggregated Statistics (Scalar Mode)
Extracts **single aggregated value per face** for each parameter across the entire albedo map.

#### What it does:
- Runs BioSkinAO encoder on all pixels of each albedo image
- Computes **mode** (peak) and **std** (spread) for each parameter
- Stores one statistic per face in **CSV format**
- Output: `skin_params.csv`

#### Output Format:
```csv
id, melanin_mode, melanin_std, hemoglobin_mode, hemoglobin_std, eumelanin_ratio, oxygenation, epidermal_thick
1,  -38.710255,   5.529956,    14.443535,       2.190942,      -0.615705,      15.829866,   24.044323
```

#### Use Case:
- Quick overview of skin parameters per face
- Statistical aggregation across entire face
- Lightweight CSV format for databases

---

### 2. **continues_main.py** — Spatial UV Maps (Continuous Space)
Extracts **parameter values at every UV coordinate** while preserving spatial structure.

#### What it does:
- Runs BioSkinAO encoder on all pixels per image
- Keeps **full spatial parameter maps** at original albedo resolution
- Normalizes each map to [0, 1] range (min-max normalization)
- **Stores mode & std in last 2 rows** of each .npy file
- Output: `.npy` files organized per sample + `metadata.json`

#### Output Structure:
```
output_maps/
├── 1/
│   ├── melanin.npy          (shape: H+2, W)
│   ├── hemoglobin.npy       
│   ├── epidermal_thick.npy  
│   ├── eumelanin_ratio.npy  
│   └── oxygenation.npy      
├── 2/
│   └── [5 .npy files]
└── metadata.json            (centralized statistics & shape info)
```

#### .npy File Structure:
```
Rows 0 to H-1  → Normalized parameter map [0, 1]
Row H          → Mode value (broadcasted)
Row H+1        → Std value (broadcasted)
```

#### Usage Example:
```python
import numpy as np

# Load melanin spatial map with statistics
data = np.load("output_maps/1/melanin.npy")
melanin_map = data[:-2, :]      # H×W normalized spatial map
melanin_mode = data[-2, 0]      # Aggregated mode
melanin_std = data[-1, 0]       # Aggregated std

# Query continuous UV value at specific coordinate
uv_value = melanin_map[y, x]    # Single value at (x, y)
```

#### Use Case:
- Per-pixel/per-region parameter analysis
- Texture mapping in 3D rendering
- Spatial variation analysis across face
- Machine learning with spatial input
- UV-space visualization

---

## Key Differences

| Feature | main.py | continues_main.py |
|---------|---------|------------------|
| **Output Format** | CSV (scalars) | .npy files (spatial maps) |
| **Values per Face** | 7 aggregated values | H×W per parameter |
| **Resolution** | N/A (aggregated) | Original albedo size |
| **Normalization** | None | [0, 1] min-max |
| **Statistics Stored** | ✓ In CSV | ✓ In last 2 rows of .npy |
| **File Size** | Very small (~1KB per face) | Medium (depends on resolution) |
| **Access Pattern** | Random row lookup | Spatial 2D indexing |

---

## Parameters Extracted

Both scripts extract these 5 biophysical parameters:

1. **Melanin** - Skin pigmentation concentration
2. **Hemoglobin** - Blood/oxygenation related parameter
3. **Epidermal Thickness** - Outer skin layer depth
4. **Eumelanin Ratio** - Type of melanin composition
5. **Oxygenation** - Blood oxygen saturation

Each parameter includes:
- **Mode** (peak value in distribution)
- **Std** (standard deviation/spread)

---

## Requirements

```
torch
PIL (Pillow)
numpy
tqdm
```

Pretrained model required:
- `Pretrain_Model/BioSkinAO.pt` — For `continues_main.py`
- `Pretrain_Model/BioSkinAO.pt` — For `main.py`

---

## Running the Scripts

```bash
# Aggregated statistics (CSV output)
python main.py

# Spatial maps (UV-space .npy output)
python continues_main.py
```

---

## When to Use Each

### Use **main.py** if you need:
- Quick face-level statistics
- Lightweight CSV database
- Single averaged values per face
- Simple data analysis

### Use **continues_main.py** if you need:
- Per-pixel/region parameter values
- Texture maps for 3D rendering
- Spatial variation analysis
- Machine learning with 2D spatial input
- Continuous UV-space representation

---

## Notes

- **Original Resolution Preservation**: `continues_main.py` keeps maps at original albedo resolution (not resized)
- **Normalization**: Per-map min-max normalization ensures [0, 1] range
- **Statistics Storage**: Both mode and std are stored for reference alongside spatial maps
- **Metadata**: `continues_main.py` generates `metadata.json` for quick reference of shapes and statistics
- **No Scaler**: Raw network outputs are used directly (no external normalization layer)

---

## Output Files

### main.py produces:
- `skin_params.csv` — Single CSV with mode/std for all faces

### continues_main.py produces:
- `output_maps/[ID]/[parameter].npy` — Spatial parameter maps (5 per sample)
- `output_maps/metadata.json` — Centralized metadata reference
