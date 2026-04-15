# BioSkin-LDM: Biophysically Conditioned Face Albedo Generation

A PyTorch implementation of the pre-training stage for **"Controllable Biophysical Human Faces"**, 
reproducing the LDM Generator conditioned on biophysical skin parameters using publicly available models and datasets.

---

## Overview

This repository implements the **pre-training stage** of a Latent Diffusion Model (LDM) that generates 
UV-unwrapped face albedo textures conditioned on biophysical skin parameters (melanin, hemoglobin, 
oxygenation, epidermal thickness, age, gender).

The pipeline is built on top of:
- **Stable Diffusion 2.1** [RBL*22] as the LDM backbone
- **RGB2X** [ZDG*24] modality switch concept using CLIP text prompts
- **CholeDYM/FFHQ_UV_albedo_sh_lit**  as the pre-training dataset
- A custom **ConditionEmbedder** conditioned on ground-truth biophysical skin parameters 
extracted using the **original Aliaga et al. [AXX*23]** skin estimator — exactly 
matching the paper's pipeline

---

## Architecture

```
Input (UV Albedo 768x768)
        |
        v
[Frozen VAE Encoder]          [Frozen CLIP Text Encoder]
   latent z (96x96x4)            tau(p) = "Albedo Map"
        |                               |
        v                               v
[DDIMScheduler] --> noisy z_t    [ConditionEmbedder]
        |                          skin scalars (9,)
        |                               |
        |                         C_c (B, 77, 1024)
        |                               |
        +---------> [UNet] <-----------+
                  combined = C_p + C_c
                       |
                  v_pred (v-prediction)
                       |
                  MSE Loss vs get_velocity()
                       |
                  Backprop -> update UNet + ConditionEmbedder
```

---

## Pretrained Models Used

### Core LDM Backbone

| Model | Reference | Source | License | Role |
|---|---|---|---|---|
| **Stable Diffusion 2.1** | Rombach et al. [RBL*22] | [HuggingFace](https://huggingface.co/sd2-community/stable-diffusion-2-1) | CreativeML RAIL-M | Base LDM backbone |
| **VAE** (AutoencoderKL) | Part of SD 2.1 | `ldm_pretrain/vae/` | CreativeML RAIL-M | Frozen encoder/decoder |
| **UNet** (UNet2DConditionModel) | Part of SD 2.1 | `ldm_pretrain/unet/` | CreativeML RAIL-M | Trainable denoiser |
| **CLIP Text Encoder** (CLIPTextModel) | Part of SD 2.1 | `ldm_pretrain/text_encoder/` | CreativeML RAIL-M | Frozen text conditioning |
| **DDIMScheduler** | Part of SD 2.1 | `ldm_pretrain/scheduler/` | CreativeML RAIL-M | Noise scheduler |
| **Aliaga et al. skin estimator** | Aliaga et al. [AXX*23] |[BioSkin](https://github.com/facebookresearch/BioSkin) | Research access | Biophysical skin parameter extraction |

**Architecture values read from JSON configs (not hardcoded):**
```
unet/config.json              -> cross_attention_dim=1024, sample_size=96
vae/config.json               -> scaling_factor=0.18215, latent_channels=4
text_encoder/config.json      -> hidden_size=1024, max_position_embeddings=77
scheduler/scheduler_config.json -> prediction_type=v_prediction, DDIMScheduler
```

---

### Modality Switch Concept

| Model | Reference | Source | License | Role |
|---|---|---|---|---|
| **RGB2X** | Zheng et al. [ZDG*24] | [HuggingFace](https://huggingface.co/zheng95z/rgb-to-x) | Adobe Research | Modality switch idea |

The RGB2X concept uses CLIP text prompts as a modality switch to control output map type from a single network:

```python
PROMPTS = {
    "albedo":      "Albedo Map",
    "deformation": "Vector field",
    "height":      "Bump and Displacement Map",
}
```

---

### Dataset

| Dataset | Reference | Source | Size | Role |
|---|---|---|---|---|
| **FFHQ-UV** | CholeDYM | [HuggingFace](https://huggingface.co/datasets/CholeDYM/FFHQ_UV_albedo_sh_lit) | 30,155 samples | Pre-training data |

**Dataset columns used:**
```
albedo_path   -> primary training input (1024x1024 UV albedo maps)
normal_path   -> surface normal maps
sh_path       -> spherical harmonics lighting
lit_path      -> lit face renders
age           -> condition variable (normalized 0-1)
gender        -> condition variable (0=female, 1=male)
bread         -> beard presence
```

---

### Skin Parameter CSV

The `skin_params_biophysical.csv` contains 9 biophysical scalars per face,
generated using the **original Aliaga et al. [AXX*23] skin estimator** —
the same tool used in the paper. This is a key differentiator from other
reproductions which must approximate these values.

The estimator takes each UV albedo map as input and outputs 7 physical
measurements via a frozen forward pass (no fine-tuning):

```
melanin_mode      -> melanin concentration (mode)
melanin_std       -> melanin concentration (std)
hemoglobin_mode   -> hemoglobin concentration (mode)
hemoglobin_std    -> hemoglobin concentration (std)
eumelanin_ratio   -> eumelanin/pheomelanin ratio
oxygenation       -> blood oxygenation
epidermal_thick   -> epidermal thickness
age               -> age (0-100)
gender            -> gender (0=female, 1=male)
```

---

## Custom Modules

### ConditionEmbedder

Converts 9 biophysical scalars into condition vector C for UNet cross-attention injection:

```
scalars (B, 9) + timestep (B,)
        |
sinusoidal encoding per scalar (B, 1280)
        |
independent MLP per scalar (B, 1280)
        |
concatenate all 10 embeddings (B, 12800)
        |
linear projection (B, 77*1024)
        |
reshape -> C_c (B, 77, 1024)   <- ready for UNet cross-attention
```

**Dual CFG at inference (from paper):**
```
v = v(null, null)
  + s_p * (v(null, C_p) - v(null, null))      <- modality guidance
  + s_c * (v(C_c,  C_p) - v(null, C_p))       <- skin guidance
```

---

## Repository Structure

```
FFHQ-UV/
|
|-- ldm_pretrain/                  <- SD 2.1 model components
|   |-- unet/
|   |   |-- config.json            <- UNet architecture (read by train_ldm.py)
|   |   `-- diffusion_pytorch_model.safetensors
|   |-- vae/
|   |   |-- config.json            <- VAE architecture
|   |   `-- diffusion_pytorch_model.safetensors
|   |-- text_encoder/
|   |   |-- config.json            <- Text encoder architecture
|   |   `-- model.safetensors
|   |-- tokenizer/
|   |-- scheduler/
|   |   `-- scheduler_config.json  <- DDIMScheduler config
|   |-- model_index.json
|   |-- outputs/                   <- Checkpoints saved here
|   |-- logs/                      <- TensorBoard logs
|   `-- samples/                   <- Sample images saved here
|
|-- dataset/                       <- FFHQ-UV Arrow dataset
|-- skin_params_biophysical.csv    <- 9 biophysical scalars per face
|
|-- train_ldm.py                   <- Main training script
|-- config.yaml                    <- Training hyperparameters
|-- dataset_loader.py              <- Custom dataset loader
|-- ConditionEmbedder.py           <- Biophysical condition embedder
|-- ClipTextModel.py               <- CLIP text encoder setup
`-- README.md
```

---

## Environment Setup

### Requirements

- Python 3.10
- CUDA 12.8+ (required for NVIDIA Blackwell RTX 50xx series)
- Windows 10/11 or Linux

### Installation

```bash
# Create conda environment
conda create -n ldm_pretrain python=3.10 -y
conda activate ldm_pretrain

# Install PyTorch with CUDA 12.8 (required for RTX 50xx Blackwell)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt 
```

### Verify GPU

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory/1e9, 'GB')
"
```

---



## Training Configuration

All hyperparameters in `config.yaml`. Architecture values are read automatically from JSON configs.

```yaml
training:
  batch_size:     2        # per GPU
  grad_accum:     4        # effective batch = 8
  epochs:         20
  max_steps:      50000    # ~13 epochs on 30K dataset
  learning_rate:  1e-5
  num_workers:    0        # set to 0 on Windows

model:
  mixed_precision: "bf16"  # native for Blackwell RTX 50xx

condition_embedder:
  num_scalars: 9            # 7 biophysical + age + gender
  d:           1280
  T:           1000
```


---

## Checkpoint Structure

Each checkpoint saves full training state for seamless resume:

```
outputs/checkpoint-step-5000/
|-- unet/                    <- UNet weights (diffusers format)
|-- embedder.pt              <- ConditionEmbedder weights
`-- training_state.pt        <- optimizer + lr_scheduler + step
```

---


## References

```bibtex
@misc{rombach2022highresolutionimagesynthesislatent,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2022},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2112.10752}, 
}

@inproceedings{zeng2024rgb,
author = {Zeng, Zheng and Deschaintre, Valentin and Georgiev, Iliyan and Hold-Geoffroy, Yannick and Hu, Yiwei and Luan, Fujun and Yan, Ling-Qi and Ha\v{s}an, Milo\v{s}},
title = {RGB↔X: Image decomposition and synthesis using material- and lighting-aware diffusion models},
year = {2024},
isbn = {9798400705250},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3641519.3657445},
doi = {10.1145/3641519.3657445},
booktitle = {ACM SIGGRAPH 2024 Conference Papers},
articleno = {75},
numpages = {11},
keywords = {Diffusion models, intrinsic decomposition, realistic rendering},
location = {Denver, CO, USA},
series = {SIGGRAPH '24}
}


@article {10.1111:cgf.14887,
journal = {Computer Graphics Forum},
title = {{A Hyperspectral Space of Skin Tones for Inverse Rendering of Biophysical Skin Properties}},
author = {Aliaga, Carlos and Xia, Mengqi and Xie, Hao and Jarabo, Adrian and Braun, Gustav and Hery, Christophe},
year = {2023},
publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
ISSN = {1467-8659},
DOI = {10.1111/cgf.14887}
}
```

---

## License

- Code: MIT License
- SD 2.1 weights: [CreativeML Open RAIL-M](https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/LICENSE.md)
- FFHQ-UV dataset: Research use only