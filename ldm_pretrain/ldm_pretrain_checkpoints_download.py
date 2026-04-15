import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

save_dir = r"D:\Github\PhD Code\FFHQ-UV\ldm_pretrain"
os.makedirs(save_dir, exist_ok=True)

print("📥 Downloading SD 2.1 from community mirror (no license gate)...")

pipeline = StableDiffusionPipeline.from_pretrained(
    "sd2-community/stable-diffusion-2-1",   # ← public mirror, no token needed
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipeline.save_pretrained(save_dir)

print(f"\n✅ Saved to: {save_dir}")
print(f"📂 Components: {os.listdir(save_dir)}")