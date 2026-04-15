# check_maps.py
import os
from datasets import load_from_disk
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dataset_path = r"D:\Github\PhD Code\FFHQ-UV\dataset"
dataset = load_from_disk(dataset_path)["train"]

# Check first 3 samples
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i in range(3):
    sample = dataset[i]

    albedo = sample["albedo_path"]
    normal = sample["normal_path"]
    sh     = sample["sh_path"]
    lit    = sample["lit_path"]

    for img, ax, title in zip(
        [albedo, normal, sh, lit],
        axes[i],
        ["Albedo", "Normal", "SH", "Lit"]
    ):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        ax.imshow(img)
        ax.set_title(f"Sample {i} — {title}")
        ax.axis("off")

        # Print pixel stats
        arr = np.array(img)
        print(f"Sample {i} {title}: shape={arr.shape} "
              f"min={arr.min()} max={arr.max()} "
              f"mean={arr.mean():.1f}")

plt.tight_layout()
plt.savefig("map_check.png")
print("\nSaved map_check.png")