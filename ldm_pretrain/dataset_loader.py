import os
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from PIL import Image
import torchvision.transforms as T

class FFHQUVDataset(Dataset):

    def __init__(self, resolution=512, max_samples=None):
        super().__init__()
        self.resolution = resolution

        # ── Load directly from local Arrow files ──────────────
        dataset_path = r'D:\Github\PhD Code\Biophysical-LDM\dataset'
        print(f"📂 Loading from local Arrow files: {dataset_path}")

        self.dataset = load_from_disk(dataset_path)

        # If it loaded as DatasetDict, get the train split
        if hasattr(self.dataset, 'keys'):
            print(f"📦 Splits found: {list(self.dataset.keys())}")
            self.dataset = self.dataset['train']

        if max_samples:
            self.dataset = self.dataset.select(range(max_samples))

        print(f"✅ Loaded {len(self.dataset)} samples")
        print(f"📋 Columns: {self.dataset.column_names}")

        # Image transforms → resize + normalize to [-1, 1] for LDM
        self.transform = T.Compose([
            T.Resize((resolution, resolution),
                     interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # [0,1] → [-1,1]
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # ── Load albedo image ──────────────────────────────────
        albedo = sample["albedo_path"]
        if not isinstance(albedo, Image.Image):
            albedo = Image.fromarray(albedo)
        albedo = albedo.convert("RGB")
        pixel_values = self.transform(albedo)

        # ── Text caption ───────────────────────────────────────
        text = sample.get("text", "") or "a UV unwrapped face albedo texture map"

        # ── Condition metadata ─────────────────────────────────
        age    = float(sample.get("age",   0) or 0) / 100.0
        gender = float(sample.get("gender",0) or 0)
        beard  = float(sample.get("bread", 0) or 0)  # "bread" = beard

        return {
            "pixel_values": pixel_values,
            "text":         text,
            "age":          torch.tensor(age,    dtype=torch.float32),
            "gender":       torch.tensor(gender, dtype=torch.float32),
            "beard":        torch.tensor(beard,  dtype=torch.float32),
        }


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Testing dataset loader...\n")
    dataset = FFHQUVDataset(resolution=512, max_samples=5)

    sample = dataset[0]
    print("\n📊 Sample keys    :", list(sample.keys()))
    print("🖼️  pixel_values   :", sample["pixel_values"].shape)
    print("📝 text            :", sample["text"])
    print("👤 age             :", sample["age"].item())
    print("⚧  gender          :", sample["gender"].item())
    print("🧔 beard            :", sample["beard"].item())
    print("\n✅ Dataset loader working!")