import os
import sys
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

# Windows encoding fix
sys.stdout.reconfigure(encoding="utf-8")

class FFHQUVDataset(Dataset):
    """
    Dataset loader for UV-unwrapped albedo face maps stored as PNG files.

    Loads images directly from a folder — no Arrow format needed.
    Skin parameters come separately from SkinParamDataset (CSV).

    Folder structure expected:
        D:/Github/PhD Code/Biophysical-LDM/dataset/Final UV Maps/
            00001.png
            00002.png
            ...
            54165.png
    """

    def __init__(self,
                 resolution  = 768,
                 dataset_dir = None,
                 max_samples = None):
        super().__init__()

        self.resolution = resolution

        # ── Dataset folder ─────────────────────────────────────
        if dataset_dir is None:
            dataset_dir = r"D:\Github\PhD Code\Biophysical-LDM\dataset\Final UV Maps"

        print(f"Loading PNG images from: {dataset_dir}")

        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(
                f"Dataset folder not found: {dataset_dir}\n"
                f"Please check the path in config.yaml → paths.dataset_dir"
            )

        # ── Collect all PNG files ──────────────────────────────
        valid_exts  = {".png", ".jpg", ".jpeg"}
        self.image_paths = sorted([
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if os.path.splitext(f.lower())[1] in valid_exts
        ])

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No PNG/JPG images found in: {dataset_dir}"
            )

        if max_samples:
            self.image_paths = self.image_paths[:max_samples]

        print(f"Found {len(self.image_paths):,} images")
        print(f"Resolution: {resolution}x{resolution}px")
        print(f"Example: {os.path.basename(self.image_paths[0])}")

        # ── Image transforms → normalize to [-1, 1] for LDM ───
        self.transform = T.Compose([
            T.Resize(
                (resolution, resolution),
                interpolation = T.InterpolationMode.BILINEAR,
                antialias     = True,
            ),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),   # [0, 1] -> [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # ── Load image ─────────────────────────────────────────
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        pixel_values = self.transform(image)

        return {
            "pixel_values": pixel_values,              # (3, H, W) in [-1, 1]
            "image_path":   img_path,                  # for debugging
        }


# ── Quick test ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("Testing dataset loader...\n")

    dataset = FFHQUVDataset(resolution=768, max_samples=5)

    sample = dataset[0]
    print("\nSample keys     :", list(sample.keys()))
    print("pixel_values    :", sample["pixel_values"].shape)
    print("pixel range     :",
          sample["pixel_values"].min().item(),
          "to",
          sample["pixel_values"].max().item())
    print("image_path      :", os.path.basename(sample["image_path"]))
    print("\nDataset loader working!")
    print(f"Total samples   : {len(dataset):,}")