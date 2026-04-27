import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────
# STEP 1 — Dataset class
# ─────────────────────────────────────────────────────────────
class SkinParamDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        # ✅ 9 columns — 7 biophysical + age + gender
        scalar_cols = [
            'melanin_mode',
            'melanin_std',
            'hemoglobin_mode',
            'hemoglobin_std',
            'eumelanin_ratio',
            'oxygenation',
            'epidermal_thick',
            'age',           # ← added
            'gender',        # ← added
        ]

        self.ids     = list(range(len(df)))   # 0, 1, 2, 3...
        self.scalars = torch.tensor(df[scalar_cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            'id':      self.ids[idx],
            'scalars': self.scalars[idx],   # (9,)
        }


# ─────────────────────────────────────────────────────────────
# STEP 2 — DataLoader
# ─────────────────────────────────────────────────────────────
dataset    = SkinParamDataset(r"D:\Github\PhD Code\Biophysical-LDM\Pretrain_Model\skin_params_final_uv.csv")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# ─────────────────────────────────────────────────────────────
# STEP 3 — ConditionEmbedder
# ─────────────────────────────────────────────────────────────
class ConditionEmbedder(nn.Module):
    """
    Converts 9 raw scalars (7 biophysical + age + gender)
    into condition vector C → shape (batch, 77, 1024)
    ready for SD 2.1 UNet cross-attention injection.
    """

    def __init__(self,
                 num_scalars = 9,     # ✅ 7 skin + age + gender
                 d           = 1280,  # sinusoidal + MLP hidden dim
                 T           = 1000,  # timestep range
                 seq_len     = 77,    # CLIP sequence length
                 unet_dim    = 1024): # SD 2.1 cross-attention dim
        super().__init__()
        self.d       = d
        self.T       = T
        self.seq_len = seq_len
        self.unet_dim= unet_dim

        # ✅ One MLP per scalar — SiLU not ReLU
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d),
                nn.SiLU(),    # ← changed from ReLU
                nn.Linear(d, d),
                nn.SiLU(),    # ← changed from ReLU
            )
            for _ in range(num_scalars)
        ])

        # ✅ Dedicated MLP for timestep
        self.t_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
            nn.SiLU(),
        )

        # ✅ Projection: (batch, 12800) → (batch, 77, 1024)
        self.proj = nn.Linear((num_scalars + 1) * d, seq_len * unet_dim)

    def sinusoidal_encode(self, c):
        """
        c: (batch,) — one scalar
        returns: (batch, d=1280)
        """
        half  = self.d // 2
        freqs = torch.pow(
            self.T,
            -torch.arange(half, dtype=torch.float32) * 2 / self.d
        ).to(c.device, c.dtype)          # ← add c.dtype here
        args  = c.unsqueeze(1) * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, scalars, timestep):
        """
        scalars:  (batch, 9)  — 9 condition values per face
        timestep: (batch,)    — diffusion timestep t
        returns:  (batch, 77, 1024) — ready for UNet cross-attention
        """
        target_dtype = self.mlps[0][0].weight.dtype
        scalars      = scalars.to(target_dtype)
        timestep     = timestep.to(target_dtype)

        embeddings = []
        for i, mlp in enumerate(self.mlps):
            sin_enc = self.sinusoidal_encode(scalars[:, i])
            emb     = mlp(sin_enc)
            embeddings.append(emb)

        t_enc = self.sinusoidal_encode(timestep.to(target_dtype))
        t_emb = self.t_mlp(t_enc)
        embeddings.append(t_emb)

        C = torch.cat(embeddings, dim=-1)
        C = self.proj(C)
        C = C.view(-1, self.seq_len, self.unet_dim)
        return C


# ─────────────────────────────────────────────────────────────
# STEP 4 — Quick test
# ─────────────────────────────────────────────────────────────
embedder = ConditionEmbedder(num_scalars=9, d=1280)

for batch in dataloader:
    scalars  = batch['scalars']                                    # (4, 9)
    timestep = torch.randint(0, 1000, (scalars.size(0),)).float() # (4,)

    C = embedder(scalars, timestep)
    print(f"✅ C shape: {C.shape}")   # → (4, 77, 1024)
    break