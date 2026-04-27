"""
train_ldm.py  LDM Pre-training Script
SD 2.1 fine-tuned on FFHQ-UV albedo maps

Architecture values read directly from model JSON configs.
Only training hyperparameters come from config.yaml.

Usage:
    # Fresh start
    python train_ldm.py --config config.yaml

    # Resume from checkpoint
    python train_ldm.py --config config.yaml --resume outputs/checkpoint-step-11000
"""

# ── Imports ────────────────────────────────────────────────────
import os
import sys
import json
import argparse
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from PIL import Image
import numpy as np
import yaml
from dataset_loader import FFHQUVDataset
from ConditionEmbedder import ConditionEmbedder, SkinParamDataset

# ── Windows encoding fix ───────────────────────────────────────
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ── Argument Parser ────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml",
                    help="Path to config YAML file")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint folder to resume from "
                         "e.g. outputs/checkpoint-step-11000")
args = parser.parse_args()

# ── Load YAML config ───────────────────────────────────────────
with open(args.config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

P   = cfg["paths"]
M   = cfg["model"]
T   = cfg["training"]
LOG = cfg["logging"]
G   = cfg["guidance"]
CE  = cfg["condition_embedder"]
INF = cfg["inference"]
MP  = cfg["modality_prompts"]

# ── Reproducibility ────────────────────────────────────────────
torch.manual_seed(T["seed"])
np.random.seed(T["seed"])

# ── Setup Dirs ─────────────────────────────────────────────────
for d in [P["output_dir"], P["log_dir"], P["sample_dir"]]:
    os.makedirs(d, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler(
            os.path.join(P["log_dir"], "train.log"),
            encoding = "utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── Device & Dtype ─────────────────────────────────────────────
device = torch.device("cuda")
dtype  = torch.bfloat16 if M["mixed_precision"] == "bf16" \
         else torch.float16

logger.info(f"GPU   : {torch.cuda.get_device_name(0)}")
logger.info(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
logger.info(f"dtype : {dtype}")
logger.info(f"seed  : {T['seed']}")
if args.resume:
    logger.info(f"resume: {args.resume}")

# ══════════════════════════════════════════════════════════════
#  READ ARCHITECTURE FROM JSON CONFIGS
# ══════════════════════════════════════════════════════════════
logger.info("\nReading architecture from JSON configs...")

model_dir = P["model_dir"]

with open(os.path.join(model_dir, "unet/config.json"),
          encoding="utf-8") as f:
    unet_json = json.load(f)

with open(os.path.join(model_dir, "vae/config.json"),
          encoding="utf-8") as f:
    vae_json = json.load(f)

with open(os.path.join(model_dir, "text_encoder/config.json"),
          encoding="utf-8") as f:
    te_json = json.load(f)

with open(os.path.join(model_dir, "scheduler/scheduler_config.json"),
          encoding="utf-8") as f:
    sch_json = json.load(f)

# ── Extract architecture values ────────────────────────────────
cross_attention_dim = unet_json["cross_attention_dim"]    # 1024
unet_sample_size    = unet_json["sample_size"]            # 96
unet_in_channels    = unet_json["in_channels"]            # 4
scaling_factor      = vae_json["scaling_factor"]          # 0.18215
latent_channels     = vae_json["latent_channels"]         # 4
hidden_size         = te_json["hidden_size"]              # 1024
max_token_length    = te_json["max_position_embeddings"]  # 77
prediction_type     = sch_json["prediction_type"]         # v_prediction
num_train_timesteps = sch_json["num_train_timesteps"]     # 1000
resolution          = unet_sample_size * 8                # 768
latent_size         = unet_sample_size                    # 96

logger.info(f"  unet     : cross_attention_dim={cross_attention_dim}"
            f" | sample_size={unet_sample_size}"
            f" | in_channels={unet_in_channels}")
logger.info(f"  vae      : scaling_factor={scaling_factor}"
            f" | latent_channels={latent_channels}")
logger.info(f"  text_enc : hidden_size={hidden_size}"
            f" | max_length={max_token_length}")
logger.info(f"  scheduler: prediction_type={prediction_type}"
            f" | timesteps={num_train_timesteps}")
logger.info(f"  derived  : resolution={resolution}px"
            f" | latent_size={latent_size}")

# ══════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════
logger.info("\nLoading SD 2.1 components...")

tokenizer = CLIPTokenizer.from_pretrained(
    model_dir, subfolder="tokenizer")

text_encoder = CLIPTextModel.from_pretrained(
    model_dir,
    subfolder   = "text_encoder",
    torch_dtype = dtype,
).to(device)
text_encoder.requires_grad_(False)
text_encoder.eval()
logger.info(f"  Text encoder : hidden_size={hidden_size} | frozen")

vae = AutoencoderKL.from_pretrained(
    model_dir,
    subfolder   = "vae",
    torch_dtype = dtype,
).to(device)
vae.requires_grad_(False)
vae.eval()
logger.info(f"  VAE          : scaling_factor={scaling_factor} | frozen")

# ── UNet — load from checkpoint if resuming, else from base ───
if args.resume:
    unet_load_path = os.path.join(args.resume, "unet")
    logger.info(f"  UNet         : loading from checkpoint {unet_load_path}")
else:
    unet_load_path = model_dir
    logger.info(f"  UNet         : loading from base model")

unet = UNet2DConditionModel.from_pretrained(
    unet_load_path,
    subfolder   = None if args.resume else "unet",
    torch_dtype = dtype,
).to(device)
unet.train()
unet.enable_gradient_checkpointing()
logger.info(f"  UNet         : grad_checkpointing=ON | trainable")

noise_scheduler = DDIMScheduler.from_pretrained(
    model_dir, subfolder="scheduler")
logger.info(f"  Scheduler    : {sch_json['_class_name']}"
            f" | {prediction_type}")

# ── ConditionEmbedder ─────────────────────────────────────────
embedder = ConditionEmbedder(
    num_scalars = CE["num_scalars"],
    d           = CE["d"],
    T           = CE["T"],
    seq_len     = max_token_length,
    unet_dim    = cross_attention_dim,
).to(device, dtype=dtype)

# Load embedder from checkpoint if resuming
if args.resume:
    emb_path = os.path.join(args.resume, "embedder.pt")
    embedder.load_state_dict(torch.load(emb_path,
                             map_location=device))
    logger.info(f"  ConditionEmb : loaded from checkpoint")
else:
    logger.info(f"  ConditionEmb : initialized fresh")

embedder.train()
logger.info(f"  ConditionEmb : scalars={CE['num_scalars']}"
            f" | seq_len={max_token_length}"
            f" | unet_dim={cross_attention_dim} | trainable")
logger.info(f"VRAM after loading : {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

# ══════════════════════════════════════════════════════════════
#  DATASETS
# ══════════════════════════════════════════════════════════════
logger.info("Loading datasets...")

image_dataset = FFHQUVDataset(
    resolution  = resolution,
    dataset_dir = P["dataset_dir"],
)
skin_dataset  = SkinParamDataset(P["skin_params"])

logger.info(f"  Image dataset : {len(image_dataset):,} samples"
            f" | resolution={resolution}px")
logger.info(f"  Skin params   : {len(skin_dataset):,} samples")

image_loader = DataLoader(
    image_dataset,
    batch_size  = T["batch_size"],
    shuffle     = True,
    num_workers = T["num_workers"],
    pin_memory  = True,
    drop_last   = True,
)

skin_loader = DataLoader(
    skin_dataset,
    batch_size  = T["batch_size"],
    shuffle     = True,
    num_workers = T["num_workers"],
    pin_memory  = True,
    drop_last   = True,
)
logger.info(f"  Batches/epoch : {len(image_loader):,}\n")

# ══════════════════════════════════════════════════════════════
#  OPTIMIZER + LR SCHEDULER
# ══════════════════════════════════════════════════════════════
optimizer = torch.optim.AdamW(
    list(unet.parameters()) + list(embedder.parameters()),
    lr           = float(T["learning_rate"]),
    betas        = (float(T["adam_beta1"]), float(T["adam_beta2"])),
    weight_decay = float(T["adam_weight_decay"]),
    eps          = float(T["adam_eps"]),
)

lr_scheduler = OneCycleLR(
    optimizer,
    max_lr      = float(T["learning_rate"]),
    total_steps = T["max_steps"],
    pct_start   = T["lr_warmup_steps"] / T["max_steps"],
)

# ── Resume optimizer + lr state ────────────────────────────────
global_step   = 0
start_epoch   = 0

if args.resume:
    state_path = os.path.join(args.resume, "training_state.pt")
    if os.path.exists(state_path):
        state         = torch.load(state_path, map_location=device)
        global_step   = state["global_step"]
        start_epoch   = state.get("epoch", 0)
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        logger.info(f"Resumed optimizer + lr_scheduler state")
        logger.info(f"Resuming from step={global_step} | epoch={start_epoch}")
    else:
        import re
        match = re.search(r"checkpoint-step-(\d+)", args.resume)
        if match:
            global_step  = int(match.group(1))         # 11000
            steps_per_epoch = len(image_loader) // T["grad_accum"]
            start_epoch  = global_step // steps_per_epoch + 1  # +1 because current epoch finished
            # Cap at max epochs
            start_epoch  = min(start_epoch, T["epochs"] - 1)
            logger.info(f"Inferred: step={global_step} | starting epoch={start_epoch+1}")
        logger.info("Optimizer restarted — LR will warmup again")

# ── TensorBoard ────────────────────────────────────────────────
writer = SummaryWriter(log_dir=P["log_dir"])

# ── Active modality prompt ─────────────────────────────────────
active_prompt = MP[MP["active"]]
logger.info(f"Active modality : '{active_prompt}'\n")


# ══════════════════════════════════════════════════════════════
#  HELPER — encode text
# ══════════════════════════════════════════════════════════════
def encode_text(prompt):
    """Frozen CLIP encode. Returns (1, 77, 1024)"""
    tokens = tokenizer(
        prompt,
        padding        = "max_length",
        max_length     = max_token_length,
        truncation     = True,
        return_tensors = "pt",
    ).input_ids.to(device)
    with torch.no_grad():
        return text_encoder(tokens)[0]


# ══════════════════════════════════════════════════════════════
#  SAMPLE IMAGE SAVING
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def save_samples(step, sample_scalars):
    """
    Generate + save sample images at checkpoint.
    Single UNet pass - VRAM efficient for monitoring.
    Skin params still drive generation via ConditionEmbedder.
    """
    logger.info(f"\nSaving samples at step {step}...")
    unet.eval()
    embedder.eval()

    n           = min(LOG["num_samples"],
                      len(cfg["sample_prompts"]),
                      sample_scalars.shape[0])
    cond_hidden = encode_text(active_prompt)
    images      = []

    for i in range(n):
        scalars_i = sample_scalars[i:i+1].to(device, dtype=dtype)
        latents   = torch.randn(
            1, latent_channels, latent_size, latent_size,
            device=device, dtype=dtype,
        )
        noise_scheduler.set_timesteps(INF["num_steps"])

        for t in noise_scheduler.timesteps:
            t_batch = t.unsqueeze(0).to(device)
            dummy_t = torch.zeros(
                scalars_i.shape[0], device=device, dtype=dtype)
            C_c      = embedder(scalars_i, dummy_t)
            combined = cond_hidden + C_c
            v_pred   = unet(
                latents, t_batch,
                encoder_hidden_states = combined,
            ).sample
            latents  = noise_scheduler.step(
                v_pred, t, latents,
                eta = INF["eta"],
            ).prev_sample

        decoded = vae.decode(latents / scaling_factor).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = decoded.squeeze(0).permute(1, 2, 0)
        decoded = (decoded.float().cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(decoded))

    w, h  = images[0].size
    cols  = min(n, 2)
    rows  = (n + cols - 1) // cols
    grid  = Image.new("RGB", (cols * w, rows * h))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(img, (c * w, r * h))

    path = os.path.join(P["sample_dir"], f"step_{step:06d}.png")
    grid.save(path)
    logger.info(f"Sample saved -> {path}")

    grid_t = torch.from_numpy(
        np.array(grid)).permute(2, 0, 1).float() / 255.0
    writer.add_image("samples", grid_t, step)

    torch.cuda.empty_cache()
    unet.train()
    embedder.train()


# ══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════
logger.info("\nStarting pre-training...\n")
optimizer.zero_grad()
skin_iter = iter(skin_loader)

# Step 0 baseline — skip if resuming
if global_step == 0:
    baseline_skin = next(iter(skin_loader))["scalars"]
    save_samples(0, baseline_skin)

try:
    for epoch in range(start_epoch, T["epochs"]):
        epoch_loss = 0.0
        pbar = tqdm(image_loader,
                    desc=f"Epoch {epoch+1}/{T['epochs']}"
                         f" | step {global_step}")

        for step, batch in enumerate(pbar):

            # Skip batches already trained when resuming
            # (approximate skip by global_step position)
            batches_done_this_epoch = (
                global_step - start_epoch *
                (len(image_loader) // T["grad_accum"])
            ) * T["grad_accum"]

            # Cycle skin loader
            try:
                skin_batch = next(skin_iter)
            except StopIteration:
                skin_iter  = iter(skin_loader)
                skin_batch = next(skin_iter)

            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            scalars      = skin_batch["scalars"].to(device, dtype=dtype)
            texts        = [active_prompt] * pixel_values.shape[0]

            with torch.amp.autocast("cuda", dtype=dtype):

                # 1 -- VAE encode
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * scaling_factor

                # 2 -- Noise + timesteps
                noise     = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, num_train_timesteps,
                    (latents.shape[0],), device=device,
                ).long()

                # 3 -- Forward diffusion
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps)

                # 4 -- CLIP encode
                text_inputs = tokenizer(
                    texts,
                    padding        = "max_length",
                    max_length     = max_token_length,
                    truncation     = True,
                    return_tensors = "pt",
                ).input_ids.to(device)

                with torch.no_grad():
                    C_p = text_encoder(text_inputs)[0]

                # 5 -- ConditionEmbedder
                C_c = embedder(scalars, timesteps.to(dtype))

                # 6 -- Combine + UNet forward
                combined = C_p + C_c
                v_pred   = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states = combined,
                ).sample

                # 7 -- v_prediction loss
                if prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    target = noise

                loss = F.mse_loss(
                    v_pred.float(),
                    target.float(),
                    reduction = "mean",
                ) / T["grad_accum"]

            loss.backward()

            if (step + 1) % T["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(unet.parameters()) +
                    list(embedder.parameters()),
                    T["max_grad_norm"],
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                loss_val   = loss.item() * T["grad_accum"]
                epoch_loss += loss_val
                lr_now     = optimizer.param_groups[0]["lr"]

                if global_step % LOG["log_every"] == 0:
                    writer.add_scalar("train/loss",
                                      loss_val, global_step)
                    writer.add_scalar("train/lr",
                                      lr_now,   global_step)
                    writer.add_scalar("train/vram_gb",
                        torch.cuda.memory_allocated()/1e9,
                        global_step)
                    pbar.set_postfix({
                        "loss": f"{loss_val:.4f}",
                        "lr":   f"{lr_now:.2e}",
                        "step": global_step,
                        "vram": f"{torch.cuda.memory_allocated()/1e9:.1f}GB",
                    })

                if global_step % LOG["sample_every"] == 0:
                    s = skin_batch["scalars"][:LOG["num_samples"]]
                    save_samples(global_step, s)

                # ── Save checkpoint + full training state ──────
                if global_step % LOG["save_every"] == 0:
                    ckpt = os.path.join(
                        P["output_dir"],
                        f"checkpoint-step-{global_step}")
                    os.makedirs(ckpt, exist_ok=True)

                    # Model weights
                    unet.save_pretrained(os.path.join(ckpt, "unet"))
                    torch.save(embedder.state_dict(),
                               os.path.join(ckpt, "embedder.pt"))

                    # Full training state for resume
                    torch.save({
                        "global_step":  global_step,
                        "epoch":        epoch,
                        "optimizer":    optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "loss":         loss_val,
                    }, os.path.join(ckpt, "training_state.pt"))

                    logger.info(
                        f"Checkpoint + training state -> {ckpt}")

                if global_step >= T["max_steps"]:
                    break

        avg = epoch_loss / max(len(image_loader), 1)
        logger.info(f"Epoch {epoch+1} avg loss: {avg:.4f}")
        writer.add_scalar("train/epoch_loss", avg, epoch)

        if global_step >= T["max_steps"]:
            break

except Exception as e:
    logger.error(
        f"Training crashed at step {global_step}: {e}",
        exc_info=True)
    ckpt = os.path.join(P["output_dir"],
                        f"emergency-step-{global_step}")
    os.makedirs(ckpt, exist_ok=True)
    unet.save_pretrained(os.path.join(ckpt, "unet"))
    torch.save(embedder.state_dict(),
               os.path.join(ckpt, "embedder.pt"))
    torch.save({
        "global_step":  global_step,
        "epoch":        epoch,
        "optimizer":    optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }, os.path.join(ckpt, "training_state.pt"))
    logger.info(f"Emergency checkpoint -> {ckpt}")
    raise

# ── Save Final Model ───────────────────────────────────────────
final = os.path.join(P["output_dir"], "final")
os.makedirs(final, exist_ok=True)
unet.save_pretrained(os.path.join(final, "unet"))
torch.save(embedder.state_dict(),
           os.path.join(final, "embedder.pt"))
torch.save({
    "global_step":  global_step,
    "epoch":        epoch,
    "optimizer":    optimizer.state_dict(),
    "lr_scheduler": lr_scheduler.state_dict(),
}, os.path.join(final, "training_state.pt"))

logger.info(f"\nTraining complete!")
logger.info(f"  UNet     -> {os.path.join(final, 'unet')}")
logger.info(f"  Embedder -> {os.path.join(final, 'embedder.pt')}")
logger.info(f"  State    -> {os.path.join(final, 'training_state.pt')}")
writer.close()