from transformers import CLIPTextModel, CLIPTokenizer
import torch



# ─────────────────────────────────────────────────────────────
# CLIP setup — load once, freeze completely
# This never gets trained — purely frozen inference
# ─────────────────────────────────────────────────────────────
clip_model     = CLIPTextModel.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
clip_tokenizer = CLIPTokenizer.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

clip_model.eval()
clip_model.requires_grad_(False)


# ─────────────────────────────────────────────────────────────
# The 3 hardcoded prompts — written once, never changed
# ─────────────────────────────────────────────────────────────
PROMPTS = {
    'deformation': "Vector field",
    'albedo':      "Albedo Map",
    'height':      "Bump and Displacement Map",
}


def encode_prompt(prompt_text):
    """
    Encode a text prompt into τ(p) embedding via frozen CLIP.
    Run once per prompt and cache the result — never re-run during training.
    """
    tokens = clip_tokenizer(
        prompt_text,
        return_tensors = "pt",
        padding        = True,
        truncation     = True,
        max_length     = 77
    )
    with torch.no_grad():
            output = clip_model(**tokens)
            # last_hidden_state for cross-attention — shape (1, 77, 1024)
            return output.last_hidden_state  # τ(p)


# ─────────────────────────────────────────────────────────────
# Pre-compute and cache all 3 embeddings at startup
# You never call CLIP again after this point
# ─────────────────────────────────────────────────────────────
tau = {
    name: encode_prompt(text)
    for name, text in PROMPTS.items()
}

# tau['deformation'] → tensor shape (1, 77, 1024)
# tau['albedo']      → tensor shape (1, 77, 1024)
# tau['height']      → tensor shape (1, 77, 1024)