# ===== C4 Perplexity Eval (streaming) =====
# Matches the HF sliding-window method and the SqueezeLLM paper's datasets (C4 + WikiText-2). 
# Ref: HF PPL guide (sliding window) and SqueezeLLM eval on C4. 
# - HF: https://huggingface.co/docs/transformers/en/perplexity
# - SqueezeLLM paper uses C4 + WikiText-2 for PPL comparisons.

import os, math, time, torch
from itertools import chain
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# ========= CONFIG =========
MODEL_OBJ_PATH   = "/workspace/SqueezeLLM/Packed_obj.pt"         # your quantized torch.save(...) object
BASE_MODEL_PATH  = "/workspace/SqueezeLLM/models/Llama-2-7b-hf"  # tokenizer path
C4_CONFIG        = "en"             # C4 English (cleaned)
C4_SPLIT         = "validation"     # standard for PPL
STREAMING        = True             # avoids Apache Beam install
TARGET_TOKENS    = 3_000_000        # cap tokens for a quick, comparable run; set to None for full split
MAX_LENGTH       = 2048             # context len used in many papers
STRIDE           = 2048             # only last STRIDE tokens per window are predicted
LOG_EVERY        = 50
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================

print("HF_DATASETS_CACHE:", os.environ.get("HF_DATASETS_CACHE"))

# --- tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- unpickle safety for your quantized object ---
import transformers.models.llama.modeling_llama
import transformers.models.llama.configuration_llama
import squeezellm.quant
torch.serialization.add_safe_globals([
    transformers.models.llama.modeling_llama.LlamaForCausalLM,
    transformers.models.llama.modeling_llama.LlamaModel,
    torch.nn.modules.sparse.Embedding,
    torch.nn.modules.container.ModuleList,
    transformers.models.llama.modeling_llama.LlamaDecoderLayer,
    transformers.models.llama.modeling_llama.LlamaAttention,
    squeezellm.quant.QuantLinearLUT,
    transformers.models.llama.configuration_llama.LlamaConfig,
    transformers.models.llama.modeling_llama.LlamaMLP,
    torch.nn.modules.activation.SiLU,
    transformers.models.llama.modeling_llama.LlamaRMSNorm,
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding,
    transformers.modeling_rope_utils._compute_default_rope_parameters,
    torch.nn.modules.linear.Linear,
    transformers.generation.configuration_utils.GenerationConfig,
])

print("Loading quantized model...")
model = torch.load(MODEL_OBJ_PATH, map_location=DEVICE, weights_only=False).to(DEVICE).eval()
torch.set_grad_enabled(False)
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# --- load C4 (streaming) ---
print(f"Loading C4:{C4_CONFIG} [{C4_SPLIT}] (streaming={STREAMING}) ...")
ds = load_dataset("allenai/c4", C4_CONFIG, split=C4_SPLIT, streaming=STREAMING)

# --- tokenize & collect up to TARGET_TOKENS ---
print("Tokenizing stream...")
all_ids = []
tokd = 0
pbar = tqdm(total=(TARGET_TOKENS if TARGET_TOKENS else None), desc="Tokens")
for ex in ds:
    text = ex.get("text") or ""
    if not text:
        continue
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        continue
    # ensure continuity between docs
    all_ids.extend(ids)
    tokd += len(ids)
    if TARGET_TOKENS and tokd >= TARGET_TOKENS:
        # make sure we have at least one extra token for shifting
        if len(all_ids) < TARGET_TOKENS + 1:
            all_ids.extend([tokenizer.eos_token_id])
        break
    if TARGET_TOKENS:
        pbar.update(len(ids))
if TARGET_TOKENS:
    pbar.close()

input_ids = torch.tensor(all_ids, dtype=torch.long)
del all_ids  # free CPU RAM

# --- sliding-window perplexity ---
print("Computing perplexity...")
total_steps = max(1, (input_ids.size(0) - 1) // STRIDE)
nlls = []
total_pred_tokens = 0
t0 = time.time()

for step, i in enumerate(tqdm(range(0, input_ids.size(0) - 1, STRIDE), total=total_steps, desc="Chunks")):
    begin = max(i + STRIDE - MAX_LENGTH, 0)
    end   = min(i + STRIDE, input_ids.size(0) - 1)
    trg_len = end - i
    if trg_len <= 0:
        break

    chunk = input_ids[begin:end].unsqueeze(0).to(DEVICE)
    targets = chunk.clone()
    targets[:, :-trg_len] = -100  # only predict the last STRIDE tokens

    with torch.no_grad():
        out = model(chunk, labels=targets)
        nlls.append(out.loss * trg_len)
        total_pred_tokens += trg_len

    if (step + 1) % LOG_EVERY == 0:
        interim_ppl = math.exp(torch.stack(nlls).sum() / total_pred_tokens)
        toks_per_sec = total_pred_tokens / max(1e-6, (time.time() - t0))
        tqdm.write(f"[{step+1}/{total_steps}] interim PPL={interim_ppl:.3f} | tokens={total_pred_tokens} | {toks_per_sec:.1f} tok/s")

final_ppl = math.exp(torch.stack(nlls).sum() / total_pred_tokens)
dt = time.time() - t0
toks_per_sec = total_pred_tokens / max(1e-6, dt)
print(f"\nDataset: C4:{C4_CONFIG} [{C4_SPLIT}] (streaming={STREAMING})")
print(f"Context={MAX_LENGTH}, Stride={STRIDE}")
print(f"Tokens evaluated: {total_pred_tokens}, Windows: {len(nlls)}, Elapsed: {dt:.1f}s, Throughput: {toks_per_sec:.1f} tok/s")
print(f"Final Perplexity: {final_ppl:.3f}")
