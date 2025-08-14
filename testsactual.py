import os, math, time, torch
from transformers import AutoTokenizer
from datasets import load_dataset
from itertools import chain
from tqdm import tqdm

# ======== CONFIG ========
MODEL_OBJ_PATH  = "/workspace/SqueezeLLM/Packed_obj.pt"        # your quantized model object (torch.save(model, ...))
BASE_MODEL_PATH = "/workspace/SqueezeLLM/models/Llama-2-7b-hf" # tokenizer path
DATASET_CONFIG  = "wikitext-2-raw-v1"   # << SqueezeLLM paper uses this
# DATASET_CONFIG = "wikitext-2-raw-v1"    # (optional) quick smoke test
SPLIT           = "test"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH      = 2048     # context window for evaluation (common in papers)
STRIDE          = 2048      # overlap; predicts only the last STRIDE tokens each window
LOG_EVERY       = 50       # print interim PPL every N steps
# ========================

print("HF_DATASETS_CACHE:", os.environ.get("HF_DATASETS_CACHE"))

# --- tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- load quantized model object (PyTorch 2.6 safe load) ---
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

# --- dataset (cached locally already) ---
print(f"Loading WikiText: {DATASET_CONFIG} [{SPLIT}] ...")
dataset = load_dataset("wikitext", DATASET_CONFIG, split=SPLIT)

# --- tokenize in batches to avoid giant strings ---
print("Tokenizing dataset...")
def encode_batch(batch):
    return tokenizer(batch["text"], add_special_tokens=False)

tok_ds = dataset.map(encode_batch, batched=True, remove_columns=["text"])
all_tokens = list(chain.from_iterable(tok_ds["input_ids"]))
input_ids = torch.tensor(all_tokens, dtype=torch.long)

# --- perplexity with sliding windows ---
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
    targets[:, :-trg_len] = -100

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
print(f"\nDataset: {DATASET_CONFIG} [{SPLIT}]")
print(f"Context={MAX_LENGTH}, Stride={STRIDE}")
print(f"Tokens evaluated: {total_pred_tokens}, Windows: {len(nlls)}, Elapsed: {dt:.1f}s, Throughput: {toks_per_sec:.1f} tok/s")
print(f"Final Perplexity: {final_ppl:.3f}")
