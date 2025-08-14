import math
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from itertools import chain
from tqdm import tqdm  # progress bar

# ======== CONFIG ========
MODEL_OBJ_PATH  = "/workspace/SqueezeLLM/Packed_obj.pt"        # quantized model object
BASE_MODEL_PATH = "/workspace/SqueezeLLM/models/Llama-2-7b-hf" # tokenizer path
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH      = 1024   # context length
STRIDE          = 512    # overlap for sliding window
# ========================

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Load quantized model (PyTorch 2.6 safe load) ---
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

# --- Load cached WikiText-2 dataset ---
print("Loading WikiText-2 from cache...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# --- Tokenize in batches (avoid large memory spike) ---
print("Tokenizing dataset...")
def encode_batch(batch):
    return tokenizer(batch["text"], add_special_tokens=False)

tokenized_dataset = dataset.map(encode_batch, batched=True, remove_columns=["text"])
all_tokens = list(chain.from_iterable(tokenized_dataset["input_ids"]))
input_ids = torch.tensor(all_tokens, dtype=torch.long)

# --- Perplexity computation ---
nlls = []
total_pred_tokens = 0

print("Computing perplexity...")
total_steps = (input_ids.size(0) - 1) // STRIDE

for step, i in enumerate(tqdm(range(0, input_ids.size(0) - 1, STRIDE), total=total_steps, desc="Processing chunks")):
    begin_loc = max(i + STRIDE - MAX_LENGTH, 0)
    end_loc   = min(i + STRIDE, input_ids.size(0) - 1)
    trg_len   = end_loc - i
    if trg_len <= 0:
        break

    input_ids_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(DEVICE)
    target_ids = input_ids_chunk.clone()
    target_ids[:, :-trg_len] = -100  # mask context tokens

    with torch.no_grad():
        outputs = model(input_ids_chunk, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)
    total_pred_tokens += trg_len

    # Optional: print intermediate PPL every 50 steps
    if (step + 1) % 50 == 0:
        interim_ppl = math.exp(torch.stack(nlls).sum() / total_pred_tokens)
        tqdm.write(f"Step {step+1}/{total_steps} - Interim PPL: {interim_ppl:.3f}")

ppl = math.exp(torch.stack(nlls).sum() / total_pred_tokens)
print(f"Final Perplexity: {ppl:.3f}")
