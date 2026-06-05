# DEPLOYMENT.md

Operations guide: secrets, artifact storage, running inference, RunPod, and serving.

> **Scope:** this is a research/portfolio ~80M model. "Deployment" here means handling credentials, storing/sharing checkpoints & logs, and running inference — not a high-traffic production service. **Important:** the model uses custom hybrid attention, so off-the-shelf runtimes (llama.cpp, vLLM, Ollama, plain `transformers`) **cannot** load it. You run/serve it with *this repo's own code* (`generate.py`, `generate_instruct.py`, `chat.py`).

---

## 1. Secrets & credentials

**Rule: secrets never live in code, configs, or git.** Nothing in this repo hardcodes a key — they are read from the environment / the CLI credential store.

| Secret | Env var | Get it from |
|---|---|---|
| Hugging Face token | `HF_TOKEN` | https://huggingface.co/settings/tokens (write scope to upload) |
| Weights & Biases key | `WANDB_API_KEY` | https://wandb.ai/authorize |

**Set it (pick one):**
- **Local (Windows):** `[Environment]::SetEnvironmentVariable("HF_TOKEN","hf_xxx","User")` — then open a new terminal.
- **RunPod:** add `HF_TOKEN` (and `WANDB_API_KEY`) as **pod Secrets / environment variables** in the template. Injected on every launch; nothing on disk to leak.
- **CLI login (local convenience):** `huggingface-cli login` stores it at `~/.cache/huggingface/token`. Optional — the env var is enough.
- **`.env` file:** copy `.env.example` → `.env` and fill in. `.env` is gitignored; `.env.example` (no real values) is tracked.

**Precedence:** an explicit `token=` argument > `HF_TOKEN` env var > stored login file. Setting `HF_TOKEN` alone is sufficient.

**Do not:** print a token to stdout (it would land in `--logs` files, which you may upload), commit a real `.env`, or paste a key into a config/flag. If a key leaks, **revoke and regenerate it** immediately.

---

## 2. Storing checkpoints & logs

Checkpoints are large binaries (full ≈ 1 GB with optimizer state) and are gitignored. Store them as artifacts, not in git.

**Primary — Hugging Face Hub (private model repo).** `huggingface_hub` is already a dependency. Use the helper (reads `HF_TOKEN` from the env):

```bash
# inference-only, ~3x smaller (drops optimizer state):
python scripts/push_to_hf.py --repo_id <user>/sambhav-80m --checkpoint out/v3_sft_chat/best.pt --slim

# full checkpoint + its logs (resumable):
python scripts/push_to_hf.py --repo_id <user>/sambhav-80m --checkpoint out/v3_ctx16k/final.pt --logs_dir out/v3_ctx16k/logs
```

Pull one back:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("<user>/sambhav-80m", "checkpoints/best.slim.pt")
```

**During training — RunPod persistent volume.** Mount a Network Volume at `/workspace` so checkpoints survive pod stop/restart and spot preemption (the trainer auto-resumes). Push the keepers (`best.pt`, `final.pt`) to HF when a stage finishes; let the volume churn the rest. Keep only keepers on the Hub, not every milestone.

**Bulk / archival alternatives.** Cloudflare R2 (no egress fees) or Backblaze B2 (cheapest), accessed via `rclone copy out/ r2:bucket/sambhav/ -P`; Google Drive (15 GB free) also via `rclone`.

**Logs.** Small text — store the `out/<run>/logs/*.log` files alongside checkpoints (helper's `--logs_dir`), or wire up Weights & Biases / TensorBoard for live metrics (optional; needs a few lines added to `src/trainer.py`).

**Slim vs full:** full checkpoints carry AdamW optimizer state (~2× params). `--slim` keeps only `model` + `config` (~320 MB fp32) — loads fine in `generate.py`/`chat.py`, but can't resume training.

---

## 3. Running inference locally

```bash
# one-time: venv + torch (CUDA build) + deps
python -m venv .venv && .\.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# interactive multi-turn chat (the usable interface):
python chat.py --checkpoint out/v3_sft_chat/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json

# single instruction:
python generate_instruct.py --checkpoint out/v3_sft_reasoning/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --instruction "Explain why the sky is blue."

# raw completion (base model):
python generate.py --checkpoint out/v3_base_2k/best.pt --tokenizer tokenizer_fineweb_16k/tokenizer.json --prompt "Cloud computing is"
```

CPU works for inference (it's an 80M model); a GPU is faster but not required.

---

## 4. Running / training on RunPod

Full runbook: [`docs/runpod_longcontext_plan.md`](docs/runpod_longcontext_plan.md). In short:
1. Rent a pod with a persistent volume at `/workspace`; set `HF_TOKEN` (and `WANDB_API_KEY`) as pod Secrets.
2. `git clone` the repo, `pip install` torch + requirements.
3. Run the training phases (`nohup python train.py --config ... --resume auto --logs &`).
4. `python scripts/push_to_hf.py ...` the keepers before terminating the pod.

Use **community/spot** GPUs — the trainer saves on SIGTERM and auto-resumes, so preemption is safe and cheap.

---

## 5. Serving as an API (optional)

There is no built-in server. Because of the custom architecture you serve with this repo's model code. A minimal HTTP wrapper:

```python
# serve.py (sketch) — run: uvicorn serve:app --host 0.0.0.0 --port 8000
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer
from src.model import GPT, ModelConfig
from generate_instruct import format_prompt, extract_response

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load("out/v3_sft_chat/best.pt", map_location=device, weights_only=False)
cfg = dict(ck["config"]["model"]); cfg["vocab_size"] = ck["model"]["tok_embeddings.weight"].shape[0]
model = GPT(ModelConfig(**cfg)).to(device); model.load_state_dict(ck["model"]); model.eval()
tok = Tokenizer.from_file("tokenizer_fineweb_16k/tokenizer.json")

app = FastAPI()
class Req(BaseModel):
    instruction: str
    max_new_tokens: int = 160

@app.post("/generate")
def generate(r: Req):
    ids = tok.encode(format_prompt(r.instruction, ""), add_special_tokens=False).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=r.max_new_tokens, temperature=0.7, top_k=40, top_p=0.9,
                           repetition_penalty=1.2, no_repeat_ngram_size=3)
    return {"response": extract_response(tok.decode(y[0].tolist()))}
```

`pip install fastapi uvicorn` to use it. This is a starting point — there's no batching, streaming, or KV-cache, and quality is bounded by the 80M scale (see `docs/PROJECT_NOTES.md §5`).

---

## Quick checklist

- [ ] `HF_TOKEN` set via env / RunPod Secret (not in code or git)
- [ ] Persistent volume mounted on the pod
- [ ] Push `best.pt`/`final.pt` (+ `--slim` copy) to a **private** HF repo per stage
- [ ] Logs uploaded with the checkpoints (or W&B wired up)
- [ ] `.env` never committed (`.env.example` is the tracked template)
