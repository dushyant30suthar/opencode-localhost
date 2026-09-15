# Qwen3.8-27B — Setup Review + 8-bit (Q8_0) Context Runbook

> Read this first: **do NOT stop/reload the server on port 9337 while this opencode
> session is live.** The model answering you right now *is* that loaded model. Run the
> "switch" steps only after you're done with this conversation (or from a second terminal
> on a machine/session that isn't routed through that server).

---

## 0. Your hardware (measured)

| Item | Value |
|---|---|
| GPUs | 2× NVIDIA RTX 5060 Ti |
| VRAM each | 16311 MiB (≈15.93 GiB) |
| **VRAM total** | **32622 MiB ≈ 31.86 GiB** |
| Split | `--split-mode tensor`, `--n-gpu-layers 99` (full offload, even split) |

---

## 1. Current setup (port 9337) — verdict

| Aspect | Current value | Verdict |
|---|---|---|
| Quant | Q4_K_M (18.97 GB) | Fine for 2×16GB; quality is the 4-bit trade-off |
| Offload | ngl 99, tensor split, flash-attn on | **Correct** — nothing to change |
| KV cache | q8_0 K/V | **Correct** — halves cache vs f16 |
| Sampling | temp 1.0, top_p 0.95, top_k 20, min_p 0, presence 0, repeat 1.0 | **One off** — you're in thinking+coding; official temp is **0.6** (all other 5 values match exactly) |
| Context | 196608 (192K) | Fine on Q4_K_M (fits with headroom) |
| **MTP decode** | `--spec-type draft-mtp --spec-draft-n-max 4` | **Mis-tuned** → see below |

### The two real mis-tunes (confirmed: thinking mode + coding)

**1. `temperature 1.0` → should be `0.6`.**
Qwen's official *"thinking mode, precise coding"* recipe is temp **0.6** with
top_p 0.95 / top_k 20 / min_p 0 / presence 0 / repeat 1.0 — the other five you **already
match exactly**. You're at 1.0, which is the *general* thinking value, not the coding one.
Lower temp also raises MTP acceptance (flatter distribution at 1.0 → draft head misses the
sampler's pick more often).

**2. `spec-draft-n-max 4` → should be `3` (or `2`).**
Benchmarks on the dense 27B MTP head (single-head) show the peak is at
**n=2 or n=3**; by n=4 acceptance has already dropped and the bigger verify batch no longer
pays for itself:

| n-max | relative vs no-MTP |
|---|---|
| 1 | +47% |
| 2 | +61%  ← peak |
| 3 | +61%  ← peak (upstream default) |
| 4 | +57%  ← you are here |
| 6 | +23% |
| 8 | −20% |

**Fix: change `--spec-draft-n-max 4` → `3`** (or `2`). Expected acceptance in the server log
should sit around **60–75%**.

### Two things that lower acceptance (informational, not errors)
- **temp 1.0** is the official Qwen recipe, but higher temperature = flatter distribution =
  lower MTP acceptance. If you mostly do **coding/agent** work, temp **0.6** (Qwen's
  "precise coding" recipe) raises acceptance and is the recommended setting for that use.
- Qwen3.8 is **hybrid attention** (48 linear + 16 full-attention layers). Watch the server
  log for `draft acceptance rate = ...` and for
  `W ... forcing full prompt re-processing due to lack of cache data`. If you see repeated
  full re-processing, acceptance will collapse toward ~35% — that's a known hybrid/SWA+MTP
  interaction, not your config's fault.

---

## 2. Switching to the official 8-bit (Q8_0) — how much context?

The official ggml-org pack for this model ships:

| File | Size |
|---|---|
| Qwen3.8-27B-**BF16**.gguf | 53.8 GB (won't fit 31.86 GiB — skip) |
| Qwen3.8-27B-**Q8_0**.gguf | **28.6 GB** ← the "8-bit" |
| mtp-Qwen3.8-27B-Q4_0.gguf | 1.68 GB (draft head, use this) |
| mtp-Qwen3.8-27B-Q8_0.gguf | 3.16 GB (draft head, +1.5 GB, better draft) |

### KV-cache cost per token (why this model is cheap on context)
Only the **16 full-attention** layers store a growing KV cache (the 48 linear layers use a
fixed-size state). Per token, at **q8_0** K/V:

```
2 (K+V) × 4 KV heads × 256 dim × 1 byte × 16 layers = 32 KB / token
```

(At f16 it would be 64 KB/token — another reason to keep `--cache-type-k/v q8_0`.)

### Memory budget for Q8_0 on your 31.86 GiB

| Component | Size |
|---|---|
| Q8_0 weights | 28.6 GB ≈ 26.6 GiB |
| mtp Q4_0 draft head | 1.68 GB ≈ 1.6 GiB |
| **weights subtotal** | **≈ 28.2 GiB** |
| CUDA context + compute buffers + draft KV (2 GPUs) | ≈ 2–4 GiB |
| **left for KV cache** | **≈ 1.5–3 GiB** |

### Resulting context window

| KV budget | context |
|---|---|
| 1.5 GiB | ~48K |
| 2.0 GiB | ~64K |
| 3.0 GiB | ~96K (optimistic) |

**Bottom line:** Q8_0 realistically gives you **~32K–64K** of context on 2×16GB (vs the
**192K** you have now on Q4_K_M). The exact number depends on runtime overhead, so **start
at 32K, check VRAM, then raise it** (section 3). You trade context for a near-lossless
quant (Q8_0 has far less error than Q4_K_M) and a bigger *relative* MTP speedup (heavier
per-pass cost → MTP helps more; expect ~2×+ decode speedup with n=3).

> Note: 32K–64K is well inside the model's native 256K window — no YaRN/RoPE scaling needed.

---

## 3. Runbook — switch to Q8_0 (do this AFTER closing this session)

### Step 1 — download the 8-bit weights + draft head
```bash
hf download ggml-org/Qwen3.8-27B-GGUF \
  --local-dir /home/dushyant30suthar/.lmstudio/models/ggml-org/Qwen3.8-27B-GGUF \
  --include "Qwen3.8-27B-Q8_0.gguf" \
  --include "mtp-Qwen3.8-27B-Q4_0.gguf"
```

### Step 2 — stop the current server (this unloads the model)
- In LM Studio: stop the server on port 9337, **or**
- `kill` the `llama-server` process. Confirm with `nvidia-smi` that VRAM is freed.

### Step 3 — launch Q8_0 (reference command; mirror these flags in your LM Studio preset)
```bash
/home/dushyant30suthar/Projects/llama.cpp/build/bin/llama-server \
  --host 127.0.0.1 \
  --port 9337 \
  --jinja \
  --model /home/dushyant30suthar/.lmstudio/models/ggml-org/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf \
  --model-draft /home/dushyant30suthar/.lmstudio/models/ggml-org/Qwen3.8-27B-GGUF/mtp-Qwen3.8-27B-Q4_0.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --alias ggml-org/Qwen3.8-27B-Q8_0 \
  --temperature 1.0 \
  --top-k 20 \
  --top-p 0.95 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --ctx-size 32768 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  --n-gpu-layers 99 \
  --parallel 1 \
  --split-mode tensor
```

### Step 4 — verify + tune context up
```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```
- If it loads with room to spare (e.g. >1.5 GiB free total), bump `--ctx-size` to
  **49152** then **65536**, restarting each time. Stop at the value that still leaves
  headroom (don't OOM mid-conversation).
- Watch the server log for `draft acceptance rate = X` (want ~0.6–0.75) and the
  `forcing full prompt re-processing` warning (hybrid-attention; see section 1).

### Step 5 (optional) — better draft head
If you have VRAM to spare after tuning context, swap the draft to the Q8_0 head
(`mtp-...-Q8_0.gguf`, +1.5 GB) for a slightly higher acceptance rate.

---

## 4. Minimal-change alternative (keep Q4_K_M, just fix the two flags)
If you want to stay on the current 192K Q4_K_M and only fix the mis-tunes, make exactly
two changes: **`--spec-draft-n-max 4` → `3`** and **`--temperature 0.6` → `1.0`**.
Same context, same quant — higher acceptance and a fully official thinking+coding setup.

---

### TL;DR
- Config is **correct** except two flags for thinking+coding: **temp 0.6 → 1.0** and
  **n-max 4 → 3** (all other sampling values already match the official recipe).
- Q8_0 (28.6 GB) on your 31.86 GiB → **~32K–64K context** (start 32K, verify, raise).
  You lose context vs 192K but gain near-lossless quality + a bigger relative MTP speedup.
