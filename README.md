# opencode-localhost

Run local models in [opencode](https://opencode.ai). Your `.gguf` files show up
in the model picker as an ordinary provider, `llama-server` is started and
supervised for you, and a panel shows GPU, VRAM and CPU while you work.

```
 HARDWARE      memory  compute │ PROVIDER       │ MODEL
 GPU0     14.2/16.0G      87%  │ llama.cpp      │ Qwen3.6-35B-A3B
 GPU1     13.8/16.0G      82%  │ ● running      │ 245760 ctx · q8_0 KV
 CPU      18.1/31.0G      24%  │ :9337 web off  │ ngl 99 · 68.2 tok/s
```

No fork of opencode, no patches. It is a plugin.

## Requirements

- opencode
- a `llama-server` binary — [build it](https://github.com/ggml-org/llama.cpp) or
  install it however you like, then make sure it is on `$PATH`
- some `.gguf` files

## Install

opencode loads server-side and TUI plugins separately, so it goes in two files.

```jsonc
// ~/.config/opencode/opencode.jsonc
{ "plugin": ["opencode-localhost"] }
```

```jsonc
// ~/.config/opencode/tui.jsonc
{ "plugin": ["opencode-localhost"] }
```

Start opencode. The panel appears under the prompt and tells you what is still
missing.

## Setup

On first run it writes `~/.config/opencode/providers/llamacpp/server.ini`.
Exactly one thing has no sensible default:

```ini
[server]
bin =                              # empty = look on $PATH
models-dir = ~/models              # ← set this
host = 127.0.0.1
port = 9337
models-max = 1
api-key =
```

Set `models-dir` and restart. Your models are found, `models.ini` is generated,
`llama-server` starts, and the provider appears in the picker.

Nothing is guessed. If `llama-server` is not on `$PATH`, set `bin` — the panel
says so rather than failing silently.

## Per-model settings

`~/.config/opencode/providers/llamacpp/models.ini` is a llama.cpp preset file,
handed to `llama-server --models-preset` verbatim. A section is appended when a
new `.gguf` appears and **never modified afterwards** — it is yours.

```ini
[unsloth/Qwen3.6-35B-A3B-GGUF]
model        = /path/to/Qwen3.6-35B-A3B-Q4_K_M.gguf
ctx-size     = 245760
ubatch-size  = 2048
gpu-layers   = 99
flash-attn   = on
cache-type-k = q8_0
cache-type-v = q8_0
temp         = 0.6
top-p        = 0.95
```

Any `llama-server` flag works as a key, without the leading dashes.

Two different lifecycles:

| Change | Takes effect |
|---|---|
| launch flags — `ctx-size`, `gpu-layers`, `tensor-split`, `cache-type-*` | next model load |
| sampling — `temp`, `top-p`, `top-k`, `min-p` | next message, no reload |

Sampling is sent per request rather than baked into the server, so opencode's
own per-model defaults never override what you set here.

`ctx-size` is also what opencode compacts against — set it to what the model
really loads with, or long conversations will compact far too early.

## Adding another backend

`src/server/llamacpp/` implements `src/server/backend.ts`. vLLM or anything else
is a sibling folder implementing the same three methods — discover models, start
the server, say where it is. Its own config file, in its own native format,
under `~/.config/opencode/providers/<backend>/`.

## Security

`host` defaults to `127.0.0.1`, so the server is reachable only from this
machine. If you set `0.0.0.0` to use it from elsewhere on your network, set
`api-key` as well — `llama-server` then enforces bearer auth, and any
OpenAI-compatible client can connect with it.

## Limitations

- **No warm-on-select.** opencode's TUI does not expose the selected model to
  plugins, so a model starts loading on your first message rather than the
  moment you pick it.
- **Local only.** The panel reads `nvidia-smi` and the server directly, so it
  goes blank if the TUI runs on a different machine from opencode.
- **NVIDIA only** for GPU stats. Everything else works without a GPU; the
  hardware rows are simply omitted.

## License

MIT
