# opencode-localhost

[![npm](https://img.shields.io/npm/v/opencode-localhost)](https://www.npmjs.com/package/opencode-localhost)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

Run local models in [opencode](https://opencode.ai).

Your GGUF files appear in the model picker like any other provider.
`llama-server` is started, supervised and stopped for you. A panel keeps GPU,
VRAM and CPU on screen while you work.

```
 HARDWARE     memory  compute │ llama.cpp         │ lmstudio-community/
 GPU0     14.3/15.9G      87% │ ● 127.0.0.1:9337  │ Qwen3.6-35B-A3B-GGUF
 GPU1     13.7/15.9G      82% │ [stop]            │ 245760 ctx · q8_0 KV
 CPU      10.8/31.2G      31% │                   │ [change]
```

It is a plugin. It does not fork or patch opencode.

## Contents

- [What it does](#what-it-does)
- [Requirements](#requirements)
- [Installation](#installation)
- [First run](#first-run)
- [Configuration](#configuration)
- [Daily use](#daily-use)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Contributing](#contributing)

## What it does

**Discovers your models.** Scans a directory you choose for `.gguf` files,
including nested layouts like `<publisher>/<repo>/model.gguf`, and registers
each one — every quantisation as its own entry.

**Runs the server.** Starts `llama-server` on your first message and keeps it
running. Nothing is spawned at launch; `[start]` and `[stop]` are in the panel
when you want to decide yourself.

**Reports the truth about context.** opencode compacts a conversation against
the model's advertised window. The context you configure per model is what gets
advertised, so a model loaded at 245k is not compacted as though it were 32k.

**Shows what the hardware is doing.** Per-GPU memory and utilisation, system RAM
and CPU, the loaded model with its launch flags, and live progress while weights
stream into VRAM.

**Keeps settings in llama.cpp's own format.** Per-model options live in a
standard llama.cpp preset file, passed to `llama-server` unmodified. Any flag
llama.cpp accepts works, whether or not this plugin knows about it.

## Requirements

| | |
|---|---|
| **opencode** | tested against 1.18.5; older versions are untested |
| **llama-server** | build [llama.cpp](https://github.com/ggml-org/llama.cpp) yourself, or install it (`brew install llama.cpp`, a release archive, your package manager) |
| **models** | one or more `.gguf` files in a directory |
| **GPU stats** | optional. NVIDIA only; without it the hardware rows are simply omitted |

## Installation

Clone the repository and install its dependencies:

```sh
git clone https://github.com/dushyant30suthar/opencode-localhost
cd opencode-localhost
bun install
```

Then point opencode at the directory. It goes in **two files**, because opencode
loads server-side and terminal-side plugins through separate hosts:

```jsonc
// ~/.config/opencode/opencode.jsonc
{ "plugin": ["/absolute/path/to/opencode-localhost"] }
```

```jsonc
// ~/.config/opencode/tui.jsonc
{ "plugin": ["/absolute/path/to/opencode-localhost"] }
```

> **Why a path rather than the package name?**
>
> The package is published to npm, but installing it by name currently gives
> you the provider without the panel. opencode resolves an npm plugin against a
> wrapper `package.json` it generates in `~/.cache/opencode/packages/`, which
> carries no `exports` field, so the terminal-side entry point is never found
> and is skipped without an error. The server-side entry survives on a
> different fallback, which makes the result look half-broken rather than
> unconfigured.
>
> Referencing the directory avoids that resolution path entirely. When opencode
> resolves npm plugins against the real package, `{ "plugin":
> ["opencode-localhost"] }` will work with no change here.

## First run

Start opencode. The panel appears below the prompt and reports what is missing:

```
 HARDWARE     memory  compute │ llama.cpp         │ no model loaded
 GPU0      0.4/15.9G      12% │ ✕ not set up      │ [change]
 GPU1      0.4/15.9G       1% │ models-dir not set│
 CPU       7.1/31.2G      16% │                   │
```

Run `/localhost` and set two things:

```
 ✓ llama.cpp          /usr/local/bin/llama-server
 ✓ vLLM               ~/Projects/vllm/venv/bin/vllm
 ✗ MLX                Apple Silicon only
 ✗ OpenVINO           not installed — download the python_on build
 ✗ exllamav3          not installed — clone tabbyAPI, venv with the exl3 wheel
 ✗ ComfyUI            not installed — clone ComfyUI, uv venv, torch cu128+
 ✓ Models directory   ~/models
 ○ Server stopped   [start]   127.0.0.1:9337
```

- **llama.cpp** — every `llama-server` found on `$PATH`, plus *Enter a path…*
  for a build that is not on it
- **Models directory** — where your `.gguf` files live

Nothing is guessed. `$PATH` is searched for the binary and nothing else; the
models directory has no default, because only you know where it is.

Once both are set the models appear in the picker within a couple of seconds.
No restart.

## Configuration

Two files, both under `~/.config/opencode/providers/llamacpp/`.

### `server.ini` — how the server is run

Written on first start. Read by this plugin, which builds `llama-server`'s
command line from it.

```ini
[server]
bin        = /usr/local/bin/llama-server
models-dir = ~/models
remote     =
host       = 127.0.0.1
port       = 9337
models-max = 1
api-key    =
```

| Key | Meaning |
|---|---|
| `bin` | path to `llama-server`. Empty means search `$PATH` |
| `models-dir` | directory to scan for `.gguf` files. Required |
| `remote` | use **another machine's** server instead of running one here. See below |
| `host` | `127.0.0.1` keeps the server on this machine. `0.0.0.0` serves your LAN |
| `port` | listen port |
| `models-max` | how many models may occupy VRAM at once. `1` swaps instead of stacking |
| `api-key` | when set, `llama-server` enforces bearer auth |

#### Serving other machines

Set `host = 0.0.0.0` and the panel shows the address to point them at —
`fedora.local:9337 (192.168.1.14)` — rather than the loopback address, which is
the one address that tells you nothing once you are serving a network. The mDNS
name is listed first because the IP moves on DHCP renewal and the hostname does
not. `llama-server` has no auth unless you set `api-key`, so weigh that before
binding a network you do not control.

#### Using another machine's models

On the second machine, the whole configuration is one line:

```ini
[server]
remote = fedora.local:9337
```

No `bin`, no `models-dir`, no `models.ini`, no `.gguf` files. The model list
comes from that server, including each model's real `--ctx-size` — opencode
compacts against that number, so it is read rather than assumed.

Sampling is deliberately not sent from the client: the machine holding the
weights already applies its own `models.ini`, and a second machine pushing its
own `temp`/`top-k` would fight it.

Switching models needs nothing special — the server runs `models-max = 1` and
swaps when a request names a different model. `stop` unloads the model to free
VRAM rather than killing the process, since that server may have other clients.

### `models.ini` — how each model is loaded

Generated for you the first time your models are scanned. This is a standard
llama.cpp preset file, handed to `llama-server --models-preset` unmodified.

A section is appended when a new `.gguf` appears and **is never edited
afterwards**. It is yours.

```ini
[unsloth/Qwen3.6-35B-A3B-GGUF]
model        = /home/you/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf
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

The section name is the model's identity. Renaming a section is safe — it keeps
governing its file and its settings keep being used.

Two settings have different lifecycles:

| Change | Takes effect |
|---|---|
| launch flags — `ctx-size`, `gpu-layers`, `tensor-split`, `cache-type-*` | next time the model loads |
| sampling — `temp`, `top-p`, `top-k`, `min-p` | next message, no reload |

Sampling is attached to each request rather than baked into the server process,
so opencode's own per-model defaults never override what you set here, and
changing it does not cost a reload.

`ctx-size` is also what opencode compacts against. Set it to what the model
really loads with, or long conversations will compact far earlier than they
need to.

## Daily use

Open the model picker (`/model`, or `[change]` in the panel) and choose a model
under **Localhost-llama.cpp**.

Send a message. The first one is slow — that is the server starting and the
weights streaming into VRAM. The panel shows it:

```
 HARDWARE     memory  compute │ llama.cpp         │ ◐ unsloth/
 GPU0      8.1/15.9G      64% │ ● 127.0.0.1:9337  │ Laguna-S-2.1-GGUF
 GPU1      7.4/15.9G      61% │ [stop]            │ ████░░░░░░ 43%
 CPU      11.2/31.2G      38% │                   │ text_model
```

Subsequent messages are warm.

The same panel appears in the session sidebar, stacked rather than in columns.

**Controls**

| | |
|---|---|
| `[start]` / `[stop]` | run or stop the server |
| `[change]` | opencode's model picker |
| `/localhost` | paths and server state |
| `ctrl+p` → *Local models* | the same actions, from the command palette |

Both actions are registered commands (`localhost.server.toggle` and opencode's
`model.list`), so they can be bound to keys in your keybinds config.

## Troubleshooting

**The panel is missing entirely.** The terminal-side plugin is not loading.
Check `tui.jsonc` exists and points at the directory — it is a separate file
from `opencode.jsonc` and easy to miss. Confirm `bun install` has been run in
the clone; the panel needs its own dependencies.

**No provider in the model picker.** The engine is not configured. Run
`/localhost` — a backend with no binary or no models directory does not
register. If both are set and the picker is still empty, no `.gguf` files were
found under `models-dir`.

**A model fails to load and retries.** The server log has the reason:

```sh
tail -40 ~/.local/state/opencode/providers/llamacpp/server.log
```

The usual cause is a path in `models.ini` that `llama-server` cannot open.
Paths must be absolute — `llama-server` does not expand `~`.

**Conversations compact too early.** `ctx-size` for that model does not match
what it actually loads with. The panel shows the value in effect.

**Settings changed but nothing happened.** Launch flags apply on the next model
load, not immediately. Stop and start the server, or switch models and back.

## Limitations

**No warm-on-select.** opencode does not expose the selected model to plugins,
so a model begins loading on your first message rather than the moment you pick
it.

**Local only.** The panel reads `nvidia-smi` and the server directly, so it goes
blank if the terminal runs on a different machine from the model server.

**NVIDIA only for GPU statistics.** Everything else works without a GPU; the
hardware rows are omitted.

**Five backends, not all of them providers.** llama.cpp, OpenVINO, exllamav3
(via TabbyAPI), vLLM and ComfyUI are implemented; MLX is listed on the setup
screen with its install state but is not.

vLLM is shaped like the exllamav3 backend — one model per process, chosen at
startup — and takes one vLLM `--config` YAML per model under
`~/.config/opencode/providers/vllm/models/`. The picker keys on the filename, so
several files may serve one checkpoint and differ only in how; give each a
distinct `served-model-name` and the backend can adopt a running server instead
of paying for a reload.

ComfyUI is the odd one. It generates images and video, not chat completions, so
it deliberately contributes **nothing** to opencode's model picker — you will
never see a ComfyUI model there, and that is not a bug. It is here because it
competes for the same VRAM as the engines above: a video model and a 27B coding
model do not fit on the same two cards at once, and the panel is the only place
that can see all four and stop one before starting another. What you get is
`[start]`, `[stop]`, live per-GPU VRAM and sampling progress, plus the address to
open in a browser.

## Contributing

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) covers how the two halves fit
together, the constraints opencode's plugin API imposes, and what implementing
another backend involves.

```sh
bun install
bunx tsc --noEmit -p tsconfig.json
```

## License

MIT — see [LICENSE](./LICENSE).
