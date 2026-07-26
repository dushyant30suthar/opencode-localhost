# opencode-localhost — architecture

Local model providers for opencode, with live hardware telemetry.
Replaces the `opencode` fork entirely. No fork, no patches.

---

## 1. What this is

One npm package containing **two opencode plugins**:

| Plugin | Thread | Job |
|---|---|---|
| **server** | server worker | registers a provider whose models are discovered from disk; starts and supervises the model server |
| **tui** | TUI main | draws the hardware / provider / model panel |

opencode forbids one module exporting both, so the package exposes them as two
subpath exports and both config files reference the same package.

### What opencode actually sees

Only two things:

1. a **provider** named `llama.cpp` with models under it
2. **pixels** in two slots

Everything else — `llama-server`, the INI files, `nvidia-smi`, VRAM — is invisible to it.

---

## 2. Terminology

opencode's own words, taken from its source:

| Term | Meaning |
|---|---|
| **plugin** | a module opencode loads. Two kinds; **one module cannot be both** |
| **server plugin** | `export default { id, server }`. Declared in `opencode.json` → `plugin: []` |
| **TUI plugin** | `export default { id, tui }`. Declared in **`tui.json`** → `plugin: []` |
| **hook** | a callback a server plugin implements — `config`, `chat.params`, `tool`, `event` |
| **provider** | a named source of models: `{ id, name, source, options, models }` |
| **config provider** | a provider declared in config — `source: "config"`. Ours is one of these |
| **model** | `{ id, providerID, name, api{url,npm}, limit{context,output}, capabilities }` |
| **slot** | a named UI insertion point — `home_bottom`, `sidebar_content`, … |

Words opencode has no concept for, so we define them:

| Term | Meaning |
|---|---|
| **backend** | llama.cpp, vLLM, OpenVINO — the thing that runs inference |
| **model server** | the process itself (`llama-server`). Needs no name in the design; it's just a process we start |

"Engine" is not used anywhere.

---

## 3. Two repositories

The split follows one rule: **would this be wrong on someone else's machine?**
If yes, it isn't package code.

### `opencode-localhost` — the product

```
opencode-localhost/
├── package.json                name · version · exports { "./server", "./tui" }
├── README.md
│
├── src/
│   ├── server/                 ── PLUGIN 1 · server thread ──
│   │   ├── index.ts              export default { id, server }
│   │   ├── provider.ts           turns a backend into a config provider
│   │   ├── backend.ts            the interface every backend implements
│   │   ├── llamacpp/
│   │   │   ├── index.ts          detect · spawn · probe · report
│   │   │   ├── discover.ts       scan models-dir for GGUFs
│   │   │   ├── models-ini.ts     read/write models.ini + generic defaults
│   │   │   ├── server-ini.ts     read server.ini → CLI flags
│   │   │   └── status.ts         /models/sse, load progress
│   │   └── vllm/                 same interface, later
│   │
│   ├── tui/                    ── PLUGIN 2 · TUI thread ──
│   │   ├── index.ts              export default { id, tui }
│   │   ├── panel.tsx             the three sections
│   │   ├── home.tsx              home_bottom layout
│   │   ├── sidebar.tsx           sidebar_content layout
│   │   └── hardware/
│   │       ├── nvidia.ts         nvidia-smi
│   │       └── system.ts         RAM · CPU
│   │
│   └── shared/
│       ├── paths.ts              where config and state live
│       └── types.ts              display types both halves use
│
└── docs/
```

No submodules. No build scripts. No benchmark data. Nothing machine-specific.

### `opencode-llama.cpp` — the rig

The existing repo, minus the package.

```
opencode-llama.cpp/
├── llama.cpp/            submodule → ggml-org/llama.cpp   (pin: c588c4f47 / b10103)
├── scripts/              build-llama · download-models · tune-model
├── bench/                sweeps and results
├── config/               this machine's server.ini / models.ini, backed up
└── docs/                 how this rig is set up
```

Deleted: `scripts/build-opencode.sh`, the `opencode/` submodule, and the
fork-feature docs (`no-self-update.md`, `zero-config-provider.md`, `upgrading.md`).

---

## 4. Files on a user's machine

```
~/.config/opencode/
├── opencode.jsonc      plugin: ["opencode-localhost"]
├── tui.jsonc           plugin: ["opencode-localhost"]
└── providers/
    └── llamacpp/
        ├── server.ini      daemon settings
        └── models.ini      per-model settings

~/.local/state/opencode/providers/llamacpp/
├── server.pid
└── server.log
```

Nothing user-specific ships in the package. The package is read-only and
identical for everyone.

### `server.ini` — ours

We read it and build llama-server's command line. Changing it restarts the daemon.

```ini
[server]
bin        = llama-server          ; $PATH, or an absolute path
models-dir = ~/.lmstudio/models     ; probed on first run, then written here
host       = 0.0.0.0
port       = 9337
api-key    = <generated>
models-max = 1
web        = off
```

`models-dir` is **probed, then recorded**. The package checks a few known
layouts, writes whichever it found into this file, and never assumes again —
so the path is always visible and editable, never a hidden default.

### `models.ini` — llama.cpp's

Passed to `llama-server --models-preset` **verbatim**. We generate a section
when a new GGUF appears and never touch it again. Changing a section reloads
that model.

```ini
[lmstudio-community/Qwen3.6-35B-A3B-GGUF]
model        = /home/…/Qwen3.6-35B-A3B-Q4_K_M.gguf
mmproj       = /home/…/mmproj-Qwen3.6-35B-A3B-BF16.gguf
ctx-size     = 245760
ubatch-size  = 2048
gpu-layers   = 99
flash-attn   = on
cache-type-k = q8_0
cache-type-v = q8_0
temp         = 0.6
top-p        = 0.95
```

Sampling stays here. The plugin reads it and supplies it per request through
`chat.params`, so opencode's own sampling heuristics never override it.

vLLM gets `~/.config/opencode/providers/vllm/` with its own two files in its
own formats. Same split, different syntax.

---

## 5. How a model reaches the picker

```
server plugin  config() hook
      │
      ├─ read server.ini → find llama-server, models-dir
      ├─ scan models-dir → GGUFs
      ├─ read/extend models.ini → per-model launch settings
      ├─ spawn llama-server --models-preset models.ini
      └─ write provider into config.provider["llamacpp"]
                    │
                    ▼
opencode reads cfg.provider → provider appears with its models
```

The `config()` hook is the load-bearing mechanism. opencode loads plugins
*first*, specifically so their `config()` hook can modify config before
providers are read — the code comment says so, and the spike proved it.

---

## 6. The panel

Three sections in priority order: **hardware → provider → model**.
Numbers only, no bar glyphs, right-aligned so digits stack.

### Home — `home_bottom`, 75 columns

```
 HARDWARE      memory  compute │ PROVIDER       │ MODEL
 GPU0     14.2/16.0G      87%  │ llama.cpp      │ Qwen3.6-35B-A3B
 GPU1     13.8/16.0G      82%  │ ● running      │ 245760 ctx · q8_0 KV
 CPU      18.1/31.0G      24%  │ :9337 web off  │ ngl 99 · 68.2 tok/s
```

Hardware is a device × metric grid — three devices, two metrics each. The CPU
row reuses the same two columns (its memory is system RAM, its compute is CPU
utilization), so the headers hold for all three rows.

### Session — `sidebar_content`, 42 columns

```
 HARDWARE      memory  compute
 GPU0     14.2/16.0G      87%
 GPU1     13.8/16.0G      82%
 CPU      18.1/31.0G      24%

 PROVIDER
 llama.cpp
 ● running · :9337
 web off

 MODEL
 Qwen3.6-35B-A3B
 245760 ctx · q8_0 KV
 ngl 99 · split 0.67/0.33
 68.2 tok/s
```

The hardware grid is 30 columns wide in both layouts — **literally the same
component**, not two layouts to keep in sync.

### States

```
loading    │ ◐ loading   │ Qwen3.6-35B-A3B
           │ :9337       │ 68%  weights
stopped    │ ○ stopped   │ no model loaded
           │ [start]     │
failed     │ ✕ failed    │ not enough VRAM
```

Hardware keeps updating in every state.

### Color

| Element | Token |
|---|---|
| memory value | `text` → `warning` ≥90% → `error` ≥97% |
| compute value | `text` always — a pinned GPU is the goal, never an alarm |
| labels, units, `·`, `/` | `textMuted` |
| status glyph + word | `success` / `warning` / `error` / `textMuted` |

**State is never colour alone.** The glyph shape carries it (`●` `◐` `○` `✕`)
*and* a word follows — works on a monochrome terminal and for a colourblind
reader.

### Liveness

- **model status** — llama-server's `/models/sse`. Push, not poll
- **hardware** — `nvidia-smi` every 2s while visible, stopped when not

### Narrow terminals

```
< 64 cols   drop the RAM row, abbreviate
< 48 cols   one line:  ● llama.cpp · Qwen3.6-35B · 87%/82%
```

---

## 7. Verified by execution

A throwaway plugin was run against a real opencode.

| # | Assumption | Result |
|---|---|---|
| 1 | `config()` can inject a provider absent from models.dev | ✅ proven |
| 2 | external plugin registers into `home_bottom` / `sidebar_content` | ✅ proven |
| 3 | `chat.params` overrides sampling | ✅ proven **on the wire** |
| 4 | plugin can observe model selection | ❌ not available |

Evidence for #3 — captured request body from a fake OpenAI endpoint:

```
model      : spike-alpha
temperature: 0.123        ← the plugin's value, not opencode's
```

### Things learned that weren't in the plan

- **A plugin exports `server()` or `tui()` — never both.** Enforced at load.
- **TUI plugins are not auto-discovered.** `.opencode/plugin/*.ts` is
  server-only. TUI plugins must be declared in **`tui.json` / `tui.jsonc`**,
  a separate file from `opencode.json`. Undocumented.
- **Declaring a TUI plugin in `opencode.json`** makes the server host log a
  load error on every startup. Keep them in their own files.
- **A provider ID does not need a schema enum entry.** `ProviderV2.ID.make()`
  accepts any string.

---

## 8. Known limitations

**No warm-on-select.** `TuiState` exposes providers, sessions and parts but not
the currently selected model, and `TuiEventBus` carries only server events —
model selection is TUI-local state. So a model can't start loading the moment
you pick it; it loads on the first message instead.

Replacement: the panel lists models and clicking one loads it. Explicit, and it
works mid-conversation with a cloud model, which the fork's version could not.

**Remote TUI.** The TUI half reads `nvidia-smi` and port 9337 directly, so if
the TUI ever runs on a different machine from the server, the panel goes blank.
Fixing it means routing telemetry through the server half, which is not
possible today — a server plugin cannot register HTTP endpoints.

---

## 9. What the fork did, and where it goes

| Fork change | Replacement |
|---|---|
| `provider/llamastack.ts` custom loader | `config()` hook injects a config provider |
| `provider.ts` detect hook | same |
| `schema/provider.ts` enum entry | not needed |
| `transform.ts` sampling suppression ×3 | `chat.params` supplies sampling |
| `registry.ts` websearch | `OPENCODE_ENABLE_EXA=1` |
| `builtins.ts` sidebar registration | external TUI plugin |
| `app.tsx` `/config` command | deleted — edit the file |
| `prompt/index.tsx` load status | folded into the panel |
| `installation/*` self-update guard | **not needed** — stock binary, nothing to protect |
| `upgrade-stack.ts` | script in the rig repo, llama.cpp only |

The self-update guard existed because a fork binary would be overwritten by
stock opencode. With no fork, you *are* stock opencode.
