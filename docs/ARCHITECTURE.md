# Architecture

How opencode-localhost is put together, why it is split the way it is, and what
adding another backend involves.

## Contents

- [The shape of it](#the-shape-of-it)
- [Vocabulary](#vocabulary)
- [Source layout](#source-layout)
- [The server half](#the-server-half)
- [The terminal half](#the-terminal-half)
- [Files on disk](#files-on-disk)
- [Adding a backend](#adding-a-backend)
- [Constraints worth knowing](#constraints-worth-knowing)

## The shape of it

One package containing **two plugins**:

| Plugin | Runs in | Responsibility |
|---|---|---|
| `./server` | opencode's server worker | discover models, run the backend, register the provider |
| `./tui` | opencode's terminal thread | draw the panel, offer the controls |

They are separate because opencode enforces it — a module may export `server()`
or `tui()`, never both. The package exposes them as subpath exports, and both
of opencode's config files reference the same package.

The two halves do not talk to each other. A server plugin cannot expose
endpoints, so the terminal half reads the backend directly rather than asking
its own server side. See [Constraints](#constraints-worth-knowing).

**What opencode sees is only ever two things:** a provider with models under it,
and content in two UI slots. Everything else — the model server process, the
configuration files, `nvidia-smi`, VRAM — is invisible to it.

## Vocabulary

opencode's own terms, used here with the same meanings:

| Term | Meaning |
|---|---|
| **plugin** | a module opencode loads; server-side or terminal-side |
| **hook** | a callback a server plugin implements — `config`, `chat.params`, … |
| **provider** | a named source of models, as it appears in the model picker |
| **config provider** | a provider contributed through configuration rather than built in. This plugin's providers are these |
| **slot** | a named UI insertion point — `home_bottom`, `sidebar_content` |

Terms this project defines:

| Term | Meaning |
|---|---|
| **backend** | an inference engine: llama.cpp, vLLM, MLX, OpenVINO |
| **model server** | the process a backend manages, e.g. `llama-server` |

Each backend becomes its own opencode provider, so a machine with several
engines shows several providers.

## Source layout

```
src/
├── server/                  ── plugin 1: opencode's server worker
│   ├── index.ts               config() and chat.params hooks
│   ├── backend.ts             the interface every backend implements
│   └── llamacpp/
│       ├── index.ts           detect · start · stop · report
│       ├── discover.ts        find .gguf files under a directory
│       ├── models-ini.ts      per-model settings, in llama.cpp's preset format
│       └── server-ini.ts      how the server is run; builds its argv
│
├── tui/                     ── plugin 2: opencode's terminal thread
│   ├── index.tsx              slot registration, commands, layout
│   ├── panel.tsx              the three sections, and the shared data source
│   ├── setup.tsx              the /localhost screen
│   ├── backends.ts            shared backend instances
│   └── hardware/
│       ├── nvidia.ts          GPU memory and utilisation
│       └── system.ts          RAM and CPU
│
└── shared/
    ├── backends.ts            the engines this plugin knows about
    ├── ini.ts                 an INI reader that preserves comments
    ├── paths.ts               where configuration and state live
    └── types.ts               shapes both halves use
```

## The server half

### Registering a provider

opencode loads plugins before reading its provider configuration, so a plugin's
`config()` hook can contribute to it. That is the whole integration:

```
config() hook
   │
   ├─ read server.ini      where is the binary, where are the models
   ├─ scan models-dir      find .gguf files
   ├─ sync models.ini      append a section for anything new
   └─ write provider into the config opencode is about to read
```

From there it is an ordinary provider and opencode treats it like any other.

A backend contributes only when it is **configured** — a binary and a models
directory. Whether a server happens to be running is deliberately not part of
that test: the models exist on disk either way, and a picker that stays empty
until you find and press a start button is indistinguishable from a plugin that
does not work.

### Starting the model server

Nothing is spawned when opencode launches. The `chat.params` hook runs before
every request, and starts the backend there if it is not already up:

```
first message → chat.params → backend not running → start it → request proceeds
```

opencode awaits that hook, so the request cannot fire before the server is
listening. Sending a message is treated as sufficient intent; the panel's
`[start]` and `[stop]` are there for deciding explicitly.

### Sampling

`chat.params` also applies per-model sampling read from `models.ini`.

Sampling is deliberately *not* baked into the server's launch flags. As request
parameters it takes effect on the next message rather than the next model load,
it can differ per agent, and it overrides opencode's own per-model heuristics,
which know nothing about local models.

## The terminal half

### The panel

Three sections, defined once and used by both layouts. The home strip arranges
them in columns; the session sidebar stacks them. Only a `stacked` prop and the
wrapper differ.

| Section | Scales how |
|---|---|
| **HARDWARE** | fixed. GPUs are a shared pool however many engines exist |
| **PROVIDER** | one entry per configured engine, each with its own control |
| **MODEL** | one entry per loaded model, naming its engine only when more than one is loaded |

The strip keeps the prompt's width and grows downward rather than sideways, so
it stays aligned with everything else on the home screen.

### Where the data comes from

A single shared source, not a per-component hook — the home strip and the
session sidebar are separate components, and a hook would mean two polling
loops, two event subscriptions and two `nvidia-smi` calls whenever both were
alive. The first mount starts the work; the last one stops it.

Two channels, because one is not enough:

| Channel | Carries | Why |
|---|---|---|
| **poll** (2s) | hardware, server state, loaded model | there is no push interface for `nvidia-smi` |
| **stream** (SSE) | load progress | the REST listing flips *unloaded* to *loaded* at the end and says nothing in between |

Polling tightens to 500ms while a load is in flight and relaxes afterwards, so
an idle panel is close to free.

### Registration refresh

opencode reads its configuration once per instance and offers no endpoint to
re-read it. A provider that was not ready at startup would therefore not appear
until the next launch.

Its terminal binds `SIGUSR2` to a configuration invalidation, and a terminal
plugin runs inside that process — so the panel signals itself once a backend
becomes available, and the provider registers without a restart. This is only
done from the home screen: the reload disposes live instances, which is
acceptable before you start work and disruptive in the middle of a session.

## Files on disk

```
~/.config/opencode/providers/<backend>/     configuration — yours to edit
~/.local/state/opencode/providers/<backend>/    pidfile, log — ours
```

Configuration follows XDG so it sits beside opencode's own, which is where
people look for it. State holds files a program owns and a human never edits.

Nothing machine-specific ships in the package. Paths are recorded in
configuration at runtime, never compiled in — a default that is right on one
machine is wrong on the next, and a wrong path silently costs performance
rather than failing loudly.

## Adding a backend

A backend implements one interface:

```ts
interface Backend {
  readonly id: string            // also its configuration directory name
  readonly name: string          // short, for the panel
  readonly providerName: string  // how it appears in the model picker

  status(): Promise<ProviderStatus>
  models(): Promise<DiscoveredModel[]>
  start(): Promise<ProviderStatus>
  stop(): Promise<boolean>
  loaded(): Promise<LoadedModel | undefined>
  watch(onEvent: (event: LoadEvent) => void): () => void

  baseURL(): string
  apiKey(): string | undefined
}
```

Everything engine-specific stays behind it: configuration format, discovery
layout, launch flags, how progress is reported. Nothing outside the folder
needs to know that llama.cpp uses INI presets while another engine uses YAML.

To add one:

1. create `src/server/<backend>/` implementing the interface
2. give it a configuration file in **that engine's own native format**, under
   `~/.config/opencode/providers/<backend>/`
3. add it to the list in `src/shared/backends.ts` with its binary name and
   install command
4. add the instance to `src/tui/backends.ts`

The panel, the provider registration, the setup screen and the commands all
adapt on their own. Each backend becomes its own provider, so several can run
side by side.

Backends are listed in the setup screen whether or not they are installed,
with what installing them would take. They are never installed for you:
llama.cpp ships a native binary while the others are Python packages, so there
is no single thing to fetch, and choosing the wrong llama.cpp build — CUDA
version, GPU architecture — quietly costs a large multiple of performance.

## Constraints worth knowing

These shaped the design and are not obvious from the code.

**A module exports `server()` or `tui()`, never both.** Enforced at load. Hence
two entry points from one package.

**Terminal plugins are declared in a different file.** Server plugins can be
auto-discovered from `.opencode/plugin/`; terminal plugins must be listed in
`tui.json` / `tui.jsonc`. Listing a terminal plugin in `opencode.json` makes the
server host log a load error on every start.

**Terminal plugins cannot be installed by package name.** opencode resolves an
npm plugin against a wrapper `package.json` it generates, which has no `exports`
field, so the terminal entry point is never found and is skipped silently. The
server entry survives on a different fallback. Referencing the package by
directory avoids this.

**Plugins cannot see the selected model.** The terminal's state exposes
providers, sessions and messages, but not the current model — that is local UI
state. This is why there is no warm-on-select, and why `[change]` dispatches
opencode's own picker rather than offering its own: a custom list could display
models but never switch to one.

**Plugins cannot take keyboard focus.** Tab is bound to cycling agents and slot
content is not in a focus ring plugins can join, so panel controls are reachable
by mouse or by binding the registered commands.

**The two halves cannot call each other.** A server plugin cannot register HTTP
endpoints, so the terminal half reads the backend directly. This is the reason
the panel goes blank when the terminal runs on a different machine from the
model server.
