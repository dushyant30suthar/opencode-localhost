import fs from "fs/promises"
import path from "path"
import * as Ini from "../../shared/ini.ts"
import { configDir, expandHome, collapseHome } from "../../shared/paths.ts"

/**
 * server.ini — ours, not OVMS's. We read it and build the ovms-serve command
 * line from it.
 *
 * This file is flatter than llama.cpp's because OVMS is: one server process
 * serves exactly one model, loaded onto the GPU at startup. There is no preset
 * file and no swapping, so the model choice and its sampling live here rather
 * than in a per-model file.
 *
 * Two values have no safe default and stay empty until set:
 *   bin        — filled from $PATH if ovms-serve is there, else empty
 *   models-dir — only the user knows where their IR models are
 */

export const BACKEND = "openvino"
export const FILE = path.join(configDir(BACKEND), "server.ini")

/**
 * 65536 is well below the 262144 these checkpoints advertise, and deliberately
 * conservative: what OVMS can actually serve depends on two values this file
 * does not control — cache_size and cache_interval_multiplier, both in the
 * servable's graph.pbtxt.
 *
 * Which of the two dominates depends on the multiplier, so neither can be
 * called "the real ceiling" on its own. At the default of 8 the fp32 recurrent
 * checkpoints dwarf the KV cache; raise it far enough and the checkpoint term
 * shrinks toward nothing and KV alone sets the floor. Measured on an Arc 140T
 * at cache_size 4 and multiplier 128: the 19.7GB MoE served 90,615 tokens,
 * while the dense 27B — 2.4x the checkpoint, 3.2x the KV — answered 44,215 and
 * returned empty at 90,617.
 *
 * Over-context does not error: OVMS returns HTTP 200 with an empty completion.
 * So advertising a window the server cannot serve makes opencode compact
 * against it, and the failure looks like the model refusing to answer. Hence a
 * default that is safe on both, to be raised only once you have actually served
 * that length.
 */
const DEFAULT_CONTEXT = 65_536

const TEMPLATE = `# OpenVINO Model Server settings for opencode-localhost.
# Changing anything here restarts the server.
#
#   bin         path to ovms-serve. empty = look on $PATH
#               NOTE the wrapper, not bare 'ovms': ovms links libpython and
#               exits before main() unless LD_LIBRARY_PATH is already set
#   models-dir  where your OpenVINO IR model directories live. REQUIRED
#   model       which one to serve. empty = the only one, if there is only one
#               OVMS holds one model per process, so this is a real choice:
#               changing it and restarting is how you switch models
#   remote      point at ANOTHER machine's OVMS instead of running one here,
#               e.g. fedora.local:8100. When set, bin/models-dir/model/host/port
#               are ignored and nothing is started locally. Note stopping is not
#               possible on a remote: OVMS holds one model for the life of the
#               process and has no unload endpoint, so the only way to free that
#               GPU is on the machine that owns it
#   host        127.0.0.1 keeps it on this machine; 0.0.0.0 serves the LAN so
#               another machine can point at it. This is passed to OVMS as
#               --rest_bind_address, whose own default is 0.0.0.0 — so leaving
#               this at 127.0.0.1 is what actually makes it loopback-only
#   port        OVMS REST port
#   context     window to advertise to opencode. keep it under what the KV cache
#               can actually hold — going over returns an empty reply, not an
#               error, so too high looks like the model refusing to answer
#
# CACHE TUNING IS NOT HERE. ovms-serve takes no flags for it — OVMS reads
# cache_size and cache_interval_multiplier from the servable's graph.pbtxt, and
# this plugin passes only a model name and lets the wrapper resolve that file.
# Edit it there. Both settings used to appear below and did nothing, which was
# worse than absent: the panel displayed a cache size the server was not using.
#
# What to put in graph.pbtxt, measured on an Arc 140T with a 19.7GB MoE:
#
#   cache_interval_multiplier  REQUIRED for long context on hybrid models
#               (Qwen3.6 etc). OVMS checkpoints the whole fp32 recurrent state
#               every kv_block_size * this. The default of 8 makes those
#               snapshots dwarf the KV cache: 267 KiB/token, so ~15k context.
#               THE RIGHT VALUE IS PER MODEL. It scales with the checkpoint,
#               and both terms come out of the model's own config.json:
#                 checkpoint  = linear_layers * linear_num_value_heads
#                             * linear_key_head_dim * linear_value_head_dim * 4
#                 KV KiB/tok  = full_attn_layers * num_key_value_heads
#                             * head_dim * 2          (1 byte each at u8)
#                 KiB/token   = checkpoint / (kv_block_size * this) + KV
#               Qwen3.6-35B-A3B: 63.8 MiB checkpoint, 10.0 KiB/tok of KV. At
#               128 that is 33 KiB/token and 90,615 answered in seconds.
#               Qwen3.6-27B: 151.5 MiB and 32.0 KiB/tok — 2.4x and 3.2x more.
#               The same 128 gives 107.8 KiB/token and caps near 38k: measured,
#               44,215 answered and 90,617 came back empty. It needs 1024+.
#               Copying another model's number is the easy way to lose most of
#               your context and think the model is refusing to answer.
#               Raising this has a floor: the checkpoint term shrinks toward
#               nothing and KV alone caps you. At cache_size 4 that is ~128k
#               for the 27B and ~409k for the 35B, whatever you set here.
#               openvino.genai PR #4050 fixes the underlying bug in 2026.3.
#   cache_size  KV cache in GB. Caps context AND footprint: model + cache must
#               leave room for the OS. On a 30GB box, 6 left 3GB free and
#               throughput fell from 28 to 5 tok/s; 4 leaves ~9GB free and runs
#               at full speed. Context ceiling scales with it — at 4 a 13.8k
#               prompt answered and 19.2k came back empty; at 6 the ceiling sat
#               between 19k and 39k.
#
# Sampling is per-request, so editing it takes effect on the next message.
# The values below are Qwen's published "precise coding" preset.

[server]
bin =
models-dir =
model =
remote =
host = 127.0.0.1
port = 8100
context = ${DEFAULT_CONTEXT}
api-key =

[sampling]
temperature = 0.6
top_p = 0.95
top_k = 20
`

export type ServerSettings = {
  bin: string
  modelsDir: string
  model: string
  remote: string
  host: string
  port: number
  context: number
  apiKey: string
  sampling: Record<string, number>
}

const DEFAULTS: ServerSettings = {
  bin: "",
  modelsDir: "",
  model: "",
  remote: "",
  host: "127.0.0.1",
  port: 8100,
  context: DEFAULT_CONTEXT,
  apiKey: "",
  sampling: {},
}

function number(value: string | undefined, fallback: number) {
  const parsed = Number.parseInt((value ?? "").trim(), 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

/** Creates the file on first call so there is always something to point at. */
export async function load(): Promise<ServerSettings> {
  let text = await fs.readFile(FILE, "utf8").catch(() => undefined)
  if (text === undefined) {
    await fs.mkdir(path.dirname(FILE), { recursive: true }).catch(() => {})
    await fs.writeFile(FILE, TEMPLATE).catch(() => {})
    text = TEMPLATE
  }
  const doc = Ini.parse(text)
  const section = Ini.find(doc, "server")
  const sampling: Record<string, number> = {}
  const samplingSection = Ini.find(doc, "sampling")
  if (samplingSection) {
    for (const [key, raw] of Object.entries(Ini.entries(samplingSection))) {
      const parsed = Number.parseFloat(raw)
      if (Number.isFinite(parsed)) sampling[key] = parsed
    }
  }
  if (!section) return { ...DEFAULTS, sampling }
  const raw = Ini.entries(section)
  return {
    bin: expandHome((raw["bin"] ?? "").trim()),
    modelsDir: expandHome((raw["models-dir"] ?? "").trim()),
    model: (raw["model"] ?? "").trim(),
    // normalised to bare host:port so origin() can prefix the scheme exactly once
    remote: (raw["remote"] ?? "").trim().replace(/^https?:\/\//, "").replace(/\/+$/, ""),
    host: (raw["host"] || DEFAULTS.host).trim(),
    port: number(raw["port"], DEFAULTS.port),
    context: number(raw["context"], DEFAULTS.context),
    apiKey: (raw["api-key"] ?? "").trim(),
    sampling,
  }
}

export { looksRemote } from "../../shared/backends.ts"

/**
 * Writes one key back, preserving comments and everything else in the file.
 *
 * Setting `bin` clears `remote` and vice versa: they are the two answers to one
 * question — run a server here, or use one over there — and leaving the other
 * behind means the file says both and the reader has to guess which wins.
 */
export async function update(key: "bin" | "models-dir" | "model" | "remote", value: string): Promise<void> {
  const text = await fs.readFile(FILE, "utf8").catch(() => TEMPLATE)
  const doc = Ini.parse(text)
  const section = Ini.find(doc, "server")
  if (!section) return
  if (key === "remote") Ini.set(section, "bin", "")
  if (key === "bin" && value) Ini.set(section, "remote", "")
  Ini.set(section, key, key === "remote" ? value : collapseHome(value))
  await fs.mkdir(path.dirname(FILE), { recursive: true }).catch(() => {})
  await fs.writeFile(FILE, Ini.serialize(doc)).catch(() => {})
}

/**
 * ovms-serve's command line. The wrapper resolves the servable directory from
 * the model name itself, so the name is positional and comes first.
 *
 * `--rest_bind_address` is not optional. OVMS defaults it to 0.0.0.0, so
 * without it every server is on the LAN whatever `host` says — and this file
 * tells the reader that 127.0.0.1 "keeps it on this machine". Passing it makes
 * that true instead of aspirational.
 */
export function argv(settings: ServerSettings, model: string): string[] {
  return [model, "--rest_port", String(settings.port), "--rest_bind_address", settings.host]
}
