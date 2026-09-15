import fs from "fs/promises"
import path from "path"
import * as Ini from "../../shared/ini.ts"
import { configDir, expandHome, collapseHome } from "../../shared/paths.ts"

/**
 * server.ini — ours, not MLX's. We read it and build the server command line
 * from it. Engine-level settings live here rather than per-model files because
 * mlx_lm/mlx_vlm servers take everything as flags: model, bind address, KV
 * budget. There is no preset-file concept to hand through unmodified.
 *
 * Two servers, one backend. `mlx_vlm.server` serves vision-capable checkpoints
 * (Qwen3.5-9B-MLX) and takes --max-kv-size; `mlx_lm.server` is text-only but
 * has a cross-request PROMPT CACHE: opencode repeats its system prompt every
 * turn, so the second message reuses it (measured 42s -> 0.3s on 7k tokens)
 * instead of paying prefill again. Prefer mlx_lm unless you need images.
 * `mlx_lm.server` has no KV flag — its cache grows to what the prompt needs.
 *
 * No auto-fit here. LM Studio's mlx-llm extension pack silently clamps the
 * configured context to what its working-set budget allows (see the
 * `context_fit ... fitted=32,768` lines in its server log); these servers take
 * --max-kv-size at face value and fail loudly instead of loading something
 * smaller quietly.
 */

export const BACKEND = "mlx"
export const FILE = path.join(configDir(BACKEND), "server.ini")

const DEFAULT_PORT = 8081
const DEFAULT_CONTEXT = 65_536
const DEFAULT_MAX_KV = 65_536
/** Tokens per prefill step. 2048 is upstream default; larger steps use more
 *  transient memory and measured SLOWER on an M1 Pro 16GB (8192: 45s vs
 *  2048: 40s on 7k tokens, peak 11.2 vs 8.4 GiB). Leave at 2048. */
const DEFAULT_PREFILL_STEP = 2048
/** KV cache quantization bits (0 = off). 8 halves KV memory (~2.1GB -> ~1GB
 *  at 64k on this hybrid) for negligible quality loss; single-user serving is
 *  unaffected by the batching it disables. */
const DEFAULT_KV_BITS = 8
/** APC prefix cache (mlx_vlm.server only): reuses matching prompt prefixes
 *  across requests. THE prefill fix for agentic use — opencode repeats the
 *  system prompt every turn (measured 21x warm speedup upstream). */
const DEFAULT_APC = true
/** In-memory APC budget in GiB. Prefix blocks live here; spill to disk after. */
const DEFAULT_APC_MEMORY_GB = 4
/** Disk APC budget in GiB. Survives restarts; warm-restores long prefixes. */
const DEFAULT_APC_DISK_GB = 20
/** Prompt-cache budget for mlx_lm.server: distinct cached prefixes. 8 covers
 *  system prompt + a few conversation shapes; each entry holds full KV. */
const DEFAULT_CACHE_SIZE = 8
/** Cap for the whole cache in bytes. 8 GiB leaves room for weights + live KV. */
const DEFAULT_CACHE_BYTES = 8_589_934_592

const TEMPLATE = `# MLX settings for opencode-localhost. Apple Silicon only.
# Changing anything here restarts the server.
#
#   bin        path to the server executable. REQUIRED. Normally the mlx_vlm
#              or mlx_lm entry point of a venv, e.g.
#              ~/Projects/mlx-llm-server/.venv/bin/mlx_vlm.server
#              NOT on \$PATH: a venv binary, so set it explicitly.
#   model      OPTIONAL single-checkpoint override. Leave blank to serve
#              models-dir (see below); when set, that checkpoint is served
#              and models-dir is only the panel's listing.
#   models-dir directory scanned for MLX checkpoints: subdirectories holding
#              a config.json (nested publisher/repo layouts work).
#   max-kv-size  KV cache budget in tokens (mlx_vlm.server only). THIS is the
#              context window: unlike LM Studio there is no auto-fit
#              second-guessing it.
#   kv-bits    KV cache quantization bits, 0 = off. 8 halves KV memory
#              (~2.1GB -> ~1GB at 64k on this hybrid) for negligible quality
#              loss.
#   apc        APC prefix cache (mlx_vlm.server only), true/false. Repeat
#              prompts reuse matching prefixes instead of re-prefilling.
#   apc-memory-gb / apc-disk-gb  APC budgets. Memory holds hot prefixes,
#              disk warm-restores them across restarts.
#   prefill-step  tokens per prefill step (--prefill-step-size). Upstream
#              default 2048 measured fastest on an M1 Pro 16GB; larger was
#              slower AND hungrier. Leave it unless you measure better.
#   cache-size / cache-bytes  mlx_lm.server prompt cache: how many distinct
#              prefixes and total bytes to keep. THE speed lever for agentic
#              use — opencode repeats the system prompt every turn, and a hit
#              skips prefill entirely (42s -> 0.3s measured on 7k tokens).
#              Ignored by mlx_vlm.server, which has no prompt cache.
#   thinking   pass enable_thinking to the chat template. mlx_lm's default
#              template thinks (slow first tokens); false keeps replies snappy.
#              Ignored by mlx_vlm.server (thinking off unless requested).
#   context    window advertised to opencode for compaction. Keep it equal to
#              max-kv-size, or long sessions compact earlier than they must.
#   remote     point at ANOTHER machine's MLX server, e.g. fedora.local:8081
#   host/port  must match what the server binds; used to reach, not to bind
#              (host IS passed as --host, so 127.0.0.1 really is loopback-only)
#
# Selecting a model relaunches the server against it: one model per process,
# like vLLM and TabbyAPI. Engine settings are fixed at construction.

[server]
bin =
model =
models-dir =
max-kv-size = ${DEFAULT_MAX_KV}
kv-bits = ${DEFAULT_KV_BITS}
apc = true
apc-memory-gb = ${DEFAULT_APC_MEMORY_GB}
apc-disk-gb = ${DEFAULT_APC_DISK_GB}
prefill-step = ${DEFAULT_PREFILL_STEP}
cache-size = ${DEFAULT_CACHE_SIZE}
cache-bytes = ${DEFAULT_CACHE_BYTES}
thinking = false
context = ${DEFAULT_CONTEXT}
remote =
host = 127.0.0.1
port = ${DEFAULT_PORT}
api-key =

[sampling]
temperature = 0.6
top_p = 0.95
top_k = 20
`

export type ServerSettings = {
  bin: string
  model: string
  modelsDir: string
  remote: string
  host: string
  port: number
  maxKv: number
  kvBits: number
  apc: boolean
  apcMemoryGb: number
  apcDiskGb: number
  prefillStep: number
  cacheSize: number
  cacheBytes: number
  thinking: boolean
  context: number
  apiKey: string
  sampling: Record<string, number>
}

const DEFAULTS: ServerSettings = {
  bin: "",
  model: "",
  modelsDir: "",
  remote: "",
  host: "127.0.0.1",
  port: DEFAULT_PORT,
  maxKv: DEFAULT_MAX_KV,
  kvBits: DEFAULT_KV_BITS,
  apc: DEFAULT_APC,
  apcMemoryGb: DEFAULT_APC_MEMORY_GB,
  apcDiskGb: DEFAULT_APC_DISK_GB,
  prefillStep: DEFAULT_PREFILL_STEP,
  cacheSize: DEFAULT_CACHE_SIZE,
  cacheBytes: DEFAULT_CACHE_BYTES,
  thinking: false,
  context: DEFAULT_CONTEXT,
  apiKey: "",
  sampling: {},
}

function number(value: string | undefined, fallback: number) {
  const parsed = Number.parseInt((value ?? "").trim(), 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

/** Like number() but 0 is a real value (kv-bits = off), not "unset". */
function bits(value: string | undefined, fallback: number) {
  const parsed = Number.parseInt((value ?? "").trim(), 10)
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback
}

function withDefaultPort(remote: string, port: number): string {
  if (!remote || /:\d+$/.test(remote)) return remote
  return `${remote}:${port}`
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
    model: expandHome((raw["model"] ?? "").trim()),
    modelsDir: expandHome((raw["models-dir"] ?? "").trim()),
    remote: withDefaultPort(
      (raw["remote"] ?? "").trim().replace(/^https?:\/\//, "").replace(/\/+$/, ""),
      DEFAULTS.port,
    ),
    host: (raw["host"] || DEFAULTS.host).trim(),
    port: number(raw["port"], DEFAULTS.port),
    maxKv: number(raw["max-kv-size"], DEFAULTS.maxKv),
    kvBits: bits(raw["kv-bits"], DEFAULTS.kvBits),
    apc: (raw["apc"] ?? "true").trim().toLowerCase() !== "false",
    apcMemoryGb: bits(raw["apc-memory-gb"], DEFAULTS.apcMemoryGb),
    apcDiskGb: bits(raw["apc-disk-gb"], DEFAULTS.apcDiskGb),
    prefillStep: number(raw["prefill-step"], DEFAULTS.prefillStep),
    cacheSize: number(raw["cache-size"], DEFAULTS.cacheSize),
    cacheBytes: number(raw["cache-bytes"], DEFAULTS.cacheBytes),
    thinking: (raw["thinking"] ?? "").trim().toLowerCase() === "true",
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
export async function update(
  key: "bin" | "model" | "models-dir" | "remote",
  value: string,
): Promise<void> {
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
 * Server command line. Flag set follows the binary: mlx_vlm.server takes
 * --max-kv-size/--trust-remote-code/--api-key, mlx_lm.server the prompt-cache
 * and chat-template flags instead. --kv-bits exists on both. Detection is by
 * basename so a renamed venv binary still works when it contains "vlm".
 */
/**
 * Environment for the server process. APC is env-only (no CLI flags): without
 * these the server runs with prefix caching off and every repeated system
 * prompt pays full prefill. Disk path defaults under the XDG cache so prefixes
 * survive restarts; memory budget bounds hot blocks on small Macs.
 */
export function env(settings: ServerSettings): Record<string, string> {
  if (!settings.apc || path.basename(settings.bin).includes("vlm") === false) return {}
  const home = process.env["HOME"] ?? ""
  const cache = process.env["XDG_CACHE_HOME"] || (home ? `${home}/.cache` : ".cache")
  return {
    APC_ENABLED: "1",
    APC_MEMORY_MAX_GB: String(settings.apcMemoryGb),
    APC_DISK_PATH: `${cache}/opencode/providers/mlx/apc`,
    APC_DISK_MAX_GB: String(settings.apcDiskGb),
  }
}

/**
 * Server command line. Flag set follows the binary: mlx_vlm.server takes
 * --max-kv-size/--trust-remote-code/--api-key, mlx_lm.server the prompt-cache
 * and chat-template flags instead. --kv-bits exists on both. Detection is by
 * basename so a renamed venv binary still works when it contains "vlm".
 */
export function argv(settings: ServerSettings, model: string): string[] {
  const vlm = path.basename(settings.bin).includes("vlm")
  const kv = settings.kvBits > 0 ? ["--kv-bits", String(settings.kvBits)] : []
  return [
    "--model",
    model,
    "--host",
    settings.host,
    "--port",
    String(settings.port),
    "--prefill-step-size",
    String(settings.prefillStep),
    ...kv,
    ...(vlm
      ? [
          "--max-kv-size",
          String(settings.maxKv),
          "--trust-remote-code",
          ...(settings.apiKey ? ["--api-key", settings.apiKey] : []),
        ]
      : [
          "--prompt-cache-size",
          String(settings.cacheSize),
          "--prompt-cache-bytes",
          String(settings.cacheBytes),
          "--trust-remote-code",
          ...(settings.thinking
            ? []
            : ["--chat-template-args", JSON.stringify({ enable_thinking: false })]),
        ]),
  ]
}
