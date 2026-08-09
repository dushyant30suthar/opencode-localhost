import fs from "fs/promises"
import path from "path"
import * as Ini from "../../shared/ini.ts"
import { configDir, expandHome, collapseHome } from "../../shared/paths.ts"

/**
 * server.ini — ours, not vLLM's. We read it and build the `vllm serve` command
 * line from it. The per-model engine settings do NOT live here: they live in
 * vLLM's own config YAMLs under models/, one per model (see models-dir.ts).
 *
 * Split that way because the two have different lifetimes. Everything here is
 * about reaching the server — where the binary is, what address to bind, who
 * may call it — and is the same whichever model is loaded. Everything in a
 * model YAML is fixed when the engine is constructed and can only change by
 * relaunching.
 *
 * One value has no safe default and stays empty until set:
 *   bin — the `vllm` executable. Almost never on $PATH system-wide: vLLM is a
 *         Python package and belongs in its own venv, so this is normally
 *         <venv>/bin/vllm. Point it at the venv you installed into.
 */

export const BACKEND = "vllm"
export const FILE = path.join(configDir(BACKEND), "server.ini")

/**
 * Advertised only when a model YAML declares no max-model-len of its own.
 * Deliberately modest: vLLM allocates its KV cache up front from
 * gpu-memory-utilization, and a window the cache cannot cover fails at load
 * with "The model's max seq len is larger than the maximum number of tokens
 * that can be stored in KV cache" rather than degrading. Per-model files should
 * always set their own.
 */
const DEFAULT_CONTEXT = 32_768

/** Port the vLLM control daemon listens on. See `control` in the template. */
const CONTROL_PORT = 8900

const TEMPLATE = `# vLLM settings for opencode-localhost.
# Changing anything here restarts the server.
#
#   bin        path to the 'vllm' executable. REQUIRED
#              NOT usually on \$PATH: vLLM is a Python package and wants its own
#              venv, so this is normally <venv>/bin/vllm
#   config     OPTIONAL single-model override. Leave blank to use models/.
#              When set, that one YAML is served and models/ is ignored — an
#              escape hatch for a one-off config without adding it to models/
#   models-dir where checkpoints live, for the panel's listing only. vLLM
#              resolves each model from its YAML's own 'model:' key, which may
#              equally be an HF repo id it downloads itself
#   remote     point at ANOTHER machine's vLLM, e.g. fedora.local:8000
#   control    OPTIONAL, remote only. Address of that machine's vLLM control
#              daemon, e.g. fedora.local:8900. Without it a remote shows only
#              the ONE model currently loaded and cannot be switched from here,
#              because vLLM serves a single model per process and /v1/models
#              reports just that one — there is nothing to enumerate. (llama.cpp
#              needs no equivalent: llama-server scans models-dir itself and
#              swaps on demand, so its /models already lists everything.)
#              With it set, this backend lists every model the far machine has
#              and selecting one restarts it there.
#   host/port  must match what the YAML binds; used to reach, not to bind
#              (host IS passed as --host, so 127.0.0.1 really is loopback-only)
#   context    fallback window advertised when a YAML declares no max-model-len
#
# MODEL SELECTION lives in models/, one vLLM config YAML per model:
#
#     ~/.config/opencode/providers/vllm/models/
#         <name>.yaml     any 'vllm serve' flag, minus the leading dashes
#
# The panel lists them by FILENAME, so two files may serve one checkpoint with
# different settings and both stay selectable. Give each a distinct
# 'served-model-name' as well — it is what /v1/models reports, and it is how
# this backend tells an already-running server apart from one it must relaunch.
# (exl3 needs a directory symlink for the same effect, because TabbyAPI names a
# model after its folder and has no alias. vLLM has one; use it.)
#
# Selecting a model relaunches vLLM against that file. Engine-level settings —
# tensor-parallel-size, kv-cache-dtype, speculative-config, quantization — are
# fixed when the engine is constructed, so a relaunch is the only way to change
# them. That is a real cost here: unlike a GGUF mmap, vLLM re-runs CUDA graph
# capture and KV-cache profiling on every start.
#
# Sampling is per-request, so editing it takes effect on the next message
# rather than requiring a reload.

[server]
bin =
config =
models-dir =
remote =
control =
host = 0.0.0.0
port = 8000
context = ${DEFAULT_CONTEXT}
api-key =

[sampling]
temperature = 0.6
top_p = 0.95
top_k = 20
`

export type ServerSettings = {
  bin: string
  config: string
  modelsDir: string
  remote: string
  control: string
  host: string
  port: number
  context: number
  apiKey: string
  sampling: Record<string, number>
}

const DEFAULTS: ServerSettings = {
  bin: "",
  config: "",
  modelsDir: "",
  remote: "",
  control: "",
  host: "0.0.0.0",
  port: 8000,
  context: DEFAULT_CONTEXT,
  apiKey: "",
  sampling: {},
}

function number(value: string | undefined, fallback: number) {
  const parsed = Number.parseInt((value ?? "").trim(), 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
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
    config: expandHome((raw["config"] ?? "").trim()),
    modelsDir: expandHome((raw["models-dir"] ?? "").trim()),
    // normalised to bare host:port; a bare host gets this backend's default
    // port, or "192.168.1.23" probes port 80 and reads as a dead server
    remote: withDefaultPort(
      (raw["remote"] ?? "").trim().replace(/^https?:\/\//, "").replace(/\/+$/, ""),
      DEFAULTS.port,
    ),
    // same normalisation as remote, but defaulted to the control daemon's port
    // rather than vLLM's — they are different services on the same machine
    control: withDefaultPort(
      (raw["control"] ?? "").trim().replace(/^https?:\/\//, "").replace(/\/+$/, ""),
      CONTROL_PORT,
    ),
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
export async function update(key: "bin" | "config" | "models-dir" | "remote", value: string): Promise<void> {
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
 * `vllm serve` command line.
 *
 * The model is passed POSITIONALLY as well as living in the YAML. That is not
 * redundancy for its own sake: `--config` is merged underneath the command
 * line, and passing the same value both ways is well-defined (they agree),
 * whereas relying on the config file alone depends on a `model:` key being
 * accepted there — which has moved between vLLM versions. Reading it from the
 * YAML and passing it explicitly works on every version that has `--config`.
 *
 * --host and --port likewise override whatever the YAML says, so the address
 * this plugin probes is guaranteed to be the address vLLM binds. A YAML that
 * disagreed would otherwise produce a server nobody can find.
 */
export function argv(settings: ServerSettings, model: string, configFile: string): string[] {
  return [
    "serve",
    model,
    "--config",
    configFile,
    "--host",
    settings.host,
    "--port",
    String(settings.port),
    ...(settings.apiKey ? ["--api-key", settings.apiKey] : []),
  ]
}
