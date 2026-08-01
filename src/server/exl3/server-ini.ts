import fs from "fs/promises"
import path from "path"
import * as Ini from "../../shared/ini.ts"
import { configDir, expandHome, collapseHome } from "../../shared/paths.ts"

/**
 * server.ini — ours, not TabbyAPI's. TabbyAPI reads its own YAML; this file
 * records how to launch it and how to reach it.
 *
 * Flat like OpenVINO's, and for the same reason: one server process serves
 * exactly one EXL3 model, loaded at startup. The model choice, split mode,
 * cache sizing and draft (MTP) settings all live in the TabbyAPI YAML this
 * file points at — duplicating them here would give the panel numbers the
 * server is not using.
 *
 * Three values have no safe default and stay empty until set:
 *   bin        — the python that has exllamav3 importable. NOT the system
 *                python: exllamav3 needs its own venv, and as of v1.2.1 the
 *                release wheels crash MTP on multi-GPU splits — a venv with
 *                the two local fixes applied is what actually works (see
 *                backends-qwen27b-benchmarks.md, issue turboderp-org/
 *                exllamav3#260)
 *   tabby-dir  — the tabbyAPI checkout; main.py is launched from here
 *   config     — which TabbyAPI YAML to launch with. This is the whole model
 *                choice: the YAML names the model, the split mode and whether
 *                MTP drafting is on
 */

export const BACKEND = "exl3"
export const FILE = path.join(configDir(BACKEND), "server.ini")

/**
 * Matches the served model YAML's max_seq_len. Advertising more than the YAML's
 * value gets requests rejected at the window edge; advertising less wastes
 * cache the server allocated. There is no way to read the YAML's number from
 * the wire, so this must be kept in step by hand.
 */
const DEFAULT_CONTEXT = 196_608

const TEMPLATE = `# exllamav3 (TabbyAPI) settings for opencode-localhost.
# Changing anything here restarts the server.
#
#   bin        python with exllamav3 importable. REQUIRED
#              NOT the system python. As of exllamav3 v1.2.1 the release wheels
#              crash MTP speculative decoding on multi-GPU splits; use a venv
#              carrying the fixes from turboderp-org/exllamav3#260 until they
#              land upstream
#   tabby-dir  the tabbyAPI checkout (contains main.py). REQUIRED
#   config     TabbyAPI YAML to launch with. REQUIRED. The YAML owns the model
#              name, gpu split, cache size and draft_mode — change models by
#              editing/switching the YAML and restarting
#   models-dir where EXL3 model directories live, for the panel's listing only;
#              TabbyAPI resolves models from its own YAML
#   model      the id to advertise while a REMOTE server is not answering, so
#              the provider stays in the picker on this machine. Must match the
#              far YAML's model_name. Ignored when running locally
#   remote     point at ANOTHER machine's TabbyAPI instead of running one here,
#              e.g. fedora.local:5000. When set, nothing is started locally
#   host       must match the YAML's network.host; used to reach the server,
#              not to bind it
#   port       must match the YAML's network.port
#   context    window to advertise to opencode. Keep equal to the YAML's
#              max_seq_len: higher gets requests rejected at the edge, lower
#              wastes cache the server already allocated
#   api-key    only if the YAML has auth enabled (disable_auth: false)
#
# Sampling is per-request, so editing it takes effect on the next message.
# The values below are Qwen's published "precise coding" preset.

[server]
bin =
tabby-dir =
config =
model =
models-dir =
remote =
host = 127.0.0.1
port = 5000
context = ${DEFAULT_CONTEXT}
api-key =

[sampling]
temperature = 0.6
top_p = 0.95
top_k = 20
min_p = 0
`

export type ServerSettings = {
  bin: string
  tabbyDir: string
  config: string
  model: string
  modelsDir: string
  remote: string
  host: string
  port: number
  context: number
  apiKey: string
  sampling: Record<string, number>
}

const DEFAULTS: ServerSettings = {
  bin: "",
  tabbyDir: "",
  config: "",
  model: "",
  modelsDir: "",
  remote: "",
  host: "127.0.0.1",
  port: 5000,
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
    tabbyDir: expandHome((raw["tabby-dir"] ?? "").trim()),
    config: expandHome((raw["config"] ?? "").trim()),
    model: (raw["model"] ?? "").trim(),
    modelsDir: expandHome((raw["models-dir"] ?? "").trim()),
    // normalised to bare host:port so origin() can prefix the scheme exactly once
    // normalised to bare host:port; a bare host gets this backend's default
    // port, or "192.168.1.23" probes port 80 and reads as a dead server
    remote: withDefaultPort((raw["remote"] ?? "").trim().replace(/^https?:\/\//, "").replace(/\/+$/, ""), 5000),
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
 * question — run a server here, or use one over there.
 */
export async function update(key: "bin" | "models-dir" | "tabby-dir" | "config" | "remote", value: string): Promise<void> {
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

/** TabbyAPI's command line: main.py resolves everything else from the YAML. */
export function argv(settings: ServerSettings): string[] {
  return [path.join(settings.tabbyDir, "main.py"), "--config", settings.config]
}
