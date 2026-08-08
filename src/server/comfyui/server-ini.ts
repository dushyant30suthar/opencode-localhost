import fs from "fs/promises"
import path from "path"
import * as Ini from "../../shared/ini.ts"
import { configDir, expandHome, collapseHome } from "../../shared/paths.ts"

/**
 * server.ini for ComfyUI — how to launch it and how to reach it.
 *
 * Unlike every other backend here, ComfyUI is not an inference server for chat:
 * it has no /v1/chat/completions and nothing it produces belongs in opencode's
 * model picker. It is in this plugin for the other half of what the plugin does
 * — supervising a process that owns the GPUs. Video generation and a 27B coding
 * model cannot both be resident on 32 GiB of VRAM, so the thing you actually
 * need is one panel that can stop one and start the other.
 *
 * Two values have no safe default and stay empty until set:
 *   bin        — the python with torch importable. NOT the system python;
 *                ComfyUI wants its own venv (uv venv --python 3.12, torch from
 *                a cu128+ index — sm_120 is not in the older wheels)
 *   comfy-dir  — the ComfyUI checkout; main.py is launched from here
 *
 * `args` is the escape hatch and the interesting one on a two-GPU box. ComfyUI
 * places a whole model on one device, so the split that matters is across
 * COMPONENTS rather than layers: MiniMax H3's DiT is 20.97 GB against 15.5 GiB
 * of usable VRAM per 5060 Ti, so something has to give. --reserve-vram leaves
 * room for the VAE decode at the end of a run, and --cuda-device pins the whole
 * process to one card if you would rather keep the other free for llama.cpp.
 */

export const BACKEND = "comfyui"
export const FILE = path.join(configDir(BACKEND), "server.ini")

/** ComfyUI's own default; changing it means changing the browser bookmark too. */
const DEFAULT_PORT = 8188

const TEMPLATE = `# ComfyUI settings for opencode-localhost.
# Changing anything here restarts the server.
#
# ComfyUI is supervised here, not registered as a model provider: it serves
# images and video, not chat completions, so it deliberately contributes
# nothing to opencode's model picker. What the panel gives you is [start],
# [stop] and live VRAM — which is the part that matters when a video model
# and a coding model are competing for the same two cards.
#
#   bin         python with torch importable. REQUIRED
#               NOT the system python. Use the ComfyUI venv; sm_120 (Blackwell,
#               the 5060 Ti) needs torch from a cu128 or newer index
#   comfy-dir   the ComfyUI checkout (contains main.py). REQUIRED
#   models-dir  where checkpoints live, for the panel's listing only. Defaults
#               to <comfy-dir>/models, which is where ComfyUI looks anyway
#   remote      point at ANOTHER machine's ComfyUI instead of running one here,
#               e.g. prajna.local:8188. When set, nothing is started locally
#   host        127.0.0.1 keeps it on this machine; 0.0.0.0 makes it reachable
#               from the LAN and is passed through as --listen
#   port        ComfyUI's default is 8188
#   args        extra flags passed to main.py verbatim, space separated.
#               Worth knowing on a two-GPU box:
#                 --reserve-vram 2.0   leave headroom for the VAE decode that
#                                      runs after sampling; H3's video VAE is
#                                      5.2 GB and OOMs at the last step without
#                                      it on a 16 GB card
#                 --cuda-device 0      pin the whole process to one card, so
#                                      the other stays free for llama.cpp/exl3
#                 --disable-smart-memory  evict aggressively between runs rather
#                                      than caching a model you are done with
#
# H3 on this box: the DiT is 20.97 GB and the Qwen3-VL-32B text encoder is
# 15.69 GB at NVFP4 (native on sm_120). They do not fit at once on one card,
# but they do not need to — the encoder runs once per generation and ComfyUI
# evicts it before sampling starts. Use MultiGPU_WorkUnits in the workflow to
# put the encoder on cuda:1 if eviction is costing more than it saves.

[server]
bin =
comfy-dir =
models-dir =
remote =
host = 127.0.0.1
port = ${DEFAULT_PORT}
args =
`

export type ServerSettings = {
  bin: string
  comfyDir: string
  modelsDir: string
  remote: string
  host: string
  port: number
  args: string
}

const DEFAULTS: ServerSettings = {
  bin: "",
  comfyDir: "",
  modelsDir: "",
  remote: "",
  host: "127.0.0.1",
  port: DEFAULT_PORT,
  args: "",
}

function number(value: string | undefined, fallback: number) {
  const parsed = Number.parseInt((value ?? "").trim(), 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

/** Same reasoning as exl3's: a bare host would otherwise probe port 80. */
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
  if (!section) return { ...DEFAULTS }
  const raw = Ini.entries(section)
  const comfyDir = expandHome((raw["comfy-dir"] ?? "").trim())
  const declared = expandHome((raw["models-dir"] ?? "").trim())
  return {
    bin: expandHome((raw["bin"] ?? "").trim()),
    comfyDir,
    // ComfyUI resolves models relative to its own checkout, so defaulting here
    // keeps the panel's listing pointed at the same place the server reads from
    // rather than at nothing on a config that never set it.
    modelsDir: declared || (comfyDir ? path.join(comfyDir, "models") : ""),
    remote: withDefaultPort(
      (raw["remote"] ?? "").trim().replace(/^https?:\/\//, "").replace(/\/+$/, ""),
      DEFAULT_PORT,
    ),
    host: (raw["host"] || DEFAULTS.host).trim(),
    port: number(raw["port"], DEFAULTS.port),
    args: (raw["args"] ?? "").trim(),
  }
}

export { looksRemote } from "../../shared/backends.ts"

/**
 * Writes one key back, preserving comments and everything else in the file.
 *
 * Setting `bin` clears `remote` and vice versa — the two answers to one
 * question, exactly as in the other backends.
 */
export async function update(
  key: "bin" | "models-dir" | "comfy-dir" | "remote",
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
 * ComfyUI's command line.
 *
 * --listen takes the bind address, so 0.0.0.0 in the ini becomes the flag while
 * 127.0.0.1 passes nothing and lets ComfyUI keep its loopback default. Extra
 * args go last so anything in `args` wins over what we chose.
 */
export function argv(settings: ServerSettings): string[] {
  const flags = [path.join(settings.comfyDir, "main.py"), "--port", String(settings.port)]
  if (settings.host === "0.0.0.0") flags.push("--listen", "0.0.0.0")
  if (settings.args) flags.push(...settings.args.split(/\s+/).filter(Boolean))
  return flags
}
