import fs from "fs/promises"
import path from "path"
import os from "os"

/**
 * The backends this plugin knows about.
 *
 * Only llama.cpp is implemented. The others are listed anyway so the setup
 * screen can tell you what it found and what installing them would take —
 * a greyed row with a real install command is more use than pretending the
 * option does not exist.
 *
 * Note these are not the same kind of thing: llama.cpp ships a native binary
 * with prebuilt releases, while vLLM, MLX and OpenVINO are Python packages.
 * That is why this plugin never installs anything for you — "install" means
 * something different for each, and picking the wrong llama.cpp build (CUDA
 * version, GPU arch) silently costs a large multiple of performance.
 */

export type BackendSpec = {
  id: string
  name: string
  /** Executable to look for on $PATH. */
  binary: string
  /** What a user would actually run to get it. */
  install: string
  /** Restricts availability; MLX is Apple Silicon only. */
  requires?: { platform: NodeJS.Platform; arch: string }
  implemented: boolean
}

export const BACKENDS: BackendSpec[] = [
  {
    id: "llamacpp",
    name: "llama.cpp",
    binary: "llama-server",
    install: "build from source, or: brew install llama.cpp",
    implemented: true,
  },
  {
    id: "vllm",
    name: "vLLM",
    binary: "vllm",
    install: "pip install vllm",
    implemented: false,
  },
  {
    id: "mlx",
    name: "MLX",
    binary: "mlx_lm.server",
    install: "pip install mlx-lm",
    requires: { platform: "darwin", arch: "arm64" },
    implemented: false,
  },
  {
    id: "openvino",
    name: "OpenVINO",
    // the generated wrapper, not bare `ovms`: that binary links libpython and
    // exits before main() unless LD_LIBRARY_PATH is already pointing at it
    binary: "ovms-serve",
    // Not a pip package — `pip install openvino-genai` gets you the Python
    // library, which has no server in it. OVMS ships as a prebuilt archive,
    // and the python_on build is the one to take: the C++-only python_off
    // build renders chat templates with a cut-down engine that drops the
    // system message and cannot emit tool calls, which an agent client needs.
    install: "download the python_on build from github.com/openvinotoolkit/model_server/releases",
    implemented: true,
  },
]

export function supported(spec: BackendSpec): boolean {
  if (!spec.requires) return true
  return process.platform === spec.requires.platform && process.arch === spec.requires.arch
}

/** First match on $PATH, or undefined. We never guess at build directories. */
export async function onPath(binary: string): Promise<string | undefined> {
  const dirs = (process.env["PATH"] || "").split(path.delimiter).filter(Boolean)
  for (const dir of dirs) {
    const candidate = path.join(dir, binary)
    const stat = await fs.stat(candidate).catch(() => undefined)
    if (stat?.isFile()) return candidate
  }
  return undefined
}

/** Every match on $PATH — several llama.cpp builds commonly coexist. */
export async function allOnPath(binary: string): Promise<string[]> {
  const dirs = (process.env["PATH"] || "").split(path.delimiter).filter(Boolean)
  const found: string[] = []
  for (const dir of dirs) {
    const candidate = path.join(dir, binary)
    const stat = await fs.stat(candidate).catch(() => undefined)
    if (stat?.isFile() && !found.includes(candidate)) found.push(candidate)
  }
  return found
}

export function expand(value: string): string {
  const trimmed = value.trim()
  if (trimmed === "~") return os.homedir()
  if (trimmed.startsWith("~/")) return path.join(os.homedir(), trimmed.slice(2))
  return trimmed
}

/**
 * A path means "run a server here"; an address means "use the one over there".
 * One field answers both, because from the user's side it is one question —
 * where does inference happen — and asking it twice invites a config that says
 * both.
 *
 * Anything with a scheme, or a host:port, or a bare hostname that is clearly
 * not a path, is an address. A leading / or ~ is always a path.
 *
 * Lives here rather than beside one backend's settings because every backend
 * that can point at another machine needs the same answer.
 */
export function looksRemote(input: string): boolean {
  const text = input.trim()
  if (!text) return false
  if (text.startsWith("/") || text.startsWith("~") || text.startsWith(".")) return false
  if (/^https?:\/\//i.test(text)) return true
  return /^[a-z0-9][a-z0-9.-]*(:\d+)?$/i.test(text) && (text.includes(":") || text.includes("."))
}

/** Bare host:port, however the user typed it. */
export function normalizeRemote(input: string): string {
  return input.trim().replace(/^https?:\/\//i, "").replace(/\/+$/, "")
}
