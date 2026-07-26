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
    binary: "ovms",
    install: "pip install openvino-genai",
    implemented: false,
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
