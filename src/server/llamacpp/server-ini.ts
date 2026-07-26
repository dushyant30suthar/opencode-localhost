import fs from "fs/promises"
import path from "path"
import * as Ini from "../../shared/ini.ts"
import { configDir, expandHome, collapseHome } from "../../shared/paths.ts"

/**
 * server.ini — ours, not llama.cpp's. We read it and build llama-server's
 * command line from it. llama-server never sees this file.
 *
 * Two values have no safe default and are left empty until set:
 *   bin        — filled from $PATH if llama-server is there, else empty
 *   models-dir — always empty on a fresh install; only the user knows this
 *
 * Empty is not a failure mode to hide. The panel reads it and asks.
 */

export const BACKEND = "llamacpp"
export const FILE = path.join(configDir(BACKEND), "server.ini")

const TEMPLATE = `# llama-server settings for opencode-localhost.
# Changing anything here restarts the server.
#
#   bin         path to llama-server. empty = look on $PATH
#   models-dir  where your .gguf files live. REQUIRED, no default
#   host        127.0.0.1 keeps it on this machine.
#               0.0.0.0 exposes it to your network — set api-key if you do
#   port        llama-server listen port
#   models-max  how many models may sit in VRAM at once. 1 swaps instead of stacking
#   api-key     required by the server when set. leave empty for localhost-only

[server]
bin =
models-dir =
host = 127.0.0.1
port = 9337
models-max = 1
api-key =
`

export type ServerSettings = {
  bin: string
  modelsDir: string
  host: string
  port: number
  modelsMax: number
  apiKey: string
}

const DEFAULTS: ServerSettings = {
  bin: "",
  modelsDir: "",
  host: "127.0.0.1",
  port: 9337,
  modelsMax: 1,
  apiKey: "",
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
  const section = Ini.find(Ini.parse(text), "server")
  if (!section) return { ...DEFAULTS }
  const raw = Ini.entries(section)
  return {
    bin: expandHome((raw["bin"] ?? "").trim()),
    modelsDir: expandHome((raw["models-dir"] ?? "").trim()),
    host: (raw["host"] || DEFAULTS.host).trim(),
    port: number(raw["port"], DEFAULTS.port),
    modelsMax: number(raw["models-max"], DEFAULTS.modelsMax),
    apiKey: (raw["api-key"] ?? "").trim(),
  }
}

/** Writes one key back, preserving comments and everything else in the file. */
export async function update(key: "bin" | "models-dir", value: string): Promise<void> {
  const text = await fs.readFile(FILE, "utf8").catch(() => TEMPLATE)
  const doc = Ini.parse(text)
  const section = Ini.find(doc, "server")
  if (!section) return
  Ini.set(section, key, collapseHome(value))
  await fs.mkdir(path.dirname(FILE), { recursive: true }).catch(() => {})
  await fs.writeFile(FILE, Ini.serialize(doc)).catch(() => {})
}

/** llama-server's command line, built from the settings plus the preset file. */
export function argv(settings: ServerSettings, preset: string): string[] {
  return [
    "--models-dir",
    settings.modelsDir,
    "--models-preset",
    preset,
    "--models-max",
    String(settings.modelsMax),
    "--host",
    settings.host,
    "--port",
    String(settings.port),
    ...(settings.apiKey ? ["--api-key", settings.apiKey] : []),
  ]
}
