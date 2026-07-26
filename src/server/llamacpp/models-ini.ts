import fs from "fs/promises"
import path from "path"
import * as Ini from "../../shared/ini.ts"
import { configDir, expandHome } from "../../shared/paths.ts"
import type { LocalModel } from "./discover.ts"

/**
 * models.ini — llama.cpp's file, not ours. Handed to
 * `llama-server --models-preset` verbatim; llama.cpp parses it, we only append.
 *
 * The contract with the user: we add a section when a new .gguf appears and
 * never modify or remove one afterwards. Everything in here is a llama-server
 * flag, so any flag llama.cpp accepts works whether or not we know about it.
 */

export const FILE = path.join(configDir("llamacpp"), "models.ini")

const HEADER = `version = 1

# Per-model llama-server settings. YOURS TO EDIT.
# New models get a section appended; existing sections are never touched.
# Any llama-server flag works as a key, without the leading dashes:
#
#   ctx-size = 32768          context window, in tokens
#   gpu-layers = 99           layers on GPU (99 = all)
#   tensor-split = 0.6,0.4    split across multiple GPUs
#   cache-type-k = q8_0       KV cache quantisation: f16 | q8_0 | q4_0
#   flash-attn = on
#   temp = 0.7                sampling; sent per-request, no reload needed
#
# Launch flags apply on the next model load. Sampling applies immediately.`

/**
 * Conservative on purpose: these must not OOM on a machine we know nothing
 * about. 32k context with a quantised KV cache fits alongside most models on
 * a single mid-range card. Tune upward per model; that is what the file is for.
 */
const DEFAULTS: Record<string, string> = {
  "ctx-size": "32768",
  "gpu-layers": "99",
  "flash-attn": "on",
  "cache-type-k": "q8_0",
  "cache-type-v": "q8_0",
  jinja: "true",
}

export type ModelSettings = {
  context: number
  sampling: Record<string, number>
}

const SAMPLING_KEYS = ["temp", "top-p", "top-k", "min-p", "repeat-penalty", "presence-penalty"] as const

/** llama.cpp flag names -> OpenAI request body fields. */
const SAMPLING_WIRE: Record<string, string> = {
  temp: "temperature",
  "top-p": "top_p",
  "top-k": "top_k",
  "min-p": "min_p",
  "repeat-penalty": "repetition_penalty",
  "presence-penalty": "presence_penalty",
}

function section(model: LocalModel): string {
  const lines = [`[${model.id}]`, `model = ${model.file}`]
  if (model.mmproj) lines.push(`mmproj = ${model.mmproj}`)
  for (const [key, value] of Object.entries(DEFAULTS)) lines.push(`${key} = ${value}`)
  return lines.join("\n")
}

export type SyncResult = {
  preset: string
  /** Discovered file -> the section name that actually governs it. */
  names: Map<string, string>
}

/**
 * Append sections for models that do not have one yet.
 *
 * The section name IS the model id, so a section the user renamed keeps
 * winning: we match on the `model =` path first and only fall back to the
 * proposed name for genuinely new files. Getting this backwards would append a
 * duplicate default section beside a hand-tuned one and then read `ctx-size`
 * from the wrong one — silently compacting far below the real window.
 */
export async function sync(models: LocalModel[]): Promise<SyncResult> {
  const existing = await fs.readFile(FILE, "utf8").catch(() => "")
  const doc = Ini.parse(existing)
  // llama-server opens these paths literally — it does not expand `~`, so a
  // hand-written `model = ~/...` fails with "failed to open GGUF file". This is
  // the one case where an existing section is rewritten, because leaving it
  // alone means the model silently never loads.
  let rewrote = false
  for (const item of doc.sections) {
    for (const key of ["model", "mmproj"] as const) {
      const raw = Ini.get(item, key)?.trim()
      if (!raw || !raw.startsWith("~")) continue
      Ini.set(item, key, expandHome(raw))
      rewrote = true
    }
  }
  const names = new Set(doc.sections.map((item) => item.name))

  // Compare expanded paths. Hand-written sections routinely use `~/...` while
  // discovery always produces absolute paths; comparing the raw strings makes
  // an already-tuned model look new, and appends a default 32k section beside
  // it that then wins — silently replacing tuned settings.
  const byFile = new Map<string, string>()
  for (const item of doc.sections) {
    const file = Ini.get(item, "model")?.trim()
    if (file) byFile.set(expandHome(file), item.name)
  }

  const added: string[] = []
  const resolved = new Map<string, string>()
  for (const model of models) {
    const owner = byFile.get(model.file)
    if (owner) {
      resolved.set(model.file, owner)
      continue
    }
    // a section named for this model but pointing elsewhere still owns the name
    let id = model.id
    for (let n = 2; names.has(id); n++) id = `${model.id}#${n}`
    names.add(id)
    resolved.set(model.file, id)
    added.push(section({ ...model, id }))
  }

  const source = rewrote ? Ini.serialize(doc) : existing
  const body = source.trim().length > 0 ? source.replace(/\s+$/, "") : HEADER
  const next = added.length > 0 ? [body, "", added.join("\n\n"), ""].join("\n") : body + "\n"

  await fs.mkdir(path.dirname(FILE), { recursive: true }).catch(() => {})
  await fs.writeFile(FILE, next).catch(() => {})
  return { preset: FILE, names: resolved }
}

/**
 * Per-model context and sampling, read back out of the file.
 *
 * Context matters beyond display: opencode compacts a conversation against the
 * model's advertised window, so if we reported a default while the model
 * actually loaded with 245k, it would compact at a fraction of the real limit.
 */
export async function settings(): Promise<Record<string, ModelSettings>> {
  const text = await fs.readFile(FILE, "utf8").catch(() => "")
  const out: Record<string, ModelSettings> = {}
  for (const item of Ini.parse(text).sections) {
    const raw = Ini.entries(item)
    const context = Number.parseInt(raw["ctx-size"] ?? "", 10)
    const sampling: Record<string, number> = {}
    for (const key of SAMPLING_KEYS) {
      const value = Number.parseFloat(raw[key] ?? "")
      if (Number.isFinite(value)) sampling[SAMPLING_WIRE[key]] = value
    }
    out[item.name] = {
      context: Number.isFinite(context) && context > 0 ? context : 0,
      sampling,
    }
  }
  return out
}

/** Reset one section to the shipped defaults, keeping model/mmproj paths. */
export async function restoreDefaults(id: string): Promise<boolean> {
  const text = await fs.readFile(FILE, "utf8").catch(() => "")
  const doc = Ini.parse(text)
  const target = Ini.find(doc, id)
  if (!target) return false
  const raw = Ini.entries(target)
  const lines = [`model = ${raw["model"] ?? ""}`]
  if (raw["mmproj"]) lines.push(`mmproj = ${raw["mmproj"]}`)
  for (const [key, value] of Object.entries(DEFAULTS)) lines.push(`${key} = ${value}`)
  target.lines = lines
  await fs.writeFile(FILE, Ini.serialize(doc)).catch(() => {})
  return true
}
