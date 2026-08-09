import fs from "fs/promises"
import path from "path"
import { configDir } from "../../shared/paths.ts"

/**
 * One vLLM config YAML per model, under
 * ~/.config/opencode/providers/vllm/models/.
 *
 * Same shape as exl3's for the same reason: vLLM serves exactly one model per
 * process, so there is nothing to route and nothing to translate. These files
 * ARE vLLM's own `--config` format — every key is a `vllm serve` flag with the
 * leading dashes dropped — so what you read here is what the engine is given.
 *
 * The per-file cost is repeating shared settings (tensor-parallel-size,
 * kv-cache-dtype, gpu-memory-utilization...) in each. That is deliberate and
 * matches the rest of this plugin: a file is complete and readable on its own,
 * and the tuning comments live next to the values they explain.
 *
 * Selecting a model relaunches vLLM against its file. Unlike exl3 that is not
 * strictly forced by a VRAM-return bug — vLLM frees on exit normally — but it
 * is still the only way to change engine-level settings (parallelism, cache
 * dtype, speculative config), which are fixed at construction.
 */

export type ModelConfig = {
  /**
   * The YAML's basename. The internal handle — what launches, and what the
   * control daemon keys its entries by.
   *
   * NOT what the picker advertises: that is `served`, because the selected id
   * becomes the request's `model` field and vLLM answers only to
   * served-model-name. A file named NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4.yaml
   * serving `Nemotron-3-Nano-30B-A3B-NVFP4` made every request 404 with
   * "The model X does not exist".
   *
   * The consequence is that **served names must be unique across files**.
   * Several files can serve one checkpoint and differ only in how — 163k
   * context without vision against 98k with it, say — and if they share a
   * served name they collapse into one entry, leaving the others unreachable.
   * Give each its own alias; vLLM treats served-model-name as free-form.
   */
  id: string
  /** Absolute path to the YAML to launch with. */
  file: string
  /**
   * What /v1/models will report: `served-model-name` when set, else `model`.
   *
   * Prefer setting served-model-name explicitly and uniquely per file. vLLM
   * accepts it as a plain alias, so two configs over one checkpoint can each
   * announce themselves distinctly — the same problem exl3 can only solve with
   * a directory symlink, because TabbyAPI derives the served name from the
   * checkpoint folder and has no alias of its own.
   */
  served: string
  /** The model vLLM loads: an HF repo id, or an absolute path to a checkout. */
  model: string
  /** max-model-len, so the panel advertises the window the model really loads. */
  context?: number
}

export const DIR = path.join(configDir("vllm"), "models")

/**
 * No YAML parser on purpose — the same trade exl3's models-dir.ts makes. Every
 * key read here is a single plain scalar on its own line in any file this
 * plugin writes or documents, and a dependency to read four fields would
 * outweigh the fragility: a file that does not match simply does not list,
 * which is visible rather than silent.
 *
 * Quotes are tolerated because HF repo ids contain a slash and people quote
 * them out of habit; vLLM itself does not care either way.
 */
function field(raw: string, key: string): string | undefined {
  const match = raw.match(new RegExp(`^\\s*${key}:\\s*(.+?)\\s*$`, "m"))
  const value = match?.[1]
  if (value === undefined) return undefined
  const unquoted = value.replace(/^["']|["']$/g, "").trim()
  // a commented-out or empty value is not a value
  if (!unquoted || unquoted.startsWith("#")) return undefined
  return unquoted
}

export async function scan(dir: string = DIR): Promise<ModelConfig[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => [])
  const found: ModelConfig[] = []
  for (const entry of entries) {
    if (!entry.isFile()) continue
    if (!/\.ya?ml$/i.test(entry.name)) continue
    const file = path.join(dir, entry.name)
    const raw = await fs.readFile(file, "utf8").catch(() => undefined)
    if (!raw) continue
    const model = field(raw, "model")
    // A config with no `model` cannot be launched into a known state, so it is
    // not a model — skip rather than advertise something unselectable.
    if (!model) continue
    const len = Number.parseInt(field(raw, "max-model-len") ?? "", 10)
    found.push({
      id: entry.name.replace(/\.ya?ml$/i, ""),
      file,
      served: field(raw, "served-model-name") ?? model,
      model,
      context: Number.isFinite(len) ? len : undefined,
    })
  }
  return found.sort((a, b) => a.id.localeCompare(b.id))
}

/** The file that serves `id`, or undefined when nothing declares it. */
export async function fileFor(id: string, dir: string = DIR): Promise<string | undefined> {
  const declared = await scan(dir)
  // served-model-name first: that is what the backend advertises and what vLLM
  // answers to. Falling back to the filename keeps ids recorded before this
  // resolving, and covers files whose basename and served name agree.
  return (declared.find((model) => model.served === id) ?? declared.find((model) => model.id === id))?.file
}
