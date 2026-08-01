import fs from "fs/promises"
import path from "path"
import { configDir } from "../../shared/paths.ts"

/**
 * One TabbyAPI YAML per model, under
 * ~/.config/opencode/providers/exl3/models/.
 *
 * Why a directory of YAMLs rather than a models.ini like llamacpp's: that file
 * is llama.cpp's own format, handed to `llama-server --models-preset` verbatim,
 * and llama.cpp does the model routing itself. TabbyAPI has no equivalent — it
 * takes one YAML naming one model — so there is nothing to hand it and no
 * translation worth writing. These files ARE TabbyAPI's format, and the backend
 * simply picks which one to launch.
 *
 * The per-file cost is repeating shared settings (cache_mode, tensor_parallel,
 * chunk_size...) in each. That is deliberate: a file is complete and readable on
 * its own, and the tuning comments live next to the values they explain.
 *
 * Selecting a model relaunches TabbyAPI against its file. The relaunch is not
 * incidental — exllamav3 does not return VRAM to the driver on unload, so an
 * in-process swap strands it (measured 2026-08-01: 6.00 -> 5.00bpw left 29.2 GiB
 * allocated with no model loaded, and could not reload). Killing the process is
 * what reclaims it, which is why inline_model_loading stays off.
 */

export type ModelConfig = {
  /**
   * The YAML's basename, and the id the picker selects by.
   *
   * NOT the YAML's model_name. Several files can name the same checkpoint and
   * differ only in how they serve it — 5.00bpw with MTP at four slots versus
   * 5.00bpw with the DFlash drafter at one, say. Keying on model_name collapses
   * those into one entry, and whichever sorted first won every lookup, leaving
   * the others unreachable from the panel.
   */
  id: string
  /** Absolute path to the YAML to launch with. */
  file: string
  /** TabbyAPI's model_name — what the server will report once loaded. */
  served: string
  /** max_seq_len, so the panel advertises the window the model really loads. */
  context?: number
}

export const DIR = path.join(configDir("exl3"), "models")

/**
 * No YAML parser on purpose. Both keys are single plain-scalar lines in every
 * file this plugin writes or documents, and a dependency to read two fields
 * would outweigh the fragility — a file that does not match simply does not
 * list, which is visible rather than silent.
 */
function field(raw: string, key: string): string | undefined {
  return raw.match(new RegExp(`^\\s*${key}:\\s*(\\S+)\\s*$`, "m"))?.[1]
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
    const served = field(raw, "model_name")
    // A YAML with no model_name cannot be launched into a known state, so it is
    // not a model — skip rather than advertise something unselectable.
    if (!served) continue
    const seq = Number.parseInt(field(raw, "max_seq_len") ?? "", 10)
    found.push({
      id: entry.name.replace(/\.ya?ml$/i, ""),
      file,
      served,
      context: Number.isFinite(seq) ? seq : undefined,
    })
  }
  return found.sort((a, b) => a.id.localeCompare(b.id))
}

/** The file that serves `id`, or undefined when nothing declares it. */
export async function fileFor(id: string, dir: string = DIR): Promise<string | undefined> {
  return (await scan(dir)).find((model) => model.id === id)?.file
}
