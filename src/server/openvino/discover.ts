import fs from "fs/promises"
import path from "path"

/**
 * Find OpenVINO IR models under the models directory.
 *
 * An IR model is a directory, not a file: weights in a .bin, topology in a
 * .xml, tokenizer beside them. Two shapes exist and both are servable —
 *
 *   openvino_model.xml           a text-only export
 *   openvino_language_model.xml  a vision-language export, whose language
 *                                half is the same thing with the vision
 *                                towers packaged alongside
 *
 * The directory name is the id, because that is also what OVMS advertises as
 * the servable name — keeping them the same means the id in opencode's picker
 * is the id the server answers to.
 */

const MAX_DEPTH = 3

const TEXT_ONLY = "openvino_model.xml"
const LANGUAGE_HALF = "openvino_language_model.xml"

export type LocalModel = {
  /** Directory name, which is also the OVMS servable name. */
  id: string
  dir: string
  /**
   * A vision-language export. Worth knowing beyond mere capability: OVMS
   * classifies these as VLM servables and then allows only the chat endpoints,
   * and OpenVINO refuses speculative decoding for them ("not supported for
   * models with embeddings"). Several speed levers are simply closed here.
   */
  vlm: boolean
}

async function has(dir: string, file: string): Promise<boolean> {
  const stat = await fs.stat(path.join(dir, file)).catch(() => undefined)
  return !!stat?.isFile()
}

async function walk(dir: string, relative: string[], out: LocalModel[]): Promise<void> {
  if (relative.length > MAX_DEPTH) return
  const textOnly = await has(dir, TEXT_ONLY)
  const language = await has(dir, LANGUAGE_HALF)
  if (textOnly || language) {
    out.push({
      id: relative.at(-1) ?? path.basename(dir),
      dir,
      vlm: language && !textOnly,
    })
    // a model directory holds no further models
    return
  }
  const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => [])
  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    await walk(path.join(dir, entry.name), [...relative, entry.name], out)
  }
}

export async function scan(root: string): Promise<LocalModel[]> {
  if (!root) return []
  const found: LocalModel[] = []
  await walk(root, [], found)
  const seen = new Set<string>()
  return found
    .filter((model) => (seen.has(model.id) ? false : (seen.add(model.id), true)))
    .sort((a, b) => a.id.localeCompare(b.id))
}
