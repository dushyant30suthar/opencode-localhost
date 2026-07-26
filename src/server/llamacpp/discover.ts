import fs from "fs/promises"
import path from "path"

/**
 * Find .gguf files under the models directory.
 *
 * llama-server's own --models-dir scan is one level deep, but the common
 * layouts nest: LM Studio uses <root>/<publisher>/<repo>/*.gguf and a plain
 * download folder is flat. We walk a bounded depth so both work without the
 * user having to know which one they have.
 */

const MAX_DEPTH = 3

export type LocalModel = {
  /** Section name: "publisher/repo:Q6_K", or "publisher/repo" when unambiguous. */
  id: string
  file: string
  /** Vision projector beside the weights, if present. */
  mmproj?: string
}

type Found = {
  dir: string
  /** Path segments below the models root, for naming. */
  relative: string[]
  files: string[]
  mmproj?: string
}

/** "Qwen3.6-27B-Q6_K.gguf" -> "Q6_K". Multi-shard suffixes stripped first. */
export function quantOf(filename: string): string | undefined {
  const stem = filename.replace(/\.gguf$/i, "").replace(/-\d{5}-of-\d{5}$/, "")
  return stem.match(/[-.]((?:I?Q\d[\w.]*)|F16|F32|BF16|MXFP4[\w]*|UD-[\w.]+)$/i)?.[1]
}

function nameFor(relative: string[], filename: string, multiple: boolean): string {
  const base = relative.length > 0 ? relative.join("/").replace(/-GGUF$/i, "") : "models"
  if (!multiple) return base
  const quant = quantOf(filename)
  return `${base}:${quant ?? filename.replace(/\.gguf$/i, "")}`
}

async function walk(dir: string, relative: string[], out: Found[]): Promise<void> {
  if (relative.length > MAX_DEPTH) return
  const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => [])
  const files: string[] = []
  let mmproj: string | undefined
  for (const entry of entries) {
    if (entry.isDirectory()) {
      await walk(path.join(dir, entry.name), [...relative, entry.name], out)
      continue
    }
    if (!entry.isFile() || !entry.name.toLowerCase().endsWith(".gguf")) continue
    if (/mmproj/i.test(entry.name)) {
      mmproj = entry.name
      continue
    }
    // multi-shard models: only the first shard is passed to llama-server
    if (/-\d{5}-of-\d{5}\.gguf$/i.test(entry.name)) {
      if (entry.name.includes("-00001-of-")) files.push(entry.name)
      continue
    }
    files.push(entry.name)
  }
  if (files.length > 0) out.push({ dir, relative, files: files.sort(), mmproj })
}

export async function scan(root: string): Promise<LocalModel[]> {
  if (!root) return []
  const found: Found[] = []
  await walk(root, [], found)

  const models: LocalModel[] = []
  const taken = new Set<string>()
  for (const group of found) {
    // one entry per quant file, so every variant is separately selectable
    const multiple = group.files.length > 1
    for (const file of group.files) {
      let id = nameFor(group.relative, file, multiple)
      if (taken.has(id)) id = nameFor(group.relative, file, true)
      if (taken.has(id)) continue
      taken.add(id)
      models.push({
        id,
        file: path.join(group.dir, file),
        mmproj: group.mmproj ? path.join(group.dir, group.mmproj) : undefined,
      })
    }
  }
  return models.sort((a, b) => a.id.localeCompare(b.id))
}
