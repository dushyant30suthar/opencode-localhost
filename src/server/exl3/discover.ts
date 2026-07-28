import fs from "fs/promises"
import path from "path"

/**
 * EXL3 model directories under models-dir.
 *
 * An EXL3 checkpoint is a directory of safetensors shards with a config.json
 * carrying a quantization block. TabbyAPI resolves the served model from its
 * own YAML, so this listing exists for the panel — to show what could be
 * served — not to decide what is.
 */

export type LocalModel = {
  id: string
  dir: string
}

async function isExl3Model(dir: string): Promise<boolean> {
  const entries = await fs.readdir(dir).catch(() => [] as string[])
  if (!entries.some((entry) => entry.endsWith(".safetensors"))) return false
  const raw = await fs.readFile(path.join(dir, "config.json"), "utf8").catch(() => undefined)
  if (!raw) return false
  try {
    const config = JSON.parse(raw)
    // exllamav3's converter stamps quantization_config.quant_method = "exl3";
    // an unquantized HF checkpoint also passes the shard test above, and
    // TabbyAPI would try to load it onto GPUs it cannot fit
    return config?.quantization_config?.quant_method === "exl3"
  } catch {
    return false
  }
}

/** One level deep, like the other backends: models-dir/<name>/config.json. */
export async function scan(modelsDir: string): Promise<LocalModel[]> {
  const found: LocalModel[] = []
  const entries = await fs.readdir(modelsDir, { withFileTypes: true }).catch(() => [])
  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    const dir = path.join(modelsDir, entry.name)
    if (await isExl3Model(dir)) found.push({ id: entry.name, dir })
  }
  return found.sort((a, b) => a.id.localeCompare(b.id))
}
