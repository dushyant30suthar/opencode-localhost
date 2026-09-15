import fs from "fs/promises"
import path from "path"
import { configDir } from "../../shared/paths.ts"

/**
 * MLX checkpoints under a models directory.
 *
 * An MLX checkpoint is a DIRECTORY — config.json plus *.safetensors — not a
 * single file, so discovery walks directories instead of matching extensions.
 * Both flat (`Qwen3.5-9B-MLX-4bit/`) and nested (`publisher/repo/`) layouts
 * are recognised; a directory counts when it holds a config.json.
 *
 * The advertised id is the checkpoint's ABSOLUTE path: that is what
 * mlx_vlm.server reports in /v1/models and what it answers to in the request's
 * "model" field, so advertising anything shorter 404s every request (the same
 * lesson vLLM's served-model-name taught).
 */

export type ModelConfig = {
  /** Absolute checkpoint path. Also the picker id and the request model id. */
  id: string
  /** Relative display path under the models directory. */
  rel: string
  /** Architecture's native window (max_position_embeddings), if readable. */
  archMax?: number
  /** Presence of vision_config.model_type in config.json. */
  vision?: boolean
}

export const DIR = path.join(configDir("mlx"), "models")

async function readConfig(dir: string): Promise<{ archMax?: number; vision?: boolean } | undefined> {
  const raw = await fs.readFile(path.join(dir, "config.json"), "utf8").catch(() => undefined)
  if (!raw) return undefined
  try {
    const config = JSON.parse(raw)
    const text = config?.text_config
    const max =
      config?.max_position_embeddings ??
      text?.max_position_embeddings ??
      config?.max_model_len
    const vision: unknown = config?.vision_config
    return {
      archMax: Number.isFinite(max) && max > 0 ? max : undefined,
      // a declared tower carries its own model_type; token ids alone are not proof
      vision:
        typeof vision === "object" && vision !== null && "model_type" in vision ? true : undefined,
    }
  } catch {
    return undefined
  }
}

async function consider(dir: string, rel: string, out: ModelConfig[]): Promise<void> {
  const info = await readConfig(dir)
  if (info) {
    out.push({ id: dir, rel, archMax: info.archMax, vision: info.vision })
    return
  }
  // not a checkpoint itself: one level of nesting (publisher/repo)
  const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => [])
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith(".")) continue
    const child = path.join(dir, entry.name)
    const childInfo = await readConfig(child)
    if (childInfo) {
      out.push({
        id: child,
        rel: path.join(rel, entry.name),
        archMax: childInfo.archMax,
        vision: childInfo.vision,
      })
    }
  }
}

export async function scan(dir: string = DIR): Promise<ModelConfig[]> {
  if (!dir) return []
  const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => [])
  const found: ModelConfig[] = []
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith(".")) continue
    await consider(path.join(dir, entry.name), entry.name, found)
  }
  return found.sort((a, b) => a.id.localeCompare(b.id))
}

/** The checkpoint serving `id` (absolute path), or undefined. */
export async function fileFor(id: string, dir: string = DIR): Promise<string | undefined> {
  if (path.isAbsolute(id)) {
    const info = await readConfig(id)
    return info ? id : undefined
  }
  return (await scan(dir)).find((model) => model.id === id || model.rel === id)?.id
}
