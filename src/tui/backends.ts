import { create as llamacpp } from "../server/llamacpp/index.ts"
import { create as openvino } from "../server/openvino/index.ts"
import { create as exl3 } from "../server/exl3/index.ts"
import * as LlamacppIni from "../server/llamacpp/server-ini.ts"
import * as OpenvinoIni from "../server/openvino/server-ini.ts"
import * as Exl3Ini from "../server/exl3/server-ini.ts"
import type { Backend } from "../server/backend.ts"

/**
 * The engines this TUI half talks to, shared so the panel, the setup screen
 * and the commands all see the same instances and the same busy state.
 *
 * A machine can have several; each is its own opencode provider. Keep this in
 * step with the server half's list — a backend registered there but not here
 * serves models while the panel offers no way to start or stop it.
 */
export const BACKENDS: Backend[] = [llamacpp(), openvino(), exl3()]

export function backendById(id: string): Backend | undefined {
  return BACKENDS.find((backend) => backend.id === id)
}

/**
 * The settings file behind each engine, keyed the same way.
 *
 * The setup screen has to read and write whichever backend's row the user is
 * standing on. Reaching straight for one backend's module — as it did while
 * llama.cpp was the only implemented one — silently shows and edits llama.cpp's
 * config under every other backend's name.
 *
 * Structural typing keeps this honest: a settings module that drifts from this
 * shape fails to compile here rather than misbehaving at runtime.
 */
export type BackendConfig = {
  FILE: string
  load(): Promise<{
    bin: string
    modelsDir: string
    remote: string
    host: string
    port: number
  }>
  update(key: "bin" | "models-dir" | "remote", value: string): Promise<void>
  looksRemote(input: string): boolean
}

export const CONFIGS: Record<string, BackendConfig> = {
  llamacpp: LlamacppIni,
  openvino: OpenvinoIni,
  exl3: Exl3Ini,
}

export function configById(id: string): BackendConfig | undefined {
  return CONFIGS[id]
}
