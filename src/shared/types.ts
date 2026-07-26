/** Shapes the panel renders. Deliberately backend-agnostic. */

export type GpuStat = {
  index: number
  usedMiB: number
  totalMiB: number
  /** Percent 0-100. Undefined when the driver does not report it. */
  utilization?: number
}

export type SystemStat = {
  usedMiB: number
  totalMiB: number
  utilization?: number
}

export type LoadedModel = {
  id: string
  /** Launch flags the server reports, e.g. ctx-size, cache-type-k, gpu-layers. */
  args: Record<string, string>
  loading?: boolean
  /** 0-1 while weights stream in. */
  progress?: number
}

/**
 * "unconfigured" is a first-class state, not an error: a fresh install has no
 * models directory and the panel is where the user is told so.
 */
export type ProviderStatus =
  | { state: "unconfigured"; missing: "binary" | "models-dir"; message: string; hint?: string }
  | { state: "stopped" }
  | { state: "starting" }
  | { state: "running"; endpoint: string; loaded?: LoadedModel }
  | { state: "failed"; message: string; hint?: string }

export type PanelData = {
  backend: { id: string; name: string }
  status: ProviderStatus
  /**
   * Whether opencode has actually picked up the provider. False while the
   * server is up but opencode has not re-read its config — the one case where
   * a restart is genuinely required, so the panel has to say so.
   */
  registered?: boolean
  gpus: GpuStat[]
  memory?: SystemStat
  cpu?: SystemStat
  /** Tokens per second, while generating. */
  throughput?: number
}
