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

/** One engine. A machine can have several, each its own opencode provider. */
export type BackendPanel = {
  id: string
  name: string
  status: ProviderStatus
  loaded?: LoadedModel
}

export type PanelData = {
  /**
   * Only backends the user has actually configured. Listing every engine we
   * know about would put three dead rows on screen forever; /localhost is
   * where the full list with install state lives.
   */
  backends: BackendPanel[]
  /** Whether opencode has picked any of these up as providers yet. */
  registered?: boolean
  /** Shared: one GPU pool no matter how many engines are installed. */
  gpus: GpuStat[]
  memory?: SystemStat
  /** Tokens per second, while generating. */
  throughput?: number
}
