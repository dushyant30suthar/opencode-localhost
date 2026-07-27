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
  /** Which stage is streaming, e.g. "text_model" or "mmproj". */
  stage?: string
}

/** A model changing state, streamed while it loads. */
export type LoadEvent = {
  model: string
  loading: boolean
  loaded?: boolean
  failed?: boolean
  /** 0-1 for the stage named below. */
  progress?: number
  stage?: string
}

/**
 * "unconfigured" is a first-class state, not an error: a fresh install has no
 * models directory and the panel is where the user is told so.
 */
export type ProviderStatus =
  | { state: "unconfigured"; missing: "binary" | "models-dir"; message: string; hint?: string }
  | { state: "stopped" }
  | { state: "starting" }
  // `lan` is set only when the server binds 0.0.0.0. The endpoint stays loopback
  // because that is how this machine reaches it; the panel needs the routable
  // address separately, or you cannot tell what to point another machine at.
  | { state: "running"; endpoint: string; lan?: string; loaded?: LoadedModel }
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
