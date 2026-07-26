import type { ProviderStatus } from "../shared/types.ts"

/**
 * What every backend implements. llama.cpp today; vLLM and OpenVINO are new
 * folders implementing this, with no change to the plugin around them.
 *
 * A backend owns exactly three things: knowing what models exist, getting its
 * server running, and saying where to reach it. Everything backend-specific —
 * config file format, launch flags, discovery layout — stays inside.
 */

export type DiscoveredModel = {
  /** Stable id shown in opencode's picker and used as the API model id. */
  id: string
  name: string
  /** Real context window this model loads with. opencode compacts against it. */
  context: number
  output: number
  /** Sampling to send per request, so opencode's own defaults never override. */
  sampling: Record<string, number>
}

export interface Backend {
  readonly id: string
  readonly name: string

  /** Never throws. "unconfigured" is a normal answer on a fresh install. */
  status(): Promise<ProviderStatus>

  /** Empty when unconfigured or when the directory holds nothing usable. */
  models(): Promise<DiscoveredModel[]>

  /** Idempotent: starts the server only if it is not already answering. */
  start(): Promise<ProviderStatus>

  /** Stops a server we started. Never touches one we did not spawn. */
  stop(): Promise<boolean>

  /** Whether the user wants it started without being asked. */
  autostart(): Promise<boolean>

  baseURL(): string
  apiKey(): string | undefined
}
