import type { LoadEvent, LoadedModel, ProviderStatus } from "../shared/types.ts"

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

  /** Short, for the panel — everything there is local, so no prefix. */
  readonly name: string

  /**
   * How it appears in opencode's provider list, where it sits beside cloud
   * providers and several engines should group together.
   */
  readonly providerName: string

  /** Never throws. "unconfigured" is a normal answer on a fresh install. */
  status(): Promise<ProviderStatus>

  /** Empty when unconfigured or when the directory holds nothing usable. */
  models(): Promise<DiscoveredModel[]>

  /** Idempotent: starts the server only if it is not already answering. */
  start(): Promise<ProviderStatus>

  /**
   * Make `id` the served model, if the backend can only serve one at a time.
   *
   * Optional, and most backends should not implement it: llama.cpp's router
   * holds every model in models.ini at once, so naming one in a request is all
   * the selection it needs. TabbyAPI is the exception — one model per process —
   * so exl3 relaunches against that model's YAML. Called from chat.params,
   * which is the only place the plugin sees which model a request is for.
   *
   * Idempotent: a no-op when `id` is already loaded. Expensive otherwise (a
   * full stop and reload), so callers should not treat it as free.
   */
  ensure?(id: string): Promise<ProviderStatus>

  /** Stops a server we started. Never touches one we did not spawn. */
  stop(): Promise<boolean>

  /** What the server currently holds, if anything. Undefined when it is down. */
  loaded(): Promise<LoadedModel | undefined>

  /**
   * Live load progress. Polling cannot see this: a model streaming into VRAM
   * takes tens of seconds and the REST listing only flips from unloaded to
   * loaded at the end. Returns an unsubscribe.
   */
  watch(onEvent: (event: LoadEvent) => void): () => void

  baseURL(): string
  apiKey(): string | undefined
}
