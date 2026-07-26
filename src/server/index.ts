import type { Backend, DiscoveredModel } from "./backend.ts"
import { create as llamacpp } from "./llamacpp/index.ts"

/**
 * The server half.
 *
 * opencode loads plugins before it reads `cfg.provider`, specifically so a
 * plugin's config() hook can add to it. That is the whole integration: we
 * discover models on disk, start the backend, and write a provider into the
 * config opencode is about to read. From there it is an ordinary provider.
 *
 * Sampling is handled in chat.params rather than baked into the server launch,
 * so changing `temp` in models.ini takes effect on the next message instead of
 * requiring a model reload — and so opencode's own per-model heuristics, which
 * know nothing about local models, never override it.
 */

const BACKENDS: Backend[] = [llamacpp()]

/** Cached per process: discovery walks the filesystem and starts a server. */
const sampling = new Map<string, Record<string, number>>()

function providerEntry(backend: Backend, models: DiscoveredModel[]) {
  return {
    name: `${backend.name} (local)`,
    api: backend.baseURL(),
    npm: "@ai-sdk/openai-compatible",
    options: {
      baseURL: backend.baseURL(),
      // the SDK requires some value; an unauthenticated server ignores it
      apiKey: backend.apiKey() ?? "local",
    },
    models: Object.fromEntries(
      models.map((model) => [
        model.id,
        {
          name: model.name,
          limit: { context: model.context, output: model.output },
          cost: { input: 0, output: 0 },
          options: {},
        },
      ]),
    ),
  }
}

async function register(input: any) {
  for (const backend of BACKENDS) {
    const status = await backend.status().catch(() => undefined)
    // unconfigured or missing binary: contribute nothing. The panel explains why.
    if (!status || status.state === "unconfigured" || status.state === "failed") continue

    // Not up yet: start it in the background and register nothing this run.
    //
    // Awaiting here would stall opencode's startup for as long as the server
    // takes to answer — once on a fresh setup, and on *every* launch if the
    // server is failing to start. opencode reads its config once per instance
    // and exposes no way to re-read it, so the provider genuinely cannot appear
    // until the next launch; the panel says so rather than leaving it silent.
    if (status.state !== "running") {
      void backend.start().catch(() => {})
      continue
    }

    const models = await backend.models().catch(() => [])
    if (models.length === 0) continue

    for (const model of models) {
      if (Object.keys(model.sampling).length > 0) {
        sampling.set(`${backend.id}/${model.id}`, model.sampling)
      }
    }

    input.provider = input.provider ?? {}
    input.provider[backend.id] = providerEntry(backend, models)
  }
}

const server = async () => ({
  config: async (input: any) => {
    await register(input).catch(() => {})
  },

  "chat.params": async (input: any, output: any) => {
    const providerID = input?.model?.providerID
    if (!BACKENDS.some((backend) => backend.id === providerID)) return
    const values = sampling.get(`${providerID}/${input?.model?.id}`)
    if (!values) return
    // opencode names three of these directly and passes the rest through options
    if ("temperature" in values) output.temperature = values["temperature"]
    if ("top_p" in values) output.topP = values["top_p"]
    if ("top_k" in values) output.topK = values["top_k"]
    const extra = { ...values }
    delete extra["temperature"]
    delete extra["top_p"]
    delete extra["top_k"]
    if (Object.keys(extra).length > 0) {
      output.options = { ...(output.options ?? {}), ...extra }
    }
  },
})

export default { id: "opencode-localhost", server }
