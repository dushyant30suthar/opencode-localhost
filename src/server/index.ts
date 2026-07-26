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
    name: backend.providerName,
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

    // Configured is enough. The models exist on disk whether or not a server
    // happens to be up, and a picker that stays empty until you find and press
    // [start] is indistinguishable from the plugin not working at all.
    //
    // Nothing is spawned here: the server comes up on the first request, in
    // chat.params below.

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
    const backend = BACKENDS.find((item) => item.id === providerID)
    if (!backend) return

    // Bring the engine up if this is the first request against it. Sending a
    // message is a clear enough signal of intent — the objection was to
    // spawning a server at launch, not to starting one you are about to use.
    const status = await backend.status().catch(() => undefined)
    if (status && status.state === "stopped") await backend.start().catch(() => {})
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
