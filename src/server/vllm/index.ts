import fs from "fs/promises"
import path from "path"
import os from "os"
import net from "net"
import { spawn } from "child_process"
import { stateDir, collapseHome } from "../../shared/paths.ts"
import type { LoadEvent, LoadedModel, ProviderStatus } from "../../shared/types.ts"
import type { Backend, DiscoveredModel } from "../backend.ts"
import * as Server from "./server-ini.ts"
import * as ModelsDir from "./models-dir.ts"

/**
 * The vLLM backend.
 *
 * Shaped like exl3's, for the same reason: one model per process, chosen at
 * startup, held for the life of the process. The choice lives in the vLLM
 * config YAML that this backend launches with — model, parallelism, cache
 * dtype, speculative config — all fixed when the engine is constructed, so
 * changing any of them means switching the YAML and restarting.
 *
 * Where it differs from exl3, and why the code does too:
 *
 *  - vLLM has `served-model-name`, a real alias. TabbyAPI names a model after
 *    its checkpoint folder, which is why exl3 needs a directory symlink to keep
 *    two configs over one checkpoint distinguishable. Here the YAML says its
 *    own name, so adoption can be decided on that alone.
 *  - /v1/models lists only what is SERVED, so data[0] is the loaded model.
 *    (exl3 cannot use the plural endpoint: TabbyAPI enumerates every checkpoint
 *    under model_dir, loaded or not, so data[0] is whichever sorts first.)
 *  - Startup is much slower than a weight copy. vLLM profiles the KV cache and
 *    captures CUDA graphs after the shards land, and on a two-GPU tensor-
 *    parallel setup that phase alone can outlast the load itself — hence the
 *    long START_TIMEOUT and a watcher that names the phase rather than leaving
 *    the panel on "loading weights" for minutes after the weights are in.
 */

const STATE = stateDir(Server.BACKEND)
const PID_FILE = path.join(STATE, "server.pid")
/**
 * Which YAML the running server was launched with.
 *
 * On disk rather than in memory because the panel process restarts far more
 * often than the server does. vLLM can answer "what model" via
 * served-model-name, but not "which of my configs" — two files may name one
 * checkpoint and differ in parallelism or cache dtype, and adopting on the
 * served name alone would silently keep the wrong one.
 */
const CONFIG_FILE = path.join(STATE, "server.yaml-path")
const LOG_FILE = path.join(STATE, "server.log")

const PROBE_TIMEOUT = 1_500
/**
 * Generous on purpose. A ~22 GiB NVFP4 checkpoint over tensor-parallel 2 has to
 * land on both cards, then vLLM profiles peak activation memory to size the KV
 * cache, then captures CUDA graphs for every batch shape. The last two phases
 * are compute, not I/O, and do not get faster with a warm page cache. Giving up
 * early reads as "no server here" downstream, which is worse than waiting.
 */
const START_TIMEOUT = 600_000
const POLL_INTERVAL = 1_000

async function executable(file: string): Promise<boolean> {
  const stat = await fs.stat(file).catch(() => undefined)
  return !!stat?.isFile()
}

/**
 * Servable-and-which, from /v1/models.
 *
 * vLLM serves one model per process and lists exactly what it is serving, so
 * data[0] is the answer rather than a guess. The entry also carries
 * max_model_len, which is the window the engine actually built — worth more
 * than the YAML's request, because vLLM silently clamps it when the KV cache
 * cannot cover what was asked for.
 */
type ModelInfo = { id: string; params: Record<string, unknown> }

async function modelInfo(origin: string, timeout: number, apiKey?: string): Promise<ModelInfo | undefined> {
  try {
    const res = await fetch(`${origin}/v1/models`, {
      signal: AbortSignal.timeout(timeout),
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
    })
    if (!res.ok) return undefined
    const body: any = await res.json()
    const first = Array.isArray(body?.data) ? body.data[0] : undefined
    if (typeof first?.id !== "string") return undefined
    return { id: first.id, params: first }
  } catch {
    return undefined
  }
}

async function servedModel(origin: string, timeout: number, apiKey?: string): Promise<string | undefined> {
  return (await modelInfo(origin, timeout, apiKey))?.id
}

async function ready(origin: string, timeout: number, apiKey?: string): Promise<boolean> {
  return (await servedModel(origin, timeout, apiKey)) !== undefined
}

/**
 * The vLLM process, found by scanning /proc.
 *
 * Needed for the same reason as exl3's: the pid file only covers servers we
 * launched. The binary is a console script whose process name is just "python",
 * so the script path plus the `serve` subcommand is the discriminator.
 */
async function pidForVllm(bin: string): Promise<number | undefined> {
  if (!bin) return undefined
  const entries = await fs.readdir("/proc").catch(() => [] as string[])
  for (const entry of entries) {
    const pid = Number.parseInt(entry, 10)
    if (!Number.isFinite(pid) || pid <= 0) continue
    const raw = await fs.readFile(`/proc/${pid}/cmdline`, "utf8").catch(() => "")
    if (raw.includes(bin) && raw.includes("serve")) return pid
  }
  return undefined
}

/**
 * Resolve once nothing is listening on the port, or when the deadline passes.
 *
 * Separate from terminate() because they answer different questions: terminate
 * waits for the PARENT to exit, this waits for the port and the tensor-parallel
 * workers it spawned. vLLM's TP workers are separate processes holding their
 * share of VRAM, and a launch that only waits for the parent lands on top of
 * them — two engines bidding for the same cards, neither able to allocate.
 */
async function released(host: string, port: number, timeout: number): Promise<boolean> {
  const deadline = Date.now() + timeout
  while (Date.now() < deadline) {
    const listening = await new Promise<boolean>((resolve) => {
      const socket = net.connect({ host, port })
      const done = (value: boolean) => {
        socket.destroy()
        resolve(value)
      }
      socket.once("connect", () => done(true))
      socket.once("error", () => done(false))
      socket.setTimeout(500, () => done(false))
    })
    if (!listening) return true
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  return false
}

/** SIGTERM, wait out the GPU unload, then SIGKILL. */
async function terminate(pid: number): Promise<boolean> {
  try {
    process.kill(pid, "SIGTERM")
  } catch {
    return false
  }
  for (let i = 0; i < 60; i++) {
    try {
      process.kill(pid, 0)
    } catch {
      return true
    }
    await new Promise((resolve) => setTimeout(resolve, 250))
  }
  try {
    process.kill(pid, "SIGKILL")
  } catch {
    // already gone
  }
  return true
}

export function create(): Backend {
  let settings: Server.ServerSettings | undefined
  let starting: Promise<ProviderStatus> | undefined
  /** The YAML the running server was launched with. See CONFIG_FILE above. */
  let launchedWith: string | undefined
  /**
   * Last answer from a live server, and when. vLLM keeps answering /v1/models
   * under load far better than TabbyAPI does, but a saturated engine can still
   * miss a 1.5s probe, and reporting "stopped" mid-session is the worst
   * possible lie for a panel to tell.
   */
  let lastSeen: { model: LoadedModel; at: number } | undefined
  /**
   * The quantization kernel vLLM announced at load, scraped from the log.
   *
   * Surfaced in the panel because on consumer Blackwell it is THE thing worth
   * knowing and nothing else reports it: a ModelOpt W4A16 checkpoint is routed
   * through Marlin and dequantised to the activation dtype, so the FP4 tensor
   * cores never run, while a W4A4 one takes the native CUTLASS path. Both load
   * fine and both say "NVFP4" everywhere else.
   */
  let kernelNote: string | undefined

  // Re-read every time rather than caching, so editing the file takes effect
  // without restarting the process the panel polls from.
  const config = async () => (settings = await Server.load())
  const host = () => (settings?.host === "0.0.0.0" ? "127.0.0.1" : (settings?.host ?? "127.0.0.1"))
  const port = () => settings?.port ?? 8000

  const isRemote = () => !!settings?.remote

  const origin = () => {
    if (settings?.remote) return `http://${settings.remote}`
    return `http://${host()}:${port()}`
  }

  const baseURL = () => `${origin()}/v1`

  /** Same contract as the other backends': the address another machine uses. */
  const lanAddress = () => {
    if (settings?.remote) return undefined
    if (settings?.host !== "0.0.0.0") return undefined
    const ipv4 = Object.values(os.networkInterfaces())
      .flat()
      .find((nic) => nic && nic.family === "IPv4" && !nic.internal)?.address
    const name = os.hostname()
    if (name && ipv4) return `${name}.local:${port()} (${ipv4})`
    return ipv4 ? `${ipv4}:${port()}` : undefined
  }

  /**
   * Is the server process we launched still there?
   *
   * Distinct from ready(): this asks whether the process exists, not whether it
   * can answer. The two diverge for the whole of CUDA graph capture, which is
   * exactly when the panel must not claim it is stopped.
   */
  async function serverAlive(cfg: Server.ServerSettings): Promise<boolean> {
    const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
    const pid = Number.parseInt(raw.trim(), 10)
    if (Number.isFinite(pid) && pid > 0) {
      const cmdline = await fs.readFile(`/proc/${pid}/cmdline`, "utf8").catch(() => "")
      if (cmdline.includes("serve")) return true
    }
    // a server started outside this process still counts as running
    return (await pidForVllm(cfg.bin)) !== undefined
  }

  /** Everything launch needs, or the first thing missing. */
  async function unconfigured(cfg: Server.ServerSettings): Promise<ProviderStatus | undefined> {
    if (!cfg.bin || !(await executable(cfg.bin))) {
      return {
        state: "unconfigured",
        missing: "binary",
        message: "vllm executable not set",
        hint: `set bin in ${collapseHome(Server.FILE)}`,
      }
    }
    // `config` is the single-model override; models/ is the normal path. Either
    // satisfies this, but at least one must produce a YAML to launch.
    if (cfg.config) {
      if (!(await executable(cfg.config))) {
        return {
          state: "unconfigured",
          missing: "models-dir",
          message: "config points at nothing",
          hint: `fix or blank config in ${collapseHome(Server.FILE)}`,
        }
      }
      return undefined
    }
    if ((await ModelsDir.scan()).length === 0) {
      return {
        state: "unconfigured",
        missing: "models-dir",
        message: "no model YAMLs found",
        hint: `add a vLLM config YAML to ${collapseHome(ModelsDir.DIR)}`,
      }
    }
    return undefined
  }

  /**
   * Which YAML to launch. The `config` override wins; otherwise the selected
   * model's file, falling back to the last one served so [start] brings back
   * what you had rather than whatever sorts first.
   */
  async function yamlFor(cfg: Server.ServerSettings, id?: string): Promise<string | undefined> {
    if (cfg.config) return cfg.config
    if (id) {
      const file = await ModelsDir.fileFor(id)
      if (file) return file
    }
    const declared = await ModelsDir.scan()
    const remembered = (await fs.readFile(CONFIG_FILE, "utf8").catch(() => "")).trim()
    if (remembered && declared.some((model) => model.file === remembered)) return remembered
    return declared[0]?.file
  }

  async function status(): Promise<ProviderStatus> {
    const cfg = await config()
    if (cfg.remote) {
      if (await ready(origin(), PROBE_TIMEOUT, cfg.apiKey)) {
        return { state: "running", endpoint: baseURL() }
      }
      // "stopped", not "failed" — same reasoning as exl3's remote: the far
      // machine starts vLLM when a session wants it, so "not answering right
      // now" is its normal resting state, and reporting failed would make
      // register() drop the provider exactly when you want to see what
      // starting it would offer.
      return { state: "stopped" }
    }
    const missing = await unconfigured(cfg)
    if (missing) return missing
    if (await ready(origin(), PROBE_TIMEOUT, cfg.apiKey)) {
      return { state: "running", endpoint: baseURL(), lan: lanAddress() }
    }
    if (await serverAlive(cfg)) {
      return { state: "running", endpoint: baseURL(), lan: lanAddress() }
    }
    return { state: "stopped" }
  }

  /** Shared shaping so every path advertises a model the same way. */
  function describe(id: string, context: number, remote: boolean): DiscoveredModel {
    return {
      id,
      // Strip a leading HF org and any quant marker, keep everything after it:
      // on this backend two entries usually differ only in the tail (context,
      // vision, parallelism), and collapsing to the base name makes them
      // indistinguishable in the picker.
      name: id.replace(/^[^/]+\//, "").replace(/-+/g, " ").trim(),
      context,
      output: Math.min(32_768, Math.max(4_096, Math.floor(context / 2))),
      // a remote applies its own sampling; overriding from here fights it
      sampling: remote ? {} : settings!.sampling,
    }
  }


  /**
   * A remote vLLM cannot be enumerated or switched on its own: it serves ONE
   * model per process, so /v1/models reports only that one and there is no
   * discovery endpoint to ask. (llama.cpp needs nothing like this — its
   * llama-server scans models-dir itself and swaps on demand, so its /models
   * already lists everything. The difference is the server, not the backend.)
   *
   * `control` in server.ini points at a small daemon on the far machine that
   * closes exactly that gap: it lists the YAMLs, and starts/stops the engine.
   * Inference still goes straight to the vLLM port; this is not a proxy.
   */
  const controlURL = (cfg: Server.ServerSettings) => (cfg.control ? `http://${cfg.control}` : undefined)

  async function controlFetch(
    cfg: Server.ServerSettings,
    route: string,
    init?: { method?: string; body?: unknown; timeout?: number },
  ): Promise<any | undefined> {
    const base = controlURL(cfg)
    if (!base) return undefined
    const res = await fetch(`${base}${route}`, {
      method: init?.method ?? "GET",
      signal: AbortSignal.timeout(init?.timeout ?? PROBE_TIMEOUT * 4),
      headers: init?.body ? { "Content-Type": "application/json" } : undefined,
      body: init?.body ? JSON.stringify(init.body) : undefined,
    }).catch(() => undefined)
    if (!res?.ok) return undefined
    return res.json().catch(() => undefined)
  }

  /**
   * Advertised ids are served-model-names; the control daemon keys its entries
   * by YAML basename. Map back before asking it to start something.
   */
  async function controlEntryFor(cfg: Server.ServerSettings, id: string): Promise<any | undefined> {
    const entries: any[] = (await controlFetch(cfg, "/models"))?.data ?? []
    return entries.find((entry) => entry?.served === id) ?? entries.find((entry) => entry?.id === id)
  }

  /** Every model the far machine has, not just the one it happens to be serving. */
  async function remoteModels(cfg: Server.ServerSettings): Promise<DiscoveredModel[]> {
    const body = await controlFetch(cfg, "/models")
    const entries: any[] = Array.isArray(body?.data) ? body.data : []
    return entries.flatMap((entry) => {
      const id = typeof entry?.id === "string" ? entry.id : undefined
      if (!id) return []
      const context = Number.isFinite(entry?.context) && entry.context > 0 ? entry.context : cfg.context
      // Advertise served-model-name, NOT the filename. The id we hand opencode
      // goes straight into the request's "model" field, and vLLM answers only
      // to served-model-name — a YAML whose basename differs from it produced
      // "The model X does not exist" for every request.
      const served = typeof entry?.served === "string" && entry.served ? entry.served : id
      return [describe(served, context, true)]
    })
  }

  async function models(): Promise<DiscoveredModel[]> {
    const cfg = await config()
    if (!cfg.remote) {
      const missing = await unconfigured(cfg)
      if (missing) return []
    }

    // Remote with a control daemon: it can enumerate, so show everything.
    // Without one we can only report what /v1/models admits to serving.
    if (cfg.remote && cfg.control) {
      const listed = await remoteModels(cfg)
      if (listed.length > 0) return listed
    }

    // Local, models/ in use: list every YAML. Only one can be SERVED at a time,
    // but all are selectable — picking one relaunches (see ensure). Each file's
    // own max-model-len is the truthful window, so a model built with a smaller
    // KV cache does not get opencode compacting against the wrong number.
    if (!cfg.remote && !cfg.config) {
      const declared = await ModelsDir.scan()
      if (declared.length > 0) {
        // served-model-name, not the filename — the advertised id becomes the
        // request's "model" field and vLLM answers only to the served name.
        return declared.map((model) =>
          describe(model.served || model.id, model.context ?? cfg.context, false),
        )
      }
    }

    const served = await servedModelFrom(origin(), PROBE_TIMEOUT, cfg.apiKey)
    const id = served ?? (cfg.config ? (await ModelsDir.scan()).find((m) => m.file === cfg.config)?.served : undefined)
    if (!id) return []
    return [describe(id, cfg.context, !!cfg.remote)]
  }

  async function launch(cfg: Server.ServerSettings, yaml: string): Promise<ProviderStatus> {
    const declared = (await ModelsDir.scan()).find((model) => model.file === yaml)
    // The positional model comes from the YAML itself — see Server.argv for why
    // it is passed both ways. Without it there is nothing to serve.
    const model = declared?.model ?? (await modelFromYaml(yaml))
    if (!model) {
      return { state: "failed", message: "config declares no model:", hint: collapseHome(yaml) }
    }

    await fs.mkdir(STATE, { recursive: true }).catch(() => {})
    const log = await fs.open(LOG_FILE, "a").catch(() => undefined)
    launchedWith = yaml
    lastSeen = undefined
    kernelNote = undefined
    await fs.writeFile(CONFIG_FILE, `${yaml}\n`).catch(() => {})
    try {
      const child = spawn(cfg.bin, Server.argv(cfg, model, yaml), {
        detached: true,
        stdio: ["ignore", log?.fd ?? "ignore", log?.fd ?? "ignore"],
      })
      child.on("error", () => {})
      child.unref()
      if (child.pid) await fs.writeFile(PID_FILE, `${child.pid}\n`).catch(() => {})
    } catch (error) {
      return { state: "failed", message: "could not start", hint: String(error).slice(0, 60) }
    } finally {
      await log?.close().catch(() => {})
    }

    const deadline = Date.now() + START_TIMEOUT
    while (Date.now() < deadline) {
      if (await ready(origin(), PROBE_TIMEOUT, settings?.apiKey)) {
        return { state: "running", endpoint: baseURL(), lan: lanAddress() }
      }
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL))
    }
    return { state: "failed", message: "server did not start", hint: collapseHome(LOG_FILE) }
  }

  /** Fallback for a `config` override, which is not in models/ and so not scanned. */
  async function modelFromYaml(file: string): Promise<string | undefined> {
    const raw = await fs.readFile(file, "utf8").catch(() => undefined)
    const match = raw?.match(/^\s*model:\s*(.+?)\s*$/m)
    return match?.[1]?.replace(/^["']|["']$/g, "").trim() || undefined
  }

  async function stop(): Promise<boolean> {
    const cfg = await config()
    // Not ours to stop — unless the far machine gave us a control daemon.
    if (isRemote()) {
      if (!cfg.control) return false
      const body = await controlFetch(cfg, "/stop", { method: "POST", timeout: START_TIMEOUT })
      return body?.stopped === true
    }
    const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
    const recorded = Number.parseInt(raw.trim(), 10)
    let pid: number | undefined
    if (Number.isFinite(recorded) && recorded > 0) {
      const cmdline = await fs.readFile(`/proc/${recorded}/cmdline`, "utf8").catch(() => "")
      // never signal a pid that has been recycled into something else
      if (cmdline.includes("serve")) pid = recorded
      else await fs.rm(PID_FILE, { force: true }).catch(() => {})
    }
    if (pid === undefined) pid = await pidForVllm(cfg.bin)
    if (pid === undefined) return false
    const killed = await terminate(pid)
    await fs.rm(PID_FILE, { force: true }).catch(() => {})
    // Wait for the PORT, not just the parent: TP workers outlive it and go on
    // holding both the socket and their share of VRAM.
    await released(host(), port(), 60_000)
    launchedWith = undefined
    lastSeen = undefined
    kernelNote = undefined
    // CONFIG_FILE is deliberately NOT removed. It records the last model
    // served, so [start] brings back what you had.
    return killed
  }

  async function loaded(): Promise<LoadedModel | undefined> {
    const cfg = await config()
    const active = launchedWith ?? cfg.config
    const declared = active
      ? (await ModelsDir.scan()).find((model) => model.file === active)
      : undefined

    const info = await modelInfo(origin(), PROBE_TIMEOUT, cfg.apiKey)
    if (info) {
      const p = info.params
      const str = (key: string) => (p[key] === undefined || p[key] === null ? undefined : String(p[key]))
      const args: Record<string, string> = {
        engine: "vllm",
        // the window the engine BUILT, which vLLM clamps down when the KV cache
        // cannot cover what the YAML asked for
        context: str("max_model_len") ?? String(declared?.context ?? cfg.context),
        // the whole point of this backend on consumer Blackwell; see kernelNote
        kernel: kernelNote ?? "",
        config: collapseHome(active || ""),
      }
      for (const key of Object.keys(args)) if (!args[key]) delete args[key]
      const model: LoadedModel = { id: info.id, args: cfg.remote ? { host: cfg.remote } : args }
      lastSeen = { model, at: Date.now() }
      return model
    }

    if (cfg.remote) return undefined

    if (lastSeen && (await serverAlive(cfg))) return lastSeen.model

    // not answering yet — is it still coming up, or simply not running?
    const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
    const pid = Number.parseInt(raw.trim(), 10)
    if (!Number.isFinite(pid) || pid <= 0) return undefined
    try {
      process.kill(pid, 0)
    } catch {
      return undefined
    }
    const pending: Record<string, string> = {
      engine: "vllm",
      context: String(declared?.context ?? cfg.context),
      config: collapseHome(active || ""),
    }
    return {
      id: declared?.served ?? "model",
      args: pending,
      loading: true,
      stage: "starting engine",
    }
  }

  /**
   * Live load progress, read from the server log.
   *
   * vLLM narrates a startup in distinct phases, and they are worth telling
   * apart: the shard loader gives a real fraction, but everything after it —
   * KV-cache profiling and CUDA graph capture — reports no numbers and can run
   * longer than the load did. A watcher that only knew about shards would sit
   * at 100% for minutes, which reads as a hang.
   */
  function watch(onEvent: (event: LoadEvent) => void): () => void {
    let stopped = false
    let offset = -1 // -1 = start from the end, so old loads are not replayed
    let model = ""

    void (async () => {
      const cfg = await config().catch(() => undefined)
      if (!cfg || model) return
      if (cfg.remote) {
        const id = await servedModel(origin(), PROBE_TIMEOUT * 4, cfg.apiKey)
        if (id && !stopped) onEvent({ model: id, loading: false, loaded: true } as LoadEvent)
        return
      }
      const active = launchedWith ?? cfg.config
      model = (await ModelsDir.scan()).find((m) => m.file === active)?.served ?? ""
    })()

    const emit = (event: Partial<LoadEvent> & { loading: boolean }) =>
      onEvent({ model, ...event } as LoadEvent)

    const scanLine = (line: string) => {
      // "Starting to load model nvidia/Qwen3.6-35B-A3B-NVFP4..."
      const naming = line.match(/Starting to load model\s+(\S+?)\.{0,3}\s*$/)
      if (naming) {
        model = naming[1] ?? model
        return emit({ loading: true, stage: "loading weights" })
      }
      // The FP4 question this backend exists to answer. Captured for the panel
      // rather than only emitted, because it matters after the load too.
      if (/does not have native support for FP4/i.test(line) || /Marlin/i.test(line)) {
        kernelNote = "marlin (FP4 dequantised, no native FP4)"
      } else if (/cutlass/i.test(line) && /fp4|nvfp4/i.test(line)) {
        kernelNote = "cutlass nvfp4 (native FP4)"
      } else if (/flashinfer/i.test(line) && /fp4|nvfp4|moe/i.test(line)) {
        kernelNote = "flashinfer fp4 (native FP4)"
      }
      // "Loading safetensors checkpoint shards:  50% Completed | 2/4 [...]"
      const shards = line.match(/checkpoint shards:\s*\d+%[^|]*\|\s*(\d+)\/(\d+)/)
      if (shards) {
        const done = Number(shards[1])
        const total = Number(shards[2])
        if (total > 0) {
          return emit({ loading: true, progress: Math.min(1, done / total), stage: "loading weights" })
        }
      }
      // Post-load phases. No fraction is reported for either, so they carry the
      // stage only — but naming them is the difference between "still working"
      // and an apparent hang at 100%.
      if (/Memory profiling|profile.*kv cache|Determining available memory/i.test(line)) {
        return emit({ loading: true, stage: "profiling KV cache" })
      }
      if (/Capturing CUDA graph|graph capturing/i.test(line)) {
        return emit({ loading: true, stage: "capturing CUDA graphs" })
      }
      if (line.includes("Application startup complete")) {
        return emit({ loading: false, loaded: true })
      }
      // Terminal load failures. The two that actually happen on a 2x16 GiB box
      // are a KV cache that cannot cover max-model-len, and plain OOM.
      if (
        /No available memory for the cache blocks/i.test(line) ||
        /max seq len.*larger than the maximum number of tokens/i.test(line) ||
        /torch\.OutOfMemoryError|CUDA out of memory/i.test(line) ||
        /Engine core initialization failed/i.test(line)
      ) {
        return emit({ loading: false, failed: true })
      }
    }

    const tick = async () => {
      let carry = ""
      while (!stopped) {
        try {
          const stat = await fs.stat(LOG_FILE).catch(() => undefined)
          if (stat) {
            if (offset < 0 || stat.size < offset) {
              offset = stat.size
              carry = ""
            } else if (stat.size > offset) {
              const handle = await fs.open(LOG_FILE, "r")
              try {
                const length = stat.size - offset
                const buffer = Buffer.alloc(length)
                await handle.read(buffer, 0, length, offset)
                offset = stat.size
                const lines = (carry + buffer.toString("utf8")).split("\n")
                carry = lines.pop() ?? ""
                for (const line of lines) scanLine(line)
              } finally {
                await handle.close().catch(() => {})
              }
            }
          }
        } catch {
          // an unreadable log is not worth tearing the watcher down for
        }
        await new Promise((resolve) => setTimeout(resolve, 500))
      }
    }

    void tick()
    return () => {
      stopped = true
    }
  }

  /** Idempotent and single-flight: concurrent callers share one attempt. */
  async function start(id?: string): Promise<ProviderStatus> {
    if (starting) return starting
    starting = (async () => {
      const cfg = await config()
      if (cfg.remote) {
        if (!cfg.control) return status()
        const entry = id ? await controlEntryFor(cfg, id) : ((await controlFetch(cfg, "/models"))?.data ?? [])[0]
        if (entry?.id) await controlFetch(cfg, "/start", { method: "POST", body: { id: entry.id } })
        return status()
      }
      const missing = await unconfigured(cfg)
      if (missing) return missing
      const yaml = await yamlFor(cfg, id)
      if (!yaml) {
        return {
          state: "unconfigured",
          missing: "models-dir",
          message: "no model YAML to launch",
          hint: collapseHome(ModelsDir.DIR),
        } as ProviderStatus
      }
      if (await ready(origin(), PROBE_TIMEOUT, settings?.apiKey)) {
        return { state: "running", endpoint: baseURL(), lan: lanAddress() } as ProviderStatus
      }
      return launch(cfg, yaml)
    })()
    try {
      return await starting
    } finally {
      starting = undefined
    }
  }

  /**
   * Make `id` the served model. vLLM holds one per process — parallelism, cache
   * dtype and speculative config are all fixed at engine construction — so
   * switching is stop-then-launch.
   *
   * Returns early whenever the wanted model is already up, because the reload
   * is expensive here in a way it is not for a mmap'd GGUF: the shards have to
   * stream again AND the engine re-profiles and re-captures CUDA graphs.
   */
  async function ensure(id: string): Promise<ProviderStatus> {
    const cfg = await config()
    // A remote decides its own model — unless it exposes a control daemon, in
    // which case selecting a model here is meant to switch it there.
    if (cfg.remote) {
      if (!cfg.control) return status()
      const entry = await controlEntryFor(cfg, id)
      // Unknown id: leave the far machine alone rather than unloading a working
      // model for one it does not have.
      if (!entry) return status()
      const current = await servedModel(origin(), PROBE_TIMEOUT, cfg.apiKey)
      if (current && entry.served && current === entry.served) return status()
      await controlFetch(cfg, "/start", { method: "POST", body: { id: entry.id } })
      await ready(origin(), START_TIMEOUT, cfg.apiKey)
      return status()
    }
    // The single-model override means there is nothing to choose between.
    if (cfg.config) return start()

    const declaredAll = await ModelsDir.scan()
    const wanted =
      declaredAll.find((model) => model.served === id) ?? declaredAll.find((model) => model.id === id)
    // Unknown id: leave whatever is running alone rather than tearing down a
    // working server for a model this machine cannot serve.
    if (!wanted) return status()

    if (!launchedWith) {
      const recorded = (await fs.readFile(CONFIG_FILE, "utf8").catch(() => "")).trim()
      if (recorded && (await serverAlive(cfg))) launchedWith = recorded
    }
    const up = await ready(origin(), PROBE_TIMEOUT, cfg.apiKey)
    if (up && launchedWith === wanted.file) {
      return { state: "running", endpoint: baseURL(), lan: lanAddress() }
    }
    // Up from a process whose choice we never recorded — this one, restarted,
    // or one started by hand. Ask the server what it holds before paying for a
    // reload.
    //
    // The uniqueness guard is the same one exl3 needs, and for the same reason:
    // a served-name match is necessary but not sufficient when two files can
    // announce the same name, since adopting on it alone would silently keep
    // the wrong config. The difference is that here you can simply FIX it —
    // vLLM's served-model-name is a free-form alias, so giving each YAML a
    // distinct one makes this branch reliable instead of merely safe.
    if (up && !launchedWith) {
      const served = await servedModel(origin(), PROBE_TIMEOUT, cfg.apiKey)
      const only = (await ModelsDir.scan()).filter((m) => m.served === wanted.served)
      if (served === wanted.served && only.length === 1) {
        launchedWith = wanted.file
        return { state: "running", endpoint: baseURL(), lan: lanAddress() }
      }
    }
    if (up) {
      await stop()
      await released(host(), port(), START_TIMEOUT)
    }
    return start(id)
  }

  const servedModelFrom = servedModel

  return {
    id: Server.BACKEND,
    name: "vLLM",
    providerName: "Localhost-vLLM",
    status,
    models,
    start,
    ensure,
    stop,
    loaded,
    watch,
    baseURL,
    apiKey: () => settings?.apiKey || undefined,
  }
}
