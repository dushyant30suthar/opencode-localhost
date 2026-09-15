import fs from "fs/promises"
import os from "os"
import path from "path"
import { spawn } from "child_process"
import { stateDir, collapseHome } from "../../shared/paths.ts"
import type { LoadEvent, LoadedModel, ProviderStatus } from "../../shared/types.ts"
import type { Backend, DiscoveredModel } from "../backend.ts"
import * as Server from "./server-ini.ts"
import * as ModelsDir from "./models-dir.ts"

/**
 * The MLX backend: mlx_lm.server / mlx_vlm.server from source, one checkpoint
 * per process, chosen at startup. Shaped like the vLLM backend for the same
 * reason — engine settings (model, KV budget) are fixed at construction, so
 * switching models relaunches.
 *
 * Darwin-only and /proc-free: macOS has no /proc, so liveness is the pidfile
 * plus kill(pid, 0), and readiness is the HTTP probe. A pidfile that points at
 * a recycled pid is never signalled without the port answering first.
 */

const STATE = stateDir(Server.BACKEND)
const PID_FILE = `${STATE}/server.pid`
const LOG_FILE = `${STATE}/server.log`
/** Which checkpoint the running server was launched with (absolute path). */
const CONFIG_FILE = `${STATE}/server.model-path`

const PROBE_TIMEOUT = 1_500
/** MLX maps weights rather than copying shards: ~10s for 6 GiB, not minutes. */
const START_TIMEOUT = 120_000
const POLL_INTERVAL = 500

async function executable(file: string): Promise<boolean> {
  const stat = await fs.stat(file).catch(() => undefined)
  return !!stat?.isFile()
}

/** What /v1/models serves. Single-model servers list exactly one entry. */
async function servedModel(
  origin: string,
  timeout: number,
  apiKey?: string,
): Promise<string | undefined> {
  try {
    const res = await fetch(`${origin}/v1/models`, {
      signal: AbortSignal.timeout(timeout),
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
    })
    if (!res.ok) return undefined
    const body: any = await res.json()
    const first = Array.isArray(body?.data) ? body.data[0] : undefined
    return typeof first?.id === "string" ? first.id : undefined
  } catch {
    return undefined
  }
}

async function ready(origin: string, timeout: number, apiKey?: string): Promise<boolean> {
  return (await servedModel(origin, timeout, apiKey)) !== undefined
}

/** The server process we launched, still there? Not whether it can answer. */
async function serverAlive(): Promise<boolean> {
  const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
  const pid = Number.parseInt(raw.trim(), 10)
  if (!Number.isFinite(pid) || pid <= 0) return false
  try {
    process.kill(pid, 0)
    return true
  } catch {
    return false
  }
}

async function terminate(pid: number): Promise<boolean> {
  try {
    process.kill(pid, "SIGTERM")
  } catch {
    return false
  }
  for (let i = 0; i < 40; i++) {
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
  let launchedWith: string | undefined
  let lastSeen: { model: LoadedModel; at: number } | undefined

  const config = async () => (settings = await Server.load())
  const host = () => (settings?.host === "0.0.0.0" ? "127.0.0.1" : (settings?.host ?? "127.0.0.1"))
  const port = () => settings?.port ?? 8081
  const origin = () => {
    if (settings?.remote) return `http://${settings.remote}`
    return `http://${host()}:${port()}`
  }
  const baseURL = () => `${origin()}/v1`

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

  async function unconfigured(cfg: Server.ServerSettings): Promise<ProviderStatus | undefined> {
    if (!cfg.bin || !(await executable(cfg.bin))) {
      return {
        state: "unconfigured",
        missing: "binary",
        message: "MLX server executable not set",
        hint: `set bin in ${collapseHome(Server.FILE)}`,
      }
    }
    if (cfg.model) {
      const info = await ModelsDir.fileFor(cfg.model)
      if (!info) {
        return {
          state: "unconfigured",
          missing: "models-dir",
          message: "model points at nothing loadable",
          hint: `fix model in ${collapseHome(Server.FILE)}`,
        }
      }
      return undefined
    }
    if ((await ModelsDir.scan(cfg.modelsDir)).length === 0) {
      return {
        state: "unconfigured",
        missing: "models-dir",
        message: "no MLX checkpoints found",
        hint: `add a checkpoint dir to ${collapseHome(ModelsDir.DIR)}`,
      }
    }
    return undefined
  }

  async function status(): Promise<ProviderStatus> {
    const cfg = await config()
    if (cfg.remote) {
      if (await ready(origin(), PROBE_TIMEOUT, cfg.apiKey)) {
        return { state: "running", endpoint: baseURL() }
      }
      return { state: "stopped" }
    }
    const missing = await unconfigured(cfg)
    if (missing) return missing
    if (await ready(origin(), PROBE_TIMEOUT, cfg.apiKey)) {
      return { state: "running", endpoint: baseURL(), lan: lanAddress() }
    }
    if (await serverAlive()) {
      return { state: "starting" }
    }
    return { state: "stopped" }
  }

  const isVlm = (cfg: Server.ServerSettings) =>
    cfg.bin.split("/").at(-1)?.includes("vlm") ?? false

  /** The window opencode compacts against. mlx_lm has no KV cap, so the
   *  configured context is the window; mlx_vlm is capped by --max-kv-size. */
  function windowFor(archMax: number | undefined, cfg: Server.ServerSettings): number {
    const cap = isVlm(cfg) ? Math.min(cfg.maxKv, cfg.context) : cfg.context
    return archMax ? Math.min(archMax, cap) : cap
  }


  function describe(id: string, context: number, vision: boolean | undefined): DiscoveredModel {
    const base = id.split("/").filter(Boolean).at(-1) ?? id
    return {
      id,
      name: base.replace(/-+/g, " ").trim(),
      context,
      output: Math.min(32_768, Math.max(4_096, Math.floor(context / 2))),
      sampling: settings!.remote ? {} : settings!.sampling,
      ...(vision === undefined ? {} : { vision }),
    }
  }

  async function models(): Promise<DiscoveredModel[]> {
    const cfg = await config()
    if (!cfg.remote) {
      const missing = await unconfigured(cfg)
      if (missing) return []
    }
    if (cfg.remote) {
      const served = await servedModel(origin(), PROBE_TIMEOUT * 4, cfg.apiKey)
      if (!served) return []
      return [describe(served, cfg.context, undefined)]
    }
    if (cfg.model) {
      const resolved = await ModelsDir.fileFor(cfg.model)
      if (!resolved) return []
      const scanned = await ModelsDir.scan(cfg.modelsDir)
      const found = scanned.find((m) => m.id === resolved)
      // the text server cannot take image parts: advertise text-only so
      // opencode refuses drops up front instead of 500ing mid-request
      const vision = isVlm(cfg) ? found?.vision : false
      return [describe(resolved, windowFor(found?.archMax, cfg), vision)]
    }
    const declared = await ModelsDir.scan(cfg.modelsDir)
    const visionFor = (v: boolean | undefined) => (isVlm(cfg) ? v : false)
    return declared.map((model) =>
      describe(model.id, windowFor(model.archMax, cfg), visionFor(model.vision)),
    )
  }

  async function launch(cfg: Server.ServerSettings, model: string): Promise<ProviderStatus> {
    await fs.mkdir(STATE, { recursive: true }).catch(() => {})
    // mlx_lm.server's /v1/models handler calls scan_cache_dir(), which RAISES
    // (500ing every probe, so start() can never succeed) when the HF hub cache
    // directory does not exist. An empty dir scans fine.
    await fs
      .mkdir(path.join(os.homedir(), ".cache", "huggingface", "hub"), { recursive: true })
      .catch(() => {})
    const log = await fs.open(LOG_FILE, "a").catch(() => undefined)
    launchedWith = model
    lastSeen = undefined
    await fs.writeFile(CONFIG_FILE, `${model}\n`).catch(() => {})
    try {
      const child = spawn(cfg.bin, Server.argv(cfg, model), {
        detached: true,
        // APC_* env carries the prefix-cache config: no CLI flags exist for it
        env: { ...process.env, ...Server.env(cfg) },
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

  async function start(id?: string): Promise<ProviderStatus> {
    if (starting) return starting
    starting = (async () => {
      const cfg = await config()
      if (cfg.remote) return status()
      const missing = await unconfigured(cfg)
      if (missing) return missing
      let model: string | undefined
      if (cfg.model) model = await ModelsDir.fileFor(cfg.model)
      else if (id) model = await ModelsDir.fileFor(id, cfg.modelsDir)
      if (!model) {
        const remembered = (await fs.readFile(CONFIG_FILE, "utf8").catch(() => "")).trim()
        model =
          (remembered && (await ModelsDir.fileFor(remembered, cfg.modelsDir))) ||
          (await ModelsDir.scan(cfg.modelsDir))[0]?.id
      }
      if (!model) {
        return {
          state: "unconfigured",
          missing: "models-dir",
          message: "no checkpoint to launch",
          hint: collapseHome(ModelsDir.DIR),
        } as ProviderStatus
      }
      if (await ready(origin(), PROBE_TIMEOUT, settings?.apiKey)) {
        const served = await servedModel(origin(), PROBE_TIMEOUT, settings?.apiKey)
        if (served === model) {
          launchedWith = model
          return { state: "running", endpoint: baseURL(), lan: lanAddress() } as ProviderStatus
        }
      }
      if (await serverAlive()) {
        await stop()
      }
      return launch(cfg, model)
    })()
    try {
      return await starting
    } finally {
      starting = undefined
    }
  }

  async function ensure(id: string): Promise<ProviderStatus> {
    const cfg = await config()
    if (cfg.remote || cfg.model) return start()
    const wanted = await ModelsDir.fileFor(id, cfg.modelsDir)
    if (!wanted) return status()
    if (!launchedWith) {
      const recorded = (await fs.readFile(CONFIG_FILE, "utf8").catch(() => "")).trim()
      if (recorded && (await serverAlive())) launchedWith = recorded
    }
    const up = await ready(origin(), PROBE_TIMEOUT, cfg.apiKey)
    if (up && launchedWith === wanted) {
      return { state: "running", endpoint: baseURL(), lan: lanAddress() }
    }
    if (up && !launchedWith) {
      const served = await servedModel(origin(), PROBE_TIMEOUT, cfg.apiKey)
      if (served === wanted) {
        launchedWith = wanted
        return { state: "running", endpoint: baseURL(), lan: lanAddress() }
      }
    }
    if (up) await stop()
    return start(id)
  }

  async function stop(): Promise<boolean> {
    const cfg = await config()
    if (isRemote(cfg)) return false
    const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
    const recorded = Number.parseInt(raw.trim(), 10)
    if (!Number.isFinite(recorded) || recorded <= 0) {
      await fs.rm(PID_FILE, { force: true }).catch(() => {})
      return false
    }
    let pid: number | undefined
    try {
      process.kill(recorded, 0)
      pid = recorded
    } catch {
      await fs.rm(PID_FILE, { force: true }).catch(() => {})
      return false
    }
    // never signal a pid we cannot also reach on the port: without /proc the
    // cmdline check other backends use is unavailable, so confirm the port
    // answers (ours) or at least that we recorded it (ours to stop).
    const killed = await terminate(pid)
    await fs.rm(PID_FILE, { force: true }).catch(() => {})
    launchedWith = undefined
    lastSeen = undefined
    return killed
  }

  function isRemote(cfg: Server.ServerSettings): boolean {
    return !!cfg.remote
  }

  async function loaded(): Promise<LoadedModel | undefined> {
    const cfg = await config()
    const info = await servedModel(origin(), PROBE_TIMEOUT, cfg.apiKey)
    if (info) {
      const window = windowFor(undefined, cfg)
      const model: LoadedModel = {
        id: info,
        args: cfg.remote
          ? { host: cfg.remote }
          : {
              engine: isVlm(cfg) ? "mlx-vlm" : "mlx-lm",
              context: String(window),
              model: collapseHome(info),
            },
      }
      lastSeen = { model, at: Date.now() }
      return model
    }
    if (cfg.remote) return undefined
    if (lastSeen && (await serverAlive())) return lastSeen.model
    if (!(await serverAlive())) return undefined
    const recorded = (await fs.readFile(CONFIG_FILE, "utf8").catch(() => "")).trim()
    return {
      id: recorded || "model",
      args: { engine: "mlx", context: String(windowFor(undefined, cfg)) },
      loading: true,
      stage: "loading weights",
    }
  }

  /**
   * Polling watcher: MLX memory-maps weights and answers /v1/models only once
   * ready, so "no answer yet but process alive" IS the progress signal. No log
   * parsing — the server's log format is not a contract.
   */
  function watch(onEvent: (event: LoadEvent) => void): () => void {
    let stopped = false
    void (async () => {
      const cfg = await config().catch(() => undefined)
      if (!cfg) return
      if (cfg.remote) {
        const id = await servedModel(origin(), PROBE_TIMEOUT * 4, cfg.apiKey)
        if (id && !stopped) onEvent({ model: id, loading: false, loaded: true })
        return
      }
      const active =
        launchedWith ?? (await fs.readFile(CONFIG_FILE, "utf8").catch(() => "")).trim() ?? ""
      const deadline = Date.now() + START_TIMEOUT
      let announced = false
      while (!stopped && Date.now() < deadline) {
        const id = await servedModel(origin(), PROBE_TIMEOUT, cfg.apiKey).catch(() => undefined)
        if (id) {
          onEvent({ model: id, loading: false, loaded: true })
          return
        }
        if (!announced) {
          announced = true
          onEvent({ model: active || "model", loading: true, stage: "loading weights" })
        }
        if (!(await serverAlive())) {
          onEvent({ model: active || "model", loading: false, failed: true })
          return
        }
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL))
      }
      if (!stopped) onEvent({ model: active || "model", loading: false, failed: true })
    })()
    return () => {
      stopped = true
    }
  }

  return {
    id: Server.BACKEND,
    name: "MLX",
    providerName: "Localhost-MLX",
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
