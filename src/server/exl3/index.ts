import fs from "fs/promises"
import path from "path"
import os from "os"
import { spawn } from "child_process"
import { stateDir, collapseHome } from "../../shared/paths.ts"
import type { LoadEvent, LoadedModel, ProviderStatus } from "../../shared/types.ts"
import type { Backend, DiscoveredModel } from "../backend.ts"
import * as Server from "./server-ini.ts"

/**
 * The exllamav3 backend: serve an EXL3 model through TabbyAPI.
 *
 * Shaped like OpenVINO, for the same reason: one model per process, chosen at
 * startup, held for the life of the process. The choice lives in the TabbyAPI
 * YAML that `config` points at — model name, gpu split, cache sizing, and
 * whether MTP speculative decoding is on all come from there, so switching any
 * of them means switching the YAML and restarting.
 *
 * Why this backend exists at all: exllamav3's page-hashed cache keeps several
 * conversations resident at once, so agent fan-out switches conversations in
 * ~1-2s where llama.cpp re-prefills for minutes; with tensor_parallel and MTP
 * both on it also decodes 1.4-1.8x faster at depth than llama.cpp on the same
 * two GPUs. Measured on this machine, 2026-07-28 — see
 * backends-qwen27b-benchmarks.md for the tables and the two local exllamav3
 * fixes (turboderp-org/exllamav3#260) the MTP path currently needs.
 */

const STATE = stateDir(Server.BACKEND)
const PID_FILE = path.join(STATE, "server.pid")
const LOG_FILE = path.join(STATE, "server.log")

const PROBE_TIMEOUT = 1_500
/**
 * A ~20GB EXL3 checkpoint streams onto the GPUs in 25-60s with a warm page
 * cache; a cold read adds the disk on top. Same reasoning as OpenVINO's
 * timeout: giving up early reads as "no server here" downstream.
 */
const START_TIMEOUT = 300_000
const POLL_INTERVAL = 1_000

async function executable(file: string): Promise<boolean> {
  const stat = await fs.stat(file).catch(() => undefined)
  return !!stat?.isFile()
}

/** TabbyAPI's /health answers 200 once the model is actually servable. */
async function ready(origin: string, timeout: number, apiKey?: string): Promise<boolean> {
  try {
    const res = await fetch(`${origin}/health`, {
      signal: AbortSignal.timeout(timeout),
      headers: apiKey ? { "x-api-key": apiKey } : {},
    })
    return res.ok
  } catch {
    return false
  }
}

/** The id the server is actually serving right now, or undefined. */
async function servedModelFrom(origin: string, timeout: number, apiKey?: string): Promise<string | undefined> {
  try {
    const res = await fetch(`${origin}/v1/models`, {
      signal: AbortSignal.timeout(timeout),
      headers: apiKey ? { "x-api-key": apiKey } : {},
    })
    if (!res.ok) return undefined
    const body: any = await res.json()
    const first = Array.isArray(body?.data) ? body.data[0] : undefined
    return typeof first?.id === "string" ? first.id : undefined
  } catch {
    return undefined
  }
}

/**
 * The python process launched from `tabby-dir`, found by scanning /proc.
 *
 * Needed for the same reason as OpenVINO's: the pid file only covers servers
 * we launched. The process name is just "python", so main.py's path is the
 * discriminator — unambiguous because one checkout serves one process.
 */
async function pidForTabby(tabbyDir: string): Promise<number | undefined> {
  const needle = path.join(tabbyDir, "main.py")
  const entries = await fs.readdir("/proc").catch(() => [] as string[])
  for (const entry of entries) {
    const pid = Number.parseInt(entry, 10)
    if (!Number.isFinite(pid) || pid <= 0) continue
    const raw = await fs.readFile(`/proc/${pid}/cmdline`, "utf8").catch(() => "")
    if (raw.includes(needle)) return pid
  }
  return undefined
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

  // Re-read every time rather than caching, so editing the file takes effect
  // without restarting the process the panel polls from.
  const config = async () => (settings = await Server.load())
  const host = () => (settings?.host === "0.0.0.0" ? "127.0.0.1" : (settings?.host ?? "127.0.0.1"))
  const port = () => settings?.port ?? 5000

  const isRemote = () => !!settings?.remote

  const origin = () => {
    if (settings?.remote) return `http://${settings.remote}`
    return `http://${host()}:${port()}`
  }

  const baseURL = () => `${origin()}/v1`

  /** Same contract as OpenVINO's: the address another machine would use. */
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

  /** Everything launch needs, or the first thing missing. */
  async function unconfigured(cfg: Server.ServerSettings): Promise<ProviderStatus | undefined> {
    if (!cfg.bin || !(await executable(cfg.bin))) {
      return {
        state: "unconfigured",
        missing: "binary",
        message: "python with exllamav3 not set",
        hint: `set bin in ${collapseHome(Server.FILE)}`,
      }
    }
    if (!cfg.tabbyDir || !(await executable(path.join(cfg.tabbyDir, "main.py")))) {
      return {
        state: "unconfigured",
        missing: "binary",
        message: "tabby-dir not set or has no main.py",
        hint: `set tabby-dir in ${collapseHome(Server.FILE)}`,
      }
    }
    if (!cfg.config || !(await executable(cfg.config))) {
      return {
        state: "unconfigured",
        missing: "models-dir",
        message: "config not set",
        hint: `point config at a TabbyAPI YAML in ${collapseHome(Server.FILE)}`,
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
      return {
        state: "failed",
        message: `no answer from ${cfg.remote}`,
        hint: "check it is running and bound to 0.0.0.0",
      }
    }
    const missing = await unconfigured(cfg)
    if (missing) return missing
    if (await ready(origin(), PROBE_TIMEOUT, cfg.apiKey)) {
      return { state: "running", endpoint: baseURL(), lan: lanAddress() }
    }
    return { state: "stopped" }
  }

  /**
   * Exactly one model: whatever the server answers for, else the YAML's next
   * start decides and there is nothing truthful to advertise while it is down —
   * except that TabbyAPI names the model after its directory, so when the
   * config is set we read the directory name out of the YAML without a YAML
   * parser: `model_name:` is a single plain-scalar line in every config this
   * plugin writes or documents.
   */
  async function configuredModelName(cfg: Server.ServerSettings): Promise<string | undefined> {
    const raw = await fs.readFile(cfg.config, "utf8").catch(() => undefined)
    const match = raw?.match(/^\s*model_name:\s*(\S+)\s*$/m)
    return match?.[1]
  }

  async function models(): Promise<DiscoveredModel[]> {
    const cfg = await config()
    if (!cfg.remote) {
      const missing = await unconfigured(cfg)
      if (missing) return []
    }
    const served = await servedModelFrom(origin(), PROBE_TIMEOUT, cfg.apiKey)
    const id = served ?? (cfg.remote ? undefined : await configuredModelName(cfg))
    if (!id) return []
    const context = cfg.context
    return [
      {
        id,
        name: id.replace(/-exl3.*$/i, ""),
        context,
        // thinking models spend the reasoning budget from the output allowance
        output: Math.min(32_768, Math.max(4_096, Math.floor(context / 2))),
        // a remote applies its own sampling; overriding from here fights it
        sampling: cfg.remote ? {} : cfg.sampling,
      },
    ]
  }

  async function launch(cfg: Server.ServerSettings): Promise<ProviderStatus> {
    await fs.mkdir(STATE, { recursive: true }).catch(() => {})
    const log = await fs.open(LOG_FILE, "a").catch(() => undefined)
    try {
      const child = spawn(cfg.bin, Server.argv(cfg), {
        cwd: cfg.tabbyDir,
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

  async function stop(): Promise<boolean> {
    const cfg = await config()
    // Not ours to stop, and no unload endpoint to call instead.
    if (isRemote()) return false
    const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
    const recorded = Number.parseInt(raw.trim(), 10)
    let pid: number | undefined
    if (Number.isFinite(recorded) && recorded > 0) {
      const cmdline = await fs.readFile(`/proc/${recorded}/cmdline`, "utf8").catch(() => "")
      // never signal a pid that has been recycled into something else
      if (cmdline.includes("main.py")) pid = recorded
      else await fs.rm(PID_FILE, { force: true }).catch(() => {})
    }
    if (pid === undefined && cfg.tabbyDir) pid = await pidForTabby(cfg.tabbyDir)
    if (pid === undefined) return false
    const killed = await terminate(pid)
    await fs.rm(PID_FILE, { force: true }).catch(() => {})
    return killed
  }

  async function loaded(): Promise<LoadedModel | undefined> {
    const cfg = await config()
    const args: Record<string, string> = {
      engine: "exllamav3",
      context: String(cfg.context),
      config: collapseHome(cfg.config || ""),
    }
    const id = await servedModelFrom(origin(), PROBE_TIMEOUT, cfg.apiKey)
    if (id) return { id, args: cfg.remote ? { host: cfg.remote } : args }

    if (cfg.remote) return undefined

    // not answering yet — is it still coming up, or simply not running?
    const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
    const pid = Number.parseInt(raw.trim(), 10)
    if (!Number.isFinite(pid) || pid <= 0) return undefined
    try {
      process.kill(pid, 0)
    } catch {
      return undefined
    }
    const name = (await configuredModelName(cfg)) ?? "model"
    return { id: name, args, loading: true, stage: "loading weights" }
  }

  /**
   * Live load progress, read from the server log. TabbyAPI narrates the load
   * ("Loading model: ...", uvicorn's "Application startup complete.") without
   * a byte fraction, so `progress` stays undefined and `stage` carries the
   * phase — the OpenVINO approach, and the same offset-tail so restarts keep
   * working.
   */
  function watch(onEvent: (event: LoadEvent) => void): () => void {
    let stopped = false
    let offset = -1 // -1 = start from the end, so old loads are not replayed
    let model = ""

    void (async () => {
      const cfg = await config().catch(() => undefined)
      if (!cfg || model) return
      if (cfg.remote) {
        const id = await servedModelFrom(origin(), PROBE_TIMEOUT * 4, cfg.apiKey)
        if (id && !stopped) onEvent({ model: id, loading: false, loaded: true } as LoadEvent)
        return
      }
      model = (await configuredModelName(cfg).catch(() => undefined)) ?? ""
    })()

    const emit = (event: Partial<LoadEvent> & { loading: boolean }) =>
      onEvent({ model, ...event } as LoadEvent)

    const scanLine = (line: string) => {
      const naming = line.match(/Loading model:\s+(\S+)/)
      if (naming) {
        model = path.basename(naming[1] ?? "") || model
        return emit({ loading: true, stage: "loading weights" })
      }
      if (line.includes("Loading draft model") || line.includes("draft model")) {
        return emit({ loading: true, stage: "loading draft (MTP)" })
      }
      if (line.includes("Application startup complete")) {
        return emit({ loading: false, loaded: true })
      }
      // TabbyAPI recreates its generator on fatal generation errors; the
      // process survives, so only a failed *load* is terminal here
      if (line.includes("Sending SIGKILL") || line.includes("RuntimeError: Insufficient VRAM")) {
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
  async function start(): Promise<ProviderStatus> {
    if (starting) return starting
    starting = (async () => {
      const cfg = await config()
      if (cfg.remote) return status()
      const missing = await unconfigured(cfg)
      if (missing) return missing
      if (await ready(origin(), PROBE_TIMEOUT, settings?.apiKey)) {
        return { state: "running", endpoint: baseURL(), lan: lanAddress() } as ProviderStatus
      }
      return launch(cfg)
    })()
    try {
      return await starting
    } finally {
      starting = undefined
    }
  }

  return {
    id: Server.BACKEND,
    name: "exllamav3",
    providerName: "Localhost-EXL3",
    status,
    models,
    start,
    stop,
    loaded,
    watch,
    baseURL,
    apiKey: () => settings?.apiKey || undefined,
  }
}
