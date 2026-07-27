import fs from "fs/promises"
import path from "path"
import os from "os"
import { spawn } from "child_process"
import { stateDir, collapseHome } from "../../shared/paths.ts"
import type { LoadEvent, LoadedModel, ProviderStatus } from "../../shared/types.ts"
import type { Backend, DiscoveredModel } from "../backend.ts"
import * as Server from "./server-ini.ts"
import { scan, type LocalModel } from "./discover.ts"

/**
 * The OpenVINO backend: serve an OpenVINO IR model through OVMS.
 *
 * Shaped differently from llama.cpp in one way that matters. OVMS compiles a
 * model onto the GPU at startup and serves that one model for the life of the
 * process — there is no preset file and no swapping. Worse, the GPU allocation
 * is only released when the process exits: the pipeline exposes no close(),
 * and release_memory() lives a layer below on ov.CompiledModel, which it never
 * surfaces. So a second model cannot simply be loaded alongside the first, and
 * switching means restarting. That is why `model` is a setting here rather
 * than a picker entry, and why this backend reports exactly one model.
 */

const STATE = stateDir(Server.BACKEND)
const PID_FILE = path.join(STATE, "server.pid")
const LOG_FILE = path.join(STATE, "server.log")

const PROBE_TIMEOUT = 1_500
/**
 * Far longer than llama.cpp's. OVMS compiles the IR graph and streams the
 * weights onto the GPU before it answers anything: measured 24-31s for a
 * ~19GB model with a warm page cache, and a cold one adds the read on top.
 * Giving up early is indistinguishable from "no server here" downstream.
 */
const START_TIMEOUT = 180_000
const POLL_INTERVAL = 1_000

/** Only $PATH. Guessing at install directories bakes one machine into everyone's. */
async function onPath(): Promise<string | undefined> {
  const dirs = (process.env["PATH"] || "").split(path.delimiter).filter(Boolean)
  for (const dir of dirs) {
    const candidate = path.join(dir, "ovms-serve")
    const stat = await fs.stat(candidate).catch(() => undefined)
    if (stat?.isFile()) return candidate
  }
  return undefined
}

async function executable(file: string): Promise<boolean> {
  const stat = await fs.stat(file).catch(() => undefined)
  return !!stat?.isFile()
}

/**
 * Readiness comes from /v2/health/ready, not from the models endpoint.
 *
 * /v3/models answers 200 with an empty list for the whole graph-compile
 * window, which reads as a healthy server that has nothing to offer — the
 * provider would be registered with no models and quietly dropped. The health
 * endpoint returns 503 for exactly that window and 200 once the graph is
 * actually servable.
 */
async function ready(host: string, port: number, timeout: number): Promise<boolean> {
  try {
    const res = await fetch(`http://${host}:${port}/v2/health/ready`, {
      signal: AbortSignal.timeout(timeout),
    })
    return res.ok
  } catch {
    return false
  }
}

/**
 * The id the server is actually serving right now, or undefined.
 *
 * Note this is not a readiness check: `/v3/models` answers 200 with an empty
 * list all through the graph-compile window, which is why `ready()` above
 * exists separately.
 */
async function servedModelFrom(host: string, port: number, timeout: number): Promise<string | undefined> {
  try {
    const res = await fetch(`http://${host}:${port}/v3/models`, { signal: AbortSignal.timeout(timeout) })
    if (!res.ok) return undefined
    const body: any = await res.json()
    const first = Array.isArray(body?.data) ? body.data[0] : undefined
    return typeof first?.id === "string" ? first.id : undefined
  } catch {
    return undefined
  }
}

/** The model to serve: the chosen one, or the only one. Never a guess. */
function chosen(cfg: Server.ServerSettings, local: LocalModel[]): LocalModel | undefined {
  if (cfg.model) return local.find((model) => model.id === cfg.model)
  return local.length === 1 ? local[0] : undefined
}

/**
 * The ovms process holding `port`, found by scanning /proc.
 *
 * Needed because the pid file only covers servers *we* launched. A server
 * started by hand — or left behind by a crashed panel — answers the health
 * probe just the same, so `status()` reports running while `stop()` has no pid
 * to signal: the panel shows a server the user cannot turn off. Since OVMS is
 * one model per process, matching the port on an `ovms` cmdline is unambiguous.
 */
async function pidOnPort(port: number): Promise<number | undefined> {
  const entries = await fs.readdir("/proc").catch(() => [] as string[])
  for (const entry of entries) {
    const pid = Number.parseInt(entry, 10)
    if (!Number.isFinite(pid) || pid <= 0) continue
    const comm = await fs.readFile(`/proc/${pid}/comm`, "utf8").catch(() => "")
    if (comm.trim() !== "ovms") continue
    const raw = await fs.readFile(`/proc/${pid}/cmdline`, "utf8").catch(() => "")
    const args = raw.split("\0")
    const at = args.indexOf("--rest_port")
    if (at >= 0 && args[at + 1] === String(port)) return pid
  }
  return undefined
}

/** SIGTERM, wait out the GPU unload, then SIGKILL. Shared by both stop paths. */
async function terminate(pid: number): Promise<boolean> {
  try {
    process.kill(pid, "SIGTERM")
  } catch {
    return false
  }
  // unloading ~20GB from the GPU is not instant; give it room before SIGKILL
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
  const port = () => settings?.port ?? 8100
  // OVMS mounts its OpenAI-compatible endpoints under /v3; /v1 carries only
  // /v1/config and 404s for models and chat.
  const baseURL = () => `http://${host()}:${port()}/v3`

  /**
   * Where *other* machines reach this server, or undefined when it is
   * loopback-only. baseURL() stays loopback even on 0.0.0.0 — that is how this
   * machine connects — so without this nothing on screen tells you the address
   * to put in a second machine's config.
   *
   * Prefers the mDNS name: the IPv4 moves on DHCP renewal and the hostname does
   * not, so `fedora.local` is the address worth writing down. The IP comes
   * along because mDNS is not reliable on every client.
   */
  const lanAddress = () => {
    if (settings?.host !== "0.0.0.0") return undefined
    const ipv4 = Object.values(os.networkInterfaces())
      .flat()
      .find((nic) => nic && nic.family === "IPv4" && !nic.internal)?.address
    const name = os.hostname()
    if (name && ipv4) return `${name}.local:${port()} (${ipv4})`
    return ipv4 ? `${ipv4}:${port()}` : undefined
  }

  async function resolveBin(cfg: Server.ServerSettings): Promise<string | undefined> {
    if (cfg.bin) return (await executable(cfg.bin)) ? cfg.bin : undefined
    const found = await onPath()
    // record it so the file always shows what is actually being used
    if (found) await Server.update("bin", found)
    return found
  }

  async function status(): Promise<ProviderStatus> {
    const cfg = await config()
    const bin = await resolveBin(cfg)
    if (!bin) {
      return {
        state: "unconfigured",
        missing: "binary",
        message: "ovms-serve not found",
        hint: `set bin in ${collapseHome(Server.FILE)}`,
      }
    }
    if (!cfg.modelsDir) {
      return {
        state: "unconfigured",
        missing: "models-dir",
        message: "models-dir not set",
        hint: `set it in ${collapseHome(Server.FILE)}`,
      }
    }
    if (await ready(host(), port(), PROBE_TIMEOUT)) {
      return { state: "running", endpoint: baseURL(), lan: lanAddress() }
    }
    return { state: "stopped" }
  }

  async function models(): Promise<DiscoveredModel[]> {
    const cfg = await config()
    if (!cfg.modelsDir) return []
    const local = await scan(cfg.modelsDir)
    // What a running server actually answers for wins over what the file asks
    // for. OVMS serves exactly one model and 404s every other name, so
    // advertising the configured id while a different one is loaded produces a
    // provider whose every request fails — and a panel whose name never
    // matches what is running. Config decides what the *next* start loads.
    const served = await servedModelFrom(host(), port(), PROBE_TIMEOUT)
    const model = (served && local.find((entry) => entry.id === served)) || chosen(cfg, local)
    if (!model) return []
    const context = cfg.context
    return [
      {
        id: model.id,
        name: model.id.replace(/-ov$/i, ""),
        context,
        // thinking models spend the reasoning budget from the output allowance;
        // half the window, capped, leaves room for a long plan without truncating
        output: Math.min(32_768, Math.max(4_096, Math.floor(context / 2))),
        sampling: cfg.sampling,
      },
    ]
  }

  async function launch(cfg: Server.ServerSettings, bin: string): Promise<ProviderStatus> {
    const local = await scan(cfg.modelsDir)
    if (local.length === 0) {
      return {
        state: "unconfigured",
        missing: "models-dir",
        message: "no OpenVINO IR models found",
        hint: collapseHome(cfg.modelsDir),
      }
    }
    const model = chosen(cfg, local)
    if (!model) {
      // Refusing to pick is deliberate: each model is tens of GB and occupies
      // the GPU for the life of the process, so there is no cheap default.
      return {
        state: "failed",
        message: cfg.model ? `model "${cfg.model}" not found` : `${local.length} models found, none chosen`,
        hint: `set model in ${collapseHome(Server.FILE)}`,
      }
    }

    await fs.mkdir(STATE, { recursive: true }).catch(() => {})
    const log = await fs.open(LOG_FILE, "a").catch(() => undefined)
    try {
      const child = spawn(bin, Server.argv(cfg, model.id), {
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
      if (await ready(host(), port(), PROBE_TIMEOUT)) {
        return { state: "running", endpoint: baseURL(), lan: lanAddress() }
      }
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL))
    }
    return { state: "failed", message: "server did not start", hint: collapseHome(LOG_FILE) }
  }

  /**
   * Stop a server we started. Matters more here than for llama.cpp: OVMS holds
   * its model on the GPU for the life of the process and there is no in-process
   * release, so stopping is the *only* way to get the memory back — and the
   * only way to switch models.
   *
   * The recorded pid is `ovms` itself, not the wrapper: `ovms-serve` sets its
   * environment and then `exec`s the binary, so the process keeps the pid.
   */
  async function stop(): Promise<boolean> {
    await config()
    const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
    const recorded = Number.parseInt(raw.trim(), 10)
    let pid: number | undefined
    if (Number.isFinite(recorded) && recorded > 0) {
      const exe = await fs.readlink(`/proc/${recorded}/exe`).catch(() => "")
      // never signal a pid that has been recycled into something else
      if (exe.includes("ovms")) pid = recorded
      else await fs.rm(PID_FILE, { force: true }).catch(() => {})
    }
    // Fall back to whoever actually holds the port. Without this, a server we
    // did not launch — or one whose pid file was lost — is unstoppable from
    // the panel, and OVMS only releases its GPU memory on exit.
    if (pid === undefined) pid = await pidOnPort(port())
    if (pid === undefined) return false
    const killed = await terminate(pid)
    await fs.rm(PID_FILE, { force: true }).catch(() => {})
    return killed
  }

  /**
   * What the server currently holds.
   *
   * OVMS spends 25-60s compiling the graph onto the GPU before it answers
   * anything, and during that window health is 503 and the model list is
   * empty — so "process alive but not ready" is reported as loading rather
   * than as nothing, which is the difference between a panel that looks stuck
   * and one that tells you what it is doing.
   */
  async function loaded(): Promise<LoadedModel | undefined> {
    const cfg = await config()
    const args: Record<string, string> = {
      device: "GPU",
      "cache-size": `${cfg.cacheSize}G`,
      context: String(cfg.context),
    }
    const id = await servedModelFrom(host(), port(), PROBE_TIMEOUT)
    if (id) return { id, args }

    // not answering yet — is it still coming up, or simply not running?
    const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
    const pid = Number.parseInt(raw.trim(), 10)
    if (!Number.isFinite(pid) || pid <= 0) return undefined
    try {
      process.kill(pid, 0)
    } catch {
      return undefined
    }
    const local = await scan(cfg.modelsDir)
    const model = chosen(cfg, local)
    if (!model) return undefined
    // no byte-level progress to report: OVMS logs stages, not a fraction
    return { id: model.id, args, loading: true, stage: "compiling graph" }
  }

  /**
   * Live load progress, read from the server log.
   *
   * OVMS has no equivalent of llama-server's `/models/sse`; the REST surface
   * only flips at the end, and the load takes 25-60s. What it does have is a
   * log that narrates the transitions, so we tail it. That means **no numeric
   * progress** — OVMS reports stages, not a fraction — so `progress` stays
   * undefined and `stage` carries the phase.
   *
   * Tailing by offset rather than `fs.watch` keeps this working across server
   * restarts, since the log is appended to and the next load simply continues
   * the file.
   */
  function watch(onEvent: (event: LoadEvent) => void): () => void {
    let stopped = false
    let offset = -1 // -1 = start from the end, so old loads are not replayed
    let model = ""

    // Seed the name from config rather than waiting for the log to mention it.
    // OVMS only names the model on the final "state changed to" line, so
    // without this the panel shows a blank name for the whole ~56s load —
    // exactly the window this exists to narrate.
    void (async () => {
      const cfg = await config().catch(() => undefined)
      if (!cfg || model) return
      const local = await scan(cfg.modelsDir).catch(() => [])
      model = chosen(cfg, local)?.id ?? ""
    })()

    const emit = (event: Partial<LoadEvent> & { loading: boolean }) =>
      onEvent({ model, ...event } as LoadEvent)

    const scanLine = (line: string) => {
      // "Mediapipe: <name> state changed to: <STATE>" carries the model name
      const named = line.match(/Mediapipe:\s+(\S+)\s+state changed to:\s+(\S+)/)
      if (named) {
        model = named[1] ?? model
        const state = named[2] ?? ""
        if (state === "AVAILABLE") return emit({ loading: false, loaded: true })
        if (state.includes("FAILED")) return emit({ loading: false, failed: true })
        return emit({ loading: true, stage: state.toLowerCase() })
      }
      if (line.includes("ServableManagerModule starting")) {
        return emit({ loading: true, stage: "starting" })
      }
      if (line.includes("servable") && line.includes("Initializing")) {
        return emit({ loading: true, stage: "compiling graph" })
      }
      // Only these are fatal. The log also carries "BOS token was not found in
      // model files." at error level on a perfectly successful load — treating
      // any [error] line as failure would mark every good start as broken.
      if (line.includes("Couldn't start model manager") || line.includes("Failed to process LLM node graph")) {
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
              // first look, or the file was truncated
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
      const bin = await resolveBin(cfg)
      if (!bin || !cfg.modelsDir) return status()
      if (await ready(host(), port(), PROBE_TIMEOUT)) {
        return { state: "running", endpoint: baseURL(), lan: lanAddress() } as ProviderStatus
      }
      return launch(cfg, bin)
    })()
    try {
      return await starting
    } finally {
      starting = undefined
    }
  }

  return {
    id: Server.BACKEND,
    name: "OpenVINO",
    providerName: "Localhost-OpenVINO",
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
