import fs from "fs/promises"
import path from "path"
import os from "os"
import { spawn } from "child_process"
import { stateDir, collapseHome } from "../../shared/paths.ts"
import type { LoadEvent, LoadedModel, ProviderStatus } from "../../shared/types.ts"
import type { Backend, DiscoveredModel } from "../backend.ts"
import * as Server from "./server-ini.ts"
import * as Models from "./models-ini.ts"
import { scan } from "./discover.ts"

/**
 * The llama.cpp backend: find llama-server, start it against models.ini, and
 * report what it has. Everything llama.cpp-specific lives here and below.
 */

const STATE = stateDir(Server.BACKEND)
const PID_FILE = path.join(STATE, "server.pid")
const LOG_FILE = path.join(STATE, "server.log")

const PROBE_TIMEOUT = 1_500
const START_TIMEOUT = 20_000
const POLL_INTERVAL = 400

/** Only $PATH. Guessing at build directories bakes one machine into everyone's. */
async function onPath(): Promise<string | undefined> {
  const dirs = (process.env["PATH"] || "").split(path.delimiter).filter(Boolean)
  for (const dir of dirs) {
    const candidate = path.join(dir, "llama-server")
    const stat = await fs.stat(candidate).catch(() => undefined)
    if (stat?.isFile()) return candidate
  }
  return undefined
}

async function executable(file: string): Promise<boolean> {
  const stat = await fs.stat(file).catch(() => undefined)
  return !!stat?.isFile()
}

/** Reachable when it answers /v1/models; undefined means nothing is listening. */
async function reachable(baseURL: string, apiKey: string, timeout: number): Promise<boolean> {
  try {
    const res = await fetch(`${baseURL}/models`, {
      signal: AbortSignal.timeout(timeout),
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
    })
    return res.ok
  } catch {
    return false
  }
}

export function create(): Backend {
  let settings: Server.ServerSettings | undefined
  let starting: Promise<ProviderStatus> | undefined

  // Re-read every time rather than caching: server.ini is small, and a cached
  // copy meant editing it did nothing until the process restarted — including
  // for the panel, which polls this several times a minute.
  const config = async () => (settings = await Server.load())
  const baseURL = () => `${origin()}/v1`

  /** Pointing at another machine: we are a client, not a supervisor. */
  const isRemote = () => !!settings?.remote

  /**
   * Scheme+host+port, no path. Everything that is not /v1 (the SSE stream, the
   * router's load/unload) must build from this — hand-rolling host:port from the
   * settings silently ignored `remote` and pointed the laptop at its own loopback.
   */
  const origin = () => {
    if (settings?.remote) return `http://${settings.remote}`
    const host = settings?.host === "0.0.0.0" ? "127.0.0.1" : (settings?.host ?? "127.0.0.1")
    return `http://${host}:${settings?.port ?? 9337}`
  }

  /**
   * Where *other* machines reach this server, or undefined when it is loopback-only.
   *
   * baseURL() is deliberately loopback even on 0.0.0.0 — that is how this machine
   * connects. But then nothing on screen tells you the address to put in a second
   * machine's config, which is the one thing you need once you bind the LAN.
   *
   * Prefers the mDNS name: the IPv4 moves on DHCP renewal and the hostname does not,
   * so `fedora.local` is the address worth copying down. The IP is shown alongside
   * because mDNS is not reliable on every client.
   */
  const lanAddress = () => {
    if (settings?.remote) return undefined
    if (settings?.host !== "0.0.0.0") return undefined
    const port = settings?.port ?? 9337
    const ipv4 = Object.values(os.networkInterfaces())
      .flat()
      .find((nic) => nic && nic.family === "IPv4" && !nic.internal)?.address
    const name = os.hostname()
    if (name && ipv4) return `${name}.local:${port} (${ipv4})`
    return ipv4 ? `${ipv4}:${port}` : undefined
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
    // A remote server needs no binary and no models directory here. The only
    // question worth asking is whether it answers.
    if (cfg.remote) {
      if (await reachable(baseURL(), cfg.apiKey, PROBE_TIMEOUT)) {
        return { state: "running", endpoint: baseURL() }
      }
      return {
        state: "failed",
        message: `no answer from ${cfg.remote}`,
        hint: `check it is running and bound to 0.0.0.0`,
      }
    }
    const bin = await resolveBin(cfg)
    if (!bin) {
      return {
        state: "unconfigured",
        missing: "binary",
        message: "llama-server not found",
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
    if (await reachable(baseURL(), cfg.apiKey, PROBE_TIMEOUT)) {
      return { state: "running", endpoint: baseURL(), lan: lanAddress() }
    }
    return { state: "stopped" }
  }

  /**
   * Models on someone else's server. Its /v1/models carries each entry's launch
   * argv, so the real --ctx-size comes back with the listing — opencode compacts
   * against that number, and guessing it would silently truncate long sessions.
   *
   * Sampling stays empty on purpose: the remote server already applies its own
   * models.ini, and overriding it from this machine would fight it.
   */
  async function remoteModels(cfg: Server.ServerSettings): Promise<DiscoveredModel[]> {
    const res = await fetch(`${baseURL()}/models`, {
      signal: AbortSignal.timeout(PROBE_TIMEOUT * 4),
      headers: cfg.apiKey ? { Authorization: `Bearer ${cfg.apiKey}` } : {},
    }).catch(() => undefined)
    if (!res?.ok) return []
    const body: any = await res.json().catch(() => undefined)
    const entries: any[] = Array.isArray(body?.data) ? body.data : []
    return entries.flatMap((entry) => {
      const id = typeof entry?.id === "string" ? entry.id : undefined
      if (!id) return []
      const args = argsOf(entry?.status?.args)
      const parsed = Number.parseInt(args["ctx-size"] ?? "", 10)
      const context = Number.isFinite(parsed) && parsed > 0 ? parsed : 32_768
      return [{
        id,
        name: id.split("/").filter(Boolean).at(-1) ?? id,
        context,
        output: Math.min(32_768, Math.max(4_096, Math.floor(context / 2))),
        sampling: {},
      }]
    })
  }

  async function models(): Promise<DiscoveredModel[]> {
    const cfg = await config()
    if (cfg.remote) return remoteModels(cfg)
    if (!cfg.modelsDir) return []
    const local = await scan(cfg.modelsDir)
    if (local.length === 0) return []
    const { names } = await Models.sync(local)
    const perModel = await Models.settings()
    return local.map((model) => {
      // the section name governs, so a renamed section keeps its tuned values
      const id = names.get(model.file) ?? model.id
      const entry = perModel[id]
      const context = entry?.context || 32_768
      return {
        id,
        name: id.split("/").filter(Boolean).at(-1) ?? id,
        context,
        // thinking models spend the reasoning budget from the output allowance;
        // half the window, capped, leaves room for a long plan without truncating
        output: Math.min(32_768, Math.max(4_096, Math.floor(context / 2))),
        sampling: entry?.sampling ?? {},
      }
    })
  }

  async function launch(cfg: Server.ServerSettings, bin: string): Promise<ProviderStatus> {
    const local = await scan(cfg.modelsDir)
    if (local.length === 0) {
      return {
        state: "unconfigured",
        missing: "models-dir",
        message: "no .gguf files found",
        hint: collapseHome(cfg.modelsDir),
      }
    }
    const { preset } = await Models.sync(local)
    await fs.mkdir(STATE, { recursive: true }).catch(() => {})
    const log = await fs.open(LOG_FILE, "a").catch(() => undefined)
    try {
      const child = spawn(bin, Server.argv(cfg, preset), {
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
      if (await reachable(baseURL(), cfg.apiKey, PROBE_TIMEOUT)) {
        return { state: "running", endpoint: baseURL(), lan: lanAddress() }
      }
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL))
    }
    return { state: "failed", message: "server did not start", hint: collapseHome(LOG_FILE) }
  }

  /** Idempotent and single-flight: concurrent callers share one attempt. */
  async function start(): Promise<ProviderStatus> {
    if (starting) return starting
    starting = (async () => {
      const cfg = await config()
      // Not ours to start. Report what the remote says and leave it alone.
      if (cfg.remote) return status()
      const bin = await resolveBin(cfg)
      if (!bin || !cfg.modelsDir) return status()
      if (await reachable(baseURL(), cfg.apiKey, PROBE_TIMEOUT)) {
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

  /**
   * Stop the server we spawned. Checks /proc before signalling so a recycled
   * pid is never killed, and only kills a process that is actually
   * llama-server — the pidfile outlives crashes and reboots.
   */
  /**
   * Every llama-server on this machine that belongs to this router: the router
   * itself (matched by --port, since several checkouts' routers could coexist)
   * and the model servers it spawned (matched by parentage, with a cmdline
   * fallback for children re-parented to init after a router crash — those are
   * exactly the ones that keep 20-30GB of VRAM after "stop" appears to work).
   */
  async function routerFamily(port: number): Promise<{ router?: number; children: number[] }> {
    const family: { router?: number; children: number[] } = { children: [] }
    const orphans: number[] = []
    const entries = await fs.readdir("/proc").catch(() => [] as string[])
    for (const entry of entries) {
      const pid = Number.parseInt(entry, 10)
      if (!Number.isFinite(pid) || pid <= 0) continue
      const exe = await fs.readlink(`/proc/${pid}/exe`).catch(() => "")
      if (!exe.includes("llama-server")) continue
      const raw = await fs.readFile(`/proc/${pid}/cmdline`, "utf8").catch(() => "")
      const args = raw.split("\0")
      const isRouter = args.includes("--models-preset") || args.includes("--models-dir")
      if (isRouter) {
        const at = args.indexOf("--port")
        if (at >= 0 && args[at + 1] === String(port)) family.router = pid
        continue
      }
      // a spawned model server carries an explicit --model path
      if (args.includes("--model")) orphans.push(pid)
    }
    // Every spawned model server counts, whatever its parent says: after a
    // router crash or restart the children re-parent to init, and those are
    // exactly the ones that keep the VRAM.
    family.children = orphans
    return family
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

  async function stop(): Promise<boolean> {
    // On a remote we unload the model rather than kill the process. Freeing VRAM
    // is the useful half of "stop"; killing a server other people may be using is
    // not ours to do. Model *switching* needs neither — the router runs with
    // models-max=1 and swaps on its own when a request names another model.
    if (isRemote()) {
      const cfg = await config()
      const current = await loaded()
      if (!current) return false
      const res = await fetch(`${origin()}/models/unload`, {
        method: "POST",
        signal: AbortSignal.timeout(PROBE_TIMEOUT * 4),
        headers: {
          "Content-Type": "application/json",
          ...(cfg.apiKey ? { Authorization: `Bearer ${cfg.apiKey}` } : {}),
        },
        body: JSON.stringify({ model: current.id }),
      }).catch(() => undefined)
      return !!res?.ok
    }

    // The pid file only covers routers we launched. A router started by hand in
    // a terminal — the normal way during tuning — answers the health probe just
    // the same, so without the /proc fallback the panel shows a server the stop
    // button silently cannot touch. And killing only the router is not stopping:
    // the model server it spawned holds the VRAM and must go first, or the next
    // request races a half-dead family.
    const cfg = await config()
    const family = await routerFamily(cfg.port)

    const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
    const recorded = Number.parseInt(raw.trim(), 10)
    if (Number.isFinite(recorded) && recorded > 0 && family.router === undefined) {
      const exe = await fs.readlink(`/proc/${recorded}/exe`).catch(() => "")
      if (exe.includes("llama-server")) family.router = recorded
    }

    if (family.router === undefined && family.children.length === 0) {
      await fs.rm(PID_FILE, { force: true }).catch(() => {})
      return false
    }

    // router first so nothing respawns a child mid-stop
    let stopped = false
    if (family.router !== undefined) stopped = (await terminate(family.router)) || stopped
    for (const child of family.children) stopped = (await terminate(child)) || stopped
    await fs.rm(PID_FILE, { force: true }).catch(() => {})
    return stopped
  }

  /** llama-server reports its launch flags as a flat argv array. */
  function argsOf(argv: unknown): Record<string, string> {
    if (!Array.isArray(argv)) return {}
    const out: Record<string, string> = {}
    for (let i = 0; i < argv.length; i++) {
      const token = String(argv[i])
      if (!token.startsWith("--")) continue
      const next = argv[i + 1]
      if (next === undefined || String(next).startsWith("--")) continue
      out[token.slice(2)] = String(next)
    }
    return out
  }

  async function loaded(): Promise<LoadedModel | undefined> {
    const cfg = await config()
    try {
      const res = await fetch(`${baseURL()}/models`, {
        signal: AbortSignal.timeout(PROBE_TIMEOUT),
        headers: cfg.apiKey ? { Authorization: `Bearer ${cfg.apiKey}` } : {},
      })
      if (!res.ok) return undefined
      const body: any = await res.json()
      const entries: any[] = Array.isArray(body?.data) ? body.data : []
      // llama-server reports state as status.value: "loaded" | "unloaded" |
      // "loading". Every preset carries args, so matching on args alone picked
      // whichever model sorted first rather than the one actually loaded.
      const active =
        entries.find((entry) => entry?.status?.value === "loading") ??
        entries.find((entry) => entry?.status?.value === "loaded")
      if (!active) return undefined
      // llama-server reports load progress as {stages, current, value} while
      // weights stream in; value is 0-1 for the stage named by `current`
      return {
        id: String(active.id ?? ""),
        args: argsOf(active?.status?.args),
        loading: active?.status?.value === "loading",
      }
    } catch {
      return undefined
    }
  }

  /**
   * Stream model state from /models/sse. The REST listing only flips from
   * unloaded to loaded at the very end, so a thirty-second load looks like a
   * hang unless we read this.
   *
   * Captured from a real load: one "model_status" frame announces the load,
   * then ~20 "status_change" frames carry the progress, ending with
   * status "loaded". The outgoing model gets its own "unloaded" frame.
   *
   *   data: {"model":"...","event":"status_change",
   *          "data":{"status":"loading",
   *                  "progress":{"stages":["text_model"],"current":"text_model","value":0.19}}}
   */
  function watch(onEvent: (event: LoadEvent) => void): () => void {
    const controller = new AbortController()
    let stopped = false

    const connect = async () => {
      while (!stopped) {
        try {
          const cfg = await config()
          const res = await fetch(`${origin()}/models/sse`, {
            signal: controller.signal,
            headers: cfg.apiKey ? { Authorization: `Bearer ${cfg.apiKey}` } : {},
          })
          if (!res.ok || !res.body) throw new Error("no stream")
          const reader = res.body.getReader()
          const decoder = new TextDecoder()
          let buffer = ""
          while (!stopped) {
            const { done, value } = await reader.read()
            if (done) break
            buffer += decoder.decode(value, { stream: true })
            // frames are separated by a blank line; keep any partial tail
            const frames = buffer.split("\n\n")
            buffer = frames.pop() ?? ""
            for (const frame of frames) {
              const line = frame.split("\n").find((item) => item.startsWith("data:"))
              if (!line) continue
              try {
                const payload = JSON.parse(line.slice(5).trim())
                // progress rides on status_change; model_status only announces
                // the start. Filtering to model_status dropped every update.
                const kind = payload?.event
                if (kind && kind !== "model_status" && kind !== "status_change") continue
                const body = payload?.data ?? {}
                const progress = body?.progress
                onEvent({
                  model: String(payload?.model ?? ""),
                  loading: body?.status === "loading",
                  failed: body?.status === "failed" || body?.status === "error",
                  loaded: body?.status === "loaded",
                  progress: typeof progress?.value === "number" ? progress.value : undefined,
                  stage: typeof progress?.current === "string" ? progress.current : undefined,
                })
              } catch {
                // a malformed frame is not worth tearing the stream down for
              }
            }
          }
        } catch {
          // server down or restarted; wait before reconnecting
        }
        if (stopped) return
        await new Promise((resolve) => setTimeout(resolve, 2_000))
      }
    }

    void connect()
    return () => {
      stopped = true
      controller.abort()
    }
  }

  return {
    id: Server.BACKEND,
    name: "llama.cpp",
    providerName: "Localhost-llama.cpp",
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
