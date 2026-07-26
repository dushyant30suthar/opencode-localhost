import fs from "fs/promises"
import path from "path"
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
  const baseURL = () => {
    const host = settings?.host === "0.0.0.0" ? "127.0.0.1" : (settings?.host ?? "127.0.0.1")
    return `http://${host}:${settings?.port ?? 9337}/v1`
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
      return { state: "running", endpoint: baseURL() }
    }
    return { state: "stopped" }
  }

  async function models(): Promise<DiscoveredModel[]> {
    const cfg = await config()
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
        return { state: "running", endpoint: baseURL() }
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
      const bin = await resolveBin(cfg)
      if (!bin || !cfg.modelsDir) return status()
      if (await reachable(baseURL(), cfg.apiKey, PROBE_TIMEOUT)) {
        return { state: "running", endpoint: baseURL() } as ProviderStatus
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
  async function stop(): Promise<boolean> {
    const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
    const pid = Number.parseInt(raw.trim(), 10)
    if (!Number.isFinite(pid) || pid <= 0) return false
    const exe = await fs.readlink(`/proc/${pid}/exe`).catch(() => "")
    if (!exe.includes("llama-server")) {
      await fs.rm(PID_FILE, { force: true }).catch(() => {})
      return false
    }
    try {
      process.kill(pid, "SIGTERM")
    } catch {
      return false
    }
    for (let i = 0; i < 40; i++) {
      try {
        process.kill(pid, 0)
      } catch {
        break
      }
      await new Promise((resolve) => setTimeout(resolve, 250))
    }
    try {
      process.kill(pid, "SIGKILL")
    } catch {
      // already gone
    }
    await fs.rm(PID_FILE, { force: true }).catch(() => {})
    return true
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
          const host = cfg.host === "0.0.0.0" ? "127.0.0.1" : cfg.host
          const res = await fetch(`http://${host}:${cfg.port}/models/sse`, {
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
