import fs from "fs/promises"
import path from "path"
import os from "os"
import { spawn } from "child_process"
import { stateDir, collapseHome } from "../../shared/paths.ts"
import type { LoadEvent, LoadedModel, ProviderStatus } from "../../shared/types.ts"
import type { Backend, DiscoveredModel } from "../backend.ts"
import * as Server from "./server-ini.ts"

/**
 * The ComfyUI backend: supervise a diffusion server, do not advertise it.
 *
 * Every other backend in this plugin exists to put models in opencode's picker.
 * This one exists for the opposite reason. ComfyUI generates images and video —
 * there is no chat completion to register, and `models()` returning empty is
 * the load-bearing part of the design, not a stub (see index.ts's register(),
 * which skips a backend contributing no models).
 *
 * What it is here for: on 32 GiB of VRAM across two 5060 Tis, a video model and
 * a 27B coding model are mutually exclusive. MiniMax H3's DiT alone is 20.97 GB
 * against 15.5 GiB usable per card. Before this, switching between them meant
 * finding the right terminal and remembering which of three servers was holding
 * the GPUs. The panel already knows how to stop llama.cpp and exl3; teaching it
 * the fourth process is what makes "one platform for all the backends" true.
 *
 * ComfyUI is also the only backend here with a UI of its own. `endpoint` is a
 * URL you open rather than one opencode calls, which is why status() reports the
 * bare origin instead of a /v1 base.
 */

const STATE = stateDir(Server.BACKEND)
const PID_FILE = path.join(STATE, "server.pid")
const LOG_FILE = path.join(STATE, "server.log")

const PROBE_TIMEOUT = 1_500
/**
 * ComfyUI imports torch and walks every node package before it binds the port;
 * on this box that is 20-40s cold, and a custom_nodes tree makes it longer.
 * Giving up early reads as "no server here" and the panel offers [start] against
 * a process that is already coming up.
 */
const START_TIMEOUT = 240_000
const POLL_INTERVAL = 1_000

async function executable(file: string): Promise<boolean> {
  const stat = await fs.stat(file).catch(() => undefined)
  return !!stat?.isFile()
}

/**
 * ComfyUI's status endpoint. Returns per-device VRAM, which is the whole reason
 * the panel wants it: `vram_free` here is the number that decides whether the
 * coding model still fits alongside whatever the last render left resident.
 */
type Device = {
  name: string
  type: string
  index: number
  vram_total: number
  vram_free: number
  torch_vram_total: number
  torch_vram_free: number
}
type Stats = { system?: { comfyui_version?: string }; devices?: Device[] }

async function stats(origin: string, timeout: number): Promise<Stats | undefined> {
  try {
    const res = await fetch(`${origin}/system_stats`, { signal: AbortSignal.timeout(timeout) })
    if (!res.ok) return undefined
    const body: any = await res.json()
    if (!body || typeof body !== "object") return undefined
    return body as Stats
  } catch {
    return undefined
  }
}

async function ready(origin: string, timeout: number): Promise<boolean> {
  return (await stats(origin, timeout)) !== undefined
}

/** How much work is outstanding, for the panel's detail line. */
async function queueDepth(origin: string, timeout: number): Promise<number | undefined> {
  try {
    const res = await fetch(`${origin}/prompt`, { signal: AbortSignal.timeout(timeout) })
    if (!res.ok) return undefined
    const body: any = await res.json()
    const remaining = body?.exec_info?.queue_remaining
    return typeof remaining === "number" ? remaining : undefined
  } catch {
    return undefined
  }
}

/**
 * The python process launched from `comfy-dir`, found by scanning /proc.
 *
 * Same reason as exl3's and OpenVINO's: the pid file only covers servers we
 * launched, and a ComfyUI someone started by hand in a terminal is still a
 * ComfyUI holding both GPUs. main.py's full path is the discriminator.
 */
async function pidForComfy(comfyDir: string): Promise<number | undefined> {
  const needle = path.join(comfyDir, "main.py")
  const entries = await fs.readdir("/proc").catch(() => [] as string[])
  for (const entry of entries) {
    const pid = Number.parseInt(entry, 10)
    if (!Number.isFinite(pid) || pid <= 0) continue
    const raw = await fs.readFile(`/proc/${pid}/cmdline`, "utf8").catch(() => "")
    if (raw.includes(needle)) return pid
  }
  return undefined
}

/** SIGTERM, wait out the CUDA teardown, then SIGKILL. */
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

const GiB = 1024 ** 3
const gib = (bytes: number) => `${(bytes / GiB).toFixed(1)}G`

export function create(): Backend {
  let settings: Server.ServerSettings | undefined
  let starting: Promise<ProviderStatus> | undefined
  /**
   * The model named by the most recent load in the log, and the step counter
   * while sampling runs. ComfyUI's REST surface cannot answer either: it has no
   * "what is loaded" endpoint, and the sampler's progress goes out over the
   * websocket to the browser. The log is the only place both appear in a form
   * this process can read without holding a socket open.
   */
  let lastModel: string | undefined
  let sampling: { step: number; total: number } | undefined

  const config = async () => (settings = await Server.load())
  const host = () => (settings?.host === "0.0.0.0" ? "127.0.0.1" : (settings?.host ?? "127.0.0.1"))
  const port = () => settings?.port ?? 8188

  const origin = () => {
    if (settings?.remote) return `http://${settings.remote}`
    return `http://${host()}:${port()}`
  }

  /** Same contract as the others: the address another machine would use. */
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

  async function serverAlive(cfg: Server.ServerSettings): Promise<boolean> {
    const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
    const pid = Number.parseInt(raw.trim(), 10)
    if (Number.isFinite(pid) && pid > 0) {
      const cmdline = await fs.readFile(`/proc/${pid}/cmdline`, "utf8").catch(() => "")
      if (cmdline.includes("main.py")) return true
    }
    return cfg.comfyDir ? (await pidForComfy(cfg.comfyDir)) !== undefined : false
  }

  /** Everything launch needs, or the first thing missing. */
  async function unconfigured(cfg: Server.ServerSettings): Promise<ProviderStatus | undefined> {
    if (!cfg.bin || !(await executable(cfg.bin))) {
      return {
        state: "unconfigured",
        missing: "binary",
        message: "python with torch not set",
        hint: `set bin in ${collapseHome(Server.FILE)}`,
      }
    }
    if (!cfg.comfyDir || !(await executable(path.join(cfg.comfyDir, "main.py")))) {
      return {
        state: "unconfigured",
        missing: "binary",
        message: "comfy-dir not set or has no main.py",
        hint: `set comfy-dir in ${collapseHome(Server.FILE)}`,
      }
    }
    return undefined
  }

  async function status(): Promise<ProviderStatus> {
    const cfg = await config()
    if (cfg.remote) {
      if (await ready(origin(), PROBE_TIMEOUT)) return { state: "running", endpoint: origin() }
      // Same call as exl3's remote: a ComfyUI on another machine is started
      // when someone wants to render, so "not answering" is its resting state
      // and reporting failed would be noise.
      return { state: "stopped" }
    }
    const missing = await unconfigured(cfg)
    if (missing) return missing
    if (await ready(origin(), PROBE_TIMEOUT)) {
      return { state: "running", endpoint: origin(), lan: lanAddress() }
    }
    // Alive but silent. ComfyUI serves the API from the same event loop that
    // runs VAE decode, and H3's video VAE is 5.2 GB — a decode blocks HTTP long
    // enough for a probe to time out. Reporting "stopped" mid-render is the bug
    // exl3 already paid for; do not repeat it here.
    if (await serverAlive(cfg)) {
      return { state: "running", endpoint: origin(), lan: lanAddress() }
    }
    return { state: "stopped" }
  }

  /**
   * Always empty, deliberately.
   *
   * ComfyUI has no chat endpoint, so there is nothing here opencode could send
   * a message to. register() skips a backend that returns no models, which is
   * exactly the wanted outcome: the panel supervises this server, and the model
   * picker never learns it exists. Returning a fake entry to make the row look
   * like the others would put a model in the picker that errors on first use.
   */
  async function models(): Promise<DiscoveredModel[]> {
    return []
  }

  async function launch(cfg: Server.ServerSettings): Promise<ProviderStatus> {
    await fs.mkdir(STATE, { recursive: true }).catch(() => {})
    const log = await fs.open(LOG_FILE, "a").catch(() => undefined)
    lastModel = undefined
    sampling = undefined
    try {
      const child = spawn(cfg.bin, Server.argv(cfg), {
        cwd: cfg.comfyDir,
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
      if (await ready(origin(), PROBE_TIMEOUT)) {
        return { state: "running", endpoint: origin(), lan: lanAddress() }
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
      if (cfg.remote) return status()
      const missing = await unconfigured(cfg)
      if (missing) return missing
      if (await ready(origin(), PROBE_TIMEOUT)) {
        return { state: "running", endpoint: origin(), lan: lanAddress() } as ProviderStatus
      }
      return launch(cfg)
    })()
    try {
      return await starting
    } finally {
      starting = undefined
    }
  }

  async function stop(): Promise<boolean> {
    const cfg = await config()
    if (cfg.remote) return false
    const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
    const recorded = Number.parseInt(raw.trim(), 10)
    let pid: number | undefined
    if (Number.isFinite(recorded) && recorded > 0) {
      const cmdline = await fs.readFile(`/proc/${recorded}/cmdline`, "utf8").catch(() => "")
      // never signal a pid that has been recycled into something else
      if (cmdline.includes("main.py")) pid = recorded
      else await fs.rm(PID_FILE, { force: true }).catch(() => {})
    }
    if (pid === undefined && cfg.comfyDir) pid = await pidForComfy(cfg.comfyDir)
    if (pid === undefined) return false
    const killed = await terminate(pid)
    await fs.rm(PID_FILE, { force: true }).catch(() => {})
    lastModel = undefined
    sampling = undefined
    return killed
  }

  /**
   * What the server is holding.
   *
   * There is no model id to report the way TabbyAPI reports one, so the honest
   * answer is what the GPUs actually contain: torch's allocation per device, the
   * free VRAM beside it, and whether anything is queued. That is also the answer
   * you need to decide whether to stop this before starting the coding model.
   */
  async function loaded(): Promise<LoadedModel | undefined> {
    const cfg = await config()
    const info = await stats(origin(), PROBE_TIMEOUT)
    if (!info) {
      if (cfg.remote) return undefined
      // Coming up, or blocked in a decode — either way, not gone.
      if (!(await serverAlive(cfg))) return undefined
      return { id: lastModel ?? "comfyui", args: { engine: "comfyui" }, loading: true, stage: "starting" }
    }

    const devices = (info.devices ?? []).filter((device) => device.type === "cuda")
    const args: Record<string, string> = { engine: "comfyui" }
    if (info.system?.comfyui_version) args["version"] = info.system.comfyui_version
    // Per-card, because the whole point on this box is which card has room left.
    for (const device of devices) {
      args[`cuda:${device.index}`] =
        `${gib(device.vram_total - device.vram_free)}/${gib(device.vram_total)}` +
        (device.torch_vram_total > 0 ? ` (torch ${gib(device.torch_vram_total)})` : "")
    }
    const depth = await queueDepth(origin(), PROBE_TIMEOUT)
    if (depth !== undefined && depth > 0) args["queue"] = String(depth)
    if (cfg.remote) args["host"] = cfg.remote

    const busy = sampling && sampling.total > 0
    return {
      id: lastModel ?? "comfyui",
      args,
      loading: busy || undefined,
      progress: busy ? Math.min(1, sampling!.step / sampling!.total) : undefined,
      stage: busy ? `sampling ${sampling!.step}/${sampling!.total}` : undefined,
    }
  }

  /**
   * Live progress, read from the server log.
   *
   * ComfyUI narrates loads ("Requested to load X" / "loaded completely") and
   * draws a tqdm bar per sampling run. The websocket carries the same thing in a
   * cleaner form, but holding a socket open from the panel process means
   * reconnect logic for a server that restarts far more often than the panel
   * does — the log tail is what the other backends already do, and it survives a
   * panel restart for free.
   */
  function watch(onEvent: (event: LoadEvent) => void): () => void {
    let stopped = false
    let offset = -1 // -1 = start from the end, so old runs are not replayed

    const emit = (event: Partial<LoadEvent> & { loading: boolean }) =>
      onEvent({ model: lastModel ?? "comfyui", ...event } as LoadEvent)

    const scanLine = (line: string) => {
      // "Requested to load MiniMaxH3" — the closest thing to a model id here.
      const requested = line.match(/Requested to load\s+(\S+)/)
      if (requested) {
        lastModel = requested[1]
        return emit({ loading: true, stage: "loading weights" })
      }
      // "loaded completely 9.5 20971.5 True" — weights are resident.
      if (line.includes("loaded completely") || line.includes("loaded partially")) {
        const partial = line.includes("partially")
        return emit({
          loading: false,
          loaded: true,
          // Worth surfacing: partial means ComfyUI spilled layers to system RAM
          // and every sampling step now crosses PCIe. On GPU1's x4 link that is
          // the difference between minutes and tens of minutes per clip.
          stage: partial ? "loaded (partial — spilling to RAM)" : undefined,
        })
      }
      // tqdm sampling bar: " 8%|▊         | 4/50 [00:12<02:18,  3.01s/it]"
      const step = line.match(/(\d+)%\|.*?\|\s*(\d+)\/(\d+)/)
      if (step) {
        const done = Number(step[2])
        const total = Number(step[3])
        if (total > 0) {
          sampling = { step: done, total }
          return emit({ loading: true, progress: Math.min(1, done / total), stage: `sampling ${done}/${total}` })
        }
      }
      if (line.includes("Prompt executed in")) {
        sampling = undefined
        return emit({ loading: false, loaded: true })
      }
      // OOM is the failure that actually happens here, and it is worth naming:
      // on 16 GiB cards it lands at the VAE decode after sampling has finished.
      if (line.includes("torch.OutOfMemoryError") || line.includes("CUDA out of memory")) {
        sampling = undefined
        return emit({ loading: false, failed: true, stage: "out of VRAM" })
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
                // tqdm rewrites its bar with \r, so split on both or every
                // sampling update arrives as one enormous line at the end.
                const lines = (carry + buffer.toString("utf8")).split(/[\r\n]/)
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

  return {
    id: Server.BACKEND,
    name: "ComfyUI",
    // Never reaches opencode's provider list — models() is always empty — but
    // the interface requires it and the setup screen prints it.
    providerName: "Localhost-ComfyUI",
    status,
    models,
    start,
    stop,
    loaded,
    watch,
    // Not an OpenAI base URL: this is the address you open in a browser. Kept
    // truthful rather than suffixed with /v1 to look like the others.
    baseURL: () => origin(),
    apiKey: () => undefined,
  }
}
