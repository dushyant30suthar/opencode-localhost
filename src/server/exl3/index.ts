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

/**
 * Servable-and-which, from /v1/model.
 *
 * NOT /health, despite that being the obvious endpoint. TabbyAPI's /health
 * blocks behind the generation queue: measured 2026-08-01 on a busy server it
 * took 10.3-16.3s to answer while /v1/model returned in 0.34s. Against a
 * 1.5s probe budget that meant every poll timed out exactly when the server was
 * working hardest, so the panel showed "stopped / no model loaded" through an
 * entire live session — GPUs pinned at 100% and the UI insisting nothing ran.
 *
 * /v1/model is also strictly more informative: 200 with an id means loaded and
 * says which, and it reports "No models are currently loaded" during the tens
 * of seconds a checkpoint is streaming into VRAM, which is the answer start()
 * wants while polling anyway.
 */
type ModelInfo = { id: string; params: Record<string, unknown> }

async function modelInfo(origin: string, timeout: number, apiKey?: string): Promise<ModelInfo | undefined> {
  try {
    const res = await fetch(`${origin}/v1/model`, {
      signal: AbortSignal.timeout(timeout),
      headers: apiKey ? { "x-api-key": apiKey } : {},
    })
    if (!res.ok) return undefined
    const body: any = await res.json()
    if (typeof body?.id !== "string") return undefined
    return { id: body.id, params: body?.parameters ?? {} }
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
 * The id the server is actually serving right now, or undefined.
 *
 * Was reading /v1/models and taking data[0] — but that endpoint enumerates
 * every checkpoint under model_dir, loaded or not, so data[0] is simply
 * whichever sorts first. With three directories present it reported the wrong
 * model whenever the loaded one was not alphabetically first. /v1/model
 * (singular) is the one that answers "what is loaded".
 */
const servedModelFrom = servedModel

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

/**
 * Resolve once nothing is listening on the port and no TabbyAPI process is
 * left, or when the deadline passes.
 *
 * Separate from terminate() because they answer different questions: terminate
 * waits for the PARENT to exit, this waits for the port and the tensor-parallel
 * workers it spawned. A launch that only waits for the former lands on top of a
 * server that is still holding VRAM.
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
  /**
   * The YAML the running server was launched with. TabbyAPI serves one model
   * per process, so this is also "which model is loaded" — and it has to be
   * remembered rather than derived, because after a launch the only other way
   * to know is to ask the server, which is unavailable for the tens of seconds
   * a checkpoint takes to stream into VRAM.
   */
  let launchedWith: string | undefined
  /**
   * Last answer from a live server, and when.
   *
   * The panel polls while the server is working, and TabbyAPI stops answering
   * HTTP for the duration of a deep prefill — a 98k prompt at ~525 T/s is around
   * three minutes of silence. Without this the panel blanks out mid-session:
   * provider "stopped", "no model loaded", every field empty, while both GPUs
   * sit at 100%. Reporting the last known good answer while the process is
   * still alive is far closer to the truth than reporting nothing.
   *
   * Deliberately not time-limited beyond the process check: the thing that
   * invalidates it is the server going away, and stop()/launch() clear it
   * directly.
   */
  let lastSeen: { model: LoadedModel; at: number } | undefined

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

  /**
   * Is the server process we launched still there?
   *
   * Distinct from `ready()`: this asks whether the process exists, not whether
   * it can answer. The two diverge for minutes at a time while TabbyAPI is
   * prefilling, which is exactly when the panel must not claim it is stopped.
   */
  async function serverAlive(cfg: Server.ServerSettings): Promise<boolean> {
    const raw = await fs.readFile(PID_FILE, "utf8").catch(() => "")
    const pid = Number.parseInt(raw.trim(), 10)
    if (Number.isFinite(pid) && pid > 0) {
      const cmdline = await fs.readFile(`/proc/${pid}/cmdline`, "utf8").catch(() => "")
      if (cmdline.includes("main.py")) return true
    }
    // a server started outside this process still counts as running
    return cfg.tabbyDir ? (await pidForTabby(cfg.tabbyDir)) !== undefined : false
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
        hint: `add a TabbyAPI YAML to ${collapseHome(ModelsDir.DIR)}`,
      }
    }
    return undefined
  }

  /**
   * Which YAML to launch. The `config` override wins; otherwise the selected
   * model's file, falling back to the first listed so a fresh install starts
   * something rather than refusing until a model is picked.
   */
  async function yamlFor(cfg: Server.ServerSettings, id?: string): Promise<string | undefined> {
    if (cfg.config) return cfg.config
    if (id) {
      const file = await ModelsDir.fileFor(id)
      if (file) return file
    }
    return (await ModelsDir.scan())[0]?.file
  }

  async function status(): Promise<ProviderStatus> {
    const cfg = await config()
    if (cfg.remote) {
      if (await ready(origin(), PROBE_TIMEOUT, cfg.apiKey)) {
        return { state: "running", endpoint: baseURL() }
      }
      // "stopped", not "failed" — a deliberate divergence from OpenVINO's
      // remote. TabbyAPI is not an always-on router: the far machine starts it
      // when a session wants EXL3, so "not answering right now" is its normal
      // resting state. Reporting failed would make register() skip the provider
      // and the model would vanish from the picker on this machine — the far
      // server being down is exactly when you want to see what starting it
      // would offer.
      return { state: "stopped" }
    }
    const missing = await unconfigured(cfg)
    if (missing) return missing
    if (await ready(origin(), PROBE_TIMEOUT, cfg.apiKey)) {
      return { state: "running", endpoint: baseURL(), lan: lanAddress() }
    }
    // Silent but alive — a deep prefill holds TabbyAPI's HTTP for minutes.
    // Reporting "stopped" here is what made the panel drop the provider,
    // the model and every field mid-session with both GPUs saturated.
    if (await serverAlive(cfg)) {
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

  /** Shared shaping so every path advertises a model the same way. */
  function describe(id: string, context: number, remote: boolean): DiscoveredModel {
    return {
      id,
      // Drop only the "-exl3" marker — the format is already implied by the
      // provider name in the picker. Everything AFTER it stays: on this backend
      // the bitrate is usually the sole difference between two entries, and
      // stripping to the tail collapsed "…-exl3-5.00bpw" and "…-exl3-6.00bpw"
      // into the same unusable "Qwen3.6-27B".
      name: id.replace(/-exl3(?=-|$)/i, "").replace(/-+/g, " ").trim(),
      context,
      // thinking models spend the reasoning budget from the output allowance
      output: Math.min(32_768, Math.max(4_096, Math.floor(context / 2))),
      // a remote applies its own sampling; overriding from here fights it
      sampling: remote ? {} : settings!.sampling,
    }
  }

  async function models(): Promise<DiscoveredModel[]> {
    const cfg = await config()
    if (!cfg.remote) {
      const missing = await unconfigured(cfg)
      if (missing) return []
    }

    // Local, models/ in use: list every YAML. Only one can be SERVED at a time,
    // but all of them are selectable — picking one relaunches (see ensure).
    // Each file's own max_seq_len is the truthful window, so a model with a
    // smaller cache does not get opencode compacting against the wrong number.
    if (!cfg.remote && !cfg.config) {
      const declared = await ModelsDir.scan()
      if (declared.length > 0) {
        return declared.map((model) => describe(model.id, model.context ?? cfg.context, false))
      }
    }

    const served = await servedModelFrom(origin(), PROBE_TIMEOUT, cfg.apiKey)
    // A remote that is not answering still advertises `model` from the ini —
    // the far end has no always-on router to ask, and a provider that vanishes
    // whenever the far server rests reads as "exl3 not acknowledged" on the
    // machine pointing at it.
    const id = served ?? (cfg.remote ? cfg.model : await configuredModelName(cfg))
    if (!id) return []
    return [describe(id, cfg.context, !!cfg.remote)]
  }

  async function launch(cfg: Server.ServerSettings, yaml: string): Promise<ProviderStatus> {
    await fs.mkdir(STATE, { recursive: true }).catch(() => {})
    const log = await fs.open(LOG_FILE, "a").catch(() => undefined)
    launchedWith = yaml
    lastSeen = undefined
    try {
      const child = spawn(cfg.bin, Server.argv({ ...cfg, config: yaml }), {
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
    launchedWith = undefined
    lastSeen = undefined
    return killed
  }

  async function loaded(): Promise<LoadedModel | undefined> {
    const cfg = await config()
    // The YAML actually launched, not server.ini's `config` — that key is the
    // single-model override and is normally blank now that models/ holds one
    // file per model, so reporting it left the panel with an empty field.
    const active = launchedWith ?? cfg.config
    const declared = active
      ? (await ModelsDir.scan()).find((model) => model.file === active)
      : undefined

    const info = await modelInfo(origin(), PROBE_TIMEOUT, cfg.apiKey)
    if (info) {
      const p = info.params
      const str = (key: string) => (p[key] === undefined || p[key] === null ? undefined : String(p[key]))
      // Everything worth showing comes from the server's own view of what it
      // loaded, so the panel cannot drift from reality the way a config-derived
      // listing does. llamacpp gets the equivalent by parsing llama-server's
      // launch argv; TabbyAPI hands it over as /v1/model parameters.
      const args: Record<string, string> = {
        engine: "exllamav3",
        context: str("max_seq_len") ?? String(declared?.context ?? cfg.context),
        cache: [str("cache_size"), str("cache_mode") && `mode ${str("cache_mode")}`]
          .filter(Boolean)
          .join(", "),
        slots: str("max_batch_size") ?? "",
        chunk: str("chunk_size") ?? "",
        draft: p["draft"] ? "model" : "mtp (built-in head)",
        config: collapseHome(active || ""),
      }
      for (const key of Object.keys(args)) if (!args[key]) delete args[key]
      const model: LoadedModel = { id: info.id, args: cfg.remote ? { host: cfg.remote } : args }
      lastSeen = { model, at: Date.now() }
      return model
    }

    if (cfg.remote) return undefined

    // Not answering. A deep prefill blocks TabbyAPI's HTTP for minutes, so
    // "silent" is not "gone" — if the process we launched is still alive, the
    // last good answer is the honest one.
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
    // Streaming weights into VRAM: the server cannot describe itself yet, so
    // the YAML about to be served is the only source. Fields the file declares
    // are real; the rest appear once /v1/model answers.
    const name = declared?.served ?? (await configuredModelName(cfg)) ?? "model"
    const pending: Record<string, string> = {
      engine: "exllamav3",
      context: String(declared?.context ?? cfg.context),
      config: collapseHome(active || ""),
    }
    return { id: name, args: pending, loading: true, stage: "loading weights" }
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
  async function start(id?: string): Promise<ProviderStatus> {
    if (starting) return starting
    starting = (async () => {
      const cfg = await config()
      if (cfg.remote) return status()
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
   * Make `id` the served model. TabbyAPI holds one per process, so switching is
   * stop-then-launch rather than an unload/load inside the running server.
   *
   * That is not laziness about a faster path — the faster path is broken here.
   * TabbyAPI's inline_model_loading does unload first, but exllamav3 does not
   * hand the VRAM back to the driver: measured 2026-08-01, swapping 6.00bpw ->
   * 5.00bpw in-process threw "Insufficient VRAM in split for model and cache"
   * and left the server with NO model loaded and 29.2 GiB still held, needing a
   * kill to recover. Ending the process is what frees it.
   *
   * Costs a full reload (~40s for a ~20 GiB checkpoint), so it returns early
   * whenever the wanted model is already up.
   */
  async function ensure(id: string): Promise<ProviderStatus> {
    const cfg = await config()
    // A remote decides its own model; asking it to switch is not ours to do.
    if (cfg.remote) return status()
    // The single-model override means there is nothing to choose between.
    if (cfg.config) return start()

    const wanted = (await ModelsDir.scan()).find((model) => model.id === id)
    // Unknown id: leave whatever is running alone rather than tearing down a
    // working server for a model this machine cannot serve.
    if (!wanted) return status()

    const up = await ready(origin(), PROBE_TIMEOUT, cfg.apiKey)
    if (up && launchedWith === wanted.file) {
      return { state: "running", endpoint: baseURL(), lan: lanAddress() }
    }
    // Up from a process whose choice we never recorded — this one, restarted,
    // or one started by hand. Ask the server what it holds before paying for a
    // reload. Only `served` can be compared: TabbyAPI reports the checkpoint's
    // model_name and knows nothing about which of our YAMLs launched it, so a
    // match here is necessary but not sufficient — two files can serve the same
    // checkpoint with different slots and drafters. Adopting on a served-name
    // match alone would silently keep the wrong one; requiring launchedWith for
    // the fast path and restarting otherwise is the safe direction to err.
    if (up && !launchedWith) {
      const served = await servedModelFrom(origin(), PROBE_TIMEOUT, cfg.apiKey)
      const only = (await ModelsDir.scan()).filter((m) => m.served === wanted.served)
      if (served === wanted.served && only.length === 1) {
        launchedWith = wanted.file
        return { state: "running", endpoint: baseURL(), lan: lanAddress() }
      }
    }
    if (up) {
      await stop()
      // Wait for the PORT, not just the parent process. terminate() waits out
      // the parent, but under tensor_parallel the TP workers outlive it and go
      // on holding both the socket and their share of VRAM. Launching into that
      // gives two servers competing for the same GPUs: observed 2026-08-01 as
      // 29.9 GiB held across two half-loaded processes and neither answering.
      // deep-confirm.sh guards the same way for the same reason.
      await released(host(), port(), START_TIMEOUT)
    }
    return start(id)
  }

  return {
    id: Server.BACKEND,
    name: "exllamav3",
    providerName: "Localhost-EXL3",
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
