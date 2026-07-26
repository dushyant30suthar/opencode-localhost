import { createMemo, createSignal, createEffect, onCleanup, For, Show } from "solid-js"
import type { GpuStat, PanelData, ProviderStatus, SystemStat } from "../shared/types.ts"
import { gpus } from "./hardware/nvidia.ts"
import { system } from "./hardware/system.ts"
import * as Server from "../server/llamacpp/server-ini.ts"
import { collapseHome } from "../shared/paths.ts"

/**
 * The panel: hardware, then provider, then model.
 *
 * Rows are built as fixed-width cells and padded explicitly rather than laid
 * out with flex, because in a monospace grid padStart/padEnd is what actually
 * guarantees the columns line up.
 */

const POLL_MS = 2_000
const MIB_PER_GIB = 1024

type Theme = Record<string, any>

function gib(mib: number) {
  return (mib / MIB_PER_GIB).toFixed(1)
}

/** Memory is an alarm when it is nearly full; compute never is. */
function memoryColor(theme: Theme, used: number, total: number) {
  if (total <= 0) return theme.text
  const ratio = used / total
  if (ratio >= 0.97) return theme.error
  if (ratio >= 0.9) return theme.warning
  return theme.text
}

const GLYPH: Record<ProviderStatus["state"], string> = {
  running: "●",
  starting: "◐",
  stopped: "○",
  failed: "✕",
  unconfigured: "✕",
}

function statusColor(theme: Theme, state: ProviderStatus["state"]) {
  if (state === "running") return theme.success
  if (state === "starting") return theme.warning
  if (state === "failed" || state === "unconfigured") return theme.error
  return theme.textMuted
}

/** The word always accompanies the glyph, so state never depends on colour. */
function statusWord(status: ProviderStatus) {
  switch (status.state) {
    case "running":
      return "running"
    case "starting":
      return "starting"
    case "stopped":
      return "stopped"
    case "failed":
      return "failed"
    case "unconfigured":
      return status.missing === "binary" ? "no binary" : "not set up"
  }
}

function DeviceRow(props: {
  theme: Theme
  label: string
  stat: GpuStat | SystemStat
  labelWidth: number
  memWidth: number
  pctWidth: number
}) {
  const used = () => props.stat.usedMiB
  const total = () => props.stat.totalMiB
  return (
    <box flexDirection="row">
      <text fg={props.theme.textMuted}>{props.label.padEnd(props.labelWidth)}</text>
      <text fg={memoryColor(props.theme, used(), total())}>
        {`${gib(used())}/${gib(total())}G`.padStart(props.memWidth)}
      </text>
      <text fg={props.theme.text}>
        {(props.stat.utilization === undefined ? "—" : `${props.stat.utilization}%`).padStart(props.pctWidth)}
      </text>
    </box>
  )
}

/** Rows the hardware grid will render, so the divider can match its height. */
export function hardwareRows(data: PanelData) {
  return data.gpus.length + (data.memory ? 1 : 0)
}

export function Hardware(props: { theme: Theme; data: PanelData }) {
  const rows = createMemo(() => {
    const out: { label: string; stat: GpuStat | SystemStat }[] = []
    for (const gpu of props.data.gpus) out.push({ label: `GPU${gpu.index}`, stat: gpu })
    if (props.data.memory) out.push({ label: "CPU", stat: props.data.memory })
    return out
  })
  return (
    <Show when={rows().length > 0}>
      <box flexDirection="column">
        <box flexDirection="row">
          <text fg={props.theme.textMuted}>{"HARDWARE".padEnd(9)}</text>
          <text fg={props.theme.textMuted}>{"memory".padStart(11)}</text>
          <text fg={props.theme.textMuted}>{"compute".padStart(10)}</text>
        </box>
        <For each={rows()}>
          {(row) => (
            <DeviceRow theme={props.theme} label={row.label} stat={row.stat} labelWidth={6} memWidth={14} pctWidth={10} />
          )}
        </For>
      </box>
    </Show>
  )
}

export function Provider(props: { theme: Theme; data: PanelData; stacked?: boolean }) {
  const status = () => props.data.status
  return (
    <box flexDirection="column">
      <Show when={props.stacked}>
        <text fg={props.theme.textMuted}>PROVIDER</text>
      </Show>
      <text fg={props.theme.text} wrapMode="none">
        {props.data.backend.name}
      </text>
      <box flexDirection="row">
        <text fg={statusColor(props.theme, status().state)}>{`${GLYPH[status().state]} `}</text>
        <text fg={props.theme.text} wrapMode="none">
          {statusWord(status())}
        </text>
      </box>
      <Show when={status().state === "running" && "endpoint" in status()}>
        <text fg={props.theme.textMuted} wrapMode="none">
          {(status() as { endpoint: string }).endpoint.replace(/^https?:\/\//, "").replace(/\/v1$/, "")}
        </text>
      </Show>
      <Show when={"message" in status()}>
        <text fg={props.theme.textMuted} wrapMode="none">
          {(status() as { message: string }).message}
        </text>
      </Show>
      {/* the file path is only useful where there is room to read it */}
      <Show when={status().state === "running" && props.data.registered === false}>
        <text fg={props.theme.warning} wrapMode="none">
          restart opencode
        </text>
      </Show>
      <Show when={props.stacked && "hint" in status() && (status() as { hint?: string }).hint}>
        <text fg={props.theme.textMuted} wrapMode="word">
          {(status() as { hint: string }).hint}
        </text>
      </Show>
    </box>
  )
}

export function Model(props: { theme: Theme; data: PanelData; stacked?: boolean }) {
  const loaded = () => (props.data.status.state === "running" ? props.data.status.loaded : undefined)
  const detail = createMemo(() => {
    const model = loaded()
    if (!model) return []
    const args = model.args
    const parts: string[] = []
    if (args["ctx-size"]) parts.push(`${args["ctx-size"]} ctx`)
    if (args["cache-type-k"]) parts.push(`${args["cache-type-k"]} KV`)
    const second: string[] = []
    if (args["gpu-layers"]) second.push(`ngl ${args["gpu-layers"]}`)
    if (args["tensor-split"]) second.push(`split ${args["tensor-split"]}`)
    if (props.data.throughput) second.push(`${props.data.throughput.toFixed(1)} tok/s`)
    return [parts.join(" · "), second.join(" · ")].filter(Boolean)
  })
  return (
    <box flexDirection="column">
      <Show when={props.stacked}>
        <text fg={props.theme.textMuted}>MODEL</text>
      </Show>
      <text fg={props.theme.text} wrapMode="none">
        {loaded()?.id ?? "no model loaded"}
      </text>
      <For each={detail()}>
        {(line) => (
          <text fg={props.theme.textMuted} wrapMode="none">
            {line}
          </text>
        )}
      </For>
    </box>
  )
}

/** Polls only while mounted, so a hidden panel costs nothing. */
export function usePanelData(isRegistered?: () => boolean) {
  const [data, setData] = createSignal<PanelData>({
    backend: { id: "llamacpp", name: "llama.cpp" },
    status: { state: "stopped" },
    gpus: [],
  })

  async function refresh() {
    const [gpuStats, sysStat] = await Promise.all([gpus().catch(() => []), system().catch(() => undefined)])
    const status = await probe().catch<ProviderStatus>(() => ({ state: "stopped" }))
    setData((current) => ({
      ...current,
      gpus: gpuStats,
      memory: sysStat,
      status,
      registered: isRegistered?.() ?? true,
    }))
  }

  createEffect(() => {
    void refresh()
    const timer = setInterval(() => void refresh(), POLL_MS)
    onCleanup(() => clearInterval(timer))
  })

  return data
}

/**
 * The TUI half reads the backend directly rather than asking opencode, because
 * a server plugin cannot expose endpoints for it to ask. Documented limitation:
 * this breaks if the TUI ever runs on a different machine from the server.
 */
async function probe(): Promise<ProviderStatus> {
  const cfg = await Server.load()
  if (!cfg.modelsDir) {
    return {
      state: "unconfigured",
      missing: "models-dir",
      message: "models-dir not set",
      hint: `set it in ${collapseHome(Server.FILE)}`,
    }
  }
  const host = cfg.host === "0.0.0.0" ? "127.0.0.1" : cfg.host
  const baseURL = `http://${host}:${cfg.port}/v1`
  try {
    const res = await fetch(`${baseURL}/models`, {
      signal: AbortSignal.timeout(1_500),
      headers: cfg.apiKey ? { Authorization: `Bearer ${cfg.apiKey}` } : {},
    })
    if (!res.ok) return { state: "stopped" }
    const body: any = await res.json()
    const entries: any[] = Array.isArray(body?.data) ? body.data : []
    const active = entries.find((entry) => entry?.status?.args || entry?.status?.status === "loading")
    if (!active) return { state: "running", endpoint: baseURL }
    return {
      state: "running",
      endpoint: baseURL,
      loaded: {
        id: String(active.id ?? ""),
        args: argsOf(active?.status?.args),
        loading: active?.status?.status === "loading",
      },
    }
  } catch {
    return { state: "stopped" }
  }
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
