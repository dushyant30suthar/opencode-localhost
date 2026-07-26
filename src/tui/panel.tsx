import { createMemo, createSignal, createEffect, onCleanup, For, Show } from "solid-js"
import type { BackendPanel, GpuStat, PanelData, ProviderStatus, SystemStat } from "../shared/types.ts"
import { gpus } from "./hardware/nvidia.ts"
import { system } from "./hardware/system.ts"
import { BACKENDS } from "./backends.ts"

/**
 * Three sections: hardware, provider, model.
 *
 * Hardware is one fixed block because the GPUs are shared no matter how many
 * engines are installed. PROVIDER is the section that multiplies — one entry
 * per configured engine. MODEL describes what is actually loaded, naming the
 * engine only when more than one holds something.
 *
 * Rows are padded explicitly rather than laid out with flex: in a monospace
 * grid padStart/padEnd is what actually makes the columns line up.
 */

const POLL_MS = 2_000
/** While weights stream in, 2s makes the bar jump in visible steps. */
const POLL_LOADING_MS = 500
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

function DeviceRow(props: { theme: Theme; label: string; stat: GpuStat | SystemStat }) {
  const used = () => props.stat.usedMiB
  const total = () => props.stat.totalMiB
  return (
    <box flexDirection="row">
      <text fg={props.theme.textMuted}>{props.label.padEnd(5)}</text>
      <text fg={memoryColor(props.theme, used(), total())}>{`${gib(used())}/${gib(total())}G`.padStart(14)}</text>
      <text fg={props.theme.text}>
        {(props.stat.utilization === undefined ? "—" : `${props.stat.utilization}%`).padStart(9)}
      </text>
    </box>
  )
}

/** Rows the hardware grid renders, so the divider can match its height. */
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
          <text fg={props.theme.textMuted}>{"HARDWARE".padEnd(8)}</text>
          <text fg={props.theme.textMuted}>{"memory".padStart(11)}</text>
          <text fg={props.theme.textMuted}>{"compute".padStart(9)}</text>
        </box>
        <For each={rows()}>{(row) => <DeviceRow theme={props.theme} label={row.label} stat={row.stat} />}</For>
      </box>
    </Show>
  )
}

function endpointOf(status: ProviderStatus) {
  if (status.state !== "running") return undefined
  return status.endpoint.replace(/^https?:\/\//, "").replace(/\/v1$/, "")
}

/** Two lines per engine: name, then state or endpoint, then the control. */
function ProviderEntry(props: { theme: Theme; backend: BackendPanel; busy?: boolean; onToggle?: (id: string) => void }) {
  const status = () => props.backend.status
  const actionable = () => status().state === "running" || status().state === "stopped"
  return (
    <box flexDirection="column">
      <text fg={props.theme.text} wrapMode="word">
        {props.backend.name}
      </text>
      <box flexDirection="row">
        <text fg={statusColor(props.theme, status().state)}>{`${GLYPH[status().state]} `}</text>
        <text fg={props.theme.textMuted} wrapMode="none">
          {endpointOf(status()) ?? statusWord(status())}
        </text>
      </box>
      <Show when={"message" in status()}>
        <text fg={props.theme.textMuted} wrapMode="word">
          {(status() as { message: string }).message}
        </text>
      </Show>
      <Show when={props.onToggle && actionable()}>
        <text
          fg={props.busy ? props.theme.textMuted : props.theme.primary}
          wrapMode="none"
          onMouseUp={() => !props.busy && props.onToggle?.(props.backend.id)}
        >
          {props.busy ? "…working" : status().state === "running" ? "[stop]" : "[start]"}
        </text>
      </Show>
    </box>
  )
}

export function Provider(props: {
  theme: Theme
  data: PanelData
  stacked?: boolean
  busy?: string
  onToggle?: (id: string) => void
}) {
  return (
    <box flexDirection="column">
      <Show when={props.stacked}>
        <text fg={props.theme.textMuted}>PROVIDER</text>
      </Show>
      <Show
        when={props.data.backends.length > 0}
        fallback={
          <text fg={props.theme.textMuted} wrapMode="word">
            nothing configured — /localhost
          </text>
        }
      >
        <For each={props.data.backends}>
          {(backend) => (
            <ProviderEntry
              theme={props.theme}
              backend={backend}
              busy={props.busy === backend.id}
              onToggle={props.onToggle}
            />
          )}
        </For>
      </Show>
    </box>
  )
}

/** A tenth-width bar: enough to read at a glance, cheap in a 25-column column. */
function bar(fraction: number) {
  const filled = Math.max(0, Math.min(10, Math.round(fraction * 10)))
  return "█".repeat(filled) + "░".repeat(10 - filled)
}

function detailOf(backend: BackendPanel, throughput?: number) {
  const model = backend.loaded
  if (!model) return []
  // while weights stream in there are no launch flags to report yet, and the
  // thing worth showing is how far along it is
  if (model.loading) {
    const pct = model.progress === undefined ? undefined : Math.round(model.progress * 100)
    const line = pct === undefined ? "loading…" : `${bar(model.progress!)} ${pct}%`
    return [line, model.stage ?? ""].filter(Boolean)
  }
  const args = model.args
  const first: string[] = []
  if (args["ctx-size"]) first.push(`${args["ctx-size"]} ctx`)
  if (args["cache-type-k"]) first.push(`${args["cache-type-k"]} KV`)
  const second: string[] = []
  if (args["gpu-layers"]) second.push(`ngl ${args["gpu-layers"]}`)
  if (args["tensor-split"]) second.push(`split ${args["tensor-split"]}`)
  if (throughput) second.push(`${throughput.toFixed(1)} tok/s`)
  return [first.join(" · "), second.join(" · ")].filter(Boolean)
}

export function Model(props: { theme: Theme; data: PanelData; stacked?: boolean; onChange?: () => void }) {
  const withModel = createMemo(() => props.data.backends.filter((backend) => backend.loaded))
  // the engine name is only worth repeating when more than one holds something
  const grouped = createMemo(() => withModel().length > 1)
  return (
    <box flexDirection="column">
      <Show when={props.stacked}>
        <text fg={props.theme.textMuted}>MODEL</text>
      </Show>
      <Show
        when={withModel().length > 0}
        fallback={
          <text fg={props.theme.textMuted} wrapMode="none">
            no model loaded
          </text>
        }
      >
        <For each={withModel()}>
          {(backend) => (
            <box flexDirection="column">
              <Show when={grouped()}>
                <text fg={props.theme.textMuted} wrapMode="none">
                  {backend.name}
                </text>
              </Show>
              {/* ids carry a publisher prefix; wrap rather than truncate */}
              <text
                fg={backend.loaded!.loading ? props.theme.warning : props.theme.text}
                wrapMode="word"
                onMouseUp={() => props.onChange?.()}
              >
                {backend.loaded!.loading ? `◐ ${backend.loaded!.id}` : backend.loaded!.id}
              </text>
              <For each={detailOf(backend, props.data.throughput)}>
                {(line) => (
                  <text fg={props.theme.textMuted} wrapMode="word">
                    {line}
                  </text>
                )}
              </For>
            </box>
          )}
        </For>
      </Show>
      <Show when={props.onChange}>
        <text fg={props.theme.primary} wrapMode="none" onMouseUp={() => props.onChange?.()}>
          [change]
        </text>
      </Show>
    </box>
  )
}

/** Polls only while mounted, so a hidden panel costs nothing. */
export function usePanelData(isRegistered?: () => boolean) {
  const [data, setData] = createSignal<PanelData>({ backends: [], gpus: [] })

  async function refresh() {
    const [gpuStats, sysStat] = await Promise.all([gpus().catch(() => []), system().catch(() => undefined)])
    const backends = await Promise.all(
      BACKENDS.map(async (backend) => {
        const status = await backend.status().catch<ProviderStatus>(() => ({ state: "stopped" }))
        const loaded = status.state === "running" ? await backend.loaded().catch(() => undefined) : undefined
        return { id: backend.id, name: backend.name, status, loaded }
      }),
    )
    // an engine with no binary or no models directory is not configured;
    // carrying it on screen forever is noise, and /localhost lists them all
    setData({
      backends: backends.filter((backend) => backend.status.state !== "unconfigured"),
      gpus: gpuStats,
      memory: sysStat,
      registered: isRegistered?.() ?? true,
    })
  }

  // poll faster while something is loading, so the bar moves smoothly, and
  // drop back afterwards so an idle panel costs almost nothing
  createEffect(() => {
    const loading = data().backends.some((backend) => backend.loaded?.loading)
    void refresh()
    const timer = setInterval(() => void refresh(), loading ? POLL_LOADING_MS : POLL_MS)
    onCleanup(() => clearInterval(timer))
  })

  return data
}
