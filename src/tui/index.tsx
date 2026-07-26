import { createEffect, For, Show } from "solid-js"
import { Hardware, Provider, Model, hardwareRows, usePanelData } from "./panel.tsx"
import { openSetup } from "./setup.tsx"

/**
 * The TUI half.
 *
 * Two slots, one component, two arrangements:
 *   home_bottom      always on screen, so it is also where a fresh install is
 *                    told what is missing
 *   sidebar_content  stacked, during a session
 *
 * The hardware grid is the same width in both, so it is genuinely one
 * component rather than two layouts to keep in sync.
 */

/**
 * Hardware and provider are fixed and cannot shrink — without that the row
 * collapses under its own content and the sections overprint each other.
 *
 * MODEL takes whatever is left rather than a fixed width, and wraps: model ids
 * carry a publisher prefix ("lmstudio-community/Qwen3.6-35B-A3B-GGUF"), which a
 * fixed column truncated at exactly the identifying part. The strip keeps the
 * prompt's 75-column max so it stays centred with everything else on the home
 * screen — full width made it hug the left edge while the prompt stayed centred.
 */
const HARDWARE_WIDTH = 30
const PROVIDER_WIDTH = 18

/** One `│` per row, so the rule spans the whole block rather than its first line. */
function Divider(props: { color: string; rows: number }) {
  const lines = () => Array.from({ length: Math.max(1, props.rows) })
  return (
    <box width={3} flexShrink={0} flexDirection="column">
      <For each={lines()}>{() => <text fg={props.color}> │ </text>}</For>
    </box>
  )
}

/** opencode exposes the providers it loaded, which is how we detect the gap. */
function registered(api: any) {
  return () => (api.state?.provider ?? []).some((item: any) => item?.id === "llamacpp")
}

/**
 * Ask opencode to re-read its config, so a provider that was not ready at
 * startup appears without restarting.
 *
 * opencode's TUI binds SIGUSR2 to a config invalidate + instance dispose, and
 * a TUI plugin runs inside that same process — so it can signal itself. This
 * is the only reload path opencode exposes; there is no endpoint for it.
 *
 * Only fired from the home screen. The reload disposes live instances, which
 * is fine before you start working and rude in the middle of a session.
 */
function requestReload(api: any): boolean {
  if (api.route?.current?.name !== "home") return false
  try {
    process.kill(process.pid, "SIGUSR2")
    return true
  } catch {
    return false
  }
}

/**
 * Once the backend is up but opencode has not picked it up, reload — once.
 * Guarded because the reload is asynchronous: without it, every poll before
 * the provider list refreshes would fire another signal.
 */
function useAutoReload(api: any, data: () => { status: { state: string }; registered?: boolean }) {
  let fired = false
  createEffect(() => {
    const ready = data().status.state === "running"
    if (!ready || data().registered !== false) {
      // reset once opencode has caught up, so a later restart of the backend
      // can trigger exactly one more reload
      if (data().registered) fired = false
      return
    }
    if (fired) return
    fired = requestReload(api)
  })
}

function HomePanel(props: { api: any }) {
  const theme = () => props.api.theme.current
  const data = usePanelData(registered(props.api))
  useAutoReload(props.api, data)
  return (
    <box width="100%" maxWidth={75} flexDirection="row" flexShrink={0} paddingTop={1}>
      <box width={HARDWARE_WIDTH} flexShrink={0} flexDirection="column">
        <Hardware theme={theme()} data={data()} />
      </box>
      <Divider color={theme().border} rows={hardwareRows(data()) + 1} />
      <box width={PROVIDER_WIDTH} flexShrink={0} flexDirection="column">
        <Provider theme={theme()} data={data()} />
      </box>
      <Divider color={theme().border} rows={hardwareRows(data()) + 1} />
      <box flexGrow={1} minWidth={20} flexDirection="column">
        <Model theme={theme()} data={data()} />
      </box>
    </box>
  )
}

function SidebarPanel(props: { api: any }) {
  const theme = () => props.api.theme.current
  const data = usePanelData(registered(props.api))
  return (
    <box flexDirection="column">
      <Hardware theme={theme()} data={data()} />
      <box height={1} />
      <Provider theme={theme()} data={data()} stacked />
      <box height={1} />
      <Show when={data().status.state === "running"}>
        <Model theme={theme()} data={data()} stacked />
      </Show>
    </box>
  )
}

const tui = async (api: any) => {
  // Setup lives behind a command rather than in the strip: the strip answers
  // "is it working", this answers "where is everything".
  api.command?.register(() => [
    {
      title: "Local models: setup",
      value: "localhost.setup",
      category: "Provider",
      slash: { name: "localhost" },
      onSelect: () => openSetup(api),
    },
  ])

  api.slots.register({
    order: 350,
    slots: {
      home_bottom() {
        return <HomePanel api={api} />
      },
      sidebar_content() {
        return <SidebarPanel api={api} />
      },
    },
  })
}

export default { id: "opencode-localhost-tui", tui }
