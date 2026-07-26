import { For, Show } from "solid-js"
import { Hardware, Provider, Model, hardwareRows, usePanelData } from "./panel.tsx"

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
 * Columns are fixed and flexShrink is off. Without this the row collapses
 * under its own content in a narrow terminal and the three sections overprint
 * each other. Total is 74, matching opencode's default prompt width of 75.
 */
const HARDWARE_WIDTH = 30
const PROVIDER_WIDTH = 18
const MODEL_WIDTH = 23

/** One `│` per row, so the rule spans the whole block rather than its first line. */
function Divider(props: { color: string; rows: number }) {
  const lines = () => Array.from({ length: Math.max(1, props.rows) })
  return (
    <box width={3} flexShrink={0} flexDirection="column">
      <For each={lines()}>{() => <text fg={props.color}> │ </text>}</For>
    </box>
  )
}

function HomePanel(props: { api: any }) {
  const theme = () => props.api.theme.current
  const data = usePanelData()
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
      <box width={MODEL_WIDTH} flexShrink={0} flexDirection="column">
        <Model theme={theme()} data={data()} />
      </box>
    </box>
  )
}

function SidebarPanel(props: { api: any }) {
  const theme = () => props.api.theme.current
  const data = usePanelData()
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
