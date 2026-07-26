import fs from "fs/promises"
import { BACKENDS, allOnPath, expand, supported, type BackendSpec } from "../shared/backends.ts"
import { collapseHome } from "../shared/paths.ts"
import * as Server from "../server/llamacpp/server-ini.ts"
import { create as llamacpp } from "../server/llamacpp/index.ts"

const backend = llamacpp()

/**
 * Setup, kept separate from the status strip.
 *
 * The strip answers "is it working right now". This answers "where is
 * everything" — a different question, asked at a different time, and cramming
 * both into three narrow columns is what made the unconfigured state unreadable.
 *
 * Built from DialogSelect and DialogPrompt rather than a hand-rolled screen, so
 * filtering, keyboard navigation and theming come from opencode itself.
 */

type Row = {
  title: string
  description?: string
  run?: () => void | Promise<void>
}

const OK = "✓"
const MISSING = "✗"

async function isFile(file: string): Promise<boolean> {
  const stat = await fs.stat(file).catch(() => undefined)
  return !!stat?.isFile()
}

async function backendRow(spec: BackendSpec, api: any, reopen: () => void): Promise<Row> {
  if (!supported(spec)) {
    return {
      title: `${MISSING} ${spec.name}`,
      description: "Apple Silicon only",
      run: () => api.ui.toast({ message: `${spec.name} requires macOS on Apple Silicon`, variant: "info" }),
    }
  }
  if (!spec.implemented) {
    const found = await allOnPath(spec.binary)
    return {
      title: `${found.length > 0 ? OK : MISSING} ${spec.name}`,
      description: found.length > 0 ? `installed, not supported yet` : `not installed — ${spec.install}`,
      run: () =>
        api.ui.toast({
          message: found.length > 0 ? `${spec.name} support is not implemented yet` : spec.install,
          variant: "info",
        }),
    }
  }

  // llama.cpp: the configured path wins, otherwise whatever is on $PATH
  const settings = await Server.load()
  const configured = settings.bin && (await isFile(settings.bin)) ? settings.bin : undefined
  const found = await allOnPath(spec.binary)
  const active = configured ?? found[0]

  return {
    title: `${active ? OK : MISSING} ${spec.name}`,
    description: active ? collapseHome(active) : `not found — ${spec.install}`,
    run: () => chooseBinary(api, spec, found, active, reopen),
  }
}

/**
 * Several llama.cpp builds commonly coexist — a tuned local build plus a
 * release binary — so this offers the ones it found and always allows typing
 * a path, rather than assuming the first hit on $PATH is the wanted one.
 */
function chooseBinary(api: any, spec: BackendSpec, found: string[], active: string | undefined, reopen: () => void) {
  const options = [
    ...found.map((file) => ({
      title: collapseHome(file),
      value: file,
      description: file === active ? "in use" : undefined,
    })),
    { title: "Enter a path…", value: "__custom__", description: `e.g. a local build of ${spec.binary}` },
  ]
  api.ui.dialog.replace(() => (
    <api.ui.DialogSelect
      title={`${spec.name} binary`}
      options={options}
      current={active}
      onSelect={(option: { value: string }) => {
        if (option.value === "__custom__") return promptPath(api, "Path to " + spec.binary, active ?? "", "bin", reopen)
        void Server.update("bin", option.value).then(reopen)
      }}
    />
  ))
}

function promptPath(api: any, title: string, value: string, key: "bin" | "models-dir", reopen: () => void) {
  api.ui.dialog.replace(() => (
    <api.ui.DialogPrompt
      title={title}
      value={collapseHome(value)}
      placeholder="~/..."
      onConfirm={(next: string) => {
        const resolved = expand(next)
        if (!resolved) return reopen()
        void Server.update(key, resolved).then(reopen)
      }}
      onCancel={reopen}
    />
  ))
}

async function rows(api: any, reopen: () => void): Promise<Row[]> {
  const settings = await Server.load()
  const out: Row[] = []

  for (const spec of BACKENDS) out.push(await backendRow(spec, api, reopen))

  const dir = settings.modelsDir
  const exists = dir ? await fs.stat(dir).then((s) => s.isDirectory()).catch(() => false) : false
  out.push({
    title: `${exists ? OK : MISSING} Models directory`,
    description: dir ? `${collapseHome(dir)}${exists ? "" : " — not found"}` : "not set — required",
    run: () => promptPath(api, "Models directory", dir, "models-dir", reopen),
  })

  const status = await backend.status().catch(() => undefined)
  const running = status?.state === "running"

  out.push({
    title: `${running ? "● Server running" : "○ Server stopped"}`,
    description: `${settings.host}:${settings.port} · ${settings.apiKey ? "api-key set" : "no api-key"} · autostart ${settings.autostart ? "on" : "off"}`,
    run: () => serverActions(api, running, settings.autostart, reopen),
  })

  return out
}

/** Start/stop is explicit: setting a path should not spawn a server by itself. */
function serverActions(api: any, running: boolean, autostart: boolean, reopen: () => void) {
  const options = [
    running
      ? { title: "Stop server", value: "stop", description: "kills the llama-server we started" }
      : { title: "Start server", value: "start", description: "loads models.ini and listens" },
    {
      title: autostart ? "Autostart: on → turn off" : "Autostart: off → turn on",
      value: "autostart",
      description: autostart ? "currently starts with opencode" : "currently only starts when you ask",
    },
    { title: "Open server.ini", value: "edit", description: collapseHome(Server.FILE) },
  ]
  api.ui.dialog.replace(() => (
    <api.ui.DialogSelect
      title="Server"
      options={options}
      onSelect={(option: { value: string }) => {
        if (option.value === "edit") {
          api.ui.toast({ message: `Edit ${collapseHome(Server.FILE)}`, variant: "info" })
          return reopen()
        }
        if (option.value === "autostart") {
          void Server.update("autostart", autostart ? "off" : "on").then(reopen)
          return
        }
        api.ui.toast({ message: running ? "stopping…" : "starting…", variant: "info" })
        const action = running ? backend.stop() : backend.start()
        void Promise.resolve(action).then(reopen, reopen)
      }}
    />
  ))
}

export function openSetup(api: any) {
  const reopen = () => openSetup(api)
  void rows(api, reopen).then((list) => {
    api.ui.dialog.replace(() => (
      <api.ui.DialogSelect
        title="Local models — setup"
        options={list.map((row) => ({
          title: row.title,
          value: row.title,
          description: row.description,
        }))}
        onSelect={(option: { value: string }) => {
          const row = list.find((item) => item.title === option.value)
          if (!row?.run) return
          void row.run()
        }}
      />
    ))
  })
}
