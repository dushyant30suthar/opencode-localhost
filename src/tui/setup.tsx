import fs from "fs/promises"
import { BACKENDS, allOnPath, expand, normalizeRemote, supported, type BackendSpec } from "../shared/backends.ts"
import { collapseHome } from "../shared/paths.ts"
import { backendById, configById, type BackendConfig } from "./backends.ts"

/**
 * Setup, kept separate from the status strip.
 *
 * The strip answers "is it working right now". This answers "where is
 * everything" — a different question, asked at a different time, and cramming
 * both into three narrow columns is what made the unconfigured state unreadable.
 *
 * Built from DialogSelect and DialogPrompt rather than a hand-rolled screen, so
 * filtering, keyboard navigation and theming come from opencode itself.
 *
 * Every row is scoped to one backend. Nothing here may reach for a specific
 * engine's settings module: while llama.cpp was the only implemented one this
 * screen loaded its config for every row, so a second backend showed — and
 * edited — llama.cpp's paths under its own name.
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

/**
 * Save, then say what the save did *not* do.
 *
 * The provider list is built once, in the server half's config() hook, because
 * opencode loads plugins before it reads cfg.provider — that is the whole
 * integration. Nothing in the plugin API re-runs it, so a setting that changes
 * *which models exist* only takes effect on the next launch. Without saying so
 * the write looks like it failed: the file is correct, the dialog closes, and
 * the model picker is unchanged.
 */
async function saved(api: any, cfg: BackendConfig, key: "bin" | "models-dir" | "remote", value: string, note: string) {
  await cfg.update(key, value)
  api.ui.toast({ message: `saved — restart opencode to ${note}`, variant: "info" })
}

async function backendRow(spec: BackendSpec, cfg: BackendConfig | undefined, api: any, reopen: () => void): Promise<Row> {
  if (!supported(spec)) {
    return {
      title: `${MISSING} ${spec.name}`,
      description: "Apple Silicon only",
      run: () => api.ui.toast({ message: `${spec.name} requires macOS on Apple Silicon`, variant: "info" }),
    }
  }
  // No settings module is the same situation as not implemented: there is
  // nothing this screen could read or write for it.
  if (!spec.implemented || !cfg) {
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

  // the configured path wins, otherwise whatever is on $PATH
  const settings = await cfg.load()

  // Pointed at another machine: a missing local binary is the expected state,
  // not a problem to flag. Reporting "not found" here sends people hunting for
  // something they deliberately do not need.
  if (settings.remote) {
    return {
      title: `${OK} ${spec.name}`,
      description: `using ${settings.remote} — no local server`,
      run: () => promptBinaryOrRemote(api, spec, cfg, settings.remote, reopen),
    }
  }

  const configured = settings.bin && (await isFile(settings.bin)) ? settings.bin : undefined
  const found = await allOnPath(spec.binary)
  const active = configured ?? found[0]

  return {
    title: `${active ? OK : MISSING} ${spec.name}`,
    description: active ? collapseHome(active) : `not found — ${spec.install}`,
    run: () => chooseBinary(api, spec, cfg, found, active, reopen),
  }
}

/**
 * Several llama.cpp builds commonly coexist — a tuned local build plus a
 * release binary — so when there is a genuine choice this offers the ones it
 * found, and always allows typing a path or an address instead.
 *
 * With nothing to choose between, it goes straight to the prompt. A select
 * whose only row is "type something else" is a trap: its text box filters, so
 * typing an address matches no row, the escape hatch disappears, and the dialog
 * shows "No results found" with no way to commit what you typed.
 */
function chooseBinary(
  api: any,
  spec: BackendSpec,
  cfg: BackendConfig,
  found: string[],
  active: string | undefined,
  reopen: () => void,
) {
  if (found.length < 2) return promptBinaryOrRemote(api, spec, cfg, active ?? found[0] ?? "", reopen)

  let typed = ""
  const options = [
    ...found.map((file) => ({
      title: collapseHome(file),
      value: file,
      description: file === active ? "in use" : undefined,
    })),
    {
      title: "Enter a path or an address…",
      value: "__custom__",
      description: `a local build of ${spec.binary}, or another machine e.g. fedora.local:${spec.id === "openvino" ? 8100 : 9337}`,
    },
  ]
  api.ui.dialog.replace(() => (
    <api.ui.DialogSelect
      title={`${spec.name} binary`}
      options={options}
      current={active}
      // Filtering would hide the "enter something else" row the moment someone
      // types an address into the box — which reads as the dialog refusing the
      // input. Keep every row reachable and carry what was typed into the
      // prompt instead, so the keystrokes are not thrown away.
      skipFilter
      onFilter={(query: string) => {
        typed = query
      }}
      onSelect={(option: { value: string }) => {
        if (option.value === "__custom__") {
          return promptBinaryOrRemote(api, spec, cfg, typed || active || "", reopen)
        }
        void saved(api, cfg, "bin", option.value, `use ${collapseHome(option.value)}`).then(reopen)
      }}
    />
  ))
}

function promptPath(
  api: any,
  cfg: BackendConfig,
  title: string,
  value: string,
  key: "bin" | "models-dir",
  reopen: () => void,
) {
  api.ui.dialog.replace(() => (
    <api.ui.DialogPrompt
      title={title}
      value={collapseHome(value)}
      placeholder="~/..."
      onConfirm={(next: string) => {
        const resolved = expand(next)
        if (!resolved) return reopen()
        void saved(api, cfg, key, resolved, "pick up the change").then(reopen)
      }}
      onCancel={reopen}
    />
  ))
}

/**
 * One prompt, two destinations. Typing an address writes `remote` and the backend
 * stops managing a process entirely — no binary, no models directory, no models.ini
 * on this machine; the model list comes from that server.
 */
function promptBinaryOrRemote(api: any, spec: BackendSpec, cfg: BackendConfig, value: string, reopen: () => void) {
  api.ui.dialog.replace(() => (
    <api.ui.DialogPrompt
      title={`Path to ${spec.binary}, or an address`}
      value={collapseHome(value)}
      placeholder={`~/…/${spec.binary}  —  or  fedora.local:9337`}
      onConfirm={(next: string) => {
        const text = (next ?? "").trim()
        if (!text) return reopen()
        if (cfg.looksRemote(text)) {
          const addr = normalizeRemote(text)
          return void saved(api, cfg, "remote", addr, `load models from ${addr}`).then(reopen)
        }
        const resolved = expand(text)
        if (!resolved) return reopen()
        void saved(api, cfg, "bin", resolved, `use ${collapseHome(resolved)}`).then(reopen)
      }}
      onCancel={reopen}
    />
  ))
}

async function rows(api: any, reopen: () => void): Promise<Row[]> {
  const out: Row[] = []

  for (const spec of BACKENDS) {
    const cfg = configById(spec.id)
    out.push(await backendRow(spec, cfg, api, reopen))

    const engine = backendById(spec.id)
    if (!cfg || !engine || !spec.implemented || !supported(spec)) continue
    const settings = await cfg.load()

    // A remote serves its own models, so there is nothing here to scan. Showing
    // "not set — required" would be false: it is neither set nor needed.
    if (!settings.remote) {
      const dir = settings.modelsDir
      const exists = dir ? await fs.stat(dir).then((s) => s.isDirectory()).catch(() => false) : false
      out.push({
        // Titles double as the select's values, so they carry the backend name:
        // two engines both offering "Models directory" would collide and the
        // wrong one would open.
        title: `  ${exists ? OK : MISSING} ${spec.name} models directory`,
        description: dir ? `${collapseHome(dir)}${exists ? "" : " — not found"}` : "not set — required",
        run: () => promptPath(api, cfg, `${spec.name} models directory`, dir, "models-dir", reopen),
      })
    }

    const status = await engine.status().catch(() => undefined)
    const running = status?.state === "running"
    const where = settings.remote || `${settings.host}:${settings.port}`

    // the action is on the row itself — selecting it starts or stops, rather
    // than opening a submenu to find the one obvious thing to do
    out.push({
      title: running ? `  ● ${spec.name} running   [stop]` : `  ○ ${spec.name} stopped   [start]`,
      description: settings.remote ? `${where} — remote, not ours to start or stop` : where,
      run: async () => {
        api.ui.toast({ message: running ? "stopping…" : "starting…", variant: "info" })
        const ok = await (running ? engine.stop() : engine.start()).catch(() => undefined)
        // stop() answering false is meaningful on a remote: OVMS has no unload
        // endpoint, so there is nothing this machine can do about that GPU.
        if (running && ok === false && settings.remote) {
          api.ui.toast({ message: `${where} is not ours to stop`, variant: "info" })
        }
        reopen()
      },
    })
  }

  return out
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
