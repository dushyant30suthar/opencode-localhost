import os from "os"
import path from "path"

/**
 * Where a backend's files live. Config is what you edit; state is what we own.
 *
 * Config follows XDG so it sits next to opencode's own config, which is where
 * people look for it. State holds pidfiles and logs — things you never edit and
 * that should not survive in a dotfiles repo.
 */

const xdg = (envVar: string, fallback: string) => process.env[envVar] || path.join(os.homedir(), fallback)

export const CONFIG_ROOT = path.join(xdg("XDG_CONFIG_HOME", ".config"), "opencode", "providers")
export const STATE_ROOT = path.join(xdg("XDG_STATE_HOME", ".local/state"), "opencode", "providers")

export function configDir(backend: string) {
  return path.join(CONFIG_ROOT, backend)
}

export function stateDir(backend: string) {
  return path.join(STATE_ROOT, backend)
}

/** `~/models` and `$HOME/models` both appear in hand-edited config files. */
export function expandHome(value: string): string {
  if (value === "~") return os.homedir()
  if (value.startsWith("~/")) return path.join(os.homedir(), value.slice(2))
  if (value.startsWith("$HOME/")) return path.join(os.homedir(), value.slice(6))
  return value
}

/** Written back to config so paths stay readable rather than machine-specific. */
export function collapseHome(value: string): string {
  const home = os.homedir()
  return value.startsWith(home + path.sep) ? "~" + value.slice(home.length) : value
}
