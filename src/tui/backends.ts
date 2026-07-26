import { create as llamacpp } from "../server/llamacpp/index.ts"
import type { Backend } from "../server/backend.ts"

/**
 * The engines this TUI half talks to, shared so the panel, the setup screen
 * and the commands all see the same instances and the same busy state.
 *
 * A machine can have several; each is its own opencode provider.
 */
export const BACKENDS: Backend[] = [llamacpp()]

export function backendById(id: string): Backend | undefined {
  return BACKENDS.find((backend) => backend.id === id)
}
