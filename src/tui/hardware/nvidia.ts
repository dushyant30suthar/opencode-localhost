import { execFile } from "child_process"
import type { GpuStat } from "../../shared/types.ts"

/**
 * GPU stats via nvidia-smi. There is no push interface, so this is polled —
 * but only while the panel is visible.
 *
 * Absent nvidia-smi is a normal state, not an error: plenty of machines run
 * llama.cpp on CPU, Metal, or ROCm. We return nothing and the panel omits the
 * GPU rows rather than showing a failure.
 */

const QUERY = "index,memory.used,memory.total,utilization.gpu"
const TIMEOUT = 2_000

let available = true

function run(): Promise<string | undefined> {
  return new Promise((resolve) => {
    execFile(
      "nvidia-smi",
      [`--query-gpu=${QUERY}`, "--format=csv,noheader,nounits"],
      { timeout: TIMEOUT },
      (error, stdout) => {
        if (error) {
          // ENOENT means no NVIDIA tooling at all; stop trying every tick
          if ((error as NodeJS.ErrnoException).code === "ENOENT") available = false
          return resolve(undefined)
        }
        resolve(stdout)
      },
    )
  })
}

export async function gpus(): Promise<GpuStat[]> {
  if (!available) return []
  const output = await run()
  if (!output) return []
  const stats: GpuStat[] = []
  for (const line of output.trim().split("\n")) {
    const parts = line.split(",").map((value) => Number.parseInt(value.trim(), 10))
    if (parts.length < 3 || parts.some((value, index) => index < 3 && !Number.isFinite(value))) continue
    stats.push({
      index: parts[0],
      usedMiB: parts[1],
      totalMiB: parts[2],
      utilization: Number.isFinite(parts[3]) ? parts[3] : undefined,
    })
  }
  return stats
}
