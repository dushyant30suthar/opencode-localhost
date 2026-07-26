import fs from "fs/promises"
import os from "os"
import type { SystemStat } from "../../shared/types.ts"

/**
 * System RAM and CPU.
 *
 * RAM uses MemAvailable rather than os.freemem(): free memory excludes page
 * cache, so on a box that has just mmap'd a 20GB model it reads as almost
 * nothing and looks alarming. MemAvailable is what "how much can I still use"
 * actually means.
 */

const MIB = 1024 * 1024

type CpuSample = { idle: number; total: number }

let previous: CpuSample | undefined

async function meminfo(): Promise<SystemStat | undefined> {
  const text = await fs.readFile("/proc/meminfo", "utf8").catch(() => undefined)
  if (!text) {
    // non-Linux: free memory is the best available approximation
    const total = os.totalmem() / MIB
    const used = (os.totalmem() - os.freemem()) / MIB
    return { usedMiB: Math.round(used), totalMiB: Math.round(total) }
  }
  const read = (key: string) => {
    const match = text.match(new RegExp(`^${key}:\\s+(\\d+) kB`, "m"))
    return match ? Number.parseInt(match[1], 10) / 1024 : undefined
  }
  const total = read("MemTotal")
  const availableKiB = read("MemAvailable")
  if (total === undefined || availableKiB === undefined) return undefined
  return { usedMiB: Math.round(total - availableKiB), totalMiB: Math.round(total) }
}

/** Utilisation between the previous call and now; undefined on the first. */
async function cpuUtilisation(): Promise<number | undefined> {
  const text = await fs.readFile("/proc/stat", "utf8").catch(() => undefined)
  let sample: CpuSample | undefined
  if (text) {
    const line = text.split("\n")[0]
    const values = line.split(/\s+/).slice(1).map(Number).filter(Number.isFinite)
    if (values.length >= 5) {
      const total = values.reduce((sum, value) => sum + value, 0)
      sample = { idle: values[3] + (values[4] ?? 0), total }
    }
  }
  if (!sample) {
    const load = os.loadavg()[0]
    const cores = os.cpus().length || 1
    return Math.min(100, Math.round((load / cores) * 100))
  }
  const last = previous
  previous = sample
  if (!last) return undefined
  const totalDelta = sample.total - last.total
  const idleDelta = sample.idle - last.idle
  if (totalDelta <= 0) return undefined
  return Math.max(0, Math.min(100, Math.round(((totalDelta - idleDelta) / totalDelta) * 100)))
}

export async function system(): Promise<SystemStat | undefined> {
  const memory = await meminfo()
  if (!memory) return undefined
  return { ...memory, utilization: await cpuUtilisation() }
}
