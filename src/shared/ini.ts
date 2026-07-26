/**
 * INI reader/writer that preserves everything it does not understand.
 *
 * These files are hand-edited — comments explaining why a model needs
 * `ubatch-size = 2048` are the most valuable thing in them. A parse/serialize
 * round trip that drops comments, ordering, or blank lines would quietly
 * destroy that, so sections keep their raw lines and are only rewritten when a
 * key in them actually changes.
 */

export type Section = {
  name: string
  /** Raw lines between this header and the next, verbatim. */
  lines: string[]
}

export type Document = {
  /** Lines before the first section header. */
  preamble: string[]
  sections: Section[]
}

const HEADER = /^\s*\[([^\]]+)\]\s*$/
/** Keys may repeat (llama.cpp allows it); we address the first occurrence. */
const entry = (key: string) => new RegExp(`^(\\s*)${escape(key)}(\\s*=\\s*)(.*?)(\\s*)$`)

function escape(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

export function parse(text: string): Document {
  const doc: Document = { preamble: [], sections: [] }
  let current: Section | undefined
  for (const line of text.split("\n")) {
    const header = line.match(HEADER)
    if (header) {
      current = { name: header[1], lines: [] }
      doc.sections.push(current)
      continue
    }
    if (current) current.lines.push(line)
    else doc.preamble.push(line)
  }
  return doc
}

export function serialize(doc: Document): string {
  const out: string[] = [...doc.preamble]
  for (const section of doc.sections) {
    out.push(`[${section.name}]`)
    out.push(...section.lines)
  }
  return out.join("\n").replace(/\n*$/, "\n")
}

export function get(section: Section, key: string): string | undefined {
  for (const line of section.lines) {
    const match = line.match(entry(key))
    if (match) return match[3]
  }
  return undefined
}

/**
 * Set a key in place, keeping its original spacing. Undefined removes it.
 * Appends after the last existing entry rather than at the very end, so a
 * trailing comment block stays at the bottom where the user put it.
 */
export function set(section: Section, key: string, value: string | undefined): void {
  const pattern = entry(key)
  for (let i = 0; i < section.lines.length; i++) {
    const match = section.lines[i].match(pattern)
    if (!match) continue
    if (value === undefined) section.lines.splice(i, 1)
    else section.lines[i] = `${match[1]}${key}${match[2]}${value}`
    return
  }
  if (value === undefined) return
  let insert = section.lines.length
  while (insert > 0 && !section.lines[insert - 1].includes("=")) insert--
  section.lines.splice(insert, 0, `${key} = ${value}`)
}

export function find(doc: Document, name: string): Section | undefined {
  return doc.sections.find((section) => section.name === name)
}

/** Every `key = value` in a section, first occurrence wins. Comments skipped. */
export function entries(section: Section): Record<string, string> {
  const out: Record<string, string> = {}
  for (const line of section.lines) {
    if (/^\s*[#;]/.test(line)) continue
    const match = line.match(/^\s*([^=\s]+)\s*=\s*(.*?)\s*$/)
    if (match && !(match[1] in out)) out[match[1]] = match[2]
  }
  return out
}
