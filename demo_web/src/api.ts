// Types mirror the events yielded by src/demo/pipeline.py, one per stage.

export type Quantity = { value: number; unit: string }
export type Pred = { value: number; unit: string; rule: string } | null

export type OcrEvent = {
  stage: 'ocr'; ms: number; source: string
  lines: { text: string; box: [number, number, number, number] | null }[]
}
export type CandidatesEvent = {
  stage: 'candidates'; ms: number; family: string; cleaned: string
  items: { value: number; unit: string; pos: number }[]
}
export type CascadeEvent = {
  stage: 'cascade'; ms: number; input: string; unit_head: string
  spans: { kind: 'SIZE' | 'UNIT' | 'PACK'; start: number; end: number }[]
  tagger: Pred; rules: Pred; cascade: Pred; source: 'tagger' | 'rules'
}
export type VlmEvent = {
  stage: 'vlm'; ms: number; options: [number, string][]
  skipped?: string; control?: 'real' | 'blank'; prompt?: string
  letters?: string[]; probs?: number[]; letter_mass?: number
  raw?: string; pick?: string | null; pred?: [number, string] | null; visual_tokens?: number
}
export type FinalEvent = {
  stage: 'final'; ms: number; source: 'cascade' | 'vlm'; reason: string
  answer: Quantity | null
  gold?: { low: number; high: number; unit: string }; correct?: boolean
}
export type ErrorEvent = { stage: 'error'; message: string }
export type StageEvent = (OcrEvent | CandidatesEvent | CascadeEvent | VlmEvent | FinalEvent | ErrorEvent)
  & { replayed?: boolean }

export type Example = {
  id: string; image_file: string; attribute: string
  gold: [number, number, string]; title: string; note: string
}
export type Health = {
  mode: 'live' | 'replay'; ready: boolean; device: string | null; vlm: boolean
  load_s: number | null; warmup_ms: number | null; attributes: string[]
}

async function detail(res: Response): Promise<string> {
  try { return (await res.json()).detail ?? res.statusText } catch { return res.statusText }
}

export async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(await detail(res))
  return res.json()
}

// POST /api/extract answers with NDJSON: one event per line, flushed as each
// stage finishes.  Yield them as they arrive rather than waiting for the end.
export async function* extract(form: FormData, signal: AbortSignal): AsyncGenerator<StageEvent> {
  const res = await fetch('/api/extract', { method: 'POST', body: form, signal })
  if (!res.ok || !res.body) throw new Error(await detail(res))
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += value
    let nl
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl)
      buf = buf.slice(nl + 1)
      if (line.trim()) yield JSON.parse(line)
    }
  }
  if (buf.trim()) yield JSON.parse(buf)
}

const SHORT: Record<string, string> = {
  centimeter: 'cm', centimetre: 'cm', millimeter: 'mm', millimetre: 'mm', meter: 'm', metre: 'm',
  inch: 'in', foot: 'ft', gram: 'g', kilogram: 'kg', milligram: 'mg', microgram: 'µg',
  pound: 'lb', ounce: 'oz', ton: 't', milliliter: 'ml', liter: 'L', fluid_ounce: 'fl oz',
  volt: 'V', kilovolt: 'kV', millivolt: 'mV', watt: 'W', kilowatt: 'kW',
}
export const num = (v: number) => String(+v.toPrecision(6))
export const qty = (v: number, u: string) => `${num(v)} ${SHORT[u] ?? u.replace(/_/g, ' ')}`
export const label = (attr: string) => attr.replace(/_/g, ' ')
