import { useEffect, useMemo, useRef, useState } from 'react'
import {
  extract, getJSON, label, num, qty,
  type CandidatesEvent, type CascadeEvent, type Example, type FinalEvent, type Health,
  type OcrEvent, type StageEvent, type VlmEvent,
} from './api'

type Source =
  | { kind: 'example'; id: string; ex: Example; url: string }
  | { kind: 'upload'; id: string; file: File; url: string }
type Control = 'real' | 'blank'
type Run = { events: StageEvent[]; running: boolean; error?: string; started: number }

const LENGTH = new Set(['depth', 'width', 'height'])
const STAGES = [
  ['ocr', 'OCR'], ['candidates', 'Candidates'], ['cascade', 'Tagger + rules'],
  ['vlm', 'VLM selector'], ['final', 'Router → answer'],
] as const

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthErr, setHealthErr] = useState<string>()
  const [examples, setExamples] = useState<Example[]>([])
  const [source, setSource] = useState<Source | null>(null)
  const [attribute, setAttribute] = useState('width')
  const [control, setControl] = useState<Control>('real')
  const [runs, setRuns] = useState<Record<string, Run>>({})
  const abort = useRef<AbortController | null>(null)

  // The server loads two models before answering; poll until it is ready.
  useEffect(() => {
    let stop = false
    const poll = async () => {
      try {
        const h = await getJSON<Health>('/api/health')
        if (stop) return
        setHealth(h); setHealthErr(undefined)
        if (!h.ready) setTimeout(poll, 1500)
        else setExamples(await getJSON<Example[]>('/api/examples'))
      } catch (e) {
        if (stop) return
        setHealthErr(String((e as Error).message)); setTimeout(poll, 2000)
      }
    }
    poll()
    return () => { stop = true }
  }, [])

  const key = (c: Control) => source ? `${source.id}|${attribute}|${c}` : ''
  const run = runs[key(control)]
  const other = runs[key(control === 'real' ? 'blank' : 'real')]
  const busy = Object.values(runs).some(r => r.running)

  const pickExample = (ex: Example) => {
    setSource({ kind: 'example', id: ex.id, ex, url: `/api/images/${ex.image_file}` })
    setAttribute(ex.attribute); setControl('real')
  }
  const pickFile = (file: File) => {
    if (source?.kind === 'upload') URL.revokeObjectURL(source.url)
    setSource({ kind: 'upload', id: `upload-${Date.now()}`, file, url: URL.createObjectURL(file) })
    setControl('real')
  }

  const start = async (c: Control = control) => {
    if (!source) return
    const k = `${source.id}|${attribute}|${c}`
    abort.current?.abort()
    const ac = new AbortController(); abort.current = ac
    const form = new FormData()
    form.set('attribute', attribute); form.set('control', c)
    if (source.kind === 'example') form.set('example_id', source.id)
    else form.set('image', source.file)
    setControl(c)
    setRuns(r => ({ ...r, [k]: { events: [], running: true, started: performance.now() } }))
    const patch = (f: (run: Run) => Run) => setRuns(r => ({ ...r, [k]: f(r[k]) }))
    try {
      for await (const ev of extract(form, ac.signal)) {
        patch(run => ({ ...run, events: [...run.events, ev],
                        error: ev.stage === 'error' ? ev.message : run.error }))
      }
      patch(run => ({ ...run, running: false }))
    } catch (e) {
      if (ac.signal.aborted) return
      patch(run => ({ ...run, running: false, error: String((e as Error).message) }))
    }
  }

  const ev = useMemo(() => {
    const by: Partial<Record<string, StageEvent>> = {}
    for (const e of run?.events ?? []) by[e.stage] = e
    return by as {
      ocr?: OcrEvent; candidates?: CandidatesEvent; cascade?: CascadeEvent; vlm?: VlmEvent; final?: FinalEvent
    }
  }, [run])
  // The blank run never shows the photo's OCR boxes differently, so reuse either run's.
  const ocrEv = ev.ocr ?? (other?.events.find(e => e.stage === 'ocr') as OcrEvent | undefined)
  const otherVlm = other?.events.find(e => e.stage === 'vlm') as VlmEvent | undefined

  const canBlank = LENGTH.has(attribute)
  const exGold = source?.kind === 'example' && source.ex.attribute === attribute ? source.ex.gold : null

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Product attribute extraction</h1>
          <p className="sub">Photo + attribute name → value. Apple Vision OCR · DistilRoBERTa tagger + rules ·
            Qwen2-VL-2B LoRA selector · all local</p>
        </div>
        <Status health={health} err={healthErr} />
      </header>

      <main className="grid">
        <aside className="panel side">
          <h2>Input</h2>
          <label className="field">
            <span>Attribute</span>
            <select value={attribute} onChange={e => setAttribute(e.target.value)} disabled={busy}>
              {(health?.attributes ?? []).map(a => <option key={a} value={a}>{label(a)}</option>)}
            </select>
          </label>
          <label className={`upload ${health?.mode === 'replay' ? 'disabled' : ''}`}>
            <input type="file" accept="image/*" disabled={busy || health?.mode === 'replay'}
                   onChange={e => e.target.files?.[0] && pickFile(e.target.files[0])} />
            <span>Upload a product photo</span>
          </label>

          <h2>Test-split examples</h2>
          <ul className="examples">
            {examples.map(ex => (
              <li key={ex.id}>
                <button className={source?.id === ex.id ? 'on' : ''} disabled={busy}
                        onClick={() => pickExample(ex)}>
                  <img src={`/api/images/${ex.image_file}`} alt="" loading="lazy" />
                  <span><b>{ex.title}</b><small>{ex.note}</small></span>
                </button>
              </li>
            ))}
            {!examples.length && <li className="muted">{health?.ready ? 'No examples found.' : 'Loading…'}</li>}
          </ul>
        </aside>

        <section className="panel photo">
          <div className="photo-head">
            <h2>{source ? (source.kind === 'example' ? source.ex.image_file : source.file.name) : 'No image yet'}</h2>
            <div className="actions">
              <button className="primary" disabled={!source || !health?.ready || busy} onClick={() => start('real')}>
                {run?.running && control === 'real' ? 'Running…' : `Extract ${label(attribute)}`}
              </button>
              {canBlank && (
                <button disabled={!source || !health?.ready || busy} onClick={() => start('blank')}
                        title="Send the VLM a white image of the same size - the S4 control">
                  {run?.running && control === 'blank' ? 'Running…' : 'Blank-image control'}
                </button>
              )}
            </div>
          </div>
          {source ? (
            <Photo url={source.url} ocr={ocrEv} cands={ev.candidates} final={ev.final} blank={control === 'blank'} />
          ) : (
            <div className="empty">Pick an example or upload a photo, choose the attribute, then extract.</div>
          )}
          {exGold && <p className="gold">Gold label: <b>{goldText(exGold[0], exGold[1], exGold[2])}</b></p>}
        </section>

        <section className="panel stages">
          <Stepper run={run} ev={ev} attribute={attribute} />
          {run?.error && <div className="error">{run.error}</div>}
          {!run && <div className="empty">The pipeline's stages will appear here as the server streams them.</div>}
          {ev.ocr && <OcrCard e={ev.ocr} />}
          {ev.candidates && <CandCard e={ev.candidates} />}
          {ev.cascade && <CascadeCard e={ev.cascade} />}
          {ev.vlm && <VlmCard e={ev.vlm} other={otherVlm} />}
          {ev.final && <FinalCard e={ev.final} />}
        </section>
      </main>
    </div>
  )
}

const goldText = (lo: number, hi: number, u: string) => lo === hi ? qty(lo, u) : `${num(lo)}–${qty(hi, u)}`

function Status({ health, err }: { health: Health | null; err?: string }) {
  if (err) return <span className="pill bad">server unreachable</span>
  if (!health) return <span className="pill">connecting…</span>
  if (!health.ready) return <span className="pill">loading models…</span>
  if (health.mode === 'replay') return <span className="pill warn">replay · saved runs</span>
  return (
    <span className="pill good" title={`models loaded in ${health.load_s}s, warm-up ${health.warmup_ms} ms`}>
      live · {health.device}{health.vlm ? '' : ' · no VLM'}
    </span>
  )
}

function Stepper({ run, ev, attribute }: { run?: Run; ev: Record<string, StageEvent | undefined>; attribute: string }) {
  const reached = STAGES.findIndex(([s]) => !ev[s])
  return (
    <ol className="stepper">
      {STAGES.map(([s, name], i) => {
        const e = ev[s] as (StageEvent & { ms?: number; skipped?: string }) | undefined
        const skipped = s === 'vlm' && (!LENGTH.has(attribute) || e?.skipped)
        const state = e ? (skipped ? 'skip' : 'done') : run?.running && i === reached ? 'active' : 'idle'
        return (
          <li key={s} className={state}>
            <span className="dot" />
            <span>{name}</span>
            {e?.ms !== undefined && <small>{e.ms} ms</small>}
          </li>
        )
      })}
    </ol>
  )
}

function Photo({ url, ocr, cands, final, blank }: {
  url: string; ocr?: OcrEvent; cands?: CandidatesEvent; final?: FinalEvent; blank: boolean
}) {
  // A line is highlighted when it contains a candidate's number; the answer's line more strongly.
  const has = (t: string, v: number) => new RegExp(`(^|[^\\d.])${num(v).replace('.', '[.,]')}(?![\\d])`).test(t)
  const boxes = (ocr?.lines ?? []).filter(l => l.box)
  return (
    <div className="photo-wrap">
      <div className="photo-frame">
        <img src={url} alt="product" />
        <svg viewBox="0 0 1 1" preserveAspectRatio="none">
          {boxes.map((l, i) => {
            const [x, y, w, h] = l.box!
            const isAns = final?.answer && has(l.text, final.answer.value)
            const isCand = cands?.items.some(c => has(l.text, c.value))
            return (
              <rect key={i} x={x} y={y} width={w} height={h}
                    className={isAns ? 'ans' : isCand ? 'cand' : 'line'}>
                <title>{l.text}</title>
              </rect>
            )
          })}
        </svg>
        {blank && <div className="blank-veil"><span>The VLM is shown a blank white page of the same size</span></div>}
      </div>
      {boxes.length > 0 && (
        <p className="legend"><i className="k-line" /> OCR line <i className="k-cand" /> candidate
          <i className="k-ans" /> answer</p>
      )}
    </div>
  )
}

function Card({ n, title, ms, children, extra }: {
  n: number; title: string; ms?: number; children: React.ReactNode; extra?: React.ReactNode
}) {
  return (
    <article className="card">
      <header><span className="n">{n}</span><h3>{title}</h3>{extra}{ms !== undefined && <small>{ms} ms</small>}</header>
      {children}
    </article>
  )
}

function OcrCard({ e }: { e: OcrEvent }) {
  return (
    <Card n={1} title={`OCR · ${e.lines.length} lines`} ms={e.ms}
          extra={<span className="tag">{e.source}{(e as StageEvent).replayed ? ' · replayed' : ''}</span>}>
      <pre className="ocr">{e.lines.map(l => l.text).join('\n') || '(no text found)'}</pre>
    </Card>
  )
}

function CandCard({ e }: { e: CandidatesEvent }) {
  return (
    <Card n={2} title={`Same-family candidates · ${e.family}`} ms={e.ms}>
      <p className="hint">Only a {e.family} quantity can answer this attribute; every other number is discarded.</p>
      <div className="chips">
        {e.items.length ? e.items.map((c, i) => <span key={i} className="chip">{qty(c.value, c.unit)}</span>)
          : <span className="muted">none found</span>}
      </div>
    </Card>
  )
}

function CascadeCard({ e }: { e: CascadeEvent }) {
  const parts: React.ReactNode[] = []
  let at = 0
  for (const [i, s] of e.spans.entries()) {
    if (s.start > at) parts.push(e.input.slice(at, s.start))
    parts.push(<mark key={i} className={s.kind.toLowerCase()} title={s.kind}>{e.input.slice(s.start, s.end)}</mark>)
    at = s.end
  }
  parts.push(e.input.slice(at))
  const p = (x: CascadeEvent['tagger']) => x ? qty(x.value, x.unit) : 'abstains'
  return (
    <Card n={3} title="Tagger + rules cascade (S2)" ms={e.ms}>
      <pre className="tagged">{parts}</pre>
      <p className="legend"><mark className="size">SIZE</mark> <mark className="unit">UNIT</mark>{' '}
        <mark className="pack">PACK</mark> spans · unit head says <b>{e.unit_head}</b></p>
      <table className="kv">
        <tbody>
          <tr className={e.source === 'tagger' ? 'used' : ''}><td>Tagger</td><td>{p(e.tagger)}</td></tr>
          <tr className={e.source === 'rules' ? 'used' : ''}><td>Rules (S0)</td><td>{p(e.rules)}</td></tr>
          <tr><td>Cascade</td><td><b>{p(e.cascade)}</b> <small>from the {e.source}</small></td></tr>
        </tbody>
      </table>
    </Card>
  )
}

function VlmCard({ e, other }: { e: VlmEvent; other?: VlmEvent }) {
  if (e.skipped) {
    return (
      <Card n={4} title="VLM selector (S3)" ms={e.ms}>
        <p className="muted">Skipped: {e.skipped}.
          {e.options.length === 1 && <> The only candidate, <b>{qty(...e.options[0])}</b>, is taken.</>}</p>
      </Card>
    )
  }
  const cmp = other?.probs && other.options.length === e.options.length ? other : undefined
  return (
    <Card n={4} title={`VLM selector (S3) · ${e.control === 'blank' ? 'blank image' : 'photo'}`} ms={e.ms}
          extra={<span className={`tag ${e.control === 'blank' ? 'warn' : ''}`}>{e.visual_tokens} visual tokens</span>}>
      <p className="hint">Qwen2-VL-2B + LoRA chooses the letter of the requested dimension. Probabilities are
        over the shown letters ({Math.round(100 * (e.letter_mass ?? 0))}% of its mass lands on them).</p>
      <div className="bars">
        {e.options.map((o, i) => {
          const pr = e.probs![i]; const cp = cmp?.probs?.[i]
          return (
            <div key={i} className={`bar ${e.letters![i] === e.pick ? 'pick' : ''}`}>
              <span className="opt">({e.letters![i]}) {qty(...o)}</span>
              <span className="track">
                <span className="fill" style={{ width: `${100 * pr}%` }} />
                {cp !== undefined && <span className="ghost" style={{ left: `${100 * cp}%` }}
                                           title={`${cmp!.control === 'blank' ? 'blank' : 'photo'}: ${(100 * cp).toFixed(1)}%`} />}
              </span>
              <span className="pct">{(100 * pr).toFixed(1)}%</span>
            </div>
          )
        })}
      </div>
      {cmp && <p className="legend"><i className="k-ghost" /> same options with the {cmp.control === 'blank' ? 'blank image' : 'photo'} — pick ({cmp.pick ?? '–'})</p>}
      <details><summary>Prompt sent to the model</summary><pre className="prompt">{e.prompt}</pre></details>
    </Card>
  )
}

function FinalCard({ e }: { e: FinalEvent }) {
  return (
    <Card n={5} title="Router → answer (S5)" ms={e.ms}>
      <div className="answer">
        <span className="big">{e.answer ? qty(e.answer.value, e.answer.unit) : 'no answer'}</span>
        <span className={`tag ${e.source}`}>{e.source === 'vlm' ? 'from the VLM' : 'from the cascade'}</span>
        {e.correct !== undefined && (
          <span className={`verdict ${e.correct ? 'ok' : 'bad'}`}>
            {e.correct ? '✓ matches gold' : '✗ gold is'} {!e.correct && e.gold && goldText(e.gold.low, e.gold.high, e.gold.unit)}
          </span>
        )}
      </div>
      <p className="hint">{e.reason}</p>
    </Card>
  )
}
