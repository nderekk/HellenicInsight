import { useEffect, useRef, useState, useCallback } from 'react'
import * as d3 from 'd3'

// ── Eye geometry ──────────────────────────────────────────────────
const CX = 330, CY = 200, W = 400, R = 273
const D_OFFSET = Math.sqrt(R * R - (W / 2) * (W / 2))
const UPPER_CY = CY + D_OFFSET
const LOWER_CY = CY - D_OFFSET
const EYE_H    = 2 * (R - D_OFFSET)
const DOT_RADIUS  = 3.5
const DOT_SPACING = 10
const MAX_DOTS    = 500
const REFRESH_MS  = 5000
const EYE_PATH    = `M 130,200 A ${R},${R} 0 0,1 530,200 A ${R},${R} 0 0,1 130,200 Z`

const DOT_POSITIONS: [number, number][] = (() => {
  const out: [number, number][] = []
  const left = CX - W / 2
  const top  = CY - EYE_H / 2
  const r2   = R * R
  for (let row = 0; row <= Math.ceil(EYE_H / DOT_SPACING) + 2; row++) {
    for (let col = 0; col <= Math.ceil(W / DOT_SPACING) + 2; col++) {
      const px = left + col * DOT_SPACING
      const py = top  + row * DOT_SPACING
      if (
        (px - CX) ** 2 + (py - UPPER_CY) ** 2 <= r2 &&
        (px - CX) ** 2 + (py - LOWER_CY) ** 2 <= r2
      ) out.push([px, py])
    }
  }
  return out.slice(0, MAX_DOTS)
})()

// ── 24-hour sliding window via localStorage ───────────────────────
const HISTORY_KEY = 'vllm_req_history'
const DAY_MS      = 86_400_000

interface Snap { ts: number; v: number }

function recordSnap(finished: number) {
  const now  = Date.now()
  const hist: Snap[] = JSON.parse(localStorage.getItem(HISTORY_KEY) ?? '[]')
  const last = hist[hist.length - 1]
  if (last && now - last.ts < 60_000) {
    last.v = finished                           // update in-place, one entry per minute max
  } else {
    hist.push({ ts: now, v: finished })
  }
  // prune anything older than 25 hours
  localStorage.setItem(HISTORY_KEY, JSON.stringify(hist.filter(s => s.ts >= now - DAY_MS - 3_600_000)))
}

function dailyDone(currentFinished: number): number {
  const hist: Snap[] = JSON.parse(localStorage.getItem(HISTORY_KEY) ?? '[]')
  if (hist.length === 0) return 0
  const cutoff  = Date.now() - DAY_MS
  // most-recent entry that is ≥ 24 h old; fall back to oldest available
  const baseline = [...hist].reverse().find(s => s.ts <= cutoff) ?? hist[0]
  return Math.max(0, currentFinished - baseline.v)
}

// ── Prometheus parser ─────────────────────────────────────────────
const PROM_RE = /^([\w:]+)(?:\{[^}]*\})?\s+([\d.e+\-]+)/gm

interface Metrics { running: number; waiting: number; finished: number; gpuCache: number }

function parsePrometheus(text: string): Metrics {
  const m: Metrics = { running: 0, waiting: 0, finished: 0, gpuCache: 0 }
  let match: RegExpExecArray | null
  while ((match = PROM_RE.exec(text)) !== null) {
    const v = parseFloat(match[2])
    switch (match[1]) {
      case 'vllm:num_requests_running':  m.running   = v;        break
      case 'vllm:num_requests_waiting':  m.waiting   = v;        break
      case 'vllm:request_success_total': m.finished += v;        break
      case 'vllm:kv_cache_usage_perc':   m.gpuCache  = v * 100; break
    }
  }
  PROM_RE.lastIndex = 0
  return m
}

function dotColor(i: number, fEnd: number, wEnd: number) {
  if (i < fEnd) return '#42a5f5'
  if (i < wEnd) return '#ff9800'
  return '#00bcd4'
}
function dotClass(i: number, fEnd: number, wEnd: number) {
  if (i < fEnd) return 'vllm-dot-finished'
  if (i < wEnd) return 'vllm-dot-waiting'
  return 'vllm-dot-running'
}

function renderDots(
  el: SVGGElement | null,
  active: [number, number][],
  nF: number, nW: number
) {
  if (!el) return
  const sel = d3.select(el)
    .selectAll<SVGCircleElement, [number, number]>('circle')
    .data(active, (_, i) => i)

  sel.enter().append('circle')
    .attr('r', DOT_RADIUS).attr('cx', d => d[0]).attr('cy', d => d[1]).attr('opacity', 0)
    .merge(sel)
    .attr('fill',  (_, i) => dotColor(i, nF, nF + nW))
    .attr('class', (_, i) => dotClass(i, nF, nF + nW))
    .transition().duration(280).attr('opacity', 1)

  sel.exit().transition().duration(280).attr('opacity', 0).remove()
}

// ── Component ─────────────────────────────────────────────────────
export default function VllmEye() {
  const desktopDotsRef = useRef<SVGGElement>(null)
  const mobileDotsRef  = useRef<SVGGElement>(null)
  const inFlight = useRef(false)
  const [metrics,  setMetrics]  = useState<Metrics | null>(null)
  const [error,    setError]    = useState(false)
  const [progress, setProgress] = useState(0)

  const fetchMetrics = useCallback(async () => {
    if (inFlight.current) return
    inFlight.current = true
    const ctrl = new AbortController()
    const tid  = setTimeout(() => ctrl.abort(), 4000)
    try {
      const res = await fetch('/vllm-metrics', { signal: ctrl.signal })
      clearTimeout(tid)
      if (!res.ok) throw new Error()
      const parsed = parsePrometheus(await res.text())
      recordSnap(parsed.finished)
      setMetrics(parsed)
      setError(false)
    } catch {
      setError(true)
    } finally {
      inFlight.current = false
    }
  }, [])

  useEffect(() => {
    fetchMetrics()
    let elapsed = 0
    const id = setInterval(() => {
      elapsed += 100
      setProgress(Math.min(elapsed / REFRESH_MS, 1))
      if (elapsed >= REFRESH_MS) { elapsed = 0; fetchMetrics() }
    }, 100)
    return () => clearInterval(id)
  }, [fetchMetrics])

  useEffect(() => {
    if (!metrics) return
    const done24 = dailyDone(metrics.finished)
    const total  = metrics.running + metrics.waiting + done24
    const n  = Math.min(total, DOT_POSITIONS.length)
    const nR = Math.min(metrics.running, n)
    const nW = Math.min(metrics.waiting, n - nR)
    const nF = n - nR - nW
    const active = DOT_POSITIONS.slice(0, n)
    renderDots(desktopDotsRef.current, active, nF, nW)
    renderDots(mobileDotsRef.current,  active, nF, nW)
  }, [metrics])

  const statusDot = (
    <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${error ? 'bg-red-500' : 'bg-[#00bcd4] animate-pulse'}`} />
  )

  return (
    <>
      <style>{`
        @keyframes vllm-run  { 0%,100%{opacity:.65} 50%{opacity:1}   }
        @keyframes vllm-wait { 0%,100%{opacity:.55} 50%{opacity:.9}  }
        .vllm-dot-running  { animation: vllm-run  1.6s ease-in-out infinite;      }
        .vllm-dot-waiting  { animation: vllm-wait 2.4s ease-in-out infinite .5s;  }
      `}</style>

      {/* ── DESKTOP: fixed top-left card ──────────────────────────── */}
      <div className="hidden md:flex md:flex-col fixed top-24 left-6 z-40 w-60 rounded-xl border border-slate-700/60 bg-[#0d1117]/90 backdrop-blur-sm shadow-lg overflow-hidden">

        <div className="flex items-center justify-between px-3 pt-2.5 pb-1">
          <span className="text-[9px] font-bold tracking-[0.22em] text-[#00bcd4]/80 uppercase">Insight</span>
          {statusDot}
        </div>

        <div className="px-2" style={{ height: 108 }}>
          <svg viewBox="0 0 660 400" className="w-full h-full">
            <defs>
              <clipPath id="vllm-clip-d">
                <path d={EYE_PATH} />
              </clipPath>
              <radialGradient id="vllm-glow-d" cx="50%" cy="50%" r="50%">
                <stop offset="0%"   stopColor="#00bcd4" stopOpacity="0.07" />
                <stop offset="100%" stopColor="#00bcd4" stopOpacity="0" />
              </radialGradient>
            </defs>
            <ellipse cx="330" cy="200" rx="210" ry="100" fill="url(#vllm-glow-d)" />
            <path d={EYE_PATH} fill="none" stroke="#1f2a33" strokeWidth="1.5" />
            <g ref={desktopDotsRef} clipPath="url(#vllm-clip-d)" />
            <circle cx="330" cy="200" r="46" fill="none" stroke="#0d1117" strokeWidth="1" opacity="0.4" />
            <circle cx="330" cy="200" r="24" fill="#0d1117" opacity="0.6" />
            <circle cx="330" cy="200" r="6"  fill="#00bcd4" opacity="0.2" />
          </svg>
        </div>

        <div className="border-t border-slate-800/80 text-center font-mono py-2">
          <div className="text-[7px] text-slate-500 uppercase tracking-wider">done · 24h</div>
          <div className="text-sm font-bold text-[#42a5f5]">
            {metrics ? dailyDone(metrics.finished) : '—'}
          </div>
        </div>

        <div className="px-3 py-1.5 border-t border-slate-800/80 flex items-center justify-between">
          <span className="text-[7px] text-slate-500 uppercase tracking-wider font-mono">GPU cache</span>
          <span className="text-[10px] font-bold text-slate-300 font-mono">
            {metrics ? `${metrics.gpuCache.toFixed(1)}%` : '—'}
          </span>
        </div>

        <div className="h-0.5 bg-slate-800/60">
          <div className="h-full bg-[#00bcd4]/70 transition-all duration-100" style={{ width: `${progress * 100}%` }} />
        </div>
      </div>

      {/* ── MOBILE: fixed bottom strip ────────────────────────────── */}
      <div className="flex md:hidden fixed bottom-0 left-0 right-0 z-40 border-t border-slate-700/50 bg-[#0d1117]/95 backdrop-blur-md items-center gap-3 px-4 h-14">

        <div className="flex-shrink-0" style={{ width: 72, height: 44 }}>
          <svg viewBox="0 0 660 400" className="w-full h-full">
            <defs>
              <clipPath id="vllm-clip-m">
                <path d={EYE_PATH} />
              </clipPath>
              <radialGradient id="vllm-glow-m" cx="50%" cy="50%" r="50%">
                <stop offset="0%"   stopColor="#00bcd4" stopOpacity="0.07" />
                <stop offset="100%" stopColor="#00bcd4" stopOpacity="0" />
              </radialGradient>
            </defs>
            <ellipse cx="330" cy="200" rx="210" ry="100" fill="url(#vllm-glow-m)" />
            <path d={EYE_PATH} fill="none" stroke="#1f2a33" strokeWidth="1.5" />
            <g ref={mobileDotsRef} clipPath="url(#vllm-clip-m)" />
            <circle cx="330" cy="200" r="46" fill="none" stroke="#0d1117" strokeWidth="1" opacity="0.4" />
            <circle cx="330" cy="200" r="24" fill="#0d1117" opacity="0.6" />
            <circle cx="330" cy="200" r="6"  fill="#00bcd4" opacity="0.2" />
          </svg>
        </div>

        <div className="w-px h-6 bg-slate-700/60 flex-shrink-0" />

        <div className="flex flex-1 items-center justify-around font-mono">
          <div className="text-center">
            <div className="text-[7px] text-slate-500 uppercase tracking-wider">done · 24h</div>
            <div className="text-xs font-bold text-[#42a5f5]">
              {metrics ? dailyDone(metrics.finished) : '—'}
            </div>
          </div>
          <div className="text-center">
            <div className="text-[7px] text-slate-500 uppercase tracking-wider">gpu</div>
            <div className="text-xs font-bold text-slate-300">
              {metrics ? `${metrics.gpuCache.toFixed(0)}%` : '—'}
            </div>
          </div>
        </div>

        {statusDot}

        <div
          className="absolute bottom-0 left-0 h-px bg-[#00bcd4]/60 transition-all duration-100"
          style={{ width: `${progress * 100}%` }}
        />
      </div>
    </>
  )
}
