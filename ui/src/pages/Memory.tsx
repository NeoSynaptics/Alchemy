import { useState, useEffect, useCallback, useMemo, useRef, type KeyboardEvent } from 'react'

// ── Types ──────────────────────────────────────────────

interface TimelineEvent {
  id: number
  ts: number
  event_type: string
  source: string
  summary: string
  app_name: string
  screenshot_url: string | null
  score: number
  meta: Record<string, unknown>
}

interface DayGroup {
  date: string
  ts: number
  label: string
  count: number
  events: TimelineEvent[]
}

interface MonthGroup {
  key: string
  label: string
  ts: number
  count: number
  events: TimelineEvent[]
  days: DayGroup[]
}

// ── Constants ──────────────────────────────────────────

const DAY = 86400
const MONTH = 30 * DAY
const ZOOM_FACTOR = 0.2

// ── API ────────────────────────────────────────────────

async function fetchAllPhotos(): Promise<TimelineEvent[]> {
  const res = await fetch('/api/v1/memory/timeline/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event_types: ['photo'], limit: 2000, semantic: false }),
  })
  if (!res.ok) return []
  return res.json()
}

async function searchPhotos(query: string): Promise<TimelineEvent[]> {
  const res = await fetch('/api/v1/memory/timeline/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, limit: 200, semantic: true, event_types: ['photo'] }),
  })
  if (!res.ok) return []
  const all: TimelineEvent[] = await res.json()
  // Only keep results with meaningful relevance scores
  return all.filter(e => e.score >= 0.55)
}

// ── Grouping ───────────────────────────────────────────

function groupByDay(events: TimelineEvent[]): DayGroup[] {
  const map = new Map<string, TimelineEvent[]>()
  for (const ev of events) {
    const d = new Date(ev.ts * 1000)
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
    ;(map.get(key) || (map.set(key, []), map.get(key)!)).push(ev)
  }
  return Array.from(map.entries()).sort((a, b) => a[0].localeCompare(b[0])).map(([date, evts]) => {
    const d = new Date(date + 'T12:00:00')
    return { date, ts: d.getTime() / 1000, label: d.toLocaleDateString('en', { month: 'short', day: 'numeric' }), count: evts.length, events: evts.sort((a, b) => b.ts - a.ts) }
  })
}

function groupByMonth(events: TimelineEvent[]): MonthGroup[] {
  const days = groupByDay(events)
  const map = new Map<string, { events: TimelineEvent[]; days: DayGroup[] }>()
  for (const day of days) {
    const key = day.date.slice(0, 7)
    const entry = map.get(key) || { events: [], days: [] }
    entry.events.push(...day.events)
    entry.days.push(day)
    map.set(key, entry)
  }
  return Array.from(map.entries()).sort((a, b) => a[0].localeCompare(b[0])).map(([key, { events: evts, days: d }]) => {
    const date = new Date(key + '-15T12:00:00')
    return { key, label: date.toLocaleDateString('en', { month: 'short', year: 'numeric' }), ts: date.getTime() / 1000, count: evts.length, events: evts, days: d }
  })
}

type ZoomView = 'years' | 'months' | 'weeks' | 'days'
function deriveView(span: number): ZoomView {
  if (span > 365 * DAY) return 'years'
  if (span > 60 * DAY) return 'months'
  if (span > 10 * DAY) return 'weeks'
  return 'days'
}

// ── Stack Icon ─────────────────────────────────────────

function StackIcon({ count, size = 28 }: { count: number; size?: number }) {
  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox="0 0 28 28" fill="none" className="text-zinc-500">
        <rect x="4" y="14" width="16" height="11" rx="2" stroke="currentColor" strokeWidth="1.2" />
        <rect x="6" y="10" width="16" height="11" rx="2" stroke="currentColor" strokeWidth="1.2" fill="#18181b" />
        <rect x="8" y="6" width="16" height="11" rx="2" stroke="currentColor" strokeWidth="1.2" fill="#18181b" />
      </svg>
      <span className="absolute bottom-0 right-0 text-[8px] font-mono font-bold text-amber-400 bg-zinc-900 rounded px-0.5 leading-tight">{count}</span>
    </div>
  )
}

// ── Photo Viewer ───────────────────────────────────────

function PhotoViewer({ event, onClose, onPrev, onNext }: {
  event: TimelineEvent; onClose: () => void; onPrev?: () => void; onNext?: () => void
}) {
  useEffect(() => {
    const h = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowLeft' && onPrev) onPrev()
      if (e.key === 'ArrowRight' && onNext) onNext()
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [onClose, onPrev, onNext])

  const fullDate = new Date(event.ts * 1000).toLocaleDateString('en', {
    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-md" onClick={onClose}>
      <div className="relative max-w-[85vw] max-h-[85vh] flex flex-col items-center gap-3" onClick={e => e.stopPropagation()}>
        {onPrev && <button onClick={onPrev} className="absolute left-[-48px] top-1/2 -translate-y-1/2 p-2 rounded-full bg-zinc-800/80 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M15 18l-6-6 6-6" /></svg></button>}
        {onNext && <button onClick={onNext} className="absolute right-[-48px] top-1/2 -translate-y-1/2 p-2 rounded-full bg-zinc-800/80 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M9 18l6-6-6-6" /></svg></button>}
        <button onClick={onClose} className="absolute top-[-40px] right-0 p-1.5 rounded-lg text-zinc-500 hover:text-white transition-colors"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M18 6L6 18M6 6l12 12" /></svg></button>
        {event.screenshot_url && <img src={`/api${event.screenshot_url}`} alt="" className="max-w-full max-h-[75vh] rounded-xl object-contain" />}
        <div className="flex flex-col gap-1.5 px-4 py-2.5 rounded-xl bg-zinc-900/80 border border-zinc-800/60 max-w-2xl">
          <div className="flex items-center gap-3">
            <span className="text-xs text-zinc-400 font-mono">{fullDate}</span>
            {event.meta?.vlm_status === 'done' && <span className="text-[9px] px-1.5 py-0.5 bg-emerald-500/15 text-emerald-400 rounded font-mono">classified</span>}
            {event.meta?.vlm_status === 'pending' && <span className="text-[9px] px-1.5 py-0.5 bg-zinc-500/15 text-zinc-500 rounded font-mono">pending</span>}
            {event.meta?.vlm_status === 'failed' && <span className="text-[9px] px-1.5 py-0.5 bg-red-500/15 text-red-400 rounded font-mono">failed</span>}
          </div>
          {event.summary && <p className="text-xs text-zinc-300 leading-relaxed">{event.summary}</p>}
          {Array.isArray(event.meta?.tags) && (event.meta.tags as string[]).length > 0 && (
            <div className="flex flex-wrap gap-1">
              {(event.meta.tags as string[]).map(tag => (
                <span key={tag} className="text-[10px] px-1.5 py-0.5 bg-amber-500/15 text-amber-400/80 rounded">{tag}</span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Expanded Day Panel ─────────────────────────────────

function ExpandedDayPanel({ group, onClose, onViewPhoto, anchorX, maxHeight }: {
  group: DayGroup; onClose: () => void; onViewPhoto: (ev: TimelineEvent) => void; anchorX: number; maxHeight: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) onClose() }
    setTimeout(() => window.addEventListener('mousedown', h), 10)
    return () => window.removeEventListener('mousedown', h)
  }, [onClose])
  useEffect(() => {
    const h = (e: globalThis.KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [onClose])

  const w = Math.min(480, window.innerWidth - 40)
  const left = Math.max(12, Math.min(anchorX - w / 2, window.innerWidth - w - 12))
  const fullDate = new Date(group.date + 'T12:00:00').toLocaleDateString('en', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })

  return (
    <div ref={ref} className="absolute z-50 bg-zinc-900/98 backdrop-blur-2xl border border-zinc-700/50 rounded-2xl shadow-[0_-20px_80px_rgba(0,0,0,0.6)] overflow-hidden"
      style={{ left, bottom: 56, width: w, maxHeight: maxHeight - 70, animation: 'slideUp 200ms ease-out' }}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/60">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-medium text-zinc-200">{fullDate}</h3>
          <span className="text-[10px] font-mono text-amber-400/80 bg-amber-400/10 px-1.5 py-0.5 rounded">{group.count} photo{group.count !== 1 ? 's' : ''}</span>
        </div>
        <button onClick={onClose} className="p-1 rounded-lg text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M18 6L6 18M6 6l12 12" /></svg>
        </button>
      </div>
      <div className="p-3 overflow-y-auto" style={{ maxHeight: maxHeight - 130 }}>
        <div className="grid grid-cols-4 sm:grid-cols-5 md:grid-cols-6 gap-2">
          {group.events.map(ev => {
            const tags = Array.isArray(ev.meta?.tags) ? (ev.meta.tags as string[]) : []
            return (
              <button key={ev.id} onClick={() => onViewPhoto(ev)}
                className="aspect-square rounded-lg overflow-hidden bg-zinc-800 border border-zinc-700/30 hover:border-amber-500/50 hover:scale-[1.03] transition-all duration-150 group relative" title={ev.summary || undefined}>
                {ev.screenshot_url ? <img src={`/api${ev.screenshot_url}`} alt="" loading="lazy" className="w-full h-full object-cover" /> :
                  <div className="w-full h-full flex items-center justify-center text-zinc-600"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="m21 15-5-5L5 21" /></svg></div>}
                {/* Classification dot */}
                {ev.summary && ev.meta?.vlm_status === 'done' && <div className="absolute top-1 right-1 w-2 h-2 rounded-full bg-emerald-400 shadow-sm" />}
                {/* Hover overlay with summary + tags */}
                {(ev.summary || tags.length > 0) && (
                  <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity p-1.5">
                    {ev.summary && <p className="text-[9px] text-zinc-200 leading-tight line-clamp-2">{ev.summary}</p>}
                    {tags.length > 0 && (
                      <div className="flex flex-wrap gap-0.5 mt-0.5">
                        {tags.slice(0, 2).map(tag => (
                          <span key={tag} className="text-[8px] px-1 py-px bg-amber-500/20 text-amber-300 rounded">{tag}</span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// ── Search Bar ─────────────────────────────────────────

function SearchBar({ onSearch, onClear, isSearching, resultCount }: {
  onSearch: (q: string) => void; onClear: () => void; isSearching: boolean; resultCount: number | null
}) {
  const [value, setValue] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const hasSearched = resultCount !== null
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = e.target.value
    setValue(v)
    if (!v.trim() && hasSearched) onClear()
  }
  const handleClearClick = () => { setValue(''); onClear(); inputRef.current?.focus() }
  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && value.trim()) onSearch(value.trim())
    if (e.key === 'Escape') { setValue(''); onClear() }
  }
  useEffect(() => {
    const h = (e: globalThis.KeyboardEvent) => { if (e.key === '/' && document.activeElement !== inputRef.current) { e.preventDefault(); inputRef.current?.focus() } }
    window.addEventListener('keydown', h); return () => window.removeEventListener('keydown', h)
  }, [])

  return (
    <div className="relative w-full max-w-md">
      <div className="relative flex items-center bg-zinc-900/60 backdrop-blur-xl border border-zinc-700/50 rounded-xl transition-all duration-300 focus-within:border-zinc-500/60">
        <div className="pl-3 pr-2 text-zinc-500">
          {isSearching ? <div className="w-3.5 h-3.5 border-2 border-zinc-600 border-t-zinc-300 rounded-full animate-spin" /> :
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></svg>}
        </div>
        <input ref={inputRef} type="text" value={value} onChange={handleChange} onKeyDown={handleKeyDown}
          placeholder="Search photos..." className="flex-1 py-2.5 pr-3 bg-transparent text-zinc-100 text-sm placeholder:text-zinc-600 focus:outline-none tracking-wide" />
        {resultCount !== null && <span className="pr-3 text-[11px] font-mono text-amber-400/80 tabular-nums whitespace-nowrap">{resultCount}</span>}
        {hasSearched ? (
          <button onClick={handleClearClick} className="pr-3 text-zinc-500 hover:text-zinc-200 transition-colors" title="Clear search">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18" /><path d="m6 6 12 12" /></svg>
          </button>
        ) : (
          <div className="pr-3 text-zinc-600 text-[10px] font-mono tracking-widest select-none">{value ? 'ENTER' : '/'}</div>
        )}
      </div>
    </div>
  )
}

// ── Search Photo Strip ────────────────────────────────

function SearchPhotoStrip({ results, onViewPhoto }: {
  results: TimelineEvent[]; onViewPhoto: (ev: TimelineEvent, all: TimelineEvent[]) => void
}) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [canScrollL, setCanScrollL] = useState(false)
  const [canScrollR, setCanScrollR] = useState(false)

  const updateScroll = useCallback(() => {
    const el = scrollRef.current; if (!el) return
    setCanScrollL(el.scrollLeft > 2)
    setCanScrollR(el.scrollLeft < el.scrollWidth - el.clientWidth - 2)
  }, [])

  useEffect(() => {
    const el = scrollRef.current; if (!el) return
    updateScroll()
    el.addEventListener('scroll', updateScroll, { passive: true })
    const ro = new ResizeObserver(updateScroll); ro.observe(el)
    return () => { el.removeEventListener('scroll', updateScroll); ro.disconnect() }
  }, [updateScroll, results])

  useEffect(() => { scrollRef.current?.scrollTo({ left: 0 }) }, [results])

  const scroll = useCallback((dir: 1 | -1) => {
    scrollRef.current?.scrollBy({ left: dir * 400, behavior: 'smooth' })
  }, [])

  useEffect(() => {
    const el = scrollRef.current; if (!el) return
    const h = (e: WheelEvent) => {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) { e.preventDefault(); el.scrollLeft += e.deltaY }
    }
    el.addEventListener('wheel', h, { passive: false })
    return () => el.removeEventListener('wheel', h)
  }, [])

  if (results.length === 0) return null

  return (
    <div className="relative flex-shrink-0 border-b border-zinc-800/40 bg-zinc-950/60 backdrop-blur-sm group/strip"
      style={{ animation: 'slideUp 250ms ease-out' }}>
      {canScrollL && <div className="absolute left-0 top-0 bottom-0 w-16 bg-gradient-to-r from-zinc-950 to-transparent z-10 pointer-events-none" />}
      {canScrollR && <div className="absolute right-0 top-0 bottom-0 w-16 bg-gradient-to-l from-zinc-950 to-transparent z-10 pointer-events-none" />}

      {canScrollL && (
        <button onClick={() => scroll(-1)}
          className="absolute left-2 top-1/2 -translate-y-1/2 z-20 p-1.5 rounded-full bg-zinc-800/90 border border-zinc-700/50 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-all opacity-0 group-hover/strip:opacity-100">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M15 18l-6-6 6-6" /></svg>
        </button>
      )}
      {canScrollR && (
        <button onClick={() => scroll(1)}
          className="absolute right-2 top-1/2 -translate-y-1/2 z-20 p-1.5 rounded-full bg-zinc-800/90 border border-zinc-700/50 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-all opacity-0 group-hover/strip:opacity-100">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M9 18l6-6-6-6" /></svg>
        </button>
      )}

      <div className="absolute top-2 right-3 z-20 text-[10px] font-mono text-amber-400/70 bg-zinc-900/80 border border-zinc-800/50 px-2 py-0.5 rounded-full">
        {results.length} result{results.length !== 1 ? 's' : ''}
      </div>

      <div ref={scrollRef} className="flex items-center gap-2 px-4 py-3 overflow-x-auto" style={{ scrollbarWidth: 'none' }}>
        {results.map(ev => {
          const d = new Date(ev.ts * 1000)
          const dateStr = d.toLocaleDateString('en', { month: 'short', day: 'numeric', year: '2-digit' })
          return (
            <button key={ev.id} onClick={() => onViewPhoto(ev, results)}
              className="flex-shrink-0 group/card relative rounded-xl overflow-hidden border border-zinc-700/30 hover:border-amber-500/50 transition-all duration-200 hover:scale-[1.04] hover:shadow-[0_0_20px_rgba(245,158,11,0.1)]"
              style={{ width: 120, height: 96 }}>
              {ev.screenshot_url ? (
                <img src={`/api${ev.screenshot_url}`} alt="" loading="lazy" className="w-full h-full object-cover" />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-zinc-800 text-zinc-600">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="m21 15-5-5L5 21" /></svg>
                </div>
              )}
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent opacity-0 group-hover/card:opacity-100 transition-opacity duration-200">
                <div className="absolute bottom-0 left-0 right-0 p-2">
                  {ev.summary && <p className="text-[8px] text-zinc-200 leading-tight line-clamp-2 mb-1">{ev.summary}</p>}
                  <span className="text-[8px] font-mono text-zinc-400">{dateStr}</span>
                </div>
              </div>
              {ev.score > 0 && (
                <div className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full" style={{
                  backgroundColor: ev.score > 0.7 ? '#34d399' : ev.score > 0.5 ? '#fbbf24' : '#6b7280',
                }} />
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ── Main Page ──────────────────────────────────────────

export function MemoryPage() {
  const [allEvents, setAllEvents] = useState<TimelineEvent[]>([])
  const [searchResults, setSearchResults] = useState<TimelineEvent[] | null>(null)
  const [isSearching, setIsSearching] = useState(false)
  const [loading, setLoading] = useState(true)
  const [viewingPhoto, setViewingPhoto] = useState<{ event: TimelineEvent; dayEvents: TimelineEvent[] } | null>(null)
  const [expandedDay, setExpandedDay] = useState<{ group: DayGroup; anchorX: number } | null>(null)
  const [viewStart, setViewStart] = useState(0)
  const [viewEnd, setViewEnd] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef({ active: false, lastX: 0 })
  const [containerW, setContainerW] = useState(1200)
  const [containerH, setContainerH] = useState(600)

  const viewSpan = viewEnd - viewStart
  const zoomView = deriveView(viewSpan)

  // Load
  useEffect(() => { fetchAllPhotos().then(evts => { setAllEvents(evts); setLoading(false) }) }, [])

  // Initial view range
  useEffect(() => {
    if (allEvents.length > 0 && viewStart === 0) {
      const min = Math.min(...allEvents.map(e => e.ts))
      const max = Math.max(...allEvents.map(e => e.ts))
      const pad = (max - min) * 0.05 || DAY
      setViewStart(min - pad)
      setViewEnd(max + pad)
    }
  }, [allEvents]) // eslint-disable-line

  // Resize
  useEffect(() => {
    const el = containerRef.current; if (!el) return
    const ro = new ResizeObserver(entries => { setContainerW(entries[0].contentRect.width); setContainerH(entries[0].contentRect.height) })
    ro.observe(el); return () => ro.disconnect()
  }, [])

  const handleSearch = useCallback(async (q: string) => {
    setIsSearching(true)
    const results = await searchPhotos(q)
    setSearchResults(results)
    setIsSearching(false)
    if (results.length > 0) {
      const min = Math.min(...results.map(e => e.ts))
      const max = Math.max(...results.map(e => e.ts))
      const pad = (max - min) * 0.1 || 30 * DAY
      setViewStart(min - pad); setViewEnd(max + pad)
    }
  }, [])

  const handleClear = useCallback(() => {
    setSearchResults(null)
    if (allEvents.length > 0) {
      const min = Math.min(...allEvents.map(e => e.ts))
      const max = Math.max(...allEvents.map(e => e.ts))
      const pad = (max - min) * 0.05; setViewStart(min - pad); setViewEnd(max + pad)
    }
  }, [allEvents])

  const activeEvents = searchResults ?? allEvents
  const totalPhotos = activeEvents.length
  const classified = activeEvents.filter(e => e.summary).length

  // Build columns positioned by timestamp
  const columns = useMemo(() => {
    if (activeEvents.length === 0 || viewSpan <= 0) return []
    const visible = activeEvents.filter(e => e.ts >= viewStart && e.ts <= viewEnd)

    if (zoomView === 'years' || zoomView === 'months') {
      return groupByMonth(visible).map(m => ({
        key: m.key, label: m.label.split(' ')[0], subLabel: m.label.split(' ')[1],
        ts: m.ts, events: m.events, dayGroup: undefined as DayGroup | undefined,
        maxShow: zoomView === 'years' ? 2 : 3,
      }))
    }
    return groupByDay(visible).map(d => ({
      key: d.date, label: d.label, subLabel: undefined as string | undefined,
      ts: d.ts, events: d.events, dayGroup: d,
      maxShow: zoomView === 'weeks' ? 4 : 8,
    }))
  }, [activeEvents, viewStart, viewEnd, zoomView, viewSpan])

  // Thumb size based on zoom
  const thumbSize = zoomView === 'years' ? 32 : zoomView === 'months' ? 40 : zoomView === 'weeks' ? 48 : 56

  // Position helper
  const tsToX = useCallback((ts: number) => ((ts - viewStart) / viewSpan) * containerW, [viewStart, viewSpan, containerW])

  // Wheel zoom
  useEffect(() => {
    const el = containerRef.current; if (!el) return
    const handler = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const ratio = (e.clientX - rect.left) / rect.width
      const span = viewEnd - viewStart
      const factor = e.deltaY > 0 ? ZOOM_FACTOR : -ZOOM_FACTOR
      const newSpan = Math.max(5 * DAY, Math.min(12 * 365 * DAY, span * (1 + factor)))
      const center = viewStart + span * ratio
      setViewStart(center - newSpan * ratio)
      setViewEnd(center + newSpan * (1 - ratio))
      setExpandedDay(null)
    }
    el.addEventListener('wheel', handler, { passive: false })
    return () => el.removeEventListener('wheel', handler)
  }, [viewStart, viewEnd])

  // Drag pan
  const handleMouseDown = useCallback((e: React.MouseEvent) => { dragRef.current = { active: true, lastX: e.clientX } }, [])
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current.active) return
    const dx = e.clientX - dragRef.current.lastX
    dragRef.current.lastX = e.clientX
    const dt = -(dx / containerW) * viewSpan
    setViewStart(s => s + dt); setViewEnd(s => s + dt)
  }, [containerW, viewSpan])
  const handleMouseUp = useCallback(() => { dragRef.current.active = false }, [])

  // Jump presets
  const jumpTo = useCallback((days: number) => {
    const now = Date.now() / 1000
    setViewStart(now - days * DAY); setViewEnd(now); setExpandedDay(null)
  }, [])
  const jumpAll = useCallback(() => {
    if (allEvents.length === 0) return
    const min = Math.min(...allEvents.map(e => e.ts)); const max = Math.max(...allEvents.map(e => e.ts))
    const pad = (max - min) * 0.05; setViewStart(min - pad); setViewEnd(max + pad); setExpandedDay(null)
  }, [allEvents])

  // Keyboard
  useEffect(() => {
    const h = (e: globalThis.KeyboardEvent) => {
      if (document.activeElement?.tagName === 'INPUT') return
      const span = viewEnd - viewStart
      if (e.key === 'ArrowLeft') { setViewStart(s => s - span * 0.1); setViewEnd(s => s - span * 0.1) }
      if (e.key === 'ArrowRight') { setViewStart(s => s + span * 0.1); setViewEnd(s => s + span * 0.1) }
      if (e.key === '+' || e.key === '=') { const d = span * 0.15; setViewStart(s => s + d); setViewEnd(s => s - d) }
      if (e.key === '-') { const d = span * 0.15; setViewStart(s => s - d); setViewEnd(s => s + d) }
      if (e.key === 'Escape') setExpandedDay(null)
    }
    window.addEventListener('keydown', h); return () => window.removeEventListener('keydown', h)
  }, [viewStart, viewEnd])

  // Viewer nav
  const viewerNav = useMemo(() => {
    if (!viewingPhoto) return { onPrev: undefined, onNext: undefined }
    const { event, dayEvents } = viewingPhoto
    const idx = dayEvents.findIndex(e => e.id === event.id)
    return {
      onPrev: idx > 0 ? () => setViewingPhoto({ event: dayEvents[idx - 1], dayEvents }) : undefined,
      onNext: idx < dayEvents.length - 1 ? () => setViewingPhoto({ event: dayEvents[idx + 1], dayEvents }) : undefined,
    }
  }, [viewingPhoto])

  const handleExpandDay = useCallback((group: DayGroup, anchorX: number) => {
    if (group.count === 1) setViewingPhoto({ event: group.events[0], dayEvents: group.events })
    else setExpandedDay(prev => prev?.group.date === group.date ? null : { group, anchorX })
  }, [])

  // Timeline ticks
  const ticks = useMemo(() => {
    if (viewSpan <= 0) return []
    const result: { ts: number; label: string; x: number }[] = []
    let interval: number; let fmt: Intl.DateTimeFormatOptions
    if (viewSpan > 3 * 365 * DAY) { interval = 365 * DAY; fmt = { year: 'numeric' } }
    else if (viewSpan > 365 * DAY) { interval = 3 * MONTH; fmt = { month: 'short', year: '2-digit' } }
    else if (viewSpan > 90 * DAY) { interval = MONTH; fmt = { month: 'short' } }
    else if (viewSpan > 14 * DAY) { interval = 7 * DAY; fmt = { month: 'short', day: 'numeric' } }
    else { interval = DAY; fmt = { weekday: 'short', day: 'numeric' } }
    const first = Math.ceil(viewStart / interval) * interval
    for (let ts = first; ts <= viewEnd; ts += interval) {
      result.push({ ts, label: new Date(ts * 1000).toLocaleDateString('en', fmt), x: ((ts - viewStart) / viewSpan) * 100 })
    }
    return result
  }, [viewStart, viewEnd, viewSpan])

  // Strip photo viewer handler
  const handleStripViewPhoto = useCallback((ev: TimelineEvent, all: TimelineEvent[]) => {
    setViewingPhoto({ event: ev, dayEvents: all })
  }, [])

  // Area height available for photos (container minus axis bar and controls)
  const photoAreaH = containerH - 52

  return (
    <div className="h-[calc(100vh-4rem)] flex flex-col bg-[#0a0a0c] overflow-hidden relative">
      <style>{`@keyframes slideUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }`}</style>

      {/* Top bar */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-800/40 flex-shrink-0">
        <div className="flex items-center gap-4">
          <h1 className="text-sm font-medium text-zinc-300 tracking-wide">Memory</h1>
          <SearchBar onSearch={handleSearch} onClear={handleClear} isSearching={isSearching} resultCount={searchResults ? searchResults.length : null} />
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono tracking-wider">
          <span className="text-zinc-500">{totalPhotos} photos</span>
          <span className="text-zinc-800">·</span>
          <span className="text-emerald-500/80">{classified} classified</span>
          {totalPhotos - classified > 0 && <>
            <span className="text-zinc-800">·</span>
            <span className="text-amber-500/70">{totalPhotos - classified} pending</span>
          </>}
          <span className="text-zinc-800">·</span>
          <span className="text-zinc-600 uppercase">{zoomView}</span>
        </div>
      </div>

      {/* Search results photo strip */}
      {searchResults && searchResults.length > 0 && (
        <SearchPhotoStrip results={searchResults} onViewPhoto={handleStripViewPhoto} />
      )}

      {/* Timeline container */}
      <div ref={containerRef} className="flex-1 relative overflow-hidden select-none"
        style={{ cursor: dragRef.current.active ? 'grabbing' : 'grab' }}
        onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>

        {loading ? (
          <div className="flex items-center justify-center h-full"><div className="w-5 h-5 border-2 border-zinc-600 border-t-zinc-300 rounded-full animate-spin" /></div>
        ) : columns.length === 0 ? (
          <div className="flex items-center justify-center h-full text-zinc-600 text-sm">{searchResults ? 'No photos match' : 'No photos in view'}</div>
        ) : (
          /* Photo columns — absolutely positioned by timestamp */
          columns.map(col => {
            const x = tsToX(col.ts)
            const colW = Math.max(thumbSize + 8, 50)
            if (x < -colW || x > containerW + colW) return null

            const showStack = col.events.length > col.maxShow
            const photosToShow = showStack ? col.events.slice(0, col.maxShow - 1) : col.events.slice(0, col.maxShow)

            return (
              <div key={col.key} className="absolute flex flex-col items-center"
                style={{ left: x - colW / 2, bottom: 52, width: colW }}>

                {/* Photos stacking upward */}
                <div className="flex flex-col-reverse items-center gap-1" style={{ maxHeight: photoAreaH - 20 }}>
                  {showStack && (
                    <button onClick={() => col.dayGroup && handleExpandDay(col.dayGroup, x)}
                      className="flex-shrink-0 hover:scale-110 transition-transform cursor-pointer"
                      title={`${col.events.length} photos`}>
                      <StackIcon count={col.events.length} size={Math.max(20, thumbSize * 0.65)} />
                    </button>
                  )}
                  {photosToShow.map(ev => {
                    const tags = Array.isArray(ev.meta?.tags) ? (ev.meta.tags as string[]) : []
                    const isClassified = ev.summary && ev.meta?.vlm_status === 'done'
                    const showLabel = (zoomView === 'days' || zoomView === 'weeks') && ev.summary

                    return (
                      <div key={ev.id} className="flex-shrink-0 flex items-end gap-1.5">
                        <button
                          onClick={() => {
                            if (col.events.length === 1 || !col.dayGroup) setViewingPhoto({ event: ev, dayEvents: col.events })
                            else handleExpandDay(col.dayGroup, x)
                          }}
                          className="flex-shrink-0 rounded-md overflow-hidden bg-zinc-800 border border-zinc-700/30 hover:border-amber-500/50 hover:scale-105 transition-all duration-150 relative"
                          style={{ width: thumbSize, height: thumbSize }} title={ev.summary || undefined}>
                          {ev.screenshot_url ? <img src={`/api${ev.screenshot_url}`} alt="" loading="lazy" className="w-full h-full object-cover" /> :
                            <div className="w-full h-full flex items-center justify-center text-zinc-600">
                              <svg width={thumbSize * 0.4} height={thumbSize * 0.4} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="8.5" cy="8.5" r="1.5" /><path d="m21 15-5-5L5 21" /></svg>
                            </div>}
                          {/* Classification indicator dot */}
                          {isClassified && <div className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-emerald-400" />}
                        </button>
                        {/* Side label with summary + tags */}
                        {showLabel && (
                          <div className="max-w-[140px] min-w-0">
                            <p className="text-[9px] text-zinc-400 leading-tight line-clamp-2">{ev.summary}</p>
                            {tags.length > 0 && (
                              <div className="flex flex-wrap gap-0.5 mt-0.5">
                                {tags.slice(0, 3).map(tag => (
                                  <span key={tag} className="text-[8px] px-1 py-px bg-amber-500/15 text-amber-400/80 rounded">{tag}</span>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>

                {/* Label */}
                <div className="mt-1 text-center flex-shrink-0">
                  <p className="text-[9px] font-mono text-zinc-500 leading-tight whitespace-nowrap">{col.label}</p>
                  {col.subLabel && <p className="text-[8px] font-mono text-zinc-600 leading-tight">{col.subLabel}</p>}
                </div>
              </div>
            )
          })
        )}

        {/* Timeline axis */}
        <div className="absolute bottom-0 left-0 right-0 h-[52px]">
          {/* Axis line */}
          <div className="absolute top-0 left-0 right-0 h-px bg-zinc-700/60" />

          {/* Ticks */}
          {ticks.map((t, i) => (
            <div key={i} className="absolute top-0 flex flex-col items-center" style={{ left: `${t.x}%`, transform: 'translateX(-50%)' }}>
              <div className="w-px h-2 bg-zinc-600" />
              <span className="text-[9px] font-mono text-zinc-500 mt-1 whitespace-nowrap">{t.label}</span>
            </div>
          ))}

          {/* Now */}
          {(() => {
            const nx = ((Date.now() / 1000 - viewStart) / viewSpan) * 100
            if (nx < 0 || nx > 100) return null
            return <div className="absolute top-0" style={{ left: `${nx}%` }}>
              <div className="w-0.5 h-5 bg-emerald-500/60 mx-auto" />
              <span className="text-[8px] font-mono text-emerald-400 -ml-2">NOW</span>
            </div>
          })()}

          {/* Controls */}
          <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-4 py-1">
            <div className="flex items-center gap-1">
              {[{ l: '1M', d: 30 }, { l: '3M', d: 90 }, { l: '1Y', d: 365 }, { l: '5Y', d: 5 * 365 }, { l: 'ALL', d: 0 }].map(({ l, d }) => (
                <button key={l} onClick={() => d === 0 ? jumpAll() : jumpTo(d)}
                  className="px-2 py-0.5 text-[10px] font-mono text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 rounded transition-colors uppercase tracking-wider">{l}</button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <button onClick={() => { const d = viewSpan * 0.15; setViewStart(s => s + d); setViewEnd(s => s - d) }}
                className="p-1 text-zinc-500 hover:text-zinc-300 transition-colors">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
              </button>
              <button onClick={() => { const d = viewSpan * 0.15; setViewStart(s => s - d); setViewEnd(s => s + d) }}
                className="p-1 text-zinc-500 hover:text-zinc-300 transition-colors">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="5" y1="12" x2="19" y2="12" /></svg>
              </button>
            </div>
          </div>
        </div>

        {/* Expanded day */}
        {expandedDay && (
          <ExpandedDayPanel group={expandedDay.group} anchorX={expandedDay.anchorX} maxHeight={containerH}
            onClose={() => setExpandedDay(null)} onViewPhoto={ev => setViewingPhoto({ event: ev, dayEvents: expandedDay.group.events })} />
        )}
      </div>

      {/* Photo viewer */}
      {viewingPhoto && <PhotoViewer event={viewingPhoto.event} onClose={() => setViewingPhoto(null)} onPrev={viewerNav.onPrev} onNext={viewerNav.onNext} />}
    </div>
  )
}
