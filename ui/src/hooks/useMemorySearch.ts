import { useState, useCallback, useEffect } from 'react'

// ── Types ────────────────────────────────────────────────────

export interface TimelineEvent {
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

export interface ContextPack {
  activity: string
  recent: string[]
  apps: string[]
  preferences: Record<string, string>
  generated_at: number
  text_summary: string
}

export interface MemoryHealth {
  status: string
  timeline: { total_events: number; oldest_ts: number | null; by_type: Record<string, number> }
  vectors: { count: number }
  stm: { active_events: number; preferences: number }
  activity: string
  storage_path: string
}

interface SearchLane<T> {
  status: 'idle' | 'loading' | 'done' | 'error'
  results: T[]
}

export interface SearchState {
  stm: SearchLane<TimelineEvent>
  ltm: SearchLane<TimelineEvent>
  synthesis: { status: 'idle' | 'loading' | 'done'; text: string }
}

// ── Fetch helpers ────────────────────────────────────────────

async function memoryFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`/api/v1/memory${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) throw new Error(`Memory API ${res.status}`)
  return res.json()
}

// ── Hooks ────────────────────────────────────────────────────

export function useMemoryHealth(pollMs = 10_000) {
  const [data, setData] = useState<MemoryHealth | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setData(await memoryFetch<MemoryHealth>('/health'))
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed')
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, pollMs)
    return () => clearInterval(id)
  }, [refresh, pollMs])

  return { data, error, refresh }
}

export function useMemoryContext(pollMs = 5_000) {
  const [data, setData] = useState<ContextPack | null>(null)

  useEffect(() => {
    const load = async () => {
      try {
        setData(await memoryFetch<ContextPack>('/stm/context'))
      } catch { /* silent */ }
    }
    load()
    const id = setInterval(load, pollMs)
    return () => clearInterval(id)
  }, [pollMs])

  return data
}

export function useRecentTimeline(limit = 12) {
  const [events, setEvents] = useState<TimelineEvent[]>([])
  const [loading, setLoading] = useState(true)

  const refresh = useCallback(async () => {
    try {
      setEvents(await memoryFetch<TimelineEvent[]>(`/timeline/recent?limit=${limit}`))
    } catch { /* silent */ }
    setLoading(false)
  }, [limit])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 30_000) // refresh every 30s
    return () => clearInterval(id)
  }, [refresh])

  return { events, loading, refresh }
}

export function useMemorySearch() {
  const [query, setQuery] = useState('')
  const [state, setState] = useState<SearchState>({
    stm: { status: 'idle', results: [] },
    ltm: { status: 'idle', results: [] },
    synthesis: { status: 'idle', text: '' },
  })
  const [isSearching, setIsSearching] = useState(false)

  const search = useCallback(async (q: string) => {
    if (!q.trim()) return
    setQuery(q)
    setIsSearching(true)

    setState({
      stm: { status: 'loading', results: [] },
      ltm: { status: 'loading', results: [] },
      synthesis: { status: 'idle', text: '' },
    })

    // Fire timeline query (semantic search across LTM)
    const ltmPromise = memoryFetch<TimelineEvent[]>('/timeline/query', {
      method: 'POST',
      body: JSON.stringify({ query: q, limit: 20, semantic: true }),
    }).then(results => {
      setState(prev => ({ ...prev, ltm: { status: 'done', results } }))
      return results
    }).catch(() => {
      setState(prev => ({ ...prev, ltm: { status: 'error', results: [] } }))
      return []
    })

    // Fire recent STM check
    const stmPromise = memoryFetch<TimelineEvent[]>('/timeline/recent?limit=10')
      .then(all => {
        const filtered = all.filter(e =>
          e.summary.toLowerCase().includes(q.toLowerCase())
        )
        setState(prev => ({ ...prev, stm: { status: 'done', results: filtered } }))
      })
      .catch(() => {
        setState(prev => ({ ...prev, stm: { status: 'error', results: [] } }))
      })

    await Promise.all([ltmPromise, stmPromise])
    setIsSearching(false)
  }, [])

  const clear = useCallback(() => {
    setQuery('')
    setIsSearching(false)
    setState({
      stm: { status: 'idle', results: [] },
      ltm: { status: 'idle', results: [] },
      synthesis: { status: 'idle', text: '' },
    })
  }, [])

  return { query, state, isSearching, search, clear }
}
