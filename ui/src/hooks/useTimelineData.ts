import { useState, useEffect, useRef, useCallback } from 'react'
import { type ZoomLevel, bucketSecondsForLevel } from './useTimelineZoom'
import { type TimelineEvent } from './useMemorySearch'

// ── Types ──────────────────────────────────────────────

export interface Bucket {
  bucket_ts: number
  count: number
  types: Record<string, number>
}

export interface TimelineData {
  buckets: Bucket[]
  events: TimelineEvent[]
  loading: boolean
}

// ── Fetch helpers ──────────────────────────────────────

async function fetchBuckets(
  startTs: number, endTs: number, bucketSeconds: number
): Promise<Bucket[]> {
  const params = new URLSearchParams({
    start_ts: startTs.toString(),
    end_ts: endTs.toString(),
    bucket_seconds: bucketSeconds.toString(),
  })
  const res = await fetch(`/api/v1/memory/timeline/buckets?${params}`)
  if (!res.ok) return []
  return res.json()
}

async function fetchEvents(
  startTs: number, endTs: number, limit: number
): Promise<TimelineEvent[]> {
  const res = await fetch('/api/v1/memory/timeline/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ start_ts: startTs, end_ts: endTs, limit, semantic: false }),
  })
  if (!res.ok) return []
  return res.json()
}

// ── Hook ───────────────────────────────────────────────

export function useTimelineData(
  startTs: number,
  endTs: number,
  zoomLevel: ZoomLevel,
) {
  const [data, setData] = useState<TimelineData>({
    buckets: [],
    events: [],
    loading: false,
  })
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined)
  const cacheRef = useRef<Map<string, Bucket[] | TimelineEvent[]>>(new Map())

  const load = useCallback(async () => {
    const bucketSeconds = bucketSecondsForLevel(zoomLevel)
    const cacheKey = `${Math.floor(startTs / 60)}_${Math.floor(endTs / 60)}_${zoomLevel}`

    const cached = cacheRef.current.get(cacheKey)
    if (cached) {
      if (zoomLevel <= 2) {
        setData({ buckets: cached as Bucket[], events: [], loading: false })
      } else {
        setData({ buckets: [], events: cached as TimelineEvent[], loading: false })
      }
      return
    }

    setData(prev => ({ ...prev, loading: true }))

    if (zoomLevel <= 2 && bucketSeconds > 0) {
      const buckets = await fetchBuckets(startTs, endTs, bucketSeconds)
      cacheRef.current.set(cacheKey, buckets)
      // Keep cache bounded
      if (cacheRef.current.size > 50) {
        const first = cacheRef.current.keys().next().value
        if (first) cacheRef.current.delete(first)
      }
      setData({ buckets, events: [], loading: false })
    } else {
      const events = await fetchEvents(startTs, endTs, 200)
      cacheRef.current.set(cacheKey, events)
      if (cacheRef.current.size > 50) {
        const first = cacheRef.current.keys().next().value
        if (first) cacheRef.current.delete(first)
      }
      setData({ buckets: [], events, loading: false })
    }
  }, [startTs, endTs, zoomLevel])

  useEffect(() => {
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(load, 150)
    return () => clearTimeout(debounceRef.current)
  }, [load])

  return data
}
