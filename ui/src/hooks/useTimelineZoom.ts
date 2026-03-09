import { useState, useCallback, useRef } from 'react'

export interface ViewRange {
  start: number // unix timestamp
  end: number   // unix timestamp
}

export type ZoomLevel = 0 | 1 | 2 | 3 | 4

const HOUR = 3600
const DAY = 86400
const MONTH = 30 * DAY
const SIX_MONTHS = 180 * DAY

/** Derive zoom level from visible time span */
function deriveZoomLevel(span: number): ZoomLevel {
  if (span > SIX_MONTHS) return 0
  if (span > MONTH) return 1
  if (span > DAY) return 2
  if (span > HOUR) return 3
  return 4
}

/** Bucket size in seconds for each zoom level */
export function bucketSecondsForLevel(level: ZoomLevel): number {
  switch (level) {
    case 0: return MONTH
    case 1: return DAY
    case 2: return HOUR
    case 3: return 0  // individual events
    case 4: return 0
  }
}

const MIN_SPAN = 300        // 5 minutes minimum view
const ZOOM_FACTOR = 0.15    // 15% per wheel tick

export function useTimelineZoom(initialSpanDays = 7) {
  const now = Date.now() / 1000
  const [viewRange, setViewRange] = useState<ViewRange>({
    start: now - initialSpanDays * DAY,
    end: now,
  })
  const animRef = useRef<number>(0)

  const span = viewRange.end - viewRange.start
  const zoomLevel = deriveZoomLevel(span)

  /** Zoom in/out centered at a proportional position (0-1 across the view) */
  const zoom = useCallback((direction: 'in' | 'out', centerRatio = 0.5) => {
    setViewRange(prev => {
      const span = prev.end - prev.start
      const factor = direction === 'in' ? -ZOOM_FACTOR : ZOOM_FACTOR
      const delta = span * factor
      const centerTs = prev.start + span * centerRatio
      const newSpan = Math.max(MIN_SPAN, span + delta)
      return {
        start: centerTs - newSpan * centerRatio,
        end: centerTs + newSpan * (1 - centerRatio),
      }
    })
  }, [])

  /** Pan by a pixel delta (requires canvasWidth to convert to time) */
  const pan = useCallback((deltaPx: number, canvasWidth: number) => {
    setViewRange(prev => {
      const span = prev.end - prev.start
      const deltaTs = (deltaPx / canvasWidth) * span
      return { start: prev.start - deltaTs, end: prev.end - deltaTs }
    })
  }, [])

  /** Animate to a specific range (for search zoom-to-results) */
  const animateTo = useCallback((target: ViewRange, durationMs = 600) => {
    cancelAnimationFrame(animRef.current)
    const startTime = performance.now()

    setViewRange(current => {
      const from = { ...current }
      const step = (now: number) => {
        const t = Math.min(1, (now - startTime) / durationMs)
        const ease = 1 - (1 - t) * (1 - t) // easeOutQuad
        setViewRange({
          start: from.start + (target.start - from.start) * ease,
          end: from.end + (target.end - from.end) * ease,
        })
        if (t < 1) animRef.current = requestAnimationFrame(step)
      }
      animRef.current = requestAnimationFrame(step)
      return current // don't change yet, animation handles it
    })
  }, [])

  /** Jump to show the last N days */
  const jumpToRecent = useCallback((days: number) => {
    const now = Date.now() / 1000
    animateTo({ start: now - days * DAY, end: now })
  }, [animateTo])

  return {
    viewRange,
    setViewRange,
    zoomLevel,
    span,
    zoom,
    pan,
    animateTo,
    jumpToRecent,
  }
}
