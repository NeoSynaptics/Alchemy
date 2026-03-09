import { useRef, useEffect, useCallback, useState } from 'react'
import type { ViewRange, ZoomLevel } from '@/hooks/useTimelineZoom'
import type { Bucket } from '@/hooks/useTimelineData'
import type { TimelineEvent } from '@/hooks/useMemorySearch'

// ── Color map ──────────────────────────────────────────

const TYPE_COLORS: Record<string, string> = {
  screenshot: '#f59e0b',  // amber
  voice:      '#10b981',  // emerald
  action:     '#0ea5e9',  // sky
  search:     '#8b5cf6',  // violet
  photo:      '#ec4899',  // pink
  app_switch: '#6b7280',  // gray
}

const SEARCH_HIT_COLOR = '#fbbf24'    // amber-400
const AXIS_COLOR = '#3f3f46'          // zinc-700
const LABEL_COLOR = '#a1a1aa'         // zinc-400
const NOW_COLOR = '#10b981'           // emerald-500
const BG_COLOR = '#0c0c0e'

// ── Helpers ────────────────────────────────────────────

function tsToX(ts: number, view: ViewRange, width: number): number {
  return ((ts - view.start) / (view.end - view.start)) * width
}

function formatTickLabel(ts: number, zoomLevel: ZoomLevel): string {
  const d = new Date(ts * 1000)
  switch (zoomLevel) {
    case 0: return d.toLocaleDateString('en', { month: 'short', year: '2-digit' })
    case 1: return d.toLocaleDateString('en', { month: 'short', day: 'numeric' })
    case 2: return d.toLocaleDateString('en', { weekday: 'short', day: 'numeric' })
    case 3: return d.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit' })
    case 4: return d.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }
}

function tickInterval(zoomLevel: ZoomLevel): number {
  switch (zoomLevel) {
    case 0: return 30 * 86400   // monthly
    case 1: return 7 * 86400    // weekly
    case 2: return 86400        // daily
    case 3: return 3600         // hourly
    case 4: return 600          // 10 min
  }
}

// ── Hit target for click detection ─────────────────────

interface HitTarget {
  x: number
  y: number
  radius: number
  event?: TimelineEvent
  bucket?: Bucket
}

// ── Component ──────────────────────────────────────────

interface Props {
  viewRange: ViewRange
  zoomLevel: ZoomLevel
  buckets: Bucket[]
  events: TimelineEvent[]
  searchHitIds: Set<number>
  onZoom: (direction: 'in' | 'out', centerRatio: number) => void
  onPan: (deltaPx: number, canvasWidth: number) => void
  onSelectEvent: (event: TimelineEvent | null) => void
  onSelectBucket: (bucket: Bucket | null) => void
}

export function TimelineCanvas({
  viewRange, zoomLevel, buckets, events, searchHitIds,
  onZoom, onPan, onSelectEvent, onSelectBucket,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 800, height: 400 })
  const dragRef = useRef<{ active: boolean; lastX: number }>({ active: false, lastX: 0 })
  const hitsRef = useRef<HitTarget[]>([])
  const [hoverInfo, setHoverInfo] = useState<{ x: number; y: number; text: string } | null>(null)

  // Resize observer
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      setSize({ width: Math.floor(width), height: Math.floor(height) })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = size.width * dpr
    canvas.height = size.height * dpr
    ctx.scale(dpr, dpr)

    const W = size.width
    const H = size.height
    const AXIS_Y = H - 40
    const PLOT_TOP = 50
    const PLOT_H = AXIS_Y - PLOT_TOP

    // Clear
    ctx.fillStyle = BG_COLOR
    ctx.fillRect(0, 0, W, H)

    // ── Time axis ──
    ctx.strokeStyle = AXIS_COLOR
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, AXIS_Y)
    ctx.lineTo(W, AXIS_Y)
    ctx.stroke()

    // Ticks
    const interval = tickInterval(zoomLevel)
    const firstTick = Math.ceil(viewRange.start / interval) * interval
    ctx.font = '11px ui-monospace, SFMono-Regular, monospace'
    ctx.textAlign = 'center'

    for (let ts = firstTick; ts <= viewRange.end; ts += interval) {
      const x = tsToX(ts, viewRange, W)
      if (x < 0 || x > W) continue

      ctx.strokeStyle = AXIS_COLOR
      ctx.beginPath()
      ctx.moveTo(x, AXIS_Y)
      ctx.lineTo(x, AXIS_Y + 6)
      ctx.stroke()

      // Vertical grid line (subtle)
      ctx.strokeStyle = 'rgba(63, 63, 70, 0.3)'
      ctx.beginPath()
      ctx.moveTo(x, PLOT_TOP)
      ctx.lineTo(x, AXIS_Y)
      ctx.stroke()

      ctx.fillStyle = LABEL_COLOR
      ctx.fillText(formatTickLabel(ts, zoomLevel), x, AXIS_Y + 20)
    }

    // ── "Now" indicator ──
    const nowX = tsToX(Date.now() / 1000, viewRange, W)
    if (nowX >= 0 && nowX <= W) {
      ctx.strokeStyle = NOW_COLOR
      ctx.lineWidth = 2
      ctx.setLineDash([4, 4])
      ctx.beginPath()
      ctx.moveTo(nowX, PLOT_TOP)
      ctx.lineTo(nowX, AXIS_Y)
      ctx.stroke()
      ctx.setLineDash([])
      ctx.fillStyle = NOW_COLOR
      ctx.font = '10px ui-monospace, SFMono-Regular, monospace'
      ctx.fillText('NOW', nowX, PLOT_TOP - 6)
    }

    // ── Render data ──
    const hits: HitTarget[] = []

    if (zoomLevel <= 2 && buckets.length > 0) {
      // Bucket bars
      const maxCount = Math.max(...buckets.map(b => b.count), 1)

      for (const bucket of buckets) {
        const x = tsToX(bucket.bucket_ts, viewRange, W)
        const barWidth = Math.max(3, (tsToX(bucket.bucket_ts + tickInterval(zoomLevel), viewRange, W) - x) * 0.7)
        const barH = (bucket.count / maxCount) * PLOT_H * 0.85
        const barY = AXIS_Y - barH

        if (x + barWidth < 0 || x > W) continue

        // Stacked bar by event type
        let yOffset = 0
        const typeEntries = Object.entries(bucket.types)
        for (const [type, count] of typeEntries) {
          const segH = (count / bucket.count) * barH
          const color = TYPE_COLORS[type] || '#6b7280'
          ctx.fillStyle = color
          ctx.globalAlpha = 0.7
          ctx.beginPath()
          ctx.roundRect(x, barY + yOffset, barWidth, segH, [2, 2, 0, 0])
          ctx.fill()
          yOffset += segH
        }
        ctx.globalAlpha = 1.0

        // Count label
        if (barWidth > 20) {
          ctx.fillStyle = LABEL_COLOR
          ctx.font = '10px ui-monospace, SFMono-Regular, monospace'
          ctx.textAlign = 'center'
          ctx.fillText(bucket.count.toString(), x + barWidth / 2, barY - 4)
        }

        hits.push({
          x: x + barWidth / 2, y: barY + barH / 2,
          radius: Math.max(barWidth, barH) / 2,
          bucket,
        })
      }
    } else if (events.length > 0) {
      // Individual event dots
      const DOT_R = zoomLevel === 4 ? 6 : 4

      // Vertical lane assignment to avoid overlap
      const lanes = new Map<number, number>()
      let laneCount = 0
      const sorted = [...events].sort((a, b) => a.ts - b.ts)

      for (const ev of sorted) {
        const x = tsToX(ev.ts, viewRange, W)
        if (x < -20 || x > W + 20) continue

        // Simple lane: hash event_type to a row
        let lane = lanes.get(ev.event_type.charCodeAt(0) % 5)
        if (lane === undefined) {
          lane = laneCount++
          lanes.set(ev.event_type.charCodeAt(0) % 5, lane)
        }
        const laneH = PLOT_H / Math.max(laneCount + 1, 5)
        const y = PLOT_TOP + laneH * (lane + 0.5)

        const isHit = searchHitIds.has(ev.id)
        const color = TYPE_COLORS[ev.event_type] || '#6b7280'

        // Glow for search hits
        if (isHit) {
          ctx.shadowColor = SEARCH_HIT_COLOR
          ctx.shadowBlur = 12
        }

        ctx.fillStyle = isHit ? SEARCH_HIT_COLOR : color
        ctx.globalAlpha = isHit ? 1.0 : 0.75
        ctx.beginPath()
        ctx.arc(x, y, DOT_R, 0, Math.PI * 2)
        ctx.fill()

        ctx.shadowBlur = 0
        ctx.globalAlpha = 1.0

        // Screenshot indicator (small ring)
        if (ev.screenshot_url) {
          ctx.strokeStyle = color
          ctx.lineWidth = 1.5
          ctx.beginPath()
          ctx.arc(x, y, DOT_R + 3, 0, Math.PI * 2)
          ctx.stroke()
        }

        // Summary text at zoom level 4
        if (zoomLevel === 4 && ev.summary) {
          ctx.fillStyle = LABEL_COLOR
          ctx.font = '11px ui-monospace, SFMono-Regular, monospace'
          ctx.textAlign = 'left'
          const label = ev.summary.length > 60 ? ev.summary.slice(0, 57) + '...' : ev.summary
          ctx.fillText(label, x + DOT_R + 6, y + 4)
        }

        hits.push({ x, y, radius: DOT_R + 4, event: ev })
      }
    }

    hitsRef.current = hits
  }, [size, viewRange, zoomLevel, buckets, events, searchHitIds])

  // ── Wheel zoom ──
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    const centerRatio = (e.clientX - rect.left) / rect.width
    onZoom(e.deltaY > 0 ? 'out' : 'in', centerRatio)
  }, [onZoom])

  // ── Drag pan ──
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    dragRef.current = { active: true, lastX: e.clientX }
  }, [])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (dragRef.current.active) {
      const delta = e.clientX - dragRef.current.lastX
      dragRef.current.lastX = e.clientX
      onPan(delta, size.width)
    } else {
      // Hover detection
      const rect = canvasRef.current?.getBoundingClientRect()
      if (!rect) return
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      const hit = hitsRef.current.find(h =>
        Math.hypot(mx - h.x, my - h.y) < h.radius
      )
      if (hit) {
        const text = hit.event
          ? `${hit.event.summary || hit.event.event_type} — ${new Date(hit.event.ts * 1000).toLocaleString()}`
          : hit.bucket
            ? `${hit.bucket.count} events`
            : ''
        setHoverInfo({ x: mx, y: my, text })
      } else {
        setHoverInfo(null)
      }
    }
  }, [onPan, size.width])

  const handleMouseUp = useCallback(() => {
    dragRef.current.active = false
  }, [])

  // ── Click ──
  const handleClick = useCallback((e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    const hit = hitsRef.current.find(h =>
      Math.hypot(mx - h.x, my - h.y) < h.radius
    )
    if (hit?.event) {
      onSelectEvent(hit.event)
    } else if (hit?.bucket) {
      onSelectBucket(hit.bucket)
    } else {
      onSelectEvent(null)
      onSelectBucket(null)
    }
  }, [onSelectEvent, onSelectBucket])

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[300px] select-none">
      <canvas
        ref={canvasRef}
        width={size.width}
        height={size.height}
        style={{ width: '100%', height: '100%', cursor: dragRef.current.active ? 'grabbing' : 'grab' }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onClick={handleClick}
      />
      {/* Tooltip overlay */}
      {hoverInfo && (
        <div
          className="absolute pointer-events-none z-10 px-3 py-1.5 rounded-lg bg-zinc-800/90 backdrop-blur-sm border border-zinc-700/50 text-xs text-zinc-200 font-mono max-w-xs truncate shadow-lg"
          style={{
            left: Math.min(hoverInfo.x + 12, size.width - 200),
            top: hoverInfo.y - 36,
          }}
        >
          {hoverInfo.text}
        </div>
      )}
    </div>
  )
}
