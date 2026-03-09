import type { TimelineEvent } from '@/hooks/useMemorySearch'
import type { Bucket } from '@/hooks/useTimelineData'

function timeAgo(ts: number): string {
  const delta = (Date.now() / 1000) - ts
  if (delta < 60) return 'just now'
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`
  return `${Math.floor(delta / 86400)}d ago`
}

const TYPE_BADGE_COLORS: Record<string, string> = {
  screenshot: 'bg-amber-500/20 text-amber-400',
  voice:      'bg-emerald-500/20 text-emerald-400',
  action:     'bg-sky-500/20 text-sky-400',
  search:     'bg-violet-500/20 text-violet-400',
  photo:      'bg-pink-500/20 text-pink-400',
  app_switch: 'bg-zinc-500/20 text-zinc-400',
}

// ── Event Detail ───────────────────────────────────────

function EventDetail({ event, onClose }: { event: TimelineEvent; onClose: () => void }) {
  const badgeColor = TYPE_BADGE_COLORS[event.event_type] || 'bg-zinc-500/20 text-zinc-400'
  const fullDate = new Date(event.ts * 1000).toLocaleString()

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="space-y-1.5">
          <span className={`inline-block px-2 py-0.5 rounded-md text-[11px] font-mono uppercase tracking-wider ${badgeColor}`}>
            {event.event_type}
          </span>
          <p className="text-xs text-zinc-500 font-mono tabular-nums">{fullDate}</p>
          <p className="text-xs text-zinc-600">{timeAgo(event.ts)}</p>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Screenshot */}
      {event.screenshot_url && (
        <div className="relative rounded-xl overflow-hidden bg-zinc-950 border border-zinc-800/40">
          <img
            src={`/api${event.screenshot_url}`}
            alt=""
            loading="lazy"
            className="w-full object-cover"
          />
        </div>
      )}

      {/* Summary */}
      {event.summary && (
        <p className="text-sm text-zinc-300 leading-relaxed">{event.summary}</p>
      )}

      {/* App name */}
      {event.app_name && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-zinc-600 uppercase tracking-wider">App</span>
          <span className="text-xs text-zinc-400">{event.app_name}</span>
        </div>
      )}

      {/* Score */}
      {event.score > 0 && (
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-zinc-600 uppercase tracking-wider">Relevance</span>
          <span className="text-xs text-amber-400 font-mono">{(event.score * 100).toFixed(0)}%</span>
        </div>
      )}

      {/* Tags */}
      {Array.isArray(event.meta?.tags) && (event.meta.tags as string[]).length > 0 && (
        <div className="space-y-1.5">
          <span className="text-[10px] font-mono text-zinc-600 uppercase tracking-wider">Tags</span>
          <div className="flex flex-wrap gap-1.5">
            {(event.meta.tags as string[]).map(tag => (
              <span key={tag} className="px-2 py-0.5 text-[11px] bg-zinc-800/60 text-zinc-400 rounded-md">
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Bucket Detail ──────────────────────────────────────

function BucketDetail({ bucket, onClose }: { bucket: Bucket; onClose: () => void }) {
  const date = new Date(bucket.bucket_ts * 1000).toLocaleDateString('en', {
    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
  })

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-sm font-medium text-zinc-200">{bucket.count} events</p>
          <p className="text-xs text-zinc-500 font-mono">{date}</p>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Type breakdown */}
      <div className="space-y-2">
        <span className="text-[10px] font-mono text-zinc-600 uppercase tracking-wider">Breakdown</span>
        {Object.entries(bucket.types).map(([type, count]) => {
          const badgeColor = TYPE_BADGE_COLORS[type] || 'bg-zinc-500/20 text-zinc-400'
          const pct = ((count / bucket.count) * 100).toFixed(0)
          return (
            <div key={type} className="flex items-center gap-3">
              <span className={`px-2 py-0.5 rounded-md text-[11px] font-mono uppercase tracking-wider ${badgeColor}`}>
                {type}
              </span>
              <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-zinc-500 transition-all"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="text-xs text-zinc-500 font-mono tabular-nums w-8 text-right">
                {count}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Panel ──────────────────────────────────────────────

interface Props {
  event: TimelineEvent | null
  bucket: Bucket | null
  onClose: () => void
}

export function TimelineDetailPanel({ event, bucket, onClose }: Props) {
  const isOpen = event !== null || bucket !== null

  return (
    <div
      className={`
        absolute right-0 top-0 bottom-0 w-80
        bg-zinc-900/95 backdrop-blur-xl
        border-l border-zinc-800/60
        shadow-[-20px_0_60px_rgba(0,0,0,0.5)]
        transition-transform duration-300 ease-out
        overflow-y-auto
        ${isOpen ? 'translate-x-0' : 'translate-x-full'}
      `}
    >
      <div className="p-5">
        {event && <EventDetail event={event} onClose={onClose} />}
        {!event && bucket && <BucketDetail bucket={bucket} onClose={onClose} />}
      </div>
    </div>
  )
}
