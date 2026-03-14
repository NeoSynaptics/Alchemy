import { useState, useRef, useEffect, useCallback } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useVoiceStatus, useSettings, streamChat } from '@/hooks/useAlchemy'
import { Mic, Send, Zap, MessageSquare, Brain, Cpu, Volume2, Sparkles } from 'lucide-react'

// ── Types ────────────────────────────────────────────────────

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

// ── Helpers ──────────────────────────────────────────────────

/** Resolve a dot-path like "pw.think" against a nested object */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getNestedValue(obj: Record<string, any> | null, path: string): any {
  if (!obj) return undefined
  return path.split('.').reduce((acc, key) => acc?.[key], obj)
}

// ── Status Bar ───────────────────────────────────────────────

function StatusBar() {
  const { data: voice } = useVoiceStatus(3000)
  const isListening = voice?.running && voice.pipeline_state !== 'idle'
  const isRunning = voice?.running

  return (
    <div className="flex items-center justify-center py-6">
      <div className="flex items-center gap-3 rounded-full border bg-card px-5 py-2.5 shadow-sm">
        {isListening ? (
          <>
            <div className="flex items-end gap-[3px] h-4">
              {[1, 2, 3, 4, 5].map((i) => (
                <span
                  key={i}
                  className="w-[3px] rounded-full bg-emerald-400"
                  style={{
                    animation: `audioBar 1.2s ease-in-out ${i * 0.1}s infinite`,
                  }}
                />
              ))}
            </div>
            <span className="text-sm font-medium text-emerald-400">Alchemy is listening...</span>
          </>
        ) : isRunning ? (
          <>
            <span className="relative flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
            </span>
            <span className="text-sm font-medium">Alchemy is ready</span>
          </>
        ) : (
          <>
            <span className="h-2.5 w-2.5 rounded-full bg-muted-foreground/40" />
            <span className="text-sm text-muted-foreground">Voice offline</span>
          </>
        )}

        {voice?.mode && (
          <Badge variant="secondary" className="ml-1 text-[11px]">
            {voice.mode}
          </Badge>
        )}
      </div>
    </div>
  )
}

// ── Chat Section ─────────────────────────────────────────────

function InlineChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const conversationId = useRef(crypto.randomUUID())
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || streaming) return
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    setStreaming(true)
    setMessages((prev) => [...prev, { role: 'assistant', content: '' }])

    try {
      for await (const chunk of streamChat(text, conversationId.current)) {
        if (chunk.content) {
          setMessages((prev) => {
            const updated = [...prev]
            const last = updated[updated.length - 1]
            if (last.role === 'assistant') {
              updated[updated.length - 1] = { ...last, content: last.content + chunk.content }
            }
            return updated
          })
        }
      }
    } catch (e) {
      setMessages((prev) => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last.role === 'assistant' && !last.content) {
          updated[updated.length - 1] = {
            ...last,
            content: `Error: ${e instanceof Error ? e.message : 'Connection failed'}`,
          }
        }
        return updated
      })
    } finally {
      setStreaming(false)
      inputRef.current?.focus()
    }
  }, [input, streaming])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="mx-auto w-full max-w-2xl space-y-4">
      {/* Input */}
      <div className="relative">
        <Input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Tell me what to change..."
          className="h-12 rounded-xl bg-card pl-4 pr-24 text-sm shadow-sm"
          disabled={streaming}
        />
        <div className="absolute right-2 top-1/2 flex -translate-y-1/2 gap-1">
          <Button size="icon" variant="ghost" className="h-8 w-8 text-muted-foreground" disabled>
            <Mic className="h-4 w-4" />
          </Button>
          <Button
            size="icon"
            className="h-8 w-8"
            onClick={handleSend}
            disabled={!input.trim() || streaming}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Messages */}
      {messages.length > 0 && (
        <div ref={scrollRef} className="max-h-80 space-y-3 overflow-y-auto rounded-xl border bg-card p-4">
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[85%] rounded-xl px-3.5 py-2.5 text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-foreground'
                }`}
              >
                <p className="whitespace-pre-wrap break-words">{msg.content}</p>
                {msg.role === 'assistant' && streaming && i === messages.length - 1 && (
                  <span className="ml-0.5 inline-block h-4 w-1 animate-pulse bg-foreground/50" />
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Setting Card ─────────────────────────────────────────────

interface ToggleCardProps {
  icon: React.ReactNode
  title: string
  description: string
  checked: boolean
  onCheckedChange: (v: boolean) => void
  disabled?: boolean
}

function ToggleCard({ icon, title, description, checked, onCheckedChange, disabled }: ToggleCardProps) {
  return (
    <Card className="transition-colors hover:bg-accent/30">
      <CardContent className="flex items-center gap-4 p-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted">
          {icon}
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium">{title}</p>
          <p className="text-xs text-muted-foreground">{description}</p>
        </div>
        <Switch checked={checked} onCheckedChange={onCheckedChange} disabled={disabled} />
      </CardContent>
    </Card>
  )
}

interface SelectCardProps {
  icon: React.ReactNode
  title: string
  description: string
  value: string
  options: { value: string; label: string }[]
  onValueChange: (v: string) => void
  disabled?: boolean
}

function SelectCard({ icon, title, description, value, options, onValueChange, disabled }: SelectCardProps) {
  return (
    <Card className="transition-colors hover:bg-accent/30">
      <CardContent className="flex items-center gap-4 p-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted">
          {icon}
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium">{title}</p>
          <p className="text-xs text-muted-foreground">{description}</p>
        </div>
        <Select value={value} onValueChange={onValueChange} disabled={disabled}>
          <SelectTrigger className="w-36">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {options.map((o) => (
              <SelectItem key={o.value} value={o.value}>
                {o.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </CardContent>
    </Card>
  )
}

// ── Manual Controls ──────────────────────────────────────────

function ManualControls() {
  const { data, loading, update } = useSettings()

  const get = (path: string) => getNestedValue(data, path)

  return (
    <div className="mx-auto w-full max-w-2xl">
      <p className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">
        Quick Controls
      </p>
      <div className="grid gap-3 sm:grid-cols-2">
        <ToggleCard
          icon={<Zap className="h-5 w-5 text-amber-400" />}
          title="Auto-Pilot Mode"
          description="Allow AI to take actions on your behalf"
          checked={get('pw.approval_enabled') === false}
          onCheckedChange={(v) => update('pw.approval_enabled', !v)}
          disabled={loading}
        />
        <ToggleCard
          icon={<Volume2 className="h-5 w-5 text-sky-400" />}
          title="Voice Output"
          description="Respond with spoken audio along with text"
          checked={!!get('voice.enabled')}
          onCheckedChange={(v) => update('voice.enabled', v)}
          disabled={loading}
        />
        <ToggleCard
          icon={<Brain className="h-5 w-5 text-purple-400" />}
          title="Context Memory"
          description="Remember details from past interactions"
          checked={!!get('voice.knowledge_enabled')}
          onCheckedChange={(v) => update('voice.knowledge_enabled', v)}
          disabled={loading}
        />
        <ToggleCard
          icon={<Sparkles className="h-5 w-5 text-rose-400" />}
          title="Think Mode"
          description="Enable deep reasoning (slower but smarter)"
          checked={!!get('pw.think')}
          onCheckedChange={(v) => update('pw.think', v)}
          disabled={loading}
        />
        <SelectCard
          icon={<Cpu className="h-5 w-5 text-emerald-400" />}
          title="GPU Mode"
          description="GPU allocation strategy"
          value={get('voice.gpu_mode') ?? 'single'}
          options={[
            { value: 'single', label: 'Single GPU' },
            { value: 'dual', label: 'Dual GPU' },
          ]}
          onValueChange={(v) => update('voice.gpu_mode', v)}
          disabled={loading}
        />
        <SelectCard
          icon={<MessageSquare className="h-5 w-5 text-teal-400" />}
          title="TTS Engine"
          description="Text-to-speech provider"
          value={get('voice.tts_engine') ?? 'piper'}
          options={[
            { value: 'piper', label: 'Piper' },
            { value: 'fish', label: 'Fish Speech' },
            { value: 'kokoro', label: 'Kokoro' },
          ]}
          onValueChange={(v) => update('voice.tts_engine', v)}
          disabled={loading}
        />
      </div>
    </div>
  )
}

// ── Page ─────────────────────────────────────────────────────

export function SettingsPage() {
  return (
    <div className="space-y-2 pb-12">
      {/* Status Bar */}
      <StatusBar />

      {/* Chat Input + Messages */}
      <InlineChat />

      {/* Spacer */}
      <div className="py-4" />

      {/* Manual Controls */}
      <ManualControls />
    </div>
  )
}
