import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useVoiceStatus, useVoiceControl } from '@/hooks/useAlchemy'

const MODES = [
  { id: 'conversation', label: 'Chat', icon: '💬' },
  { id: 'command', label: 'Cmd', icon: '⚡' },
  { id: 'dictation', label: 'Dict', icon: '✏️' },
  { id: 'muted', label: 'Mute', icon: '🔇' },
] as const

export function VoiceControls() {
  const { data: voice, error } = useVoiceStatus(3000)
  const control = useVoiceControl()
  const [pending, setPending] = useState(false)

  if (error) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Voice</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Voice system unavailable</p>
        </CardContent>
      </Card>
    )
  }

  const running = voice?.running ?? false

  async function handleToggle() {
    setPending(true)
    try {
      if (running) await control.stop()
      else await control.start()
    } finally {
      setPending(false)
    }
  }

  async function handleMode(mode: string) {
    setPending(true)
    try {
      await control.setMode(mode)
    } finally {
      setPending(false)
    }
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div
              className={`h-2.5 w-2.5 rounded-full ${
                running ? 'bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)]' : 'bg-zinc-500'
              }`}
            />
            <CardTitle className="text-base">Voice</CardTitle>
            {voice && (
              <Badge variant="secondary" className="text-[11px]">
                {voice.pipeline_state}
              </Badge>
            )}
          </div>
          <Button
            size="sm"
            variant={running ? 'destructive' : 'default'}
            onClick={handleToggle}
            disabled={pending}
          >
            {running ? 'Stop' : 'Start'}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex gap-1.5">
          {MODES.map((m) => (
            <Button
              key={m.id}
              size="sm"
              variant={voice?.mode === m.id ? 'default' : 'outline'}
              className="flex-1 text-xs"
              onClick={() => handleMode(m.id)}
              disabled={pending || !running}
            >
              <span className="mr-1">{m.icon}</span>
              {m.label}
            </Button>
          ))}
        </div>
        {voice && (
          <div className="flex gap-3 text-[11px] text-muted-foreground">
            <span>TTS: {voice.tts_engine}</span>
            <span>Wake: {voice.wake_word}</span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
