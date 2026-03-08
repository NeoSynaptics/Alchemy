import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { VoiceControls } from '@/components/VoiceControls'
import { useVoiceStatus } from '@/hooks/useAlchemy'

export function VoiceSettings() {
  const { data: voice } = useVoiceStatus(5000)

  return (
    <div className="space-y-6">
      <VoiceControls />

      <Card>
        <CardHeader>
          <CardTitle>Voice Configuration</CardTitle>
          <CardDescription>Current voice pipeline state</CardDescription>
        </CardHeader>
        <CardContent>
          {voice ? (
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Pipeline State</p>
                <Badge variant="secondary">{voice.pipeline_state}</Badge>
              </div>
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">TTS Engine</p>
                <Badge variant="secondary">{voice.tts_engine}</Badge>
              </div>
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Wake Word</p>
                <Badge variant="secondary">{voice.wake_word}</Badge>
              </div>
              <div className="space-y-1">
                <p className="text-xs text-muted-foreground">Mode</p>
                <Badge variant="secondary">{voice.mode}</Badge>
              </div>
              {voice.conversation_id && (
                <div className="space-y-1 sm:col-span-2">
                  <p className="text-xs text-muted-foreground">Conversation</p>
                  <p className="font-mono text-xs">{voice.conversation_id}</p>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Voice system not available</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Voice Modes</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 text-sm">
            <div>
              <p className="font-medium">💬 Conversation</p>
              <p className="text-muted-foreground">Full voice chat — listen, think, speak</p>
            </div>
            <div>
              <p className="font-medium">⚡ Command</p>
              <p className="text-muted-foreground">Short commands — execute actions, no conversation</p>
            </div>
            <div>
              <p className="font-medium">✏️ Dictation</p>
              <p className="text-muted-foreground">Speech-to-text only — transcribe without responding</p>
            </div>
            <div>
              <p className="font-medium">🔇 Muted</p>
              <p className="text-muted-foreground">Pipeline running but mic disabled</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
