import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { useApuStatus } from '@/hooks/useAlchemy'

function LocationBadge({ location }: { location: string }) {
  if (location === 'vram')
    return <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/20 text-[10px]">VRAM</Badge>
  if (location === 'ram')
    return <Badge className="bg-yellow-500/15 text-yellow-600 border-yellow-500/20 text-[10px]">RAM</Badge>
  return <Badge variant="secondary" className="text-[10px]">Disk</Badge>
}

export function SystemSettings() {
  const { data: apu, loading, error } = useApuStatus()

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Ollama Connection</CardTitle>
          <CardDescription>
            Local LLM server configuration
            <Badge variant="outline" className="ml-2 text-[10px]">Read-only</Badge>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="ollama-host">Ollama Host</Label>
              <Input id="ollama-host" defaultValue="http://localhost:11434" disabled />
            </div>

            <div className="space-y-2">
              <Label htmlFor="ollama-keepalive">Keep Alive</Label>
              <Input id="ollama-keepalive" defaultValue="10m" disabled />
            </div>

            <div className="space-y-2">
              <Label htmlFor="ollama-retries">Retry Attempts</Label>
              <Input id="ollama-retries" type="number" defaultValue={3} disabled />
            </div>

            <div className="space-y-2">
              <Label htmlFor="ollama-retry-delay">Retry Delay (s)</Label>
              <Input id="ollama-retry-delay" type="number" defaultValue={2} disabled />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Model Fleet</CardTitle>
          <CardDescription>
            {loading ? 'Loading...' : error ? 'Backend offline' : `${apu?.models.length ?? 0} models registered`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {error && (
            <p className="text-sm text-muted-foreground">
              Cannot reach Alchemy API. Start the backend on port 8000.
            </p>
          )}
          {apu && (
            <div className="space-y-4">
              {/* GPUs */}
              <div className="flex flex-wrap gap-3">
                {apu.gpus.map((gpu) => (
                  <div key={gpu.index} className="rounded-lg border p-3 text-sm">
                    <p className="font-medium">GPU {gpu.index}: {gpu.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(gpu.used_vram_mb / 1024).toFixed(1)} / {(gpu.total_vram_mb / 1024).toFixed(1)} GB VRAM
                    </p>
                  </div>
                ))}
              </div>

              {/* Models grouped by location */}
              {['vram', 'ram', 'disk'].map((loc) => {
                const group = apu.models.filter((m) => m.current_location === loc)
                if (group.length === 0) return null
                return (
                  <div key={loc}>
                    <p className="mb-2 text-xs font-semibold uppercase text-muted-foreground">{loc}</p>
                    <div className="flex flex-wrap gap-2">
                      {group.map((m) => (
                        <Badge key={m.name} variant="outline" className="gap-1.5">
                          <LocationBadge location={m.current_location} />
                          {m.display_name || m.name}
                          {m.vram_mb > 0 && (
                            <span className="text-muted-foreground">
                              {(m.vram_mb / 1024).toFixed(1)}G
                            </span>
                          )}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )
              })}

              {/* RAM */}
              <p className="text-sm text-muted-foreground">
                System RAM: {(apu.ram.available_mb / 1024).toFixed(1)} GB free / {(apu.ram.total_mb / 1024).toFixed(0)} GB total
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Server</CardTitle>
          <CardDescription>
            Alchemy API server settings
            <Badge variant="outline" className="ml-2 text-[10px]">Read-only</Badge>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="srv-port">Port</Label>
              <Input id="srv-port" type="number" defaultValue={8000} disabled />
            </div>

            <div className="space-y-2">
              <Label htmlFor="srv-log">Log Level</Label>
              <Select defaultValue="INFO" disabled>
                <SelectTrigger id="srv-log">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="DEBUG">DEBUG</SelectItem>
                  <SelectItem value="INFO">INFO</SelectItem>
                  <SelectItem value="WARNING">WARNING</SelectItem>
                  <SelectItem value="ERROR">ERROR</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="srv-auth">Require Auth</Label>
            <Switch id="srv-auth" defaultChecked={false} disabled />
          </div>
        </CardContent>
      </Card>

    </div>
  )
}
