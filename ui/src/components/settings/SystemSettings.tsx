import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { useModels } from '@/hooks/useAlchemy'

export function SystemSettings() {
  const { data: models, loading, error } = useModels()

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Ollama Connection</CardTitle>
          <CardDescription>Local LLM server configuration</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="ollama-host">Ollama Host</Label>
              <Input id="ollama-host" defaultValue="http://localhost:11434" />
            </div>

            <div className="space-y-2">
              <Label htmlFor="ollama-keepalive">Keep Alive</Label>
              <Input id="ollama-keepalive" defaultValue="10m" />
            </div>

            <div className="space-y-2">
              <Label htmlFor="ollama-retries">Retry Attempts</Label>
              <Input id="ollama-retries" type="number" min="0" max="10" defaultValue={3} />
            </div>

            <div className="space-y-2">
              <Label htmlFor="ollama-retry-delay">Retry Delay (s)</Label>
              <Input id="ollama-retry-delay" type="number" step="0.5" min="0" max="30" defaultValue={2} />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Loaded Models</CardTitle>
          <CardDescription>
            {loading ? 'Checking...' : error ? 'Backend offline' : `${models?.models.length ?? 0} models available`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {error && (
            <p className="text-sm text-muted-foreground">
              Cannot reach Alchemy API. Start the backend on port 8000.
            </p>
          )}
          {models && (
            <div className="space-y-3">
              <div className="flex flex-wrap gap-2">
                {models.models.map((m) => (
                  <Badge key={m.name} variant="secondary">
                    {m.name} ({m.size_gb.toFixed(1)} GB)
                  </Badge>
                ))}
              </div>
              <p className="text-sm text-muted-foreground">
                RAM: {models.system.available_gb.toFixed(1)} GB free / {models.system.total_gb.toFixed(0)} GB total
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Server</CardTitle>
          <CardDescription>Alchemy API server settings</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="srv-port">Port</Label>
              <Input id="srv-port" type="number" defaultValue={8000} disabled />
            </div>

            <div className="space-y-2">
              <Label htmlFor="srv-log">Log Level</Label>
              <Select defaultValue="INFO">
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
            <Switch id="srv-auth" defaultChecked={false} />
          </div>
        </CardContent>
      </Card>

    </div>
  )
}
