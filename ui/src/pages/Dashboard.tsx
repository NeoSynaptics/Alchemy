import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useModels, useShadowHealth } from '@/hooks/useAlchemy'
import { Button } from '@/components/ui/button'

export function DashboardPage() {
  const { data: models, loading: modelsLoading, error: modelsError, refresh: refreshModels } = useModels()
  const { data: shadow, loading: shadowLoading, refresh: refreshShadow } = useShadowHealth()

  const backendOnline = !modelsError
  const shadowOnline = shadow?.status === 'running'

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">AlchemyFrontDev</h1>
          <p className="text-muted-foreground">Agent engine status overview</p>
        </div>
        <Button variant="outline" onClick={() => { refreshModels(); refreshShadow() }}>
          Refresh
        </Button>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Backend API</CardDescription>
            <CardTitle className="flex items-center gap-2">
              {modelsLoading ? (
                <Badge variant="secondary">Checking...</Badge>
              ) : backendOnline ? (
                <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/20">Online</Badge>
              ) : (
                <Badge variant="destructive">Offline</Badge>
              )}
              <span className="text-sm font-normal text-muted-foreground">:8000</span>
            </CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Shadow Desktop</CardDescription>
            <CardTitle className="flex items-center gap-2">
              {shadowLoading ? (
                <Badge variant="secondary">Checking...</Badge>
              ) : shadowOnline ? (
                <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/20">Running</Badge>
              ) : (
                <Badge variant="secondary">Stopped</Badge>
              )}
            </CardTitle>
          </CardHeader>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>System RAM</CardDescription>
            <CardTitle className="text-lg">
              {models
                ? `${models.system.available_gb.toFixed(0)} GB free / ${models.system.total_gb.toFixed(0)} GB`
                : '—'}
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      {models && models.models.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Loaded Models</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {models.models.map((m) => (
                <Badge key={m.name} variant="outline" className="text-sm">
                  {m.name} — {m.size_gb.toFixed(1)} GB
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
