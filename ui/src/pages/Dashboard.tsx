import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { VoiceControls } from '@/components/VoiceControls'
import { useHealth, useApuStatus, useModules } from '@/hooks/useAlchemy'

function VramBar({ used, total, label }: { used: number; total: number; label: string }) {
  const pct = total > 0 ? Math.round((used / total) * 100) : 0
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono">
          {(used / 1024).toFixed(1)} / {(total / 1024).toFixed(1)} GB
          <span className="ml-1 text-muted-foreground">({pct}%)</span>
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-muted">
        <div
          className={`h-2 rounded-full transition-all ${
            pct > 90 ? 'bg-red-500' : pct > 70 ? 'bg-yellow-500' : 'bg-emerald-500'
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

function LocationBadge({ location }: { location: string }) {
  if (location === 'vram')
    return <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/20 text-[10px]">VRAM</Badge>
  if (location === 'ram')
    return <Badge className="bg-yellow-500/15 text-yellow-600 border-yellow-500/20 text-[10px]">RAM</Badge>
  return <Badge variant="secondary" className="text-[10px]">Disk</Badge>
}

function TierBadge({ tier }: { tier: string }) {
  const colors: Record<string, string> = {
    core: 'bg-blue-500/15 text-blue-400 border-blue-500/20',
    infra: 'bg-purple-500/15 text-purple-400 border-purple-500/20',
    app: 'bg-zinc-500/15 text-zinc-400 border-zinc-500/20',
  }
  return <Badge className={`${colors[tier] || colors.app} text-[10px]`}>{tier}</Badge>
}

export function DashboardPage() {
  const { data: health, error: healthError } = useHealth()
  const { data: apu, loading: apuLoading, refresh: refreshApu } = useApuStatus()
  const { data: modules } = useModules()

  const online = !healthError && health?.status === 'ok'

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Alchemy</h1>
          <p className="text-muted-foreground">System overview</p>
        </div>
        <Button variant="outline" size="sm" onClick={refreshApu}>
          Refresh
        </Button>
      </div>

      {/* Status row */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* Backend */}
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Backend</CardDescription>
            <CardTitle className="flex items-center gap-2 text-lg">
              {online ? (
                <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/20">Online</Badge>
              ) : (
                <Badge variant="destructive">Offline</Badge>
              )}
              <span className="text-sm font-normal text-muted-foreground">:8000</span>
            </CardTitle>
          </CardHeader>
        </Card>

        {/* Voice */}
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Voice</CardDescription>
            <CardTitle className="text-lg">
              {health?.voice_enabled ? (
                <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/20">Enabled</Badge>
              ) : (
                <Badge variant="secondary">Disabled</Badge>
              )}
            </CardTitle>
          </CardHeader>
        </Card>

        {/* GPUs */}
        {apu?.gpus.map((gpu) => (
          <Card key={gpu.index}>
            <CardHeader className="pb-2">
              <CardDescription>GPU {gpu.index}</CardDescription>
              <CardTitle className="text-sm font-medium">{gpu.name}</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <VramBar used={gpu.used_vram_mb} total={gpu.total_vram_mb} label="VRAM" />
              <div className="mt-2 flex gap-3 text-[11px] text-muted-foreground">
                <span>{gpu.temperature_c}°C</span>
                <span>{gpu.utilization_pct}% util</span>
              </div>
            </CardContent>
          </Card>
        ))}

        {apuLoading && !apu && (
          <>
            <Card><CardContent className="pt-6"><Skeleton className="h-16" /></CardContent></Card>
            <Card><CardContent className="pt-6"><Skeleton className="h-16" /></CardContent></Card>
          </>
        )}
      </div>

      {/* RAM */}
      {apu && (
        <Card>
          <CardContent className="pt-6">
            <VramBar used={apu.ram.used_mb} total={apu.ram.total_mb} label="System RAM" />
          </CardContent>
        </Card>
      )}

      {/* Module Contracts */}
      {modules && modules.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Modules</CardTitle>
            <CardDescription>
              {modules.filter((m) => m.contract_satisfied).length}/{modules.length} contracts satisfied
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {modules.map((m) => (
                <div key={m.id} className="flex items-center gap-3 text-sm">
                  <div
                    className={`h-2 w-2 rounded-full ${
                      m.contract_satisfied ? 'bg-emerald-500' : 'bg-red-500'
                    }`}
                  />
                  <span className="w-36 font-medium">{m.name}</span>
                  <TierBadge tier={m.tier} />
                  {m.contract_missing.length > 0 && (
                    <span className="text-xs text-red-400">
                      Missing: {m.contract_missing.join(', ')}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Model Fleet */}
      {apu && apu.models.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Model Fleet</CardTitle>
            <CardDescription>{apu.models.length} models registered</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-xs text-muted-foreground">
                    <th className="pb-2 pr-4">Model</th>
                    <th className="pb-2 pr-4">Location</th>
                    <th className="pb-2 pr-4">Tier</th>
                    <th className="pb-2 pr-4 text-right">VRAM</th>
                    <th className="pb-2">Capabilities</th>
                  </tr>
                </thead>
                <tbody>
                  {apu.models.map((m) => (
                    <tr key={m.name} className="border-b border-border/50">
                      <td className="py-2 pr-4 font-medium">{m.display_name || m.name}</td>
                      <td className="py-2 pr-4"><LocationBadge location={m.current_location} /></td>
                      <td className="py-2 pr-4"><Badge variant="outline" className="text-[10px]">{m.current_tier}</Badge></td>
                      <td className="py-2 pr-4 text-right font-mono text-xs">
                        {m.vram_mb > 0 ? `${(m.vram_mb / 1024).toFixed(1)} GB` : '—'}
                      </td>
                      <td className="py-2">
                        <div className="flex flex-wrap gap-1">
                          {m.capabilities.map((c) => (
                            <Badge key={c} variant="secondary" className="text-[10px]">{c}</Badge>
                          ))}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Voice Controls */}
      <VoiceControls />
    </div>
  )
}
