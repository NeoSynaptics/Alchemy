import { useState, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

// ── Types ──────────────────────────────────────────────────

interface BPComponent {
  [key: string]: string
}

interface ExperimentPhase {
  [key: string]: string
}

interface BPStatus {
  enabled: boolean
  memory_patterns: number
  phase: string
  components: BPComponent
  experiment_plan: ExperimentPhase
}

interface BPStepResult {
  iteration: number
  resolution: string
  nodes: number
  action: string | null
  prediction_confidence: number | null
  error_magnitude: number | null
  refined: boolean
  distilled: boolean
  elapsed_ms: number
  error?: string
  hint?: string
}

// ── Fetch helpers ──────────────────────────────────────────

async function fetchStatus(): Promise<BPStatus> {
  const res = await fetch('/api/v1/brain-physics/status')
  if (!res.ok) throw new Error(`API ${res.status}`)
  return res.json()
}

async function fetchStep(goal: string): Promise<BPStepResult> {
  const res = await fetch('/api/v1/brain-physics/step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ goal }),
  })
  if (!res.ok) throw new Error(`API ${res.status}`)
  return res.json()
}

// ── Status badge ───────────────────────────────────────────

function StatusBadge({ value }: { value: string }) {
  if (value === 'stub')
    return <Badge variant="secondary" className="text-[10px]">stub</Badge>
  if (value === 'active')
    return <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/20 text-[10px]">active</Badge>
  return <Badge variant="outline" className="text-[10px]">{value}</Badge>
}

// ── Phase card ─────────────────────────────────────────────

const PHASE_META: Record<string, { title: string; color: string }> = {
  phase_1: { title: 'Phase 1 — Perception', color: 'border-l-blue-500' },
  phase_2: { title: 'Phase 2 — Spatial Relations', color: 'border-l-cyan-500' },
  phase_3: { title: 'Phase 3 — Physics Sim', color: 'border-l-amber-500' },
  phase_4: { title: 'Phase 4 — Prediction Loop', color: 'border-l-orange-500' },
  phase_5: { title: 'Phase 5 — Consolidation', color: 'border-l-purple-500' },
  phase_6: { title: 'Phase 6 — Benchmark', color: 'border-l-emerald-500' },
}

// ── Page ───────────────────────────────────────────────────

export function BrainPhysicsPage() {
  const [status, setStatus] = useState<BPStatus | null>(null)
  const [stepResult, setStepResult] = useState<BPStepResult | null>(null)
  const [goal, setGoal] = useState('')
  const [loading, setLoading] = useState(false)
  const [stepping, setStepping] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadStatus = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setStatus(await fetchStatus())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch status')
    } finally {
      setLoading(false)
    }
  }, [])

  const runStep = useCallback(async () => {
    setStepping(true)
    setError(null)
    try {
      setStepResult(await fetchStep(goal))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Step failed')
    } finally {
      setStepping(false)
    }
  }, [goal])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">BrainPhysics</h1>
          <p className="text-muted-foreground">
            Coarse-to-fine cognitive router — spatial scene graphs, intuitive physics, predictive processing
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={loadStatus} disabled={loading}>
          {loading ? 'Loading...' : status ? 'Refresh' : 'Connect'}
        </Button>
      </div>

      {error && (
        <Card className="border-red-500/30">
          <CardContent className="pt-6">
            <p className="text-sm text-red-400">{error}</p>
            <p className="text-xs text-muted-foreground mt-1">Make sure Alchemy server is running on :8000</p>
          </CardContent>
        </Card>
      )}

      {/* Architecture overview — always visible */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Architecture</CardTitle>
          <CardDescription>The 6-step cognitive loop</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2 text-xs">
            {[
              { step: '1', label: 'Perceive', desc: 'Coarse-to-fine VLM', color: 'bg-blue-500/15 text-blue-400 border-blue-500/20' },
              { step: '2', label: 'Scene Graph', desc: 'Spatial objects + relations', color: 'bg-cyan-500/15 text-cyan-400 border-cyan-500/20' },
              { step: '3', label: 'Physics Sim', desc: '"What happens if I click X?"', color: 'bg-amber-500/15 text-amber-400 border-amber-500/20' },
              { step: '4', label: 'Predict', desc: 'Compare vs reality', color: 'bg-orange-500/15 text-orange-400 border-orange-500/20' },
              { step: '5', label: 'Refine', desc: 'Loop if error high', color: 'bg-rose-500/15 text-rose-400 border-rose-500/20' },
              { step: '6', label: 'Distill', desc: 'Compress to fast lookup', color: 'bg-purple-500/15 text-purple-400 border-purple-500/20' },
            ].map((s, i) => (
              <div key={s.step} className="flex items-center gap-1.5">
                <Badge className={`${s.color} font-mono`}>{s.step}</Badge>
                <span className="font-medium">{s.label}</span>
                <span className="text-muted-foreground hidden sm:inline">— {s.desc}</span>
                {i < 5 && <span className="text-muted-foreground mx-1">&rarr;</span>}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Component status */}
      {status && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Engine Status</CardTitle>
              <CardDescription>
                {status.enabled ? 'Engine initialized' : 'Engine not running'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Status</span>
                  {status.enabled ? (
                    <Badge className="bg-emerald-500/15 text-emerald-600 border-emerald-500/20">Active</Badge>
                  ) : (
                    <Badge variant="destructive">Disabled</Badge>
                  )}
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Phase</span>
                  <span className="font-mono text-xs">{status.phase}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Memory patterns</span>
                  <span className="font-mono text-xs">{status.memory_patterns}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Components</CardTitle>
              <CardDescription>Pipeline implementation status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {Object.entries(status.components).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">{key.replace(/_/g, ' ')}</span>
                    <StatusBadge value={value} />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Experiment plan */}
      {status && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Experiment Plan</CardTitle>
            <CardDescription>Phases to implement over coming weeks</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {Object.entries(status.experiment_plan).map(([key, desc]) => {
                const meta = PHASE_META[key] || { title: key, color: 'border-l-zinc-500' }
                return (
                  <div key={key} className={`border-l-2 ${meta.color} pl-3 py-1`}>
                    <div className="text-sm font-medium">{meta.title}</div>
                    <div className="text-xs text-muted-foreground">{desc}</div>
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Step runner */}
      {status?.enabled && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Run Step</CardTitle>
            <CardDescription>
              Execute one perceive &rarr; simulate &rarr; predict &rarr; refine cycle
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2">
              <input
                type="text"
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                placeholder='Goal, e.g. "open Chrome"'
                className="flex-1 rounded-md border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                onKeyDown={(e) => e.key === 'Enter' && runStep()}
              />
              <Button onClick={runStep} disabled={stepping} size="sm">
                {stepping ? 'Running...' : 'Step'}
              </Button>
            </div>

            {stepResult && (
              <div className="mt-4 rounded-md border bg-muted/50 p-3 space-y-1 text-xs font-mono">
                {stepResult.error ? (
                  <div className="text-red-400">{stepResult.error}</div>
                ) : (
                  <>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Resolution</span>
                      <span>{stepResult.resolution}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Nodes detected</span>
                      <span>{stepResult.nodes}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Action</span>
                      <span>{stepResult.action ?? 'none'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Prediction confidence</span>
                      <span>{stepResult.prediction_confidence ?? '—'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Error magnitude</span>
                      <span>{stepResult.error_magnitude ?? '—'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Refined</span>
                      <span>{stepResult.refined ? 'yes' : 'no'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Distilled</span>
                      <span>{stepResult.distilled ? 'yes' : 'no'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Elapsed</span>
                      <span>{stepResult.elapsed_ms}ms</span>
                    </div>
                  </>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Models needed */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Model Requirements</CardTitle>
          <CardDescription>Uses existing fleet — no new models needed</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {[
              { model: 'Qwen2.5-VL 7B', role: 'Perception — coarse-to-fine scene extraction', gpu: 'GPU 1', vram: '4.4 GB' },
              { model: 'Qwen3 14B', role: 'Reasoning — physics simulation & prediction refinement', gpu: 'GPU 1', vram: '9 GB' },
              { model: 'nomic-embed-text', role: 'Consolidation — memory pattern lookup', gpu: 'RAM', vram: '—' },
            ].map((m) => (
              <div key={m.model} className="flex items-center gap-3 text-sm">
                <Badge variant="outline" className="text-[10px] font-mono w-28 justify-center">{m.gpu}</Badge>
                <span className="font-medium w-32">{m.model}</span>
                <span className="text-muted-foreground text-xs">{m.role}</span>
                <span className="ml-auto font-mono text-xs text-muted-foreground">{m.vram}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
