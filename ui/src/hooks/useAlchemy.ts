import { useState, useEffect, useCallback } from 'react'

// ── Types ────────────────────────────────────────────────────

export interface AlchemyModels {
  models: { name: string; size_gb: number }[]
  system: { total_gb: number; available_gb: number }
}

export interface TaskStatus {
  task_id: string
  status: string
  current_step: number
  max_steps: number
  steps: {
    step: number
    action: string
    inference_ms: number
    execution_ms: number
  }[]
}

export interface HealthStatus {
  status: string
  version?: string
  ollama_connected: boolean
  voice_enabled: boolean
  connect_enabled: boolean
  connect_devices: number
  gpu_orchestrator: boolean
  gate_enabled: boolean
  desktop_agent: boolean
  research_enabled: boolean
  playwright_agent: boolean
}

export interface GPUInfo {
  index: number
  name: string
  total_vram_mb: number
  used_vram_mb: number
  free_vram_mb: number
  temperature_c: number
  utilization_pct: number
}

export interface RAMInfo {
  total_mb: number
  used_mb: number
  free_mb: number
  available_mb: number
}

export interface ModelCard {
  name: string
  display_name: string
  backend: string
  vram_mb: number
  ram_mb: number
  disk_mb: number
  preferred_gpu: number | null
  default_tier: string
  current_tier: string
  current_location: string
  capabilities: string[]
  last_used: string | null
  owner_app: string | null
}

export interface StackStatus {
  gpus: GPUInfo[]
  ram: RAMInfo
  models: ModelCard[]
  mode: string
}

export interface ModuleInfo {
  id: string
  name: string
  description: string
  tier: string
  contract_satisfied: boolean
  contract_missing: string[]
  contract_optional_missing: string[]
}

export interface VoiceStatus {
  running: boolean
  mode: string
  pipeline_state: string
  tts_engine: string
  wake_word: string
  conversation_id: string | null
}

// ── Fetch helpers ────────────────────────────────────────────

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`/api/v1${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

async function apiRawFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

// ── Generic hook factory ─────────────────────────────────────

function useApiFetch<T>(fetcher: () => Promise<T>, pollMs?: number) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setData(await fetcher())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Fetch failed')
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [fetcher])

  useEffect(() => { refresh() }, [refresh])

  useEffect(() => {
    if (!pollMs) return
    const id = setInterval(refresh, pollMs)
    return () => clearInterval(id)
  }, [pollMs, refresh])

  return { data, loading, error, refresh }
}

// ── Hooks ────────────────────────────────────────────────────

export function useModels() {
  const fetcher = useCallback(() => apiFetch<AlchemyModels>('/models'), [])
  return useApiFetch(fetcher)
}

export function useTaskStatus(taskId: string | null) {
  const fetcher = useCallback(
    () => apiFetch<TaskStatus>(`/playwright/task/${taskId}/status`),
    [taskId],
  )
  const result = useApiFetch(taskId ? fetcher : async () => null as any)
  return { data: taskId ? result.data : null, loading: result.loading, refresh: result.refresh }
}

export function useHealth() {
  const fetcher = useCallback(() => apiRawFetch<HealthStatus>('/health'), [])
  return useApiFetch(fetcher, 10_000)
}

export function useApuStatus() {
  const fetcher = useCallback(() => apiFetch<StackStatus>('/apu/status'), [])
  return useApiFetch(fetcher)
}

export function useModules() {
  const fetcher = useCallback(() => apiFetch<ModuleInfo[]>('/modules'), [])
  return useApiFetch(fetcher)
}

export function useVoiceStatus(pollMs = 5000) {
  const fetcher = useCallback(() => apiFetch<VoiceStatus>('/voice/status'), [])
  return useApiFetch(fetcher, pollMs)
}

export function useVoiceControl() {
  const start = useCallback(
    () => apiFetch<unknown>('/voice/start', { method: 'POST' }),
    [],
  )
  const stop = useCallback(
    () => apiFetch<unknown>('/voice/stop', { method: 'POST' }),
    [],
  )
  const setMode = useCallback(
    (mode: string) =>
      apiFetch<unknown>('/voice/mode', {
        method: 'POST',
        body: JSON.stringify({ mode }),
      }),
    [],
  )
  return { start, stop, setMode }
}

// ── Settings ─────────────────────────────────────────────────

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function useSettings() {
  const fetcher = useCallback(() => apiFetch<Record<string, any>>('/settings'), [])
  const { data, loading, error, refresh } = useApiFetch(fetcher)

  const update = useCallback(
    async (path: string, value: unknown) => {
      // Build nested object from dot path: "pw.think" → { pw: { think: value } }
      const keys = path.split('.')
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let payload: any = value
      for (let i = keys.length - 1; i >= 0; i--) {
        payload = { [keys[i]]: payload }
      }
      await apiFetch('/settings', {
        method: 'PATCH',
        body: JSON.stringify(payload),
      })
      refresh()
    },
    [refresh],
  )

  return { data, loading, error, refresh, update }
}

// ── SSE Streaming Chat ──────────────────────────────────────

export async function* streamChat(
  message: string,
): AsyncGenerator<{ content: string; done: boolean; model?: string }> {
  const res = await fetch('/api/v1/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, source: 'ui' }),
  })
  if (!res.ok) {
    throw new Error(`Chat ${res.status}: ${await res.text()}`)
  }
  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop()!
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          yield JSON.parse(line.slice(6))
        } catch {
          // skip malformed SSE lines
        }
      }
    }
  }
}
