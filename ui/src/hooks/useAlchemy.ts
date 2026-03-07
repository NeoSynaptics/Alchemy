import { useState, useEffect, useCallback } from 'react'

const API_BASE = '/api/v1'

// --- Types (read-only mirrors of backend responses) ---

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

// --- Fetch helper ---

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

// --- Read-only hooks ---

export function useModels() {
  const [data, setData] = useState<AlchemyModels | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setData(await apiFetch<AlchemyModels>('/models'))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch models')
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])
  return { data, loading, error, refresh }
}

export function useTaskStatus(taskId: string | null) {
  const [data, setData] = useState<TaskStatus | null>(null)
  const [loading, setLoading] = useState(false)

  const refresh = useCallback(async () => {
    if (!taskId) return
    setLoading(true)
    try {
      setData(await apiFetch<TaskStatus>(`/playwright/task/${taskId}/status`))
    } catch {
      setData(null)
    } finally {
      setLoading(false)
    }
  }, [taskId])

  useEffect(() => { refresh() }, [refresh])
  return { data, loading, refresh }
}
