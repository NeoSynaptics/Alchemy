import { useState, useCallback, useEffect } from 'react'

interface Props {
  onUploadComplete: () => void
}

export function PhotoDropZone({ onUploadComplete }: Props) {
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState('')

  // Listen for drag events on the window so we don't block clicks
  useEffect(() => {
    let dragCounter = 0

    const onDragEnter = (e: DragEvent) => {
      e.preventDefault()
      dragCounter++
      if (e.dataTransfer?.types.includes('Files')) {
        setIsDragging(true)
      }
    }

    const onDragLeave = (e: DragEvent) => {
      e.preventDefault()
      dragCounter--
      if (dragCounter <= 0) {
        dragCounter = 0
        setIsDragging(false)
      }
    }

    const onDragOver = (e: DragEvent) => {
      e.preventDefault()
    }

    window.addEventListener('dragenter', onDragEnter)
    window.addEventListener('dragleave', onDragLeave)
    window.addEventListener('dragover', onDragOver)

    return () => {
      window.removeEventListener('dragenter', onDragEnter)
      window.removeEventListener('dragleave', onDragLeave)
      window.removeEventListener('dragover', onDragOver)
    }
  }, [])

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files).filter(f =>
      f.type.startsWith('image/')
    )
    if (files.length === 0) return

    setUploading(true)
    let uploaded = 0

    for (const file of files) {
      setProgress(`Uploading ${uploaded + 1}/${files.length}: ${file.name}`)
      const form = new FormData()
      form.append('file', file)

      try {
        await fetch('/api/v1/memory/timeline/photo/upload', {
          method: 'POST',
          body: form,
        })
        uploaded++
      } catch {
        // continue with remaining files
      }
    }

    setProgress(`Done — ${uploaded} photo${uploaded !== 1 ? 's' : ''} added`)
    setTimeout(() => {
      setUploading(false)
      setProgress('')
      onUploadComplete()
    }, 1500)
  }, [onUploadComplete])

  // Only render the overlay when dragging or uploading
  if (!isDragging && !uploading) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <div className={`
        flex flex-col items-center gap-4 p-12
        rounded-2xl border-2 border-dashed
        transition-all duration-300
        ${isDragging
          ? 'border-amber-400/60 bg-amber-500/5 scale-105'
          : 'border-zinc-600/40 bg-zinc-900/80'
        }
      `}>
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-amber-400">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" strokeLinecap="round" strokeLinejoin="round" />
          <polyline points="17 8 12 3 7 8" strokeLinecap="round" strokeLinejoin="round" />
          <line x1="12" y1="3" x2="12" y2="15" strokeLinecap="round" />
        </svg>
        <p className="text-sm text-zinc-300">
          {uploading ? progress : 'Drop photos to add to timeline'}
        </p>
        {uploading && (
          <div className="w-48 h-1 bg-zinc-800 rounded-full overflow-hidden">
            <div className="h-full bg-amber-400 rounded-full animate-pulse" style={{ width: '60%' }} />
          </div>
        )}
      </div>
    </div>
  )
}
