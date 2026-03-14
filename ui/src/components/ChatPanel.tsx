import { useState, useRef, useEffect, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { streamChat } from '@/hooks/useAlchemy'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const conversationId = useRef(crypto.randomUUID())
  const scrollRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || streaming) return
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    setStreaming(true)

    // Add empty assistant message to stream into
    setMessages((prev) => [...prev, { role: 'assistant', content: '' }])

    try {
      for await (const chunk of streamChat(text, conversationId.current)) {
        if (chunk.content) {
          setMessages((prev) => {
            const updated = [...prev]
            const last = updated[updated.length - 1]
            if (last.role === 'assistant') {
              updated[updated.length - 1] = { ...last, content: last.content + chunk.content }
            }
            return updated
          })
        }
      }
    } catch (e) {
      setMessages((prev) => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last.role === 'assistant' && !last.content) {
          updated[updated.length - 1] = {
            ...last,
            content: `Error: ${e instanceof Error ? e.message : 'Connection failed'}`,
          }
        }
        return updated
      })
    } finally {
      setStreaming(false)
      textareaRef.current?.focus()
    }
  }, [input, streaming])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex h-full flex-col rounded-xl border bg-card">
      {/* Header */}
      <div className="border-b px-4 py-3">
        <h3 className="text-sm font-semibold">Chat with Alchemy</h3>
        <p className="text-[11px] text-muted-foreground">Ask about settings, models, or capabilities</p>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center">
            <p className="text-sm text-muted-foreground text-center">
              Ask anything about your Alchemy setup
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-xl px-3.5 py-2.5 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-foreground'
              }`}
            >
              <p className="whitespace-pre-wrap break-words">{msg.content}</p>
              {msg.role === 'assistant' && streaming && i === messages.length - 1 && (
                <span className="inline-block h-4 w-1 animate-pulse bg-foreground/50 ml-0.5" />
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Input */}
      <div className="border-t p-3">
        <div className="flex gap-2">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            className="min-h-[40px] max-h-[120px] resize-none text-sm"
            rows={1}
            disabled={streaming}
          />
          <Button
            size="sm"
            onClick={handleSend}
            disabled={!input.trim() || streaming}
            className="self-end"
          >
            Send
          </Button>
        </div>
      </div>
    </div>
  )
}
