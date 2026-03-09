/**
 * AlchemyConnect WebSocket client — speaks the AlchemyMessage protocol.
 * Connects to Alchemy server at /ws/connect with QR-based device auth.
 */

import type { AlchemyMessage, AlchemyBinaryHeader, AlchemyBinaryFrame, AlchemyConnectConfig, AlchemyConnectState, AlchemyHello } from "./types";
import { ALCHEMY_CLOSE_CODES } from "./types";

// ── State ────────────────────────────────────────────────────

let ws: WebSocket | null = null;
let config: AlchemyConnectConfig | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectAttempt = 0;
let deadTimer: ReturnType<typeof setTimeout> | null = null;
let seqCounter = 0;
let authFailed = false;

// ── Helpers ──────────────────────────────────────────────────

function makeId(): string {
  const bytes = new Uint8Array(6);
  if (typeof crypto !== "undefined" && crypto.getRandomValues) {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < 6; i++) bytes[i] = Math.floor(Math.random() * 256);
  }
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function makeMsg(
  agent: string,
  type: string,
  payload?: Record<string, any>,
  ref?: string,
): AlchemyMessage {
  const msg: AlchemyMessage = {
    agent,
    type,
    v: 1,
    id: makeId(),
    ts: Date.now(),
  };
  if (payload && Object.keys(payload).length > 0) msg.payload = payload;
  if (ref) msg.ref = ref;
  return msg;
}

function send(msg: AlchemyMessage): void {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

function setState(state: AlchemyConnectState): void {
  config?.onStateChange(state);
}

// ── Keepalive ────────────────────────────────────────────────

function resetDeadTimer(): void {
  if (deadTimer) clearTimeout(deadTimer);
  deadTimer = setTimeout(() => {
    // No ping received for 45s — connection is dead
    if (ws) {
      ws.close();
      ws = null;
    }
    setState("disconnected");
    scheduleReconnect();
  }, 45_000);
}

function clearDeadTimer(): void {
  if (deadTimer) {
    clearTimeout(deadTimer);
    deadTimer = null;
  }
}

// ── Message Handling ─────────────────────────────────────────

function handleSystemMsg(msg: AlchemyMessage): void {
  switch (msg.type) {
    case "hello": {
      const hello = msg.payload as AlchemyHello | undefined;
      if (hello?.available_agents) {
        config?.onAgentList?.(hello.available_agents);
      }
      // Immediately send auth
      setState("authenticating");
      send(makeMsg("system", "auth", { token: config!.token }));
      break;
    }
    case "auth_ok":
      setState("connected");
      reconnectAttempt = 0;
      authFailed = false;
      resetDeadTimer();
      break;
    case "auth_fail":
      authFailed = true;
      config?.onError?.(
        (msg.payload as any)?.reason || "Authentication failed",
        "system",
      );
      setState("error");
      // Don't reconnect on auth failure
      break;
    case "ping":
      send(makeMsg("system", "pong", { ts: Date.now() }));
      resetDeadTimer();
      break;
    case "pong":
      resetDeadTimer();
      break;
    case "error":
      config?.onError?.((msg.payload as any)?.reason || "Unknown error", "system");
      break;
  }
}

function handleAgentMsg(msg: AlchemyMessage): void {
  // Chat streaming
  if (msg.agent === "chat") {
    if (msg.type === "token" && msg.ref) {
      config?.onChatToken?.((msg.payload as any)?.text || "", msg.ref);
      return;
    }
    if (msg.type === "done" && msg.ref) {
      config?.onChatDone?.(
        (msg.payload as any)?.text || "",
        (msg.payload as any)?.model || "",
        msg.ref,
      );
      return;
    }
  }

  // Agent errors
  if (msg.type === "error") {
    config?.onError?.((msg.payload as any)?.reason || "Agent error", msg.agent);
    return;
  }
}

function handleMessage(event: MessageEvent): void {
  // ── Binary frame ─────────────────────────────────────
  if (event.data instanceof ArrayBuffer) {
    decodeBinaryFrame(event.data);
    return;
  }

  // ── Text frame (JSON) ─────────────────────────────────
  let msg: AlchemyMessage;
  try {
    msg = JSON.parse(event.data);
  } catch {
    return;
  }
  if (!msg.agent || !msg.type) return;

  if (msg.agent === "system") {
    handleSystemMsg(msg);
  } else {
    handleAgentMsg(msg);
  }

  config?.onMessage?.(msg);
}

// ── Binary encoding / decoding ────────────────────────────────
// Wire format: [4-byte LE uint32: header JSON length][header JSON bytes][binary data]

function decodeBinaryFrame(buffer: ArrayBuffer): void {
  if (!config?.onBinary) return;
  try {
    const view = new DataView(buffer);
    const headerLen = view.getUint32(0, true); // little-endian
    const headerBytes = new Uint8Array(buffer, 4, headerLen);
    const header: AlchemyBinaryHeader = JSON.parse(new TextDecoder().decode(headerBytes));
    const data = buffer.slice(4 + headerLen);
    config.onBinary({ header, data });
  } catch {
    // Malformed binary frame — ignore
  }
}

function encodeBinaryFrame(header: AlchemyBinaryHeader, data: ArrayBuffer): ArrayBuffer {
  const headerJson = new TextEncoder().encode(JSON.stringify(header));
  const frame = new ArrayBuffer(4 + headerJson.byteLength + data.byteLength);
  const view = new DataView(frame);
  view.setUint32(0, headerJson.byteLength, true); // little-endian
  new Uint8Array(frame, 4, headerJson.byteLength).set(headerJson);
  new Uint8Array(frame, 4 + headerJson.byteLength).set(new Uint8Array(data));
  return frame;
}

// ── Reconnect ────────────────────────────────────────────────

function scheduleReconnect(): void {
  if (reconnectTimer || authFailed) return;
  const delay = Math.min(1000 * Math.pow(2, reconnectAttempt), 30_000);
  reconnectAttempt++;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    doConnect();
  }, delay);
}

// ── Connect ──────────────────────────────────────────────────

function doConnect(): void {
  if (!config) return;
  setState("connecting");

  try {
    const wsUrl = config.serverUrl.replace(/^http/, "ws").replace(/\/+$/, "") + "/ws/connect";
    // React Native WebSocket accepts headers as 3rd arg — needed for ngrok free tier
    ws = new (WebSocket as any)(wsUrl, undefined, {
      headers: { "ngrok-skip-browser-warning": "1", "User-Agent": "AlchemyCode/1.0" },
    });

    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      // Wait for system:hello before sending auth
    };

    ws.onmessage = handleMessage;

    ws.onclose = (event) => {
      clearDeadTimer();
      ws = null;

      // Don't reconnect on auth failures
      if (
        event.code === ALCHEMY_CLOSE_CODES.AUTH_TIMEOUT ||
        event.code === ALCHEMY_CLOSE_CODES.INVALID_TOKEN ||
        event.code === ALCHEMY_CLOSE_CODES.DEVICE_REVOKED
      ) {
        authFailed = true;
        setState("error");
        return;
      }

      setState("disconnected");
      scheduleReconnect();
    };

    ws.onerror = () => {
      setState("error");
    };
  } catch {
    setState("error");
    scheduleReconnect();
  }
}

// ── Public API ───────────────────────────────────────────────

export function connectAlchemy(cfg: AlchemyConnectConfig): void {
  disconnectAlchemy();
  config = cfg;
  authFailed = false;
  reconnectAttempt = 0;
  seqCounter = 0;
  doConnect();
}

export function disconnectAlchemy(): void {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  clearDeadTimer();
  authFailed = false;
  if (ws) {
    ws.close();
    ws = null;
  }
}

export function isAlchemyConnected(): boolean {
  return ws !== null && ws.readyState === WebSocket.OPEN && !authFailed;
}

/** Send a raw AlchemyMessage to any agent. Returns the message ID. */
export function sendAlchemyMessage(
  agent: string,
  type: string,
  payload?: Record<string, any>,
): string {
  const msg = makeMsg(agent, type, payload);
  msg.seq = ++seqCounter;
  send(msg);
  return msg.id!;
}

/** Send a chat message. Returns the message ID for ref-tracking. */
export function sendChat(text: string): string {
  return sendAlchemyMessage("chat", "message", { text });
}

/**
 * Send binary data (image, audio, file) to any agent.
 * Wire format: [4-byte LE header length][header JSON][binary data]
 *
 * @param agent  - Target agent ("code", "voice", "browser", etc.)
 * @param type   - Frame type ("image", "audio_chunk", "file", "screenshot")
 * @param mime   - MIME type ("image/jpeg", "audio/pcm", "audio/opus", etc.)
 * @param data   - Raw binary payload (ArrayBuffer)
 * @param meta   - Optional metadata (filename, width, height, sample_rate, etc.)
 * @returns Frame ID
 */
export function sendBinary(
  agent: string,
  type: string,
  mime: string,
  data: ArrayBuffer,
  meta?: Record<string, any>,
): string {
  const id = makeId();
  const header: AlchemyBinaryHeader = {
    agent, type, id, mime, size: data.byteLength, meta,
  };
  const frame = encodeBinaryFrame(header, data);
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(frame);
  }
  return id;
}

/** Convenience: send an image to an agent. */
export function sendImage(
  agent: string,
  data: ArrayBuffer,
  mime: "image/jpeg" | "image/png" | "image/webp" = "image/jpeg",
  meta?: Record<string, any>,
): string {
  return sendBinary(agent, "image", mime, data, meta);
}

/** Convenience: send a raw PCM audio chunk to an agent (e.g. voice). */
export function sendAudioChunk(
  agent: string,
  data: ArrayBuffer,
  sampleRate = 16000,
  channels = 1,
): string {
  return sendBinary(agent, "audio_chunk", "audio/pcm", data, { sample_rate: sampleRate, channels });
}

/** Convenience: send a file to an agent. */
export function sendFile(
  agent: string,
  data: ArrayBuffer,
  filename: string,
  mime = "application/octet-stream",
): string {
  return sendBinary(agent, "file", mime, data, { filename });
}

/** Query voice pipeline status. */
export function voiceStatus(): string {
  return sendAlchemyMessage("voice", "status");
}

/** Start voice pipeline. */
export function voiceStart(): string {
  return sendAlchemyMessage("voice", "start");
}

/** Stop voice pipeline. */
export function voiceStop(): string {
  return sendAlchemyMessage("voice", "stop");
}

/** Switch voice mode (conversation/command/dictation/muted). */
export function voiceMode(mode: string): string {
  return sendAlchemyMessage("voice", "mode", { mode });
}

/** Speak text through PC speakers via TTS. */
export function voiceSay(text: string): string {
  return sendAlchemyMessage("voice", "say", { text });
}
