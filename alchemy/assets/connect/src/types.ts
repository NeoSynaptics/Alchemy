/** AlchemyConnect protocol types — separate from AlchemyCode relay types. */

export interface AlchemyMessage {
  agent: string;
  type: string;
  payload?: Record<string, any>;
  v?: number;
  id?: string;
  ts?: number;
  ref?: string;
  seq?: number;
}

export type AlchemyConnectState =
  | "disconnected"
  | "connecting"
  | "authenticating"
  | "connected"
  | "error";

/**
 * Binary frame header — prepended to every binary WebSocket frame.
 * Wire format: [4-byte LE uint32: header JSON byte length][header JSON][binary data]
 */
export interface AlchemyBinaryHeader {
  agent: string;       // Target agent: "code", "voice", "browser", etc.
  type: string;        // e.g. "image", "audio_chunk", "file", "screenshot"
  id: string;          // Unique frame ID
  mime: string;        // "image/jpeg", "audio/pcm", "audio/opus", "application/octet-stream"
  size: number;        // Binary payload byte length
  ref?: string;        // Optional reference to a prior message
  meta?: Record<string, any>; // Extra context: filename, sample_rate, width, height, etc.
}

/** Received binary frame — decoded from the wire. */
export interface AlchemyBinaryFrame {
  header: AlchemyBinaryHeader;
  data: ArrayBuffer;
}

export interface AlchemyConnectConfig {
  serverUrl: string;
  token: string;
  onStateChange: (state: AlchemyConnectState) => void;
  onAgentList?: (agents: string[]) => void;
  onChatToken?: (token: string, ref: string) => void;
  onChatDone?: (fullText: string, model: string, ref: string) => void;
  onMessage?: (msg: AlchemyMessage) => void;
  onBinary?: (frame: AlchemyBinaryFrame) => void;  // image, audio, file from server
  onError?: (reason: string, agent?: string) => void;
}

export interface AlchemyHello {
  session_id: number;
  server_version: string;
  available_agents: string[];
}

export interface AlchemyAuthOk {
  device_name: string;
  paired_at: number;
}

export const ALCHEMY_CLOSE_CODES = {
  AUTH_TIMEOUT: 4000,
  INVALID_TOKEN: 4001,
  DEVICE_REVOKED: 4002,
  SERVER_SHUTDOWN: 1001,
} as const;
