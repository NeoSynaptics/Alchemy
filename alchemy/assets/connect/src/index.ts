/**
 * @alchemy/connect — AlchemyConnect WebSocket client SDK
 *
 * Supports text messages, images, audio chunks, and files.
 * Works in React Native, Node.js, and browser environments.
 *
 * Usage:
 *   import { connectAlchemy, sendAlchemyMessage, sendImage, sendAudioChunk } from "@alchemy/connect"
 */

// Connection lifecycle
export { connectAlchemy, disconnectAlchemy, isAlchemyConnected } from "./client";

// Text messaging
export { sendAlchemyMessage, sendChat } from "./client";

// Voice helpers
export { voiceStatus, voiceStart, voiceStop, voiceMode, voiceSay } from "./client";

// Binary — images, audio, files
export { sendBinary, sendImage, sendAudioChunk, sendFile } from "./client";

// Types
export type {
  AlchemyMessage,
  AlchemyBinaryHeader,
  AlchemyBinaryFrame,
  AlchemyConnectConfig,
  AlchemyConnectState,
  AlchemyHello,
  AlchemyAuthOk,
} from "./types";

export { ALCHEMY_CLOSE_CODES } from "./types";
