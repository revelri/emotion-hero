/**
 * WebSocket Broadcast Layer - Real-time Emotion State Streaming
 *
 * This module provides a WebSocket server that broadcasts smoothed emotion states
 * to connected clients in real time.
 *
 * Key behaviors:
 * - On client connect: Immediately send the current emotion state
 * - On state update: Broadcast the latest emotion state to all connected clients
 * - Rate limiting: ≤10 updates/second
 * - JSON-only payloads
 * - Stateless connections (no per-client buffers)
 */

import * as WebSocket from 'ws';
import * as http from 'node:http';
import { EventEmitter } from 'events';
import { EmotionState } from '@zeitgeist/shared/emotions.js';

/**
 * WebSocket message payload for emotion state updates
 */
interface EmotionStateMessage {
  type: 'emotions';
  /** Message version for future extensibility */
  version: '1.0';
  /** Timestamp when the state was generated */
  timestamp: number;
  /** Map of emotion IDs to their current states */
  emotions: Record<string, EmotionState>;
}

/**
 * WebSocket message payload for content updates (ASCII art and colors)
 */
interface ContentMessage {
  type: 'content';
  version: '1.0';
  timestamp: number;
  asciiArt: string | null;
  artFileName: string | null;
  colors: Record<string, { hex: string; rgb: [number, number, number] }>;
}

/**
 * Configuration options for the WebSocket server
 */
export interface WsServerConfig {
  /** Port number for the WebSocket server (ignored if httpServer is provided) */
  port: number;
  /** Host to bind to (default: '0.0.0.0', ignored if httpServer is provided) */
  host?: string;
  /** Maximum updates per second (default: 10) */
  maxUpdatesPerSecond?: number;
  /** Optional shared HTTP server to attach to (e.g. the settings API server). When provided, the WS server reuses that port instead of binding its own. */
  httpServer?: http.Server;
}

/**
 * WebSocket server that broadcasts emotion states to connected clients
 *
 * Artistic Intent: The WebSocket server acts as the bridge between the emotional
 * core and the visual world, broadcasting the living, breathing emotional state
 * to all who listen. As an EventEmitter, it can signal when things go wrong,
 * allowing the system to respond gracefully to the inevitable hiccups of
 * networked communication.
 */
export class WsServer extends EventEmitter {
  private server: WebSocket.WebSocketServer | null = null;
  private clients: Set<WebSocket.WebSocket> = new Set();
  private config: WsServerConfig;
  private currentState: Record<string, EmotionState> = {};
  private lastBroadcastTime: number = 0;
  private minBroadcastInterval: number;

  // Content state
  private currentContent: {
    asciiArt: string | null;
    artFileName: string | null;
    colors: Record<string, { hex: string; rgb: [number, number, number] }>;
  } = {
    asciiArt: null,
    artFileName: null,
    colors: {},
  };

  constructor(config: WsServerConfig) {
    super();
    this.config = config;
    this.minBroadcastInterval = 1000 / (config.maxUpdatesPerSecond ?? 10);
  }

  /**
   * Start the WebSocket server
   */
  async start(): Promise<void> {
    if (this.server) {
      throw new Error('WebSocket server is already running');
    }

    if (this.config.httpServer) {
      this.server = new WebSocket.WebSocketServer({ server: this.config.httpServer });
    } else {
      this.server = new WebSocket.WebSocketServer({
        port: this.config.port,
        host: this.config.host ?? '0.0.0.0',
      });
    }

    this.server!.on('connection', (socket: WebSocket.WebSocket) => {
      this.handleConnection(socket);
    });

    this.server!.on('listening', () => {
      // Artistic Intent: The server is now listening, ready to share emotional states
      // with all who connect. The broadcast begins.
    });

    this.server!.on('error', (error: Error) => {
      // Artistic Intent: Forward server errors to our EventEmitter
      // This allows the backend to respond to network issues gracefully
      this.emit('error', error);
    });
  }

  /**
   * Stop the WebSocket server
   */
  async stop(): Promise<void> {
    if (!this.server) {
      return;
    }

    // Close all client connections
    this.clients.forEach((socket) => {
      if (socket.readyState === WebSocket.WebSocket.OPEN) {
        socket.close();
      }
    });
    this.clients.clear();

    // Close the server
    return new Promise<void>((resolve) => {
      this.server!.close(() => {
        this.server = null;
        resolve();
      });
    });
  }

  /**
   * Update the current emotion state and broadcast to clients
   * Rate limited to ≤10 updates/second
   */
  updateState(emotions: Record<string, EmotionState>): void {
    const now = Date.now();
    const timeSinceLastBroadcast = now - this.lastBroadcastTime;

    // Rate limiting: drop frames if we're broadcasting too fast
    if (timeSinceLastBroadcast < this.minBroadcastInterval) {
      return;
    }

    // Update current state
    this.currentState = emotions;
    this.lastBroadcastTime = now;

    // Broadcast to all connected clients
    this.broadcastEmotions();
  }

  /**
   * Update the current content (ASCII art and colors) and broadcast to clients
   */
  updateContent(content: {
    asciiArt: string | null;
    artFileName: string | null;
    colors: Record<string, { hex: string; rgb: [number, number, number] }>;
  }): void {
    this.currentContent = content;
    this.broadcastContent();
  }

  /**
   * Handle new client connection
   * Immediately sends the current content and emotion state
   */
  private handleConnection(socket: WebSocket.WebSocket): void {
    this.clients.add(socket);

    // Send current content immediately on connect
    this.sendContent(socket);

    // Send current emotion state immediately on connect
    this.sendEmotions(socket);

    // Handle client disconnect
    socket.on('close', () => {
      this.handleDisconnection(socket);
    });

    // Handle client errors
    socket.on('error', (error: Error) => {
      this.handleDisconnection(socket);
    });
  }

  /**
   * Handle client disconnection
   */
  private handleDisconnection(socket: WebSocket.WebSocket): void {
    this.clients.delete(socket);
  }

  /**
   * Send the current emotion state to a specific client
   */
  private sendEmotions(socket: WebSocket.WebSocket): void {
    if (socket.readyState !== WebSocket.WebSocket.OPEN) {
      return;
    }

    const message: EmotionStateMessage = {
      type: 'emotions',
      version: '1.0',
      timestamp: Date.now(),
      emotions: this.currentState,
    };

    try {
      socket.send(JSON.stringify(message));
    } catch (error) {
      this.emit('error', error);
      this.handleDisconnection(socket);
    }
  }

  /**
   * Send the current content to a specific client
   */
  private sendContent(socket: WebSocket.WebSocket): void {
    if (socket.readyState !== WebSocket.WebSocket.OPEN) {
      return;
    }

    const message: ContentMessage = {
      type: 'content',
      version: '1.0',
      timestamp: Date.now(),
      asciiArt: this.currentContent.asciiArt,
      artFileName: this.currentContent.artFileName,
      colors: this.currentContent.colors,
    };

    try {
      socket.send(JSON.stringify(message));
    } catch (error) {
      this.emit('error', error);
      this.handleDisconnection(socket);
    }
  }

  /**
   * Broadcast a JSON payload to all connected clients.
   * Stateless: no per-client buffers, drop frames on error.
   */
  private broadcast(message: object): void {
    const payload = JSON.stringify(message);
    const failed: WebSocket.WebSocket[] = [];

    for (const socket of this.clients) {
      if (socket.readyState === WebSocket.WebSocket.OPEN) {
        try {
          socket.send(payload);
        } catch (error) {
          this.emit('error', error);
          failed.push(socket);
        }
      }
    }

    for (const socket of failed) {
      this.handleDisconnection(socket);
    }
  }

  /**
   * Broadcast the current emotion state to all connected clients
   */
  private broadcastEmotions(): void {
    const message: EmotionStateMessage = {
      type: 'emotions',
      version: '1.0',
      timestamp: Date.now(),
      emotions: this.currentState,
    };
    this.broadcast(message);
  }

  /**
   * Broadcast the current content to all connected clients
   */
  private broadcastContent(): void {
    const message: ContentMessage = {
      type: 'content',
      version: '1.0',
      timestamp: Date.now(),
      asciiArt: this.currentContent.asciiArt,
      artFileName: this.currentContent.artFileName,
      colors: this.currentContent.colors,
    };
    this.broadcast(message);
  }

  broadcastSettings(settings: { shaderMode?: number; feedbackStrength?: number; reducedMotion?: boolean }): void {
    const message = {
      type: 'settings',
      version: '1.0',
      timestamp: Date.now(),
      settings,
    };
    this.broadcast(message);
  }

  /**
   * Get the number of connected clients
   */
  getClientCount(): number {
    return this.clients.size;
  }

  /**
   * Check if the server is running
   */
  isRunning(): boolean {
    return this.server !== null;
  }

  /**
   * Get the current status of the WebSocket server
   *
   * Artistic Intent: Status reveals the health of the emotional broadcast channel,
   * showing how many souls are connected to receive the emotional stream and whether
   * the server is actively transmitting the living emotional state.
   *
   * @returns Status object with server state information
   */
  getStatus(): { isRunning: boolean; clientCount: number; port: number } {
    return {
      isRunning: this.isRunning(),
      clientCount: this.getClientCount(),
      port: this.config.port,
    };
  }
}
