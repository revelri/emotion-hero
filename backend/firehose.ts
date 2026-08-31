/**
 * Firehose - Data Ingestion Layer
 *
 * This module handles the ingestion of raw data from the Bluesky firehose.
 * It uses Bluesky's Jetstream API which provides JSON-formatted messages
 * with full post content, making it straightforward to extract post text.
 *
 * The firehose is responsible for:
 * - Connecting to the Bluesky Jetstream firehose
 * - Extracting post text from create events
 * - Emitting raw post text as plain strings
 * - Managing connection health and error recovery
 * - Graceful reconnection with exponential backoff
 */

import { EventEmitter } from 'events';
import { WebSocket } from 'ws';

export interface FirehoseConfig {
  endpoint?: string;
  retryInterval: number;
  maxRetries: number;
}

interface ConnectionState {
  status: 'disconnected' | 'connecting' | 'connected' | 'error';
  retryCount: number;
  lastError?: Error;
}

/**
 * Jetstream message structure for post creation events
 * See: https://github.com/bluesky-social/jetstream
 */
interface JetstreamMessage {
  did: string;
  time_us: number;
  kind: 'commit' | 'identity' | 'account';
  commit?: {
    rev: string;
    operation: 'create' | 'update' | 'delete';
    collection: string;
    rkey: string;
    record?: {
      $type: string;
      text?: string;
      createdAt?: string;
      langs?: string[];
      [key: string]: unknown;
    };
    cid?: string;
  };
}

export class Firehose extends EventEmitter {
  private config: FirehoseConfig;
  private isRunning: boolean = false;
  private ws: WebSocket | null = null;
  private connectionState: ConnectionState = {
    status: 'disconnected',
    retryCount: 0,
  };
  private metrics: {
    processed: number;
    errors: number;
    reconnects: number;
  } = {
    processed: 0,
    errors: 0,
    reconnects: 0,
  };
  private metricsInterval: NodeJS.Timeout | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;

  // Jetstream provides JSON-formatted firehose data with full records
  private readonly DEFAULT_ENDPOINT = process.env.JETSTREAM_URL || 'wss://jetstream1.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post';
  private readonly METRICS_LOG_INTERVAL = 60000; // 1 minute

  constructor(config: FirehoseConfig) {
    super();
    this.config = config;
  }

  /** The endpoint actually used (override or default). */
  getEffectiveEndpoint(): string {
    return this.config.endpoint || this.DEFAULT_ENDPOINT;
  }

  /** The built-in default endpoint, used when no override is set. */
  getDefaultEndpoint(): string {
    return this.DEFAULT_ENDPOINT;
  }

  /**
   * Start the firehose and establish connection to Bluesky Jetstream
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error('Firehose is already running');
    }

    this.isRunning = true;
    this.connectionState = { status: 'disconnected', retryCount: 0 };
    this.metrics = { processed: 0, errors: 0, reconnects: 0 };

    // Start metrics logging
    this.metricsInterval = setInterval(() => this.logMetrics(), this.METRICS_LOG_INTERVAL);

    // Start connection
    await this.connect();
    this.emit('started');
  }

  /**
   * Stop the firehose and close connection
   */
  async stop(): Promise<void> {
    if (!this.isRunning) {
      return;
    }

    this.isRunning = false;

    // Clear reconnection timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Clear metrics interval
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
      this.metricsInterval = null;
    }

    // Close WebSocket connection
    this.disconnect();

    // Log final metrics
    this.logMetrics();

    this.emit('stopped');
  }

  /**
   * Establish WebSocket connection to Bluesky Jetstream
   */
  private async connect(): Promise<void> {
    if (!this.isRunning) {
      return;
    }

    this.connectionState.status = 'connecting';

    const endpoint = this.config.endpoint || this.DEFAULT_ENDPOINT;

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(endpoint, {
        headers: {
          'User-Agent': 'zeitgeist-firehose/1.0',
        },
      });

      // Set connection timeout
      const connectionTimeout = setTimeout(() => {
        if (this.ws?.readyState === WebSocket.CONNECTING) {
          this.ws.close();
          this.handleConnectionError(new Error('Connection timeout'));
          reject(new Error('Connection timeout'));
        }
      }, 30000);

      this.ws.on('open', () => {
        clearTimeout(connectionTimeout);
        this.connectionState.status = 'connected';
        this.connectionState.retryCount = 0;
        console.log('[Firehose] Connected to Bluesky Jetstream');
        resolve();
      });

      this.ws.on('message', (data: Buffer) => {
        this.onMessage(data);
      });

      this.ws.on('close', (code: number, reason: Buffer) => {
        clearTimeout(connectionTimeout);
        console.log(`[Firehose] Disconnected (code: ${code})`);
        this.handleDisconnection();
      });

      this.ws.on('error', (error: Error) => {
        clearTimeout(connectionTimeout);
        this.handleConnectionError(error);
        reject(error);
      });
    });
  }

  /**
   * Handle WebSocket disconnection and trigger reconnection
   */
  private handleDisconnection(): void {
    if (!this.isRunning) {
      return;
    }

    this.connectionState.status = 'disconnected';

    // Trigger reconnection
    this.handleReconnection();
  }

  /**
   * Handle connection errors and trigger reconnection
   */
  private handleConnectionError(error: Error): void {
    if (!this.isRunning) {
      return;
    }

    this.connectionState.status = 'error';
    this.connectionState.lastError = error;
    this.metrics.errors++;
    console.error('[Firehose] Connection error:', error.message);
    this.emit('error', error);

    // Trigger reconnection
    this.handleReconnection();
  }

  /**
   * Handle reconnection with exponential backoff
   */
  private handleReconnection(): void {
    if (!this.isRunning) {
      return;
    }

    // Check if we've exceeded max retries
    if (this.connectionState.retryCount >= this.config.maxRetries) {
      console.error('[Firehose] Max retries exceeded, stopping');
      this.emit('error', new Error('Max retries exceeded'));
      this.stop();
      return;
    }

    // Calculate exponential backoff delay
    const backoffDelay = Math.min(
      this.config.retryInterval * Math.pow(2, this.connectionState.retryCount),
      60000 // Cap at 1 minute
    );

    this.connectionState.retryCount++;
    this.metrics.reconnects++;

    console.log(`[Firehose] Reconnecting in ${backoffDelay}ms (attempt ${this.connectionState.retryCount})`);

    this.reconnectTimeout = setTimeout(async () => {
      try {
        await this.connect();
      } catch (error) {
        // Error already handled in connect()
      }
    }, backoffDelay);
  }

  /**
   * Close WebSocket connection
   */
  private disconnect(): void {
    if (this.ws) {
      if (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) {
        this.ws.close();
      }
      this.ws = null;
    }
  }

  /**
   * Handle incoming WebSocket messages
   * This is the processing layer - separate from connection management
   */
  private onMessage(data: Buffer): void {
    // Never block on processing - use setImmediate to yield to event loop
    setImmediate(() => {
      try {
        this.processMessage(data);
      } catch (error) {
        this.metrics.errors++;
      }
    });
  }

  /**
   * Process a single Jetstream message
   * Jetstream sends JSON messages with full record data included
   */
  private processMessage(data: Buffer): void {
    let message: JetstreamMessage;

    try {
      const text = data.toString('utf-8');
      message = JSON.parse(text);
    } catch {
      // Skip malformed messages
      return;
    }

    // Only process commit messages
    if (message.kind !== 'commit') {
      return;
    }

    // Only process post creation events
    if (!message.commit || message.commit.operation !== 'create') {
      return;
    }

    // Only process app.bsky.feed.post records
    if (message.commit.collection !== 'app.bsky.feed.post') {
      return;
    }

    // Extract the post text from the record
    const postText = this.extractPostText(message);

    if (postText) {
      this.metrics.processed++;
      this.emit('data', postText);
    }
  }

  /**
   * Extract post text from a Jetstream message
   * Returns plain string or null if not extractable
   */
  private extractPostText(message: JetstreamMessage): string | null {
    // The record is directly included in Jetstream messages
    const record = message.commit?.record;

    if (!record) {
      return null;
    }

    // Verify this is a post record type
    if (record.$type !== 'app.bsky.feed.post') {
      return null;
    }

    // Extract the text field
    const text = record.text;

    if (typeof text !== 'string' || text.length === 0) {
      return null;
    }

    return text;
  }

  /**
   * Log aggregate metrics
   */
  private logMetrics(): void {
    if (this.metrics.processed > 0 || this.metrics.errors > 0) {
      console.log(`[Firehose] Metrics - Processed: ${this.metrics.processed}, Errors: ${this.metrics.errors}, Reconnects: ${this.metrics.reconnects}`);
    }
  }

  /**
   * Get current connection state
   */
  getConnectionState(): ConnectionState {
    return { ...this.connectionState };
  }

  /**
   * Get current metrics
   */
  getMetrics(): typeof this.metrics {
    return { ...this.metrics };
  }

  /**
   * Get firehose status (called by index.ts)
   */
  getStatus(): {
    isRunning: boolean;
    connectionState: ConnectionState;
    metrics: { processed: number; errors: number; reconnects: number };
  } {
    return {
      isRunning: this.isRunning,
      connectionState: this.getConnectionState(),
      metrics: this.getMetrics(),
    };
  }
}
