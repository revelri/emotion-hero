/**
 * Backend Entry Point - System Orchestration
 * 
 * This module serves as the entry point for the emotional visualization backend.
 * It orchestrates all backend components and manages their lifecycle.
 * The index file handles:
 * - Initializing and configuring all backend components
 * - Managing component startup and shutdown sequence
 * - Setting up inter-component communication
 * - Handling system-wide errors and graceful shutdown
 */

import { Firehose } from './firehose.js';
import { EmotionDetector } from './emotionDetector.js';
import { SignalProcessor } from './signalProcessor.js';
import { WsServer } from './wsServer.js';
import { ContentLoader } from './contentLoader.js';
import { SettingsApi } from './settingsApi.js';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration interfaces - these will be expanded as the system develops
interface SystemConfig {
  firehose: {
    sources: Array<{
      id: string;
      type: string;
      endpoint: string;
      credentials?: any;
    }>;
    retryInterval: number;
    maxRetries: number;
  };
  emotionDetector: {
    models: Array<{
      id: string;
      type: string;
      parameters: any;
    }>;
    thresholds: {
      [emotion: string]: number;
    };
  };
  signalProcessor: {
    minCutoff: number;
    beta: number;
    aggregationMethod: 'average' | 'weighted' | 'peak';
    minSources: number;
    updateInterval: number;
  };
  wsServer: {
    port: number;
    maxConnections: number;
    heartbeatInterval: number;
    compressionEnabled: boolean;
  };
  contentLoader: {
    contentDir: string;
    colorsFile: string;
    watchInterval: number;
  };
}

class EmotionalVisualizationBackend {
  private config: SystemConfig;
  private firehose: Firehose | null = null;
  private emotionDetector: EmotionDetector | null = null;
  private signalProcessor: SignalProcessor | null = null;
  private wsServer: WsServer | null = null;
  private contentLoader: ContentLoader | null = null;
  private settingsApi: SettingsApi | null = null;
  private isRunning: boolean = false;

  constructor(config: SystemConfig) {
    this.config = config;
  }

  /**
   * Start the entire backend system
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error('Backend system is already running');
    }

    try {
      // Initialize components in dependency order
      await this.initializeComponents();
      
      // Set up inter-component communication
      this.setupComponentCommunication();
      
      // Start components in reverse dependency order
      await this.startComponents();
      
      this.isRunning = true;
    } catch (error) {
      await this.stop(); // Clean up any partially started components
      throw error;
    }
  }

  /**
   * Stop the entire backend system
   */
  async stop(): Promise<void> {
    if (!this.isRunning) {
      return;
    }
    
    try {
      // Stop components in dependency order
      await this.stopComponents();
      
      this.isRunning = false;
    } catch (error) {
      throw error;
    }
  }

  /**
   * Initialize all backend components
   */
  private async initializeComponents(): Promise<void> {
    this.firehose = new Firehose(this.config.firehose);
    this.emotionDetector = new EmotionDetector(this.config.emotionDetector);
    this.signalProcessor = new SignalProcessor(this.config.signalProcessor);
    this.wsServer = new WsServer(this.config.wsServer);
    this.contentLoader = new ContentLoader(this.config.contentLoader);
  }

  /**
   * Set up communication between components
   */
  private setupComponentCommunication(): void {
    if (!this.firehose || !this.emotionDetector || !this.signalProcessor || !this.wsServer || !this.contentLoader) {
      throw new Error('Components not initialized');
    }

    // Firehose -> EmotionDetector
    this.firehose.on('data', (dataPoint) => {
      this.emotionDetector?.processRawData(dataPoint);
    });

    // EmotionDetector -> SignalProcessor
    // Artistic Intent: Windowing aggregates raw emotion hits over time, creating a
    // temporal canvas where each emotion's presence is measured as a proportion.
    // Ratio calculation normalizes these proportions, allowing emotions to be
    // compared on equal footing - a single "joy" hit carries the same weight as
    // a single "sadness" hit within the same window. This creates a balanced
    // emotional landscape where the relative strength of each feeling emerges
    // naturally, enabling the signal processor to smooth these ratios into
    // fluid, continuous emotional transitions that feel organic and responsive.
    this.emotionDetector.on('ratios', (emotionRatios) => {
      for (const [emotionId, ratio] of Object.entries(emotionRatios.ratios)) {
        this.signalProcessor?.process(emotionId, ratio as number);
      }
    });

    // SignalProcessor -> WsServer
    this.signalProcessor.on('processed', (processedSignal) => {
      this.wsServer?.updateState(processedSignal);
    });

    // ContentLoader -> WsServer (broadcast content changes to clients)
    this.contentLoader.on('content', (contentState) => {
      this.wsServer?.updateContent({
        asciiArt: contentState.asciiArt,
        artFileName: contentState.artFileName,
        colors: contentState.colors,
      });
    });

    // Error handling
    this.firehose.on('error', (error) => {
      this.handleComponentError('firehose', error);
    });

    this.emotionDetector.on('error', (error) => {
      this.handleComponentError('emotionDetector', error);
    });

    this.signalProcessor.on('error', (error) => {
      this.handleComponentError('signalProcessor', error);
    });

    this.wsServer.on('error', (error) => {
      this.handleComponentError('wsServer', error);
    });
  }

  /**
   * Start all components
   */
  private async startComponents(): Promise<void> {
    if (!this.wsServer || !this.signalProcessor || !this.emotionDetector || !this.firehose || !this.contentLoader) {
      throw new Error('Components not initialized');
    }

    // Start the settings HTTP server first so we can share its http.Server
    // with the WS server — single port for the whole consolidated backend.
    const settingsPort = parseInt(process.env.SETTINGS_PORT || '8081', 10);
    this.settingsApi = new SettingsApi({
      port: settingsPort,
      firehose: this.firehose as any,
      emotionDetector: this.emotionDetector,
      signalProcessor: this.signalProcessor,
      wsServer: this.wsServer,
      contentLoader: this.contentLoader,
    });
    await this.settingsApi.start();

    // Attach WS server to the same http.Server (so port 8081 carries both).
    const sharedHttp = this.settingsApi.getHttpServer();
    if (sharedHttp) {
      (this.wsServer as any).config.httpServer = sharedHttp;
    }
    await this.wsServer.start();
    await this.contentLoader.start();
    await this.signalProcessor.start();
    await this.emotionDetector.start();
    await this.firehose.start();

    // Send initial content state after all components are started
    const contentState = this.contentLoader.getState();
    this.wsServer.updateContent({
      asciiArt: contentState.asciiArt,
      artFileName: contentState.artFileName,
      colors: contentState.colors,
    });

    console.log(`[Settings] HTTP+WS available at http://localhost:${this.settingsApi.getPort()}`);
  }

  /**
    * Stop all components
    */
  private async stopComponents(): Promise<void> {
    if (this.settingsApi) {
      await this.settingsApi.stop();
    }

    if (this.firehose) {
      await this.firehose.stop();
    }

    if (this.emotionDetector) {
      await this.emotionDetector.stop();
    }

    if (this.signalProcessor) {
      await this.signalProcessor.stop();
    }

    if (this.contentLoader) {
      await this.contentLoader.stop();
    }

    if (this.wsServer) {
      await this.wsServer.stop();
    }
  }

  getFirehose(): Firehose | null { return this.firehose; }
  getEmotionDetector(): EmotionDetector | null { return this.emotionDetector; }
  getSignalProcessor(): SignalProcessor | null { return this.signalProcessor; }
  getWsServer(): WsServer | null { return this.wsServer; }

  /**
   * Handle component errors
   */
  private handleComponentError(component: string, error: any): void {
    // For now, we'll implement a simple strategy: critical components cause shutdown
    const criticalComponents = ['firehose', 'emotionDetector', 'signalProcessor', 'wsServer'];
    
    if (criticalComponents.includes(component)) {
      this.stop().catch(err => {
        console.error(`[Backend] Shutdown failed after ${component} error:`, err);
      });
    }
  }

  /**
   * Get system status
   */
  getStatus(): any {
    return {
      isRunning: this.isRunning,
      components: {
        firehose: this.firehose?.getStatus(),
        emotionDetector: this.emotionDetector?.getStatus(),
        signalProcessor: this.signalProcessor?.getStatus(),
        wsServer: this.wsServer?.getStatus(),
        contentLoader: this.contentLoader?.getStatus(),
      },
    };
  }
}

// Default configuration - in a real system, this would be loaded from config files
const defaultConfig: SystemConfig = {
  firehose: {
    sources: [],
    retryInterval: 5000,
    maxRetries: Infinity,  // Never give up — this is a long-running service
  },
  emotionDetector: {
    models: [],
    thresholds: {},
  },
  signalProcessor: {
    minCutoff: 1.0, // Heavy smoothing when calm
    beta: 0.007,    // Adaptive: less lag during fast changes
    aggregationMethod: 'average',
    minSources: 1,
    updateInterval: 100, // 10 times per second
  },
  wsServer: {
    port: parseInt(process.env.WS_PORT || '8080', 10),
    maxConnections: 100,
    heartbeatInterval: 30000, // 30 seconds
    compressionEnabled: true,
  },
  contentLoader: {
    // Content folder is at backend/content (relative to project root)
    contentDir: path.join(__dirname, '..', 'content'),
    colorsFile: 'colors.txt',
    watchInterval: 1000,  // Check for changes every second
  },
};

// Create and export the backend instance
const backend = new EmotionalVisualizationBackend(defaultConfig);

// Handle process signals for graceful shutdown
process.on('SIGINT', async () => {
  await backend.stop();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  await backend.stop();
  process.exit(0);
});

// Start the backend
backend.start().catch(error => {
  process.exit(1);
});

export { backend, EmotionalVisualizationBackend, SystemConfig };