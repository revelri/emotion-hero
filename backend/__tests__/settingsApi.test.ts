import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { EventEmitter } from 'events';
import { SettingsApi } from '../settingsApi.js';
import { EmotionDetector } from '../emotionDetector.js';
import { SignalProcessor } from '../signalProcessor.js';
import { WsServer } from '../wsServer.js';
import * as http from 'node:http';

const TEST_PORT = 18801;

// Minimal firehose mock
class MockFirehose extends EventEmitter {
  private _config = { endpoint: 'wss://test.example.com', retryInterval: 5000, maxRetries: 10 };
  getStatus() {
    return { isRunning: false, connectionState: { status: 'disconnected' as const, retryCount: 0 }, metrics: { processed: 0, errors: 0, reconnects: 0 } };
  }
  get config() { return this._config; }
  setConfig(c: Partial<{ endpoint: string; retryInterval: number; maxRetries: number }>) { Object.assign(this._config, c); }
}

function makeFetch(port: number, method: string, path: string, body?: unknown): Promise<{ status: number; data: string; headers: http.IncomingHttpHeaders }> {
  return new Promise((resolve, reject) => {
    const payload = body !== undefined ? JSON.stringify(body) : undefined;
    const options: http.RequestOptions = {
      hostname: 'localhost',
      port,
      path,
      method,
      headers: {
        'Content-Type': 'application/json',
        'Connection': 'close',
      },
      timeout: 5000,
    };
    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk: Buffer) => { data += chunk.toString(); });
      res.on('end', () => resolve({ status: res.statusCode ?? 0, data, headers: res.headers }));
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('Request timeout')); });
    if (payload) req.write(payload);
    req.end();
  });
}

describe('SettingsApi', () => {
  let api: SettingsApi;
  let detector: EmotionDetector;
  let processor: SignalProcessor;
  let wsServer: WsServer;
  let firehose: MockFirehose;
  const wsPort = 18802;

  beforeEach(async () => {
    detector = new EmotionDetector({ models: [], thresholds: {} });
    await detector.start();
    processor = new SignalProcessor({ minCutoff: 1.0, beta: 0.007 });
    wsServer = new WsServer({ port: wsPort });
    await wsServer.start();
    firehose = new MockFirehose();

    api = new SettingsApi({
      port: TEST_PORT,
      firehose: firehose as any,
      emotionDetector: detector,
      signalProcessor: processor,
      wsServer,
    });
    await api.start();
  });

  afterEach(async () => {
    await api.stop();
    await wsServer.stop();
    await detector.stop();
  });

  describe('GET /api/settings', () => {
    it('returns all settings as JSON', async () => {
      const res = await makeFetch(TEST_PORT, 'GET', '/api/settings');
      expect(res.status).toBe(200);
      const body = JSON.parse(res.data);
      expect(body).toHaveProperty('firehose');
      expect(body).toHaveProperty('keywords');
      expect(body).toHaveProperty('signal');
      expect(body).toHaveProperty('detector');
      expect(body).toHaveProperty('visualization');
    });

    it('returns current signal processor config', async () => {
      const res = await makeFetch(TEST_PORT, 'GET', '/api/settings');
      const body = JSON.parse(res.data);
      expect(body.signal.minCutoff).toBe(1.0);
      expect(body.signal.beta).toBe(0.007);
    });

    it('returns current keywords', async () => {
      const res = await makeFetch(TEST_PORT, 'GET', '/api/settings');
      const body = JSON.parse(res.data);
      expect(body.keywords.happy).toEqual(['happy']);
    });
  });

  describe('PUT /api/settings/signal', () => {
    it('updates signal processor config', async () => {
      const res = await makeFetch(TEST_PORT, 'PUT', '/api/settings/signal', { minCutoff: 2.0, beta: 0.01 });
      expect(res.status).toBe(200);
      expect(processor.getConfig().minCutoff).toBe(2.0);
      expect(processor.getConfig().beta).toBe(0.01);
    });

    it('rejects invalid minCutoff', async () => {
      const res = await makeFetch(TEST_PORT, 'PUT', '/api/settings/signal', { minCutoff: -1 });
      expect(res.status).toBe(400);
    });

    it('rejects invalid beta', async () => {
      const res = await makeFetch(TEST_PORT, 'PUT', '/api/settings/signal', { beta: -0.5 });
      expect(res.status).toBe(400);
    });
  });

  describe('PUT /api/settings/keywords', () => {
    it('updates emotion keywords', async () => {
      const res = await makeFetch(TEST_PORT, 'PUT', '/api/settings/keywords', { happy: ['zen', 'chill'] });
      expect(res.status).toBe(200);
      const keywords = detector.getKeywords();
      expect(keywords.happy).toEqual(['zen', 'chill']);
    });

    it('rejects non-object body', async () => {
      const res = await makeFetch(TEST_PORT, 'PUT', '/api/settings/keywords', 'not-an-object');
      expect(res.status).toBe(400);
    });

    it('rejects non-string array keywords', async () => {
      const res = await makeFetch(TEST_PORT, 'PUT', '/api/settings/keywords', { happy: 'not-array' });
      expect(res.status).toBe(400);
    });
  });

  describe('PUT /api/settings/detector', () => {
    it('updates window duration', async () => {
      const res = await makeFetch(TEST_PORT, 'PUT', '/api/settings/detector', { windowDuration: 2000 });
      expect(res.status).toBe(200);
    });

    it('rejects out-of-range window duration', async () => {
      const res = await makeFetch(TEST_PORT, 'PUT', '/api/settings/detector', { windowDuration: 100 });
      expect(res.status).toBe(400);
    });
  });

  describe('PUT /api/settings/visualization', () => {
    it('broadcasts settings to frontend via wsServer', async () => {
      const messages: any[] = [];
      const ws = new (await import('ws')).WebSocket(`ws://localhost:${wsPort}`);
      await new Promise<void>((resolve, reject) => {
        ws.on('open', resolve);
        ws.on('error', reject);
      });

      ws.on('message', (data: any) => {
        messages.push(JSON.parse(data.toString()));
      });

      // Clear initial messages
      await new Promise((r) => setTimeout(r, 200));
      messages.length = 0;

      const res = await makeFetch(TEST_PORT, 'PUT', '/api/settings/visualization', {
        shaderMode: 5,
        feedbackStrength: 0.8,
        reducedMotion: true,
      });
      expect(res.status).toBe(200);

      await new Promise((r) => setTimeout(r, 200));
      const settingsMsg = messages.find((m) => m.type === 'settings');
      expect(settingsMsg).toBeDefined();
      expect(settingsMsg.settings.shaderMode).toBe(5);
      expect(settingsMsg.settings.feedbackStrength).toBe(0.8);
      expect(settingsMsg.settings.reducedMotion).toBe(true);

      ws.close();
    });

    it('rejects invalid shader mode', async () => {
      const res = await makeFetch(TEST_PORT, 'PUT', '/api/settings/visualization', { shaderMode: 999 });
      expect(res.status).toBe(400);
    });
  });

  describe('PUT /api/settings/firehose', () => {
    it('updates firehose endpoint', async () => {
      const res = await makeFetch(TEST_PORT, 'PUT', '/api/settings/firehose', {
        endpoint: 'wss://new.example.com',
        retryInterval: 3000,
        maxRetries: 5,
      });
      expect(res.status).toBe(200);
      expect(firehose.config.endpoint).toBe('wss://new.example.com');
    });
  });

  describe('GET /api/status', () => {
    it('returns system status', async () => {
      const res = await makeFetch(TEST_PORT, 'GET', '/api/status');
      expect(res.status).toBe(200);
      const body = JSON.parse(res.data);
      expect(body).toHaveProperty('firehose');
      expect(body).toHaveProperty('emotionDetector');
      expect(body).toHaveProperty('signalProcessor');
      expect(body).toHaveProperty('wsServer');
      expect(body).toHaveProperty('clients');
      expect(body).toHaveProperty('uptime');
    });
  });

  describe('GET /', () => {
    it('serves the settings HTML page', async () => {
      const res = await makeFetch(TEST_PORT, 'GET', '/');
      expect(res.status).toBe(200);
      expect(res.data).toContain('<!DOCTYPE html>');
      expect(res.data).toContain('Zeitgeist Settings');
    });
  });

  describe('404 handling', () => {
    it('returns 404 for unknown routes', async () => {
      const res = await makeFetch(TEST_PORT, 'GET', '/api/unknown');
      expect(res.status).toBe(404);
    });
  });
});
