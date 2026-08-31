import * as http from 'node:http';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { EventEmitter } from 'events';
import { EMOTIONS, EMOTION_IDS } from '@zeitgeist/shared/emotions.js';
import { EmotionDetector } from './emotionDetector.js';
import { SignalProcessor } from './signalProcessor.js';
import { WsServer } from './wsServer.js';
import { ContentLoader } from './contentLoader.js';

const SHADER_MODE_NAMES = [
  'Voronoi', 'Curl Noise', 'Domain-Warped FBM', 'Metaballs',
  'Flow Field', 'Reaction-Diffusion', 'Chladni', 'Cymatics',
  'Julia Set', 'Gravity Lens', 'Attractor', 'Cracks',
  'Smoke', 'Topography', 'Magnetic LIC', 'Lissajous',
  'Phyllotaxis', 'Ink Flow', 'Readme Hero',
];

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface SettingsApiConfig {
  port?: number;
  firehose: { getStatus(): unknown; getEffectiveEndpoint?(): string; getDefaultEndpoint?(): string; config: { endpoint?: string; retryInterval: number; maxRetries: number }; setConfig?(c: Partial<{ endpoint: string; retryInterval: number; maxRetries: number }>): void };
  emotionDetector: EmotionDetector;
  signalProcessor: SignalProcessor;
  wsServer: WsServer;
  contentLoader?: ContentLoader;
}

function json(res: http.ServerResponse, data: unknown, status = 200): void {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

function html(res: http.ServerResponse, content: string): void {
  res.writeHead(200, {
    'Content-Type': 'text/html; charset=utf-8',
    'Content-Length': Buffer.byteLength(content),
  });
  res.end(content);
}

function findAvailablePort(startPort: number, originalPort?: number): Promise<number> {
  const base = originalPort ?? startPort;
  return new Promise((resolve) => {
    const server = http.createServer();
    server.listen(startPort, '0.0.0.0', () => {
      server.close(() => resolve(startPort));
    });
    server.on('error', () => {
      if (startPort - base < 10) {
        resolve(findAvailablePort(startPort + 1, base));
      } else {
        resolve(startPort);
      }
    });
  });
}

export class SettingsApi extends EventEmitter {
  private server: http.Server | null = null;
  private port: number;
  private config: SettingsApiConfig;
  private startTime: number = Date.now();
  private settingsHtml: string | null = null;
  private vizDir: string | null = null;

  constructor(config: SettingsApiConfig) {
    super();
    this.config = config;
    this.port = config.port ?? 8081;
  }

  async start(): Promise<void> {
    // Try dist layout first (../static), fall back to source layout (./static)
    let htmlPath = path.join(__dirname, '..', 'static', 'settings.html');
    if (!fs.existsSync(htmlPath)) {
      htmlPath = path.join(__dirname, 'static', 'settings.html');
    }
    try {
      this.settingsHtml = fs.readFileSync(htmlPath, 'utf-8');
    } catch {
      this.settingsHtml = '<h1>Settings GUI not found</h1><p>Place settings.html in backend/static/</p>';
    }

    // Locate the sibling frontend/ directory for the embedded viz preview.
    // Try dist layout (backend/dist -> ../../frontend) then source (backend -> ../frontend).
    const candidates = [
      path.join(__dirname, '..', '..', 'frontend'),
      path.join(__dirname, '..', 'frontend'),
    ];
    this.vizDir = candidates.find((p) => fs.existsSync(path.join(p, 'index.html'))) ?? null;

    this.port = await findAvailablePort(this.port);
    this.startTime = Date.now();

    this.server = http.createServer((req, res) => this.handleRequest(req, res));
    this.server.listen(this.port, '0.0.0.0');
    this.emit('started', this.port);
  }

  async stop(): Promise<void> {
    if (!this.server) return;
    return new Promise((resolve) => {
      this.server!.close(() => {
        this.server = null;
        resolve();
      });
    });
  }

  private handleRequest(req: http.IncomingMessage, res: http.ServerResponse): void {
    const url = new URL(req.url || '/', `http://localhost:${this.port}`);

    if (req.method === 'GET' && url.pathname === '/') {
      html(res, this.settingsHtml!);
      return;
    }

    if (req.method === 'GET' && url.pathname.startsWith('/viz')) {
      this.serveVizFile(url.pathname, res);
      return;
    }

    if (req.method === 'GET' && url.pathname === '/api/settings') {
      json(res, this.getAllSettings());
      return;
    }

    if (req.method === 'GET' && url.pathname === '/api/status') {
      json(res, this.getStatus());
      return;
    }

    if (req.method === 'GET' && url.pathname === '/api/presets') {
      json(res, { presets: this.listPresets() });
      return;
    }

    if (req.method === 'PUT') {
      let raw = '';
      req.on('data', (chunk: Buffer) => { raw += chunk.toString(); });
      req.on('end', () => {
        let body: unknown;
        try { body = JSON.parse(raw); } catch { body = null; }
        this.handlePut(url.pathname, body, res);
      });
      return;
    }

    json(res, { error: 'Not found' }, 404);
  }

  private getPresetsDir(): string {
    // Mirrors contentLoader's dist/source resolution
    const candidates = [
      path.join(__dirname, '..', '..', 'backend', 'content', 'presets'),
      path.join(__dirname, '..', 'content', 'presets'),
      path.join(__dirname, 'content', 'presets'),
    ];
    return candidates.find((p) => fs.existsSync(p)) ?? candidates[1];
  }

  private listPresets(): Array<{ id: string; name: string; description?: string; sections: string[] }> {
    const dir = this.getPresetsDir();
    if (!fs.existsSync(dir)) return [];
    const out: Array<{ id: string; name: string; description?: string; sections: string[] }> = [];
    for (const file of fs.readdirSync(dir).sort()) {
      if (!file.endsWith('.json')) continue;
      try {
        const raw = fs.readFileSync(path.join(dir, file), 'utf-8');
        const parsed = JSON.parse(raw) as Record<string, unknown>;
        const id = file.replace(/\.json$/, '');
        out.push({
          id,
          name: typeof parsed.name === 'string' ? parsed.name : id,
          description: typeof parsed.description === 'string' ? parsed.description : undefined,
          sections: ['visualization', 'signal', 'detector', 'colors', 'keywords', 'firehose'].filter((s) => s in parsed),
        });
      } catch {
        // skip unparseable preset
      }
    }
    return out;
  }

  private loadPreset(id: string): Record<string, unknown> | null {
    const file = path.join(this.getPresetsDir(), `${id}.json`);
    if (!fs.existsSync(file)) return null;
    if (!path.normalize(file).startsWith(this.getPresetsDir())) return null;
    try { return JSON.parse(fs.readFileSync(file, 'utf-8')) as Record<string, unknown>; } catch { return null; }
  }

  private serveVizFile(pathname: string, res: http.ServerResponse): void {
    if (!this.vizDir) {
      json(res, { error: 'frontend directory not found' }, 404);
      return;
    }
    // Strip /viz prefix and treat /viz or /viz/ as index.html
    let rel = pathname.replace(/^\/viz\/?/, '') || 'index.html';
    if (rel.endsWith('/')) rel += 'index.html';
    // Path traversal guard
    const resolved = path.normalize(path.join(this.vizDir, rel));
    if (!resolved.startsWith(this.vizDir)) {
      json(res, { error: 'forbidden' }, 403);
      return;
    }
    if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
      json(res, { error: 'not found' }, 404);
      return;
    }
    const ext = path.extname(resolved).toLowerCase();
    const mime: Record<string, string> = {
      '.html': 'text/html; charset=utf-8',
      '.js': 'application/javascript; charset=utf-8',
      '.map': 'application/json; charset=utf-8',
      '.css': 'text/css; charset=utf-8',
      '.json': 'application/json; charset=utf-8',
      '.svg': 'image/svg+xml',
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.woff2': 'font/woff2',
    };
    const body = fs.readFileSync(resolved);
    res.writeHead(200, {
      'Content-Type': mime[ext] || 'application/octet-stream',
      'Content-Length': body.length,
    });
    res.end(body);
  }

  private handlePut(pathname: string, body: unknown, res: http.ServerResponse): void {

    switch (pathname) {
      case '/api/settings/firehose':
        this.handleFirehoseUpdate(body, res);
        break;
      case '/api/settings/keywords':
        this.handleKeywordsUpdate(body, res);
        break;
      case '/api/settings/signal':
        this.handleSignalUpdate(body, res);
        break;
      case '/api/settings/detector':
        this.handleDetectorUpdate(body, res);
        break;
      case '/api/settings/visualization':
        this.handleVisualizationUpdate(body, res);
        break;
      case '/api/settings/colors':
        this.handleColorsUpdate(body, res);
        break;
      case '/api/presets/apply':
        this.handleApplyPreset(body, res);
        break;
      default:
        json(res, { error: 'Not found' }, 404);
    }
  }

  private handleFirehoseUpdate(body: unknown, res: http.ServerResponse): void {
    if (typeof body !== 'object' || body === null) {
      json(res, { error: 'Body must be an object' }, 400);
      return;
    }
    const data = body as Record<string, unknown>;
    const firehose = this.config.firehose;

    if (data.endpoint !== undefined) {
      if (typeof data.endpoint !== 'string') {
        json(res, { error: 'endpoint must be a string' }, 400);
        return;
      }
      firehose.config.endpoint = data.endpoint;
    }
    if (data.retryInterval !== undefined) {
      if (typeof data.retryInterval !== 'number' || data.retryInterval < 100) {
        json(res, { error: 'retryInterval must be >= 100' }, 400);
        return;
      }
      firehose.config.retryInterval = data.retryInterval;
    }
    if (data.maxRetries !== undefined) {
      if (typeof data.maxRetries !== 'number' || data.maxRetries < 0) {
        json(res, { error: 'maxRetries must be >= 0' }, 400);
        return;
      }
      firehose.config.maxRetries = data.maxRetries;
    }

    json(res, { ok: true });
  }

  private handleKeywordsUpdate(body: unknown, res: http.ServerResponse): void {
    if (typeof body !== 'object' || body === null || Array.isArray(body)) {
      json(res, { error: 'Body must be an object' }, 400);
      return;
    }
    const data = body as Record<string, unknown>;

    for (const [emotionId, keywords] of Object.entries(data)) {
      if (!EMOTION_IDS.includes(emotionId)) {
        json(res, { error: `Unknown emotion: ${emotionId}` }, 400);
        return;
      }
      if (!Array.isArray(keywords)) {
        json(res, { error: `Keywords for ${emotionId} must be an array` }, 400);
        return;
      }
      for (const kw of keywords) {
        if (typeof kw !== 'string') {
          json(res, { error: `All keywords must be strings` }, 400);
          return;
        }
      }
    }

    const updates: Record<string, string[]> = {};
    for (const [emotionId, keywords] of Object.entries(data)) {
      updates[emotionId] = keywords as string[];
    }

    this.config.emotionDetector.updateKeywords(updates);
    json(res, { ok: true });
  }

  private handleSignalUpdate(body: unknown, res: http.ServerResponse): void {
    if (typeof body !== 'object' || body === null) {
      json(res, { error: 'Body must be an object' }, 400);
      return;
    }
    const data = body as Record<string, unknown>;
    const updates: Partial<{ minCutoff: number; beta: number }> = {};

    if (data.minCutoff !== undefined) {
      if (typeof data.minCutoff !== 'number' || data.minCutoff < 0.01 || data.minCutoff > 10) {
        json(res, { error: 'minCutoff must be between 0.01 and 10' }, 400);
        return;
      }
      updates.minCutoff = data.minCutoff;
    }
    if (data.beta !== undefined) {
      if (typeof data.beta !== 'number' || data.beta < 0 || data.beta > 1) {
        json(res, { error: 'beta must be between 0 and 1' }, 400);
        return;
      }
      updates.beta = data.beta;
    }

    this.config.signalProcessor.updateConfig(updates);
    json(res, { ok: true });
  }

  private handleDetectorUpdate(body: unknown, res: http.ServerResponse): void {
    if (typeof body !== 'object' || body === null) {
      json(res, { error: 'Body must be an object' }, 400);
      return;
    }
    const data = body as Record<string, unknown>;

    if (data.windowDuration !== undefined) {
      if (typeof data.windowDuration !== 'number' || data.windowDuration < 500 || data.windowDuration > 5000) {
        json(res, { error: 'windowDuration must be between 500 and 5000' }, 400);
        return;
      }
      this.config.emotionDetector.setWindowDuration(data.windowDuration);
    }

    json(res, { ok: true });
  }

  private handleVisualizationUpdate(body: unknown, res: http.ServerResponse): void {
    if (typeof body !== 'object' || body === null) {
      json(res, { error: 'Body must be an object' }, 400);
      return;
    }
    const data = body as Record<string, unknown>;
    const settings: { shaderMode?: number; feedbackStrength?: number; reducedMotion?: boolean } = {};

    if (data.shaderMode !== undefined) {
      if (typeof data.shaderMode !== 'number' || data.shaderMode < 0 || data.shaderMode >= SHADER_MODE_NAMES.length) {
        json(res, { error: `shaderMode must be between 0 and ${SHADER_MODE_NAMES.length - 1}` }, 400);
        return;
      }
      settings.shaderMode = data.shaderMode;
    }
    if (data.feedbackStrength !== undefined) {
      if (typeof data.feedbackStrength !== 'number' || data.feedbackStrength < 0 || data.feedbackStrength > 0.95) {
        json(res, { error: 'feedbackStrength must be between 0 and 0.95' }, 400);
        return;
      }
      settings.feedbackStrength = data.feedbackStrength;
    }
    if (data.reducedMotion !== undefined) {
      if (typeof data.reducedMotion !== 'boolean') {
        json(res, { error: 'reducedMotion must be a boolean' }, 400);
        return;
      }
      settings.reducedMotion = data.reducedMotion;
    }

    if (Object.keys(settings).length > 0) {
      this.config.wsServer.broadcastSettings(settings);
    }
    json(res, { ok: true });
  }

  private async handleColorsUpdate(body: unknown, res: http.ServerResponse): Promise<void> {
    if (typeof body !== 'object' || body === null || Array.isArray(body)) {
      json(res, { error: 'Body must be an object mapping emotion IDs to hex colors' }, 400);
      return;
    }
    const data = body as Record<string, unknown>;
    const hexRegex = /^#?[0-9a-fA-F]{6}$/;

    for (const [emotionId, hex] of Object.entries(data)) {
      if (!EMOTION_IDS.includes(emotionId)) {
        json(res, { error: `Unknown emotion: ${emotionId}` }, 400);
        return;
      }
      if (typeof hex !== 'string' || !hexRegex.test(hex)) {
        json(res, { error: `Invalid hex color for ${emotionId}: ${hex}` }, 400);
        return;
      }
    }

    // Merge with existing colors
    const current = this.config.contentLoader?.getState().colors ?? {};
    const merged: Record<string, string> = {};
    for (const [id, color] of Object.entries(current)) {
      merged[id] = color.hex;
    }
    for (const [id, hex] of Object.entries(data)) {
      merged[id] = (hex as string).startsWith('#') ? hex as string : `#${hex}`;
    }

    try {
      await this.config.contentLoader!.updateColors(merged);
      json(res, { ok: true });
    } catch (e) {
      json(res, { error: 'Failed to write colors' }, 500);
    }
  }

  private async handleApplyPreset(body: unknown, res: http.ServerResponse): Promise<void> {
    if (typeof body !== 'object' || body === null) {
      json(res, { error: 'Body must be { id: string }' }, 400);
      return;
    }
    const data = body as Record<string, unknown>;
    if (typeof data.id !== 'string') {
      json(res, { error: 'id must be a string' }, 400);
      return;
    }
    const preset = this.loadPreset(data.id);
    if (!preset) {
      json(res, { error: `Preset '${data.id}' not found` }, 404);
      return;
    }

    const applied: string[] = [];

    // visualization → broadcast to clients
    if (typeof preset.visualization === 'object' && preset.visualization !== null) {
      const v = preset.visualization as Record<string, unknown>;
      const settings: { shaderMode?: number; feedbackStrength?: number; reducedMotion?: boolean } = {};
      if (typeof v.shaderMode === 'number') settings.shaderMode = v.shaderMode;
      if (typeof v.feedbackStrength === 'number') settings.feedbackStrength = v.feedbackStrength;
      if (typeof v.reducedMotion === 'boolean') settings.reducedMotion = v.reducedMotion;
      if (Object.keys(settings).length > 0) {
        this.config.wsServer.broadcastSettings(settings);
        applied.push('visualization');
      }
    }

    // signal smoothing
    if (typeof preset.signal === 'object' && preset.signal !== null) {
      const s = preset.signal as Record<string, unknown>;
      const updates: Partial<{ minCutoff: number; beta: number }> = {};
      if (typeof s.minCutoff === 'number') updates.minCutoff = s.minCutoff;
      if (typeof s.beta === 'number') updates.beta = s.beta;
      if (Object.keys(updates).length > 0) {
        this.config.signalProcessor.updateConfig(updates);
        applied.push('signal');
      }
    }

    // detector window
    if (typeof preset.detector === 'object' && preset.detector !== null) {
      const d = preset.detector as Record<string, unknown>;
      if (typeof d.windowDuration === 'number') {
        this.config.emotionDetector.setWindowDuration(d.windowDuration);
        applied.push('detector');
      }
    }

    // keywords
    if (typeof preset.keywords === 'object' && preset.keywords !== null && !Array.isArray(preset.keywords)) {
      const kw = preset.keywords as Record<string, unknown>;
      const updates: Record<string, string[]> = {};
      for (const [id, list] of Object.entries(kw)) {
        if (Array.isArray(list) && list.every((x) => typeof x === 'string')) {
          updates[id] = list as string[];
        }
      }
      if (Object.keys(updates).length > 0) {
        this.config.emotionDetector.updateKeywords(updates);
        applied.push('keywords');
      }
    }

    // colors (persisted to colors.txt)
    if (typeof preset.colors === 'object' && preset.colors !== null && !Array.isArray(preset.colors) && this.config.contentLoader) {
      const cols = preset.colors as Record<string, unknown>;
      const merged: Record<string, string> = {};
      const current = this.config.contentLoader.getState().colors;
      for (const [id, c] of Object.entries(current)) merged[id] = c.hex;
      const hexRegex = /^#?[0-9a-fA-F]{6}$/;
      for (const [id, hex] of Object.entries(cols)) {
        if (typeof hex === 'string' && hexRegex.test(hex)) {
          merged[id] = hex.startsWith('#') ? hex : `#${hex}`;
        }
      }
      try {
        await this.config.contentLoader.updateColors(merged);
        applied.push('colors');
      } catch {
        // silently skip; the rest of the preset still applies
      }
    }

    // firehose endpoint / retry
    if (typeof preset.firehose === 'object' && preset.firehose !== null) {
      const f = preset.firehose as Record<string, unknown>;
      const fh = this.config.firehose;
      if (typeof f.endpoint === 'string') fh.config.endpoint = f.endpoint;
      if (typeof f.retryInterval === 'number' && f.retryInterval >= 100) fh.config.retryInterval = f.retryInterval;
      if (typeof f.maxRetries === 'number' && f.maxRetries >= 0) fh.config.maxRetries = f.maxRetries;
      applied.push('firehose');
    }

    json(res, { ok: true, applied, preset: { id: data.id, name: preset.name } });
  }

  private getAllSettings(): Record<string, unknown> {
    const firehose = this.config.firehose;
    return {
      firehose: {
        endpoint: firehose.config.endpoint || null,
        defaultEndpoint: firehose.getDefaultEndpoint?.() ?? null,
        effectiveEndpoint: firehose.getEffectiveEndpoint?.() ?? firehose.config.endpoint ?? null,
        retryInterval: firehose.config.retryInterval,
        maxRetries: firehose.config.maxRetries,
      },
      keywords: this.config.emotionDetector.getKeywords(),
      signal: this.config.signalProcessor.getConfig(),
      detector: {
        windowDuration: this.config.emotionDetector.getWindowDuration(),
      },
      visualization: {
        shaderModeNames: SHADER_MODE_NAMES,
      },
      colors: this.config.contentLoader?.getState().colors ?? {},
    };
  }

  private getStatus(): Record<string, unknown> {
    return {
      firehose: this.config.firehose.getStatus(),
      emotionDetector: this.config.emotionDetector.getStatus(),
      signalProcessor: this.config.signalProcessor.getStatus(),
      wsServer: this.config.wsServer.getStatus(),
      clients: this.config.wsServer.getClientCount(),
      uptime: Date.now() - this.startTime,
    };
  }

  getPort(): number {
    return this.port;
  }

  getHttpServer(): http.Server | null {
    return this.server;
  }
}
