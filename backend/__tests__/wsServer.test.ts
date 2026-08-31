import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { WsServer } from '../wsServer.js';
import * as WebSocket from 'ws';

const TEST_PORT = 18765;

function createClient(): { ws: WebSocket.WebSocket; messages: any[]; waitForMessages: (n: number) => Promise<any[]> } {
  const ws = new WebSocket.WebSocket(`ws://localhost:${TEST_PORT}`);
  const messages: any[] = [];
  const waiters: Array<{ count: number; resolve: (msgs: any[]) => void }> = [];

  ws.on('message', (data: WebSocket.RawData) => {
    messages.push(JSON.parse(data.toString()));
    for (let i = waiters.length - 1; i >= 0; i--) {
      if (messages.length >= waiters[i].count) {
        waiters[i].resolve(messages.slice(0, waiters[i].count));
        waiters.splice(i, 1);
      }
    }
  });

  function waitForMessages(n: number): Promise<any[]> {
    if (messages.length >= n) return Promise.resolve(messages.slice(0, n));
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error(`Timeout waiting for ${n} messages (got ${messages.length})`)), 3000);
      waiters.push({
        count: n,
        resolve: (msgs) => {
          clearTimeout(timeout);
          resolve(msgs);
        },
      });
    });
  }

  return { ws, messages, waitForMessages };
}

function waitForOpen(ws: WebSocket.WebSocket): Promise<void> {
  return new Promise((resolve, reject) => {
    if (ws.readyState === WebSocket.WebSocket.OPEN) return resolve();
    ws.once('open', resolve);
    ws.once('error', reject);
  });
}

describe('WsServer', () => {
  let server: WsServer;

  beforeEach(async () => {
    server = new WsServer({ port: TEST_PORT });
    await server.start();
  });

  afterEach(async () => {
    await server.stop();
  });

  it('accepts client connections', async () => {
    const { ws } = createClient();
    await waitForOpen(ws);

    expect(server.getClientCount()).toBe(1);

    ws.close();
    await new Promise((r) => setTimeout(r, 100));
    expect(server.getClientCount()).toBe(0);
  });

  it('sends initial emotion state on connect', async () => {
    server.updateState({
      happy: { value: 0.5, velocity: 0.1 },
    });

    const { ws, waitForMessages } = createClient();
    const msgs = await waitForMessages(2);

    const types = msgs.map((m) => m.type);
    expect(types).toContain('emotions');
    expect(types).toContain('content');

    ws.close();
  });

  it('broadcasts emotion updates to all clients', async () => {
    const c1 = createClient();
    const c2 = createClient();

    await c1.waitForMessages(2);
    await c2.waitForMessages(2);

    c1.messages.length = 0;
    c2.messages.length = 0;

    server.updateState({
      happy: { value: 0.8, velocity: 0.2 },
    });

    await new Promise((r) => setTimeout(r, 200));
    expect(c1.messages.length).toBeGreaterThanOrEqual(1);
    expect(c1.messages[0].type).toBe('emotions');
    expect(c1.messages[0].emotions.happy.value).toBe(0.8);

    c1.ws.close();
    c2.ws.close();
  });

  it('broadcasts content updates', async () => {
    const c = createClient();
    await c.waitForMessages(2);
    c.messages.length = 0;

    server.updateContent({
      asciiArt: '***',
      artFileName: 'test.txt',
      colors: {},
    });

    await new Promise((r) => setTimeout(r, 200));
    expect(c.messages.length).toBeGreaterThanOrEqual(1);
    expect(c.messages[0].type).toBe('content');
    expect(c.messages[0].asciiArt).toBe('***');

    c.ws.close();
  });

  it('handles client disconnect gracefully', async () => {
    const { ws } = createClient();
    await waitForOpen(ws);
    expect(server.getClientCount()).toBe(1);

    ws.terminate();
    await new Promise((r) => setTimeout(r, 100));

    expect(server.getClientCount()).toBe(0);

    // Broadcast should not throw with 0 clients
    server.updateState({ happy: { value: 0.5, velocity: 0 } });
  });

  it('rate limits broadcasts', async () => {
    const c = createClient();
    await c.waitForMessages(2);
    c.messages.length = 0;

    for (let i = 0; i < 100; i++) {
      server.updateState({ happy: { value: i / 100, velocity: 0 } });
    }

    await new Promise((r) => setTimeout(r, 300));

    expect(c.messages.length).toBeLessThan(100);
    expect(c.messages.length).toBeGreaterThanOrEqual(1);

    c.ws.close();
  });

  describe('lifecycle', () => {
    it('throws when started twice', async () => {
      await expect(server.start()).rejects.toThrow('already running');
    });

    it('stop is idempotent', async () => {
      await server.stop();
      await server.stop();
    });

    it('reports running status', () => {
      const status = server.getStatus();
      expect(status.isRunning).toBe(true);
      expect(status.port).toBe(TEST_PORT);
    });
  });
});
