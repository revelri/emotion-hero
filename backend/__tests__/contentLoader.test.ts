import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { ContentLoader } from '../contentLoader.js';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

describe('ContentLoader', () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'zeitgeist-test-'));
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  describe('color parsing', () => {
    it('loads colors from colors.txt', async () => {
      fs.writeFileSync(
        path.join(tmpDir, 'colors.txt'),
        'happy=#87a99e\nhorny=#ad9387\n'
      );

      const loader = new ContentLoader({ contentDir: tmpDir });
      await loader.start();

      const state = loader.getState();
      expect(state.colors['happy']).toBeDefined();
      expect(state.colors['happy'].hex).toBe('#87a99e');
      expect(state.colors['happy'].rgb).toEqual([0x87, 0xa9, 0x9e]);
      expect(state.colors['horny']).toBeDefined();

      await loader.stop();
    });

    it('skips comments and blank lines', async () => {
      fs.writeFileSync(
        path.join(tmpDir, 'colors.txt'),
        '# This is a comment\n\nhappy=#87a99e\n# Another comment\n'
      );

      const loader = new ContentLoader({ contentDir: tmpDir });
      await loader.start();

      const state = loader.getState();
      expect(Object.keys(state.colors)).toHaveLength(1);
      expect(state.colors['happy']).toBeDefined();

      await loader.stop();
    });

    it('handles missing colors file gracefully', async () => {
      const loader = new ContentLoader({ contentDir: tmpDir });
      await loader.start();

      const state = loader.getState();
      expect(Object.keys(state.colors)).toHaveLength(0);

      await loader.stop();
    });

    it('normalizes emotion IDs to lowercase', async () => {
      fs.writeFileSync(path.join(tmpDir, 'colors.txt'), 'Happy=#87a99e\n');

      const loader = new ContentLoader({ contentDir: tmpDir });
      await loader.start();

      const state = loader.getState();
      expect(state.colors['happy']).toBeDefined();

      await loader.stop();
    });
  });

  describe('ASCII art loading', () => {
    it('loads art.txt', async () => {
      const art = '  ***\n *****\n  ***';
      fs.writeFileSync(path.join(tmpDir, 'art.txt'), art);

      const loader = new ContentLoader({ contentDir: tmpDir });
      await loader.start();

      const state = loader.getState();
      expect(state.asciiArt).toBe(art);
      expect(state.artFileName).toBe('art.txt');

      await loader.stop();
    });

    it('returns null when no art file exists', async () => {
      const loader = new ContentLoader({ contentDir: tmpDir });
      await loader.start();

      const state = loader.getState();
      expect(state.asciiArt).toBeNull();
      expect(state.artFileName).toBeNull();

      await loader.stop();
    });

    it('picks art.txt over other .txt files', async () => {
      fs.writeFileSync(path.join(tmpDir, 'art.txt'), 'main art');
      fs.writeFileSync(path.join(tmpDir, 'other.txt'), 'other art');

      const loader = new ContentLoader({ contentDir: tmpDir });
      await loader.start();

      const state = loader.getState();
      expect(state.asciiArt).toBe('main art');

      await loader.stop();
    });

    it('falls back to other .txt file if no art.txt', async () => {
      fs.writeFileSync(path.join(tmpDir, 'heart.txt'), 'heart art');

      const loader = new ContentLoader({ contentDir: tmpDir });
      await loader.start();

      const state = loader.getState();
      expect(state.asciiArt).toBe('heart art');
      expect(state.artFileName).toBe('heart.txt');

      await loader.stop();
    });
  });

  describe('lifecycle', () => {
    it('creates content directory if missing', async () => {
      const nestedDir = path.join(tmpDir, 'nested', 'content');
      const loader = new ContentLoader({ contentDir: nestedDir });
      await loader.start();

      expect(fs.existsSync(nestedDir)).toBe(true);

      await loader.stop();
    });

    it('throws when started twice', async () => {
      const loader = new ContentLoader({ contentDir: tmpDir });
      await loader.start();

      await expect(loader.start()).rejects.toThrow('already running');

      await loader.stop();
    });

    it('stop is idempotent', async () => {
      const loader = new ContentLoader({ contentDir: tmpDir });
      await loader.start();
      await loader.stop();
      await loader.stop();
    });
  });

  describe('getStatus', () => {
    it('returns status object', async () => {
      fs.writeFileSync(path.join(tmpDir, 'colors.txt'), 'happy=#87a99e\n');

      const loader = new ContentLoader({ contentDir: tmpDir });
      await loader.start();

      const status = loader.getStatus();
      expect(status.isRunning).toBe(true);
      expect(status.emotionCount).toBe(1);

      await loader.stop();
    });
  });
});
