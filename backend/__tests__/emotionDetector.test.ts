import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { EmotionDetector, EmotionalSignal, EmotionRatios } from '../emotionDetector.js';

function makeDetector() {
  return new EmotionDetector({ models: [], thresholds: {} });
}

describe('EmotionDetector', () => {
  let detector: EmotionDetector;

  beforeEach(async () => {
    detector = makeDetector();
    await detector.start();
  });

  afterEach(async () => {
    await detector.stop();
  });

  describe('processRawData', () => {
    it('detects a single emotion keyword', () => {
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));

      detector.processRawData('I feel so calm today');

      expect(signals).toHaveLength(1);
      expect(signals[0].emotions['serene']).toBe(1);
    });

    it('detects multiple emotions in one post', () => {
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));

      detector.processRawData('I feel calm and happy');

      expect(signals).toHaveLength(1);
      expect(signals[0].emotions['serene']).toBe(1);
      expect(signals[0].emotions['content']).toBe(1);
    });

    it('returns zero counts when no keywords match', () => {
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));

      detector.processRawData('xyzzy plugh frobnitz');

      expect(signals).toHaveLength(1);
      const allZero = Object.values(signals[0].emotions).every((v) => v === 0);
      expect(allZero).toBe(true);
    });

    it('handles empty string', () => {
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));

      detector.processRawData('');

      expect(signals).toHaveLength(1);
      const allZero = Object.values(signals[0].emotions).every((v) => v === 0);
      expect(allZero).toBe(true);
    });

    it('uses word-boundary matching to avoid false positives', () => {
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));

      // "bluetooth" should NOT match "blue" (melancholy keyword)
      detector.processRawData('my bluetooth headset broke');

      expect(signals[0].emotions['melancholy']).toBe(0);
    });

    it('matches keywords case-insensitively', () => {
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));

      detector.processRawData('I am CALM and PEACEFUL');

      expect(signals[0].emotions['serene']).toBe(2);
    });

    it('does not process data when detector is stopped', async () => {
      await detector.stop();
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));

      detector.processRawData('I feel calm');

      expect(signals).toHaveLength(0);
    });

    it('accepts RawDataPoint objects', () => {
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));

      detector.processRawData({
        timestamp: Date.now(),
        source: 'test',
        data: 'feeling peaceful',
      });

      expect(signals).toHaveLength(1);
      expect(signals[0].emotions['serene']).toBe(1);
      expect(signals[0].source).toBe('test');
    });
  });

  describe('window ratios', () => {
    it('emits ratios at the window interval', async () => {
      vi.useFakeTimers();

      const ratioDetector = new EmotionDetector({
        models: [],
        thresholds: {},
        windowDuration: 100,
      });
      await ratioDetector.start();

      const ratios: EmotionRatios[] = [];
      ratioDetector.on('ratios', (r: EmotionRatios) => ratios.push(r));

      ratioDetector.processRawData('I feel calm');
      ratioDetector.processRawData('I feel happy');

      vi.advanceTimersByTime(150);

      expect(ratios.length).toBeGreaterThanOrEqual(1);
      expect(ratios[0].totalHits).toBe(2);
      expect(ratios[0].ratios['serene']).toBe(0.5);
      expect(ratios[0].ratios['content']).toBe(0.5);

      await ratioDetector.stop();
      vi.useRealTimers();
    });

    it('produces zero ratios when no data in window', async () => {
      vi.useFakeTimers();

      const ratioDetector = new EmotionDetector({
        models: [],
        thresholds: {},
        windowDuration: 100,
      });
      await ratioDetector.start();

      const ratios: EmotionRatios[] = [];
      ratioDetector.on('ratios', (r: EmotionRatios) => ratios.push(r));

      vi.advanceTimersByTime(150);

      expect(ratios.length).toBeGreaterThanOrEqual(1);
      expect(ratios[0].totalHits).toBe(0);
      for (const ratio of Object.values(ratios[0].ratios)) {
        expect(ratio).toBe(0);
      }

      await ratioDetector.stop();
      vi.useRealTimers();
    });
  });

  describe('lifecycle', () => {
    it('throws when started twice', async () => {
      await expect(detector.start()).rejects.toThrow('already running');
    });

    it('emits started event', async () => {
      const d = makeDetector();
      const events: string[] = [];
      d.on('started', () => events.push('started'));
      await d.start();
      expect(events).toContain('started');
      await d.stop();
    });

    it('emits stopped event', async () => {
      const events: string[] = [];
      detector.on('stopped', () => events.push('stopped'));
      await detector.stop();
      expect(events).toContain('stopped');
    });
  });

  describe('getKeywords', () => {
    it('returns current keywords for all emotions', () => {
      const keywords = detector.getKeywords();
      expect(keywords).toHaveProperty('serene');
      expect(keywords).toHaveProperty('vibrant');
      expect(keywords).toHaveProperty('melancholy');
      expect(keywords).toHaveProperty('curious');
      expect(keywords).toHaveProperty('content');
      expect(Array.isArray(keywords.serene)).toBe(true);
      expect(keywords.serene.length).toBeGreaterThan(0);
    });

    it('returns the original keywords from shared emotions', () => {
      const keywords = detector.getKeywords();
      expect(keywords.serene).toContain('calm');
      expect(keywords.serene).toContain('peaceful');
      expect(keywords.vibrant).toContain('energetic');
      expect(keywords.melancholy).toContain('sad');
      expect(keywords.curious).toContain('interested');
      expect(keywords.content).toContain('happy');
    });
  });

  describe('updateKeywords', () => {
    it('updates keywords for a single emotion', () => {
      detector.updateKeywords({ serene: ['zen', 'mellow'] });
      const keywords = detector.getKeywords();
      expect(keywords.serene).toEqual(['zen', 'mellow']);
    });

    it('updates keywords for multiple emotions at once', () => {
      detector.updateKeywords({
        serene: ['zen'],
        vibrant: ['pumped'],
      });
      const keywords = detector.getKeywords();
      expect(keywords.serene).toEqual(['zen']);
      expect(keywords.vibrant).toEqual(['pumped']);
    });

    it('emits keywordsUpdated event with the new keyword map', () => {
      const updates: Record<string, string[]>[] = [];
      detector.on('keywordsUpdated', (map: Record<string, string[]>) => updates.push(map));
      const newKeywords = { serene: ['zen', 'chill'] };
      detector.updateKeywords(newKeywords);
      expect(updates).toHaveLength(1);
      expect(updates[0].serene).toEqual(['zen', 'chill']);
    });

    it('rebuilds patterns so new keywords are matched', () => {
      detector.updateKeywords({ serene: ['zen'] });
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));
      detector.processRawData('I feel so zen today');
      expect(signals).toHaveLength(1);
      expect(signals[0].emotions['serene']).toBe(1);
    });

    it('removed keywords are no longer matched', () => {
      detector.updateKeywords({ serene: [] });
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));
      detector.processRawData('I feel calm and peaceful');
      expect(signals[0].emotions['serene']).toBe(0);
    });

    it('handles empty keyword list gracefully', () => {
      expect(() => detector.updateKeywords({ serene: [] })).not.toThrow();
      const keywords = detector.getKeywords();
      expect(keywords.serene).toEqual([]);
    });

    it('does not affect other emotions when updating one', () => {
      const originalVibrant = detector.getKeywords().vibrant;
      detector.updateKeywords({ serene: ['zen'] });
      expect(detector.getKeywords().vibrant).toEqual(originalVibrant);
    });
  });
});
