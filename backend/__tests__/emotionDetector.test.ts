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

      detector.processRawData('I feel so happy today');

      expect(signals).toHaveLength(1);
      expect(signals[0].emotions['happy']).toBe(1);
    });

    it('detects multiple emotions in one post', () => {
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));

      detector.processRawData('I feel happy and depressed');

      expect(signals).toHaveLength(1);
      expect(signals[0].emotions['happy']).toBe(1);
      expect(signals[0].emotions['depressed']).toBe(1);
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

      // "angrier" should NOT match "angry" (only exact word matches)
      detector.processRawData('I am getting angrier by the minute');

      expect(signals[0].emotions['angry']).toBe(0);
    });

    it('matches keywords case-insensitively', () => {
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));

      detector.processRawData('I am HAPPY today');

      expect(signals[0].emotions['happy']).toBe(1);
    });

    it('does not process data when detector is stopped', async () => {
      await detector.stop();
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));

      detector.processRawData('I feel happy');

      expect(signals).toHaveLength(0);
    });

    it('accepts RawDataPoint objects', () => {
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));

      detector.processRawData({
        timestamp: Date.now(),
        source: 'test',
        data: 'feeling happy',
      });

      expect(signals).toHaveLength(1);
      expect(signals[0].emotions['happy']).toBe(1);
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

      ratioDetector.processRawData('I feel happy');
      ratioDetector.processRawData('I feel depressed');

      vi.advanceTimersByTime(150);

      expect(ratios.length).toBeGreaterThanOrEqual(1);
      expect(ratios[0].totalHits).toBe(2);
      expect(ratios[0].ratios['happy']).toBe(0.5);
      expect(ratios[0].ratios['depressed']).toBe(0.5);

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
      expect(keywords).toHaveProperty('happy');
      expect(keywords).toHaveProperty('horny');
      expect(keywords).toHaveProperty('angry');
      expect(keywords).toHaveProperty('depressed');
      expect(Array.isArray(keywords.happy)).toBe(true);
      expect(keywords.happy.length).toBeGreaterThan(0);
    });

    it('returns the original keywords from shared emotions', () => {
      const keywords = detector.getKeywords();
      expect(keywords.happy).toEqual(['happy']);
      expect(keywords.horny).toEqual(['horny']);
      expect(keywords.angry).toEqual(['angry']);
      expect(keywords.depressed).toEqual(['depressed']);
    });
  });

  describe('updateKeywords', () => {
    it('updates keywords for a single emotion', () => {
      detector.updateKeywords({ happy: ['zen', 'mellow'] });
      const keywords = detector.getKeywords();
      expect(keywords.happy).toEqual(['zen', 'mellow']);
    });

    it('updates keywords for multiple emotions at once', () => {
      detector.updateKeywords({
        happy: ['zen'],
        horny: ['pumped'],
      });
      const keywords = detector.getKeywords();
      expect(keywords.happy).toEqual(['zen']);
      expect(keywords.horny).toEqual(['pumped']);
    });

    it('emits keywordsUpdated event with the new keyword map', () => {
      const updates: Record<string, string[]>[] = [];
      detector.on('keywordsUpdated', (map: Record<string, string[]>) => updates.push(map));
      const newKeywords = { happy: ['zen', 'chill'] };
      detector.updateKeywords(newKeywords);
      expect(updates).toHaveLength(1);
      expect(updates[0].happy).toEqual(['zen', 'chill']);
    });

    it('rebuilds patterns so new keywords are matched', () => {
      detector.updateKeywords({ happy: ['zen'] });
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));
      detector.processRawData('I feel so zen today');
      expect(signals).toHaveLength(1);
      expect(signals[0].emotions['happy']).toBe(1);
    });

    it('removed keywords are no longer matched', () => {
      detector.updateKeywords({ happy: [] });
      const signals: EmotionalSignal[] = [];
      detector.on('signal', (s: EmotionalSignal) => signals.push(s));
      detector.processRawData('I feel happy and very happy indeed');
      expect(signals[0].emotions['happy']).toBe(0);
    });

    it('handles empty keyword list gracefully', () => {
      expect(() => detector.updateKeywords({ happy: [] })).not.toThrow();
      const keywords = detector.getKeywords();
      expect(keywords.happy).toEqual([]);
    });

    it('does not affect other emotions when updating one', () => {
      const originalHorny = detector.getKeywords().horny;
      detector.updateKeywords({ happy: ['zen'] });
      expect(detector.getKeywords().horny).toEqual(originalHorny);
    });
  });
});
