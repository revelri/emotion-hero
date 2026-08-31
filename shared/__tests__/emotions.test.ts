import { describe, it, expect } from 'vitest';
import { EMOTIONS, EMOTION_MAP, EMOTION_IDS } from '../emotions.js';

describe('EMOTIONS', () => {
  it('has exactly 4 emotion entries', () => {
    expect(EMOTIONS).toHaveLength(4);
  });

  it('each emotion has id, keywords, and colorRGB', () => {
    for (const emotion of EMOTIONS) {
      expect(typeof emotion.id).toBe('string');
      expect(emotion.id.length).toBeGreaterThan(0);
      expect(Array.isArray(emotion.keywords)).toBe(true);
      expect(emotion.keywords.length).toBeGreaterThan(0);
      expect(emotion.colorRGB).toHaveLength(3);
      for (const c of emotion.colorRGB) {
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(255);
      }
    }
  });

  it('has unique emotion IDs', () => {
    const ids = EMOTIONS.map((e) => e.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('contains expected emotion types', () => {
    const ids = EMOTIONS.map((e) => e.id);
    expect(ids).toContain('happy');
    expect(ids).toContain('horny');
    expect(ids).toContain('angry');
    expect(ids).toContain('depressed');
  });
});

describe('EMOTION_MAP', () => {
  it('maps every emotion ID to its Emotion object', () => {
    for (const emotion of EMOTIONS) {
      expect(EMOTION_MAP[emotion.id]).toBe(emotion);
    }
  });

  it('has the same count as EMOTIONS', () => {
    expect(Object.keys(EMOTION_MAP)).toHaveLength(EMOTIONS.length);
  });
});

describe('EMOTION_IDS', () => {
  it('matches EMOTIONS order', () => {
    for (let i = 0; i < EMOTIONS.length; i++) {
      expect(EMOTION_IDS[i]).toBe(EMOTIONS[i].id);
    }
  });
});
