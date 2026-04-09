import { describe, it, expect, beforeEach, vi } from 'vitest';
import { EmotionStateManager } from '../emotionStateManager.js';

describe('EmotionStateManager', () => {
  let manager: EmotionStateManager;

  beforeEach(() => {
    manager = new EmotionStateManager({
      springFrequency: 2.0,
      springDamping: 1.0,
      maxPacketAge: 1000,
      enableInterpolation: true,
    });
  });

  describe('updateFromBackend', () => {
    it('sets target values from backend data', () => {
      const now = Date.now();
      manager.updateFromBackend(
        { serene: { value: 0.8, velocity: 0.1 } },
        now
      );

      const state = manager.getState('serene');
      expect(state).toBeDefined();
      expect(state!.targetValue).toBe(0.8);
    });

    it('drops stale packets', () => {
      const staleTimestamp = Date.now() - 2000;
      manager.updateFromBackend(
        { serene: { value: 0.8, velocity: 0.1 } },
        staleTimestamp
      );

      const state = manager.getState('serene');
      expect(state!.targetValue).toBe(0);
    });

    it('ignores unknown emotion IDs without crashing', () => {
      const now = Date.now();
      expect(() => {
        manager.updateFromBackend(
          { nonexistent: { value: 0.5, velocity: 0 } },
          now
        );
      }).not.toThrow();
    });
  });

  describe('interpolate', () => {
    it('moves currentValue toward targetValue', () => {
      const now = Date.now();
      manager.updateFromBackend(
        { serene: { value: 1.0, velocity: 0 } },
        now
      );

      for (let i = 0; i < 10; i++) {
        manager.interpolate(1 / 60);
      }

      const state = manager.getState('serene');
      expect(state!.currentValue).toBeGreaterThan(0);
      expect(state!.currentValue).toBeLessThan(1.0);
    });

    it('converges to target over many frames', () => {
      const now = Date.now();
      manager.updateFromBackend(
        { serene: { value: 0.8, velocity: 0 } },
        now
      );

      for (let i = 0; i < 200; i++) {
        manager.interpolate(1 / 60);
      }

      const state = manager.getState('serene');
      expect(state!.currentValue).toBeCloseTo(0.8, 1);
    });

    it('snaps to target when interpolation is disabled', () => {
      manager.updateConfig({ enableInterpolation: false });

      const now = Date.now();
      manager.updateFromBackend(
        { serene: { value: 0.8, velocity: 0 } },
        now
      );
      manager.interpolate(1 / 60);

      const state = manager.getState('serene');
      expect(state!.currentValue).toBe(0.8);
    });

    it('clamps deltaTime to prevent huge jumps', () => {
      const now = Date.now();
      manager.updateFromBackend(
        { serene: { value: 1.0, velocity: 0 } },
        now
      );

      // Simulate a huge lag spike (10 seconds)
      manager.interpolate(10.0);

      const state = manager.getState('serene');
      // Should still be reasonable, not have jumped wildly
      expect(state!.currentValue).toBeLessThanOrEqual(1.0);
      expect(state!.currentValue).toBeGreaterThanOrEqual(0);
    });
  });

  describe('degradation mode', () => {
    it('enters degradation after timeout', () => {
      vi.useFakeTimers();

      const now = Date.now();
      manager.updateFromBackend(
        { serene: { value: 0.8, velocity: 0 } },
        now
      );

      // Run some frames to establish value
      for (let i = 0; i < 30; i++) {
        manager.interpolate(1 / 60);
      }

      // Advance time past degradation timeout
      vi.advanceTimersByTime(3000);

      // In degradation, target is driven to 0 and spring settles toward it
      for (let i = 0; i < 200; i++) {
        manager.interpolate(1 / 60);
      }

      const state = manager.getState('serene');
      // Should have decayed toward 0
      expect(state!.currentValue).toBeLessThan(0.4);

      vi.useRealTimers();
    });

    it('resets degradation on new data', () => {
      vi.useFakeTimers();

      const now = Date.now();
      manager.updateFromBackend(
        { serene: { value: 0.8, velocity: 0 } },
        now
      );

      vi.advanceTimersByTime(3000);
      manager.interpolate(1 / 60);

      // New data arrives
      manager.updateFromBackend(
        { serene: { value: 0.9, velocity: 0 } },
        Date.now()
      );

      const state = manager.getState('serene');
      expect(state!.targetValue).toBe(0.9);

      vi.useRealTimers();
    });

    it('resetDegradation exits degradation mode', () => {
      manager.resetDegradation();
      manager.interpolate(1 / 60);
    });
  });

  describe('getCurrentStates', () => {
    it('returns all emotion states', () => {
      const states = manager.getCurrentStates();
      expect(Object.keys(states).length).toBeGreaterThanOrEqual(5);
      expect(states).toHaveProperty('serene');
      expect(states).toHaveProperty('vibrant');
      expect(states).toHaveProperty('melancholy');
      expect(states).toHaveProperty('curious');
      expect(states).toHaveProperty('content');
    });

    it('returns EmotionState shape with value and velocity', () => {
      const states = manager.getCurrentStates();
      for (const state of Object.values(states)) {
        expect(state).toHaveProperty('value');
        expect(state).toHaveProperty('velocity');
        expect(typeof state.value).toBe('number');
        expect(typeof state.velocity).toBe('number');
      }
    });
  });

  describe('hasReceivedData', () => {
    it('is false initially', () => {
      expect(manager.getHasReceivedData()).toBe(false);
    });

    it('becomes true after receiving data', () => {
      manager.updateFromBackend(
        { serene: { value: 0.5, velocity: 0 } },
        Date.now()
      );
      expect(manager.getHasReceivedData()).toBe(true);
    });
  });

  describe('config', () => {
    it('uses sensible defaults', () => {
      const defaultManager = new EmotionStateManager();
      const config = defaultManager.getConfig();
      expect(config.lerpFactor).toBe(0.05);
      expect(config.maxPacketAge).toBe(1000);
      expect(config.enableInterpolation).toBe(true);
      expect(config.springFrequency).toBe(2.0);
      expect(config.springDamping).toBe(1.0);
    });

    it('merges partial config with defaults', () => {
      const config = manager.getConfig();
      expect(config.springFrequency).toBe(2.0);
      expect(config.degradationTimeout).toBe(2000);
    });
  });
});
