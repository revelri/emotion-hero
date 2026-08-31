/**
 * Signal Processor - Stabilizes emotional ratios into calm, continuous signals
 * 
 * This module implements signal processing for emotional data using:
 * - Rolling median window to remove spikes from raw emotional data
 * - EMA (Exponential Moving Average) toward the median for viscosity/smoothing
 * - Velocity tracking (delta over time) - never clamped
 * 
 * Output format: { value, velocity }
 */

import { EventEmitter } from 'events';
import { EmotionState, EMOTION_IDS } from '@zeitgeist/shared/emotions.js';

/**
 * Configuration for the signal processor (One-Euro Filter)
 */
export interface SignalProcessorConfig {
  /** Minimum cutoff frequency in Hz -- lower = more smoothing when calm (default 1.0) */
  minCutoff: number;
  /** Speed coefficient -- higher = less lag during fast changes (default 0.007) */
  beta: number;
}

/**
 * Per-emotion state for the One-Euro filter
 */
interface EmotionProcessorState {
  /** Current smoothed value */
  value: number;
  /** Previous value for velocity calculation */
  prevValue: number;
  /** Previous raw input for speed estimation */
  prevRaw: number;
  /** Timestamp of last process() call (ms) */
  lastTimestamp: number;
}

/**
 * Signal processor that stabilizes emotional ratios into calm, continuous signals
 *
 * Artistic Intent:
 * Signal smoothing transforms raw emotional data into fluid, organic movement.
 * Without smoothing, emotions would jitter and flicker like a broken lightbulb.
 * With smoothing, they flow like water - each emotion rising and falling naturally,
 * creating a living, breathing visualization that mirrors the subtle dance of human feeling.
 * The rolling median removes sudden spikes (the noise of life), while the EMA adds viscosity
 * (the weight of experience), resulting in emotional transitions that feel intentional and meaningful.
 */
export class SignalProcessor extends EventEmitter {
  private config: SignalProcessorConfig;
  private emotionStates: Map<string, EmotionProcessorState>;

  constructor(config: SignalProcessorConfig = { minCutoff: 1.0, beta: 0.007 }) {
    super();
    this.config = config;
    this.emotionStates = new Map();

    // Artistic Intent: The One-Euro filter adapts its smoothing to the signal's speed.
    // When emotions are calm, minCutoff=1.0 gives heavy filtering -- still water.
    // When emotions surge, beta=0.007 raises the cutoff to let fast changes through.
    // The result: the visualization breathes slowly in quiet moments and snaps to
    // attention during emotional events, mirroring how human attention works.

    const now = Date.now();
    for (const emotionId of EMOTION_IDS) {
      this.emotionStates.set(emotionId, {
        value: 0,
        prevValue: 0,
        prevRaw: 0,
        lastTimestamp: now,
      });
    }
  }

  /**
   * Process a raw emotional ratio and return the smoothed signal
   * 
   * Algorithm:
   * 1. Add raw value to history buffer
   * 2. Calculate median of history (removes spikes)
   * 3. Apply EMA toward median (adds viscosity)
   * 4. Calculate velocity (delta over time, never clamped)
   * 
   * @param emotionId - The emotion identifier
   * @param rawValue - The raw emotional ratio (0-1)
   * @returns Processed signal with value and velocity
   */
  process(emotionId: string, rawValue: number): EmotionState {
    let state = this.emotionStates.get(emotionId);
    const now = Date.now();

    // Initialize state for unknown emotions
    if (!state) {
      state = {
        value: 0,
        prevValue: 0,
        prevRaw: rawValue,
        lastTimestamp: now,
      };
      this.emotionStates.set(emotionId, state);
    }

    // One-Euro Filter: adaptive low-pass that adjusts cutoff based on signal speed
    const dt = Math.max((now - state.lastTimestamp) / 1000, 0.001); // seconds, min 1ms
    const speed = Math.abs(rawValue - state.prevRaw) / dt;
    const cutoff = this.config.minCutoff + this.config.beta * speed;
    const alpha = 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));

    // Store previous value for velocity calculation
    state.prevValue = state.value;

    // Apply adaptive EMA
    state.value += alpha * (rawValue - state.value);

    // Update state for next call
    state.prevRaw = rawValue;
    state.lastTimestamp = now;

    // Calculate velocity - never clamped
    const velocity = state.value - state.prevValue;

    // Emit processed event with all current emotion states
    const allStates: Record<string, EmotionState> = {};

    // Add tiny baseline so no emotion is truly invisible, then normalize
    // Old value (0.1) flattened ratios to ~0.2 when raw signals were small
    const minBaseline = 0.001;
    const numEmotions = this.emotionStates.size;

    let totalValue = 0;
    Array.from(this.emotionStates.values()).forEach(s => {
      totalValue += s.value + minBaseline;
    });

    Array.from(this.emotionStates.entries()).forEach(([id, s]) => {
      const valueWithBaseline = s.value + minBaseline;
      const normalizedValue = totalValue > 0 ? valueWithBaseline / totalValue : 1 / numEmotions;
      allStates[id] = { value: normalizedValue, velocity: s.value - s.prevValue };
    });
    this.emit('processed', allStates);

    return allStates[emotionId] ?? { value: state.value, velocity };
  }

  /**
   * Get the current state for a specific emotion
   * 
   * @param emotionId - The emotion identifier
   * @returns Current emotion state or undefined if not found
   */
  getState(emotionId: string): EmotionState | undefined {
    const state = this.emotionStates.get(emotionId);
    if (!state) {
      return undefined;
    }
    return { value: state.value, velocity: state.value - state.prevValue };
  }

  /**
   * Get the current state for all emotions
   *
   * @returns Map of emotion IDs to their current states
   */
  getAllStates(): Map<string, EmotionState> {
    const result = new Map<string, EmotionState>();
    Array.from(this.emotionStates.entries()).forEach(([emotionId, state]) => {
      result.set(emotionId, {
        value: state.value,
        velocity: state.value - state.prevValue,
      });
    });
    return result;
  }

  /**
   * Update the processor configuration
   *
   * @param newConfig - Partial configuration to update
   */
  updateConfig(newConfig: Partial<SignalProcessorConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Get the current configuration
   *
   * @returns Current processor configuration
   */
  getConfig(): SignalProcessorConfig {
    return { ...this.config };
  }

  /**
   * Start the signal processor
   *
   * Artistic Intent: Starting the processor awakens the emotional canvas,
   * preparing it to receive and transform raw emotional data into fluid signals.
   * The processor begins its journey of smoothing and stabilizing, ready to
   * capture the subtle nuances of human feeling as they flow through time.
   */
  async start(): Promise<void> {
    // Artistic Intent: The processor is stateless and always ready
    // This method exists for lifecycle consistency with other components
    // No initialization needed - the emotional canvas is prepared from birth
  }

  /**
   * Stop the signal processor
   *
   * Artistic Intent: Stopping the processor preserves the final emotional state,
   * freezing the emotional landscape at its current moment. The emotional memory
   * remains intact, allowing the system to resume where it left off, maintaining
   * the continuity of feeling across restarts.
   */
  async stop(): Promise<void> {
    // Artistic Intent: The processor gracefully halts without losing state
    // Emotional history is preserved in the emotionStates map
    // This allows for seamless resumption, maintaining emotional continuity
  }

  /**
   * Get the current status of the signal processor
   *
   * @returns Status object with processor state information
   */
  getStatus(): { isRunning: boolean; emotionCount: number; config: SignalProcessorConfig } {
    // Artistic Intent: Status provides a window into the processor's soul
    // Revealing how many emotions are being tracked and the smoothing parameters
    // that govern the viscosity of emotional transitions
    return {
      isRunning: true, // Processor is always ready to process
      emotionCount: this.emotionStates.size,
      config: this.getConfig(),
    };
  }
}
