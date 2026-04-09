/**
 * Client-side emotion state manager with smooth interpolation.
 *
 * This class handles maintaining target vs current state for each emotion,
 * smooth interpolation toward target values, velocity calculation from
 * interpolation (not from backend), and packet staleness detection.
 */

import { EMOTION_IDS } from '../shared/emotions.js';
import { EmotionState, InterpolationState, InterpolationConfig } from '../shared/types.js';

/**
 * Manages client-side emotion state with smooth interpolation.
 */
export class EmotionStateManager {
  /** Map of emotion IDs to their interpolation states */
  private states: Map<string, InterpolationState>;
  
  /** Configuration for interpolation behavior */
  private config: InterpolationConfig;

  /** Track if we're in degradation mode (no backend updates) */
  private isDegraded: boolean;

  /** Track when we last received data from backend */
  private lastBackendUpdateTime: number;

  /** Whether we've ever received data from the backend */
  private hasReceivedData: boolean;

  /**
   * Create a new EmotionStateManager.
   * @param config - Optional partial configuration for interpolation
   */
  constructor(config?: Partial<InterpolationConfig>) {
    this.config = {
      lerpFactor: 0.05,
      maxPacketAge: 1000,
      enableInterpolation: true,
      degradationTimeout: 2000,
      velocityDecayRate: 0.95,
      valueDecayRate: 0.98,
      enableDegradationMode: true,
      // Spring dynamics: slight underdamping (0.95) creates organic overshoot
      // Emotions swing into position with a satisfying settling quality
      // 2 Hz frequency means transitions complete in ~0.5 seconds
      springFrequency: 2.0,
      springDamping: 1.0, // Critical damping: fastest approach without overshoot
      ...config
    };

    // Initialize degradation tracking
    this.isDegraded = false;
    this.hasReceivedData = false;
    this.lastBackendUpdateTime = Date.now();
    
    // Initialize states for all emotions
    this.states = new Map();
    for (const id of EMOTION_IDS) {
      this.states.set(id, {
        targetValue: 0,
        currentValue: 0,
        prevValue: 0,
        currentVelocity: 0,
        springVelocity: 0,
        lastUpdateTime: Date.now()
      });
    }
  }

  /**
   * Update target values from backend emotion data.
   * Drops stale packets based on timestamp.
   * @param emotions - Record of emotion IDs to their states from backend
   * @param timestamp - Timestamp when the data was generated (ms)
   */
  updateFromBackend(emotions: Record<string, EmotionState>, timestamp: number): void {
    const now = Date.now();
    const packetAge = now - timestamp;
    
    // Drop stale packets
    if (packetAge > this.config.maxPacketAge) {
      return;
    }

    // Update last backend update time and reset degradation state
    this.lastBackendUpdateTime = now;
    this.isDegraded = false;
    this.hasReceivedData = true;

    for (const [id, emotion] of Object.entries(emotions)) {
      const state = this.states.get(id);
      if (state) {
        // Store previous value before updating
        state.prevValue = state.currentValue;
        // Update target value from backend
        state.targetValue = emotion.value;
        state.lastUpdateTime = timestamp;
      }
    }
  }

  /**
   * Perform interpolation step.
   * Should be called each frame with deltaTime in seconds.
   * @param deltaTime - Time since last frame in seconds
   */
  interpolate(deltaTime: number): void {
    // Check for degradation mode
    this.checkDegradation();

    if (!this.config.enableInterpolation) {
      for (const state of this.states.values()) {
        state.currentValue = state.targetValue;
        state.currentVelocity = 0;
        state.springVelocity = 0;
      }
      return;
    }

    // Clamp deltaTime to prevent huge jumps on lag spikes
    const clampedDelta = Math.min(deltaTime, 0.1);

    // Spring parameters
    const omega = 2 * Math.PI * this.config.springFrequency;
    const zeta = this.config.springDamping;

    for (const state of this.states.values()) {
      // In degradation mode, drive target toward zero and let spring settle naturally
      if (this.isDegraded && this.config.enableDegradationMode) {
        state.targetValue = 0;
      }

      // Critically damped spring: second-order dynamics
      // Uses semi-implicit Euler for stability (update velocity first, then position)
      const displacement = state.targetValue - state.currentValue;
      const acceleration = omega * omega * displacement - 2 * zeta * omega * state.springVelocity;

      // Semi-implicit Euler: velocity first, then position using new velocity
      state.springVelocity += acceleration * clampedDelta;
      state.currentValue += state.springVelocity * clampedDelta;

      // Clamp to valid range
      state.currentValue = Math.max(0, Math.min(1, state.currentValue));

      // Expose spring velocity as the current velocity for shader consumption
      state.currentVelocity = state.springVelocity;

      // Update prevValue for compatibility
      state.prevValue = state.currentValue;
    }
  }

  /**
   * Get current interpolated emotion states.
   * Returns EmotionState objects with interpolated values and calculated velocities.
   * @returns Record of emotion IDs to their current interpolated states
   */
  getCurrentStates(): Record<string, EmotionState> {
    const result: Record<string, EmotionState> = {};
    for (const [id, state] of this.states.entries()) {
      result[id] = {
        value: state.currentValue,
        velocity: state.currentVelocity
      };
    }
    return result;
  }

  /**
   * Get interpolation state for a specific emotion.
   * @param id - The emotion ID
   * @returns The interpolation state for the emotion, or undefined if not found
   */
  getState(id: string): InterpolationState | undefined {
    return this.states.get(id);
  }

  /**
   * Update configuration.
   * @param config - Partial configuration to update
   */
  updateConfig(config: Partial<InterpolationConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get current configuration.
   * @returns A copy of the current configuration
   */
  getConfig(): InterpolationConfig {
    return { ...this.config };
  }

  /**
   * Check if we should enter degradation mode based on time since last update.
   * @returns Whether we're currently in degradation mode
   *
   * Artistic Intent: Degradation mode transforms disconnection into poetry
   * When the emotional stream is interrupted, the field doesn't simply freeze
   * Instead, it enters a state of graceful dissolution, like a flower wilting
   * This creates an emotional narrative: the field was alive, now it fades
   * The decay rates (0.95 for velocity, 0.98 for values) create a two-stage fade
   * First movement stops, then color slowly returns to neutral
   * This mirrors the human experience of emotion after its source is gone
   * We don't stop feeling immediately; feelings linger, then slowly dissolve
   */
  private checkDegradation(): boolean {
    const now = Date.now();
    const timeSinceLastUpdate = now - this.lastBackendUpdateTime;
    
    if (timeSinceLastUpdate > this.config.degradationTimeout) {
      this.isDegraded = true;
    }
    
    return this.isDegraded;
  }

  /**
   * Reset degradation state.
   * Call this when reconnection is detected or when you want to exit degradation mode.
   */
  resetDegradation(): void {
    this.isDegraded = false;
    this.lastBackendUpdateTime = Date.now();
  }

  /**
   * Check if any data has been received from the backend.
   * Used to distinguish "waiting for first data" from "data stopped coming."
   */
  getHasReceivedData(): boolean {
    return this.hasReceivedData;
  }
}
