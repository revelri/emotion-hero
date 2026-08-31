/**
 * Shared emotion model for the real-time emotional visualization system.
 * These types and data are platform-agnostic and can be used by both the backend
 * and frontend components of the system.
 */

/**
 * Represents an emotion type with associated metadata.
 */
export interface Emotion {
  /** Unique string identifier for this emotion */
  id: string;
  /** Keywords associated with this emotion for matching and classification */
  keywords: string[];
  /** RGB color values for visualization (0-255 range) */
  colorRGB: [number, number, number];
}

/**
 * Represents the dynamic state of an emotion at a given moment.
 */
export interface EmotionState {
  /** Current intensity value of the emotion (typically 0-1) */
  value: number;
  /** Rate of change of the emotion intensity */
  velocity: number;
}

/**
 * Collection of defined emotions for the system.
 * Each emotion includes keywords for classification and a low-saturation
 * color for visualization purposes.
 */
export const EMOTIONS: readonly Emotion[] = [
  {
    id: 'happy',
    keywords: ['happy'],
    colorRGB: [212, 188, 130], // Muted warm gold
  },
  {
    id: 'horny',
    keywords: ['horny'],
    colorRGB: [195, 130, 155], // Muted dusty rose
  },
  {
    id: 'angry',
    keywords: ['angry'],
    colorRGB: [185, 95, 90], // Muted brick red
  },
  {
    id: 'depressed',
    keywords: ['depressed'],
    colorRGB: [120, 135, 165], // Muted slate blue
  },
] as const;

/**
 * Map of emotion IDs to their corresponding Emotion objects.
 * Provides efficient lookup by emotion identifier.
 */
export const EMOTION_MAP: Readonly<Record<string, Emotion>> = Object.fromEntries(
  EMOTIONS.map((emotion) => [emotion.id, emotion])
);

/**
 * Array of all emotion IDs for easy iteration.
 */
export const EMOTION_IDS: readonly string[] = EMOTIONS.map((emotion) => emotion.id);
