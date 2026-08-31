/**
 * Content Loader - File-based Content Management
 *
 * This module handles loading ASCII art and color configuration from the
 * backend content folder. It watches for file changes and emits events
 * when content is updated.
 *
 * Content folder structure:
 *   content/
 *     art.txt          - ASCII art file (or any .txt file)
 *     colors.txt       - Emotion color configuration (emotion=hexcode pairs)
 *
 * Colors file format:
 *   # Comments start with #
 *   happy=#d4bc82
 *   horny=#c3829b
 *   angry=#b95f5a
 */

import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as fsp from 'fs/promises';
import * as path from 'path';

/**
 * Parsed color configuration
 */
export interface ColorConfig {
  [emotionId: string]: {
    hex: string;
    rgb: [number, number, number];
  };
}

/**
 * Content state broadcast to frontend
 */
export interface ContentState {
  asciiArt: string | null;
  colors: ColorConfig;
  artFileName: string | null;
}

export interface ContentLoaderConfig {
  contentDir: string;
  colorsFile?: string;  // defaults to 'colors.txt'
  watchInterval?: number;  // defaults to 1000ms
}

export class ContentLoader extends EventEmitter {
  private config: ContentLoaderConfig;
  private contentDir: string;
  private colorsFile: string;
  private isRunning: boolean = false;
  private watchInterval: NodeJS.Timeout | null = null;
  private lastArtMtime: number = 0;
  private lastColorsMtime: number = 0;
  private currentArtFile: string | null = null;

  private currentState: ContentState = {
    asciiArt: null,
    colors: {},
    artFileName: null,
  };

  constructor(config: ContentLoaderConfig) {
    super();
    this.config = config;
    this.contentDir = path.resolve(config.contentDir);
    this.colorsFile = config.colorsFile || 'colors.txt';
  }

  /**
   * Start the content loader and begin watching for changes
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error('Content loader is already running');
    }

    this.isRunning = true;

    // Ensure content directory exists
    await fsp.mkdir(this.contentDir, { recursive: true });

    // Initial load
    await this.loadContent();

    // Start watching for changes
    const interval = this.config.watchInterval || 1000;
    this.watchInterval = setInterval(() => this.checkForChanges(), interval);

    console.log(`[ContentLoader] Started watching: ${this.contentDir}`);
    this.emit('started');
  }

  /**
   * Stop the content loader
   */
  async stop(): Promise<void> {
    if (!this.isRunning) {
      return;
    }

    this.isRunning = false;

    if (this.watchInterval) {
      clearInterval(this.watchInterval);
      this.watchInterval = null;
    }

    console.log('[ContentLoader] Stopped');
    this.emit('stopped');
  }

  /**
   * Load all content from the content directory
   */
  private async loadContent(): Promise<void> {
    await this.loadColors();
    await this.loadAsciiArt();
  }

  /**
   * Check for file changes and reload if necessary
   */
  private async checkForChanges(): Promise<void> {
    let changed = false;

    // Check colors file
    const colorsPath = path.join(this.contentDir, this.colorsFile);
    try {
      const stat = await fsp.stat(colorsPath);
      if (stat.mtimeMs > this.lastColorsMtime) {
        await this.loadColors();
        changed = true;
      }
    } catch {
      // Colors file doesn't exist, skip
    }

    // Check for ASCII art files
    const artFile = await this.findArtFile();
    if (artFile) {
      const artPath = path.join(this.contentDir, artFile);
      const stat = await fsp.stat(artPath);
      if (artFile !== this.currentArtFile || stat.mtimeMs > this.lastArtMtime) {
        await this.loadAsciiArt();
        changed = true;
      }
    }

    if (changed) {
      this.emit('content', this.currentState);
    }
  }

  /**
   * Find the first ASCII art file in the content directory
   * Prioritizes 'art.txt', then any other .txt file (excluding colors.txt)
   */
  private async findArtFile(): Promise<string | null> {
    try {
      await fsp.access(this.contentDir);
    } catch {
      return null;
    }

    const files = await fsp.readdir(this.contentDir);

    // First look for art.txt
    if (files.includes('art.txt')) {
      return 'art.txt';
    }

    // Then look for any .txt file that isn't colors.txt
    for (const file of files) {
      if (file.endsWith('.txt') && file !== this.colorsFile) {
        return file;
      }
    }

    return null;
  }

  /**
   * Load and parse the colors configuration file
   */
  private async loadColors(): Promise<void> {
    const colorsPath = path.join(this.contentDir, this.colorsFile);

    try {
      await fsp.access(colorsPath);
    } catch {
      console.log(`[ContentLoader] Colors file not found: ${colorsPath}`);
      return;
    }

    try {
      const content = await fsp.readFile(colorsPath, 'utf-8');
      const stat = await fsp.stat(colorsPath);
      this.lastColorsMtime = stat.mtimeMs;

      const colors = this.parseColorsFile(content);
      this.currentState.colors = colors;

      console.log(`[ContentLoader] Loaded colors: ${Object.keys(colors).length} emotions`);
      this.emit('colors', colors);
    } catch (error) {
      console.error('[ContentLoader] Error loading colors:', error);
    }
  }

  /**
   * Parse the colors file content into a ColorConfig object
   * Format: emotion=hexcode (one per line, # for comments)
   */
  private parseColorsFile(content: string): ColorConfig {
    const colors: ColorConfig = {};
    const lines = content.split('\n');

    for (const line of lines) {
      const trimmed = line.trim();

      // Skip empty lines and comments
      if (!trimmed || trimmed.startsWith('#')) {
        continue;
      }

      // Parse emotion=hexcode
      const match = trimmed.match(/^(\w+)\s*=\s*#?([0-9a-fA-F]{6})$/);
      if (match) {
        const [, emotionId, hex] = match;
        const rgb = this.hexToRgb(hex);
        colors[emotionId.toLowerCase()] = {
          hex: `#${hex.toLowerCase()}`,
          rgb,
        };
      }
    }

    return colors;
  }

  /**
   * Convert hex color to RGB tuple
   */
  private hexToRgb(hex: string): [number, number, number] {
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    return [r, g, b];
  }

  /**
   * Load ASCII art from the content directory
   */
  private async loadAsciiArt(): Promise<void> {
    const artFile = await this.findArtFile();

    if (!artFile) {
      console.log('[ContentLoader] No ASCII art file found in content directory');
      this.currentState.asciiArt = null;
      this.currentState.artFileName = null;
      return;
    }

    const artPath = path.join(this.contentDir, artFile);

    try {
      const content = await fsp.readFile(artPath, 'utf-8');
      const stat = await fsp.stat(artPath);
      this.lastArtMtime = stat.mtimeMs;
      this.currentArtFile = artFile;

      this.currentState.asciiArt = content;
      this.currentState.artFileName = artFile;

      console.log(`[ContentLoader] Loaded ASCII art: ${artFile}`);
      this.emit('ascii', content, artFile);
    } catch (error) {
      console.error('[ContentLoader] Error loading ASCII art:', error);
    }
  }

  /**
   * Update colors, write to disk, and emit content event
   */
  async updateColors(colors: Record<string, string>): Promise<void> {
    // Build colors.txt content
    const lines = ['# Emotion color configuration', '# Format: emotion=hexcode', ''];
    for (const [emotion, hex] of Object.entries(colors)) {
      const normalized = hex.startsWith('#') ? hex : `#${hex}`;
      lines.push(`${emotion}=${normalized}`);
    }
    const colorsPath = path.join(this.contentDir, this.colorsFile);
    await fsp.writeFile(colorsPath, lines.join('\n') + '\n', 'utf-8');
    await this.loadColors();
    this.emit('content', this.currentState);
  }

  /**
   * Get the current content state
   */
  getState(): ContentState {
    return { ...this.currentState };
  }

  /**
   * Get status for diagnostics
   */
  getStatus(): {
    isRunning: boolean;
    contentDir: string;
    artFile: string | null;
    emotionCount: number;
  } {
    return {
      isRunning: this.isRunning,
      contentDir: this.contentDir,
      artFile: this.currentState.artFileName,
      emotionCount: Object.keys(this.currentState.colors).length,
    };
  }
}
