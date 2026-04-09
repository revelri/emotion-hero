/**
 * Main entry point for the Emotion Hero frontend application.
 *
 * This file initializes the WebGL2 renderer and establishes a WebSocket connection
 * to the backend for real-time emotional data visualization.
 */

// Import the WebGL2 renderer and ASCII mask generator
import { Renderer } from './gl/renderer.js';
import { ASCIIMask } from './gl/asciiMask.js';
import { EmotionStateManager } from './emotionStateManager.js';
import { EMOTIONS, Emotion } from '../shared/emotions.js';
import { SHADER_MODE_NAMES } from './gl/shaders.js';

// DOM elements
let canvas: HTMLCanvasElement;
let renderer: Renderer;
let ws: WebSocket | null = null;
let emotionStateManager: EmotionStateManager;
let asciiMask: ASCIIMask | null = null;

// ASCII art management
let currentASCIIMaskTexture: WebGLTexture | null = null;

// Dynamic color configuration from backend
let dynamicColors: Map<string, [number, number, number]> = new Map();

// Default heart ASCII art (used if backend hasn't sent content yet)
const defaultASCIIMask = `      ******       ******
   **********   **********
  ************ ************
 ***************************
  *************************
   ***********************
     *******************
       ***************
         ***********
           *******
             ***
              *`;

// WebSocket connection state tracking
let isIntentionalClose = false;
let reconnectionAttempts = 0;
let reconnectionTimeout: number | null = null;

// Track time for continuous animation
let startTime = performance.now();
let lastFrameTime = performance.now();

// Resize debounce timer
let resizeTimeout: number | null = null;

// Reduced motion preference
let prefersReducedMotion = false;

/**
 * Initialize the application
 */
function init(): void {
    // Detect reduced motion preference
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    prefersReducedMotion = motionQuery.matches;
    motionQuery.addEventListener('change', (e) => {
        prefersReducedMotion = e.matches;
    });

    // Get the canvas element
    canvas = document.getElementById('canvas') as HTMLCanvasElement;

    if (!canvas) {
        throw new Error('Canvas element not found');
    }

    // Initialize the WebGL2 renderer
    renderer = new Renderer();

    // Initialize WebGL2 context
    if (!renderer.init(canvas)) {
        // Show WebGL2 fallback message
        const fallback = document.getElementById('webgl-fallback');
        if (fallback) {
            fallback.classList.add('visible');
        }
        dismissLoadingOverlay();
        return;
    }
    
    // Set initial canvas size
    renderer.resize(window.innerWidth, window.innerHeight);
    
    // Set initial color (gray for testing)
    renderer.setColor(0.5, 0.5, 0.5);
    
    // Initialize ASCII mask
    // Artistic Intent: ASCII art transforms raw pixels into a symbolic language of emotion.
    // The heart shape represents the core of human feeling - love, passion, and vulnerability.
    // By rendering emotions through this mask, we create a bridge between digital signals
    // and human emotional expression, making the invisible visible through the familiar
    // aesthetic of terminal art. The monospace characters evoke nostalgia while the heart
    // provides an immediately recognizable emotional anchor.
    asciiMask = new ASCIIMask();
    const gl = canvas.getContext('webgl2');
    if (gl) {
        asciiMask.init(gl);
        
        // Initialize with default ASCII art
        updateASCIIMask(defaultASCIIMask);
    }
    
    // Set up UI event listeners
    setupStatusUI();
    
    // Initialize the emotion state manager with interpolation
    emotionStateManager = new EmotionStateManager({
        maxPacketAge: 1000,
        enableInterpolation: true,
        springFrequency: 2.0,
        springDamping: 0.95
    });
    
    // Set up shader mode switching (keys 1-9, 0, q-t, y, u, i)
    setupShaderModeKeys();
    updateShaderModeDisplay(0);

    // Dismiss loading overlay once renderer is ready
    dismissLoadingOverlay();

    // Set up WebSocket connection to the backend
    connectWebSocket();

    // Start the render loop
    requestAnimationFrame(render);
}

/**
 * Dismiss the loading overlay with a fade transition
 */
function dismissLoadingOverlay(): void {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.add('hidden');
        overlay.addEventListener('transitionend', () => {
            overlay.remove();
        }, { once: true });
        if (prefersReducedMotion) {
            overlay.remove();
        }
    }
}

/**
 * Establish WebSocket connection to the backend
 */
function connectWebSocket(): void {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.hostname || 'localhost';
    const wsPort = '8080';
    const wsUrl = `${wsProtocol}//${wsHost}:${wsPort}`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        // Reset degradation state on successful connection
        emotionStateManager.resetDegradation();
        // Reset reconnection attempts on successful connection
        reconnectionAttempts = 0;
        // Update status display
        updateConnectionStatus('connected');
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleMessage(data);
        } catch {
            // Silently drop malformed messages from the backend
        }
    };
    
    ws.onerror = (error) => {
        // Update status display
        updateConnectionStatus('disconnected');
        // Trigger degradation by not resetting state
        // Attempt reconnection
        attemptReconnection();
    };

    ws.onclose = () => {
        // Update status display
        updateConnectionStatus('disconnected');
        // Trigger degradation by not resetting state (emotionStateManager's timeout will handle it)
        // Attempt reconnection if not intentionally closed
        attemptReconnection();
    };
}

/**
 * Attempt to reconnect to the WebSocket server with exponential backoff.
 */
function attemptReconnection(): void {
    // Don't attempt reconnection if this was an intentional close
    if (isIntentionalClose) {
        return;
    }
    
    // Clear any existing reconnection timeout
    if (reconnectionTimeout !== null) {
        clearTimeout(reconnectionTimeout);
    }
    
    // Calculate backoff delay: start at 1000ms, double each attempt, max 30000ms
    const backoffDelay = Math.min(1000 * Math.pow(2, reconnectionAttempts), 30000);
    
    reconnectionTimeout = window.setTimeout(() => {
        reconnectionAttempts++;
        updateConnectionStatus('connecting');
        connectWebSocket();
    }, backoffDelay);
}

/**
 * Handle incoming WebSocket messages from the backend
 * Routes to appropriate handler based on message type
 * @param data - The message data received from the backend
 */
function handleMessage(data: any): void {
    const messageType = data.type as string | undefined;

    switch (messageType) {
        case 'emotions':
            handleEmotionData(data);
            break;
        case 'content':
            handleContentData(data);
            break;
        case 'settings':
            handleSettingsData(data);
            break;
        default:
            // Legacy support: messages without type are treated as emotion data
            if (data.emotions) {
                handleEmotionData(data);
            }
            break;
    }
}

function handleSettingsData(data: any): void {
    if (data.settings) {
        const { shaderMode, reducedMotion } = data.settings;
        if (typeof shaderMode === 'number' && shaderMode >= 0 && shaderMode < SHADER_MODE_NAMES.length) {
            renderer.setShaderMode(shaderMode);
            updateShaderModeDisplay(shaderMode);
            showStatus(`Shader: ${SHADER_MODE_NAMES[shaderMode]} (remote)`, 'success');
        }
        if (typeof reducedMotion === 'boolean') {
            prefersReducedMotion = reducedMotion;
        }
    }
}

/**
 * Handle incoming emotional data from the backend
 * @param data - The emotional data received from the backend
 */
function handleEmotionData(data: any): void {
    // Parse the emotion data message
    const emotions = data.emotions as Record<string, { value: number; velocity: number }> | undefined;
    const timestamp = data.timestamp as number | undefined;

    if (!emotions) {
        return;
    }

    if (timestamp === undefined) {
        return;
    }

    // Update the emotion state manager with backend data
    // This sets target values; interpolation will be applied in render loop
    emotionStateManager.updateFromBackend(emotions, timestamp);
}

/**
 * Handle incoming content data from the backend (ASCII art and colors)
 * @param data - The content data received from the backend
 */
function handleContentData(data: any): void {
    // Update ASCII art if provided
    if (data.asciiArt && typeof data.asciiArt === 'string') {
        updateASCIIMask(data.asciiArt);
        updateArtStatus(data.artFileName || 'loaded');
        showStatus(`Loaded: ${data.artFileName || 'ASCII art'}`, 'success');
    } else {
        updateArtStatus(null);
    }

    // Update colors if provided
    if (data.colors && typeof data.colors === 'object') {
        dynamicColors.clear();
        for (const [emotionId, colorData] of Object.entries(data.colors)) {
            const color = colorData as { hex: string; rgb: [number, number, number] };
            if (color.rgb && Array.isArray(color.rgb)) {
                dynamicColors.set(emotionId, color.rgb);
            }
        }
        updateColorsStatus(dynamicColors.size);
        console.log(`[Content] Updated colors for ${dynamicColors.size} emotions`);
    }
}

/**
 * Main render loop — capped at 30fps to reduce GPU load.
 * Full-rate rAF is unnecessary for smoothly-interpolated generative visuals.
 */
const TARGET_FPS = 30;
const FRAME_INTERVAL = 1000 / TARGET_FPS;
let lastRenderTime = 0;

function render(timestamp: number): void {
    requestAnimationFrame(render);

    // Skip frame if we haven't hit the interval yet
    const elapsed = timestamp - lastRenderTime;
    if (elapsed < FRAME_INTERVAL) return;
    lastRenderTime = timestamp - (elapsed % FRAME_INTERVAL);

    // Calculate current time in seconds for continuous animation
    const currentTime = (performance.now() - startTime) / 1000.0;

    // Calculate deltaTime for interpolation (in seconds)
    const now = performance.now();
    const deltaTime = (now - lastFrameTime) / 1000.0;
    lastFrameTime = now;

    // Perform interpolation step
    emotionStateManager.interpolate(deltaTime);

    // Get current interpolated states
    const currentStates = emotionStateManager.getCurrentStates();

    // Extract colors, values, and velocities for all 5 emotions in consistent order
    const colors: [number, number, number][] = [];
    const values: number[] = [];
    const velocities: number[] = [];

    for (const emotion of EMOTIONS) {
        const emotionState = currentStates[emotion.id];

        // Use dynamic color from backend if available, otherwise fall back to default
        const dynamicColor = dynamicColors.get(emotion.id);
        colors.push(dynamicColor ?? emotion.colorRGB);

        // Use interpolated value or default to 0
        values.push(emotionState?.value ?? 0);

        // Use calculated velocity from interpolation or default to 0
        velocities.push(emotionState?.velocity ?? 0);
    }

    // Update the renderer with the interpolated emotional data
    renderer.setEmotionData(colors, values, velocities);

    // Update legend with current percentages
    updateLegend(values);

    // Respect reduced motion preference
    renderer.setReducedMotion(prefersReducedMotion);

    // Set the time for the simplex noise flow integration
    // In reduced motion mode, slow the time progression
    renderer.setTime(prefersReducedMotion ? currentTime * 0.1 : currentTime);

    // Render one frame
    renderer.render();
}

/**
 * Update the legend with current emotion percentages
 */
let lastLegendUpdate = 0;
function updateLegend(values: number[]): void {
    // Throttle updates to 10fps to avoid DOM thrashing
    const now = performance.now();
    if (now - lastLegendUpdate < 100) return;
    lastLegendUpdate = now;

    const ids = ['serene', 'vibrant', 'melancholy', 'curious', 'content'];
    for (let i = 0; i < ids.length; i++) {
        const el = document.getElementById('val-' + ids[i]);
        if (el) {
            const pct = Math.round((values[i] || 0) * 100);
            el.textContent = pct + '%';
        }
    }
}

/**
 * Handle window resize events
 */
function handleResize(): void {
    if (resizeTimeout !== null) {
        clearTimeout(resizeTimeout);
    }
    resizeTimeout = window.setTimeout(() => {
        renderer.resize(window.innerWidth, window.innerHeight);
        resizeTimeout = null;
    }, 100);
}

/**
 * Update the ASCII mask with new art
 * @param asciiArt - The ASCII art string to use as the mask
 */
function updateASCIIMask(asciiArt: string): void {
    // Validate input
    if (!asciiArt || asciiArt.trim().length === 0) {
        showStatus('ASCII art cannot be empty', 'error');
        return;
    }
    
    // Check for reasonable dimensions (not too many lines or too wide)
    const lines = asciiArt.trim().split('\n');
    if (lines.length > 200 || lines.some(line => line.length > 300)) {
        showStatus('ASCII art dimensions too large (max 200 lines x 300 chars)', 'error');
        return;
    }
    
    const gl = canvas.getContext('webgl2');
    if (!gl || !asciiMask) {
        showStatus('WebGL or ASCII mask not initialized', 'error');
        return;
    }
    
    // Clean up previous texture if exists
    if (currentASCIIMaskTexture) {
        gl.deleteTexture(currentASCIIMaskTexture);
    }
    
    // Generate new mask texture (512x512 resolution for crisp rendering)
    const maskTexture = asciiMask.generateMask(asciiArt, 512, 512);
    
    // Pass the texture to the renderer
    if (maskTexture) {
        currentASCIIMaskTexture = maskTexture;
        renderer.setASCIIMask(maskTexture);
        showStatus('ASCII art updated successfully', 'success');
    } else {
        showStatus('Failed to generate ASCII mask', 'error');
    }
}

/**
 * Set up UI event listeners for status panel
 */
function setupStatusUI(): void {
    const statusPanel = document.getElementById('status-panel') as HTMLDivElement;
    const panelToggle = document.getElementById('panel-toggle') as HTMLButtonElement;

    if (!statusPanel || !panelToggle) {
        console.error('Status panel elements not found');
        return;
    }

    // Toggle panel collapse/expand
    panelToggle.addEventListener('click', () => {
        const isCollapsed = statusPanel.classList.toggle('collapsed');
        panelToggle.innerHTML = isCollapsed ? '&#x25B6;' : '&#x25BC;';
        panelToggle.setAttribute('aria-expanded', String(!isCollapsed));
    });
}

/**
 * Update connection status display
 */
function updateConnectionStatus(status: 'connected' | 'disconnected' | 'connecting'): void {
    const element = document.getElementById('connection-status');
    if (!element) return;

    element.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    element.className = 'status-value ' + status;
}

/**
 * Update art status display
 */
function updateArtStatus(fileName: string | null): void {
    const element = document.getElementById('art-status');
    if (!element) return;

    element.textContent = fileName || '-';
}

/**
 * Update colors status display
 */
function updateColorsStatus(count: number): void {
    const element = document.getElementById('colors-status');
    if (!element) return;

    element.textContent = count > 0 ? `${count} emotions` : '-';
}

/**
 * Show status message in the ASCII panel
 * @param message - The message to display
 * @param type - Either 'success' or 'error'
 */
function showStatus(message: string, type: 'success' | 'error'): void {
    const statusElement = document.getElementById('status-message') as HTMLDivElement;
    if (!statusElement) return;
    
    statusElement.textContent = message;
    statusElement.className = type;
    
    // Clear the message after 3 seconds
    setTimeout(() => {
        statusElement.textContent = '';
        statusElement.className = '';
    }, 3000);
}

/**
 * Set up keyboard shortcuts for shader mode switching
 */
function setupShaderModeKeys(): void {
    const keyMap: Record<string, number> = {};
    for (let d = 1; d <= 9; d++) keyMap[String(d)] = d - 1;
    keyMap['0'] = 9;
    const letters = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i'];
    letters.forEach((k, idx) => { keyMap[k] = 10 + idx; });

    window.addEventListener('keydown', (e) => {
        const mode = keyMap[e.key.toLowerCase()];
        if (mode !== undefined && mode < SHADER_MODE_NAMES.length) {
            e.preventDefault();
            renderer.setShaderMode(mode);
            updateShaderModeDisplay(mode);
            showStatus(`Shader: ${SHADER_MODE_NAMES[mode]}`, 'success');
        }
    });
}

/**
 * Update the shader mode display in the status panel
 */
function updateShaderModeDisplay(mode: number): void {
    const el = document.getElementById('shader-mode');
    if (el) el.textContent = SHADER_MODE_NAMES[mode] || 'Unknown';
}

// Event listeners
window.addEventListener('load', init);
window.addEventListener('resize', handleResize);
window.addEventListener('beforeunload', () => {
    // Mark the close as intentional to prevent reconnection attempts
    isIntentionalClose = true;
    // Clear any pending reconnection timeout
    if (reconnectionTimeout !== null) {
        clearTimeout(reconnectionTimeout);
    }
    // Close the WebSocket connection cleanly
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
    }
});

// Export for potential external use
export { handleEmotionData };