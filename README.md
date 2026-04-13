# Zeitgeist

Real-time sentiment visualization engine. Ingests Bluesky Jetstream, classifies emotional signals via keyword detection, and renders the output as animated WebGL2 ASCII art.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-blue)]()
[![Node](https://img.shields.io/badge/node-%3E%3D18-brightgreen)]()

---

## Overview

Zeitgeist processes live text data, extracts emotional content through keyword matching, smooths the resulting signals, and visualizes the aggregate emotional state as a continuously animated WebGL2 scene. The rendering uses ASCII glyphs composited through a heart-shaped mask, with color blending performed in OKLab perceptual color space.

The project is intentionally framework-free on the frontend and minimal on dependencies — the only external package is `ws` for the backend WebSocket server.

## Architecture

```
Data Sources (Bluesky Jetstream, etc.)
        │
        ▼
┌─────────────────────────────┐
│     Backend (Node.js)       │
│  Firehose → EmotionDetector │
│  → SignalProcessor → WsServer│
└─────────────┬───────────────┘
              │ WebSocket (real-time)
              ▼
┌─────────────────────────────┐
│    Frontend (WebGL2)        │
│  WS Client → StateManager  │
│  → WebGL2 Renderer          │
└─────────────────────────────┘
```

### Backend

| Module | Responsibility |
|--------|---------------|
| `Firehose` | Ingests raw data from configured sources with retry logic |
| `EmotionDetector` | Matches keywords against a sliding 1s window, outputs emotion ratios |
| `SignalProcessor` | Two-stage smoothing: rolling median (window=5) + EMA (α=0.3) |
| `WsServer` | Broadcasts processed emotional state via WebSocket at 100ms intervals |

### Frontend

| Module | Responsibility |
|--------|---------------|
| `WebSocket Client` | Receives data, handles reconnection with exponential backoff (1s–30s) |
| `EmotionStateManager` | Client-side interpolation (lerp factor 0.05) and graceful degradation |
| `Renderer` | WebGL2 pipeline with ping-pong framebuffers for temporal feedback |
| `ASCIIMask` | Generates ASCII glyph textures via offscreen Canvas2D |

### Shared

- `emotions.ts` — Emotion definitions, colors, and keyword lists
- `types.ts` — TypeScript interfaces for the WebSocket protocol

## Emotional Model

Five emotional dimensions are tracked, each mapped to a color and keyword set:

| Emotion | Color | Keywords (sample) |
|---------|-------|-------------------|
| Serene | Muted teal `[135, 169, 158]` | calm, peaceful, tranquil, relaxed |
| Vibrant | Soft terracotta `[173, 147, 135]` | energetic, lively, dynamic, active |
| Melancholy | Slate blue `[145, 155, 175]` | sad, somber, pensive, reflective |
| Curious | Warm gray `[165, 160, 145]` | interested, inquisitive, wondering |
| Content | Muted olive `[155, 165, 145]` | satisfied, happy, pleased, grateful |

Emotions coexist as weighted ratios (e.g., 60% serene / 30% content / 10% curious) and are blended in OKLab color space for perceptually natural transitions.

## Rendering Pipeline

1. **Signal processing** — Rolling median removes spikes; EMA adds smoothing
2. **Client interpolation** — Lerp-based smoothing for frame-rate-independent transitions
3. **Color blending** — sRGB → linear → OKLab → blend → linear → sRGB
4. **Noise distortion** — 2-octave simplex noise scaled by emotional velocity (0.5–2.0 scale, max 3% distortion)
5. **Temporal feedback** — Ping-pong framebuffers blend current frame with previous, creating trails proportional to signal velocity (capped at 0.95)
6. **ASCII mask** — Heart-shaped glyph grid rendered to texture, sampled in the fragment shader

### Degradation Mode

On WebSocket disconnect (after 2s timeout), the system gracefully fades rather than freezing:
- Velocity decays at 0.95/frame (movement stops first)
- Values decay at 0.98/frame (color fades to neutral)

## Tech Stack

- **Runtime:** Node.js ≥ 18
- **Language:** TypeScript (end-to-end, shared types)
- **Backend:** `ws` WebSocket server (sole external dependency)
- **Frontend:** WebGL2, Canvas2D — no frameworks, no bundler
- **Build:** TypeScript compiler, npm workspaces (monorepo)
- **Color science:** OKLab perceptual color space
- **Graphics:** GLSL fragment shaders, ping-pong FBOs, 2D simplex noise

## Getting Started

### Prerequisites

- Node.js ≥ 18.0.0
- A browser with WebGL2 support

### Install & Build

```bash
git clone https://github.com/revelri/zeitgeist.git
cd zeitgeist
npm install
npm run build
```

### Run

```bash
# Terminal 1: Backend (WebSocket server on :8080)
cd backend && npm start

# Terminal 2: Frontend (static server on :8080)
cd frontend && npm run serve
```

Open `http://localhost:8080` in your browser.

### Development

```bash
# Watch mode
cd backend && npm run dev   # tsc --watch
cd frontend && npm run dev  # tsc --watch + serve
```

## Project Structure

```
zeitgeist/
├── backend/
│   ├── index.ts              # Entry point, orchestration
│   ├── firehose.ts           # Data source ingestion
│   ├── emotionDetector.ts    # Keyword matching, ratio calculation
│   ├── signalProcessor.ts    # Rolling median + EMA smoothing
│   └── wsServer.ts           # WebSocket broadcast server
├── frontend/
│   ├── index.html            # HTML entry
│   ├── main.ts               # Client entry, render loop
│   ├── emotionStateManager.ts # Interpolation + degradation
│   └── gl/
│       ├── renderer.ts       # WebGL2 renderer
│       ├── shaders.ts        # GLSL shaders (simplex noise, OKLab)
│       └── asciiMask.ts      # ASCII texture generation
├── shared/
│   ├── emotions.ts           # Emotion definitions
│   └── types.ts              # Shared interfaces
├── package.json              # Workspace root
└── tsconfig.json             # Root TS config
```

## Design Rationale

| Decision | Reasoning |
|----------|-----------|
| OKLab color space | Perceptually uniform blending — mixed emotions produce natural intermediate colors instead of muddy RGB artifacts |
| Temporal feedback | Emotional states persist and trail, reflecting how feelings linger rather than switching instantaneously |
| ASCII rendering | Adds texture and a symbolic layer to the visualization; the glyph grid creates visual rhythm |
| No frameworks | Full control over the WebGL2 pipeline; minimal dependency surface; small bundle |
| Two-stage smoothing | Median removes outlier spikes; EMA adds fluid continuity without sacrificing responsiveness |

## License

[MIT](LICENSE)
