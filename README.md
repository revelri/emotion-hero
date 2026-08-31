<div align="center">

# Zeitgeist

### A live, configurable view of keyword-level emotional signals

Bluesky posts in. Four project-defined signals out. WebGL2 turns their changing balance into an ASCII field.

[Quick start](#quick-start) · [How it works](#how-it-works) · [Controls and presets](#controls-and-presets) · [HTTP API](#http-api) · [Roadmap](ROADMAP.md)

[![CI](https://github.com/revoydotdev/zeitgeist/actions/workflows/ci.yml/badge.svg)](https://github.com/revoydotdev/zeitgeist/actions/workflows/ci.yml)
[![Node.js 18+](https://img.shields.io/badge/Node.js-18%2B-3c873a?logo=nodedotjs&logoColor=white)](https://nodejs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-3178c6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-4f46e5.svg)](LICENSE)

</div>

Zeitgeist is a small, end-to-end generative visualization system. Its backend
subscribes to the [Bluesky Jetstream](https://github.com/bluesky-social/jetstream)
post stream, counts configured keyword hits over short windows, smooths the
resulting ratios, and broadcasts them to a browser. The browser renders the
state as an animated, heart-shaped ASCII composition with WebGL2 shaders and
perceptual OKLab colour mixing.

It is deliberately a keyword matcher, not an affect-recognition system. The
four labels below are a configurable visual taxonomy for this project, and a
hit indicates only that a configured word appeared in a post.

## Quick start

### Requirements

- Node.js 18 or later (CI uses Node.js 22)
- A browser with WebGL2 support
- Network access to Jetstream for live input

```bash
git clone https://github.com/revoydotdev/zeitgeist.git
cd zeitgeist
npm ci
npm run build
npm start --workspace @zeitgeist/backend
```

Open [http://localhost:8081/](http://localhost:8081/) for the settings desk,
then open [http://localhost:8081/viz/](http://localhost:8081/viz/) for the live
visualization. The backend logs the selected port at startup. If 8081 is busy,
it tries the next ten ports; use that logged URL for both pages.

The backend serves HTTP and WebSocket traffic from the same port, so the
visualization does not need a separate static server. To choose a preferred
starting port or a Jetstream endpoint, set the environment before starting:

```bash
SETTINGS_PORT=8090 JETSTREAM_URL='wss://jetstream1.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post' \
  npm start --workspace @zeitgeist/backend
```

For development, rebuild after source changes, then start the backend again:

```bash
npm run build
npm start --workspace @zeitgeist/backend
```

Each workspace also offers a TypeScript watch command via `npm run dev
--workspace <workspace-name>`; the root `npm run dev` starts those watchers in
parallel.

## What you can explore

| Surface | What it provides |
| --- | --- |
| Live view | A 30 FPS WebGL2 ASCII composition driven by the smoothed stream, with a connection/status panel and graceful fading when the feed is unavailable. |
| Settings desk | A browser UI for source endpoint and retries, keyword lexicon, detector window, smoothing, colours, shader mode, reduced motion, and presets. |
| Shader library | 19 switchable modes, including Voronoi, curl noise, reaction-diffusion, smoke, magnetic LIC, ink flow, and the README Hero composition. |
| Content files | Reloadable ASCII art and a colour file under `backend/content/`; colour changes made through the API are written back to `colors.txt`. |
| Small protocol | Typed shared emotion definitions and WebSocket messages shared by the Node backend and browser client. |

## How it works

```text
Bluesky Jetstream posts
        |
        v
Firehose ──> EmotionDetector ──> SignalProcessor ──> WebSocket clients
                 |                     |                    |
          keyword hit ratios      adaptive smoothing     WebGL2 + ASCII mask
                 ^                                          |
                 |                                          v
        settings and presets <────── HTTP settings desk ── live visualization
```

1. `Firehose` consumes Bluesky post-create events and reconnects with exponential backoff.
2. `EmotionDetector` matches configured words case-insensitively at word boundaries, then emits normalized hit ratios for each aggregation window (1.5 seconds by default).
3. `SignalProcessor` applies an adaptive low-pass filter and emits values plus velocity for each emotion.
4. `WsServer` rate-limits broadcasts, sends an initial emotion/content state to each new client, and shares the HTTP server with the settings API.
5. The browser interpolates received state, applies dynamic content colours and ASCII art, and renders a selected GLSL mode. It reconnects with a 1–30 second backoff and fades the existing state when packets go stale.

### Signal vocabulary

The checked-in default lexicon has one word per signal. It can be changed at
runtime from the settings desk or API.

| ID | Default keyword | Base RGB palette |
| --- | --- | --- |
| `happy` | `happy` | `[212, 188, 130]` |
| `horny` | `horny` | `[195, 130, 155]` |
| `angry` | `angry` | `[185, 95, 90]` |
| `depressed` | `depressed` | `[120, 135, 165]` |

The base palette above lives in `shared/emotions.ts`. On startup, the
content loader can replace the display palette with the values in
`backend/content/colors.txt`; the settings desk shows and edits that live
content palette.

### Rendering notes

The renderer blends weighted emotion colours in OKLab rather than directly in
sRGB. It combines the result with glyph-mask sampling, motion-responsive noise,
and temporal feedback. Some modes use stateful simulations (reaction-diffusion,
smoke, topography, and ink flow); the renderer keeps their state in ping-pong
framebuffers. Respecting `prefers-reduced-motion` disables temporal feedback and
reduces distortion.

## Controls and presets

In the visualization, press `1`–`9`, `0`, then `Q`–`O` to choose the 19 shader
modes in order. The settings desk can also broadcast a selected mode and
reduced-motion preference to connected clients.

Four checked-in presets are available from the settings desk or `GET
/api/presets`:

| Preset | Intended character |
| --- | --- |
| Firehose Default | Voronoi with the shipping detector and smoothing defaults. |
| Curl Noise Tide | A medium-paced flow-field composition. |
| Ink Flow Slow | A deliberately slow lattice-Boltzmann ink treatment. |
| README Hero | Four fixed colour blobs with an opacity pulse and no feedback trail. |

Applying a preset updates only the sections it contains. Visualization, signal,
detector, keyword, and firehose changes are runtime configuration; colour
changes are persisted to `backend/content/colors.txt`.

## HTTP API

The following unauthenticated, same-service endpoints are useful for local
experimentation. `PUT` requests accept JSON and return `{ "ok": true }` on
success unless noted otherwise.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Settings desk. |
| `GET` | `/viz/` | Live visualization and its bundled browser assets. |
| `GET` | `/api/settings` | Current firehose, keywords, signal, detector, visualization metadata, and content colours. |
| `GET` | `/api/status` | Component status, client count, and process uptime. |
| `GET` | `/api/presets` | Available preset metadata. |
| `PUT` | `/api/settings/firehose` | Set `endpoint`, `retryInterval` (at least 100 ms), and/or `maxRetries`. |
| `PUT` | `/api/settings/keywords` | Map known emotion IDs to arrays of keyword strings. |
| `PUT` | `/api/settings/signal` | Set `minCutoff` (0.01–10) and/or `beta` (0–1). |
| `PUT` | `/api/settings/detector` | Set `windowDuration` (500–5000 ms). |
| `PUT` | `/api/settings/visualization` | Broadcast `shaderMode` (0–18), `feedbackStrength` (0–0.95), and/or `reducedMotion`. |
| `PUT` | `/api/settings/colors` | Map known emotion IDs to six-digit hex colours; writes `colors.txt`. |
| `PUT` | `/api/presets/apply` | Apply `{ "id": "readme-hero" }`; response includes applied sections. |

The WebSocket endpoint is the same origin and port as the HTTP service. Clients
receive typed `emotions`, `content`, and `settings` messages; an initial
emotion/content pair is sent when a client connects.

> **Local-use boundary:** the server listens on all interfaces and does not
> implement authentication. Run it only on a trusted local network or place it
> behind appropriate network controls before exposing it elsewhere.

## Project layout

```text
zeitgeist/
├── backend/                 # Jetstream ingestion, detector, smoothing, HTTP + WebSocket service
│   ├── content/             # ASCII art, live colour file, and JSON presets
│   ├── settingsApi.ts       # Settings UI and HTTP API
│   └── wsServer.ts          # Same-port WebSocket broadcasting
├── frontend/                # Framework-free WebGL2 client
│   └── gl/                  # Renderer, shader registry, and ASCII-mask generator
├── shared/                  # Emotion vocabulary and protocol types
├── ROADMAP.md               # Product direction and current status
└── vitest.config.ts         # Cross-workspace test configuration
```

## Development and verification

```bash
npm ci
npm run build        # builds shared, backend, then frontend
npm test             # Vitest suite across all workspaces
npm run type-check   # root TypeScript check
npm run clean        # removes workspace dist directories
```

The automated tests cover the shared vocabulary, keyword matching and update
behaviour, signal smoothing, content loading, settings validation, WebSocket
messages, state interpolation/degradation, and shader-registry contracts.

## Roadmap

The project is a functional prototype. Planned work includes broader input
sources, an extensible taxonomy, render-performance profiling, historical
replay, export, and additional mask shapes. See [ROADMAP.md](ROADMAP.md) for
the current sequence rather than treating this list as a release commitment.

## License

[MIT](LICENSE)
