# Zeitgeist — Roadmap

Real-time sentiment visualization engine. Ingests the Bluesky Jetstream, classifies emotional signals via keyword detection, smooths the resulting signals, and renders the aggregate as animated WebGL2 ASCII art. Glyphs are composited through a heart-shaped mask; color blending is in OKLab perceptual color space. Intentionally framework-free frontend; only external dependency is `ws` on the backend.

## Current status

Functional prototype — ingestion, classification, smoothing, and WebGL2 render pipeline all working end-to-end.

## Near-term

- Configurable keyword lexicon: swap or extend the emotion classifier without code changes
- Smoothing parameter tuning UI (decay rate, window size sliders)
- Performance profiling of the WebGL2 render loop under sustained load

## Medium-term

- Additional data sources beyond Bluesky (Mastodon, RSS feeds)
- Emotion taxonomy expansion: move from keyword matching to lightweight ML classifier
- Export mode: capture the animation as an MP4 or animated GIF

## Long-term

- Embeddable widget version for third-party sites
- Historical replay: scrub through archived emotional-state timelines
- Configurable composite mask shapes beyond the heart
