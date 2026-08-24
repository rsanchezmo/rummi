# Charts

Generated from committed data, not pasted:

```bash
python -m rummi.bench.bench_backends --compile --json docs/data/backends.json
python tools/capture_agents.py --suite standard-greedy --games 60
python tools/render_charts.py
```

SVG because GitHub inlines it, it scales, and it diffs as text. That rules out a
hover layer — GitHub strips scripts from SVG — so the interaction budget goes to
direct labels instead, and the README's tables serve as the table view.

## Colour

Colours come from the reference data-viz palette and were checked with its
validator in both modes rather than by eye:

| chart | palette | result |
|---|---|---|
| throughput | 6 categorical slots | passes lightness, chroma, CVD and normal-vision gates, light and dark |
| agents | 5-step ordinal blue ramp | passes monotonicity, step gaps and light-end contrast, light and dark |

The light-mode check raises a contrast warning on aqua, yellow and magenta, which
obliges visible labels rather than colour alone — so every series is both
direct-labelled and in the legend, and the tables stay in the README.

Each SVG carries both themes: light values on `:root`, dark behind
`prefers-color-scheme`. GitHub resolves the media query even through `<img>`, so
one file serves both. `tests/test_charts.py` asserts every variable is defined in
both — a colour defined in only one theme renders as text on its own background.

## Geometry

`tests/test_charts.py` also asserts nothing escapes the viewBox. This is not
paranoia: a label two pixels past the edge looks fine in a thumbnail and is
clipped in the browser, and macOS Quick Look crops SVG thumbnails to a square,
which makes eyeballing actively misleading.
