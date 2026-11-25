# Runtime GPU Bottleneck & Telemetry Scripts

Utilities for measuring live GPU power draw, clocks, VRAM growth across shader reloads, and Viewfinder GPU pass timings without manually reading multi-thousand-line diagnostics dumps.

## Scripts

- [`gpu_telemetry.py`](gpu_telemetry.py): Polls AMD `atiadlxx.dll` (ADL2) plus the Windows GPU dedicated memory counter for live ASIC power (W), GFX/MEM clocks (MHz), GFX/MEM activity (%), hotspot temperature, and dedicated VRAM (MB).
  ```sh
  python tests/gpu_telemetry.py 3.0
  python tests/gpu_telemetry.py 2.5 --out=telemetry.json
  ```
- [`record_bottleneck_run.py`](record_bottleneck_run.py): Extracts per-pass GPU timings (`deferred1_a..f`, `deferred4`, `Shadow`, `Terrain`, `composite`), FPS, wall-vs-GPU gap, and live/saved telemetry from a Viewfinder `get_diagnostics` JSON dump and appends a compact row to `bottleneck_results.jsonl` + `bottleneck_results.md`.
  ```sh
  python tests/record_bottleneck_run.py baseline path/to/diagnostics.json 2.5 --out-dir=out
  python tests/record_bottleneck_run.py step1 path/to/diagnostics.json --telemetry=telemetry.json --out-dir=out
  ```

For full automated multi-pose MCP ablation sweeps, see [`tools/viewfinder_benchmark.py`](../tools/viewfinder_benchmark.py).
