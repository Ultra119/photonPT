"""Parse Viewfinder diagnostics JSON + AMD GPU telemetry and append to compact logs.

Avoids reading multi-thousand-line diagnostics dumps directly by extracting only
per-pass GPU timings, FPS, CPU/sync gap, GPU power/clocks/activity, and VRAM into
a JSONL log and a Markdown comparison table.

Examples:
    python tests/record_bottleneck_run.py baseline diag.json 2.5
    python tests/record_bottleneck_run.py step1 diag.json --telemetry=tel.json --out-dir=out
    python tests/record_bottleneck_run.py probe diag.json 2.0 --no-save
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gpu_telemetry import AmdTelemetry, read_vram_windows_mb

# Compute subpasses already included in parent deferred/deferredN totals
SUBPASSES = {
    "deferred_a", "deferred_b", "deferred_c", "deferred_d", "deferred_e", "deferred_f", "deferred_g",
    "deferred1_a", "deferred1_b", "deferred1_c", "deferred1_d", "deferred1_e", "deferred1_f",
    "deferred2_a",
    "deferred3_a", "deferred3_b", "deferred3_c", "deferred3_d",
}


def parse_diag(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    r = d.get("render", {})
    w = r.get("window", {})
    fps = r.get("fps", 0)
    m = d.get("metrics", {})

    def ms(k):
        v = m.get(k)
        return (v.get("avg", 0.0) / 1e6) if isinstance(v, dict) else 0.0

    total_gpu_ms = sum(
        (v.get("avg", 0.0) / 1e6)
        for k, v in m.items()
        if isinstance(v, dict) and k not in SUBPASSES
    )
    comp_ms = sum(
        (v.get("avg", 0.0) / 1e6)
        for k, v in m.items()
        if isinstance(v, dict) and k.startswith("composite/")
    )
    shadow_ms = sum(
        (v.get("avg", 0.0) / 1e6)
        for k, v in m.items()
        if isinstance(v, dict) and (k.startswith("Shadow ") or k == "shadowcomp")
    )
    terrain_ms = sum(
        (v.get("avg", 0.0) / 1e6)
        for k, v in m.items()
        if isinstance(v, dict) and (k.startswith("Terrain ") or k == "Particles" or k == "Entities")
    )
    def_pre_ms = sum(ms(f"deferred_{c}") for c in "abcdefg")
    def3_comp_ms = sum(ms(f"deferred3_{c}") for c in "abcd")
    def3_ao_ms = max(0.0, ms("deferred/deferred3") - def3_comp_ms)

    leaf = [
        (k, v.get("avg", 0.0) / 1e6)
        for k, v in m.items()
        if isinstance(v, dict)
        and k not in ("deferred/deferred", "deferred/deferred1", "deferred/deferred2", "deferred/deferred3")
    ]
    leaf.sort(key=lambda x: -x[1])

    compile_errs = [e for e in d.get("errors", []) if e.get("compilerMessages")]
    return {
        "pack": d.get("shaderpack", {}).get("name", ""),
        "fps": fps,
        "wall_ms": (1000.0 / fps) if fps and fps > 0 else 0.0,
        "res": f"{w.get('width', 0)}x{w.get('height', 0)}",
        "focused": w.get("focused", False),
        "total_gpu_ms": total_gpu_ms,
        "shadow_ms": shadow_ms,
        "terrain_ms": terrain_ms,
        "def_pre_ms": def_pre_ms,
        "def1_a_ms": ms("deferred1_a"),
        "def1_b_ms": ms("deferred1_b"),
        "def1_c_ms": ms("deferred1_c"),
        "def1_d_ms": ms("deferred1_d"),
        "def1_e_ms": ms("deferred1_e"),
        "def1_f_ms": ms("deferred1_f"),
        "def1_ms": ms("deferred/deferred1"),
        "def2_a_ms": ms("deferred2_a"),
        "def3_filt_ms": def3_comp_ms,
        "def3_ao_ms": def3_ao_ms,
        "def4_ms": ms("deferred/deferred4"),
        "comp_ms": comp_ms,
        "final_ms": ms("final"),
        "top5": leaf[:5],
        "include_failures": len(d.get("shaderpack", {}).get("includeFailures", [])),
        "compile_errors": len(compile_errs),
    }


def write_markdown(jsonl_path, md_path, power_limit_w=250.0):
    rows = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    lines = [
        "# GPU Bottleneck & Power Sweep Results",
        "",
        f"GPU Power Limit: **{power_limit_w:.0f} W** | All pass times in **ms**",
        "",
        "| # | Label | FPS | Wall ms | GPU sum ms | Gap | Power W | %TBP | GFX MHz | GFX% | VRAM MB | def1 total | d1_a | d1_b | d1_c | d1_d | d1_e | def4 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows, 1):
        gap = r["wall_ms"] - r["total_gpu_ms"]
        pwr = r.get("power_w", 0.0)
        lines.append(
            f"| {i} | {r['label']} | {r['fps']} | {r['wall_ms']:.2f} | {r['total_gpu_ms']:.2f} | {gap:+.2f} "
            f"| **{pwr:.1f}** | {100.0 * pwr / power_limit_w:.0f}% | {r.get('gfx_mhz', 0):.0f} "
            f"| {r.get('gfx_busy', 0):.0f}% | {r.get('vram_mb', 0):.0f} "
            f"| {r.get('def1_ms', 0):.2f} | {r.get('def1_a_ms', 0):.2f} | {r.get('def1_b_ms', 0):.2f} "
            f"| {r.get('def1_c_ms', 0):.2f} | {r.get('def1_d_ms', 0):.2f} | {r.get('def1_e_ms', 0):.2f} "
            f"| {r.get('def4_ms', 0):.2f} |"
        )
    lines.append("")
    lines.append("## Top-5 Slowest Leaf Passes per Run")
    lines.append("")
    for i, r in enumerate(rows, 1):
        t5 = ", ".join(f"`{k}`={v:.2f}ms" for k, v in r.get("top5", []))
        lines.append(
            f"- **{i}. {r['label']}** ({r['fps']} FPS, {r.get('power_w', 0):.1f}W, "
            f"VRAM {r.get('vram_mb', 0):.0f}MB): {t5}"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: record_bottleneck_run.py <label> <diag_json_path> [telemetry_seconds] "
            "[--telemetry=<json>] [--out-dir=<dir>] [--power-limit=<W>] [--no-save]"
        )
        sys.exit(1)

    label = sys.argv[1]
    diag_path = sys.argv[2]
    dur = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 3.0
    no_save = "--no-save" in sys.argv
    tel_json = None
    out_dir = HERE
    power_limit = 250.0
    for arg in sys.argv[3:]:
        if arg.startswith("--telemetry="):
            tel_json = arg.split("=", 1)[1]
        elif arg.startswith("--out-dir="):
            out_dir = arg.split("=", 1)[1]
        elif arg.startswith("--power-limit="):
            power_limit = float(arg.split("=", 1)[1])

    info = parse_diag(diag_path)
    if tel_json:
        with open(tel_json, "r", encoding="utf-8") as f:
            info.update(json.load(f))
    elif dur > 0:
        telem = AmdTelemetry()
        try:
            s = telem.sample_window(seconds=dur, interval=0.1)
        finally:
            telem.close()
        info["power_w"] = s.get("ASIC_POWER", {}).get("avg", 0.0)
        info["power_max_w"] = s.get("ASIC_POWER", {}).get("max", 0.0)
        info["gfx_mhz"] = s.get("CLK_GFXCLK", {}).get("avg", 0.0)
        info["mem_mhz"] = s.get("CLK_MEMCLK", {}).get("avg", 0.0)
        info["gfx_busy"] = s.get("INFO_ACTIVITY_GFX", {}).get("avg", 0.0)
        info["mem_busy"] = s.get("INFO_ACTIVITY_MEM", {}).get("avg", 0.0)
        info["vram_mb"] = s.get("VRAM_MB", {}).get("avg", read_vram_windows_mb())
        info["hotspot_c"] = s.get("TEMPERATURE_HOTSPOT", {}).get("avg", 0.0)
    else:
        info["vram_mb"] = read_vram_windows_mb()

    info["label"] = label
    info["timestamp"] = time.strftime("%H:%M:%S")

    if not no_save:
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, "bottleneck_results.jsonl")
        md_path = os.path.join(out_dir, "bottleneck_results.md")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(info) + "\n")
        write_markdown(jsonl_path, md_path, power_limit_w=power_limit)

    gap = info["wall_ms"] - info["total_gpu_ms"]
    print(
        f"[{label}] FPS={info['fps']} ({info['res']}, focused={info['focused']}) | "
        f"Wall={info['wall_ms']:.2f}ms GPU={info['total_gpu_ms']:.2f}ms Gap={gap:+.2f}ms | "
        f"Power={info.get('power_w', 0):.1f}W (max {info.get('power_max_w', 0):.0f}W) "
        f"GFX={info.get('gfx_mhz', 0):.0f}MHz ({info.get('gfx_busy', 0):.0f}%) "
        f"VRAM={info.get('vram_mb', 0):.0f}MB"
    )
    print(
        f"  def1={info['def1_ms']:.2f} (a={info['def1_a_ms']:.2f} b={info['def1_b_ms']:.2f} "
        f"c={info['def1_c_ms']:.2f} d={info['def1_d_ms']:.2f} e={info['def1_e_ms']:.2f} "
        f"f={info['def1_f_ms']:.2f}) | def4={info['def4_ms']:.2f} Comp={info['comp_ms']:.2f} "
        f"| errs={info['compile_errors']}/{info['include_failures']}"
    )


if __name__ == "__main__":
    main()
