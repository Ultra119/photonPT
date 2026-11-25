"""Live AMD GPU telemetry sampler via atiadlxx.dll (ADL2) + Windows VRAM perf counter.

Reads board/ASIC power (Watts), GFX/MEM clocks (MHz), GFX/MEM activity (%),
hotspot/edge temperatures (C), and dedicated VRAM usage (MB) on Windows AMD GPUs.

Examples:
    python tests/gpu_telemetry.py 3.0
    python tests/gpu_telemetry.py 2.5 --out=telemetry.json
"""

import ctypes
from ctypes import (
    CFUNCTYPE,
    Structure,
    byref,
    c_char,
    c_int,
    c_void_p,
    create_string_buffer,
    sizeof,
)
import json
import subprocess
import sys
import time

ADL_OK = 0
ADL_MAX_PATH = 256
ADL_PMLOG_MAX_SENSORS = 256

# ADL_PMLOG_SENSORS enum values (from AMD ADL SDK adl_defines.h)
SENSOR_NAMES = {
    1: "CLK_GFXCLK",
    2: "CLK_MEMCLK",
    3: "CLK_SOCCLK",
    8: "TEMPERATURE_EDGE",
    9: "TEMPERATURE_MEM",
    14: "FAN_RPM",
    15: "FAN_PERCENTAGE",
    17: "SOC_POWER",
    19: "INFO_ACTIVITY_GFX",
    20: "INFO_ACTIVITY_MEM",
    23: "ASIC_POWER",
    27: "TEMPERATURE_HOTSPOT",
    30: "GFX_POWER",
    44: "CLK_FCLK",
    73: "BOARD_POWER",
}


class AdapterInfo(Structure):
    _fields_ = [
        ("iSize", c_int),
        ("iAdapterIndex", c_int),
        ("strUDID", c_char * ADL_MAX_PATH),
        ("iBusNumber", c_int),
        ("iDeviceNumber", c_int),
        ("iFunctionNumber", c_int),
        ("iVendorID", c_int),
        ("strAdapterName", c_char * ADL_MAX_PATH),
        ("strDisplayName", c_char * ADL_MAX_PATH),
        ("iPresent", c_int),
        ("iExist", c_int),
        ("strDriverPath", c_char * ADL_MAX_PATH),
        ("strDriverPathExt", c_char * ADL_MAX_PATH),
        ("strPNPString", c_char * ADL_MAX_PATH),
        ("iOSDisplayIndex", c_int),
    ]


class ADLSingleSensorData(Structure):
    _fields_ = [("supported", c_int), ("value", c_int)]


class ADLPMLogDataOutput(Structure):
    _fields_ = [
        ("size", c_int),
        ("sensors", ADLSingleSensorData * ADL_PMLOG_MAX_SENSORS),
    ]


MALLOC_CB = CFUNCTYPE(c_void_p, c_int)
_allocs = []


@MALLOC_CB
def _adl_malloc(size):
    buf = create_string_buffer(size)
    _allocs.append(buf)
    return ctypes.addressof(buf)


class AmdTelemetry:
    def __init__(self):
        self.adl = ctypes.WinDLL("atiadlxx.dll")
        self.ctx = c_void_p()
        rc = self.adl.ADL2_Main_Control_Create(_adl_malloc, 1, byref(self.ctx))
        if rc != ADL_OK:
            raise RuntimeError(f"ADL2_Main_Control_Create failed: {rc}")
        num = c_int(0)
        self.adl.ADL2_Adapter_NumberOfAdapters_Get(self.ctx, byref(num))
        arr = (AdapterInfo * num.value)()
        self.adl.ADL2_Adapter_AdapterInfo_Get(self.ctx, arr, sizeof(arr))
        self.adapter_idx = None
        self.adapter_name = ""
        for adapter in arr:
            if adapter.iPresent and adapter.iVendorID == 1002:
                self.adapter_idx = adapter.iAdapterIndex
                self.adapter_name = adapter.strAdapterName.decode(errors="ignore")
                break
        if self.adapter_idx is None:
            self.adapter_idx = 0

    def read_once(self):
        out = ADLPMLogDataOutput()
        out.size = sizeof(out)
        rc = self.adl.ADL2_New_QueryPMLogData_Get(self.ctx, self.adapter_idx, byref(out))
        if rc != ADL_OK:
            raise RuntimeError(f"ADL2_New_QueryPMLogData_Get failed: {rc}")
        res = {}
        for idx, name in SENSOR_NAMES.items():
            sensor = out.sensors[idx]
            if sensor.supported:
                res[name] = sensor.value
        vram = c_int(-1)
        if hasattr(self.adl, "ADL2_Adapter_DedicatedVRAMUsage_Get"):
            if self.adl.ADL2_Adapter_DedicatedVRAMUsage_Get(self.ctx, self.adapter_idx, byref(vram)) == ADL_OK:
                res["VRAM_MB"] = vram.value
        return res

    def sample_window(self, seconds=3.0, interval=0.1):
        samples = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            samples.append(self.read_once())
            time.sleep(interval)
        keys = sorted({k for s in samples for k in s})
        summary = {}
        for k in keys:
            vals = [s[k] for s in samples if k in s]
            summary[k] = {
                "avg": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }
        return summary

    def close(self):
        try:
            self.adl.ADL2_Main_Control_Destroy(self.ctx)
        except Exception:
            pass


def read_vram_windows_mb():
    """Fallback/cross-check total GPU dedicated VRAM usage via Windows perf counter."""
    cmd = (
        "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue)"
        ".CounterSamples | Measure-Object -Property CookedValue -Maximum | "
        "Select-Object -ExpandProperty Maximum"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", cmd],
            text=True,
            timeout=5,
        ).strip()
        return float(out) / (1024 * 1024) if out else -1.0
    except Exception:
        return -1.0


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else 2.0
    out_json = None
    for arg in sys.argv[1:]:
        if arg.startswith("--out="):
            out_json = arg.split("=", 1)[1]

    telem = AmdTelemetry()
    try:
        s = telem.sample_window(seconds=dur, interval=0.1)
        vram_win = read_vram_windows_mb()
        if out_json:
            payload = {
                "power_w": s.get("ASIC_POWER", {}).get("avg", 0.0),
                "power_max_w": s.get("ASIC_POWER", {}).get("max", 0.0),
                "gfx_mhz": s.get("CLK_GFXCLK", {}).get("avg", 0.0),
                "mem_mhz": s.get("CLK_MEMCLK", {}).get("avg", 0.0),
                "gfx_busy": s.get("INFO_ACTIVITY_GFX", {}).get("avg", 0.0),
                "mem_busy": s.get("INFO_ACTIVITY_MEM", {}).get("avg", 0.0),
                "vram_mb": s.get("VRAM_MB", {}).get("avg", vram_win),
                "hotspot_c": s.get("TEMPERATURE_HOTSPOT", {}).get("avg", 0.0),
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            print(
                f"Telemetry saved: Power={payload['power_w']:.1f}W "
                f"GFX={payload['gfx_mhz']:.0f}MHz ({payload['gfx_busy']:.0f}%) "
                f"VRAM={payload['vram_mb']:.0f}MB"
            )
        else:
            print(f"Adapter[{telem.adapter_idx}]: {telem.adapter_name}")
            for k in (
                "BOARD_POWER",
                "ASIC_POWER",
                "GFX_POWER",
                "SOC_POWER",
                "CLK_GFXCLK",
                "CLK_MEMCLK",
                "CLK_SOCCLK",
                "CLK_FCLK",
                "INFO_ACTIVITY_GFX",
                "INFO_ACTIVITY_MEM",
                "TEMPERATURE_EDGE",
                "TEMPERATURE_HOTSPOT",
                "VRAM_MB",
            ):
                if k in s:
                    v = s[k]
                    print(f"  {k:22s} avg={v['avg']:7.1f}  min={v['min']:6.0f}  max={v['max']:6.0f}")
            print(f"  {'WIN_VRAM_MB':22s} {vram_win:7.1f}")
    finally:
        telem.close()


if __name__ == "__main__":
    main()
