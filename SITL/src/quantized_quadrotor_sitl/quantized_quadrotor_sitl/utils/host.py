from __future__ import annotations

import os
import platform
import socket
from pathlib import Path


def _linux_cpu_model() -> str | None:
    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.exists():
        return None
    for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key == "model name" and value:
            return value
    return None


def _available_memory_bytes() -> int | None:
    sysconf_names = os.sysconf_names
    if "SC_AVPHYS_PAGES" in sysconf_names and "SC_PAGE_SIZE" in sysconf_names:
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError):
            pass

    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None
    for line in meminfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1]) * 1024
    return None


def runtime_host_snapshot(*, headless: bool) -> dict[str, object]:
    cpu_model = _linux_cpu_model() or platform.processor() or platform.machine()
    try:
        load_average_1m, load_average_5m, load_average_15m = os.getloadavg()
    except (AttributeError, OSError):
        load_average_1m, load_average_5m, load_average_15m = None, None, None

    return {
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "load_average_1m": load_average_1m,
        "load_average_5m": load_average_5m,
        "load_average_15m": load_average_15m,
        "available_memory_bytes": _available_memory_bytes(),
        "headless": bool(headless),
    }
