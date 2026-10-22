"""What the Mac is doing: power source, battery, thermal pressure, input idleness.

Each probe is a plain function so tests can replace it. All of them are cheap and
read-only; none needs a permission prompt.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class Power:
    on_battery: bool
    percent: Optional[int]


def power() -> Power:
    fake = os.environ.get("MACK_FAKE_POWER")  # tests and demos: "ac" or "battery:24"
    if fake:
        kind, _, pct = fake.partition(":")
        return Power(kind == "battery", int(pct) if pct else None)
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return Power(False, None)
    m = re.search(r"(\d+)%", out)
    return Power("Battery Power" in out, int(m.group(1)) if m else None)


THERMAL_NOMINAL, THERMAL_FAIR, THERMAL_SERIOUS, THERMAL_CRITICAL = range(4)


def thermal_state() -> int:
    """NSProcessInfo.thermalState through the Objective-C runtime (no pyobjc needed)."""
    try:
        objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        send = objc.objc_msgSend
        send.restype = ctypes.c_void_p
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        info = send(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
        send_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(ctypes.cast(send, ctypes.c_void_p).value)
        return int(send_long(info, objc.sel_registerName(b"thermalState")))
    except Exception:
        return THERMAL_NOMINAL


MEMORY_NORMAL, MEMORY_WARN, MEMORY_CRITICAL = 1, 2, 4


def memory_pressure() -> int:
    """macOS memory pressure level: 1 normal, 2 warn, 4 critical (kern.memorystatus_vm_pressure_level).
    "warn" is common on a 16 GB Mac with a browser open; only "critical" means trouble."""
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip() or 1)
    except Exception:
        return MEMORY_NORMAL


def input_idle_seconds() -> float:
    """Seconds since the last keyboard or mouse event, from CoreGraphics."""
    try:
        cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        f = cg.CGEventSourceSecondsSinceLastEventType
        f.restype = ctypes.c_double
        f.argtypes = [ctypes.c_int, ctypes.c_uint32]
        return float(f(0, 0xFFFFFFFF))  # combined session state, any input event
    except Exception:
        return 0.0
