"""Draw the app icon and compile it to StemForge.icns."""

from __future__ import annotations

import math
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

SIZE = 1024


def _blend(bottom, top, alpha):
    return tuple(round(b + (t - b) * alpha) for b, t in zip(bottom, top))


def render() -> bytearray:
    """A rounded slab with four stacked stem waveforms, drawn by hand."""
    rows = bytearray()
    radius = SIZE * 0.225
    inset = SIZE * 0.06
    left, top = inset, inset
    right, bottom = SIZE - inset, SIZE - inset

    bars = 4
    band = (bottom - top) * 0.78 / bars
    band_top = top + (bottom - top) * 0.11

    for y in range(SIZE):
        rows.append(0)  # PNG filter byte: none
        for x in range(SIZE):
            # Rounded-rectangle mask with a soft edge.
            dx = max(left + radius - x, 0, x - (right - radius))
            dy = max(top + radius - y, 0, y - (bottom - radius))
            dist = math.hypot(dx, dy)
            inside = (left <= x <= right) and (top <= y <= bottom)
            edge = 1.0 if (inside and dist <= radius) else 0.0
            if inside and radius - 1.5 < dist <= radius + 1.5:
                edge = max(0.0, (radius + 1.5 - dist) / 3.0)
            if edge <= 0.0:
                rows.extend((0, 0, 0, 0))
                continue

            # Vertical gradient, warm orange into violet.
            t = (y - top) / (bottom - top)
            color = _blend((214, 84, 42), (98, 62, 214), t * 0.85)

            index = int((y - band_top) // band)
            if 0 <= index < bars:
                local = (y - band_top) - index * band
                centre = band / 2
                # Each stem is a sine of its own frequency and amplitude.
                freq = (1.4 + index * 1.1) * math.tau / (right - left)
                amp = band * (0.34 - index * 0.04)
                wave = math.sin((x - left) * freq + index * 1.7) * amp
                thickness = band * 0.085
                if abs(local - centre - wave) < thickness:
                    color = (255, 252, 246)

            rows.extend((*color, round(255 * edge)))
    return rows


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def write_png(path: Path, raw: bytearray) -> None:
    header = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _chunk(b"IEND", b"")
    )


def main(destination: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        master = tmp / "icon.png"
        write_png(master, render())

        iconset = tmp / "StemForge.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            for scale, suffix in ((1, ""), (2, "@2x")):
                subprocess.run(
                    ["sips", "-z", str(size * scale), str(size * scale),
                     str(master), "--out", str(iconset / f"icon_{size}x{size}{suffix}.png")],
                    check=True, capture_output=True,
                )
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(destination)], check=True
        )
    print(destination)
    return 0


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/StemForge.icns")
    raise SystemExit(main(out))
