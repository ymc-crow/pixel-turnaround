"""One GIF with every motion (rows) in down/right/up (columns), on one clock.

    python -m spritehatch.showcase out/knight/anim
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

MOTIONS = ["idle", "walk", "attack", "cast", "death"]
VIEWS = ["down", "right", "up"]


def showcase(anim_dir: Path, zoom: int = 3, tick: int = 50, length: int = 2400) -> list[Image.Image]:
    meta = json.loads((anim_dir / "anim.json").read_text())
    rows = meta["rows"]
    motions = [m for m in MOTIONS if m in meta["motions"]]
    cell = max(meta["motions"][m]["frame"] for m in motions)
    strips = {}
    for m in motions:
        info = meta["motions"][m]
        img = Image.open(anim_dir / info["file"]).convert("RGBA")
        f = info["frame"]
        strips[m] = [[img.crop((i * f, rows.index(v) * f, (i + 1) * f, (rows.index(v) + 1) * f))
                      for i in range(info["frames"])] for v in VIEWS]
    frames = []
    for t in range(0, length, tick):
        canvas = Image.new("RGBA", (cell * len(VIEWS), cell * len(motions)), (236, 236, 240, 255))
        for r, m in enumerate(motions):
            n, ms = meta["motions"][m]["frames"], meta["motions"][m]["ms"]
            # Death plays once and stays down; the rest loop.
            k = min(t // ms, n - 1) if m == "death" else (t // ms) % n
            for c, strip in enumerate(strips[m]):
                fr = strip[k]
                canvas.alpha_composite(fr, (c * cell + (cell - fr.width) // 2,
                                            r * cell + cell - fr.height))
        frames.append(canvas.resize((canvas.width * zoom, canvas.height * zoom),
                                    Image.NEAREST).convert("RGB"))
    return frames


def main(argv=None) -> None:
    anim_dir = Path((argv or sys.argv[1:])[0])
    frames = showcase(anim_dir)
    out = anim_dir / "showcase.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=50, loop=0)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
