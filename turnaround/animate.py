"""pixel-animate: add motions to a sheet made by pixel-turnaround.

The standing sprite is cut into body parts along the pose skeleton saved in
sheet.json, and every motion bends those parts at the joints (see rig.py and
moves.py). Nothing is redrawn, so every frame keeps the exact look and colors
of the original. Sword, slash trail and magic are drawn on top as pixels.

idle:   breathing, arms sway a little
walk:   8 frames, legs swing with bending knees, arms swing opposite
attack: sword wind-up, swing with a slash trail, follow-through
cast:   arms rise, an orb grows over the head and is thrown forward
death:  hit flash, staggers back, falls flat, lies still

Attack, cast and death use a 1.5x bigger frame to fit the sword and the fall.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .moves import MOTIONS, render
from .pixelate import preview, sheet

ORDER = ["down", "left", "right", "up"]
# Last frame of these stays up longer in the preview GIFs.
HOLD = {"death": 1200}


def stack(rows: list[list[Image.Image]]) -> Image.Image:
    strips = [sheet(r) for r in rows]
    out = Image.new("RGBA", (max(s.width for s in strips), sum(s.height for s in strips)))
    for i, s in enumerate(strips):
        out.paste(s, (0, i * s.height))
    return out


def save_gif(frames: list[Image.Image], path: Path, ms: int, zoom: int = 6,
             hold: int | None = None) -> None:
    big = [preview(f, zoom).convert("RGB") for f in frames]
    durations = [ms] * len(big)
    if hold:
        durations[-1] = hold
    big[0].save(path, save_all=True, append_images=big[1:], duration=durations, loop=0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="pixel-animate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet_dir", type=Path, help="output folder of pixel-turnaround (-d 4)")
    ap.add_argument("--motions", default=",".join(MOTIONS),
                    help="comma separated: " + ", ".join(MOTIONS))
    args = ap.parse_args(argv)

    src = args.sheet_dir
    meta = json.loads((src / "sheet.json").read_text())
    if meta.get("order") != ORDER:
        ap.error("only 4-direction sheets are supported")
    if "skeleton" not in meta:
        ap.error("sheet.json has no skeleton; re-cut it with pixel-turnaround --from-raw")
    out = src / "anim"
    out.mkdir(exist_ok=True)
    stills = {d: Image.open(src / "frames" / f"{d}.png").convert("RGBA") for d in ORDER}

    result = {"rows": ORDER, "motions": {}}
    for name in args.motions.split(","):
        if name not in MOTIONS:
            ap.error(f"unknown motion: {name}")
        _, ms, _ = MOTIONS[name]
        # Left is drawn from the right-facing sprite and flipped.
        rows = [render(stills["right" if d == "left" else d],
                       meta["skeleton"]["right" if d == "left" else d], d, name) for d in ORDER]
        stack(rows).save(out / f"{name}.png")
        for d, frames in zip(ORDER, rows):
            save_gif(frames, out / f"{name}_{d}.gif", ms, hold=HOLD.get(name))
        result["motions"][name] = {"file": f"{name}.png", "frame": rows[0][0].width,
                                   "frames": len(rows[0]), "ms": ms}

    (out / "anim.json").write_text(json.dumps(result, indent=2))
    print(f"done: {out}")


if __name__ == "__main__":
    main()
