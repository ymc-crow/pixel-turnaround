"""pixel-animate: add motions to a sheet made by pixel-turnaround.

Motions are made by moving the finished pixels, not by drawing new ones, so
every frame keeps exactly the look and colors of the standing sprite.

idle:   the upper body sinks one pixel (breathing).
walk:   legs step and the body bobs. Side views swing the legs, front and back
        views lift one foot at a time.
attack: sword wind-up and swing with a slash trail.
cast:   magic ring at the feet, a glowing orb grows and flies off.
death:  white hit flash, knocked back, falls over, lies still.

Attack, cast and death use a 1.5x bigger frame to fit the sword and spells.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .actions import ACTIONS
from .pixelate import preview, sheet

ORDER = ["down", "left", "right", "up"]
HIP = 0.70  # where the legs start, as a share of the sprite's height


def _rows(a: np.ndarray):
    rows = np.nonzero(a[..., 3].any(axis=1))[0]
    return rows[0], rows[-1]


def _bob(a: np.ndarray, dy: int, cut: float) -> np.ndarray:
    """Move everything above `cut` (share of height) by dy pixels, + is down."""
    top, bottom = _rows(a)
    split = top + int((bottom - top) * cut)
    out = a.copy()
    out[top + dy:split + dy] = a[top:split]
    if dy > 0:
        out[top:top + dy] = 0
    return out


def _leg_masks(a: np.ndarray):
    """Opaque leg pixels split into the left and right half."""
    top, bottom = _rows(a)
    hip = top + int((bottom - top) * HIP)
    legs = np.zeros(a.shape[:2], bool)
    legs[hip:bottom + 1] = a[hip:bottom + 1, :, 3] > 0
    cx = round(float(np.nonzero(legs)[1].mean()))
    left, right = legs.copy(), legs.copy()
    left[:, cx:] = False
    right[:, :cx] = False
    return hip, bottom, left, right


def _move(a: np.ndarray, part: np.ndarray, shift) -> np.ndarray:
    """Cut `part` out and paste it back moved by shift(row) = (dx, dy)."""
    out = a.copy()
    out[part] = 0
    for y, x in zip(*np.nonzero(part)):
        dx, dy = shift(y)
        ny, nx = y + dy, x + dx
        if 0 <= ny < a.shape[0] and 0 <= nx < a.shape[1]:
            out[ny, nx] = a[y, x]
    return out


def breathe(frame: Image.Image) -> list[Image.Image]:
    """Two idle frames: as is, then the upper body sunk by one pixel."""
    a = np.asarray(frame.convert("RGBA"))
    return [frame, Image.fromarray(_bob(a, 1, 0.55))]


def walk_side(frame: Image.Image, stride: int = 2) -> list[Image.Image]:
    """Four frames for a profile: legs apart, together, apart, together."""
    a = np.asarray(frame.convert("RGBA"))
    hip, bottom, back, front = _leg_masks(a)

    def swing(k):
        return lambda y: (round(k * (y - hip) / max(1, bottom - hip)), 0)

    apart = _move(_move(a, front, swing(stride)), back, swing(-stride))
    passing = _bob(a, -1, HIP)
    return [Image.fromarray(f) for f in (apart, passing, apart, passing)]


def walk_front(frame: Image.Image) -> list[Image.Image]:
    """Four frames seen from the front or back: one foot up, then the other."""
    a = np.asarray(frame.convert("RGBA"))
    hip, _, left, right = _leg_masks(a)

    def lift(y):
        return (0, -1 if y > hip else 0)

    passing = _bob(a, -1, HIP)
    frames = (_move(a, left, lift), passing, _move(a, right, lift), passing)
    return [Image.fromarray(f) for f in frames]


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


MOTIONS = {
    "idle": (lambda d, f: breathe(f), 400),
    "walk": (lambda d, f: walk_side(f) if d in ("left", "right") else walk_front(f), 160),
    **{name: (lambda d, f, fn=fn: fn(f, d), ms) for name, (fn, ms) in ACTIONS.items()},
}
# Last frame of these stays up longer in the preview GIFs.
HOLD = {"death": 1200}


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
    out = src / "anim"
    out.mkdir(exist_ok=True)
    stills = {d: Image.open(src / "frames" / f"{d}.png").convert("RGBA") for d in ORDER}

    result = {"frame": meta["frame"], "rows": ORDER, "motions": {}}
    for name in args.motions.split(","):
        if name not in MOTIONS:
            ap.error(f"unknown motion: {name}")
        make, ms = MOTIONS[name]
        rows = [make(d, stills[d]) for d in ORDER]
        stack(rows).save(out / f"{name}.png")
        for d, frames in zip(ORDER, rows):
            save_gif(frames, out / f"{name}_{d}.gif", ms, hold=HOLD.get(name))
        result["motions"][name] = {"file": f"{name}.png", "frame": rows[0][0].width,
                                   "frames": len(rows[0]), "ms": ms}

    (out / "anim.json").write_text(json.dumps(result, indent=2))
    print(f"done: {out}")


if __name__ == "__main__":
    main()
