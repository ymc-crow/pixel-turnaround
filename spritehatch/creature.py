"""Motions for creatures (spritehatch --creature): no skeleton to bend, so the
whole sprite squashes, stretches, hops and tips over, the way slimes and
small monsters move in hand-made games.

idle:   slow squash and stretch
walk:   a hop: squash, spring up, land
attack: lean back, lunge forward with a flash, come back
death:  hit flash, squash, tip over, lie gray

Winged creatures (the top half much wider than the body below) also beat
their wings in idle and walk: see flap().
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import effects as fx
from .moves import canvas_size

FORWARD = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}
WHITE = (255, 255, 255)


def _place(still: Image.Image, size: int, sx: float = 1.0, sy: float = 1.0,
           dx: float = 0.0, dy: float = 0.0) -> Image.Image:
    """The still scaled by (sx, sy) around its feet and moved by (dx, dy),
    on a size x size canvas with the feet where they were."""
    x0, y0, x1, y1 = fx.box(still)
    body = still.crop((x0, y0, x1, y1))
    w, h = max(1, round(body.width * sx)), max(1, round(body.height * sy))
    body = body.resize((w, h), Image.NEAREST)
    pad_x, pad_y = (size - still.width) // 2, (size - still.height) // 2
    feet_x, feet_y = pad_x + (x0 + x1) / 2, pad_y + y1
    out = Image.new("RGBA", (size, size))
    out.alpha_composite(body, (round(feet_x - w / 2 + dx), round(feet_y - h + dy)))
    return out


def _wings(solid: np.ndarray):
    """(center x, half body width, shoulder row) when the top half spreads much
    wider than the narrowest row of the lower body (wings), else None."""
    ys, _ = np.nonzero(solid)
    top, bottom = ys.min(), ys.max()
    h = bottom - top + 1
    rows = [np.nonzero(solid[y])[0] for y in range(top + int(0.6 * h), top + int(0.9 * h))]
    rows = [r for r in rows if len(r)]
    if not rows:
        return None
    waist = min(rows, key=lambda r: r[-1] - r[0])
    body = waist[-1] - waist[0] + 1
    upper = [np.ptp(np.nonzero(solid[y])[0]) + 1 if solid[y].any() else 0
             for y in range(top, top + h // 2)]
    if max(upper) < 1.3 * body:
        return None
    shoulder = top + max(i for i, w in enumerate(upper) if w >= max(upper) - 1)
    return (waist[0] + waist[-1] + 1) / 2, body / 2, shoulder


def flap(still: Image.Image, view: str, amount: float) -> Image.Image:
    """Beat the wings: every column outside the body, above the shoulder row,
    slides down (amount < 0, folding) or up, more toward the wing tips. Whole
    columns move, so no holes open. Profiles only move the wing behind the
    back (the head is on the other side)."""
    a = np.asarray(still.convert("RGBA"))
    found = _wings(a[..., 3] > 0)
    if found is None or not amount:
        return still
    cx, half, shoulder = found
    out = a.copy()
    span = max(cx, a.shape[1] - cx) - half
    rows = np.arange(shoulder + 1)
    for x in range(a.shape[1]):
        side = x + 0.5 - cx
        if (view == "right" and side > 0) or (view == "left" and side < 0):
            continue
        o = abs(side) - half
        d = round(amount * o * o / span) if o > 0 else 0
        if d:
            src = np.clip(rows + d, 0, shoulder)
            col = a[src, x].copy()
            col[rows + d < 0] = 0
            out[:shoulder + 1, x] = col
    return Image.fromarray(out)


def idle(still: Image.Image, view: str) -> list[Image.Image]:
    s = still.width
    beat = (0, -0.1, -0.2, -0.1)            # wings sink a little as it breathes
    return [_place(flap(still, view, b), s, *k)
            for b, k in zip(beat, ((1, 1), (1.04, 0.96), (1, 1), (0.97, 1.03)))]


def walk(still: Image.Image, view: str) -> list[Image.Image]:
    s = canvas_size(still.width)          # room to hop
    fx_, fy = FORWARD[view]
    k = still.width / 48                  # hop sizes are for a 48px frame
    hop = [  # (sx, sy, forward, up)
        (1.08, 0.9, 0, 0),
        (0.94, 1.08, 1, 3),
        (1.0, 1.0, 2, 5),
        (0.96, 1.05, 1, 2),
    ]
    beat = (-0.3, 0.1, 0.15, -0.1)          # wings fold on landing, open in the air
    return [_place(flap(still, view, b), s, sx, sy, fx_ * f * k, (fy * f - up) * k)
            for b, (sx, sy, f, up) in zip(beat, hop)]


def attack(still: Image.Image, view: str) -> list[Image.Image]:
    s = canvas_size(still.width)
    fx_, fy = FORWARD[view]
    reach = still.width // 8

    def at(sx, sy, f):
        return _place(still, s, sx, sy, fx_ * f, fy * f)
    lunge = at(1.1, 0.92, reach) if fx_ else at(1.06, 1.06, reach)
    return [
        at(1.0, 1.0, 0),
        at(1.06, 0.92, -reach / 3),       # lean back
        lunge,
        fx.tint(lunge, WHITE, 0.5),       # the hit
        at(1.0, 1.0, reach / 2),
        at(1.0, 1.0, 0),
    ]


def death(still: Image.Image, view: str) -> list[Image.Image]:
    s = canvas_size(still.width)
    fx_, _ = FORWARD[view]
    back = -fx_ if fx_ else 1             # falls away from where it faces
    stand = _place(still, s)
    floor = fx.box(stand)[3]
    x0, _, x1, _ = fx.box(stand)
    pivot = ((x0 + x1) / 2, floor)
    out = [fx.tint(stand, WHITE, 0.8), _place(still, s, 1.1, 0.85, -back * 2 * still.width / 48)]
    # Turned on a canvas twice as big, so the body lying flat is not cut off,
    # then put back on the floor in the middle.
    m = s // 2
    big = Image.new("RGBA", (2 * s, 2 * s))
    big.alpha_composite(stand, (m, m))
    for deg in (35, 70, 90):
        # Positive turns counterclockwise: the head goes left.
        lying = fx.ground(fx.rotate(big, -deg * back, (pivot[0] + m, pivot[1] + m)),
                          floor + m, s)
        out.append(lying.crop((m, m, m + s, m + s)))
    out.append(fx.darken(out[-1], 0.5))
    return out


MOTIONS = {
    "idle": (idle, 220, False),
    "walk": (walk, 110, True),
    "attack": (attack, 80, True),
    "death": (death, 120, True),
}


def render(still: Image.Image, direction: str, motion: str) -> list[Image.Image]:
    return MOTIONS[motion][0](still.convert("RGBA"), direction)
