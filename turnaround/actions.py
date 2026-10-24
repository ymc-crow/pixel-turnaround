"""Action motions: sword attack, spell cast and death.

Like idle and walk, these move the finished sprite instead of redrawing it.
What the character holds or throws (sword, slash trail, magic) is drawn on top
as plain pixels, so it stays crisp at any frame size.

Every function takes the standing frame and a direction and returns frames on
a bigger canvas (room for the sword and spells). Feet stay on the same line.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

FORWARD = {"right": (1, 0), "left": (-1, 0), "down": (0, 1), "up": (0, -1)}

OUTLINE = (40, 32, 48, 255)
BLADE = (226, 232, 240, 255)
BLADE_EDGE = (150, 160, 178, 255)
GUARD = (230, 180, 60, 255)
GRIP = (110, 64, 40, 255)
TRAIL = (255, 255, 255, 220)
TRAIL_EDGE = (150, 220, 255, 170)
MAGIC = [(255, 255, 255, 255), (160, 236, 255, 255), (70, 170, 255, 255), (40, 90, 220, 255)]


# --- canvas helpers ---------------------------------------------------------

def action_size(frame: int) -> int:
    return -(-int(frame * 1.5) // 8) * 8


def _place(sprite: Image.Image, size: int, dx: int = 0, dy: int = 0) -> Image.Image:
    """Put a frame-sized sprite in the middle of the bigger canvas."""
    out = Image.new("RGBA", (size, size))
    out.alpha_composite(sprite, ((size - sprite.width) // 2 + dx, (size - sprite.height) // 2 + dy))
    return out


def _box(img: Image.Image):
    a = np.asarray(img)[..., 3] > 0
    ys, xs = np.nonzero(a)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _hand(img: Image.Image, direction: str):
    """Roughly where the character holds things."""
    x0, y0, x1, y1 = _box(img)
    fx, fy = FORWARD[direction]
    cx, h = (x0 + x1) / 2, y1 - y0
    return cx + fx * (x1 - x0) * 0.3, y0 + h * 0.6 + fy * h * 0.08


def _shoulder(img: Image.Image):
    x0, y0, x1, y1 = _box(img)
    return (x0 + x1) / 2, y0 + (y1 - y0) * 0.45


def _angle(direction: str, rel: float) -> float:
    """Screen angle in degrees, `rel` measured from the facing direction."""
    fx, fy = FORWARD[direction]
    return math.degrees(math.atan2(fy, fx)) + rel


def _unit(deg: float):
    return math.cos(math.radians(deg)), math.sin(math.radians(deg))


def _behind(direction: str) -> bool:
    """Seen from the back, held things go behind the body."""
    return direction == "up"


def _layer(body: Image.Image, fx: Image.Image, direction: str) -> Image.Image:
    if _behind(direction):
        return Image.alpha_composite(fx, body)
    return Image.alpha_composite(body, fx)


def _tint(img: Image.Image, color, amount: float) -> Image.Image:
    a = np.asarray(img).astype(np.float32)
    mask = a[..., 3] > 0
    a[mask, :3] = a[mask, :3] * (1 - amount) + np.array(color[:3]) * amount
    return Image.fromarray(a.astype(np.uint8))


def _shear(img: Image.Image, lean: int, direction: str) -> Image.Image:
    """Lean the upper body `lean` pixels toward the facing side (0 at the feet)."""
    fx, _ = FORWARD[direction]
    if fx == 0 or lean == 0:
        return img
    a = np.asarray(img)
    _, y0, _, y1 = _box(img)
    out = np.zeros_like(a)
    for y in range(a.shape[0]):
        k = 0 if y >= y1 else round(lean * fx * (y1 - y) / max(1, y1 - y0))
        out[y] = np.roll(a[y], k, axis=0)
    return Image.fromarray(out)


# --- drawing ----------------------------------------------------------------

def _sword(draw: ImageDraw.ImageDraw, hand, deg: float, length: float) -> None:
    ux, uy = _unit(deg)
    hx, hy = hand
    tip = (hx + ux * length, hy + uy * length)
    base = (hx + ux * 3, hy + uy * 3)
    draw.line([base, tip], fill=OUTLINE, width=4)
    draw.line([base, tip], fill=BLADE_EDGE, width=2)
    draw.line([(base[0] - uy * 0.5, base[1] + ux * 0.5), tip], fill=BLADE, width=1)
    px, py = -uy, ux
    draw.line([(base[0] + px * 3, base[1] + py * 3), (base[0] - px * 3, base[1] - py * 3)],
              fill=GUARD, width=2)
    draw.line([(hx - ux * 3, hy - uy * 3), (hx, hy)], fill=GRIP, width=2)


def _trail(draw: ImageDraw.ImageDraw, hand, a0: float, a1: float, r0: float, r1: float) -> None:
    """Crescent swipe between two angles: thick at the end, thin at the start."""
    steps = max(4, int(abs(a1 - a0) / 6))
    outer, inner = [], []
    for i in range(steps + 1):
        t = i / steps
        deg = a0 + (a1 - a0) * t
        ux, uy = _unit(deg)
        r_in = r1 - (r1 - r0) * t
        outer.append((hand[0] + ux * r1, hand[1] + uy * r1))
        inner.append((hand[0] + ux * r_in, hand[1] + uy * r_in))
    draw.polygon(outer + inner[::-1], fill=TRAIL, outline=TRAIL_EDGE)


def _orb(draw: ImageDraw.ImageDraw, center, radius: float) -> None:
    cx, cy = center
    for color, k in zip(MAGIC[::-1], (1.0, 0.75, 0.5, 0.25)):
        r = max(0.5, radius * k)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)


def _circle(draw: ImageDraw.ImageDraw, feet, width: float, phase: int, alpha: int = 255) -> None:
    """Flat magic ring at the feet with dots that turn each frame."""
    cx, cy = feet
    rx, ry = width / 2, width / 6
    edge = MAGIC[2][:3] + (alpha,)
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=edge, width=1)
    for i in range(6):
        deg = phase * 20 + i * 60
        ux, uy = _unit(deg)
        x, y = cx + ux * rx, cy + uy * ry
        draw.rectangle((x - 1, y - 1, x, y), fill=MAGIC[1][:3] + (alpha,))


def _sparkles(draw: ImageDraw.ImageDraw, center, spread: float, seed: int, count: int = 5):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        x = center[0] + rng.uniform(-spread, spread)
        y = center[1] + rng.uniform(-spread, spread)
        c = MAGIC[int(rng.integers(0, 2))]
        draw.point([(x, y)], fill=c)
        draw.point([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)], fill=MAGIC[2])


def _feet(img: Image.Image):
    x0, _, x1, y1 = _box(img)
    return (x0 + x1) / 2, y1 - 1


# --- motions ----------------------------------------------------------------

def attack(still: Image.Image, direction: str) -> list[Image.Image]:
    """Six frames: wind up, big swing with a trail, recover."""
    size = action_size(still.width)
    fx, fy = FORWARD[direction]
    length = still.height * 0.32
    # (body shift, lean, sword angle from facing, trail from-to or None)
    keys = [
        (-1, -1, -60, None),
        (-1, -2, -130, None),
        (2, 2, 30, (-130, 30)),
        (2, 2, 60, (-40, 60)),
        (1, 1, 75, None),
        (0, 0, 45, None),
    ]
    frames = []
    for shift, lean, rel, trail in keys:
        body = _shear(_place(still, size, shift * fx, max(0, shift) * fy), lean, direction)
        fx_layer = Image.new("RGBA", body.size)
        draw = ImageDraw.Draw(fx_layer)
        # The hand circles the shoulder, so a raised sword means a raised arm.
        sx, sy = _shoulder(body)
        reach = still.height * 0.14
        ux, uy = _unit(_angle(direction, rel))
        hand = (sx + ux * reach, sy + uy * reach)
        if trail:
            _trail(draw, (sx, sy), _angle(direction, trail[0]), _angle(direction, trail[1]),
                   reach + length * 0.4, reach + length + 2)
        _sword(draw, hand, _angle(direction, rel), length)
        frames.append(_layer(body, fx_layer, direction))
    return frames


def cast(still: Image.Image, direction: str) -> list[Image.Image]:
    """Six frames: crouch, ring and orb grow, the orb flies off."""
    size = action_size(still.width)
    fx, fy = FORWARD[direction]
    base = _place(still, size)
    feet = _feet(base)
    ring = (_box(base)[2] - _box(base)[0]) * 1.1
    # (body dy, orb radius, orb travel, ring alpha, glow)
    keys = [
        (1, 0, 0, 120, 0.0),
        (0, 2, 0, 255, 0.1),
        (-1, 4, 0, 255, 0.25),
        (-1, 5, 0, 255, 0.35),
        (0, 3, 14, 180, 0.15),
        (0, 2, 26, 90, 0.0),
    ]
    frames = []
    for i, (dy, radius, travel, alpha, glow) in enumerate(keys):
        body = _place(still, size, 0, dy)
        if glow:
            body = _tint(body, MAGIC[1], glow)
        back = Image.new("RGBA", body.size)
        front = Image.new("RGBA", body.size)
        _circle(ImageDraw.Draw(back), feet, ring, i, alpha)
        hand = _hand(body, direction)
        if radius:
            orb = (hand[0] + fx * (4 + travel), hand[1] + fy * (4 + travel) - (radius if fy else 0))
            target = back if _behind(direction) else front
            d = ImageDraw.Draw(target)
            if travel:
                d.line([(orb[0] - fx * travel, orb[1] - fy * travel), orb], fill=MAGIC[2], width=2)
            _orb(d, orb, radius)
            _sparkles(d, orb, radius + 4, seed=i)
        frames.append(Image.alpha_composite(Image.alpha_composite(back, body), front))
    return frames


def _rotate(img: Image.Image, deg: float, pivot) -> Image.Image:
    """Pixel-art friendly rotation: rotate at 4x, then sample pixel centers."""
    k = 4
    big = img.resize((img.width * k, img.height * k), Image.NEAREST)
    big = big.rotate(deg, resample=Image.NEAREST, center=(pivot[0] * k, pivot[1] * k))
    a = np.asarray(big)[k // 2::k, k // 2::k]
    return Image.fromarray(np.ascontiguousarray(a))


def _fall(still: Image.Image, size: int, deg: float, shift: int) -> Image.Image:
    """Tip the sprite over around its feet, keeping it on the floor.

    Done on a canvas twice as big so the head never gets cut off, then the
    lying body slides toward the middle as it falls so it still fits.
    """
    big = _place(still, size * 2, shift)
    feet = _feet(big)
    if deg:
        big = _rotate(big, deg, feet)
    x0, _, x1, y1 = _box(big)
    drift = ((x0 + x1) / 2 - feet[0]) * abs(deg) / 90
    left = round(feet[0] + drift - size / 2)
    floor = int(_feet(_place(still, size))[1])
    top = int(y1 - 1) - floor
    return big.crop((left, top, left + size, top + size))


def death(still: Image.Image, direction: str) -> list[Image.Image]:
    """Seven frames: white hit flash, knocked back, falls over, lies still."""
    size = action_size(still.width)
    fx, _ = FORWARD[direction]
    # Side views fall backward; front and back views topple sideways.
    sign = 1 if fx >= 0 else -1
    back = -fx if fx else -1
    keys = [(0, 0, "flash"), (2, 10, None), (3, 35, None), (4, 65, None),
            (4, 90, None), (4, 90, "bounce"), (4, 90, "gray")]
    frames = []
    for shift, deg, look in keys:
        body = _fall(still, size, deg * sign, shift * back)
        if look == "flash":
            body = _tint(body, (255, 255, 255), 1.0)
        elif look == "bounce":
            body = body.transform(body.size, Image.AFFINE, (1, 0, 0, 0, 1, 1))
        elif look == "gray":
            a = np.asarray(body).astype(np.float32)
            gray = a[..., :3].mean(axis=2, keepdims=True) * 0.7
            a[..., :3] = a[..., :3] * 0.4 + gray * 0.6
            body = Image.fromarray(a.astype(np.uint8))
        frames.append(body)
    return frames


ACTIONS = {"attack": (attack, 80), "cast": (cast, 110), "death": (death, 110)}
