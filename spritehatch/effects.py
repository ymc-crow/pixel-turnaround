"""Pixel drawing for things a character holds or throws, and frame filters."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

OUTLINE = (40, 32, 48, 255)
BLADE = (226, 232, 240, 255)
BLADE_EDGE = (150, 160, 178, 255)
GUARD = (230, 180, 60, 255)
GRIP = (110, 64, 40, 255)
TRAIL = (255, 255, 255, 180)
TRAIL_EDGE = (150, 220, 255, 170)
MAGIC = [(255, 255, 255, 255), (160, 236, 255, 255), (70, 170, 255, 255), (40, 90, 220, 255)]


def unit(deg: float):
    return math.cos(math.radians(deg)), math.sin(math.radians(deg))


def box(img: Image.Image):
    a = np.asarray(img)[..., 3] > 0
    ys, xs = np.nonzero(a)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _sword_sprite(length: float) -> Image.Image:
    """A hand-pixeled sword pointing right, the hand at (4, 3): outlined
    blade with a bright top edge and a pointed tip, gold guard, brown grip."""
    w = int(length + 0.5) + 6
    a = np.zeros((7, w, 4), np.uint8)
    a[1:6, 5:w - 1] = OUTLINE
    a[2, 6:w - 2] = BLADE
    a[3:5, 6:w - 2] = BLADE_EDGE
    a[2:5, w - 2] = OUTLINE
    a[3, w - 2] = BLADE
    a[3, w - 1] = OUTLINE
    a[[1, 5], w - 2] = 0
    a[0:7, 4] = OUTLINE
    a[1:6, 4] = GUARD
    a[2:5, 0:4] = OUTLINE
    a[3, 1:4] = GRIP
    return Image.fromarray(a)


def sword(layer: Image.Image, hand, deg: float, length: float) -> None:
    """Draw a sword held at `hand`, pointing `deg` degrees (clockwise on
    screen), onto `layer`. The sprite is turned RotSprite-style: drawn
    straight with lines, a diagonal blade at this size breaks into stripes."""
    hx, hy = round(hand[0]), round(hand[1])
    canvas = Image.new("RGBA", layer.size)
    canvas.alpha_composite(_sword_sprite(length), (max(0, hx - 4), max(0, hy - 3)))
    layer.alpha_composite(rotate(canvas, -deg, (hx + 0.5, hy + 0.5)))


def trail(draw: ImageDraw.ImageDraw, center, a0: float, a1: float, r0: float, r1: float) -> None:
    """Crescent swipe from angle a0 to a1: thin where it started, thick at the end."""
    if abs(a1 - a0) < 1:
        return
    steps = max(4, int(abs(a1 - a0) / 6))
    outer, inner = [], []
    for i in range(steps + 1):
        t = i / steps
        ux, uy = unit(a0 + (a1 - a0) * t)
        r_in = r1 - (r1 - r0) * t
        outer.append((center[0] + ux * r1, center[1] + uy * r1))
        inner.append((center[0] + ux * r_in, center[1] + uy * r_in))
    draw.polygon(outer + inner[::-1], fill=TRAIL, outline=TRAIL_EDGE)


def orb(draw: ImageDraw.ImageDraw, center, radius: float) -> None:
    cx, cy = center
    for color, k in zip(MAGIC[::-1], (1.0, 0.75, 0.5, 0.25)):
        r = max(0.5, radius * k)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)


def magic_ring(draw: ImageDraw.ImageDraw, feet, width: float, phase: int, alpha: int = 255) -> None:
    """Flat ring at the feet with dots that turn each frame."""
    cx, cy = feet
    rx, ry = width / 2, width / 6
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=MAGIC[2][:3] + (alpha,), width=1)
    for i in range(6):
        ux, uy = unit(phase * 20 + i * 60)
        x, y = cx + ux * rx, cy + uy * ry
        draw.rectangle((x - 1, y - 1, x, y), fill=MAGIC[1][:3] + (alpha,))


def sparkles(draw: ImageDraw.ImageDraw, center, spread: float, seed: int, count: int = 5) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(count):
        x = center[0] + rng.uniform(-spread, spread)
        y = center[1] + rng.uniform(-spread, spread)
        draw.point([(x, y)], fill=MAGIC[int(rng.integers(0, 2))])
        draw.point([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)], fill=MAGIC[2])


def tint(img: Image.Image, color, amount: float) -> Image.Image:
    a = np.asarray(img).astype(np.float32)
    mask = a[..., 3] > 0
    a[mask, :3] = a[mask, :3] * (1 - amount) + np.array(color[:3]) * amount
    return Image.fromarray(a.astype(np.uint8))


def darken(img: Image.Image, amount: float = 0.6) -> Image.Image:
    """Grayer and darker, for a body that is down for good."""
    a = np.asarray(img).astype(np.float32)
    gray = a[..., :3].mean(axis=2, keepdims=True) * 0.7
    a[..., :3] = a[..., :3] * (1 - amount) + gray * amount
    return Image.fromarray(a.astype(np.uint8))


def scale2x(a: np.ndarray) -> np.ndarray:
    """Double a pixel-art image with the Scale2x (EPX) rule: corners follow
    matching neighbors, so diagonal edges stay smooth instead of turning into
    stairs of big blocks."""
    key = a.view(np.uint32)[..., 0]          # one number per RGBA pixel
    p = np.pad(key, 1, mode="edge")
    up, down, left, right = p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]
    out = np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)
    ap = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    src = {"up": ap[:-2, 1:-1], "down": ap[2:, 1:-1], "left": ap[1:-1, :-2], "right": ap[1:-1, 2:]}
    rules = [  # (output corner, condition, neighbor it copies)
        ((0, 0), (left == up) & (left != down) & (up != right), "up"),
        ((0, 1), (up == right) & (up != left) & (right != down), "right"),
        ((1, 0), (down == left) & (down != right) & (left != up), "left"),
        ((1, 1), (right == down) & (right != up) & (down != left), "down"),
    ]
    for (dy, dx), cond, name in rules:
        view = out[dy::2, dx::2]
        view[cond] = src[name][cond]
    return out


def rotate(img: Image.Image, deg: float, pivot) -> Image.Image:
    """Pixel-art friendly rotation (the RotSprite idea): grow 8x with Scale2x,
    rotate with nearest sampling, then read back one pixel per 8x8 block.

    Positive degrees turn counterclockwise on screen.
    """
    k = 8
    big = np.ascontiguousarray(np.asarray(img.convert("RGBA")))
    for _ in range(3):
        big = scale2x(big)
    big = Image.fromarray(big).rotate(deg, resample=Image.NEAREST,
                                      center=(pivot[0] * k, pivot[1] * k))
    return Image.fromarray(np.ascontiguousarray(np.asarray(big)[k // 2::k, k // 2::k]))


def ground(img: Image.Image, floor: int, center_x: float | None = None) -> Image.Image:
    """Slide the frame so its lowest pixel is on `floor` (and optionally centered)."""
    x0, _, x1, y1 = box(img)
    dx = 0 if center_x is None else round(center_x - (x0 + x1) / 2)
    out = Image.new("RGBA", img.size)
    out.paste(img, (dx, floor - (y1 - 1)), img)
    return out
