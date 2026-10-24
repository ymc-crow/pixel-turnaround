"""Turn the raw SDXL sheet into real, game-ready pixel sprites."""
from __future__ import annotations

import numpy as np
from PIL import Image


def _grid_axis(gray: np.ndarray, axis: int, lo: int = 3, hi: int = 20):
    """Pixel size and phase along one axis, from where edges line up."""
    edges = np.abs(np.diff(gray, axis=axis)).sum(axis=1 - axis)
    edges = edges - edges.mean()
    scores = {}
    for period in range(lo, hi):
        phase = max(range(period), key=lambda o: edges[o::period].mean())
        scores[period] = (edges[phase::period].mean(), phase)
    top = max(sc for sc, _ in scores.values())
    # Multiples of the true size score just as well, so take the smallest.
    period = min(p for p, (sc, _) in scores.items() if sc >= 0.8 * top)
    return period, (scores[period][1] + 1) % period


def snap_to_grid(img: Image.Image) -> Image.Image:
    """Undo the model's fake big pixels: one real pixel per drawn block.

    pixel-art LoRAs draw blocks of roughly 8x8 screen pixels. Taking the median
    of each block's center gives clean single-color pixels instead of the mush
    a plain resize produces.
    """
    rgb = np.asarray(img.convert("RGB"))
    gray = rgb.mean(axis=2)
    px, ox = _grid_axis(gray, 1)
    py, oy = _grid_axis(gray, 0)
    xs = np.arange(ox, rgb.shape[1] - px + 1, px)
    ys = np.arange(oy, rgb.shape[0] - py + 1, py)
    ix, iy = max(1, px // 4), max(1, py // 4)
    out = np.empty((len(ys), len(xs), 3), dtype=np.uint8)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            block = rgb[y + iy:y + py - iy, x + ix:x + px - ix].reshape(-1, 3)
            out[j, i] = np.median(block, axis=0)
    return Image.fromarray(out)


def split_cells(sheet: Image.Image, count: int) -> list[Image.Image]:
    w, h = sheet.size
    step = w / count
    return [sheet.crop((round(i * step), 0, round((i + 1) * step), h)) for i in range(count)]


def background_mask(cell: Image.Image, tolerance: float = 40) -> np.ndarray:
    """True where the pixel belongs to the plain background.

    Flood-fills from the border through pixels close to the border color, so
    white parts inside the character (eyes, armor shine) are kept.
    """
    rgb = np.asarray(cell.convert("RGB"), dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    similar = np.linalg.norm(rgb - bg, axis=2) < tolerance

    mask = np.zeros_like(similar)
    mask[0], mask[-1], mask[:, 0], mask[:, -1] = similar[0], similar[-1], similar[:, 0], similar[:, -1]
    while True:
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        grown &= similar
        if (grown == mask).all():
            return mask
        mask = grown


def _bbox(fg: np.ndarray):
    ys, xs = np.nonzero(fg)
    if len(ys) == 0:
        raise ValueError("no character found in a cell; try another seed")
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _shrink(rgb: np.ndarray, alpha: np.ndarray, size: tuple[int, int]):
    """Box-downscale with alpha weighting so edges don't pick up background."""
    a = Image.fromarray((alpha * 255).astype(np.uint8)).resize(size, Image.BOX)
    a = np.asarray(a, dtype=np.float32) / 255
    channels = []
    for c in range(3):
        premul = Image.fromarray((rgb[..., c] * alpha).astype(np.float32))
        channels.append(np.asarray(premul.resize(size, Image.BOX)))
    out = np.stack(channels, axis=2) / np.maximum(a, 1e-6)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), a >= 0.5


def to_sprites(cells: list[Image.Image], size: int = 0, colors: int = 24,
               margin: int = 1) -> list[Image.Image]:
    """Cut out, scale and recolor every cell the same way.

    One scale for all views keeps the character the same height, and one shared
    palette keeps the colors identical between directions. size=0 keeps the
    model's own pixel size and picks the smallest frame that fits.
    """
    cut = []
    for cell in cells:
        rgb = np.asarray(cell.convert("RGB"), dtype=np.float32)
        fg = ~background_mask(cell)
        x0, y0, x1, y1 = _bbox(fg)
        cut.append((rgb[y0:y1, x0:x1], fg[y0:y1, x0:x1].astype(np.float32)))

    tallest = max(max(r.shape[0], r.shape[1]) for r, _ in cut)
    if size <= 0:
        size = -(-(tallest + 2 * margin) // 8) * 8
        scale = 1.0
    else:
        scale = min(1.0, (size - 2 * margin) / tallest)

    small = []
    for rgb, alpha in cut:
        w = max(1, round(rgb.shape[1] * scale))
        h = max(1, round(rgb.shape[0] * scale))
        if scale == 1.0:
            small.append((rgb.astype(np.uint8), alpha >= 0.5))
        else:
            small.append(_shrink(rgb, alpha, (w, h)))

    # Build one palette from every opaque pixel of every view.
    pixels = np.concatenate([px[m] for px, m in small])[None]
    palette = Image.fromarray(pixels).quantize(colors=colors, method=Image.Quantize.MEDIANCUT)

    sprites = []
    for px, m in small:
        recolored = Image.fromarray(px).quantize(palette=palette, dither=Image.Dither.NONE)
        rgba = np.asarray(recolored.convert("RGBA")).copy()
        rgba[..., 3] = np.where(m, 255, 0)
        sprite = Image.new("RGBA", (size, size))
        h, w = m.shape
        # Feet on the same line in every frame, centered sideways.
        sprite.paste(Image.fromarray(rgba), ((size - w) // 2, size - margin - h))
        sprites.append(sprite)
    return sprites


def sheet(sprites: list[Image.Image]) -> Image.Image:
    size = sprites[0].width
    out = Image.new("RGBA", (size * len(sprites), sprites[0].height))
    for i, s in enumerate(sprites):
        out.paste(s, (i * size, 0))
    return out


def preview(img: Image.Image, zoom: int = 8, bg=(236, 236, 240, 255)) -> Image.Image:
    big = img.resize((img.width * zoom, img.height * zoom), Image.NEAREST)
    base = Image.new("RGBA", big.size, bg)
    return Image.alpha_composite(base, big)
