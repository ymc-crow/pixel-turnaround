"""Turn the raw SDXL sheet into real, game-ready pixel sprites."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


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
    a plain resize produces. The grid is kept in out.info["grid"] as
    (px, ox, py, oy) so points on the original image can be mapped later.
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
    snapped = Image.fromarray(out)
    snapped.info["grid"] = (px, ox, py, oy)
    return snapped


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

    Each sprite keeps info["place"] = (scale, x0, y0, left, top): a cell point
    (x, y) lands on the sprite at ((x - x0) * scale + left, (y - y0) * scale + top).
    """
    cut = []
    for cell in cells:
        rgb = np.asarray(cell.convert("RGB"), dtype=np.float32)
        fg = ~background_mask(cell)
        x0, y0, x1, y1 = _bbox(fg)
        cut.append((rgb[y0:y1, x0:x1], fg[y0:y1, x0:x1].astype(np.float32), (x0, y0)))

    tallest = max(max(r.shape[0], r.shape[1]) for r, _, _ in cut)
    if size <= 0:
        size = -(-(tallest + 2 * margin) // 8) * 8
        scale = 1.0
    else:
        scale = min(1.0, (size - 2 * margin) / tallest)

    small = []
    for rgb, alpha, _ in cut:
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
    for (px, m), (_, _, (x0, y0)) in zip(small, cut):
        recolored = Image.fromarray(px).quantize(palette=palette, dither=Image.Dither.NONE)
        rgba = np.asarray(recolored.convert("RGBA")).copy()
        rgba[..., 3] = np.where(m, 255, 0)
        sprite = Image.new("RGBA", (size, size))
        h, w = m.shape
        # Feet on the same line in every frame, centered sideways.
        left, top = (size - w) // 2, size - margin - h
        sprite.paste(Image.fromarray(rgba), (left, top))
        sprite.info["place"] = (scale, x0, y0, left, top)
        sprites.append(sprite)
    return sprites


def pick_colors(pixels: np.ndarray, colors: int, method: str = "distinct",
                seed: int = 0) -> np.ndarray:
    """`colors` RGB colors for an (N, 3) pixel list.

    "median": PIL median cut, which favors big areas.
    "distinct": k-means over the distinct colors instead of over pixels, so a
    small face or gem gets its own color as much as a large cape does.
    """
    if method == "median":
        pal = Image.fromarray(pixels[None].astype(np.uint8)).quantize(
            colors=colors, method=Image.Quantize.MEDIANCUT)
        out = np.array(pal.getpalette()[:colors * 3], np.uint8).reshape(-1, 3)
        return out
    p, count = np.unique(pixels.astype(np.int32) // 8, axis=0, return_counts=True)
    p = p.astype(np.float32) * 8 + 4
    if len(p) <= colors:
        return p.astype(np.uint8)
    # sqrt: big areas still count more, but no longer drown small ones; a color
    # seen only once or twice is mostly blending between two others.
    w = np.sqrt(count) * (count > 2)
    rng = np.random.default_rng(seed)
    centers = [p[rng.choice(len(p), p=w / w.sum())]]
    near = ((p - centers[0]) ** 2).sum(1)
    for _ in range(colors - 1):      # k-means++ start: spread the colors out
        centers.append(p[rng.choice(len(p), p=near * w / (near * w).sum())])
        near = np.minimum(near, ((p - centers[-1]) ** 2).sum(1))
    c = np.array(centers)
    for _ in range(12):
        label = np.argmin(((p[:, None] - c[None]) ** 2).sum(2), 1)
        for j in range(colors):
            m = label == j
            if w[m].sum() > 0:
                c[j] = (p[m] * w[m, None]).sum(0) / w[m].sum()
    return c.round().astype(np.uint8)


def nearest(rgb: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Index of the closest color for every pixel of an (H, W, 3) image."""
    flat = rgb.reshape(-1, 3).astype(np.int32)
    keys, inverse = np.unique(flat, axis=0, return_inverse=True)
    d = ((keys[:, None] - colors[None].astype(np.int32)) ** 2).sum(2)
    return d.argmin(1)[inverse.ravel()].reshape(rgb.shape[:2])


def _components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label 4-connected parts of a mask: (label image, size of each label);
    label 0 is everything outside the mask."""
    label = np.zeros(mask.shape, np.int32)
    sizes = [0]
    h, w = mask.shape
    for y0, x0 in zip(*np.nonzero(mask)):
        if label[y0, x0]:
            continue
        n = len(sizes)
        label[y0, x0] = n
        stack, size = [(y0, x0)], 0
        while stack:
            y, x = stack.pop()
            size += 1
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not label[yy, xx]:
                    label[yy, xx] = n
                    stack.append((yy, xx))
        sizes.append(size)
    return label, np.array(sizes)


def main_body(fg: np.ndarray, keep: float = 0.02) -> np.ndarray:
    """Drop specks floating away from the character (sparkles, dust): parts
    smaller than `keep` of the biggest one go."""
    label, sizes = _components(fg)
    return (sizes >= keep * sizes.max())[label] & fg


def open_holes(cell: Image.Image, fg: np.ndarray, biggest: float = 0.004,
               tolerance: float = 40) -> np.ndarray:
    """Background caught between an arm and a staff is enclosed, so the flood
    in background_mask never reaches it. Enclosed patches of background color
    bigger than `biggest` of the cell are background too; smaller ones (eye
    whites, shine) stay."""
    rgb = np.asarray(cell.convert("RGB"), dtype=np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    holes = fg & (np.linalg.norm(rgb - np.median(border, axis=0), axis=2) < tolerance)
    label, sizes = _components(holes)
    big = sizes > biggest * fg.size
    big[0] = False
    return fg & ~big[label]


LUMA = np.array([0.3, 0.59, 0.11])


def _vote_shrink(q: np.ndarray, fg: np.ndarray, size: tuple[int, int], weight: np.ndarray):
    """Shrink a palette-index image: each output pixel takes the most common
    color of its block, with dark colors counting extra so thin lines (eyes,
    seams) survive. A pixel is opaque when at least half its block is."""
    h, w = q.shape
    ow, oh = size
    ys = np.linspace(0, h, oh + 1).round().astype(int)
    xs = np.linspace(0, w, ow + 1).round().astype(int)
    idx = np.zeros((oh, ow), int)
    alpha = np.zeros((oh, ow), bool)
    for j in range(oh):
        for i in range(ow):
            m = fg[ys[j]:max(ys[j + 1], ys[j] + 1), xs[i]:max(xs[i + 1], xs[i] + 1)]
            if m.mean() < 0.5:
                continue
            block = q[ys[j]:max(ys[j + 1], ys[j] + 1), xs[i]:max(xs[i + 1], xs[i] + 1)][m]
            alpha[j, i] = True
            idx[j, i] = (np.bincount(block, minlength=len(weight)) * weight).argmax()
    return idx, alpha


def _drop_strays(idx: np.ndarray, alpha: np.ndarray, lum: np.ndarray) -> np.ndarray:
    """A lone pixel lighter than the color all four neighbors share is noise;
    a lone darker one (an eye, a rivet) is kept."""
    out = idx.copy()
    h, w = idx.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not alpha[y, x]:
                continue
            nb = [idx[y - 1, x], idx[y + 1, x], idx[y, x - 1], idx[y, x + 1]]
            ok = [alpha[y - 1, x], alpha[y + 1, x], alpha[y, x - 1], alpha[y, x + 1]]
            if not all(ok) or idx[y, x] in nb:
                continue
            common = max(set(nb), key=nb.count)
            if nb.count(common) >= 3 and lum[idx[y, x]] > lum[common]:
                out[y, x] = common
    return out


def _flatten(idx: np.ndarray, alpha: np.ndarray, colors: np.ndarray, near: float = 80,
             need: int = 4, rounds: int = 2) -> np.ndarray:
    """Shiny metal comes out speckled with many shades. Hand-made sprites paint
    each material in a few flat shades, so a pixel takes the color most of its
    8 neighbors share when that color is a close shade of its own (same
    material). Real edges between materials (skin and eye, cape and armor)
    are far apart in color and stay."""
    rgb = colors.astype(np.float32)
    close = np.linalg.norm(rgb[:, None] - rgb[None], axis=2) < near
    out = idx.copy()
    h, w = idx.shape
    for _ in range(rounds):
        src = out.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if not alpha[y, x]:
                    continue
                solid = np.delete(alpha[y - 1:y + 2, x - 1:x + 2].ravel(), 4)
                nb = np.delete(src[y - 1:y + 2, x - 1:x + 2].ravel(), 4)[solid]
                count = np.bincount(nb, minlength=len(colors))
                c = count.argmax()
                if count[c] >= need and c != src[y, x] and close[src[y, x], c]:
                    out[y, x] = c
    return out


def outline(rgba: np.ndarray, color) -> np.ndarray:
    """1px outline around the silhouette, the way hand-made sprites read at
    small sizes."""
    out = rgba.copy()
    solid = rgba[..., 3] > 0
    p = np.pad(solid, 1)
    ring = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & ~solid
    out[ring, :3] = color
    out[ring, 3] = 255
    return out


def downscale_sprites(cells: list[Image.Image], size: int = 48, colors: int = 32,
                      dark_weight: float = 2.0, method: str = "distinct") -> list[Image.Image]:
    """Like to_sprites, for models whose fake pixels do not sit on a grid.

    Colors are picked once at full size from every view, then each sprite
    pixel votes over its block of the big image, and a dark outline goes
    around the result. info["place"] works as in to_sprites.
    """
    cut = []
    for cell in cells:
        fg = main_body(open_holes(cell, ~background_mask(cell)))
        x0, y0, x1, y1 = _bbox(fg)
        cut.append((np.asarray(cell.convert("RGB"))[y0:y1, x0:x1], fg[y0:y1, x0:x1], (x0, y0)))
    rgb_of = pick_colors(np.concatenate([rgb[m] for rgb, m, _ in cut]), colors, method)
    ink = rgb_of[(rgb_of @ LUMA).argmin()]

    margin = 1                           # room for the outline
    scale = (size - 2 * margin) / max(max(rgb.shape[:2]) for rgb, _, _ in cut)
    sprites = []
    for rgb, fg, (x0, y0) in cut:
        h, w = max(1, round(rgb.shape[0] * scale)), max(1, round(rgb.shape[1] * scale))
        small = paint_small(rgb, fg, (w, h), rgb_of, dark_weight)
        rgba = np.zeros((size, size, 4), np.uint8)
        left, top = (size - w) // 2, size - margin - h
        rgba[top:top + h, left:left + w] = small
        sprite = Image.fromarray(outline(rgba, ink))
        sprite.info["place"] = (scale, x0, y0, left, top)
        sprites.append(sprite)
    return sprites


def paint_small(rgb: np.ndarray, fg: np.ndarray, size: tuple[int, int], colors: np.ndarray,
                dark_weight: float = 2.0, smooth: int = 3) -> np.ndarray:
    """Big drawing -> small RGBA in the given colors: a light median blur
    first (the model's shading is speckled; flat areas read better small),
    then nearest color at full size, a weighted vote per sprite pixel, lone
    light specks removed, and speckled shading of one material flattened."""
    rgb = np.asarray(Image.fromarray(rgb).filter(ImageFilter.MedianFilter(smooth))) if smooth else rgb
    lum = colors @ LUMA
    weight = 1 + dark_weight * (1 - lum / 255) ** 2
    idx, alpha = _vote_shrink(nearest(rgb, colors), fg, size, weight)
    idx = _flatten(_drop_strays(idx, alpha, lum), alpha, colors)
    out = np.zeros((size[1], size[0], 4), np.uint8)
    out[..., :3] = colors[idx]
    out[..., 3] = alpha * 255
    return out


def inline(rgba: np.ndarray, color) -> np.ndarray:
    """Like outline, but on the edge pixels themselves, so the size of the
    silhouette does not change."""
    out = rgba.copy()
    solid = rgba[..., 3] > 0
    p = np.pad(solid, 1)
    edge = solid & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    out[edge, :3] = color
    return out


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
