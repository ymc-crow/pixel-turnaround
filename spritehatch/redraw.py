"""Let the model paint over rig frames so every frame looks hand drawn.

Rig frames move cut-out parts, which reads stiff: an arm hidden behind the body
never appears, a cape never sways. Here the rig frames become the rough draft
(the approach of "From Rigging to Waving", arXiv 2509.06573):

1. Four frames go into one 2x2 image, so the model paints them together and
   keeps the outfit the same across them (the way PixelLab edits animations
   as one grid).
2. img2img starts from that draft, and an OpenPose grid built from the rig's
   posed joints holds every limb where the motion wants it.
3. Each cell is shrunk back to the frame size and recolored with the original
   sprite's palette, so the frames still match the standing sprite.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .comfy import WORKFLOWS, ComfyClient
from .pixelate import LUMA, background_mask, inline, main_body, open_holes, paint_small
from .pose import VIEW_WORDS, draw_cell

# Target cell size. Small on purpose: the model's fake pixels keep their size,
# so a small cell gives few, big ones, close to what a 48px frame can hold
# (and it is much faster than SDXL's own 1024x1024).
CELL = 384
GRID_MAX = 768
GROUP = 4


def cell_size(frame: int, body: int | None = None) -> int:
    """Cell size that is a whole multiple of the frame, so every sprite pixel
    becomes an exact block in the draft and shrinks back without blur.

    `body` is the standing sprite's size. Attack and death use a bigger frame
    around the same body; scaling by the body keeps the character painted at
    the same size (a smaller painting comes out darker and blurrier)."""
    # ...but keep the 2x2 grid at most 768px; bigger ones run a 32 GB Mac
    # machine out of memory.
    return frame * max(1, min(CELL // (body or frame), GRID_MAX // (2 * frame)))

PROMPT = (
    "pixel art, sprite sheet, {n} animation frames of the same character {action}, "
    "{view}, {character}, chibi, big head, full body, identical outfit and colors "
    "in every frame, plain white background, game sprite, clean pixels"
)
ACTION = {"idle": "standing and breathing", "walk": "walking", "attack": "swinging a sword",
          "cast": "casting a spell", "death": "getting hit and falling down"}


def every_nth(items: list, count: int, keep_last: bool = False) -> list:
    """`count` evenly spaced items, starting with the first.

    Loops (walk) wrap around, so the last item is skipped like any other.
    One-shot motions (death) must end on their last frame: keep_last=True.
    """
    n = len(items)
    if n <= count:
        return list(items)
    if keep_last:
        return [items[round(i * (n - 1) / (count - 1))] for i in range(count)]
    return [items[i * n // count] for i in range(count)]


def build_grids(frames: list[Image.Image], keypoints: list,
                body: int | None = None) -> tuple[Image.Image, Image.Image]:
    """Up to four frames -> (draft on white, OpenPose skeletons on black), 2x2."""
    cell = cell_size(frames[0].width, body)
    draft = Image.new("RGB", (cell * 2, cell * 2), "white")
    pose = Image.new("RGB", (cell * 2, cell * 2), "black")
    draw = ImageDraw.Draw(pose)
    for k, (frame, pts) in enumerate(zip(frames, keypoints)):
        cx, cy = (k % 2) * cell, (k // 2) * cell
        big = frame.resize((cell, cell), Image.NEAREST)
        draft.paste(big, (cx, cy), big)
        if pts is None:
            continue
        s = cell / frame.width
        # draw_cell maps x as x0 + x*w and y as y*h, so y is a share of the grid.
        cell_pts = [None if p is None else (p[0] * s / cell, (p[1] * s + cy) / (cell * 2))
                    for p in pts]
        draw_cell(draw, cell_pts, cx, cell, cell * 2, stick=5, dot=5)
    return draft, pose


def split_grid(grid: Image.Image, count: int, frame: int, colors: np.ndarray,
               body: int | None = None) -> list[Image.Image]:
    """Cut the painted grid back into `count` frames of frame x frame pixels,
    shrunk the same way as the stills and drawn only in their colors."""
    out = []
    cell = cell_size(frame, body)
    ink = colors[(colors @ LUMA).argmin()]
    for k in range(count):
        cx, cy = (k % 2) * cell, (k // 2) * cell
        part = grid.crop((cx, cy, cx + cell, cy + cell))
        fg = main_body(open_holes(part, ~background_mask(part)))
        small = paint_small(np.asarray(part.convert("RGB")), fg, (frame, frame), colors)
        out.append(Image.fromarray(inline(small, ink)))
    return out


def sprite_colors(still: Image.Image) -> np.ndarray:
    """Every color the standing sprite uses."""
    a = np.asarray(still.convert("RGBA"))
    return np.unique(a[a[..., 3] > 0][:, :3], axis=0)


def redraw(frames: list[Image.Image], keypoints: list, *, still: Image.Image,
           character: str, view: str, motion: str, client: ComfyClient,
           denoise: float = 0.6, seed: int = 0, model: str = "zimage") -> list[Image.Image]:
    """Paint over all frames of one motion and view, four at a time.

    Some frames keep their rig drawing: ones without a skeleton (nothing to
    guide the model, it paints a lump) and ones the model mostly lost, like
    the white hit flash, which melts into the white background."""
    colors = sprite_colors(still)
    prompt = PROMPT.format(n=GROUP, action=ACTION.get(motion, motion),
                           view=VIEW_WORDS[view], character=character)
    out = []
    for g in range(0, len(frames), GROUP):
        part, pts = frames[g:g + GROUP], keypoints[g:g + GROUP]
        draft, pose = build_grids(part, pts, still.width)
        tag = f"redraw_{motion}_{view}_{g}"
        wf = WORKFLOWS[model](prompt, client.upload_image(draft, f"{tag}_draft.png"),
                             client.upload_image(pose, f"{tag}_pose.png"),
                             denoise=denoise, seed=seed)
        painted = client.run(wf)[0]
        for rig, new, p in zip(part, split_grid(painted, len(part), part[0].width, colors,
                                                   still.width), pts):
            out.append(new if p is not None and _area(new) >= 0.5 * _area(rig) else rig)
    return out


def _area(frame: Image.Image) -> int:
    return int((np.asarray(frame)[..., 3] > 0).sum())
