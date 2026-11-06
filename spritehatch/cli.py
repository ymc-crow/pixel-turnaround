"""spritehatch: one character, every direction, same look."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .comfy import ComfyClient, turnaround_workflow, zimage_turnaround_workflow
from .pixelate import (
    background_mask,
    downscale_sprites,
    preview,
    selout,
    sheet,
    snap_to_grid,
    split_cells,
    to_sprites,
)
from .pose import MODES, VIEW_WORDS, keypoints, pose_sheet, quadruped_sheet, quadruped_start

# Canvas per model and mode. SDXL wants about one megapixel. Z-Image gets a
# quarter of that: its fake pixels keep their size, so a smaller canvas means
# fewer of them per character, close to what fits in a 48px sprite (and it
# runs about 3x faster).
CANVAS = {"sdxl": {"4": (1536, 640), "8": (1600, 640)},
          "zimage": {"4": (768, 320), "8": (800, 320)}}
# A 96px boss holds four times the pixels of a 48px hero, so it gets a canvas
# twice as wide and tall.
BOSS_SIZE, BOSS_CANVAS = 96, {"4": (1536, 640), "8": (1600, 640)}


def _skeleton(view: str, index: int, count: int, raw: Image.Image,
              snapped: Image.Image, sprite: Image.Image, body: str = "chibi"):
    """Where the pose skeleton the model followed ends up on the final sprite."""
    px, ox, py, oy = snapped.info["grid"]
    scale, x0, y0, left, top = sprite.info["place"]
    cell_w = raw.width / count
    cell_x = round(index * snapped.width / count)
    points = []
    for p in keypoints(view, body):
        if p is None:
            points.append(None)
            continue
        gx = (index * cell_w + p[0] * cell_w - ox) / px - cell_x
        gy = (p[1] * raw.height - oy) / py
        points.append([round((gx - x0) * scale + left, 1), round((gy - y0) * scale + top, 1)])
    return points


# Four-legged creatures start from a silhouette (keep QUADRUPED_DENOISE of the
# way to noise) and follow the guide this hard; at 1.0 the guide's straight
# legs get painted as black sticks.
QUADRUPED_STRENGTH, QUADRUPED_DENOISE = 0.6, 0.85
_FOUR_LEGS = re.compile(r"four[ -]legg?(ed|s)|quadruped|on all fours", re.IGNORECASE)


def is_quadruped(character: str | None) -> bool:
    """A creature described as walking on four legs."""
    return bool(character and _FOUR_LEGS.search(character))


def _tail_side(cell: Image.Image) -> int:
    """-1 when a four-legged profile's tail is on the left, 1 on the right,
    0 when unclear. The tail hangs low behind the body; on the head side
    nothing reaches that far out down low (wings, up high, do not count)."""
    fg = ~background_mask(cell)
    h, w = fg.shape
    low = fg[int(0.62 * h):int(0.92 * h)]
    left, right = low[:, :int(0.24 * w)].mean(), low[:, int(0.76 * w):].mean()
    return 0 if abs(left - right) < 0.05 else (-1 if left > right else 1)


def face_profiles(cells: list[Image.Image], views: list[str]) -> list[Image.Image]:
    """Four-legged profiles come out facing either way: mirror the ones whose
    tail is on the side their head should be."""
    out = []
    for cell, view in zip(cells, views):
        tail = _tail_side(cell) if view in ("left", "right") else 0
        wrong = (view == "left" and tail < 0) or (view == "right" and tail > 0)
        out.append(ImageOps.mirror(cell) if wrong else cell)
    return out


def _pick_sides(mirror: dict, drawn: dict) -> dict:
    """Which view of each mirrored pair to keep. When the model drew both, the
    one with more drawn pixels wins: it is usually the one still holding the
    weapon or staff the other forgot."""
    out = {}
    for view, source in mirror.items():
        if view in drawn and _area(drawn[view]) > _area(drawn[source]):
            view, source = source, view
        out[view] = source
    return out


def _area(sprite: Image.Image) -> int:
    return int((np.asarray(sprite)[..., 3] > 0).sum())


def build(raw: Image.Image, mode: str, size: int, colors: int, out: Path,
          character: str | None = None, model: str = "zimage",
          swap_sides: bool = False, creature: bool = False, boss: bool = False) -> None:
    body = "boss" if boss else "chibi"
    spec = MODES[mode]
    count = len(spec["draw"])
    if model == "sdxl":
        # The SDXL pixel LoRA draws even blocks, so they can be read off a grid.
        snapped = snap_to_grid(raw)
        sprites = to_sprites(split_cells(snapped, count), size=size, colors=colors)
    else:
        snapped = raw.copy()
        snapped.info["grid"] = (1, 0, 1, 0)
        cells = split_cells(raw, count)
        if creature and is_quadruped(character):
            cells = face_profiles(cells, spec["draw"])
        sprites = downscale_sprites(cells, colors=colors,
                                    size=size or (BOSS_SIZE if boss else 48), glow=boss)
    drawn = dict(zip(spec["draw"], sprites))
    skeletons = {v: _skeleton(v, i, count, raw, snapped, s, body)
                 for i, (v, s) in enumerate(zip(spec["draw"], sprites))}
    # Mirrors win over anything drawn for the same view.
    for view, source in _pick_sides(spec["mirror"], drawn).items():
        drawn[view] = ImageOps.mirror(drawn[source])
        w = drawn[source].width
        skeletons[view] = [None if p is None else [round(w - p[0], 1), p[1]]
                           for p in skeletons[source]]
        if swap_sides:
            # Without the pose guide the model may draw both profiles facing
            # the wrong way round.
            drawn[view], drawn[source] = drawn[source], drawn[view]
            skeletons[view], skeletons[source] = skeletons[source], skeletons[view]

    frames = [selout(drawn[v]) for v in spec["order"]]
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    for view, frame in zip(spec["order"], frames):
        frame.save(out / "frames" / f"{view}.png")
    full = sheet(frames)
    full.save(out / "sheet.png")
    preview(full).save(out / "preview.png")
    meta = {"frame": frames[0].width, "order": spec["order"], "character": character,
            "skeleton": {v: skeletons[v] for v in spec["order"]}}
    if creature:
        meta["creature"] = True             # spritehatch-animate hops it instead
    if boss:
        meta["body"] = "boss"               # the redraw prompt drops the chibi words
    (out / "sheet.json").write_text(json.dumps(meta, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="spritehatch", description=__doc__)
    ap.add_argument("character", nargs="?", help='e.g. "little knight, silver armor, red cape"')
    ap.add_argument("-o", "--out", default="out", help="output folder")
    ap.add_argument("-d", "--directions", choices=sorted(MODES), default="4")
    ap.add_argument("--size", type=int, default=0,
                    help="frame size in pixels; 0 = 48 for zimage, the model's pixel size for sdxl")
    ap.add_argument("--colors", type=int, default=32, help="shared palette size")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", choices=["zimage", "sdxl"], default="zimage",
                    help="zimage (Z-Image-Turbo, default, ~1.5 min) or sdxl (~3 min)")
    ap.add_argument("--steps", type=int, default=0, help="0 = the model's default")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--checkpoint", default="sd_xl_base_1.0.safetensors")
    ap.add_argument("--lora", default="pixel-art-xl.safetensors")
    ap.add_argument("--style-lora", default=None,
                    help="zimage only: a LoRA in ComfyUI's models/loras (e.g. one trained on 48px sprites)")
    ap.add_argument("--style-strength", type=float, default=0.5, help="strength of --style-lora")
    ap.add_argument("--pose-strength", type=float, default=0.8,
                    help="zimage: how closely to follow the chibi skeleton; 0 for creatures "
                         "(dragons, slimes) the human skeleton does not fit")
    ap.add_argument("--creature", action="store_true",
                    help="zimage: a creature, not a person (prompt without chibi body words, "
                         "--pose-strength 0, --swap-sides)")
    ap.add_argument("--boss", action="store_true",
                    help="zimage: a big boss (96px, menacing, about 4 heads tall) on a canvas "
                         "twice the size")
    ap.add_argument("--swap-sides", action="store_true",
                    help="swap the left and right views (when they come out facing backwards)")
    ap.add_argument("--controlnet", default="OpenPoseXL2.safetensors")
    ap.add_argument("--from-raw", type=Path,
                    help="skip generation and re-cut an existing raw.png")
    args = ap.parse_args(argv)
    quadruped = args.creature and is_quadruped(args.character)
    if quadruped:
        # A four-legged guide replaces the human skeleton (see
        # pose.quadruped_sheet); profiles drawn facing the wrong way are
        # turned round by build (see face_profiles).
        args.pose_strength = QUADRUPED_STRENGTH
    elif args.creature:
        # Drawn freely, the model turns both profiles to face left, so the
        # sides always come out swapped (dragon, slime, wolf).
        args.pose_strength, args.swap_sides = 0.0, True
    out = Path(args.out)

    if args.from_raw:
        build(Image.open(args.from_raw), args.directions, args.size, args.colors, out,
              args.character, args.model, args.swap_sides, args.creature, args.boss)
        print(f"done: {out / 'sheet.png'}")
        return
    if not args.character:
        ap.error("describe a character, or pass --from-raw")

    spec = MODES[args.directions]
    if args.boss and args.model != "zimage":
        ap.error("--boss needs --model zimage")
    width, height = (BOSS_CANVAS if args.boss else CANVAS[args.model])[args.directions]
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    client = ComfyClient(args.server)
    if quadruped:
        pose = quadruped_sheet(width, height, spec["draw"])
    else:
        pose = pose_sheet(width, height, spec["draw"], "boss" if args.boss else "chibi")
    name = client.upload_image(pose, f"turnaround_pose_{args.directions}.png")
    start = (client.upload_image(quadruped_start(pose), "turnaround_start.png")
             if quadruped else None)
    views = [VIEW_WORDS[v] for v in spec["draw"]]
    if args.model == "zimage":
        workflow = zimage_turnaround_workflow(args.character, name, views, width=width,
                                              height=height, seed=seed, steps=args.steps or 8,
                                              control_strength=args.pose_strength,
                                              creature=args.creature, boss=args.boss,
                                              quadruped=quadruped, init_image=start,
                                              denoise=QUADRUPED_DENOISE,
                                              lora=args.style_lora,
                                              lora_strength=args.style_strength)
    else:
        workflow = turnaround_workflow(
            args.character, name, views, width=width, height=height, seed=seed,
            steps=args.steps or 30, checkpoint=args.checkpoint, lora=args.lora,
            controlnet=args.controlnet)
    print(f"generating {len(spec['draw'])} views (seed {seed}), this takes a few minutes...")
    raw = client.run(workflow)[0]

    out.mkdir(parents=True, exist_ok=True)
    raw.save(out / "raw.png")
    pose.save(out / "pose.png")
    build(raw, args.directions, args.size, args.colors, out, args.character, args.model,
          args.swap_sides, args.creature, args.boss)
    print(f"done: {out / 'sheet.png'}  (seed {seed})")


if __name__ == "__main__":
    main()
