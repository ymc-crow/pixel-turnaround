"""spritehatch: one character, every direction, same look."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageOps

from .comfy import ComfyClient, turnaround_workflow
from .pixelate import preview, sheet, snap_to_grid, split_cells, to_sprites
from .pose import MODES, VIEW_WORDS, keypoints, pose_sheet

# One SDXL-friendly canvas per mode, roughly one megapixel.
CANVAS = {"4": (1536, 640), "8": (1600, 640)}


def _skeleton(view: str, index: int, count: int, raw: Image.Image,
              snapped: Image.Image, sprite: Image.Image):
    """Where the pose skeleton the model followed ends up on the final sprite."""
    px, ox, py, oy = snapped.info["grid"]
    scale, x0, y0, left, top = sprite.info["place"]
    cell_w = raw.width / count
    cell_x = round(index * snapped.width / count)
    points = []
    for p in keypoints(view):
        if p is None:
            points.append(None)
            continue
        gx = (index * cell_w + p[0] * cell_w - ox) / px - cell_x
        gy = (p[1] * raw.height - oy) / py
        points.append([round((gx - x0) * scale + left, 1), round((gy - y0) * scale + top, 1)])
    return points


def build(raw: Image.Image, mode: str, size: int, colors: int, out: Path,
          character: str | None = None) -> None:
    spec = MODES[mode]
    snapped = snap_to_grid(raw)
    count = len(spec["draw"])
    sprites = to_sprites(split_cells(snapped, count), size=size, colors=colors)
    drawn = dict(zip(spec["draw"], sprites))
    skeletons = {v: _skeleton(v, i, count, raw, snapped, s)
                 for i, (v, s) in enumerate(zip(spec["draw"], sprites))}
    # Mirrors win over anything drawn for the same view.
    for view, source in spec["mirror"].items():
        drawn[view] = ImageOps.mirror(drawn[source])
        w = drawn[source].width
        skeletons[view] = [None if p is None else [round(w - p[0], 1), p[1]]
                           for p in skeletons[source]]

    frames = [drawn[v] for v in spec["order"]]
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    for view, frame in zip(spec["order"], frames):
        frame.save(out / "frames" / f"{view}.png")
    full = sheet(frames)
    full.save(out / "sheet.png")
    preview(full).save(out / "preview.png")
    meta = {"frame": frames[0].width, "order": spec["order"], "character": character,
            "skeleton": {v: skeletons[v] for v in spec["order"]}}
    (out / "sheet.json").write_text(json.dumps(meta, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="spritehatch", description=__doc__)
    ap.add_argument("character", nargs="?", help='e.g. "little knight, silver armor, red cape"')
    ap.add_argument("-o", "--out", default="out", help="output folder")
    ap.add_argument("-d", "--directions", choices=sorted(MODES), default="4")
    ap.add_argument("--size", type=int, default=0,
                    help="frame size in pixels, 0 = keep the model's pixel size")
    ap.add_argument("--colors", type=int, default=24, help="shared palette size")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--checkpoint", default="sd_xl_base_1.0.safetensors")
    ap.add_argument("--lora", default="pixel-art-xl.safetensors")
    ap.add_argument("--controlnet", default="OpenPoseXL2.safetensors")
    ap.add_argument("--from-raw", type=Path,
                    help="skip generation and re-cut an existing raw.png")
    args = ap.parse_args(argv)
    out = Path(args.out)

    if args.from_raw:
        build(Image.open(args.from_raw), args.directions, args.size, args.colors, out,
              args.character)
        print(f"done: {out / 'sheet.png'}")
        return
    if not args.character:
        ap.error("describe a character, or pass --from-raw")

    spec = MODES[args.directions]
    width, height = CANVAS[args.directions]
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    client = ComfyClient(args.server)
    pose = pose_sheet(width, height, spec["draw"])
    name = client.upload_image(pose, f"turnaround_pose_{args.directions}.png")
    workflow = turnaround_workflow(
        args.character, name, [VIEW_WORDS[v] for v in spec["draw"]],
        width=width, height=height, seed=seed, steps=args.steps,
        checkpoint=args.checkpoint, lora=args.lora, controlnet=args.controlnet,
    )
    print(f"generating {len(spec['draw'])} views (seed {seed}), this takes a few minutes...")
    raw = client.run(workflow)[0]

    out.mkdir(parents=True, exist_ok=True)
    raw.save(out / "raw.png")
    pose.save(out / "pose.png")
    build(raw, args.directions, args.size, args.colors, out, args.character)
    print(f"done: {out / 'sheet.png'}  (seed {seed})")


if __name__ == "__main__":
    main()
