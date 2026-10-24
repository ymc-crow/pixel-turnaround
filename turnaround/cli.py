"""pixel-turnaround: one character, every direction, same look."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageOps

from .comfy import ComfyClient, turnaround_workflow
from .pixelate import preview, sheet, snap_to_grid, split_cells, to_sprites
from .pose import MODES, VIEW_WORDS, pose_sheet

# One SDXL-friendly canvas per mode, roughly one megapixel.
CANVAS = {"4": (1536, 640), "8": (1600, 640)}


def build(raw: Image.Image, mode: str, size: int, colors: int, out: Path) -> None:
    spec = MODES[mode]
    cells = split_cells(snap_to_grid(raw), len(spec["draw"]))
    drawn = dict(zip(spec["draw"], to_sprites(cells, size=size, colors=colors)))
    for view, source in spec["mirror"].items():
        drawn[view] = ImageOps.mirror(drawn[source])

    frames = [drawn[v] for v in spec["order"]]
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    for view, frame in zip(spec["order"], frames):
        frame.save(out / "frames" / f"{view}.png")
    full = sheet(frames)
    full.save(out / "sheet.png")
    preview(full).save(out / "preview.png")
    meta = {"frame": frames[0].width, "order": spec["order"]}
    (out / "sheet.json").write_text(json.dumps(meta, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="pixel-turnaround", description=__doc__)
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
        build(Image.open(args.from_raw), args.directions, args.size, args.colors, out)
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
    build(raw, args.directions, args.size, args.colors, out)
    print(f"done: {out / 'sheet.png'}  (seed {seed})")


if __name__ == "__main__":
    main()
