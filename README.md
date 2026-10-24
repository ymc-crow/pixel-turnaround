# pixel-turnaround

One pixel-art character, every direction, **same look**.

Plain text-to-image draws each direction as a separate picture, so the front,
side and back come out as different characters. pixel-turnaround fixes that:

1. **One canvas, all views.** The model paints every direction in a single
   image, so outfit and colors stay in sync.
2. **Pose lock.** An OpenPose skeleton per view tells the model which way to face.
3. **Mirror, don't redraw.** Left is a flipped right, so the two sides always match.
4. **Real pixels.** The model's fake 8×8 "pixels" are snapped back to a true grid,
   the background is removed, all views share one palette, and feet sit on the
   same line.

![knight](examples/knight/preview.png)

Raw model output for that sheet: [examples/knight/raw.png](examples/knight/raw.png)

## Motions

```bash
pixel-animate out/knight            # idle, walk, attack, cast, death
pixel-animate out/knight --motions attack,death
```

![motions](examples/knight/anim/showcase.gif)

The model was asked to follow a pose skeleton, and that skeleton is saved in
`sheet.json`. `pixel-animate` uses it as a 2D bone rig, the way Spine or
DragonBones work:

1. The standing sprite is cut into head, torso, pelvis, upper/lower arms and legs,
   each pixel going to the nearest bone.
2. Where an arm covered the torso, the torso gets filled-in pixels so a
   moving arm does not leave a hole.
3. Each motion is a list of joint angles. Parts rotate around their joints,
   parents first, and are drawn back with nearest-pixel sampling, so the art
   stays crisp and keeps its palette. Feet are kept on the floor, and cracks
   that open at bent joints are filled.
4. Sword, slash trail and magic are drawn on top as plain pixels.

No GPU needed, it takes a second.

| motion | frames | what happens |
| --- | --- | --- |
| `idle` | 4 | breathing, arms sway a little |
| `walk` | 8 | legs swing with bending knees, pelvis rocks, arms swing opposite |
| `attack` | 8 | wind-up, swing with a slash trail, follow-through |
| `cast` | 8 | arms rise, an orb grows over the head and is thrown |
| `death` | 8 | hit flash, staggers back, falls flat, lies still |

`attack`, `cast` and `death` use a 1.5x bigger frame so the sword and the fall
fit. Output in `out/knight/anim/`: one sheet per motion (rows: down, left,
right, up), one GIF per motion and direction, and `anim.json` with frame sizes
and timing. Poses live in `turnaround/moves.py`; add your own there.

Limits: parts are cut from a single drawing, so an arm hidden behind the body
in the side view stays thin, and a cape moves with the torso.

Asking the model to draw the frames was tried first. With SDXL + OpenPoseXL2
it ignored arm poses, changed the outfit between frames, and filled-in frames
came out faded, so that path is not included.

## Requirements

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally (default `http://127.0.0.1:8188`)
- Models in ComfyUI:
  - `checkpoints/sd_xl_base_1.0.safetensors` — [SDXL base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
  - `loras/pixel-art-xl.safetensors` — [Pixel Art XL](https://huggingface.co/nerijs/pixel-art-xl)
  - `controlnet/OpenPoseXL2.safetensors` — [OpenPoseXL2](https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0)
- Python 3.9+

Any SDXL checkpoint / pixel LoRA / OpenPose ControlNet works; pass `--checkpoint`, `--lora`, `--controlnet`.

**Hardware:** anything that runs SDXL. On an Apple M1 Pro (32 GB) one sheet takes about 3.5 minutes.

## Usage

```bash
pip install -e .
pixel-turnaround "cute little knight, silver armor, red cape" -o out/knight
pixel-animate out/knight
```

Output in `out/knight/`:

| file | what |
| --- | --- |
| `sheet.png` | all frames in a row, transparent background |
| `frames/<direction>.png` | one file per direction |
| `sheet.json` | frame size and direction order |
| `preview.png` | sheet scaled ×8 for a quick look |
| `raw.png`, `pose.png` | model output and the skeleton it followed |

Options:

- `-d 4` (default): `down, left, right, up`
- `-d 8` (**experimental**): adds diagonals. SDXL often turns three-quarter views
  into plain side views, so expect to try a few seeds.
- `--size 48`: shrink to a fixed frame. Default `0` keeps the model's own pixel
  size, which looks cleanest.
- `--colors 24`: shared palette size.
- `--seed N`: repeat a result.
- `--from-raw raw.png`: re-cut an existing image without generating again.

## Limits

- Top-down or bottom-up camera angles and anything beyond 8 directions need a
  real 3D step (image → 3D model → render), which this tool does not do.
- The model sometimes mixes small details between views (a plume here, a belt
  there). Re-roll the seed.

## License

MIT. Generated images follow the licenses of the models you use.
