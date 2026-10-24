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
pixel-animate out/knight            # idle + walk
```

![walk](examples/knight/anim/walk_all.gif)

Motions move the finished pixels instead of drawing new ones, so every frame
keeps the exact look and palette of the standing sprite. No GPU needed.

- `idle`: 2 frames, upper body sinks one pixel.
- `walk`: 4 frames, legs step and the body bobs.

Output in `out/knight/anim/`: `idle.png`, `walk.png` (rows: down, left, right,
up), one GIF per direction, and `anim.json` with frame size and timing.

Asking the model to draw walk frames was tried first. With SDXL + OpenPoseXL2
the frames came out faded and the legs barely moved, so it is not included.

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
