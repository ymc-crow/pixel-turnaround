"""Tiny ComfyUI HTTP client and the turnaround workflow."""
from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.request
import uuid

from PIL import Image

PROMPT_TEMPLATE = (
    "pixel art, character turnaround sheet, the same character shown {count} times "
    "in a row, {views}, "
    "{character}, chibi, big head, full body, standing, identical outfit and colors "
    "in every view, plain white background, game sprite, clean pixels"
)
NEGATIVE = (
    "different characters, inconsistent outfit, inconsistent colors, extra limbs, "
    "extra characters, blurry, 3d render, realistic, photo, text, watermark, "
    "shadow, scenery, frame, border"
)


class ComfyClient:
    def __init__(self, server: str = "http://127.0.0.1:8188", timeout: float = 900):
        self.server = server.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(self.server + path, timeout=30) as r:
            return r.read()

    def _post(self, path: str, body: bytes, content_type: str) -> dict:
        req = urllib.request.Request(
            self.server + path, data=body, headers={"Content-Type": content_type}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def upload_image(self, img: Image.Image, name: str) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        boundary = uuid.uuid4().hex
        body = b"".join([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'.encode(),
            b"Content-Type: image/png\r\n\r\n",
            buf.getvalue(),
            f"\r\n--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue',
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        res = self._post("/upload/image", body, f"multipart/form-data; boundary={boundary}")
        return res["name"]

    def run(self, workflow: dict) -> list[Image.Image]:
        body = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode()
        prompt_id = self._post("/prompt", body, "application/json")["prompt_id"]
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            hist = json.loads(self._get(f"/history/{prompt_id}"))
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI error: {status.get('messages')}")
                return [self._fetch(img) for out in entry["outputs"].values()
                        for img in out.get("images", [])]
            time.sleep(2)
        raise TimeoutError(f"ComfyUI did not finish within {self.timeout}s")

    def _fetch(self, meta: dict) -> Image.Image:
        q = urllib.parse.urlencode(
            {"filename": meta["filename"], "subfolder": meta["subfolder"], "type": meta["type"]}
        )
        return Image.open(io.BytesIO(self._get(f"/view?{q}"))).convert("RGB")


def turnaround_workflow(
    character: str,
    pose_image: str,
    views: list[str],
    *,
    width: int = 1536,
    height: int = 640,
    seed: int = 0,
    steps: int = 30,
    cfg: float = 7.0,
    checkpoint: str = "sd_xl_base_1.0.safetensors",
    lora: str = "pixel-art-xl.safetensors",
    lora_strength: float = 1.2,
    controlnet: str = "OpenPoseXL2.safetensors",
    control_strength: float = 0.85,
) -> dict:
    """SDXL + pixel LoRA + OpenPose ControlNet, one wide image for all views."""
    return {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "lora": {"class_type": "LoraLoader", "inputs": {
            "lora_name": lora, "strength_model": lora_strength, "strength_clip": 1.0,
            "model": ["ckpt", 0], "clip": ["ckpt", 1]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {
            "text": PROMPT_TEMPLATE.format(
                character=character, count=len(views), views=", ".join(views)), "clip": ["lora", 1]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["lora", 1]}},
        "pose": {"class_type": "LoadImage", "inputs": {"image": pose_image}},
        "cn": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": controlnet}},
        "apply": {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["pos", 0], "negative": ["neg", 0], "control_net": ["cn", 0],
            "image": ["pose", 0], "strength": control_strength,
            "start_percent": 0.0, "end_percent": 0.8}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {
            "width": width, "height": height, "batch_size": 1}},
        "sample": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler_ancestral",
            "scheduler": "normal", "denoise": 1.0, "model": ["lora", 0],
            "positive": ["apply", 0], "negative": ["apply", 1], "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "SaveImage", "inputs": {
            "filename_prefix": "turnaround", "images": ["decode", 0]}},
    }
