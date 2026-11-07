"""Render the four-legged depth guides in spritehatch/guides/ from 3D models.

    uv pip install trimesh
    python tools/make_quadruped_guides.py wolf.glb lean      # front and back straight on
    python tools/make_quadruped_guides.py bull.glb heavy

Models: Quaternius "Ultimate Animated Animals" (CC0), e.g. from
https://github.com/danwahl/animasim/tree/main/assets/generated/glb
Each view is an orthographic depth image (nearer is brighter) seen from above,
the way game sprites show animals, all at one scale, feet on the bottom
edge. Y is up and the model faces +Z.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

SIZE = 512
# (yaw, pitch). Straight from the front a four-legged animal shows two legs
# (hind legs hide behind the front ones) and the model stands it up; from
# 45 degrees above, the back rises up the picture and it sits up. Turned a
# little, as game sprites draw animals, the body and all four legs show.
VIEWS = {"down": (25, 30), "right": (90, 20), "left": (-90, 20), "up": (205, 30)}
# A slim animal turned like that reads as looking back over its shoulder (front
# and back swap), and straight on it does stand on four legs, so lean bodies
# keep the front and back straight.
STRAIGHT = {"down": (0, 30), "right": (90, 20), "left": (-90, 20), "up": (180, 30)}


def _project(vertices: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    y, p = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return vertices @ ry.T @ rx.T            # looking down -Z: bigger z is nearer


def _rasterize(pts: np.ndarray, faces: np.ndarray, size: int) -> np.ndarray:
    zbuf = np.full((size, size), -np.inf)
    px, py, z = pts[:, 0], pts[:, 1], pts[:, 2]
    for f in faces:
        x0, x1 = max(int(px[f].min()), 0), min(int(np.ceil(px[f].max())), size - 1)
        y0, y1 = max(int(py[f].min()), 0), min(int(np.ceil(py[f].max())), size - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        a, b, c = np.stack([px[f], py[f]], 1)
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-9:
            continue
        l1 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / den
        l2 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / den
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-3) & (l2 >= -1e-3) & (l3 >= -1e-3)
        zz = l1 * z[f[0]] + l2 * z[f[1]] + l3 * z[f[2]]
        sub = zbuf[gy, gx]
        hit = inside & (zz > sub)
        sub[hit] = zz[hit]
        zbuf[gy, gx] = sub
    return zbuf


def render(mesh: trimesh.Trimesh, angles: dict = VIEWS) -> dict[str, Image.Image]:
    views = {v: _project(mesh.vertices, yaw, pitch) for v, (yaw, pitch) in angles.items()}
    # One scale for every view: the longest extent (the profile) fills the image.
    span = max(np.ptp(p[:, :2], axis=0).max() for p in views.values())
    s = (SIZE - 4) / span
    out = {}
    for v, p in views.items():
        q = p.copy()
        q[:, 0] = (p[:, 0] - (p[:, 0].min() + p[:, 0].max()) / 2) * s + SIZE / 2
        q[:, 1] = SIZE - 2 - (p[:, 1] - p[:, 1].min()) * s     # lowest point on the floor
        zbuf = _rasterize(q, mesh.faces, SIZE)
        solid = np.isfinite(zbuf)
        img = np.zeros((SIZE, SIZE), np.uint8)
        z = zbuf[solid]
        img[solid] = (70 + 185 * (z - z.min()) / max(np.ptp(z), 1e-6)).astype(np.uint8)
        out[v] = Image.fromarray(img)
    return out


if __name__ == "__main__":
    model, name = sys.argv[1], sys.argv[2]
    folder = Path(__file__).resolve().parent.parent / "spritehatch" / "guides"
    angles = STRAIGHT if name == "lean" else VIEWS
    for view, img in render(trimesh.load(model, force="mesh"), angles).items():
        img.save(folder / f"{name}_{view}.png", optimize=True)
        print(folder / f"{name}_{view}.png")
