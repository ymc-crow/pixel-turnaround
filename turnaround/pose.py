"""Draw OpenPose skeletons for a 4-direction turnaround sheet.

The drawing follows the OpenPose body format (18 COCO keypoints, the same
limb order and colors as comfyui_controlnet_aux), so any SDXL OpenPose
ControlNet can read it.
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

# Views the model draws. Mirrored views are then replaced by a flipped copy of
# their partner, which keeps the two sides identical for free.
MODES = {
    "4": {
        # The left view is drawn too: four poses on the canvas give steadier
        # results than three, even though the right view replaces it later.
        "draw": ["down", "left", "right", "up"],
        "mirror": {"left": "right"},
        "order": ["down", "left", "right", "up"],
    },
    "8": {
        "draw": ["down", "down_right", "right", "up_right", "up"],
        "mirror": {"left": "right", "down_left": "down_right", "up_left": "up_right"},
        "order": ["down", "down_left", "left", "up_left", "up", "up_right", "right", "down_right"],
    },
}
VIEW_WORDS = {
    "down": "front view",
    "down_right": "three-quarter front view",
    "left": "side view facing left",
    "right": "side view facing right",
    "up_right": "three-quarter back view",
    "up": "back view",
}

# 1-based keypoint pairs, same order as the reference implementation.
LIMBS = [
    (2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8), (2, 9), (9, 10),
    (10, 11), (2, 12), (12, 13), (13, 14), (2, 1), (1, 15), (15, 17),
    (1, 16), (16, 18),
]
COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

# Keypoint order: nose, neck, r_shoulder, r_elbow, r_wrist, l_shoulder,
# l_elbow, l_wrist, r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle,
# r_eye, l_eye, r_ear, l_ear. Coordinates are 0..1 inside one cell.
# Proportions are chibi: the head takes about a third of the height.
_FRONT = [
    (0.50, 0.27), (0.50, 0.39), (0.39, 0.41), (0.33, 0.52), (0.31, 0.62),
    (0.61, 0.41), (0.67, 0.52), (0.69, 0.62), (0.44, 0.63), (0.43, 0.76),
    (0.43, 0.89), (0.56, 0.63), (0.57, 0.76), (0.57, 0.89), (0.45, 0.24),
    (0.55, 0.24), (0.39, 0.26), (0.61, 0.26),
]
# Profile facing the right edge: the character's right side is toward us.
_RIGHT = [
    (0.62, 0.28), (0.50, 0.39), (0.47, 0.41), (0.49, 0.52), (0.50, 0.62),
    (0.53, 0.41), (0.55, 0.52), (0.56, 0.62), (0.48, 0.63), (0.46, 0.76),
    (0.44, 0.89), (0.52, 0.63), (0.56, 0.76), (0.57, 0.89), (0.58, 0.24),
    None, (0.47, 0.26), None,
]
# Three-quarter front, turned toward the right edge.
_DOWN_RIGHT = [
    (0.56, 0.28), (0.50, 0.39), (0.42, 0.41), (0.38, 0.52), (0.37, 0.62),
    (0.58, 0.41), (0.61, 0.52), (0.62, 0.62), (0.46, 0.63), (0.45, 0.76),
    (0.45, 0.89), (0.55, 0.63), (0.57, 0.76), (0.58, 0.89), (0.50, 0.24),
    (0.59, 0.24), (0.42, 0.26), None,
]
# Three-quarter back, walking away toward the upper right.
_UP_RIGHT = [
    None, (0.50, 0.39), (0.59, 0.41), (0.62, 0.52), (0.63, 0.62),
    (0.42, 0.41), (0.39, 0.52), (0.38, 0.62), (0.55, 0.63), (0.56, 0.76),
    (0.57, 0.89), (0.45, 0.63), (0.44, 0.76), (0.43, 0.89), None,
    None, (0.60, 0.26), (0.43, 0.26),
]


def _mirror(points):
    """Flip left/right: swap every r_* keypoint with its l_* partner."""
    pairs = {2: 5, 3: 6, 4: 7, 8: 11, 9: 12, 10: 13, 14: 15, 16: 17}
    pairs.update({v: k for k, v in pairs.items()})
    out = [None] * len(points)
    for i, p in enumerate(points):
        out[pairs.get(i, i)] = None if p is None else (1 - p[0], p[1])
    return out


def keypoints(view: str):
    if view == "up":
        # Seen from behind: sides swap and the face disappears.
        back = _mirror(_FRONT)
        back[0] = back[14] = back[15] = None
        return back
    if view == "left":
        return _mirror(_RIGHT)
    table = {"down": _FRONT, "right": _RIGHT, "down_right": _DOWN_RIGHT, "up_right": _UP_RIGHT}
    if view not in table:
        raise ValueError(f"unknown view: {view}")
    return list(table[view])


def _ellipse(cx, cy, half_len, half_w, angle_deg, steps=24):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for i in range(steps):
        t = 2 * math.pi * i / steps
        x, y = half_len * math.cos(t), half_w * math.sin(t)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return pts


def draw_cell(draw: ImageDraw.ImageDraw, points, x0, w, h, stick=6, dot=6):
    px = [None if p is None else (x0 + p[0] * w, p[1] * h) for p in points]
    for (a, b), color in zip(LIMBS, COLORS):
        p1, p2 = px[a - 1], px[b - 1]
        if p1 is None or p2 is None:
            continue
        length = math.dist(p1, p2)
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        dim = tuple(int(c * 0.6) for c in color)
        draw.polygon(_ellipse(*mid, length / 2, stick, angle), fill=dim)
    for p, color in zip(px, COLORS):
        if p is not None:
            draw.ellipse((p[0] - dot, p[1] - dot, p[0] + dot, p[1] + dot), fill=color)


def pose_sheet(width: int, height: int, views) -> Image.Image:
    """Black canvas with one skeleton per view, left to right."""
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    cell = width / len(views)
    for i, d in enumerate(views):
        draw_cell(draw, keypoints(d), i * cell, cell, height)
    return img
