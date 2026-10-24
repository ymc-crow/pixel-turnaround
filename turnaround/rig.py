"""A 2D bone rig for one pixel sprite.

The sprite is cut into body parts (head, torso, upper and lower arms and legs)
using the pose skeleton the model was asked to follow. A pose rotates the
parts around their joints, parents first, the same way 2D game tools
(Spine, DragonBones) do it, and the parts are drawn back with plain nearest
pixel sampling so the art stays crisp.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# name, parent, start point, end point, thickness as a share of body height.
# Numbers are OpenPose keypoints; strings are points worked out below.
BONES = [
    ("torso", None, "hip", 1, 0.15),
    ("head", "torso", 1, "top", 0.24),
    ("arm_r", "torso", 2, 3, 0.06),
    ("hand_r", "arm_r", 3, "wrist_r", 0.06),
    ("arm_l", "torso", 5, 6, 0.06),
    ("hand_l", "arm_l", 6, "wrist_l", 0.06),
    ("leg_r", None, 8, 9, 0.065),
    ("foot_r", "leg_r", 9, "sole_r", 0.07),
    ("leg_l", None, 11, 12, 0.065),
    ("foot_l", "leg_l", 12, "sole_l", 0.07),
]
NAMES = [b[0] for b in BONES]
LIMBS = [n for n in NAMES if n not in ("torso", "head")]

# Back to front. Profiles face right: the character's right side is near us.
DRAW_ORDER = {
    "right": ["arm_l", "hand_l", "leg_l", "foot_l", "torso", "head",
              "leg_r", "foot_r", "arm_r", "hand_r"],
    "down": ["leg_r", "foot_r", "leg_l", "foot_l", "torso",
             "arm_r", "hand_r", "arm_l", "hand_l", "head"],
    "up": ["leg_r", "foot_r", "leg_l", "foot_l", "arm_r", "hand_r",
           "arm_l", "hand_l", "torso", "head"],
}


def _rot(deg: float, pivot) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    px, py = pivot
    return np.array([[c, -s, px - c * px + s * py],
                     [s, c, py - s * px - c * py],
                     [0, 0, 1]])


def _move(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=float)


def _seg_dist(p: np.ndarray, a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-6), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _fill(tex: np.ndarray, holes: np.ndarray, steps: int = 12) -> None:
    """Paint `holes` with the colors of neighboring opaque pixels, in place."""
    todo = holes & (tex[..., 3] == 0)
    for _ in range(steps):
        if not todo.any():
            return
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            src = np.roll(tex, (dy, dx), axis=(0, 1))
            take = todo & (src[..., 3] > 0)
            tex[take] = src[take]
            todo &= ~take


class Rig:
    def __init__(self, sprite: Image.Image, skeleton, view: str):
        """sprite: RGBA image; skeleton: 18 [x, y] points (or None) on it.

        view is the drawn view: right, down or up (left = mirrored right).
        """
        self.view = view
        self.size = sprite.size
        a = np.asarray(sprite.convert("RGBA"))
        ys, xs = np.nonzero(a[..., 3] > 0)
        self.top, self.floor = ys.min(), ys.max()
        self.height = self.floor - self.top + 1
        self.pts = self._points(skeleton, xs)
        self.cx = self.pts["hip"][0]

        pix = np.stack([xs + 0.5, ys + 0.5], axis=1)
        scores = []
        for name, _, s, e, r in BONES:
            d = _seg_dist(pix, self.pts[s], self.pts[e]) / (r * self.height)
            above_neck = pix[:, 1] < self.pts[1][1]
            if name == "head":
                d = np.where(above_neck, 0.0, np.inf)
            else:
                d = np.where(above_neck, np.inf, d)
            scores.append(d)
        scores = np.stack(scores)
        label = scores.argmin(axis=0)

        self.parts = {}
        for i, name in enumerate(NAMES):
            tex = np.zeros_like(a)
            sel = label == i
            tex[ys[sel], xs[sel]] = a[ys[sel], xs[sel]]
            self.parts[name] = tex
        # Where a limb lay over the torso, the torso needs pixels of its own,
        # or a moving arm would leave a hole behind.
        under = np.zeros(a.shape[:2], bool)
        ti = NAMES.index("torso")
        inside = (scores[ti] < 1.0) & (label != ti) & (label != NAMES.index("head"))
        under[ys[inside], xs[inside]] = True
        _fill(self.parts["torso"], under)

    def _points(self, skeleton, xs):
        p = {i: None if q is None else np.array(q, float) for i, q in enumerate(skeleton)}
        p["hip"] = (p[8] + p[11]) / 2
        face = [p[i] for i in (0, 14, 15, 16, 17) if p[i] is not None]
        head_x = np.mean([f[0] for f in face]) if face else p[1][0]
        p["top"] = np.array([head_x, self.top + (p[1][1] - self.top) * 0.3])
        for side, (e, w) in {"r": (3, 4), "l": (6, 7)}.items():
            p[f"wrist_{side}"] = p[w] + (p[w] - p[e]) * 0.4
        for side, (k, an) in {"r": (9, 10), "l": (12, 13)}.items():
            p[f"sole_{side}"] = np.array([p[an][0], max(p[an][1], self.floor)])
        return p

    # --- poses --------------------------------------------------------------

    def angles(self, pose: dict) -> dict:
        """Turn friendly pose numbers into screen rotations.

        Profiles: degrees toward the facing side (arms forward, torso leaning
        in). Front/back: degrees away from the body's middle (arms out).
        """
        out = {}
        for name, _, s, e, _ in BONES:
            v = pose.get(name, 0)
            if not v:
                continue
            start, end = self.pts[s], self.pts[e]
            down = end[1] > start[1]
            if self.view == "right":
                out[name] = -v if down else v
            elif name in ("torso", "head"):
                out[name] = v
            else:
                left_side = (start[0] + end[0]) / 2 < self.cx
                out[name] = v * (1 if left_side else -1) * (1 if down else -1)
        return out

    def matrices(self, pose: dict) -> dict:
        rot = self.angles(pose)
        shift = pose.get("shift", {})
        root = _move(*pose.get("root", (0, 0)))
        world = {}
        for name, parent, s, _, _ in BONES:
            local = _rot(rot.get(name, 0), self.pts[s]) @ _move(*shift.get(name, (0, 0)))
            world[name] = (world[parent] if parent else root) @ local
        if pose.get("ground", True):
            # Keep the lowest foot on the floor, like a real step.
            low = max(self._lowest(world, f) for f in ("foot_r", "foot_l"))
            fix = _move(0, self.floor - low)
            world = {k: fix @ m for k, m in world.items()}
        return world

    def _lowest(self, world, part) -> float:
        ys, xs = np.nonzero(self.parts[part][..., 3] > 0)
        if len(ys) == 0:
            return self.floor
        pts = world[part] @ np.stack([xs + 0.5, ys + 0.5, np.ones(len(xs))])
        return pts[1].max() - 0.5

    def point(self, world: dict, bone: str, key) -> np.ndarray:
        x, y = self.pts[key]
        return (world[bone] @ np.array([x, y, 1.0]))[:2]

    def render(self, pose: dict):
        """Draw the pose. Returns the image and the bone matrices used."""
        world = self.matrices(pose)
        w, h = self.size
        out = np.zeros((h, w, 4), np.uint8)
        for name in DRAW_ORDER[self.view]:
            tex = self.parts[name]
            ys, xs = np.nonzero(tex[..., 3] > 0)
            if len(ys) == 0:
                continue
            m = world[name]
            corners = m @ np.array([[xs.min(), xs.max() + 1, xs.min(), xs.max() + 1],
                                    [ys.min(), ys.min(), ys.max() + 1, ys.max() + 1],
                                    [1, 1, 1, 1]])
            x0, y0 = np.floor(corners[:2].min(axis=1)).astype(int) - 1
            x1, y1 = np.ceil(corners[:2].max(axis=1)).astype(int) + 1
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if x0 >= x1 or y0 >= y1:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            src = np.linalg.inv(m) @ np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
            sx, sy = np.floor(src[0]).astype(int), np.floor(src[1]).astype(int)
            ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
            tx, ty = gx.ravel()[ok].astype(int), gy.ravel()[ok].astype(int)
            color = tex[sy[ok], sx[ok]]
            vis = color[:, 3] > 0
            out[ty[vis], tx[vis]] = color[vis]
        return Image.fromarray(out), world
