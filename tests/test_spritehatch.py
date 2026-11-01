import numpy as np
from PIL import Image

from spritehatch.pixelate import background_mask, snap_to_grid, to_sprites
from spritehatch.pose import MODES, keypoints, pose_sheet


def test_every_drawn_view_has_a_skeleton():
    for spec in MODES.values():
        for view in spec["draw"]:
            assert len(keypoints(view)) == 18
        img = pose_sheet(640, 320, spec["draw"])
        assert np.asarray(img).max() > 0


def test_back_view_has_no_face():
    back = keypoints("up")
    assert back[0] is None and back[14] is None and back[15] is None


def _figure(height, top=10):
    """White cell with a dark box 'character' that has a white spot inside."""
    a = np.full((120, 80, 3), 255, np.uint8)
    a[top:top + height, 30:50] = (40, 40, 160)
    a[top + 5:top + 8, 38:42] = 255
    return Image.fromarray(a)


def test_background_mask_keeps_white_inside_character():
    mask = background_mask(_figure(60))
    assert mask[0, 0]
    assert not mask[16, 39]


def test_sprites_share_scale_and_feet_line():
    sprites = to_sprites([_figure(100), _figure(50, top=40)], size=32)
    alphas = [np.asarray(s)[..., 3] > 0 for s in sprites]
    heights = [np.nonzero(a.any(axis=1))[0] for a in alphas]
    assert heights[0][-1] == heights[1][-1]
    assert len(heights[0]) > len(heights[1])


def test_snap_to_grid_recovers_fake_pixels():
    rng = np.random.default_rng(0)
    small = rng.integers(0, 255, (20, 30, 3), dtype=np.uint8)
    big = Image.fromarray(small).resize((240, 160), Image.NEAREST)
    snapped = np.asarray(snap_to_grid(big))
    assert snapped.shape[0] >= 18 and snapped.shape[1] >= 28



def _stick_figure(view="down", size=80):
    """A plain sprite drawn along the pose skeleton, plus that skeleton."""
    from PIL import ImageDraw

    from spritehatch.pose import LIMBS, keypoints

    pts = [None if p is None else [p[0] * size, p[1] * size] for p in keypoints(view)]
    img = Image.new("RGBA", (size, size))
    d = ImageDraw.Draw(img)
    for a, b in LIMBS[:12]:
        if pts[a - 1] and pts[b - 1]:
            d.line([tuple(pts[a - 1]), tuple(pts[b - 1])], fill=(200, 60, 60, 255), width=5)
    neck = pts[1]
    d.ellipse((neck[0] - 9, neck[1] - 20, neck[0] + 9, neck[1] - 2), fill=(90, 90, 200, 255))
    return img, pts


def test_rig_parts_cover_the_sprite_and_rest_pose_is_unchanged():
    from spritehatch.rig import Rig

    img, pts = _stick_figure()
    rig = Rig(img, pts, "down")
    alpha = np.asarray(img)[..., 3] > 0
    cover = sum((p[..., 3] > 0).astype(int) for n, p in rig.parts.items() if n != "torso")
    torso = rig.parts["torso"][..., 3] > 0
    assert ((cover + torso) >= alpha).all()
    still, _ = rig.render({})
    assert np.array_equal(np.asarray(still), np.asarray(img))


def test_rig_bends_joints_and_keeps_feet_on_floor():
    from spritehatch.rig import Rig

    img, pts = _stick_figure("right")
    rig = Rig(img, pts, "right")
    moved, _ = rig.render({"leg_r": 30, "foot_r": -40, "arm_r": 90})
    a = np.asarray(moved)[..., 3] > 0
    assert not np.array_equal(a, np.asarray(img)[..., 3] > 0)
    assert np.nonzero(a.any(axis=1))[0][-1] == rig.floor


def test_every_motion_renders_in_every_direction():
    from spritehatch.moves import MOTIONS, canvas_size, render

    img, pts = _stick_figure()
    side, side_pts = _stick_figure("right")
    for name, (_, _, big) in MOTIONS.items():
        for d in ("down", "left", "right", "up"):
            still, skel = (side, side_pts) if d in ("left", "right") else (img, pts)
            frames = render(still, skel, d, name)
            assert len(frames) >= 4
            size = canvas_size(80) if big else 80
            assert all(f.size == (size, size) for f in frames)
    lying = render(side, side_pts, "right", "death")[-1]
    ys, xs = np.nonzero(np.asarray(lying)[..., 3] > 0)
    assert np.ptp(xs) > np.ptp(ys)
