import numpy as np
from PIL import Image

from turnaround.pixelate import background_mask, snap_to_grid, to_sprites
from turnaround.pose import MODES, keypoints, pose_sheet


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


def test_breathe_sinks_upper_body_one_pixel():
    from turnaround.animate import breathe

    sprite = to_sprites([_figure(60)], size=80)[0]
    still, sunk = (np.asarray(f)[..., 3] > 0 for f in breathe(sprite))
    assert np.nonzero(sunk.any(axis=1))[0][0] == np.nonzero(still.any(axis=1))[0][0] + 1
    assert np.nonzero(sunk.any(axis=1))[0][-1] == np.nonzero(still.any(axis=1))[0][-1]


def test_walk_keeps_colors_and_moves_legs():
    from turnaround.animate import walk_front, walk_side

    sprite = to_sprites([_figure(60)], size=80)[0]
    colors = {tuple(c) for c in np.asarray(sprite).reshape(-1, 4) if c[3]}
    for frames in (walk_side(sprite), walk_front(sprite)):
        assert len(frames) == 4
        for f in frames:
            assert {tuple(c) for c in np.asarray(f).reshape(-1, 4) if c[3]} <= colors
        assert not np.array_equal(np.asarray(frames[0]), np.asarray(sprite))
