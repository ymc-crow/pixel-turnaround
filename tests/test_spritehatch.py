import numpy as np
from PIL import Image

from spritehatch.pixelate import background_mask, downscale_sprites, snap_to_grid, to_sprites
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


def test_downscale_keeps_scale_feet_line_eyes_and_adds_outline():
    big = _figure(100)
    a = np.asarray(big).copy()
    a[30:34, 34:37] = (10, 10, 10)          # a small dark "eye" on the body
    sprites = downscale_sprites([Image.fromarray(a), _figure(50, top=40)], size=32, colors=8)
    assert all(s.size == (32, 32) for s in sprites)
    rgba = [np.asarray(s) for s in sprites]
    bottoms = [np.nonzero((r[..., 3] > 0).any(axis=1))[0][-1] for r in rgba]
    assert bottoms[0] == bottoms[1]
    first = rgba[0]
    solid = first[..., 3] > 0
    left_col = np.nonzero(solid.any(axis=0))[0][0]
    ys = np.nonzero(solid[:, left_col])[0]
    assert (first[ys, left_col, :3].sum(axis=1) < 150).all()   # dark outline on the left edge
    lum = first[..., :3].astype(int).sum(axis=2)
    right_col = np.nonzero(solid.any(axis=0))[0][-1]
    inside = ((lum < 100) & solid)[ys[0] + 1:ys[-1], left_col + 1:right_col]
    assert inside.any()                                        # the eye survived the shrink


def test_zimage_turnaround_is_text_to_image():
    from spritehatch.comfy import zimage_turnaround_workflow

    wf = zimage_turnaround_workflow("knight", "pose.png", ["front view"], seed=3)
    assert wf["sample"]["inputs"]["denoise"] == 1.0
    assert wf["enc"]["class_type"] == "EmptySD3LatentImage"
    assert "knight" in wf["pos"]["inputs"]["text"]


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


def test_render_reports_posed_keypoints_that_follow_the_limbs():
    from spritehatch.moves import render

    side, side_pts = _stick_figure("right")
    pts: list = []
    frames = render(side, side_pts, "right", "walk", keypoints=pts)
    assert len(pts) == len(frames)
    ankles = {round(p[10][0]) for p in pts}
    assert len(ankles) > 1          # the right ankle moves through the stride
    assert all(len(p) == 18 for p in pts)
    dead: list = []
    render(side, side_pts, "right", "death", keypoints=dead)
    neck, ankle = dead[-1][1], dead[-1][10]
    assert abs(neck[1] - ankle[1]) < abs(neck[0] - ankle[0])   # the skeleton lies flat too


class _FakeComfy:
    """Stands in for ComfyUI: 'paints' by returning the draft unchanged."""

    def __init__(self):
        self.uploads, self.runs = {}, 0

    def upload_image(self, img, name):
        self.uploads[name] = img
        return name

    def run(self, wf):
        self.runs += 1
        assert wf["sample"]["inputs"]["latent_image"] == ["enc", 0]
        return [self.uploads[wf["init"]["inputs"]["image"]]]


def test_redraw_paints_four_frames_per_grid_and_keeps_size_and_palette():
    from spritehatch.moves import render
    from spritehatch.redraw import build_grids, cell_size, redraw

    side, side_pts = _stick_figure("right")
    pts: list = []
    frames = render(side, side_pts, "right", "walk", keypoints=pts)
    draft, pose = build_grids(frames[:4], pts[:4])
    assert draft.size == pose.size == (cell_size(80) * 2,) * 2
    assert np.asarray(pose).any()

    fake = _FakeComfy()
    out = redraw(frames, pts, still=side, character="knight", view="right",
                 motion="walk", client=fake)
    assert fake.runs == 2           # 8 frames -> two 2x2 grids
    assert [f.size for f in out] == [f.size for f in frames]
    src = {tuple(c) for c in np.asarray(side)[np.asarray(side)[..., 3] > 0][:, :3]}
    for f in out:
        a = np.asarray(f)
        used = {tuple(c) for c in a[a[..., 3] > 0][:, :3]}
        assert used and used <= src  # only colors of the original sprite


def test_every_nth_picks_evenly_spaced_frames():
    from spritehatch.redraw import every_nth

    assert every_nth(list(range(8)), 4) == [0, 2, 4, 6]
    assert every_nth([1, 2, 3], 4) == [1, 2, 3]
    assert every_nth(list(range(8)), 4, keep_last=True) == [0, 2, 5, 7]


def test_redraw_runs_on_every_model_workflow():
    from spritehatch.comfy import WORKFLOWS
    from spritehatch.moves import render
    from spritehatch.redraw import redraw

    side, side_pts = _stick_figure("right")
    pts: list = []
    frames = render(side, side_pts, "right", "idle", keypoints=pts)
    for model in WORKFLOWS:
        out = redraw(frames, pts, still=side, character="knight", view="right",
                     motion="idle", client=_FakeComfy(), model=model)
        assert len(out) == len(frames)


def test_mirror_keeps_the_side_view_that_drew_more():
    from spritehatch.cli import _pick_sides

    thin = Image.new("RGBA", (8, 8))
    thin.putpixel((1, 1), (255, 0, 0, 255))
    full = Image.new("RGBA", (8, 8), (0, 0, 255, 255))
    assert _pick_sides({"left": "right"}, {"left": thin, "right": full}) == {"left": "right"}
    assert _pick_sides({"left": "right"}, {"left": full, "right": thin}) == {"right": "left"}
    assert _pick_sides({"left": "right"}, {"right": thin}) == {"left": "right"}


def test_open_holes_frees_trapped_background_but_keeps_eye_whites():
    from spritehatch.pixelate import open_holes

    a = np.full((100, 100, 3), 255, np.uint8)
    a[20:80, 20:80] = (40, 40, 160)          # body
    a[30:60, 30:50] = 255                    # big gap caught inside
    a[25:27, 60:62] = 255                    # tiny eye white
    cell = Image.fromarray(a)
    fg = open_holes(cell, ~background_mask(cell))
    assert not fg[40, 40]
    assert fg[25, 60]
    assert fg[70, 70]


def test_redraw_keeps_rig_frames_it_cannot_guide_or_lost():
    from spritehatch.moves import render
    from spritehatch.redraw import redraw

    side, side_pts = _stick_figure("right")
    pts: list = []
    frames = render(side, side_pts, "right", "death", keypoints=pts)[-4:]
    pts = pts[-4:]

    class Blank(_FakeComfy):
        def run(self, wf):
            self.runs += 1
            img = self.uploads[wf["init"]["inputs"]["image"]]
            return [Image.new("RGB", img.size, "white")]    # paints nothing

    out = redraw(frames, pts, still=side, character="knight", view="right",
                 motion="death", client=Blank())
    assert [np.asarray(a).tobytes() for a in out] == [np.asarray(b).tobytes() for b in frames]


def test_animate_adds_motions_to_an_earlier_run(tmp_path):
    import json

    from spritehatch.animate import ORDER, main

    (tmp_path / "frames").mkdir()
    skeleton = {}
    for d in ORDER:
        img, pts = _stick_figure("right" if d == "left" else d)
        img.save(tmp_path / "frames" / f"{d}.png")
        skeleton[d] = pts
    (tmp_path / "sheet.json").write_text(json.dumps({"order": ORDER, "skeleton": skeleton}))
    main([str(tmp_path), "--motions", "idle"])
    main([str(tmp_path), "--motions", "walk"])
    motions = json.loads((tmp_path / "anim" / "anim.json").read_text())["motions"]
    assert set(motions) == {"idle", "walk"}


def test_cells_keep_the_body_the_same_size_in_bigger_frames():
    from spritehatch.redraw import GRID_MAX, cell_size

    walk, attack = cell_size(48, 48), cell_size(72, 48)
    assert walk % 48 == 0 and attack % 72 == 0
    assert attack / 72 >= cell_size(72) / 72   # never smaller than without the body hint
    assert 2 * attack <= GRID_MAX         # but the grid stays small enough to fit in memory


def test_showcase_has_a_row_per_motion(tmp_path):
    import json

    from spritehatch.animate import ORDER, main
    from spritehatch.showcase import showcase

    (tmp_path / "frames").mkdir()
    skeleton = {}
    for d in ORDER:
        img, pts = _stick_figure("right" if d == "left" else d)
        img.save(tmp_path / "frames" / f"{d}.png")
        skeleton[d] = pts
    (tmp_path / "sheet.json").write_text(json.dumps({"order": ORDER, "skeleton": skeleton}))
    main([str(tmp_path), "--motions", "idle,death"])
    frames = showcase(tmp_path / "anim", zoom=1)
    big = 120                                # death's frame, 1.5x the 80px sprite
    assert frames[0].size == (big * 3, big * 2)


def test_scale2x_doubles_and_smooths_a_diagonal():
    from spritehatch.effects import scale2x

    a = np.zeros((2, 2, 4), np.uint8)
    a[0, 0] = a[1, 1] = (255, 0, 0, 255)       # a diagonal line
    b = scale2x(np.ascontiguousarray(a))
    assert b.shape == (4, 4, 4)
    red = b[..., 0] == 255
    assert red[1, 2] and red[2, 1]             # the gap between the blocks is filled
    assert not red[0, 3] and not red[3, 0]     # the far corners stay empty


def test_zimage_uses_no_lora_unless_asked():
    from spritehatch.comfy import zimage_redraw_workflow

    plain = zimage_redraw_workflow("knight", "a.png", "b.png")
    assert "lora" not in plain and plain["shift"]["inputs"]["model"] == ["unet", 0]
    styled = zimage_redraw_workflow("knight", "a.png", "b.png", lora="pixel.safetensors")
    assert styled["shift"]["inputs"]["model"] == ["lora", 0]
