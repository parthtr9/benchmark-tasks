"""Scripted full-score demo for the vise task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The right arm picks the block up by its upper 25 mm (fingers closing along the jaws' length), carries it over the vise and lowers it
between the jaws until its top stands 35 mm above their tops, hard against the fixed jaw's side. The left arm, fingers
closed and its hand rolled 50 degrees away from the right hand, lowers outside the push tab on the sliding jaw and pushes the tab toward the block, a fixed 55 mm
at 15 mm/s and a 1 s hold (open loop, from the layout), which closes the gap to the block's width and takes the preload. The left arm backs off and
lifts away, then the right arm opens and lifts away: the clamp holds the block.

Negative variants (DEMO_VARIANT=...):
  release_early  the right arm lets go as soon as the block is between the jaws: it drops to the base (0.25)
  no_close       the sliding jaw is never pushed; the right arm lets go and the block drops (0.25)
  one_arm        only the left arm works: it closes the jaw on nothing and the block stays on the table (0)
  high           the block is held with its top 60 mm above the jaw tops: clamped, but outside the band (0.5)
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
ROT_WEIGHT = 1.0          # keep the fingers square to the block even when the wrist is near a limit


def _rot_axis(axis, angle) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each starting where the last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.err_mm = 0.0

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60, rot_weight=ROT_WEIGHT)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.err_mm = pe * 1000
        self.worst_mm = max(self.worst_mm, self.err_mm)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None):
        """Generator: go to ``pos``/``rot`` (world TCP); ``grip`` 1 open .. 0 closed (None keeps it)."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def set_grip(self, grip: float, duration: float = 0.8):
        yield from self.move(self.pos, self.rot, duration, grip)

    def aim(self, pos, rot):
        """Track a pose this step (for closed-loop moves)."""
        self._solve(pos, rot)


PUSH = 0.055          # fixed push distance (m) of the left fingers, from the layout


class Layout:
    """A part's pose from the task file's fixed layout (no simulator reading)."""

    def __init__(self, xy, yaw, z):
        self.pos = np.array([xy[0], xy[1], z], float)
        self._r = _rot_axis((0, 0, 1), yaw)

    def rotation(self):
        return self._r

    def to_world(self, p):
        return self.pos + self._r @ np.asarray(p, float)


def reachable_yaw(arm, pos, yaw):
    """The grip yaw, or the same parallel grip turned 180 degrees, whichever the arm reaches more exactly. A wrist at
    its joint limit otherwise leaves the fingers several degrees off and they twist light parts as they close."""
    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    best = None
    for y in (yaw, yaw + math.pi, yaw - math.pi):
        _, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(grip_rotation(y)), arm.q, base_pos=arm.base, iters=200)
        if best is None or pe + 0.05 * re < best[0]:
            best = (pe + 0.05 * re, y)
    return best[1]


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def both(*gens):
    gens = [iter(g) for g in gens]
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        if gens:
            yield


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    T = task
    L, R = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in ("", "release_early", "no_close", "high", "one_arm"):
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    vise, block = Layout(T.VISE_XY, T.VISE_YAW, T.TOP), Layout(T.BLOCK_START, T.BLOCK_YAW, T.TOP)   # the fixed layout
    up = np.array([0, 0, 0.05])
    W = lambda x, y, z: np.asarray(vise.to_world((x, y, z)), float)    # a point in the vise frame, in the world
    inward = vise.rotation()[:, 1]                    # the vise's y axis: from the fixed jaw toward the sliding one

    # Right arm: the block's upper 25 mm, fingers closing along the jaws' length (pad tips 25 mm below the top).
    # The vise stands turned 90 degrees: its gap opens toward the right arm, and its sliding jaw faces the arms.
    over = 0.060 if variant == "high" else 0.035
    grip_z = lambda top: top + 0.001            # TCP z for a block whose top is at ``top``
    g0 = np.asarray(block.to_world((0.0, 0.0, T.BLOCK_Z)), float)
    g0[2] = grip_z(g0[2])
    yaw_r = reachable_yaw(R, g0, 0.0)
    rot_r = grip_rotation(yaw_r)
    hold = W(0.0, T.FIXED_FACE + T.BLOCK_Y / 2 + 0.0005, 0.0)
    hold[2] = grip_z(T.TOP + T.JAW_TOP + over)
    mid = W(0.0, 0.5 * (T.FIXED_FACE + T.SLIDE_FACE), 0.0)
    mid[2] = hold[2]
    pre_in = W(-T.JAW_HALF_X - 0.060, 0.5 * (T.FIXED_FACE + T.SLIDE_FACE), 0.0)   # the gap's open end, nearest the right arm
    pre_in[2] = hold[2] + 0.020                      # the block's bottom 65 mm over the base: clear of it

    # Left arm: closed fingers outside the sliding jaw's push tab (a tall tab on its outer side, 70 mm from the block
    # and 40 mm above the jaw tops), 40 mm clear of the tab's outer face, pad tips 20 mm below the tab's top.
    # The fingers push 80 mm below the tab's top (34 mm below the jaw tops, under the right hand's fingertips), and
    # the left hand is rolled 50 degrees about the push direction so its wrist leans back toward the left arm: held
    # straight down beside the right hand, the two wrists touched.
    push0 = W(T.TAB_X, T.TAB_FACE + 0.040, 0.0)
    push0[2] = T.TOP + T.TAB_TOP - 0.080 + 0.013
    yaw_l = 0.0          # the other half-turn leaves the left wrist on a bad branch here
    rot_l = _rot_axis((1.0, 0.0, 0.0), -math.radians(50)) @ grip_rotation(yaw_l)
    hover = np.array([0, 0, 0.030])

    def pushing():
        """Open loop: advance the left fingers toward the fixed jaw at 15 mm/s over a fixed 55 mm (22 mm to reach the
        jaw, 20 mm to close the gap to the block's width, 2 mm of preload, a little for the finger's lag), then hold."""
        s = 0.0
        while s < PUSH:
            s = min(s + 0.015 / CONTROL_HZ, PUSH)
            L.aim(push0 - inward * s, rot_l)
            yield
        yield from wait(1.0)

    one_arm = variant == "one_arm"      # the left arm alone closes the jaw on nothing: no block is ever held

    # 1. Right arm over the block, down, grip, lift, slide it in along the gap down the middle, then over against the
    # fixed jaw. The block stays low: the wrist cannot keep the fingers square high over the vise.
    yield from both(R.move(g0 + np.array([0, 0, 0.06]), rot_r, 2.5, grip=1.0), L.move(push0 + hover, rot_l, 2.5, grip=0.0))
    if not one_arm:
        yield from R.move(g0, rot_r, 1.5)
        yield from R.set_grip(0.0, 1.0)
        yield from R.move(np.array([g0[0], g0[1], pre_in[2]]), rot_r, 1.5)
        yield from R.move(pre_in, rot_r, 2.5)
        yield from R.move(np.array([pre_in[0], pre_in[1], mid[2]]), rot_r, 1.0)
        yield from R.move(mid, rot_r, 3.0)
        yield from R.move(hold, rot_r, 1.5)
    yield from wait(0.5)

    if variant == "release_early":
        yield from R.set_grip(1.0, 0.6)
        yield from R.move(R.pos + up, rot_r, 1.0)
        yield from wait(1.0)

    # 2. Left arm: down beside the jaw's outer face, push it shut on the block, back off.
    if variant != "no_close":
        yield from L.move(push0, rot_l, 1.5)
        yield from pushing()
        yield from wait(0.5)
        yield from L.move(L.pos + inward * 0.040, rot_l, 1.0)
        yield from L.move(L.pos + hover, rot_l, 1.0)

    # 3. Right arm lets go and lifts away; the clamp holds the block.
    if variant != "release_early" and not one_arm:
        yield from R.set_grip(1.0, 0.8)
        yield from wait(0.3)
        yield from R.move(R.pos + up, rot_r, 1.5)


def _yaw_of(p):
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 4.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    def _trace(ctx, task) -> str:
        """DEMO_DEBUG logging only: reads the simulated state; never used for control."""
        from yam_arena.scene import object_states

        st = object_states(ctx.emb._env, ctx.emb.task.names)
        b = task.block_in_vise(st["vise"], st["block"])
        msg = (f" jaw_mm {1000 * task.jaw_pos(st['vise']):.1f} gap_mm {1000 * task.gap(st['vise']):.1f}"
                f" between {b['between']} lift_mm {1000 * b['lift']:.1f} over_mm {1000 * b['over']:.1f}"
                f" tilt {b['tilt']:.1f} yaw {b['yaw']:.1f} low_mm {1000 * task._MECH['low']:.1f}"
                f" LR_mm {1000 * np.linalg.norm(ctx.left.pos - ctx.right.pos):.0f}"
                f" Lblock_mm {1000 * np.linalg.norm(ctx.left.pos - st['block'].to_world((0, 0, task.BLOCK_Z / 2))):.0f}"
                f" ik_mm L {ctx.left.err_mm:.1f} R {ctx.right.err_mm:.1f} failed {task._STATE['failed']} L_mm {np.round(1000 * ctx.left.pos, 1)}"
                f" R_mm {np.round(1000 * ctx.right.pos, 1)}")
        return msg

    ctx = Ctx(emb)
    gen = script(task, ctx)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % 10 == 0):
            msg = f"step {steps} score {info.get('score')}"
            if os.environ.get("DEMO_DEBUG"):
                msg += _trace(ctx, task)
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
