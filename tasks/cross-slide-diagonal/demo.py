"""Scripted full-score demo for the cross-slide task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Open loop: it uses only the fixed layout from the task file, timing and the arms' own joint readings. Both arms come down
on their feed handles (the left one closes across x, the right one across y, so the fingers push the post the way it
travels), close firmly, and feed 74 mm together over 6 s (the slides stop at 70 mm; the extra 4 mm covers the arms' lag).
Then both let go and lift away.

Negative variants (DEMO_VARIANT=...):
  one_arm          only the left arm feeds; the right handle stays put (0)
  x_then_y         the left arm feeds 74 mm, then the right arm does (0): a step, not a diagonal
  staircase        in step to 40 mm, then 20 mm moves one after the other (0 in the simulation: peak e is 20 mm, over 15 mm)
  half_speed_pause in step to 35 mm, both stop for 3 s, then finish (0.5)
DEMO_DEBUG=1 prints each fingertip against its handle at close time, and the slides and the arms' lag over time.
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

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
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


VARIANTS = ("", "one_arm", "x_then_y", "staircase", "half_speed_pause")
FEED = 0.074                                  # commanded travel: the stop is at 70 mm


def handle_points(task):
    """World tool points on the two handle posts (a fixed layout): 12 mm above each post's middle."""
    b = np.array([*task.BASE_XY, task.TOP])
    left = b + np.array(task.LEFT_POST) + np.array([0, 0, 0.012])
    right = b + np.array(task.RIGHT_POST) + np.array([0, 0, 0.012])
    return left, right


def segments(variant):
    """(seconds, dx, dy) legs of the feed: the slides' commanded positions at the end of each leg."""
    F = FEED
    if variant == "one_arm":
        return [(6.0, F, 0.0)]
    if variant == "x_then_y":
        return [(3.0, F, 0.0), (3.0, F, F)]
    if variant == "staircase":
        return [(3.5, 0.040, 0.040), (1.0, 0.060, 0.040), (1.0, 0.060, 0.060), (1.0, F, 0.060), (1.0, F, F)]
    if variant == "half_speed_pause":
        return [(3.0, 0.035, 0.035), (3.0, 0.035, 0.035), (3.0, F, F)]
    return [(6.0, F, F)]


def _smooth(t: float) -> float:
    return t * t * (3 - 2 * t)


def feed(L, R, left0, right0, rotL, rotR, segs):
    """Generator: both arms track their handles' commanded positions along the legs (smooth within a leg)."""
    dx0 = dy0 = 0.0
    for seconds, dx1, dy1 in segs:
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = _smooth(k / n)
            dx, dy = dx0 + s * (dx1 - dx0), dy0 + s * (dy1 - dy0)
            L.aim(left0 + np.array([dx, 0.0, 0.0]), rotL)
            R.aim(right0 + np.array([0.0, dy, 0.0]), rotR)
            yield
        dx0, dy0 = dx1, dy1


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    L, R = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    left0, right0 = handle_points(task)
    yawL = reachable_yaw(L, left0, math.pi / 2)       # fingers close along x: they push the post along its travel
    yawR = reachable_yaw(R, right0, 0.0)              # fingers close along y
    rotL, rotR = grip_rotation(yawL), grip_rotation(yawR)
    up = np.array([0.0, 0.0, 0.07])

    # 1. Above the handles, down onto them, close firmly.
    yield from both(L.move(left0 + up, rotL, 2.5, grip=1.0), R.move(right0 + up, rotR, 2.5, grip=1.0))
    yield from both(L.move(left0, rotL, 1.5), R.move(right0, rotR, 1.5))
    yield from wait(0.3)
    if os.environ.get("DEMO_DEBUG"):
        ctx.report_close(task, "before close")
    yield from both(L.set_grip(0.0, 1.0), R.set_grip(0.0, 1.0))
    yield from wait(0.3)
    if os.environ.get("DEMO_DEBUG"):
        ctx.report_close(task, "after close")

    # 2. The feed.
    yield from feed(L, R, left0, right0, rotL, rotR, segments(variant))
    yield from wait(0.5)

    # 3. Let go, lift away, stay clear and still.
    yield from both(L.set_grip(1.0, 0.6), R.set_grip(1.0, 0.6))
    yield from both(L.move(L.pos + up, L.rot, 1.2), R.move(R.pos + up, R.rot, 1.2))


def _yaw_of(p):
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    def measured(self, side):
        """(tool point, approach axis) of an arm from its own joint readings."""
        from yam_arena import ik, station

        rob = self.emb._env.scene[f"{side}_robot"]
        q = np.asarray(getattr(rob.data.joint_pos, "torch", rob.data.joint_pos)[0].detach().cpu(), float)[:6]
        tcp, rot = ik.tcp_pose(q, station.BASES[side])
        return tcp, rot[:, 2]

    def report_close(self, task, label):
        """Debug: each fingertip against its handle's middle (world mm) and the finger opening."""
        m = self.states()["mill"]
        post = task.post_boxes(m.pos, *task.slides(m))
        for side, (c, _) in zip(("left", "right"), post):
            tcp, a = self.measured(side)
            tip = tcp + 0.013 * a
            print(f"[{label}] {side}: tip-post (mm) {np.round(1000 * (tip - c), 1)} tcp-post {np.round(1000 * (tcp - c), 1)}",
                  flush=True)

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
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % 5 == 0):
            msg = f"step {steps} score {info.get('score')}"
            if os.environ.get("DEMO_DEBUG"):
                st = ctx.states()
                dx, dy = task.slides(st["mill"])
                (lt, _), (rt, _) = ctx.measured("left"), ctx.measured("right")
                msg += (f" dx_mm {1000 * dx:.1f} dy_mm {1000 * dy:.1f} e_mm {1000 * abs(dx - dy):.1f}"
                        f" peak_e {1000 * task._STATE['peak_e']:.1f} pause {task._STATE['pause']}"
                        f" lagL_mm {1000 * (ctx.left.pos[0] - lt[0]):.1f} lagR_mm {1000 * (ctx.right.pos[1] - rt[1]):.1f}"
                        f" clear {task._STATE['clear']}")
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
