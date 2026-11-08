"""Scripted full-score demo for the chalk-line task, run in the yam_arena embodiment (open loop: the fixed layout from
the task file, timing and the arms' own joints; no object is ever read).

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

After 1.5 s of settling the right arm grips the reel on its pad, carries it to put the nozzle on the right pencil tick
and presses it on the plywood. The left arm lowers open over the line halfway along with its pad tips 8 mm over the
plywood, closes on the line, lifts 2 cm and opens. The right arm lifts the reel, returns it to its pad, lets go and
lifts away.

Negative variants (DEMO_VARIANT=...):
  one_arm    only the right arm works: it pins the reel, lets go, then closes on the slack line and lifts it (0)
  off_tick   the nozzle is pinned 10 mm off the right tick (0.25)
  no_return  the reel is set down 10 cm from its pad instead (0.5)
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
ROT_WEIGHT = 1.0          # keep the fingers square to the reel even when the wrist is near a limit


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


SETTLE = 1.5          # s: the scene settles before anything moves
LIFT = 0.02           # m the left hand lifts the line
TIP = 0.013           # m: pad tips below the TCP


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    T = task
    L, R = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in ("", "one_arm", "off_tick", "no_return"):
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    one_arm = variant == "one_arm"

    # Right arm: the reel's top, fingers closing along y (its 35 mm faces); TCP 1 mm over the top.
    g0 = np.array([T.REEL_START[0], T.REEL_START[1], T.PAD_TOP + T.REEL_Z + 0.001])
    rot_r = grip_rotation(reachable_yaw(R, g0, 0.0))
    # Nozzle on the right tick: the reel's middle is NOZZLE_TIP behind it along y. Command 3 mm into the plywood.
    dx = 0.010 if variant == "off_tick" else 0.0
    press = np.array([T.TICK_R[0] + dx, T.TICK_R[1] - T.NOZZLE_TIP, T.PLY_TOP - 0.003 + T.REEL_Z + 0.001])
    hover_p = press + np.array([0, 0, 0.035])
    # Put-down: the reel's underside on the pad (command 2 mm into it); no_return sets it 10 cm off, on the table.
    if variant == "no_return":
        put = np.array([T.REEL_START[0] + 0.10, T.REEL_START[1], T.TOP - 0.002 + T.REEL_Z + 0.001])
    else:
        put = np.array([T.REEL_START[0], T.REEL_START[1], T.PAD_TOP - 0.002 + T.REEL_Z + 0.001])
    up = np.array([0, 0, 0.06])

    # Left arm: closes along x on the line at (0.30, 0.04), pad tips 8 mm over the plywood.
    pluck_y = 0.04
    tcp_z = T.PLY_TOP + 0.008 + TIP
    pl = np.array([T.TICK_R[0], pluck_y, tcp_z])
    rot_l = grip_rotation(reachable_yaw(L, pl, math.pi / 2))
    hov_l = pl + np.array([0, 0, 0.06])

    # 0. settle, then both arms go over their jobs (left hand open, right hand over the reel)
    yield from wait(SETTLE)
    yield from both(R.move(g0 + up, rot_r, 2.5, grip=1.0), L.move(hov_l, rot_l, 2.5, grip=1.0))
    # 1. right arm: grip the reel, lift, carry over the right tick, lower and press
    yield from R.move(g0, rot_r, 1.5)
    yield from R.set_grip(0.0, 1.0)
    yield from wait(0.3)
    yield from R.move(g0 + up, rot_r, 1.2)
    yield from R.move(hover_p, rot_r, 3.0)
    yield from R.move(press + np.array([0, 0, 0.008]), rot_r, 1.5)
    yield from R.move(press, rot_r, 1.0)
    yield from wait(0.8)
    if one_arm:
        yield from R.set_grip(1.0, 0.8)           # the right hand lets go: the line goes slack
        yield from R.move(R.pos + up, rot_r, 1.2)
        yield from wait(0.5)
        pr = np.array([T.TICK_R[0], -0.06, tcp_z])
        rot_p = grip_rotation(reachable_yaw(R, pr, math.pi / 2))
        yield from R.move(pr + up, rot_p, 2.0)
        yield from R.move(pr, rot_p, 1.5)
        yield from R.set_grip(0.0, 1.0)
        yield from R.move(pr + np.array([0, 0, LIFT]), rot_p, 1.0)
        yield from wait(0.3)
        yield from R.set_grip(1.0, 1.0)
        yield from R.move(R.pos + up, rot_p, 1.0)
        return
    # 2. left arm: lower open, close on the line, lift it 2 cm, let go, back up
    yield from L.move(pl, rot_l, 1.5)
    yield from L.set_grip(0.0, 1.0)
    yield from wait(0.3)
    yield from L.move(pl + np.array([0, 0, LIFT]), rot_l, 1.0)
    yield from wait(0.3)
    yield from L.set_grip(1.0, 1.0)
    yield from L.move(hov_l, rot_l, 1.2)
    # 3. right arm: lift the reel clear, return it, release with margin, lift away
    yield from R.move(press + np.array([0, 0, 0.04]), rot_r, 1.2)
    yield from R.move(put + np.array([0, 0, 0.04]), rot_r, 3.0)
    yield from R.move(put, rot_r, 1.5)
    yield from wait(0.3)
    yield from R.set_grip(1.0, 1.0)
    yield from wait(0.3)
    yield from R.move(put + np.array([0, 0, 0.10]), rot_r, 1.5)


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

        st = object_states(ctx.emb._env, ["reel"])
        reel = st["reel"]
        tip = task.nozzle_tip(reel)
        err = 1000 * np.linalg.norm(tip[:2] - np.asarray(task.TICK_R))
        msg = (f" reel_mm {np.round(1000 * reel.pos, 1)} tip_mm {np.round(1000 * tip, 1)} tip_err_mm {err:.1f}"
               f" pinned {task.pinned(reel)} held {task._STATE['held']} taut {task._STATE['taut']}"
               f" snaps {[(s['taut'], round(1000 * s['err'], 1)) for s in task._STATE['snaps']]}"
               f" L_mm {np.round(1000 * ctx.left.pos, 1)} R_mm {np.round(1000 * ctx.right.pos, 1)}"
               f" ik_mm L {ctx.left.err_mm:.1f} R {ctx.right.err_mm:.1f}")
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
