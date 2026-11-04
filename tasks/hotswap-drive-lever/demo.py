"""Scripted full-score demo for the hot-swap drive task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The right arm grips the handle's pull tab and pushes the carrier into the bay along the open handle (the push runs
through the pin, so the handle does not turn) until the connector plunger bottoms out. It opens its fingers into a fork
around the tab (the carrier springs back a few millimetres) and swings the tab round the pin, tracking the measured pin
position and handle angle each step, until the handle latches across the carrier's front; the hook levers the carrier
home on the way. Then it lets go and lifts away.

Negative variants (DEMO_VARIANT=...):
  no_close     push by the tab, then shove the carrier's front face with the closed fingers, and leave the handle
               open: the plunger keeps it from seating and nothing latches
  close_first  swing the handle shut before the carrier is in, then push: the hook stops at the catch post
  short        push only 20 mm, then swing the handle: the hook passes in front of the post and catches nothing
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


def tab_point(sled, task, angle, z):
    """World position of the pull tab's centre at handle ``angle`` (rad), at TCP height ``z``."""
    px, py, _ = task.PIVOT
    local = (px - task.TAB_R * math.cos(angle), py - task.TAB_R * math.sin(angle), 0.0)
    p = sled.to_world(local)
    return np.array([p[0], p[1], z])


def swing(task, ctx, arm, z, yaw0, rate=1.0, lead=0.25, end=None):
    """Closed loop: the open fingers straddle the tab along its path; each step the fork's centre goes to where the
    tab would be ``lead`` rad further round (capped at ``end``), so the trailing finger pushes it, at most ``rate``
    rad/s ahead of the last command. Yields once per step until the command reaches ``end`` and the handle is shut."""
    from yam_arena.demo import grip_rotation

    end = task.CLOSED + 0.15 if end is None else end
    cmd = None
    for _ in range(int(8 * CONTROL_HZ)):
        sled = ctx.states()["carrier"]
        angle = task.handle_angle(sled)
        cmd = angle if cmd is None else cmd
        cmd = min(cmd + rate / CONTROL_HZ, angle + lead, end)
        arm.aim(tab_point(sled, task, cmd, z), grip_rotation(yaw0 + cmd))
        yield
        if cmd >= end - 1e-6 and angle >= task.LATCH_AT:
            break
    for _ in range(int(0.5 * CONTROL_HZ)):
        yield


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    T = task
    R = ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    s0 = ctx.states()
    sled, bay = s0["carrier"], s0["bay"]
    # TCP 4 mm below the tab's top: the pad tips (13 mm further down) stay 2 mm above the bar under the tab.
    z = float(sled.pos[2]) + 0.050
    start = tab_point(sled, T, 0.0, z)
    yaw0 = reachable_yaw(R, start, 0.0)          # fingers close along the bar's width (world y)
    rot0 = grip_rotation(yaw0)
    inward = bay.rotation()[:, 0]
    up = np.array([0, 0, 0.06])

    # 1. Above the tab, down onto it.
    yield from R.move(start + up, rot0, 2.5, grip=1.0)
    yield from R.move(start, rot0, 1.5)

    if variant == "close_first":
        # Swing the handle shut out in front of the bay, then try to push the carrier in by its tab.
        yield from R.set_grip(0.35, 0.5)
        yield from swing(T, ctx, R, z, yaw0)
        sled = ctx.states()["carrier"]
        tab = tab_point(sled, T, T.handle_angle(sled), z)
        rot = grip_rotation(yaw0 + T.CLOSED)
        yield from R.move(tab, rot, 0.8, grip=1.0)
        yield from R.set_grip(0.0, 1.0)
        yield from R.move(tab + inward * (T.STOP_F - T.START_F + 0.003), rot, 2.5)
        yield from wait(0.5)
        yield from R.set_grip(1.0, 0.8)
        yield from R.move(R.pos + up, rot, 1.5)
        return

    # 2. Grip the tab and push along the open handle until the plunger bottoms out (or only 20 mm: "short").
    yield from R.set_grip(0.0, 1.0)
    yield from wait(0.2)
    push = 0.020 if variant == "short" else T.STOP_F - T.START_F + 0.012   # 12 mm past the stop: the grip gives
    yield from R.move(start + inward * push, rot0, 2.5)
    yield from wait(0.3)

    if variant == "no_close":
        # Let go of the tab, push the carrier's front face home with the closed fingers, and back away: unlatched,
        # the plunger pushes it back out.
        yield from R.set_grip(1.0, 0.8)
        yield from R.move(R.pos + up, rot0, 1.0)
        f = T.carrier_in_bay(ctx.states()["bay"], ctx.states()["carrier"])[0]
        face = bay.to_world((f, -0.012, 0.0))
        low = float(bay.pos[2]) + T.FLOOR_Z + 0.010 + 0.013   # pad tips 10 mm above the bay floor
        yield from R.move(np.array([face[0] - 0.030, face[1], low + 0.06]), rot0, 1.0, grip=0.0)
        yield from R.move(np.array([face[0] - 0.030, face[1], low]), rot0, 1.0)
        yield from R.move(np.array([float(bay.to_world((T.STOP_F + 0.025, 0, 0))[0]) - 0.010, face[1], low]),
                          rot0, 2.0)
        yield from wait(0.5)
        yield from R.move(R.pos - inward * 0.04, rot0, 1.0)
        yield from R.move(R.pos + up, rot0, 1.0)
        return

    # 3. Open into a fork round the tab and swing it shut; the hook levers the carrier home.
    yield from R.set_grip(0.35, 0.5)
    yield from swing(T, ctx, R, z, yaw0)

    # 4. Let go and lift away.
    yield from R.set_grip(1.0, 0.6)
    yield from R.move(R.pos + up, R.rot, 1.5)


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
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % 10 == 0):
            msg = f"step {steps} score {info.get('score')}"
            if os.environ.get("DEMO_DEBUG"):
                st = ctx.states()
                f, dy, lift, tilt = task.carrier_in_bay(st["bay"], st["carrier"])
                msg += (f" f_mm {1000 * f:.1f} dy_mm {1000 * dy:.1f} lift_mm {1000 * lift:.1f} tilt {tilt:.1f}"
                        f" handle_deg {math.degrees(task.handle_angle(st['carrier'])):.1f}"
                        f" plunger_mm {1000 * st['carrier'].sliders.get('plunger', 0.0):.1f}"
                        f" latched {task._MECH['latched']} bay_mm {np.round(1000 * (st['bay'].pos - task._STATE['bay0']), 1) if task._STATE['bay0'] is not None else None} R_cmd_mm {np.round(1000 * ctx.right.pos, 1)}")
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
