"""Scripted full-score demo for the dustpan-and-brush task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The right arm grips the dustpan's handle and holds the pan still. The left arm grips the brush's handle bar, lowers
the brush to 2 mm above the table and sweeps it toward the pan, riding up the lip so the cubes go onto the floor of
the pan, then lifts the brush straight up and away.

Negative variants (DEMO_VARIANT=...), each must score below 1:
  no_hold     the right arm never holds the pan: the brush pushes the pan away
  gripper     the left arm rakes the cubes in with its bare open fingers instead of the brush (they count as touched)
  short       the brush stops at the lip and never rides onto the pan, leaving the cubes on the lip
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


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    L, R = ctx.left, ctx.right
    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    s0 = ctx.states()
    pan, brush = s0["dustpan"], s0["brush"]
    up = np.array([0.0, 0.0, 0.06])

    # 1. Right arm onto the pan handle (fingers closing across its 20 mm width, along x); left arm onto the brush's
    #    handle bar (fingers closing across it, along y). Tool point 3 mm below each handle's top.
    handle = pan.to_world((0.0, T.HANDLE_GRIP_Y, T.HANDLE_Z[1] - 0.003))
    knob = brush.to_world((0.0, 0.0, T.KNOB_TOP - 0.003))
    rot_r = grip_rotation(reachable_yaw(R, handle, math.pi / 2))
    rot_l = grip_rotation(reachable_yaw(L, knob, 0.0))
    gripper = variant == "gripper"
    moves = []
    if variant != "no_hold":
        moves.append(R.move(handle + up, rot_r, 2.5, grip=1.0))
    if not gripper:
        moves.append(L.move(knob + up, rot_l, 2.5, grip=1.0))
    yield from both(*moves)
    moves = []
    if variant != "no_hold":
        moves.append(R.move(handle, rot_r, 1.5))
    if not gripper:
        moves.append(L.move(knob, rot_l, 1.5))
    yield from both(*moves)
    moves = []
    if variant != "no_hold":
        moves.append(R.set_grip(0.0, 1.0))
    if not gripper:
        moves.append(L.set_grip(0.0, 1.0))
    yield from both(*moves)
    yield from wait(0.3)

    # 2. The sweep, along the pan's +y axis toward the pan (world -y). ``lead`` is how far the tool's front sits ahead
    #    of the tool point; ``clear`` is the gap kept under it.
    if gripper:
        # Open fingers spread along x (94 mm apart), pad tips 5 mm above the table, raked across the cubes.
        lead, clear, tool_z = 0.010, 0.005, T.TOP + 0.005 + 0.013
        start = pan.to_world((0.0, 0.0, 0.0)) + np.array([0.0, 0.135, 0.0])
        start[2] = tool_z
        rot_l = grip_rotation(reachable_yaw(L, start, math.pi / 2))
        yield from L.move(start + up, rot_l, 2.5, grip=1.0)
        yield from L.move(start, rot_l, 1.5)
        base_z = tool_z
    else:
        lead, clear = T.BRUSH_HALF_T, 0.002
        yield from L.move(L.pos + np.array([0.0, 0.0, clear + 0.0005]), rot_l, 0.5)   # lift the brush 2 mm
        base_z = L.pos[2]

    def at(front_y, extra_z):
        """Tool point for the tool's front at pan-frame y = ``front_y``, raised ``extra_z`` above its sweep height."""
        p = pan.to_world((0.0, front_y + lead, 0.0))
        return np.array([p[0], p[1], base_z + extra_z])

    rise = T.FLOOR_TOP + 0.001   # the lip ramps up to the floor's height over its length
    yield from L.move(at(0.004, 0.0), rot_l, 3.0)                    # across the table to the lip
    if variant == "short":
        yield from wait(1.0)
    else:
        yield from L.move(at(T.LIP_END_Y - 0.002, rise), rot_l, 1.2)     # up the lip, following its slope
        yield from L.move(at(-0.068, rise), rot_l, 1.5)                 # over the floor, cubes ahead of the brush
        yield from wait(0.5)
    # 3. Brush straight up and back toward its start, so it drags nothing back out.
    yield from L.move(L.pos + np.array([0.0, 0.0, 0.08]), rot_l, 1.5)
    yield from L.move(L.pos + np.array([0.0, 0.08, 0.0]), rot_l, 1.5)
    yield from wait(1.0)


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
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % 20 == 0):
            msg = f"step {steps} score {info.get('score')}"
            if os.environ.get("DEMO_DEBUG"):
                st = ctx.states()
                pan = st["dustpan"]
                msg += f" pan_moved_mm {1000 * np.linalg.norm((pan.pos - task._STATE['pan0'])[:2]) if task._STATE['pan0'] is not None else 0:.1f}"
                msg += f" tilt {task.pan_tilt(pan):.1f} touched {sorted(task._STATE['touched'])}"
                msg += f" fingers_ok {task._STATE['fingers_ok']}"
                msg += " cubes_local_mm " + str([np.round(1000 * pan.to_local(st[n].pos), 0).tolist() for n in task.CUBE_NAMES])
                msg += f" brush_mm {np.round(1000 * pan.to_local(st['brush'].pos), 0).tolist()}"
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
