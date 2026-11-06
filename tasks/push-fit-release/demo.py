"""Scripted full-score demo for the push-fit task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The left arm closes its fingers and pushes the near side bar of the orange collet in with them, beside the pipe,
closed loop on the measured slider, and holds it there. The right arm grips the pipe from above, pulls it out along its axis
at 10 mm/s until its end is clear of the fitting, the left arm lets go, and the right arm lays the pipe flat on the
table in front of the fitting.

Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder.
Set DEMO_NO_PRESS=1 to pull with the collet untouched: that must score 0. Set DEMO_EARLY_RELEASE=1 to press the
collet and let it go before pulling: the tooth bites again, so that must score below 1 (it scores 0: the hard pull
on the locked pipe moves the fitting).
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
PRESS_TO = 0.0031     # push the collet this far (the task lifts the grab tooth at 2.5 mm; the stop is at 3.5 mm)
PRESS_AT = (-0.017, 0.041)   # left TCP beside the pipe on the collet's near side bar (x from the axis, height): the
                             # closed fingers stand on the bar from 28 mm up, clear of the pipe
GRASP_AT = -0.065     # where the right arm grips the pipe, from its middle (m): 17 cm from the collet, so the two
                      # wrists stay 13 cm or more apart, and clear of the block at the end of the pull


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
        self.off = np.zeros(3)

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


def servo(arm, ctx, target, rot, seconds: float, axes=(0, 1, 2)):
    """Generator: hold ``target`` with the measured TCP, adding up the error on ``axes`` (the arm sags under load)."""
    off = np.zeros(3)
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        err = np.asarray(target) - ctx.actual(arm.name)
        for i in axes:
            off[i] = float(np.clip(off[i] + 0.5 * err[i], -0.03, 0.03))
        arm.aim(np.asarray(target) + off, rot)
        yield
    arm.off = off


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    L, R = ctx.left, ctx.right
    T = task
    s0 = ctx.states()
    fitting, pipe = s0["fitting"], s0["pipe"]
    # Fingers close roughly along x, across the pipe, with each wrist camera on the outer side, away from the other
    # hand; the turn is short of 90 degrees so the wrist's roll joint stays inside its range.
    rot_l, rot = grip_rotation(1.25), grip_rotation(-1.35)
    pressed = not os.environ.get("DEMO_NO_PRESS")

    # 1. Push the collet's side bar in with the closed fingers, beside the pipe, closed loop on the measured slider.
    if pressed:
        face = fitting.to_world((PRESS_AT[0], -0.030, PRESS_AT[1]))
        yield from L.move(face + [0, 0, 0.06], rot_l, 2.5, grip=0.0)
        yield from L.move(face, rot_l, 2.0)
        y = 0.0
        for _ in range(int(6 * CONTROL_HZ)):
            y += 0.0005  # 10 mm/s
            L.aim(face + np.array([0.0, y, 0.0]), rot_l)
            yield
            if T.collet_pressed(ctx.states()["fitting"]) >= PRESS_TO or y >= 0.040:
                break
        L.aim(face + np.array([0.0, y + 0.001, 0.0]), rot_l)   # lean on it a little more and hold

    if pressed and os.environ.get("DEMO_EARLY_RELEASE"):
        yield from L.move(L.pos + [0, -0.02, 0], rot_l, 1.0)
        yield from L.move(L.pos + [0, 0, 0.08], rot_l, 1.5)
        pressed = False

    # 2. Grip the pipe from above, servoing the measured TCP onto the pipe's axis before closing.
    grasp = pipe.to_world((0.0, GRASP_AT, 0.0))
    yield from R.move(grasp + [0, 0, 0.08], rot, 2.5, grip=1.0)
    yield from R.move(grasp, rot, 1.5)
    yield from servo(R, ctx, grasp, rot, 1.5)
    yield from R.set_grip(0.0, 1.0)
    yield from wait(0.3)

    # 3. Pull it out along its axis at 10 mm/s until its end is 3.5 cm out from the bottom of the bore.
    # The hand is kept on the pipe's axis (x and height) from its measured position, so the pipe does not cock and bind.
    y_cmd, off = float(grasp[1]), R.off.copy()
    for _ in range(int(10 * CONTROL_HZ)):
        st = ctx.states()
        if T.pulled(st["fitting"], st["pipe"]) >= 0.040:
            break
        y_cmd -= 0.0005
        target = np.array([float(grasp[0]), y_cmd, float(grasp[2])])
        err = target - ctx.actual("right")
        off[[0, 2]] = np.clip(off[[0, 2]] + 0.5 * err[[0, 2]], -0.03, 0.03)
        R.aim(target + off, rot)
        yield

    # 4. The left arm lets go; the right lifts the pipe clear and lays it flat in front of the fitting.
    if pressed:
        yield from L.move(L.pos + [0, -0.02, 0], rot_l, 1.0)
        yield from L.move(L.pos + [0, 0, 0.08], rot_l, 1.5)
    yield from R.move(R.pos + [0, 0, 0.06], rot, 1.5)
    place = np.array([0.18, -0.20, T.TOP + 0.017])   # the pipe then lies from y -0.01 to -0.31, 10 cm clear
    yield from R.move(place + [0, 0, 0.05], rot, 3.0)
    yield from R.move(place, rot, 2.0)
    yield from wait(0.5)
    yield from R.set_grip(1.0, 1.0)
    yield from R.move(R.pos + [0, 0, 0.08], R.rot, 1.5)


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def actual(self, name: str) -> np.ndarray:
        """The arm's measured TCP position (world), from its joint positions."""
        from yam_arena import ik

        arm_ids = self.emb._ids[f"{name}_robot"][0]
        q = self.emb._base.scene[f"{name}_robot"].data.joint_pos.torch[0][arm_ids].cpu().numpy().astype(np.float64)
        return ik.tcp_pose(q, self.left.base if name == "left" else self.right.base)[0]

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
                f, p = st["fitting"], st["pipe"]
                msg += (f" collet_mm {1000 * task.collet_pressed(f):.2f}"
                        f" tooth_mm {1000 * f.sliders.get('tooth', 0.0):.2f}"
                        f" pulled_mm {1000 * task.pulled(f, p):.1f}"
                        f" pipe {np.round(1000 * p.pos, 1)}"
                        f" fitting_moved_mm {1000 * np.linalg.norm(f.pos - task._STATE['fit0']):.1f}"
                        f" R_err_mm {np.round(1000 * (ctx.right.pos - ctx.actual('right')), 1)}")
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
