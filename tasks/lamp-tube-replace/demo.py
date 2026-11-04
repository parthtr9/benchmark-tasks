"""Scripted full-score demo for the lamp-tube task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Both arms grip the tube 15 cm either side of its middle, lift it, carry it over the sockets and, closed loop on the
measured tube pose, line each end up over its slot and lower it until both ends' pins are in the pockets. Both then
turn it a quarter turn together and let go. DEMO_ONE_ARM=1 has the right arm do the same alone, gripping as near the
middle as it reaches over the sockets.
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

    T = task
    dt = 1.0 / CONTROL_HZ
    one_arm = bool(os.environ.get("DEMO_ONE_ARM"))   # the right arm alone, gripping as near the middle as it reaches
    gy = 0.10 if one_arm else 0.15                    # grip points either side of the tube's middle
    arms = [(ctx.right, -1)] if one_arm else [(ctx.left, 1), (ctx.right, -1)]
    tube = ctx.states()["tube"]
    grip = {a: tube.to_world((0.0, s * gy, 0.0)) for a, s in arms}
    # Fingers closing across the tube (along x); the turn is about the tube's axis (world y), so each tool point stays
    # on the axis while its grip rolls about it.
    rot = {a: grip_rotation(reachable_yaw(a, grip[a], math.pi / 2)) for a, s in arms}
    turned = {a: _rot_axis((0.0, 1.0, 0.0), -math.pi / 2) @ rot[a] for a, s in arms}

    def all_move(target, duration, grip_to=None, rots=rot):
        yield from both(*(a.move(target(a, s), rots[a], duration, grip_to) for a, s in arms))

    # 1. Both arms grip the tube 15 cm either side of its middle and lift it out of the cradles.
    yield from all_move(lambda a, s: grip[a] + [0, 0, 0.06], 2.5, 1.0)
    yield from all_move(lambda a, s: grip[a], 1.5)
    yield from all_move(lambda a, s: grip[a], 1.0, 0.0)
    yield from wait(0.3)
    yield from all_move(lambda a, s: grip[a] + [0, 0, 0.06], 1.5)

    # 2. Carry it over the sockets, its axis 3 cm above the pockets, pins upright.
    above = lambda a, s: np.array([T.SOCKET_X, s * gy, T.AXIS_Z + 0.03])  # noqa: E731
    yield from all_move(above, 3.0)

    # 3. Closed loop on the measured tube pose: line each end up over its socket's slot, then lower it at up to 5 mm/s
    #    until it rests on the ledges with both ends' pins in the pockets.
    corr = {a: np.zeros(2) for a, s in arms}
    z = 0.03
    for k in range(int(12 * CONTROL_HZ)):
        st = ctx.states()["tube"]
        for a, s in arms:
            end = st.to_world((0.0, s * (T.CAP_FACE - 0.005), 0.0))
            corr[a] = np.clip(corr[a] - 4.0 * dt * np.array([end[0] - T.SOCKET_X, st.pos[1]]), -0.02, 0.02)
        if k > CONTROL_HZ:                            # 1 s to line up before going down
            z = max(z - 0.005 * dt, -0.002)
        for a, s in arms:
            a.aim(np.array([T.SOCKET_X + corr[a][0], s * gy + corr[a][1], T.AXIS_Z + z]), rot[a])
        off, dy, _ = T.tube_fit(st)
        if z <= -0.002 and off < 0.0015:
            break
        yield
    yield from wait(0.5)

    # 4. Both turn the tube a quarter turn together and let go. The fingers now point sideways, one above and one
    #    below the tube, so each open hand turns back upright around the tube before lifting away.
    yield from all_move(lambda a, s: a.pos, 4.0, rots=turned)
    yield from wait(2.5)
    yield from all_move(lambda a, s: a.pos, 1.2, 1.0, rots=turned)
    yield from all_move(lambda a, s: a.pos, 2.0)
    yield from all_move(lambda a, s: a.pos + [0, 0, 0.06], 1.2)


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
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % 20 == 0 and steps > int(os.environ.get("DEMO_DEBUG_FROM", 380))):
            msg = f"step {steps} score {info.get('score')}"
            if os.environ.get("DEMO_DEBUG"):
                st = ctx.states()
                off, dy, turn = task.tube_fit(st["tube"])
                msg += (f" tube_mm {np.round(1000 * st['tube'].pos, 1)} off_mm {1000 * off:.1f} dy_mm {1000 * dy:.1f}"
                        f" turn {turn:.1f} detent_mm {1000 * st['tube'].sliders.get('detent_l', 0):.2f}")
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
