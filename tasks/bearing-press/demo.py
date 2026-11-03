"""Scripted full-score demo for the bearing press task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The left arm holds the link's far end. The right arm grips the bearing's sides, lifts it from its holder, lines its
bore up with the shaft 15 mm beyond the shaft's end and, closed loop on the measured bearing pose, presses it along the
shaft at up to 10 mm/s, past the spring ball, until it seats against the shoulder. Then both let go.
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
    s0 = ctx.states()
    link, b0 = s0["robot_link"], s0["bearing"]
    link_yaw = T.LINK_YAW
    # Tool point level with the bearing's top: the pads hold its whole height (lower, they only reach its bottom half and
    # it swings in the grip). Pushing from up there tips it a few degrees; the bore's 0.5 mm play allows about 8.
    grip_dz = T.B_HALF

    # 1. Left arm holds the link's far end across its width (DEMO_NO_BRACE=1 leaves it idle).
    if not os.environ.get("DEMO_NO_BRACE"):
        hold = link.to_world((-0.055, 0.0, 0.030))
        hold_yaw = reachable_yaw(L, hold, link_yaw)
        yield from L.move(hold + [0, 0, 0.05], grip_rotation(hold_yaw), 2.5, grip=1.0)
        yield from L.move(hold, grip_rotation(hold_yaw), 1.5)
        yield from L.set_grip(0.0, 2.0)
        yield from wait(0.3)

    # 2. Right arm grips the bearing's sides (fingers closing along its x axis) and lifts it out of the holder.
    grasp = b0.pos + np.array([0, 0, grip_dz])
    pick_yaw = reachable_yaw(R, grasp, T.HOLDER_YAW + math.pi / 2)
    yield from R.move(grasp + [0, 0, 0.06], grip_rotation(pick_yaw), 2.5, grip=1.0)
    yield from R.move(grasp, grip_rotation(pick_yaw), 1.5)
    yield from R.set_grip(0.0, 1.0)
    yield from wait(0.3)
    yield from R.move(grasp + [0, 0, 0.06], grip_rotation(pick_yaw), 1.5)

    # 3. Bore in line with the shaft, 15 mm beyond its end.
    off = pick_yaw - T.HOLDER_YAW  # keep the grip branch the bearing was picked with
    start_y = T.SHAFT_END_Y + T.B_WIDTH / 2 + 0.015
    yield from R.move(link.to_world((T.SHAFT_X, start_y, T.SHAFT_Z + grip_dz + 0.04)), grip_rotation(link_yaw + off), 2.5)
    yield from R.move(link.to_world((T.SHAFT_X, start_y, T.SHAFT_Z + grip_dz)), grip_rotation(link_yaw + off), 1.5)

    # 4. Closed loop on the measured bearing pose: hold the bore on the shaft's axis and square to it, and press it
    #    along at up to 10 mm/s until it has cleared the ball; the shoulder limits how far it goes.
    state = {"y": start_y, "x": 0.0, "z": 0.0, "yaw": 0.0}
    seated_y = (T.SEATED_Y[0] + T.SEATED_Y[1]) / 2
    goal_y = T.SEATED_Y[0] - 0.006  # 6 mm past the shoulder: lead for the arm to press through the ball
    dt = 1.0 / CONTROL_HZ

    def servo(y_goal):
        st = ctx.states()
        lk, br = st["robot_link"], st["bearing"]
        c, rel_yaw, _ = T.bearing_on_shaft(lk, br)
        state["x"] -= 2.0 * dt * (c[0] - T.SHAFT_X)
        state["z"] -= 2.0 * dt * (c[2] - T.SHAFT_Z)
        state["yaw"] = float(np.clip(state["yaw"] - 2.0 * dt * math.radians(rel_yaw), -0.2, 0.2))
        state["y"] += float(np.clip(y_goal - state["y"], -0.010 * dt, 0.010 * dt))
        aim = lk.to_world((T.SHAFT_X + state["x"], state["y"], T.SHAFT_Z + state["z"] + grip_dz))
        R.aim(aim, grip_rotation(_yaw_of(lk) + off + state["yaw"]))
        return c

    for _ in range(int(30 * CONTROL_HZ)):
        c = servo(goal_y)
        if c[1] <= seated_y:
            break
        yield
    for _ in range(int(1.0 * CONTROL_HZ)):
        servo(state["y"])
        yield

    # 5. Let go, lift away, release the link.
    yield from R.set_grip(1.0, 1.0)
    yield from R.move(R.pos + [0, 0, 0.06], R.rot, 1.5)
    if not os.environ.get("DEMO_NO_BRACE"):
        yield from L.set_grip(1.0, 0.8)
        yield from L.move(L.pos + [0, 0, 0.05], L.rot, 1.5)


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
                c, yaw, tilt = task.bearing_on_shaft(st["robot_link"], st["bearing"])
                msg += (f" bearing_mm {np.round(1000 * c, 1)} yaw {yaw:.1f} tilt {tilt:.1f}"
                        f" ball_mm {1000 * st['robot_link'].sliders.get('ball', 0):.2f}"
                        f" link_moved_mm {1000 * np.linalg.norm(st['robot_link'].pos - task._STATE['link0']):.1f}")
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
