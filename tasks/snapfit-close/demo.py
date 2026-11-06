"""Scripted full-score demo for the snap-fit lid task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The demo uses only what a policy could see: the fixed layout in the first camera frame (the task's constants; the run
uses object_jitter=0), one look at the case's pose once the brace has settled (a fixed part in plain view of the
cameras; gripping it nudges it) and its own arm commands. Everything after that look is open loop; it never reads the
latch state or slider positions.
The right arm picks the lid off its rails, lowers it into the pocket, lets go and parks. The left arm grips the base's
long sides behind the pocket, its wrist leaning away from the right arm. The right arm then presses the middle of the
lid's front edge slowly down to a fixed depth past where a seated lid is flush with the case rim (the latches click
on the way), holds, and both arms let go and lift clear.
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


class Frame:
    """A fixed object frame from the task's layout constants: what the first camera frame shows."""

    def __init__(self, xy, z, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        self.pos = np.array([xy[0], xy[1], z], float)
        self.rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def to_world(self, local):
        return self.pos + self.rot @ np.asarray(local, float)


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    L, R = ctx.left, ctx.right
    T = task
    lid0 = Frame(T.RAILS_XY, T.LID.pos[2], T.RAILS_YAW)
    base = Frame(T.BASE_XY, T.BASE.pos[2], T.BASE_YAW)
    pinch_local = grip_rotation(0.0).T @ np.array([0.0, 1.0, 0.0])

    def down(yaw):
        return grip_rotation(yaw)

    # 1. Right arm picks the lid across its width.
    lid_yaw = T.RAILS_YAW
    grasp = lid0.pos + np.array([0, 0, 0.005])
    yield from R.move(grasp + [0, 0, 0.08], down(lid_yaw), 2.5, grip=1.0)
    yield from R.move(grasp, down(lid_yaw), 1.5)
    yield from R.set_grip(0.0, 1.0)
    yield from wait(0.3)
    yield from R.move(grasp + [0, 0, 0.08], down(lid_yaw), 1.5)

    # 2. Over the pocket, lid turned to the base, lowered onto the raised latches, released.
    base_yaw = T.BASE_YAW
    rest = base.to_world((0, 0, T.FLOOR_Z + T.LATCH_TRAVEL + T.LID_HALF)) + np.array([0, 0, 0.005 + 0.003])
    yield from R.move(rest + [0, 0, 0.05], down(base_yaw), 3.0)
    yield from R.move(rest, down(base_yaw), 1.5)
    yield from wait(0.3)
    yield from R.set_grip(1.0, 1.0)
    yield from wait(0.3)
    yield from R.move(rest + [0, 0, 0.09], down(base_yaw), 1.2)
    park = base.to_world((0.11, 0.0, 0.11))
    yield from R.move(park, down(base_yaw), 2.0, grip=0.0)

    # 3. Left arm braces the base's long sides 35 mm behind its centre, wrist leaning away.
    #    DEMO_NO_BRACE=1 skips this, to check that one arm alone tips the base.
    if not os.environ.get("DEMO_NO_BRACE"):
        brace_rot = down(base_yaw) @ _rot_axis(pinch_local, math.radians(-35))
        brace = base.to_world((-0.035, 0.0, 0.0))
        yield from L.move(brace + [0, 0, 0.09], brace_rot, 3.0, grip=1.0)
        yield from L.move(brace, brace_rot, 1.5)
        yield from L.set_grip(0.0, 1.2)
        yield from wait(0.3)

    # 4. One look at the case once the brace has settled (gripping it nudges it): the case is a fixed part in plain
    #    view of the cameras. Everything after this is open loop from that pose. The right arm presses the middle of
    #    the lid's front edge.
    from yam_arena.constants import PAD_TIP_PAST_TCP

    yield from wait(0.5)
    base = ctx.states()["case_base"]
    if os.environ.get("DEMO_VARIANT") == "no_press":   # negative variant: lid placed, never pressed
        yield from L.set_grip(1.0, 0.8)
        yield from L.move(L.pos + [0, 0, 0.08], L.rot, 1.5)
        return
    base_yaw = math.atan2(base.rotation()[1, 0], base.rotation()[0, 0])

    lid_top = T.FLOOR_Z + T.LATCH_TRAVEL + 2 * T.LID_HALF
    over = base.to_world((0.045, 0.0, 0.06))
    yield from R.move(over, down(base_yaw), 2.0, grip=0.0)
    top = lid_top + PAD_TIP_PAST_TCP + 0.003
    yield from R.move(base.to_world((0.045, 0.0, top)), down(base_yaw), 1.2)
    # A seated lid's top is flush with the rim, LATCH_TRAVEL below where it rests on the raised latches. Command the pad
    # 20 mm past flush at 6 mm/s, without reading whether the latches clicked: the arm's position control gives under
    # the latch springs' load, so the pad itself ends up only about 1 mm below flush (12 mm past leaves the lid short).
    final = 0.003 + T.LATCH_TRAVEL + 0.020
    depth = 0.0
    while depth < final:
        depth = min(final, depth + 0.006 / CONTROL_HZ)
        R.aim(base.to_world((0.045, 0.0, top - depth)), down(base_yaw))
        yield
    yield from wait(0.5)
    yield from R.move(over, R.rot, 1.0)
    yield from both(R.move(over + [0, 0, 0.05], R.rot, 1.5), L.set_grip(1.0, 0.8))
    yield from L.move(L.pos + [0, 0, 0.08], L.rot, 1.5)


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
        cam_height: int = 176, hold_s: float = 2.0) -> dict:
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
    base0 = ctx.states()["case_base"].pos.copy()
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
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % 20 == 0 and steps > 400):
            msg = f"step {steps} score {info.get('score')} latched {task._STATE['latched']}"
            if os.environ.get("DEMO_DEBUG"):
                st = ctx.states()
                b, lid = st["case_base"], st["lid"]
                corners = [1000 * (c[2] - task.FLOOR_Z) for c in task._corner_heights(b, lid)]
                msg += (f" base_moved_mm {1000 * np.linalg.norm(b.pos - base0):.1f} corner_gap_mm {np.round(corners, 2)}"
                        f" lid_rel_mm {np.round(1000 * b.to_local(lid.pos), 1)} R_cmd_z_mm {1000 * ctx.right.pos[2]:.1f}")
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "latched": task._STATE["latched"],
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
