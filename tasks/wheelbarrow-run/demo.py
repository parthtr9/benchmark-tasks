"""Scripted full-score demo for the wheelbarrow task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Open loop from the task file's fixed layout: nothing measured from the barrow or the bricks steers it. Each arm grips its
round handle grip; together they lift the handles 4.5 cm (the barrow pitches about its axle and balances on the wheel,
legs about 2.5 cm up), push it 12 cm forward so the wheel rolls onto the orange line, lower it onto its legs, let go
and lift clear. Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder.

DEMO_VARIANT=one_handle lifts and pushes with the right arm only; DEMO_VARIANT=no_lift grips both handles and pushes
without lifting; DEMO_VARIANT=tray_grip lifts and pushes with one arm by the middle of the tray's back wall. All must
score 0.
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
LIFT = 0.045                       # handle grips raised 4.5 cm: the legs clear the table by about 2.5 cm
GRIP_Z = 0.030                     # the grips' axis above the axle (barrow frame)


def _rot_y(a) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def barrow_pose(task, pitch, advance):
    """The barrow with its handles raised by ``pitch`` about the axle and the axle moved ``advance`` along +x."""
    R = _rot_y(pitch)              # +pitch about y raises the handles (they lie along -x)
    return task.AXLE + np.array([advance, 0.0, 0.0]), R


def grip_point(task, side, pose, x=None):
    p, R = pose
    y = task.HANDLE_Y if side == "left" else -task.HANDLE_Y
    return p + R @ np.array([task.GRIP_X if x is None else x, y, GRIP_Z])


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


def _grip_yaw(task, arms) -> float:
    """0 or pi: the turn of the straight-down grip (fingers across the round grip either way) that IK reaches best at
    the lifted and pushed poses. Joint 6's limits rule one of them out in parts of the workspace."""
    from yam_arena import ik
    from yam_arena.constants import READY
    from yam_arena.demo import grip_rotation

    pitch = math.asin(LIFT / abs(task.GRIP_X))
    worst = {}
    for flip in (0.0, math.pi):
        errors = []
        for pose in (barrow_pose(task, 0.0, 0.0), barrow_pose(task, pitch, 0.0), barrow_pose(task, pitch, task.TRAVEL)):
            for side, arm in arms.items():
                rot = pose[1] @ grip_rotation(flip)
                _, pe, _ = ik.solve(grip_point(task, side, pose), ik.matrix_to_quat_xyzw(rot),
                                    np.array(READY, float), base_pos=arm.base, iters=200)
                errors.append(pe)
        worst[flip] = max(errors)
    return min(worst, key=worst.get)


def together(*gens):
    """Step several arm generators in lockstep until all have finished."""
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
    variant = os.environ.get("DEMO_VARIANT", "")
    arms = {"left": ctx.left, "right": ctx.right}
    flip = _grip_yaw(T, arms)
    sides = ("right",) if variant == "one_handle" else ("left", "right")
    pitch = 0.0 if variant == "no_lift" else math.asin(LIFT / abs(T.GRIP_X))

    def to(pose, duration, grip=None):
        yield from together(*(arms[s].move(grip_point(T, s, pose), pose[1] @ grip_rotation(flip), duration, grip)
                              for s in sides))

    if variant == "tray_grip":
        yield from tray_grip(T, ctx.right, flip)
        return
    rest = barrow_pose(T, 0.0, 0.0)
    # 1. Above the grips, then down onto them, and close.
    yield from together(*(arms[s].move(grip_point(T, s, rest) + [0, 0, 0.06], rest[1] @ grip_rotation(flip), 3.0, 1.0)
                          for s in sides))
    yield from to(rest, 1.5)
    yield from together(*(arms[s].set_grip(0.0, 1.0) for s in sides))
    yield from wait(0.3)
    # 2. Lift the handles together: it balances on its wheel. 3. Wheel it forward. 4. Set it down at the line.
    yield from to(barrow_pose(T, pitch, 0.0), 2.0)
    yield from to(barrow_pose(T, pitch, T.TRAVEL), 4.0)
    yield from to(barrow_pose(T, 0.0, T.TRAVEL), 2.0)
    yield from wait(0.3)
    # 5. Let go and lift clear.
    yield from together(*(arms[s].set_grip(1.0, 0.8) for s in sides))
    yield from together(*(arms[s].move(arms[s].pos + [0, 0, 0.07], arms[s].rot, 1.5) for s in sides))


def tray_grip(task, arm, flip):
    """One arm grips the middle of the tray's back wall and lifts and pushes from there."""
    from yam_arena.demo import grip_rotation

    p, R = barrow_pose(task, 0.0, 0.0)
    g = p + R @ np.array([-0.187, 0.0, 0.050])
    rot = R @ grip_rotation(flip + math.pi / 2)
    yield from arm.move(g + [0, 0, 0.06], rot, 3.0, 1.0)
    yield from arm.move(g, rot, 1.5)
    yield from arm.set_grip(0.0, 1.0)
    yield from arm.move(g + [0, 0, 0.03], rot, 2.0)
    yield from arm.move(g + [task.TRAVEL, 0, 0.03], rot, 4.0)
    yield from arm.move(g + [task.TRAVEL, 0, 0], rot, 2.0)
    yield from arm.set_grip(1.0, 0.8)
    yield from arm.move(arm.pos + [0, 0, 0.07], arm.rot, 1.5)


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


def arm_link_gap(emb) -> float:
    """Smallest distance (m) between any link origin of the left arm and any of the right arm: a coarse check that the
    two arms stay apart while both hold the barrow."""
    base = emb._env.unwrapped
    pts = {}
    for arm in ("left_robot", "right_robot"):
        pose = base.scene[arm].data.body_link_pose_w
        pts[arm] = np.asarray(getattr(pose, "torch", pose).detach().cpu().numpy())[0][:, :3]
    d = pts["left_robot"][:, None, :] - pts["right_robot"][None, :, :]
    return float(np.sqrt((d * d).sum(-1)).min())


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
    min_gap = float("inf")
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        try:
            min_gap = min(min_gap, arm_link_gap(emb))
        except Exception:  # noqa: BLE001 - the check is evidence only
            pass
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % 10 == 0):
            msg = f"step {steps} score {info.get('score')}"
            if os.environ.get("DEMO_DEBUG"):
                st = ctx.states()
                b = st["wheelbarrow"]
                R = b.rotation()
                heights = [1000 * float(f[2] - task.TOP) for f in task.legs(b)]
                msg += (f" axle_mm ({1000 * (b.pos[0] - task.AXLE[0]):.1f}, {1000 * b.pos[1]:.1f}) legs_mm "
                        f"({heights[0]:.1f}, {heights[1]:.1f}) roll {math.degrees(math.asin(max(-1, min(1, R[2, 1])))):.1f}"
                        f" pitch {math.degrees(math.asin(max(-1, min(1, -R[2, 0])))):.1f} drag_mm {1000 * task._STATE['drag']:.1f}"
                        f" flags tipped {task._STATE['tipped']} spilled {task._STATE['spilled']} dragged "
                        f"{task._STATE['dragged']} touched {task._STATE['touched']} released {task._STATE['released']}")
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "variant": os.environ.get("DEMO_VARIANT", ""),
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "min_arm_link_gap_mm": round(1000 * min_gap, 1) if min_gap < float("inf") else None}
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
