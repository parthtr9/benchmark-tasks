"""Scripted full-score demo for the two-latch lid task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Each arm grips one latch tab (the left the far one, the right the near one). Both slide their tabs toward the lid's
middle together, releasing the latches, lift the lid out together, carry it to the right of the case and set it down.
DEMO_ONE_ARM=1 has only the right arm act, to show one latch can't free the lid.
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
    case = s0["case"]
    yaw = T.CASE_YAW
    # Tool point 3 mm below each tab's top: the pads hold its upper 13 mm and the fingertips stay 14 mm above the lid
    # (touching the lid, they drag it along and the tab can't slide).
    tab_z = T.TAB_TOP - 0.003
    far = case.to_world((T.TAB_X, 0.0, tab_z))     # left arm
    near = case.to_world((-T.TAB_X, 0.0, tab_z))   # right arm
    yaw_l, yaw_r = reachable_yaw(L, far, yaw), reachable_yaw(R, near, yaw)
    rot_l, rot_r = grip_rotation(yaw_l), grip_rotation(yaw_r)
    one_arm = bool(os.environ.get("DEMO_ONE_ARM"))  # only the right arm acts: one latch can't free the lid
    inward = case.rotation()[:, 0]                  # the case's +x axis: the far tab slides -x, the near tab +x
    pull = T.LATCH_TRAVEL + 0.006  # past the latch's 6 mm stop: the grip and wrist give a few millimetres

    # 1. Both arms above their tabs, then down onto them, and grip (fingers closing across each tab's 30 mm width).
    moves = [R.move(near + [0, 0, 0.06], rot_r, 2.5, grip=1.0)]
    if not one_arm:
        moves.append(L.move(far + [0, 0, 0.06], rot_l, 2.5, grip=1.0))
    yield from both(*moves)
    moves = [R.move(near, rot_r, 1.5)] + ([] if one_arm else [L.move(far, rot_l, 1.5)])
    yield from both(*moves)
    close = 1.0 if os.environ.get("DEMO_NOGRIP") else 0.0  # debug: slide with the fingers left open
    moves = [R.set_grip(close, 1.0)] + ([] if one_arm else [L.set_grip(close, 1.0)])
    yield from both(*moves)
    yield from wait(0.3)

    # 2. Slide both tabs toward the lid's middle together; the tongues leave the end walls.
    moves = [R.move(near + inward * pull, rot_r, 1.5)] + ([] if one_arm else [L.move(far - inward * pull, rot_l, 1.5)])
    yield from both(*moves)
    yield from wait(0.3)

    # 3. Lift together, holding the tabs in, then carry the lid to the right of the case.
    up = np.array([0, 0, 0.06])
    moves = [R.move(R.pos + up, rot_r, 2.0)] + ([] if one_arm else [L.move(L.pos + up, rot_l, 2.0)])
    yield from both(*moves)
    if one_arm:
        yield from wait(2.0)
        return
    side = case.rotation()[:, 1] * T.PLACE_DY
    yield from both(R.move(R.pos + side, rot_r, 3.0), L.move(L.pos + side, rot_l, 3.0))

    # 4. Lower until the lid sits on the table, let go of both tabs, and lift away.
    down = np.array([0, 0, -(0.06 + 0.04 - 0.002)])   # from 6 cm above the seated height to the table (lid bottom 21 mm)
    yield from both(R.move(R.pos + down, rot_r, 2.5), L.move(L.pos + down, rot_l, 2.5))
    yield from wait(0.3)
    yield from both(R.set_grip(1.0, 1.0), L.set_grip(1.0, 1.0))
    yield from both(R.move(R.pos + up, rot_r, 1.5), L.move(L.pos + up, rot_l, 1.5))


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
                c, tilt, height = task.lid_in_case(st["case"], st["lid"])
                tr = task.latch_travel(st["lid"])
                from yam_arena import ik
                rob = emb._env.unwrapped.scene["right_robot"]
                qa = np.asarray(getattr(rob.data.joint_pos, "torch", rob.data.joint_pos)[0].cpu())[:6]
                tcp_act = ik.tcp_pose(qa, ctx.right.base)[0]
                tab = st["lid"].to_world((-task.TAB_X, 0.0, task.TAB_TOP - 0.003))
                msg += (f" R_cmd_mm {np.round(1000 * ctx.right.pos, 1)} R_act_mm {np.round(1000 * tcp_act, 1)}"
                        f" near_tab_mm {np.round(1000 * tab, 1)}")
                msg += (f" lid_mm {np.round(1000 * c, 1)} tilt {tilt:.1f} height_mm {1000 * height:.1f}"
                        f" latches_mm {[round(1000 * v, 1) for v in tr.values()]}"
                        f" case_moved_mm {1000 * np.linalg.norm(st['case'].pos - task._STATE['case0']):.1f}")
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
