"""Scripted full-score demo for the lever-and-dunnage task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The right arm grips the lever's knob and presses it down round its pin, tracking the measured lever angle each step
(the gripper pitches with the knob), until the lever reaches its lower stop and the toe has lifted the crate's front
edge about 25 mm. Holding it there, the left arm picks up the wooden block by its back end, brings it low to the
crate's front face in its lane and slides it 40 mm under the raised edge, sets it down and lets go. Then the right arm
lets the lever back up slowly, so the crate comes down onto the block, opens and lifts away.

Negative variants (DEMO_VARIANT=...):
  no_hold        the lever is never pressed: the block is pushed against the crate's closed front edge (0)
  early_release  press the lever, then let it back up before the block goes in: the edge is down again (0.25)
  short          the block goes in only 10 mm before the crate is let down on it (0.25)
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


def knob_point(stand, task, angle):
    """World TCP position on the lever's knob (4 mm below its top, on its centre line) at lever ``angle`` (rad)."""
    x, z = task.KNOB_X, task.GRIP_Z
    c, s = math.cos(angle), math.sin(angle)
    px, py, pz = task.PIVOT
    return np.asarray(stand.to_world((px + x * c + z * s, py, pz - x * s + z * c)), float)


def knob_rot(stand, angle, yaw0):
    """The gripper's rotation on the knob: pointing down with ``yaw0``, pitched with the lever about its pin."""
    from yam_arena.demo import grip_rotation

    rs = stand.rotation()
    return rs @ _rot_axis((0, 1, 0), angle) @ rs.T @ grip_rotation(yaw0)


class Lever:
    """Closed loop on the right arm: each step the TCP goes to the knob's place at the command angle, which moves
    toward ``goal`` at most ``rate`` rad/s and at most ``lead`` rad ahead of the measured lever angle, so the fingers
    push the knob round its pin instead of dragging the stand."""

    def __init__(self, task, ctx, arm, yaw0, rate=0.25, lead=0.08):
        self.T, self.ctx, self.arm, self.yaw0, self.rate, self.lead = task, ctx, arm, yaw0, rate, lead
        self.goal, self.cmd, self.on = 0.0, None, False

    def angle(self):
        return self.T.lever_angle(self.ctx.states()["lever_stand"])

    def track(self):
        if not self.on:
            return
        stand = self.ctx.states()["lever_stand"]
        a = self.T.lever_angle(stand)
        self.cmd = a if self.cmd is None else self.cmd
        step = self.rate / CONTROL_HZ
        if self.goal < self.cmd:
            self.cmd = max(self.cmd - step, a - self.lead, self.goal)
        else:
            self.cmd = min(self.cmd + step, a + self.lead, self.goal)
        self.arm.aim(knob_point(stand, self.T, self.cmd), knob_rot(stand, self.cmd, self.yaw0))

    def wrap(self, gen):
        """Run another arm's generator while this one keeps tracking the lever."""
        for _ in gen:
            self.track()
            yield

    def wait(self, seconds):
        yield from self.wrap(wait(seconds))

    def go(self, goal, max_s=8.0, tol=0.03):
        """Turn the lever to ``goal`` and stay until it is within ``tol`` rad of it (or ``max_s`` passes)."""
        self.goal = goal
        for k in range(int(max_s * CONTROL_HZ)):
            self.track()
            yield
            if abs(self.cmd - goal) < 1e-6 and abs(self.angle() - goal) < tol:
                break


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    T = task
    L, R = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in ("", "no_hold", "early_release", "short"):
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    s0 = ctx.states()
    stand, crate, block = s0["lever_stand"], s0["crate"], s0["block"]
    up = np.array([0, 0, 0.05])

    # Right arm: the knob, fingers closing across the lever (world y). Left arm: the block's back end, fingers
    # closing across its width, pad tips 4 mm above the table.
    knob0 = knob_point(stand, T, 0.0)
    yaw_r = reachable_yaw(R, knob0, 0.0)
    rot_r = knob_rot(stand, 0.0, yaw_r)
    grip_dx = T.BLOCK_GRIP_BACK - T.BLOCK_L / 2      # the grip point, behind the block's centre along its length
    tool_z = T.TOP + 0.004 + 0.013
    g0 = np.asarray(block.to_world((grip_dx, 0.0, 0.0)), float)
    g0[2] = tool_z
    yaw_l = reachable_yaw(L, g0, 0.0)
    rot_l = grip_rotation(yaw_l)
    # Where the grip point goes: the block's front end DEPTH behind the crate's front face, in the block lane.
    depth = 0.010 if variant == "short" else T.DEPTH
    goal = np.asarray(crate.to_world((depth - T.BLOCK_L + T.BLOCK_GRIP_BACK, T.BLOCK_Y, 0.0)), float)
    goal[2] = tool_z + 0.002
    inward = crate.rotation()[:, 0]
    pre = goal - inward * (depth + 0.030)            # the block's front end 30 mm in front of the crate's face

    lever = Lever(T, ctx, R, yaw_r)

    # 1. Both arms above their parts, then down; close.
    if variant == "no_hold":
        yield from L.move(g0 + up, rot_l, 2.5, grip=1.0)
    else:
        yield from both(R.move(knob0 + up, rot_r, 2.5, grip=1.0), L.move(g0 + up, rot_l, 2.5, grip=1.0))
        yield from both(R.move(knob0, rot_r, 1.5), L.move(g0, rot_l, 1.5))
        yield from both(R.set_grip(0.0, 1.0), L.set_grip(0.0, 1.0))
        lever.on = True
        # 2. Press the lever to its lower stop: the toe lifts the crate's front edge.
        yield from lever.go(T.PRESSED - 0.03, tol=0.05)   # a little past the stop: the fingers hold it there
        yield from lever.wait(0.5)
    if variant == "no_hold":
        yield from L.move(g0, rot_l, 1.5)
        yield from L.set_grip(0.0, 1.0)
    if variant == "early_release":
        yield from lever.go(0.0)
        yield from lever.wait(0.5)

    # 3. Lift the block, bring it to the crate's front face in its lane, low, and slide it under the raised edge.
    yield from lever.wrap(L.move(g0 + np.array([0, 0, 0.030]), rot_l, 1.0))
    yield from lever.wrap(L.move(pre + np.array([0, 0, 0.030]), rot_l, 2.0))
    yield from lever.wrap(L.move(pre, rot_l, 1.0))
    yield from lever.wrap(L.move(goal, rot_l, 2.5))
    yield from lever.wait(0.3)

    # 4. Set it down and let go of it.
    yield from lever.wrap(L.move(goal - np.array([0, 0, 0.002]), rot_l, 0.5))
    yield from lever.wrap(L.set_grip(1.0, 0.8))
    yield from lever.wrap(L.move(L.pos + np.array([0, 0, 0.04]) - inward * 0.03, rot_l, 1.0))

    # 5. Let the lever back up slowly: the crate comes down onto the block. Then let go of the knob.
    if lever.on:
        lever.rate = 0.15
        yield from lever.go(0.0)
        yield from lever.wait(0.3)
        lever.on = False
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
                depth, lane = task.block_under(st["crate"], st["block"])
                msg += (f" front_mm {1000 * task.front_height(st['crate']):.1f}"
                        f" lever_deg {math.degrees(task.lever_angle(st['lever_stand'])):.1f}"
                        f" depth_mm {1000 * depth:.1f} lane_mm {1000 * lane:.1f}"
                        f" block_z_mm {1000 * (st['block'].pos[2] - task.TOP):.1f}"
                        f" failed {task._STATE['failed']} L_mm {np.round(1000 * ctx.left.pos, 1)}"
                        f" R_mm {np.round(1000 * ctx.right.pos, 1)}")
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
