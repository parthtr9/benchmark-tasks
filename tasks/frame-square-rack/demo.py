"""Scripted full-score demo for the face-frame squaring task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Open loop: the layout of the task file, timing and the arms' own commands only. The left arm closes its fingers (along x)
on the far-left corner block of the frame and the right arm on the near-right corner block, the two ends of the long
diagonal. After half a second they move toward each other (left to -y, right to +y) in lockstep over 4 s, each by 4 mm
plus an allowance for the arms' lag under the joint's 2 N friction (DEMO_ALLOW mm, default set below), hold 0.5 s, open,
and lift away 6 cm.

Negative variants (DEMO_VARIANT=...):
  one_arm          only the left arm, pulling its corner 8 mm to -y: the whole frame slides, the rack stays (0)
  wrong_diagonal   both hands on the short diagonal's corners, squeezed: the rack grows (0)
  short            about 60% of the squeeze: not squared enough (0.25)
  no_release       squeeze right but never let go (0.5)
DEMO_DEBUG=1 prints (and saves in result.json) a trace every 10 steps for every variant: finger points against the
grip blocks at close, diagonals, frame displacement and each arm's lag behind its command.
The trace also logs the minimum arm-to-arm distance (URDF collision meshes, clearance.py) and result.json keeps its minimum.
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



ALLOW_MM = 12.0          # allowance per arm for the arms' lag under the joint's 2 N friction (calibrate on the GPU box)
VARIANTS = ("", "one_arm", "wrong_diagonal", "short", "no_release")


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    T = task
    L, R = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    allow = float(os.environ.get("DEMO_ALLOW", ALLOW_MM)) / 1000
    z = T.TOP + 0.029                                   # pad tips 16 mm above the table: over the 10 mm stiles
    far_x, near_x = T.NEAR_X + T.SPAN, T.NEAR_X
    inset = 0.010                                       # grip 10 mm inboard of the block centre (pads are 27 mm wide)
    if variant == "wrong_diagonal":                     # short diagonal: near-left (left arm), far-right (right arm)
        pl = np.array([near_x, T.STILE_Y - inset, z])
        pr = np.array([far_x, T.S0 - T.STILE_Y + inset, z])
    else:                                               # long diagonal: far-left (left arm), near-right (right arm)
        pl = np.array([far_x, T.S0 + T.STILE_Y - inset, z])
        pr = np.array([near_x, -T.STILE_Y + inset, z])
    move = 0.0118 if variant == "short" else 0.004 + allow   # short: about 60% of the squeeze
    ctx.grips = {"left": pl, "right": pr}
    ctx.swapped = variant == "wrong_diagonal"
    yaw_l = reachable_yaw(L, pl, math.pi / 2)
    yaw_r = reachable_yaw(R, pr, math.pi / 2)
    rot_l, rot_r = grip_rotation(yaw_l), grip_rotation(yaw_r)
    up = np.array([0.0, 0.0, 0.06])
    dy = np.array([0.0, 1.0, 0.0])

    one_arm = variant == "one_arm"
    if one_arm:
        yield from L.move(pl + up, rot_l, 2.5, grip=1.0)
        yield from L.move(pl, rot_l, 1.5)
        yield from L.set_grip(0.0, 1.0)
        ctx.mark("closed")
        yield from wait(0.5)
        yield from L.move(pl - 0.008 * dy, rot_l, 2.0)
        ctx.mark("squeezed")
        yield from wait(0.5)
        yield from L.set_grip(1.0, 0.8)
        yield from wait(0.3)
        yield from L.move(L.pos + up, rot_l, 1.0)
        return

    # 1. Both arms above their corner blocks, then down; close.
    yield from both(L.move(pl + up, rot_l, 2.5, grip=1.0), R.move(pr + up, rot_r, 2.5, grip=1.0))
    yield from both(L.move(pl, rot_l, 1.5), R.move(pr, rot_r, 1.5))
    yield from both(L.set_grip(0.0, 1.0), R.set_grip(0.0, 1.0))
    ctx.mark("closed")
    yield from wait(0.5)
    # 2. Squeeze: the left hand to -y and the right hand to +y in lockstep.
    yield from both(L.move(pl - move * dy, rot_l, 4.0), R.move(pr + move * dy, rot_r, 4.0))
    ctx.mark("squeezed")
    yield from wait(1.5)
    if variant == "no_release":
        yield from wait(4.0)
        return
    # 3. Let go and lift away.
    yield from both(L.set_grip(1.0, 0.8), R.set_grip(1.0, 0.8))
    yield from wait(0.3)
    yield from both(L.move(L.pos + up, rot_l, 1.0), R.move(R.pos + up, rot_r, 1.0))
    yield from wait(1.5)


def _measured_tcp(ctx, name):
    """The arm's own joint readings turned into a TCP position (the demo's logging only)."""
    from yam_arena import ik
    from yam_arena.constants import ARM_JOINTS

    robot = ctx.emb._env.unwrapped.scene[f"{name}_robot"]
    names = list(robot.joint_names)
    jp = robot.data.joint_pos[0].detach().cpu().numpy()
    q = np.array([jp[names.index(j)] for j in ARM_JOINTS], float)
    pos, _ = ik.tcp_pose(q, ctx.left.base if name == "left" else ctx.right.base)
    return np.asarray(pos, float)


def _clearance(ctx, task):
    """Smallest distance between any collision mesh of one arm and the other arm or the part the other holds, at the
    commanded joints (the URDF meshes, see clearance.py)."""
    import clearance

    near = clearance.box_points((task.NEAR_X, 0.0, task.TOP + 0.015), (task.RAIL_W, task.RAIL_L, task.RAIL_H))
    far = clearance.box_points((task.NEAR_X + task.SPAN, task.S0, task.TOP + 0.015),
                               (task.RAIL_W, task.RAIL_L, task.RAIL_H))
    left, right = ctx.left, ctx.right
    if getattr(ctx, "swapped", False):                  # wrong_diagonal: the left hand holds the near rail
        near, far = far, near
    return clearance.min_clearance(left.q, left.grip, left.base, right.q, right.grip, right.base, held_a=far,
                                   held_b=near)


def _trace(ctx, task, steps, event=None):
    """One logging row: finger points against the grip blocks, diagonals, frame displacement, rack, each arm's lag."""
    st = ctx.states()["frame"]
    a, b = task.diagonals(st)
    row = {"step": steps, "event": event, "rack_mm": round(1000 * task.rack(st), 2),
           "long_mm": round(1000 * a, 2), "short_mm": round(1000 * b, 2), "diff_mm": round(1000 * (a - b), 2),
           "centre_mm": [round(1000 * float(v), 1) for v in (st.pos[:2] + st.rotation()[:2, :2] @ [task.SPAN / 2,
                         task.rack(st) / 2] - np.array(task.CENTRE))],
           "yaw_deg": round(math.degrees(math.atan2(st.rotation()[1, 0], st.rotation()[0, 0])), 2),
           "failed": task._STATE["failed"], "why": task._STATE["why"], "released": task._STATE["released"]}
    try:
        for n, arm in (("left", ctx.left), ("right", ctx.right)):
            row[f"lag_{n}_mm"] = [round(1000 * float(v), 1) for v in (arm.pos - _measured_tcp(ctx, n))]
    except Exception as exc:  # noqa: BLE001
        row["lag_error"] = repr(exc)
    try:
        d, names = _clearance(ctx, task)
        row["arm_clear_mm"], row["arm_clear_pair"] = round(1000 * d, 1), list(names)
        if d < ctx.min_clear[0]:
            ctx.min_clear = (d, names, steps)
    except Exception as exc:  # noqa: BLE001
        row["clear_error"] = repr(exc)
    fp = task._STATE.get("fingers")
    if fp is not None and event in ("closed", "squeezed"):
        # finger points in the frame's own axes (x across the rail, y along it) relative to each grip block centre
        row["fingers_vs_block_mm"] = [[round(1000 * float(v), 1) for v in st.to_local(p)[:2]] for p in fp]
        row["grips_local_mm"] = {n: [round(1000 * float(v), 1) for v in st.to_local(p)[:2]]
                                 for n, p in ctx.grips.items()}
    return row


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.events, self.grips, self.min_clear = [], {}, (9.0, (), 0)

    def mark(self, event):
        self.events.append(event)

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
    trace, debug = [], bool(os.environ.get("DEMO_DEBUG"))
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        event = ctx.events.pop(0) if ctx.events else None
        ctx.events.clear()
        if debug and (event or steps % 10 == 0):
            row = _trace(ctx, task, steps, event)
            trace.append(row)
            print("TRACE " + json.dumps(row), flush=True)
        elif steps % 20 == 0:
            try:
                d, names = _clearance(ctx, task)
                if d < ctx.min_clear[0]:
                    ctx.min_clear = (d, names, steps)
            except Exception as exc:  # noqa: BLE001
                print(f"clearance check failed: {exc!r}", flush=True)
        if not debug and steps % 100 == 0:
            print(f"step {steps} score {info.get('score')}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "variant": os.environ.get("DEMO_VARIANT", ""), "allow_mm": float(os.environ.get("DEMO_ALLOW", ALLOW_MM)),
              "final": _trace(ctx, task, steps, "final"), "failed": task._STATE["failed"], "why": task._STATE["why"],
              "min_arm_clearance_mm": round(1000 * ctx.min_clear[0], 1), "min_arm_clearance_pair": list(ctx.min_clear[1]),
              "min_arm_clearance_step": ctx.min_clear[2], "peak_rack_mm": round(1000 * task._STATE["peak_rack"], 2), "trace": trace}
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
