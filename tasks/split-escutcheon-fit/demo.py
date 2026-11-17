"""Scripted full-score demo for split-escutcheon-fit, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Open loop: every target comes from the task file's fixed layout and the clock. No object pose or task state steers it.

1. Both hands come in from the sides, level, and pinch the plate standing in its block across its 11.4 mm depth, each
   on the middle of one half.
2. They lift it out of the groove and spread it open at the split: each half turns 0.25 rad about the top of the ring,
   the hands turning with it.
3. They carry it over the valve to its place 52 mm off the wall and lower it so the open split passes over the pipe.
4. They turn the halves back, so it springs shut round the pipe, let go and draw out sideways.
5. The right hand, pointing down, tipped 15 degrees toward the wall, its fingers open either side of the pipe, slides
   the plate along the pipe until it is flat against the wall, and moves away.

Negative variants (DEMO_VARIANT=...):
  no_push    shut round the pipe but left where it was put (0.5)
  one_arm    the right hand alone takes the plate by one half and lowers it onto the pipe: the shut split lands on the
             pipe (0)
  no_spread  both hands, but the plate is not spread: the same (0)
Writes one MP4 per camera, setup.png and result.json; DEMO_DEBUG=1 prints a trace every 10 steps.
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
VARIANTS = ("", "no_push", "one_arm", "no_spread")
KI, CORR_MAX = 0.2, 0.02        # joint-feedback trim: gain per step and limit (m)


def fwd(yaw: float = 0.0, pitch: float = 0.0, roll: bool = False) -> np.ndarray:
    """A forward-pointing grip (columns: closing axis, the pads' width axis, approach axis), turned ``yaw`` about
    the vertical and pitched ``pitch`` (negative: nose down). ``roll``: turned a quarter turn about its approach
    axis, so the pads are above and below."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    a = np.array([cy * cp, sy * cp, sp])
    c = np.array([-cy * sp, -sy * sp, cp]) if roll else np.array([-sy, cy, 0.0])
    return np.c_[c, np.cross(a, c), a]


FLIP = np.diag([-1.0, -1.0, 1.0])       # the same parallel grip turned half a turn about its approach axis


def best_grip(arm, positions, rot):
    """The grip (rot or rot turned half a turn about its approach) whose worst IK error over ``positions``, solved one
    after another from the arm's last command, is smaller, with the wrist roll kept off its +-120 degree limit."""
    from yam_arena import ik

    positions = [positions] if np.ndim(positions) == 1 else positions
    out = []
    for R in (rot, rot @ FLIP):
        q, worst = arm.q.copy(), 0.0
        for pos in positions:
            q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(R), q, base_pos=arm.base, iters=200)
            worst = max(worst, pe + 0.05 * re + 0.01 * max(0.0, abs(float(q[5])) - 1.9))
        out.append((worst, R))
    return min(out, key=lambda t: t[0])[1]


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each from where the last command left it. With ``servo`` the
    command is trimmed by the arm's own joint readings so the measured tool point follows the plan under load."""

    def __init__(self, name: str, base_pos, read=None):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.pos = np.asarray(self.pos, float)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.read = read
        self.corr = np.zeros(3)
        self.corr_max = CORR_MAX
        self.max_corr = 0.0

    def _solve(self, pos, rot, servo=None):
        from yam_arena import ik

        if servo is not None and self.read is not None:
            meas = self.read()
            if meas is not None:
                self.corr = np.clip(self.corr + KI * np.asarray(servo) * (self.pos - meas[0]), -self.corr_max,
                                    self.corr_max)
                self.max_corr = max(self.max_corr, float(np.abs(self.corr).max()))
        q, pe, _ = ik.solve(np.asarray(pos, float) + self.corr, ik.matrix_to_quat_xyzw(rot), self.q,
                            base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None, servo=None):
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s), servo)
            yield

    def hold(self, duration: float, grip: float | None = None, servo=None):
        yield from self.move(self.pos, self.rot, duration, grip, servo)

    def relax(self, duration: float = 0.5):
        c0 = self.corr.copy()
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self.corr = c0 * (1 - k / n)
            self._solve(self.pos, self.rot)
            yield


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


def chain(*gens):
    for g in gens:
        yield from g


def shift(arm, d, rot, duration: float, grip: float | None = None, servo=None):
    """Generator: move by ``d`` from wherever the arm's command is when this step starts (not when it was built)."""
    yield from arm.move(arm.pos + np.asarray(d, float), rot, duration, grip, servo)


def stay(arm, rot, duration: float, grip: float | None = None):
    """Generator: stay where the arm is when this step starts, turning to ``rot`` and setting the grip."""
    yield from arm.move(arm.pos, rot, duration, grip)


# -- the plan ------------------------------------------------------------------------------------------------------------
R_TCP = 0.026          # each hand's tool point this far out from the plate's axis, on its half's middle (pads reach to 13 mm)
G_OPEN = 0.26          # finger opening (1 = 94 mm) coming onto the plate's rim: 6 mm a side
G_RELEASE = 0.40       # and letting it go: 13 mm a side, so the pads leave the halves without dragging them
G_PINCH = 0.0          # closed on the plate's 11.4 mm depth
LIFT = 0.015           # out of the block's groove
SPREAD = 0.28          # each half turned this far (rad) about the top: the split opens by 0.56 rad
OFF_WALL = 0.052       # the plate's back edge this far off the wall while it goes onto the pipe (the hands' bodies clear)
OVER_VALVE = 0.075     # the plate's origin this far above the pipe's axis while it is carried over the valve
DROP = 0.048           # ... and this far above it over its place on the pipe, before it is lowered
HANG = -0.00035        # hanging on the pipe, the bore's top flat rests on the pipe's top
PUSH_PITCH = math.radians(30.0)   # the pushing hand points down, tipped this much toward the wall (its body and finger
#                                   roots lean back, clear of the wall, while the pads' lower front corners lead)
G_STRADDLE = 0.38      # its fingers open 36 mm, either side of the pipe and clear of the valve
PUSH_LOW = 0.0         # its tool point this far below the pipe's axis (the fingertips straddle the pipe)
PUSH_PAST = 0.003      # pushed this far past flat against the wall
PUSH_LEAD = 0.0122     # tipped 30 degrees, the fingers' front faces reach this far past the tool point at its height
#                        (measured on the gripper's URDF collision meshes; higher up they trail behind it)


def side(s: int) -> np.ndarray:
    """A sideways grip from the arm's own side (s=+1: the left arm, pointing -y), fingers closing along x."""
    a = np.array([0.0, -float(s), 0.0])
    c = np.array([1.0, 0.0, 0.0])
    return np.c_[c, np.cross(a, c), a]


def rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def down(pitch: float = 0.0) -> np.ndarray:
    """Pointing down, fingers closing along y, tipped ``pitch`` toward +x (the tips lead)."""
    a = np.array([math.sin(pitch), 0.0, -math.cos(pitch)])
    c = np.array([0.0, 1.0, 0.0])
    return np.c_[c, np.cross(a, c), a]


def plan(T) -> dict:
    """Plate origins and tool points (world, m) from the task file's fixed layout."""
    p = {}
    o0 = np.array(T.PLATE_START)
    mid = np.array([-T.PLATE_H / 2000, 0.0, 0.0])                    # the middle of the plate's depth from its origin
    p["start"] = o0
    p["lifted"] = o0 + np.array([0.0, 0.0, LIFT])
    place_x = T.XW - OFF_WALL
    p["over_valve"] = np.array([o0[0], 0.0, T.PIPE_Z + OVER_VALVE])
    p["above"] = np.array([place_x, 0.0, T.PIPE_Z + DROP])
    p["above_high"] = np.array([place_x, 0.0, T.PIPE_Z + OVER_VALVE])
    p["hung"] = np.array([place_x, 0.0, T.PIPE_Z + HANG])
    p["pivot"] = np.array(T.PIVOT) / 1000
    p["mid"] = mid
    lead = PUSH_LEAD
    p["push_end"] = np.array([T.XW - T.PLATE_H / 1000 - lead + PUSH_PAST, 0.0, T.PIPE_Z - PUSH_LOW])
    p["push_start"] = np.array([T.XW - OFF_WALL - T.PLATE_H / 1000 - lead - 0.006, 0.0, T.PIPE_Z - PUSH_LOW])
    return p


def hand_pose(P, origin, s: int, spread: float):
    """Hand s's tool point and rotation with the plate's origin at ``origin`` and each half turned ``spread``."""
    piv = origin + P["pivot"]
    R = rx(s * spread)
    rest = origin + P["mid"] + np.array([0.0, s * R_TCP, 0.0])
    return piv + R @ (rest - piv), R @ side(s)


def script(task, ctx):
    T, P = task, plan(task)
    variant = os.environ.get("DEMO_VARIANT", "")
    L, R = ctx.left, ctx.right
    hands = ((L, 1), (R, -1))
    rots = {}
    for arm, s in hands:
        p0, r0 = hand_pose(P, P["start"], s, 0.0)
        rots[s] = best_grip(arm, [p0, hand_pose(P, P["hung"], s, SPREAD)[0]], r0)
    flip = {s: not np.allclose(rots[s], hand_pose(P, P["start"], s, 0.0)[1]) for s in (1, -1)}

    def pose(origin, s, spread, out=0.0):
        p, r = hand_pose(P, origin, s, spread)
        return p + np.array([0.0, s * out, 0.0]), (r @ FLIP if flip[s] else r)

    def go(origin, spread, duration, grip=None, out=0.0, arms=hands, servo=None):
        return both(*[arm.move(*pose(origin, s, spread, out), duration, grip, servo) for arm, s in arms])

    spread = 0.0 if variant == "no_spread" else SPREAD
    grab = hands if variant != "one_arm" else ((R, -1),)
    # 1. both hands come in from the sides and pinch the plate's halves across its depth, at their middles
    yield from go(P["start"], 0.0, 2.5, G_OPEN, out=0.07, arms=grab)
    yield from go(P["start"], 0.0, 1.2, arms=grab)
    yield from go(P["start"], 0.0, 0.6, G_PINCH, arms=grab)
    # 2. lift it out of the groove and spread it open at the split, the halves turning about the top; from here the
    #    arms' own joint readings trim the commands (the ring's spring pushes the hands together)
    sv = np.ones(3)
    yield from go(P["lifted"], 0.0, 0.8, arms=grab, servo=sv)
    if variant != "one_arm":
        yield from go(P["lifted"], spread, 1.2, servo=sv)
    # 3. over the valve, above its place on the pipe, down over the pipe
    yield from go(P["over_valve"], spread, 1.2, arms=grab, servo=sv)
    yield from go(P["above_high"], spread, 1.8, arms=grab, servo=sv)
    yield from go(P["above"], spread, 0.8, arms=grab, servo=sv)
    yield from go(P["hung"], spread, 1.6, arms=grab, servo=sv)
    # 4. let it spring shut round the pipe, the hands following, then let go and draw the hands out sideways
    if variant != "one_arm":
        yield from go(P["hung"], 0.0, 1.2, servo=sv)
    yield from wait(0.3)
    yield from both(*[arm.relax(0.3) for arm, _ in grab])
    yield from go(P["hung"], 0.0, 0.6, G_RELEASE, arms=grab)
    yield from go(P["hung"], 0.0, 1.6, out=0.08, arms=grab)
    park_l = np.array([0.20, 0.13, T.PIPE_Z + 0.08])
    park_r = np.array([0.20, -0.13, T.PIPE_Z + 0.08])
    if variant == "no_push":
        yield from both(L.move(park_l, L.rot, 1.5), R.move(park_r, R.rot, 1.5))
        yield from wait(1.5)
        return
    # 5. the right hand, pointing down with its fingers either side of the pipe, slides the plate back to the wall
    pr = down(PUSH_PITCH)
    ps, pe = P["push_start"], P["push_end"]
    up = np.array([0.0, 0.0, 1.0])
    yield from both(L.move(park_l, L.rot, 1.5),
                    chain(shift(R, np.array([-0.05, 0.0, 0.05]), R.rot, 1.0),
                          R.move(ps + 0.09 * up, pr, 2.0, G_STRADDLE), R.move(ps, pr, 1.5)))
    sv = np.ones(3)                 # the arm's own joint readings trim the push: it keeps the pads at the pipe's height
    yield from R.move(pe, pr, 2.5, servo=sv)   # (the reaction sags the arm) and presses harder where the plate drags
    yield from R.hold(0.6, servo=sv)
    yield from R.relax(0.4)
    yield from R.move(pe - np.array([0.012, 0.0, 0.0]), pr, 0.6)
    yield from R.move(pe - np.array([0.012, 0.0, -0.09]), pr, 1.0)
    yield from R.move(park_r, pr, 1.5)
    yield from wait(1.5)


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"], read=lambda: self.joint_tcp("left"))
        self.right = Arm("right", station.BASES["right"], read=lambda: self.joint_tcp("right"))

    def joint_tcp(self, name: str):
        """The arm's tool point and gripper rotation from its own joint positions (forward kinematics)."""
        from partkit import _np
        from yam_arena import ik, station
        from yam_arena.constants import ARM_JOINTS

        art = self.emb._env.unwrapped.scene[f"{name}_robot"]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        pos, rot = ik.tcp_pose(_np(art.data.joint_pos)[0][ids], station.BASES[name])
        return np.asarray(pos, float), np.asarray(rot, float)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    variant = os.environ.get("DEMO_VARIANT", "")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    gen = script(task, ctx)
    frames: dict[str, list] = {}
    debug = os.environ.get("DEMO_DEBUG", "1") != "0"
    min_gap = float("inf")
    import clearance

    print("PLAN " + json.dumps({k: np.round(v - [0, 0, task.TOP], 4).tolist() for k, v in plan(task).items()}), flush=True)

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    last_gap = {"mm": None}

    def row(steps, info):
        """Logging only: reads simulator state for the trace."""
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)
        g = task.geometry(st)
        S = task._STATE
        r2 = lambda v: [round(float(x), 2) for x in v]  # noqa: E731
        return {"step": steps, "score": info.get("score"), "levels": S["levels"], "released": S["released"],
                "near_mm": round(float(S.get("near_mm", -1.0)), 1), "gap_mm": last_gap["mm"],
                "axis_off": round(g["axis_off_mm"], 2), "wall_gap": round(g["wall_gap_mm"], 2),
                "tilt_deg": round(math.degrees(g["tilt"]), 2), "spread": round(g["spread"], 3),
                "flags": [S["dropped"]],
                "plate": np.round(st["plate"].pos - [0, 0, task.TOP], 4).tolist(),
                "tcp": {"L_cmd": np.round(ctx.left.pos - [0, 0, task.TOP], 3).tolist(),
                        "L_meas": np.round(ctx.joint_tcp("left")[0] - [0, 0, task.TOP], 3).tolist(),
                        "R_cmd": np.round(ctx.right.pos - [0, 0, task.TOP], 3).tolist(),
                        "R_meas": np.round(ctx.joint_tcp("right")[0] - [0, 0, task.TOP], 3).tolist(),
                        "grip": [round(float(ctx.left.grip), 2), round(float(ctx.right.grip), 2)]}}

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    if debug:
        try:
            print("TRACE " + json.dumps(row(0, {}), default=str), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"trace unavailable: {exc!r}", flush=True)
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 5 == 0:
            try:
                gap, _ = clearance.min_clearance(ctx.left.q, ctx.left.grip, ctx.left.base, ctx.right.q,
                                                 ctx.right.grip, ctx.right.base)
                min_gap = min(min_gap, gap)
                last_gap["mm"] = round(1000 * gap, 1)
            except Exception as exc:  # noqa: BLE001 - logging only
                if steps == 5:
                    print(f"clearance unavailable: {exc!r}", flush=True)
        if debug and steps % 10 == 0:
            try:
                print("TRACE " + json.dumps(row(steps, info), default=str), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"trace unavailable: {exc!r}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    S = task._STATE
    result = {"task_score": score, "success": success, "steps": steps, "variant": variant,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "max_trim_mm": {"left": round(1000 * ctx.left.max_corr, 1), "right": round(1000 * ctx.right.max_corr, 1)},
              "min_arm_link_gap_mm": round(1000 * min_gap, 1) if min_gap < float("inf") else None,
              "levels": S["levels"], "part_dropped": S["dropped"], "finger_check": S["finger_check"],
              "final": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in S["geom"].items()}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)
    return result


def ik_check(task_path: str) -> None:
    """No simulator: yam_arena IK error along the whole planned path, solved step by step as the demo does."""
    import importlib

    from yam_arena import station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)

    class Dry:
        left, right = Arm("left", station.BASES["left"]), Arm("right", station.BASES["right"])

    ctx = Dry()
    n = 0
    for _ in script(task, ctx):
        n += 1
        if n % 20 == 0:
            print(f"IK step {n:4d} L {np.round(ctx.left.pos - [0, 0, task.TOP], 3).tolist()} worst {ctx.left.worst_mm:.1f} mm | "
                  f"R {np.round(ctx.right.pos - [0, 0, task.TOP], 3).tolist()} worst {ctx.right.worst_mm:.1f} mm", flush=True)
    print(f"IK steps {n} worst left {ctx.left.worst_mm:.2f} mm right {ctx.right.worst_mm:.2f} mm", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ik", action="store_true", help="only print the IK error of every planned target")
    a = ap.parse_args()
    if a.ik:
        ik_check(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
