"""Scripted full-score demo for mahjong-wall-row-lift, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]
    python demo.py task.py --ik          (no simulator: IK of every target)

Open loop: every target comes from the task file's fixed layout and the clock; the only feedback is each arm's own
joint readings (its measured tool pose, which the plan is trimmed to under load). No tile's pose steers it.

1. Each hand comes straight down over one end tile of the near row, fingers closing across the row, and pinches it.
2. Both hands press toward each other (each commanded SQUEEZE past where the row stops it): the 17 tiles are now a beam.
   While they press, each hand keeps its fingers upright by its own joint readings, so the end tiles stay square to the
   row and the thrust passes through the tiles' faces.
3. Both lift 35 mm, carry the row 60 mm to above the far row, and lower it until it bears lightly on the far row.
4. Both ease the press off, open, and leave.

Negative variants (DEMO_VARIANT=...):
  one_arm        only the left hand works: it lifts its end tile and the rest of the row stays on the mat (0)
  light          the press is 6 mm instead of SQUEEZE: the row sags apart as it leaves the mat (0)
  early_release  the press is let go half way across, 35 mm up: the tiles fall in a heap (0)
  crooked        the row is set down 8 mm off the far row's line: on top, not tile over tile (0.5)
  no_trim        the hands do not keep their fingers upright: the press tilts the end tiles and the row breaks (0)
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
VARIANTS = ("", "one_arm", "light", "early_release", "crooked", "no_trim")
KI, CORR_MAX = 0.2, 0.03        # joint-feedback trim of the tool point: gain per step and limit (m)
KI_ROT, ROT_MAX = 0.2, 0.45     # and of the hand's direction: gain per step and limit (rad)

DOWN_X = None
FLIP = np.diag([-1.0, -1.0, 1.0])       # the same parallel grip turned half a turn about its approach axis


def _rotations():
    """yam_arena's straight-down grip turned a quarter turn: fingers closing along x, across the row."""
    global DOWN_X
    if DOWN_X is None:
        from yam_arena.demo import grip_rotation

        DOWN_X = grip_rotation(math.pi / 2)


def rotvec(R) -> np.ndarray:
    """Rotation vector of a rotation matrix (angle below pi)."""
    R = np.asarray(R, float)
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = float(np.linalg.norm(v))
    return np.zeros(3) if n < 1e-12 else v / n * ang


def rot_exp(v) -> np.ndarray:
    v = np.asarray(v, float)
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


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
    """One arm: smooth Cartesian moves (IK on the URDF), each from where the last command left it. With ``servo`` (a
    0/1 mask over x, y, z) the commanded tool point is trimmed by the arm's own joint readings so the measured one
    follows the plan on those axes under load; with ``square`` the hand's direction is trimmed the same way."""

    def __init__(self, name: str, base_pos, read=None):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.pos, self.rot = np.asarray(self.pos, float), np.asarray(self.rot, float)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.last_mm = 0.0
        self.read = read
        self.corr = np.zeros(3)
        self.corr_rot = np.zeros(3)
        self.max_corr = 0.0
        self.max_corr_rot = 0.0

    def _solve(self, pos, rot, servo=None, square=False):
        from yam_arena import ik

        meas = self.read() if (self.read is not None and (servo is not None or square)) else None
        if meas is not None and servo is not None:
            self.corr = np.clip(self.corr + KI * np.asarray(servo, float) * (self.pos - meas[0]), -CORR_MAX, CORR_MAX)
            self.max_corr = max(self.max_corr, float(np.abs(self.corr).max()))
        if meas is not None and square:
            self.corr_rot = np.clip(self.corr_rot + KI_ROT * rotvec(self.rot @ np.asarray(meas[1], float).T),
                                    -ROT_MAX, ROT_MAX)
            self.max_corr_rot = max(self.max_corr_rot, float(np.abs(self.corr_rot).max()))
        q, pe, _ = ik.solve(np.asarray(pos, float) + self.corr, ik.matrix_to_quat_xyzw(rot_exp(self.corr_rot) @ rot),
                            self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), np.asarray(rot, float)
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.last_mm = pe * 1000
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None, servo=None, square=False):
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s), servo, square)
            yield

    def to_joints(self, q_target, duration: float, grip: float | None = None):
        """Move in joint space to ``q_target`` (no IK)."""
        from yam_arena import ik

        q0, g0 = self.q.copy(), self.grip
        g1 = g0 if grip is None else grip
        self.corr, self.corr_rot = np.zeros(3), np.zeros(3)
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.q = q0 + s * (np.asarray(q_target, float) - q0)
            self.grip = g0 + s * (g1 - g0)
            pos, rot = ik.tcp_pose(self.q, self.base)
            self.pos, self.rot = np.asarray(pos, float), np.asarray(rot, float)
            self.cmd = np.r_[self.q, self.grip]
            yield

    def relax(self, duration: float = 0.5):
        """Let the trims go."""
        c0, r0 = self.corr.copy(), self.corr_rot.copy()
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self.corr, self.corr_rot = c0 * (1 - k / n), r0 * (1 - k / n)
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


def shift(arm, d, duration: float, grip: float | None = None, servo=None, square=False):
    """Generator: move by ``d`` from wherever the arm's command is when this step starts (not when it was built)."""
    yield from arm.move(arm.pos + np.asarray(d, float), arm.rot, duration, grip, servo, square)


def stay(arm, duration: float, grip: float | None = None, servo=None, square=False):
    yield from arm.move(arm.pos, arm.rot, duration, grip, servo, square)


def ease_off(arm, outward: float, press: float, duration: float, servo=None, square=False):
    """Generator: back the hand off along y by its own measured give (commanded minus measured tool point, from its joint
    readings), less EASE, so the press goes to almost nothing without the pinched end tile being pulled off the row.
    The trims that held the hand upright and on its line against the press are let go in step with it (they were
    earned by the press, so they go as it goes); the joint readings keep correcting what is left."""
    from yam_arena import ik

    meas = arm.read() if arm.read is not None else None
    give = press if meas is None else min(press, abs(float(arm.pos[1] - meas[0][1])))
    p0, c0, r0 = arm.pos.copy(), arm.corr.copy(), arm.corr_rot.copy()
    back = np.array([0.0, outward * max(0.0, give - EASE), 0.0])
    n, last = max(1, round(duration * CONTROL_HZ)), 0.0
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        arm.corr, arm.corr_rot = arm.corr - (s - last) * c0, arm.corr_rot - (s - last) * r0
        last = s
        arm._solve(p0 + s * back, arm.rot, servo, square)
        yield


# -- the plan ------------------------------------------------------------------------------------------------------------
PAD_TIP = 0.013          # the fingertips reach 13 mm past the tool point
TIP_UP = 0.0015          # the fingertips stop this far above the mat
OUT = 0.003              # each pinch is centred this far outward of its end tile's centre
OPEN = 0.45              # grip command for the approach: about 42 mm between the pads, clear of the far row
PINCH = 0.15             # grip command on the 26 mm tile: each finger is asked 6 mm into it, which its 5,000 N/m drive
                         # turns into about 30 N. Asked to close fully, both drives sit on their 50 N limit and the tile
                         # is free to drift between them (batch a4ee342a: an end tile slid 3 mm as the press was eased off)
SQUEEZE = float(os.environ.get("DEMO_SQUEEZE_MM", "60")) / 1000   # each hand is commanded this far past the row's end
LIGHT = 0.006            # the "light" variant's press
SQUEEZE_S = 2.5
LIFT = 0.035
EASE = 0.0005            # the press is eased off to this much give (about 0.1 N), not to zero
EASE_S = 3.5             # slowly: as the press goes, so does its sideways pull on the hands, which the trim must follow
CARRY_TRIM = 0.0         # no offset on the carry: the pinch centres each end tile on its tool point
SETTLE = -0.0015         # the row is lowered to 1.5 mm past touching the far row, so it bears on it lightly (about 1 N)
                         # and stays down while the press is eased off (batch a3558eec: lowered to 1 mm above, the
                         # hands rose 2 mm as the press went and the row dropped the last 3 mm crooked)
TOUCH = 0.001            # the height trim is on down to 1 mm above touching
X_ONLY, XZ = (1.0, 0.0, 0.0), (1.0, 0.0, 1.0)


def plan(T) -> dict:
    p = {}
    z = T.FLOOR + PAD_TIP + TIP_UP
    for side, s in (("l", 1.0), ("r", -1.0)):
        y = s * (T.ROW_Y[-1] + OUT)
        p[f"{side}_pre"] = np.array([T.NEAR_X, y, z + 0.06])
        p[f"{side}_grip"] = np.array([T.NEAR_X, y, z])
        p[f"{side}_sq"] = np.array([T.NEAR_X, y - s * SQUEEZE, z])
        p[f"{side}_up"] = np.array([T.NEAR_X, y - s * SQUEEZE, z + LIFT])
        p[f"{side}_over"] = np.array([T.FAR_X + CARRY_TRIM, y - s * SQUEEZE, z + LIFT])
        p[f"{side}_down"] = np.array([T.FAR_X + CARRY_TRIM, y - s * SQUEEZE, z + T.TILE_T + SETTLE])
        p[f"{side}_off"] = np.array([T.FAR_X + CARRY_TRIM, y, z + T.TILE_T + SETTLE])
        p[f"{side}_away"] = np.array([T.FAR_X + CARRY_TRIM, y, z + T.TILE_T + 0.08])
    return p


def script(task, ctx):
    _rotations()
    T, P = task, plan(task)
    variant = os.environ.get("DEMO_VARIANT", "")
    L, R = ctx.left, ctx.right
    from yam_arena.constants import READY as READY_Q

    rot = {"l": best_grip(L, [P["l_pre"], P["l_grip"], P["l_over"]], DOWN_X),
           "r": best_grip(R, [P["r_pre"], P["r_grip"], P["r_over"]], DOWN_X)}
    arms = {"l": L, "r": R}
    sides = ("l",) if variant == "one_arm" else ("l", "r")
    square = variant != "no_trim"
    press = LIGHT if variant == "light" else SQUEEZE
    sgn = {"l": 1.0, "r": -1.0}

    def each(fn):
        return both(*[fn(s, arms[s]) for s in sides])

    def go(key, duration, grip=None, servo=None, sq=False):
        return each(lambda s, a: a.move(P[f"{s}_{key}"] + (np.array([0.0, sgn[s] * (SQUEEZE - press), 0.0])
                                                            if key in ("sq", "up", "over", "down") else 0.0),
                                        rot[s], duration, grip, servo, sq))

    # 1. over the end tiles, down, pinch
    yield from go("pre", 2.5, OPEN)
    yield from go("grip", 1.2)
    yield from each(lambda s, a: stay(a, 0.6, PINCH))
    if variant == "one_arm":
        yield from chain(L.move(P["l_grip"] + [0, 0, LIFT], rot["l"], 1.5), L.move(P["l_over"] + [0, SQUEEZE, 0], rot["l"], 2.0),
                         L.move(P["l_off"], rot["l"], 1.5), stay(L, 0.6, OPEN), L.move(P["l_away"], rot["l"], 1.0),
                         L.to_joints(READY_Q, 2.0), wait(2.0))
        return
    # 2. press the row together end to end
    yield from go("sq", SQUEEZE_S, None, XZ, square)
    yield from each(lambda s, a: stay(a, 0.5, None, XZ, square))
    # 3. lift, carry, lower
    yield from go("up", 1.5, None, XZ, square)
    if variant == "early_release":
        yield from each(lambda s, a: shift(a, [(T.FAR_X - T.NEAR_X) / 2, 0, 0], 1.0, None, XZ, square))
        yield from each(lambda s, a: shift(a, [0, sgn[s] * press, 0], 0.6))
        yield from wait(3.0)
        return
    yield from go("over", 2.5, None, XZ, square)
    if variant == "crooked":
        yield from each(lambda s, a: shift(a, [0.008, 0, 0], 0.6, None, XZ, square))
    yield from each(lambda s, a: stay(a, 0.8, None, XZ, square))
    yield from each(lambda s, a: shift(a, [0, 0, -(LIFT - T.TILE_T - TOUCH)], 1.5, None, XZ, square))
    yield from each(lambda s, a: stay(a, 0.5, None, XZ, square))
    # the last step down bears on the far row, so the height is no longer trimmed (the trim would wind up against it)
    yield from each(lambda s, a: shift(a, [0, 0, SETTLE - TOUCH], 0.5, None, X_ONLY, square))
    # 4. ease the press off (each hand backs off by how far its own joints say the press has bent it, so it neither
    #    keeps pushing nor drags its end tile away), open, leave
    yield from each(lambda s, a: ease_off(a, sgn[s], press, EASE_S, X_ONLY, square))
    yield from each(lambda s, a: stay(a, 0.5, None, X_ONLY, square))
    yield from each(lambda s, a: stay(a, 0.6, OPEN))
    yield from each(lambda s, a: shift(a, [0, 0, 0.08], 1.0))
    yield from each(lambda s, a: a.to_joints(READY_Q, 2.0))
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


def summary(task, st) -> dict:
    """Logging only: where the tiles are (reads simulator state for the trace and result.json, never for steering)."""
    near = [st[s.name] for s in task.NEAR]
    far = [st[s.name] for s in task.FAR]
    p = np.array([np.asarray(s.pos, float) for s in near])
    f = np.array([np.asarray(s.pos, float) for s in far])
    f0 = np.array([[task.FAR_X, y] for y in task.ROW_Y])
    return {"near_under_mm": [round(min(task.under(s) for s in near), 1), round(max(task.under(s) for s in near), 1)],
            "near_x_mm": [round(1000 * float(p[:, 0].min()), 1), round(1000 * float(p[:, 0].max()), 1)],
            "near_len_mm": round(1000 * float(np.linalg.norm(p[-1] - p[0])), 1),
            "near_face_down": sum(task.face_down(s) for s in near),
            "align_mm": round(1000 * float(np.abs(p[:, :2] - f[:, :2]).max()), 1),
            "align_x_mm": [round(1000 * float((p[:, 0] - f[:, 0]).min()), 1), round(1000 * float((p[:, 0] - f[:, 0]).max()), 1)],
            "align_y_mm": [round(1000 * float((p[:, 1] - f[:, 1]).min()), 1), round(1000 * float((p[:, 1] - f[:, 1]).max()), 1)],
            "far_moved_mm": round(1000 * float(np.linalg.norm(f[:, :2] - f0, axis=1).max()), 1),
            "far_under_mm": round(max(abs(task.under(s)) for s in far), 1),
            "intact": bool(task.row_intact(near))}


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
    debug = os.environ.get("DEMO_DEBUG", "1") != "0"
    min_gap = float("inf")
    import clearance

    print("PLAN " + json.dumps({k: np.round(v, 4).tolist() for k, v in plan(task).items()}), flush=True)

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    last_gap = {"mm": None}

    def row(steps, info):
        """Logging only: reads simulator state for the trace."""
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)
        S = task._STATE
        hist = S.get("hist") or []
        moved = (np.max([np.linalg.norm(h - hist[-1], axis=1) for h in hist], axis=0) if hist else np.zeros(1))
        out_row = {"step": steps, "score": info.get("score"), "levels": [bool(v) for v in task.levels(st)],
                   "as_one": S["as_one"], "released": S["released"], "still_mm": round(1000 * float(moved.max()), 2),
                   "window": len(hist), "violation": S["violation"], "gap_mm": last_gap["mm"]}
        out_row.update(summary(task, st))
        for nm, a in (("L", ctx.left), ("R", ctx.right)):
            mp, mr = ctx.joint_tcp(a.name)
            out_row[nm] = {"cmd": np.round(a.pos - [0, 0, task.FLOOR], 4).tolist(),
                           "meas": np.round(mp - [0, 0, task.FLOOR], 4).tolist(),
                           "tilt_deg": np.round(np.degrees(rotvec(a.rot @ mr.T)), 1).tolist(),
                           "trim_mm": np.round(1000 * a.corr, 1).tolist(),
                           "trim_deg": np.round(np.degrees(a.corr_rot), 1).tolist(),
                           "grip": round(float(a.grip), 2), "ik_mm": round(a.last_mm, 1)}
        return out_row

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
    result = {"task_score": score, "success": success, "steps": steps, "variant": os.environ.get("DEMO_VARIANT", ""),
              "squeeze_mm": round(1000 * SQUEEZE, 1),
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "max_trim_mm": {"left": round(1000 * ctx.left.max_corr, 1), "right": round(1000 * ctx.right.max_corr, 1)},
              "max_trim_deg": {"left": round(math.degrees(ctx.left.max_corr_rot), 1),
                               "right": round(math.degrees(ctx.right.max_corr_rot), 1)},
              "min_arm_link_gap_mm": round(1000 * min_gap, 1) if min_gap < float("inf") else None,
              "as_one": S["as_one"], "violation": S["violation"]}
    try:
        from yam_arena.scene import object_states

        result["end"] = summary(task, object_states(emb._env, emb.task.names))
    except Exception as exc:  # noqa: BLE001 - logging only
        result["end"] = repr(exc)
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
    """No simulator: yam_arena IK error of every target."""
    import importlib

    from yam_arena import ik, station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)
    _rotations()
    P = plan(task)
    left, right = Arm("left", station.BASES["left"]), Arm("right", station.BASES["right"])
    rot = {"l": best_grip(left, [P["l_pre"], P["l_grip"], P["l_over"]], DOWN_X),
           "r": best_grip(right, [P["r_pre"], P["r_grip"], P["r_over"]], DOWN_X)}
    for name, p in P.items():
        arm = left if name.startswith("l_") else right
        q, pe, re = ik.solve(np.asarray(p, float), ik.matrix_to_quat_xyzw(rot[name[0]]), arm.q, base_pos=arm.base, iters=300)
        print(f"IK {name:8s} {arm.name:5s} {np.round(p - [0, 0, task.FLOOR], 4).tolist()} {1000 * pe:.1f}mm/{re:.3f} "
              f"roll {math.degrees(float(q[5])):.0f} deg", flush=True)


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
