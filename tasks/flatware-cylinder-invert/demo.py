"""Scripted full-score demo for flatware-cylinder-invert, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]
    python demo.py task.py --ik          (no simulator: IK of every target and of the turn, step by step)

Open loop: every target comes from the task file's fixed layout and the clock; the only feedback is each arm's own
joint readings. No object pose steers it.

1. Each hand points at the stack from its side and pinches the two collars together across the seam, fingers closing
   up and down (the right hand's grip turned half a turn so both wrists can roll the same way).
2. Both lift the pair 75 mm, turn it over (each wrist rolling 180 degrees about the line between the hands), and set it
   down: the spoons slide through the seam into the clean cylinder, handles up.
3. Both open and back off, then squeeze the old cylinder's body between their closed fingers, lift it clear of the
   spoons' handles, carry it aside and set it down.

Negative variants (DEMO_VARIANT=...):
  one_arm       only the left hand pinches the seam and lifts: the loose pair parts and spoons spill (0 for good)
  no_pinch      the hands hold the old cylinder's body only (not the seam) and turn: the new one falls off (0 for good)
  early_lift    the old cylinder is lifted off while the pair is still up, before it is set down: spoons spill (0)
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
VARIANTS = ("", "one_arm", "no_pinch", "early_lift")
KI, CORR_MAX = 0.2, 0.03        # joint-feedback trim: gain per step and limit (m)


# -- grip rotations: columns are the TCP's closing axis, its y axis and its approach axis (yam_arena convention) -----------
DOWN_Y = DOWN_X = None


def _rotations():
    """yam_arena's straight-down grip (fingers closing along y) and the same turned a quarter turn (closing along x)."""
    global DOWN_Y, DOWN_X
    if DOWN_Y is None:
        from yam_arena.demo import grip_rotation

        DOWN_Y, DOWN_X = grip_rotation(0.0), grip_rotation(math.pi / 2)


FLIP = np.diag([-1.0, -1.0, 1.0])       # the same parallel grip turned half a turn about its approach axis
# pointing +x at the door, fingers closing up-down (columns: closing axis, its y, approach)
PUSH_FWD = np.column_stack([np.array([0.0, 0.0, 1.0]), np.cross([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])])


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
        self.last_mm = 0.0
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
        self.last_mm = pe * 1000
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

    def to_joints(self, q_target, duration: float, grip: float | None = None):
        """Move in joint space to ``q_target`` (no IK): a clean way back to a known configuration after a pose the IK
        can only reach from one side (the push pointing at the door)."""
        from yam_arena import ik

        q0, g0 = self.q.copy(), self.grip
        g1 = g0 if grip is None else grip
        self.corr = np.zeros(3)
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.q = q0 + s * (np.asarray(q_target, float) - q0)
            self.grip = g0 + s * (g1 - g0)
            pos, rot = ik.tcp_pose(self.q, self.base)
            self.pos, self.rot = np.asarray(pos, float), np.asarray(rot, float)
            self.cmd = np.r_[self.q, self.grip]
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
PAD_TIP = 0.013          # the fingertips reach 13 mm past the tool point
COLLAR_BITE = 0.0075     # the fingertips reach 7.5 mm in over the collars (they stand 8.5 mm proud of the body)
LIFT = 0.075             # the seam is lifted to 218 mm before the turn (the stack's corner sweeps 154 mm)
TURN_S = 6.0             # the turn over, slow so the spoons slide rather than fly
SQUEEZE = 0.010          # the closed fingertips commanded 10 mm into the old cylinder's body from each side (batch 80d4d6ab: the pair ends the turn 20 mm off the layout in x, where 4 mm barely touched)
SQUEEZE_UP = 0.100       # where they press: 100 mm above the seam, above the upside-down cylinder's centre of mass, so
                         # it hangs from the two fingertips (at 70 mm it swung over as it was carried: batch a9783e20)
SQ_DX = -0.020           # and 20 mm short of the layout in x: the pair ends the turn there (batches 95a8b3c6, 80d4d6ab,
                         # 21f9d8eb: new cylinder at x 0.335, old at 0.330), and an off-centre squeeze pushes it away


def cols(closing, approach):
    x, z = np.asarray(closing, float), np.asarray(approach, float)
    return np.column_stack([x, np.cross(z, x), z])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


L_ROT = cols((0, 0, 1), (0, -1, 0))                 # left: pointing -y, fingers closing up-down
R_ROT = cols((0, 0, 1), (0, 1, 0)) @ np.diag([-1.0, -1.0, 1.0])   # right: pointing +y, turned half a turn
L_SQ = cols((1, 0, 0), (0, -1, 0))                  # squeezing the body: fingers closed, pointing at it
R_SQ = cols((1, 0, 0), (0, 1, 0)) @ np.diag([-1.0, -1.0, 1.0])
# the same squeeze turned half a turn, for after the turn over: each wrist is then at -+90 degrees of roll, from where
# L_SQ / R_SQ would need more than its 120 degree limit (batch 95a8b3c6: the fingers stopped 10 mm short of the body)
L_SQ2, R_SQ2 = L_SQ @ FLIP, R_SQ @ FLIP


def plan(T) -> dict:
    p = {}
    x, y, top = T.STACK_X, T.STACK_Y, T.TOP
    seam = top + T.SEAM_Z
    reach = T.COLLAR_R + PAD_TIP - COLLAR_BITE        # tool point this far out from the axis
    for side, s in (("l", 1.0), ("r", -1.0)):
        p[f"{side}_pre"] = np.array([x, y + s * (reach + 0.05), seam])
        p[f"{side}_seam"] = np.array([x, y + s * reach, seam])
        p[f"{side}_up"] = np.array([x, y + s * reach, seam + LIFT])
        p[f"{side}_back"] = np.array([x, y + s * (reach + 0.05), seam])
        sq = T.CYL_R + PAD_TIP - SQUEEZE
        p[f"{side}_sq_pre"] = np.array([x + SQ_DX, y + s * (sq + 0.04), seam + SQUEEZE_UP])
        p[f"{side}_sq"] = np.array([x + SQ_DX, y + s * sq, seam + SQUEEZE_UP])
        p[f"{side}_sq_up"] = np.array([x + SQ_DX, y + s * sq, seam + SQUEEZE_UP + 0.06])
        p[f"{side}_sq_aside"] = np.array([x + 0.15, y + s * sq, seam + SQUEEZE_UP + 0.06])
        p[f"{side}_sq_off"] = np.array([x + 0.15, y + s * (sq + 0.05), top + SQUEEZE_UP + 0.002])
    # set down upside down, the old cylinder's collar is on the table: the squeeze point stays 70 mm above its collar
    for side, s in (("l", 1.0), ("r", -1.0)):
        p[f"{side}_sq_down"] = np.array([x + 0.15, y + s * (T.CYL_R + PAD_TIP - SQUEEZE), top + SQUEEZE_UP + 0.002])
    return p


def turn(arm, rot0, centre_pos, a0, a1, duration):
    """Generator: roll the hand about the world y axis through its own tool point from a0 to a1 (its position fixed)."""
    from yam_arena import ik

    n = max(1, round(duration * CONTROL_HZ))
    for k in range(1, n + 1):
        a = a0 + ik.smoothstep(k / n) * (a1 - a0)
        arm._solve(centre_pos, ry(a) @ rot0, None)
        yield


def script(task, ctx):
    _rotations()
    T, P = task, plan(task)
    variant = os.environ.get("DEMO_VARIANT", "")
    L, R = ctx.left, ctx.right
    half = -math.pi            # each wrist rolls from +-90 to -+90 degrees (the other way runs into its 120 degree limit)
    if variant == "no_pinch":                 # hold only the old (lower) cylinder's body and turn
        seam = T.TOP + T.SEAM_Z
        lo = {s: P[f"{s}_seam"] - np.array([0.0, 0.0, 0.07]) for s in ("l", "r")}
        lo = {s: lo[s] + np.array([0.0, (1 if s == "l" else -1) * (T.CYL_R + PAD_TIP - SQUEEZE - (T.COLLAR_R + PAD_TIP - COLLAR_BITE)), 0.0])
              for s in ("l", "r")}
        yield from both(chain(L.move(lo["l"] + np.array([0, 0.04, 0]), L_SQ, 2.5, 0.0), L.move(lo["l"], L_SQ, 1.0)),
                        chain(R.move(lo["r"] - np.array([0, 0.04, 0]), R_SQ, 2.5, 0.0), R.move(lo["r"], R_SQ, 1.0)))
        yield from both(shift(L, np.array([0, 0, LIFT]), L_SQ, 1.5), shift(R, np.array([0, 0, LIFT]), R_SQ, 1.5))
        yield from both(turn(L, L_SQ, L.pos.copy(), 0.0, half, 4.0), turn(R, R_SQ, R.pos.copy(), 0.0, half, 4.0))
        yield from wait(3.0)
        del seam
        return
    if os.environ.get("FLATWARE_LAYOUT") == "pour":
        yield from pour(T, R)
        return
    # 1. pinch the two collars at the seam
    yield from both(chain(L.move(P["l_pre"], L_ROT, 2.5, 1.0), L.move(P["l_seam"], L_ROT, 1.2), stay(L, L_ROT, 0.6, 0.0)),
                    chain(R.move(P["r_pre"], R_ROT, 2.5, 1.0), R.move(P["r_seam"], R_ROT, 1.2), stay(R, R_ROT, 0.6, 0.0))
                    if variant != "one_arm" else wait(0.1))
    if variant == "one_arm":
        yield from chain(L.move(P["l_up"], L_ROT, 1.5), turn(L, L_ROT, P["l_up"], 0.0, half, TURN_S), wait(3.0))
        return
    # 2. lift, turn over, set down
    yield from both(L.move(P["l_up"], L_ROT, 1.5), R.move(P["r_up"], R_ROT, 1.5))
    yield from both(turn(L, L_ROT, P["l_up"], 0.0, half, TURN_S), turn(R, R_ROT, P["r_up"], 0.0, half, TURN_S))
    l_rot, r_rot = ry(half) @ L_ROT, ry(half) @ R_ROT
    if variant == "early_lift":            # let go with one hand and pull the old one away while the pair is up
        yield from chain(stay(R, r_rot, 0.6, 1.0), shift(R, np.array([0, -0.05, 0]), r_rot, 0.8),
                         shift(L, np.array([0, 0, 0.06]), l_rot, 1.0), wait(3.0))
        return
    yield from both(L.move(P["l_seam"], l_rot, 1.5), R.move(P["r_seam"], r_rot, 1.5))
    # 3. let go, back off, squeeze the old cylinder's body, lift it clear and set it aside
    yield from both(chain(stay(L, l_rot, 0.6, 1.0), L.move(P["l_back"], l_rot, 0.8)),
                    chain(stay(R, r_rot, 0.6, 1.0), R.move(P["r_back"], r_rot, 0.8)))
    yield from both(chain(L.move(P["l_sq_pre"], L_SQ2, 2.0, 0.0), L.move(P["l_sq"], L_SQ2, 1.0)),
                    chain(R.move(P["r_sq_pre"], R_SQ2, 2.0, 0.0), R.move(P["r_sq"], R_SQ2, 1.0)))
    yield from both(chain(L.move(P["l_sq_up"], L_SQ2, 1.2), L.move(P["l_sq_aside"], L_SQ2, 2.5), L.move(P["l_sq_down"], L_SQ2, 1.5)),
                    chain(R.move(P["r_sq_up"], R_SQ2, 1.2), R.move(P["r_sq_aside"], R_SQ2, 2.5), R.move(P["r_sq_down"], R_SQ2, 1.5)))
    from yam_arena.constants import READY as READY_Q
    yield from both(chain(L.move(P["l_sq_off"], L_SQ2, 0.8), L.to_joints(READY_Q, 2.0)),
                    chain(R.move(P["r_sq_off"], R_SQ2, 0.8), R.to_joints(READY_Q, 2.0)))
    yield from wait(1.5)


def pour(T, R):
    """Rehearsal of the one-arm route (FLATWARE_LAYOUT=pour: the empty cylinder already upright beside the full one).
    The right hand pinches the full cylinder's collar on its near side, lifts it, and rolls it half a turn about the line
    through the pinch while carrying it toward the empty one: at a quarter turn its mouth faces the empty cylinder's
    mouth from just above and the spoons slide out bowl-first; it ends upside down over the empty one. Then it sets the old
    cylinder down where it came from and leaves. (Batch e11adfd9: the one-sided collar pinch let the full cylinder, 0.63 kg
    hanging 56 mm off the pinch, fall on its side as it was lifted, before any pouring.)"""
    from yam_arena import ik
    from yam_arena.constants import READY as READY_Q

    xo, xn = T.OLD_POS[0], T.NEW_POS[0]
    lip = T.TOP + T.CYL_H - T.COLLAR_T / 2
    reach = T.COLLAR_R + PAD_TIP - COLLAR_BITE
    y = T.OLD_POS[1] - reach
    yield from chain(R.move(np.array([xo, y - 0.05, lip]), R_ROT, 2.5, 1.0), R.move(np.array([xo, y, lip]), R_ROT, 1.2),
                     stay(R, R_ROT, 0.8, 0.0))
    up = np.array([xo, y, lip + 0.08])
    yield from R.move(up, R_ROT, 1.5)
    mid = np.array([xn + 0.03, y, T.TOP + 0.215])       # quarter turn: mouth 30 mm short of the empty one's axis, body clear
    end = np.array([xn, y, T.TOP + T.CYL_H + 0.025])    # upside down, mouth 20 mm over the empty one's rim
    n = round(TURN_S * CONTROL_HZ)
    for k in range(1, n + 1):
        u = ik.smoothstep(k / n)
        p = up + (mid - up) * min(1.0, 2 * u) if u <= 0.5 else mid + (end - mid) * (2 * u - 1)
        R._solve(p, ry(-math.pi * u) @ R_ROT, None)
        yield
    r_down = ry(-math.pi) @ R_ROT
    yield from wait(1.5)
    yield from chain(R.move(end + np.array([0, 0, 0.17]), r_down, 1.5), R.move(np.array([xo + 0.02, y, lip + 0.17]), r_down, 1.5),
                     R.move(np.array([xo + 0.02, y, T.TOP + T.COLLAR_T / 2 + 0.002]), r_down, 1.5),
                     stay(R, r_down, 0.6, 1.0), shift(R, np.array([0, -0.05, 0]), r_down, 0.8), R.to_joints(READY_Q, 2.0))
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


def run(task_path: str, out_dir: str, max_steps: int = 1800, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 10.0) -> dict:   # 10 s after the script for the spoons to settle (batch 5f873c8f: one still sliding into its lean 5 s after the set-down)
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
        new, old = st["new_cylinder"], st["old_cylinder"]
        spoons = [st[s.name] for s in task.SPOONS]
        hist = S.get("hist") or []
        moved = (np.max([np.linalg.norm(h - hist[-1], axis=1) for h in hist], axis=0) if hist else np.zeros(1))
        names = ["new", "old"] + [s.name for s in task.SPOONS]
        return {"step": steps, "score": info.get("score"), "levels": list(task.levels(st)), "released": S["released"],
                "still_mm": round(1000 * float(moved.max()), 2), "moving": names[int(moved.argmax())] if hist else "",
                "window": len(hist),
                "violation": S["violation"], "gap_mm": last_gap["mm"],
                "new_axis_z": round(float(task._axis(new)[2]), 3), "old_axis_z": round(float(task._axis(old)[2]), 3),
                "new": np.round(np.asarray(new.pos) - [0, 0, task.TOP], 3).tolist(),
                "old": np.round(np.asarray(old.pos) - [0, 0, task.TOP], 3).tolist(),
                "spoons_in_new": sum(task._inside(s, new) for s in spoons),
                "spoons_in_old": sum(task._inside(s, old) for s in spoons),
                "spoons_up": sum(task.spoon_up(s) for s in spoons),
                "tcp": {"L_cmd": np.round(ctx.left.pos - [0, 0, task.TOP], 3).tolist(),
                        "L_meas": np.round(ctx.joint_tcp("left")[0] - [0, 0, task.TOP], 3).tolist(),
                        "R_cmd": np.round(ctx.right.pos - [0, 0, task.TOP], 3).tolist(),
                        "R_meas": np.round(ctx.joint_tcp("right")[0] - [0, 0, task.TOP], 3).tolist(),
                        "grip": [round(float(ctx.left.grip), 2), round(float(ctx.right.grip), 2)]},
                "ik_mm": [round(ctx.left.last_mm, 1), round(ctx.right.last_mm, 1)],
                "roll_deg": [round(math.degrees(float(ctx.left.q[5])), 0), round(math.degrees(float(ctx.right.q[5])), 0)]}

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
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "min_arm_link_gap_mm": round(1000 * min_gap, 1) if min_gap < float("inf") else None,
              "violation": S["violation"]}
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
    """No simulator: yam_arena IK error of every target, and of the two-arm turn step by step."""
    import importlib

    from yam_arena import ik, station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)
    _rotations()
    P = plan(task)
    left, right = Arm("left", station.BASES["left"]), Arm("right", station.BASES["right"])
    for name, p in P.items():
        arm = left if name.startswith("l_") else right
        R = (L_SQ if "sq" in name else L_ROT) if arm is left else (R_SQ if "sq" in name else R_ROT)
        _, pe, re = ik.solve(np.asarray(p, float), ik.matrix_to_quat_xyzw(R), arm.q, base_pos=arm.base, iters=300)
        print(f"IK {name:12s} {arm.name:5s} {np.round(p - [0, 0, task.TOP], 3).tolist()} {1000 * pe:.1f}mm/{re:.3f}", flush=True)
    for arm, rot0, centre in ((left, L_ROT, P["l_up"]), (right, R_ROT, P["r_up"])):
        q, worst, q6 = arm.q.copy(), 0.0, []
        q, _, _ = ik.solve(centre, ik.matrix_to_quat_xyzw(rot0), q, base_pos=arm.base, iters=300)
        for a in np.linspace(0.0, -math.pi, 37):
            q, pe, _ = ik.solve(centre, ik.matrix_to_quat_xyzw(ry(a) @ rot0), q, base_pos=arm.base, iters=60)
            worst = max(worst, pe)
            q6.append(float(q[5]))
        print(f"SEQ {arm.name} turn: worst {1000 * worst:.1f} mm, wrist roll {math.degrees(min(q6)):.0f}..{math.degrees(max(q6)):.0f} deg",
              flush=True)


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
