"""Scripted full-score demo for softserve-door-assemble, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1800] [--no-record]

Open loop: every target comes from the task file's fixed layout and the clock. The only feedback is each arm's own
joint readings (a trim that keeps the measured tool point on the plan). No object pose or task state steers it.

1. The left hand presses down on the door's face (it would slide under the valve's drag). The right hand pushes the
   draw valve's head home with closed fingers, 2 mm past the seat, against the O-rings' drag.
2. The left hand lifts the handle off the door's face, turns it 1 degree back about the pin axis (it hangs tilted in
   the grip), carries it over its place and lowers it, the knuckle between the lugs and the fork over the valve's bar,
   and holds the lever 6.8 mm above fitted.
3. The right hand picks the pin by its bent end, presses it lightly on the knuckle while the left hand sweeps the
   lever 1 mm either way so the holes line up, then pushes it +y through lug, knuckle and lug.
4. The right hand lets go and leaves, then the left; the handle settles raised on its pin.

Negative variants (DEMO_VARIANT=...):
  pin_no_handle  the handle is left on the door's face; the pin goes through the empty lugs (0.25)
  reversed       the valve starts turned half a turn (slot away from the lugs); the rest as usual (0)
  no_pin         the handle is placed and let go without the pin (0.5 at most)
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
VARIANTS = ("", "pin_no_handle", "reversed", "no_pin")
KI, CORR_MAX = 0.2, 0.03        # joint-feedback trim: gain per step and limit (m)
PUSH_CORR_MAX = 0.045           # trim limit while a hand pushes against the O-ring drag (80 mm shoved the door: run 16)
HOLD_CORR_MAX = 0.006           # trim limit while the left hand holds the handle down on its stop


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
PUSH_PAST = 0.002        # the closed fingers push the head 2 mm past the seat (the ledge stops the valve)
PAD_TIP = 0.013          # the fingertips reach 13 mm past the tool point along the approach
PUSH_UP = 0.0            # the push on the valve's axis
LEVER_GRIP = 0.120       # the left hand holds the lever this far from the pin axis (well clear of the pin hand)
LEVER_TCP_UP = 0.004
HANDLE_Y_TRIM = -0.0035  # the carried handle arrives 3.5 mm to the left of plan (measured in a sim run, repeatable)
HANDLE_TILT = math.radians(float(os.environ.get("DEMO_HANDLE_TILT", "-1")))   # the hand turned about the pin axis:
#                          the handle hangs about 4 degrees tilted in the grip and turns further as its tines meet the bar
#                          (Isaac runs 16-18); with the plain 6.5 mm notch the finger-grip bench fits best untilted
HANDLE_X_TRIM = float(os.environ.get("DEMO_HANDLE_X_TRIM", "-0.0014"))   # and 1 mm along the valve axis (runs 14-17)
PRESS = float(os.environ.get("DEMO_PRESS", "-0.0055"))   # the lever held this far below fitted (negative: above);
#                          with the 6.5 mm notch a 2 mm press pushed the bar and valve down in their play: knuckle 3.4 mm low (run 21)
SWEEP = (-0.0058, -0.0078)  # the lever's small sweep while the pin goes in (below fitted, negative: above), run 32
PIN_GRIP = (-0.015, -0.034)   # the right hand takes the pin's bent end: (x, y) from the pin's middle, world
PIN_TCP_UP = 0.0113        # the held pin rides 1.3 mm low (run 32): aimed that much higher
PIN_END_Y = -0.014       # the pin is pushed until its middle is 14 mm short of the pivot (tip 4.5 mm past the far lug)
BRACE = (0.335, 0.072)   # where the left hand presses on the door's face: its far end, 11 mm clear of the handle
#                          lying there and well away from the right hand's push
BRACE_DOWN = 0.010       # pressed 10 mm below the face (about 10 N through the arm's position control)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def plan(T) -> dict:
    """Tool-point targets (world, m) from the task file's fixed layout."""
    p = {}
    ax = T.M @ np.array([0.0, 0.0, 1.0])            # the bore axis toward the door's top (world -x)
    door = np.array(T.DOOR_ORIGIN_W)
    head = door + ax * (T.VALVE_TOP / 1000)                           # the head's end face when seated
    lift = np.array([0.0, 0.0, PUSH_UP])
    # the push from the front: a hand pointing at the door (+x) on the valve's axis, fingers closed up-down, so its
    # fingers pass under the lugs; its fingertips (13 mm past the tool point) push the head's end face
    tip = PAD_TIP
    p["push_start"] = door + ax * (T.VALVE_START_OUT + T.VALVE_TOP / 1000) + ax * (tip + 0.010) + lift
    p["push_home"] = head + ax * (tip - PUSH_PAST) + lift
    p["push_short"] = head + ax * (tip + 0.010) + lift
    h0 = np.array(T.HANDLE_START)
    lever = LEVER_GRIP * ax + np.array([0.0, 0.0, LEVER_TCP_UP])
    p["handle_pick"] = h0 + lever
    p["handle_lift"] = p["handle_pick"] + np.array([0.0, 0.0, 0.06])
    tilted = _ry(HANDLE_TILT) @ lever                 # turned about the pin axis, so the knuckle stays put
    p["handle_over"] = T.PIVOT_W + tilted + np.array([HANDLE_X_TRIM, HANDLE_Y_TRIM, 0.075])   # the hanging fork 18 mm above the lugs
    p["handle_fit"] = T.PIVOT_W + tilted + np.array([HANDLE_X_TRIM, HANDLE_Y_TRIM, -PRESS])
    pin0 = np.array(T.PIN_START)
    pg = np.array([PIN_GRIP[0], PIN_GRIP[1], PIN_TCP_UP])
    p["pin_pick"] = pin0 + pg
    p["pin_pre"] = T.PIVOT_W + np.array([0.0, -0.060, 0.0]) + pg
    p["pin_home"] = T.PIVOT_W + np.array([0.0, PIN_END_Y, 0.0]) + pg
    # the pin's tip at the knuckle's face (12 mm from the pivot) and 3 mm past it: a light press while the hole lines up
    p["pin_touch"] = T.PIVOT_W + np.array([0.0, -(T.KNUCKLE_W / 2 + T.PIN_LEN / 2 - 3.0) / 1000, 0.0]) + pg
    p["brace"] = np.array([BRACE[0] + (T.XD - 0.24), BRACE[1], T.FACE_Z + 0.004])   # BRACE is for the door at x 0.24
    p["park_right"] = np.array([0.20, -0.22, T.TOP + 0.16])
    return p


def script(task, ctx):
    _rotations()
    T, P = task, plan(task)
    variant = os.environ.get("DEMO_VARIANT", "")
    L, R = ctx.left, ctx.right
    up = np.array([0.0, 0.0, 1.0])
    ax = T.M @ np.array([0.0, 0.0, 1.0])
    from yam_arena.constants import READY as READY_Q
    brace_rot = best_grip(L, [P["brace"]], DOWN_Y)
    push_rot = best_grip(R, [P["push_start"], P["push_home"]], PUSH_FWD)
    handle_rot = best_grip(L, [P[n] for n in ("handle_pick", "handle_lift", "handle_over", "handle_fit")], DOWN_Y)
    fit_rot = _ry(HANDLE_TILT) @ handle_rot
    pin_rot = best_grip(R, [P[n] for n in ("pin_pick", "pin_pre", "pin_home")], DOWN_Y)
    # 1. brace the door (it would slide under the O-rings' drag); the right hand moves over the valve's head
    brace = chain(L.move(P["brace"] + 0.06 * up, brace_rot, 2.5, 0.0),
                  L.move(P["brace"] - (BRACE_DOWN + 0.004) * up, brace_rot, 1.5))
    yield from both(brace, R.move(P["park_right"], push_rot, 2.5))   # parked pointing at the door, ready to push
    # the closed fingers push the head home: a down-pointing hand, fingers closed across y (narrower than the lug gap),
    # pushing with the front edge of its finger bodies
    target = P["push_short"] if variant == "short" else P["push_home"]
    R.corr_max = PUSH_CORR_MAX
    if variant == "from_seated":
        target = None
    if target is not None:
      yield from chain(R.move(P["push_start"] + 0.04 * up, push_rot, 2.0, 0.0), R.move(P["push_start"], push_rot, 1.0),
                     R.move(target, push_rot, 2.5, servo=np.array([1.0, 0.0, 1.0])), R.hold(0.5),
                     R.relax(0.3), shift(R, 0.04 * ax + 0.08 * up, push_rot, 1.0),
                     # back to the ready pose in joint space: turning from pointing at the door to pointing down
                     # through IK swung the arm into the door (run 42) or lost the IK (run 44)
                     R.to_joints(READY_Q, 2.0), R.move(P["park_right"], pin_rot, 1.5))
    R.corr_max = CORR_MAX
    # 2. the left hand leaves the brace, takes the handle off the door's face and fits it
    yield from shift(L, 0.06 * up, brace_rot, 0.8)
    if variant != "pin_no_handle":
        yield from chain(L.move(P["handle_pick"] + 0.05 * up, handle_rot, 2.0, 1.0), L.move(P["handle_pick"], handle_rot, 1.0),
                         L.move(P["handle_pick"], handle_rot, 0.6, 0.0), L.move(P["handle_lift"], handle_rot, 1.0),
                         L.move(P["handle_over"], fit_rot, 2.2),
                         L.hold(1.0, servo=np.array([1.0, 1.0, 0.0])),   # settle over the lugs on the arm's own reading
                         L.move(P["handle_fit"], fit_rot, 2.4, servo=np.array([1.0, 1.0, 0.0])),
                         L.hold(0.4))
    if variant == "no_pin":
        yield from chain(stay(L, fit_rot, 0.6, 1.0), shift(L, 0.08 * up, fit_rot, 1.0), wait(2.0))
        return
    # 3. the right hand takes the pin by its bent end and pushes it through lug, knuckle and lug
    # while the pin is pushed in slowly, the left hand sweeps the lever up and down so the knuckle's hole passes
    # through line with the lugs' and the pin drops in (where the knuckle comes to rest varies by about 2 mm)
    sz = np.array([0.0, 0.0, 1.0])

    def lever_at(press):
        return P["handle_fit"] + np.array([0.0, 0.0, PRESS - press])

    if variant != "pin_no_handle":
        pin_hold = chain(L.hold(9.2, servo=sz),   # the sweep starts as the pin meets the knuckle
                         L.move(lever_at(SWEEP[0]), fit_rot, 1.0, servo=sz), L.move(lever_at(SWEEP[1]), fit_rot, 2.0, servo=sz),
                         L.move(lever_at(SWEEP[0]), fit_rot, 2.0, servo=sz),
                         # a second sweep while the pin is pushed home, so it finds the far side too (run 45 wedged)
                         L.move(lever_at(SWEEP[1]), fit_rot, 1.5, servo=sz), L.move(lever_at(PRESS), fit_rot, 1.5, servo=sz),
                         L.hold(1.0, servo=sz))
    else:
        pin_hold = wait(0.1)
    L.corr_max = HOLD_CORR_MAX
    pin = chain(R.move(P["pin_pick"] + 0.06 * up, pin_rot, 2.0, 1.0), R.move(P["pin_pick"], pin_rot, 1.0),
                R.move(P["pin_pick"], pin_rot, 0.6, 0.0), R.move(P["pin_pick"] + 0.06 * up, pin_rot, 1.0),
                R.move(P["pin_pre"] + 0.02 * up, pin_rot, 1.8), R.move(P["pin_pre"], pin_rot, 0.8),
                R.move(P["pin_touch"], pin_rot, 2.0), R.hold(6.0),     # pressed lightly on the knuckle during the sweep
                R.move(P["pin_home"], pin_rot, 3.0), R.hold(0.6))
    yield from both(pin_hold, pin)
    # 4. let go and move away
    # open first, in place: relaxing the arm's trim while still holding moved the lever 6 mm and drew the valve out
    # and one hand at a time, slowly: both opening at once while loaded kicked the handle off its pin (run 35)
    yield from chain(stay(R, pin_rot, 1.2, 1.0), shift(R, 0.08 * up, pin_rot, 1.0), R.relax(0.3))
    yield from chain(stay(L, fit_rot, 1.2, 1.0), shift(L, 0.08 * up, fit_rot, 1.0), L.relax(0.3))
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
        cam_height: int = 176, hold_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    if os.environ.get("DEMO_VARIANT") == "reversed":
        os.environ["SOFTSERVE_VALVE_REVERSED"] = "1"
    if os.environ.get("DEMO_VARIANT") == "from_seated":
        os.environ["SOFTSERVE_VALVE_SEATED"] = "1"
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
        g = task.geometry(st)
        S = task._STATE
        door_shift = (1000 * float(np.linalg.norm(st["door"].pos - S["door0"]))) if S["door0"] is not None else None
        return {"step": steps, "score": info.get("score"), "levels": S["levels"], "released": S["released"],
                "gap_mm": last_gap["mm"],
                "drag_n": round(S.get("drag_n", 0.0), 1), "door_shift_mm": door_shift,
                "valve_door_mm": np.round(task._local(st["door"], st["valve"].pos), 2).tolist(),
                "valve_depth_mm": round(g["valve_depth_mm"], 2), "valve_axis_mm": round(g["valve_axis_mm"], 2),
                "slot_deg": round(math.degrees(g["slot_angle"]), 1), "bar_in_fork": g["bar_in_fork"],
                "knuckle_mm": round(g["knuckle_mm"], 2),
                "knuckle_door_mm": np.round(task._local(st["door"], st["handle"].pos) - [task.PIVOT_DX, 0, task.PIVOT_DZ], 2).tolist(), "lever_deg": round(math.degrees(g["lever_from_raised"]), 1),
                "pin_axis_mm": round(g["pin_axis_mm"], 2), "pin_span": [round(v, 1) for v in g["pin_span"]],
                "corr_mm": {"L": np.round(1000 * ctx.left.corr, 1).tolist(),
                            "R": np.round(1000 * ctx.right.corr, 1).tolist()},
                "w": {n: np.round(st[n].pos - [0, 0, task.TOP], 4).tolist() for n in ("door", "valve", "handle", "pin")},
                "tcp": {"L_cmd": np.round(ctx.left.pos - [0, 0, task.TOP], 3).tolist(),
                        "L_meas": np.round(ctx.joint_tcp("left")[0] - [0, 0, task.TOP], 3).tolist(),
                        "R_cmd": np.round(ctx.right.pos - [0, 0, task.TOP], 3).tolist(),
                        "R_meas": np.round(ctx.joint_tcp("right")[0] - [0, 0, task.TOP], 3).tolist(),
                        "grip": [round(float(ctx.left.grip), 2), round(float(ctx.right.grip), 2)]}}

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
              "max_trim_mm": {"left": round(1000 * ctx.left.max_corr, 1), "right": round(1000 * ctx.right.max_corr, 1)},
              "min_arm_link_gap_mm": round(1000 * min_gap, 1) if min_gap < float("inf") else None,
              "levels": S["levels"], "door_moved": S["moved"], "finger_check": S["finger_check"]}
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
    """No simulator: yam_arena IK error of every planned target, for both grips (and the one the script picks)."""
    import importlib

    from yam_arena import ik, station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)
    _rotations()
    P = plan(task)
    left, right = Arm("left", station.BASES["left"]), Arm("right", station.BASES["right"])
    for name, p in P.items():
        arm = left if name.startswith(("handle", "brace")) else right
        errs = []
        for R in (DOWN_Y, DOWN_Y @ FLIP):
            _, pe, re = ik.solve(np.asarray(p, float), ik.matrix_to_quat_xyzw(R), arm.q, base_pos=arm.base, iters=300)
            errs.append(f"{1000 * pe:.1f}mm/{re:.3f}")
        print(f"IK {name:12s} {arm.name:5s} {np.round(p - [0, 0, task.TOP], 3).tolist()} grip0 {errs[0]} grip180 {errs[1]}",
              flush=True)
    # the left arm's handle path, solved step by step from the previous solution as the demo does
    for seq_name, arm, names in (("left handle", left, ["handle_pick", "handle_lift", "handle_over", "handle_fit"]),
                                 ("right pin", right, ["pin_pick", "pin_pre", "pin_home"])):
        q = arm.q.copy()
        R = best_grip(arm, [P[n] for n in names], DOWN_Y)
        prev = None
        worst = 0.0
        for n in names:
            target = P[n]
            start = prev if prev is not None else target
            for k in range(1, 41):
                pt = start + (target - start) * k / 40
                q, pe, _ = ik.solve(pt, ik.matrix_to_quat_xyzw(R), q, base_pos=arm.base, iters=60)
                worst = max(worst, pe)
            prev = target
            print(f"SEQ {seq_name}: {n} worst so far {1000 * worst:.1f} mm  q={np.round(q, 2).tolist()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ik", action="store_true", help="only print the IK error of every planned target")
    a = ap.parse_args()
    if a.ik:
        ik_check(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
