"""Scripted full-score demo for warning-triangle-erect, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]
    python demo.py task.py --ik          # no simulator: IK error of the whole plan, step by step

Open loop: every target comes from the task file's fixed layout and the clock. The only feedback is each arm's own
joint readings (a trim that keeps a lifting hand at its planned height under an arm's weight). No object pose, hinge
angle or task state steers it.

The apex of the set-up triangle is 38 to 42 cm above the table, out of the arms' reach, and a folded arm cannot be
pinched on its own (the three bars lie face to face, so a pinch clamps the whole pack). So the arms are raised from
below, near their rivets:

1. The right hand pinches the near end of the folded pack from above and walks it a quarter turn round the swivel to
   the stop: the pack now lies across the base, facing the robot, with nothing under its bars.
2. The right hand, fingers closed and pitched 45 degrees, goes under the left arm's lower edge near its tip end and
   lifts it to 24 degrees (0.5 N there). The left hand, nearly level, goes under the same arm 13 cm from its rivet;
   the right hand lets the arm down onto it and then presses its fingertips on the folded right arm's face, which
   keeps the triangle against its swivel stop. The left hand raises the arm to just past 60 degrees and holds it
   (let go, the arm falls back).
3. The right hand goes under the right arm 7 cm from its rivet and raises it. As that arm comes up, its outer rim
   meets the pin on the left arm, the pin rides up the rim's bevel (the bars spring apart) and drops into the slot;
   the hand pushes a few millimetres further against the joined triangle.
4. The hands lower away and back out, one after the other. The arms settle on the pin and the triangle stands.

Negative variants (DEMO_VARIANT=...):
  turn_only  the pack is turned across the base and nothing else (0.25)
  one_arm    the left arm is raised and let go without the other: it falls back (0.25)
  no_hold    the left hand lets go before the right arm arrives: both arms end folded (0.25)
  short      the right hand stops 45 mm short: the pin never reaches the slot; let go, both arms fall back
  no_turn    the hands go straight to raising the arms with the pack still along the base: nothing is raised (0.0)
  drag       after the quarter turn the right hand keeps its pinch and pulls sideways: the triangle stays put (0.25)
Writes one MP4 per camera, setup.png and result.json; DEMO_DEBUG=0 silences the trace printed every 10 steps.
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
VARIANTS = ("", "turn_only", "one_arm", "no_hold", "short", "no_turn", "drag")

# -- the plan's numbers (m, rad) ------------------------------------------------------------------------------------------
TURN_R = 0.150           # the right hand pinches the pack this far from the swivel axis
TURN_GRIP_DOWN = 0.012   # tool point below the bars' upper edge while pinching
TURN_OPEN, TURN_SHUT = 0.55, 0.22        # finger opening (0..1 of 94 mm): 52 mm to straddle the 27 mm pack, then squeezed
TURN_OVER = math.radians(1.5)            # the walk ends this far past the quarter-turn stop
PITCH_IN = math.radians(45.0)            # a hand going under a folded arm (lower edge 41 mm up) points this far below level
PITCH_IN_R = math.radians(45.0)          # the right hand going under the folded right arm
PITCH_TIP = math.radians(15.0)           # ... and levels to this as it carries the left arm's tip end up
PITCH_UP = math.radians(15.0)            # a hand under a raised arm: the arm's edge then rests on a finger face 26 degrees
#                                          from level, inside the friction cone, so the hand pushes it up, not away
#                                          (flatter than this the wrist reaches its limit)
TIP_AT = 0.3666          # the right hand first lifts the left arm near its tip, this far (horizontally) from its rivet
#                          (past the end of the right arm, which lies behind it), ...
TIP_DEG = 24.0           # ... to this angle (0.5 N there; the same lift near the rivet would take 1.9 N on a 45 degree finger)
LIFT_L, LIFT_R = 0.130, 0.100   # each arm is then raised by fingers under its lower edge this far from its own rivet
TOE_T, TOE_L, TOE_R = math.radians(25.0), -math.radians(15.0), math.radians(10.0)   # the lifting hands point this much
#                          toward the midline, which keeps the two wrists apart: under the left arm's tip end, under the
#                          left arm near its rivet, and (least, so no corner of the fingertip reaches under the bottom bar
#                          behind) under the right arm
PARK_PITCH = math.radians(35.0)          # the hands wait aside pointing forward and this far down
# where an arm's edge rests on the upper finger: (in front of the tool point, the closed fingers' outer face from the
# hand's axis there). Near the rivets the fingertip must stop short of the bar behind, so the edge sits near its end.
ON_TIP, ON_PAD = (0.005, 0.0049), (0.0, 0.0060)
IN_X = 0.002             # how much deeper than its nominal place the left hand goes under the half-raised left arm
SEAT_GIVE = 0.0012       # under the folded right arm the hand goes in until its own joints show it held back this much:
SEAT_PAST = 0.012        # the arm's face is then against the step behind the finger pad (searched up to this far past the
SEAT_SPEED = 0.015       # nominal place, at this speed), so the fingertips reach 11 mm under the arm whatever the arm gives
UNDER_R = 0.003          # the pad's face goes in this far below the folded right arm's lower edge (the step is 3.6 mm high)
EASE_R = 0.003           # by the top the right hand has eased this far back, so it does not push the right arm away from
#                          the pin it has to take
PAD_HALF = 0.010         # a raised arm's sloping edge rests on the rivet-side edge of the fingers, this far off the hand's axis
UNDER = 0.006            # the fingers go in this far below a folded arm's lower edge ...
UNDER_HELD = 0.014       # ... and this far below the half-raised arm's (its height depends on how the first hand gives)
HOLD_DEG = 62.0          # the left arm is held here (the slot takes the pin from 59.9 to 62.8 degrees)
PUSH_DEG = 60.0          # the right arm's angle when the pin is home ...
PUSH_PAST = 0.006        # ... and the right hand is sent this much higher (the joined triangle stops it). For the last
GUARD_FROM = 0.008       # stretch, from this far below the joined height, its trims are held: it then presses on the
#                          joined triangle with under 1 N
RIDE_R = 0.0             # how much higher than on its pad's face the right arm's edge rides on the hand at the top
BRACE_AT = 0.160         # the right hand's fingertip end presses on the folded right arm's face this far from the middle,
BRACE_IN = 0.0025        # sent this far into the face (about 0.6 N): the triangle stays against its swivel stop while
#                          the left hand's lift pushes the other way
SERVO = True             # trim the lifting hands' height by their own joint readings
TILT_MAX = math.radians(12.0)   # limit of the pitch trim
KI, CORR_MAX = 0.2, 0.03  # height trim from the arm's own joint readings while it carries an arm: gain per step, limit (m)
LOWER = 0.035            # the hands drop this far to let go
BACK = 0.090             # and back out this far toward the robot


def down(heading: float) -> np.ndarray:
    """Tool rotation (columns: closing axis, y, approach) pointing straight down, the fingers closing across
    ``heading`` (the direction of the bar they pinch)."""
    x = np.array([-math.sin(heading), math.cos(heading), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.c_[x, np.cross(z, x), z]


def scoop(pitch: float, flip: float = 1.0, heading: float = 0.0) -> np.ndarray:
    """Tool rotation pointing ``heading`` from straight ahead (+x; positive toward +y) and ``pitch`` below level, the
    fingers closing in the vertical plane: the upper finger's outer face looks up and forward. ``flip`` -1 is the
    same hand turned half a turn about its axis."""
    h = np.array([math.cos(heading), math.sin(heading), 0.0])
    z = math.cos(pitch) * h + np.array([0.0, 0.0, -math.sin(pitch)])
    x = flip * (math.sin(pitch) * h + np.array([0.0, 0.0, math.cos(pitch)]))
    return np.c_[x, np.cross(z, x), z]


def under_edge(corner, pitch: float, on=None, heading: float = 0.0) -> np.ndarray:
    """The tool point that puts the upper finger's outer face against ``corner`` (the arm's lower front edge)."""
    on = ON_TIP if on is None else on
    h = np.array([math.cos(heading), math.sin(heading), 0.0])
    a = math.cos(pitch) * h + np.array([0.0, 0.0, -math.sin(pitch)])
    n = math.sin(pitch) * h + np.array([0.0, 0.0, math.cos(pitch)])
    return np.asarray(corner, float) - on[0] * a - on[1] * n


def edge_height(T, raised_deg: float, at: float) -> float:
    """Height above the table of an arm's lower edge ``at`` (horizontally) from its rivet when the arm is raised
    ``raised_deg``."""
    a = math.radians(raised_deg)
    return T.PIVOT_Z * T.MM + at * math.tan(a) - (T.BAR_W / 2) * T.MM / math.cos(a)


def pitch_of(rot) -> float:
    """How far below level a hand with tool rotation ``rot`` points (rad)."""
    return math.asin(max(-1.0, min(1.0, -float(rot[2, 2]))))


def tilted(rot, lift: float) -> np.ndarray:
    """``rot`` pitched up by ``lift`` rad about the hand's level cross axis."""
    if abs(lift) < 1e-9:
        return rot
    z = np.asarray(rot, float)[:, 2]
    axis = np.cross(z, [0.0, 0.0, 1.0])
    n = float(np.linalg.norm(axis))
    if n < 1e-6:
        return rot
    k = axis / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + math.sin(lift) * K + (1 - math.cos(lift)) * K @ K) @ np.asarray(rot, float)


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each from where the last command left it."""

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
        self.read = read                 # the arm's tool point from its own joint readings
        self.corr = np.zeros(3)          # the trim added to the commanded tool point
        self.tilt = 0.0                  # and to the commanded pitch (rad, positive lifts the fingertips)
        self.max_corr = 0.0

    def trim(self, axes=(2,)):
        """Move the command by what the arm's own joints say it gives under the load it carries: in height, and
        with ``axes`` (0, 2) also toward and away from the robot."""
        meas = self.read() if self.read is not None else None
        if meas is not None:
            for i in axes:
                self.corr[i] = float(np.clip(self.corr[i] + KI * (self.pos[i] - meas[0][i]), -CORR_MAX, CORR_MAX))
            self.max_corr = max(self.max_corr, float(np.abs(self.corr).max()))
            if 0 in axes:        # a seated hand also keeps its pitch: the load on its fingertips bends the wrist down
                sag = pitch_of(np.asarray(meas[1], float)) - pitch_of(self.rot)
                self.tilt = float(np.clip(self.tilt + KI * sag, -TILT_MAX, TILT_MAX))

    def _send(self, pos, rot):
        from yam_arena import ik

        q, pe, _ = ik.solve(np.asarray(pos, float) + self.corr, ik.matrix_to_quat_xyzw(tilted(rot, self.tilt)), self.q,
                            base_pos=self.base, iters=60)
        self.q = q
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def _solve(self, pos, rot, servo=False):
        if servo:
            self.trim((0, 2) if servo == "xz" else (2,))
        self._send(pos, rot)
        self.pos, self.rot = np.asarray(pos, float), rot

    def relax(self, duration: float = 0.5):
        """Let the trim go, smoothly."""
        c0, t0 = self.corr.copy(), self.tilt
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self.corr, self.tilt = c0 * (1 - k / n), t0 * (1 - k / n)
            self._solve(self.pos, self.rot)
            yield

    def move(self, pos, rot, duration: float, grip: float | None = None):
        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos, float) - p0), slerp(r0, rot, s))
            yield

    def shift(self, delta, duration: float, grip: float | None = None):
        """A move relative to where the arm is when this step starts."""
        yield from self.move(self.pos + np.asarray(delta, float), self.rot, duration, grip)

    def hold(self, duration: float, grip: float | None = None):
        yield from self.move(self.pos, self.rot, duration, grip)

    def arc(self, centre, radius: float, a0: float, a1: float, z: float, rot_at, duration: float):
        """The tool point along a circle about the vertical through ``centre``, from angle a0 to a1; ``rot_at(a)`` is
        the hand's rotation at angle a."""
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            a = a0 + smoothstep(k / n) * (a1 - a0)
            self._solve(np.array([centre[0] + radius * math.cos(a), centre[1] + radius * math.sin(a), z]), rot_at(a))
            yield


def smoothstep(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


def slerp(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    """Rotation interpolated from r0 to r1 (NumPy only)."""
    d = r1 @ r0.T
    c = max(-1.0, min(1.0, (np.trace(d) - 1) / 2))
    ang = math.acos(c)
    if ang < 1e-9:
        return r1
    w = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]]) / (2 * math.sin(ang))
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    a = s * ang
    return (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K) @ r0


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


def plan(T) -> dict:
    """Targets (world, m) from the task file's fixed layout. Each lift is (the arm's lower front edge where the
    fingers go under it, the same edge when the arm is up, the hand's pitch at each, where on the finger it rests)."""
    mm, top = T.MM, T.TOP
    x0 = T.X0
    y_l, y_r = T.HALF * mm - LIFT_L, -(T.HALF * mm - LIFT_R)
    y_tip = T.HALF * mm - TIP_AT                   # negative: on the right, near the left arm's tip
    # after the quarter turn the left arm is the layer facing the robot and the right arm the one behind it
    front_l = x0 - T.Y_LEFT[1] * mm
    front_r = x0 - T.Y_RIGHT[1] * mm
    z_fold = top + T.EDGE_Z * mm - UNDER
    p = {
        "centre": (x0, 0.0),
        "turn_z": top + (T.Z1 * mm) - TURN_GRIP_DOWN,
        "park_r": np.array([x0 - 0.10, -0.26, top + 0.20]),
        "park_l": np.array([x0 - 0.10, 0.26, top + 0.20]),
        # the right hand under the left arm's tip end: in under the folded arm, up to TIP_DEG
        "t": (np.array([front_l, y_tip, z_fold]), np.array([front_l, y_tip, top + edge_height(T, TIP_DEG, TIP_AT)]),
              PITCH_IN, PITCH_TIP, ON_PAD, TOE_T),
        # the left hand under the half-raised left arm near its rivet, up to HOLD_DEG
        "l": (np.array([front_l + IN_X, y_l, top + edge_height(T, TIP_DEG, LIFT_L - PAD_HALF) - UNDER_HELD]),
              np.array([front_l, y_l, top + edge_height(T, HOLD_DEG, LIFT_L - PAD_HALF)]), PITCH_UP, PITCH_UP, ON_TIP,
              TOE_L),
        # the right hand under the folded right arm near its rivet, up until the pin is home and a little more
        "r": (np.array([front_r, y_r, top + T.EDGE_Z * mm - UNDER_R]),
              np.array([front_r - EASE_R, y_r, top + edge_height(T, PUSH_DEG, LIFT_R - PAD_HALF) - RIDE_R + PUSH_PAST]),
              PITCH_IN_R, PITCH_UP, ON_PAD, TOE_R),
        # the right hand keeps its 45 degrees until its fingertips are above the bottom bar behind the arm it carries
        "level_from": {"r": top + T.Z1 * mm + 0.015},
    }
    p["turn_start"] = np.array([x0 - TURN_R, 0.0, p["turn_z"]])
    # the bracing fingertip's end on the right arm's face at half its height (the tool point is 12.8 mm short of it)
    p["brace"] = np.array([x0 - T.Y_RIGHT[1] * mm + BRACE_IN, -BRACE_AT, top + T.PIVOT_Z * mm]) \
        - 0.0128 * scoop(PITCH_IN, 1.0, TOE_T)[:, 2]
    return p


def lift_ends(p, side: str):
    """Tool points at the start and end of a lift."""
    c0, c1, a0, a1, on, hd = p[side]
    return under_edge(c0, a0, on, hd), under_edge(c1, a1, on, hd)


def best_flip(arm, poses, rot_fn) -> float:
    """+1 or -1: which half-turn of the hand has the smaller worst IK error over ``poses``, solved one after
    another from the arm's ready pose."""
    from yam_arena import ik
    from yam_arena.constants import READY

    out = []
    for flip in (1.0, -1.0):
        q, worst = np.array(READY, float), 0.0
        for pos in poses:
            q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot_fn(flip)), q, base_pos=arm.base,
                                 iters=300)
            worst = max(worst, pe + 0.05 * re)
        out.append((worst, flip))
    return min(out)[1]


def turn(arm, p, variant=""):
    """Pinch the near end of the folded pack from above and walk it a quarter turn round the swivel to its stop."""
    a0, a1 = math.pi, 1.5 * math.pi + TURN_OVER
    over = p["turn_start"] + [0.0, 0.0, 0.10]
    yield from chain(arm.move(over, down(a0), 2.0, TURN_OPEN), arm.move(p["turn_start"], down(a0), 1.2),
                     arm.hold(0.6, TURN_SHUT),
                     arm.arc(p["centre"], TURN_R, a0, a1, p["turn_z"], down, 4.0), arm.hold(0.3),
                     arm.shift([0.0, -0.16, 0.0], 4.0) if variant == "drag" else (),      # drags the whole triangle
                     arm.hold(0.5, TURN_OPEN), arm.shift([0.0, 0.0, 0.10], 1.0))


def go_under(arm, p, side: str, flip: float):
    """Closed fingers to just under an arm's lower edge. ``side``: 't' the left arm by its tip end, 'l' and 'r' each
    arm near its own rivet."""
    c0, _, a0, _, on, hd = p[side]
    r_in = scoop(a0, flip, hd)
    p_in = under_edge(c0, a0, on, hd)
    start = p_in - [0.050, 0.0, 0.0]
    yield from chain(arm.move(start + [0.0, 0.0, 0.06], r_in, 2.0, 0.0), arm.move(start, r_in, 1.0),
                     arm.move(p_in, r_in, 1.0))


def seat(arm, p, side: str, flip: float):
    """Closed fingers to in front of an arm's lower edge, then straight in under it at a steady pace until the arm's
    own joint readings show the hand held back by SEAT_GIVE more than while it moved freely: the arm's face is
    against the step behind the finger pad. Without joint readings (a dry run) it stops at the nominal place.
    Leaves ``arm.seat_dx``: how far past the nominal place the hand was sent."""
    c0, _, a0, _, on, hd = p[side]
    r_in = scoop(a0, flip, hd)
    p_in = under_edge(c0, a0, on, hd)
    start = p_in - [0.030, 0.0, 0.0]
    yield from chain(arm.move(start + [-0.020, 0.0, 0.06], r_in, 2.0, 0.0), arm.move(start, r_in, 1.2), arm.hold(0.3))
    step = SEAT_SPEED / CONTROL_HZ
    free, n_free = [], 12
    arm.seat_dx, k = 0.0, 0
    while True:
        k += 1
        x = start[0] + k * step
        if x > p_in[0] + (SEAT_PAST if arm.read is not None else 0.0):
            break
        arm._solve(np.array([x, start[1], start[2]]), r_in)
        yield
        meas = arm.read() if arm.read is not None else None
        if meas is None:
            continue
        give = float(meas[0][0] - x)
        if k <= n_free:
            free.append(give)
        elif give < float(np.mean(free[3:])) - SEAT_GIVE:
            break
    meas = arm.read() if arm.read is not None else None
    if meas is not None and SERVO:
        # carry on from where the hand really is (its own joints): the plan's place for the arm's edge becomes the
        # measured one, and the command keeps the push it has
        arm.corr[0] = float(arm.pos[0] - meas[0][0])
        arm.pos = np.array([float(meas[0][0]), arm.pos[1], arm.pos[2]])
    arm.seat_dx = float(arm.pos[0] - p_in[0])
    yield from arm.hold(0.3)


def rise(arm, p, side: str, flip: float, stop_short: float = 0.0, rise_s: float = 4.0, guard_from: float = 1e9):
    """The edge is carried straight up, the hand levelling as it rises about the edge it carries. ``stop_short``: end
    this far below the plan. The hand's height is trimmed by its own joint readings. Once the edge is above
    ``guard_from`` the trims are held and the hand stops leaning on the arm, so that the last few millimetres press on
    the joined triangle only lightly."""
    c0, c1, a0, a1, on, hd = p[side]
    seated = getattr(arm, "seat_dx", None) is not None and side == "r"
    c0 = c0 + [getattr(arm, "seat_dx", None) or 0.0, 0.0, 0.0]   # from where the hand seated, if it searched for the arm
    arm.seat_dx = None
    mode = "xz" if seated else True       # a seated hand also keeps its place under the edge by its own joints
    c1 = c1 - [0.0, 0.0, stop_short]
    n = max(1, round(rise_s * CONTROL_HZ))
    z_level = p["level_from"].get(side, c0[2])
    for k in range(1, n + 1):
        c = c0 + smoothstep(k / n) * (c1 - c0)
        u = min(1.0, max(0.0, (c[2] - z_level) / max(1e-6, c1[2] - z_level)))
        pitch = a0 + u * (a1 - a0)
        near_top = c[2] >= guard_from
        if near_top:                      # the hand is nearly level here: it stops leaning on the arm it carries, so
            arm.corr[0] *= 0.8            # the arm is not pushed away from the pin it has to take
        arm._solve(under_edge(c, pitch, on, hd), scoop(pitch, flip, hd), servo=(False if near_top else mode)
                   if SERVO else False)
        yield
    yield from arm.hold(0.4)


def carrying(arm, gen):
    """Run ``gen`` (the other hand's work) while ``arm`` keeps its hold at height by its own joint readings."""
    for _ in gen:
        if SERVO:
            arm._solve(arm.pos, arm.rot, servo=True)
        yield


def brace(arm, p, flip: float):
    """The closed fingertips' end against the folded right arm's face, pushing the triangle onto its swivel stop."""
    r = scoop(PITCH_IN, flip, TOE_T)
    start = p["brace"] - [0.040, 0.0, 0.0]
    yield from chain(arm.move(start, r, 1.6), arm.move(p["brace"], r, 0.8))


def drop_back(arm):
    """Lower away from under an arm's edge and back out toward the robot."""
    yield from chain(arm.shift([0.0, 0.0, -LOWER], 1.0), arm.relax(0.3), arm.shift([-BACK, 0.0, 0.0], 1.0))


def let_go(arm, p, side: str, flip: float):
    """Lower away from the arm's edge, back out toward the robot and go aside."""
    yield from chain(drop_back(arm), arm.move(p[f"park_{side}"], scoop(PARK_PITCH, flip), 1.5))


def script(task, ctx):
    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    L, R = ctx.left, ctx.right
    p = plan(T)
    fl = best_flip(L, lift_ends(p, "l"), lambda f: scoop(PITCH_UP, f, TOE_L))
    fr = best_flip(R, [*lift_ends(p, "t"), *lift_ends(p, "r")], lambda f: scoop(PITCH_IN, f, TOE_T))
    if variant != "no_turn":
        # 1. the right hand turns the folded pack a quarter turn across the base
        yield from both(turn(R, p, variant), L.move(p["park_l"], scoop(PARK_PITCH, fl), 3.0, 0.0))
    if variant in ("turn_only", "drag"):
        yield from chain(R.move(p["park_r"], scoop(PARK_PITCH, fr), 2.0, 0.0), wait(2.0))
        return
    # 2. the right hand lifts the left arm by its tip end and the left hand goes under it near its rivet
    yield from both(chain(go_under(R, p, "t", fr), rise(R, p, "t", fr, rise_s=3.0)),
                    chain(L.hold(3.6), go_under(L, p, "l", fl)))
    #    the right hand lets the arm down onto the left hand and braces the triangle against its swivel stop
    yield from chain(drop_back(R), brace(R, p, fr))
    #    the left hand raises the arm the rest of the way and holds it
    yield from rise(L, p, "l", fl)
    yield from carrying(L, R.shift([-0.040, 0.0, 0.0], 0.8))
    if variant in ("one_arm", "no_hold"):
        yield from let_go(L, p, "l", fl)
        if variant == "one_arm":
            yield from chain(R.move(p["park_r"], scoop(PARK_PITCH, fr), 1.5), wait(2.0))
            return
    # 3. the right hand raises the right arm until the pin is in the slot
    join_z = p["r"][1][2] - PUSH_PAST - GUARD_FROM     # from here the right hand watches for the triangle closing
    short = 0.045 if variant == "short" else 0.0
    work = chain(seat(R, p, "r", fr), rise(R, p, "r", fr, stop_short=short, guard_from=join_z))
    yield from (work if variant == "no_hold" else carrying(L, work))
    # 4. both let go
    if variant != "no_hold":
        yield from let_go(L, p, "l", fl)
    yield from let_go(R, p, "r", fr)
    yield from wait(1.0)


class Ctx:
    def __init__(self, emb=None):
        from yam_arena import station

        self.emb = emb
        read = (lambda name: (lambda: self.joint_tcp(name))) if emb is not None else (lambda name: None)
        self.left = Arm("left", station.BASES["left"], read=read("left"))
        self.right = Arm("right", station.BASES["right"], read=read("right"))

    def joint_tcp(self, name: str):
        """The arm's tool point and rotation from its own joint positions (forward kinematics)."""
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

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    last_gap = {"mm": None}

    def row(steps, info):
        """Logging only: reads simulator state for the trace."""
        from yam_arena.scene import object_states

        tri = object_states(emb._env, ["triangle"])["triangle"]
        st = task.status(tri)
        S = task._STATE
        return {"step": steps, "score": info.get("score"), "turn_deg": round(math.degrees(st["turn"]), 1),
                "left_deg": round(math.degrees(st["left"]), 1), "right_deg": round(math.degrees(st["right"]), 1),
                "flex_deg": [round(math.degrees(tri.hinges.get(k, 0.0)), 2) for k in ("flex_l", "flex_r")],
                "pin_mm": {k: round(v, 1) for k, v in st["pin"].items()}, "joined": st["joined"],
                "lean_deg": round(math.degrees(st["lean"]), 1), "clear": S["clear"],
                "base": np.round(tri.pos - [task.X0, 0, task.TOP], 4).tolist(), "gap_mm": last_gap["mm"],
                "tcp": {"L": np.round(ctx.left.pos - [task.X0, 0, task.TOP], 4).tolist(),
                        "R": np.round(ctx.right.pos - [task.X0, 0, task.TOP], 4).tolist(),
                        "L_give_mm": np.round(1000 * (ctx.joint_tcp("left")[0] - ctx.left.pos), 1).tolist(),
                        "R_give_mm": np.round(1000 * (ctx.joint_tcp("right")[0] - ctx.right.pos), 1).tolist(),
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
              "report": S["report"], "moved": S["moved"], "tipped": S["tipped"], "finger_check": S["finger_check"]}
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


def dry_run(task_path: str, variant: str = ""):
    """No simulator: play the script and return every step's joint commands and tool points, and each arm's worst IK
    error (mm)."""
    import importlib

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)
    os.environ["DEMO_VARIANT"] = variant
    ctx = Ctx()
    steps = []
    for _ in script(task, ctx):
        steps.append((ctx.left.q.copy(), float(ctx.left.grip), ctx.right.q.copy(), float(ctx.right.grip),
                      ctx.left.pos.copy(), ctx.right.pos.copy(), ctx.left.rot.copy(), ctx.right.rot.copy()))
    return task, steps, {"left": ctx.left.worst_mm, "right": ctx.right.worst_mm}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ik", action="store_true", help="only play the plan through IK and print the worst errors")
    a = ap.parse_args()
    if a.ik:
        for v in VARIANTS:
            _, s, worst = dry_run(a.task, v)
            print(f"IK variant '{v}': {len(s)} steps, worst error left {worst['left']:.1f} mm, right {worst['right']:.1f} mm",
                  flush=True)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
