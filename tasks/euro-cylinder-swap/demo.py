"""Scripted full-score demo for euro-cylinder-swap, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1800] [--no-record]
    python demo.py task.py --dry         # no simulator: play the script on the IK alone

Open loop: every target comes from the task file's fixed layout (the door, the lock's axis, where the new cylinder
and the tray lie) and the clock. The only feedback is each arm's own joint readings: a trim that keeps the measured
tool point on the plan under load, and the "feel" for the cam: the hand turns the key a few degrees with no load on it,
gives a light tug (or push) by setting its target 5 mm ahead of where its joints say it is, and if its joints show
it has not moved 4 mm, eases off and turns a little further. No object pose, joint state or task state steers it.

1. The left hand grips the door's free edge above the lock and holds it for the whole job.
2. The right hand takes the old key by its bow, turns it 7 degrees with no load on it and gives it a light tug; if
   the hand does not move out it eases off and tries 7 degrees the other way, then 14, 21, 28, 35 and 42 degrees
   each way in turn (the old cam's angle is a trial setting it does not read). When the hand moves, the cam is in
   line, and it draws the cylinder out by the key.
3. It turns the hand down so the cylinder hangs from its key, sets the cylinder's far end on the tray's floor, lays
   it down and lets go.
4. It comes down over the new cylinder's key and takes the barrel by its sides, the bow between its fingers, turns
   level and pushes the cylinder into the door until the cam stops on the door's face (25 mm in).
5. It opens, slides back to the key's bow, closes on it and lifts it level. A push with the key at 14 degrees
   meets the cam on the door's face (the reference for centring); with the key at 29 degrees the push goes in, and
   the hand pushes the cylinder on by the known distance to centred, and turns the key back to 5 degrees, where the
   cam stands 25 degrees out of line and the cylinder cannot be drawn out.
6. The right hand lets go of the key and backs away; the left hand lets go of the door; both hands move away.
The keys stay in their cylinders throughout (a turned key is captive in the plug; see the task file).

Negative variants (DEMO_VARIANT=...):
  no_turn      the old key is pulled without turning: the cam holds the cylinder in the case (0)
  wrong_angle  the old key is turned 10 degrees only and pulled hard: the cam holds the cylinder in the case (0)
  cam_in_line  the new cylinder is centred and left with its cam in line, free to be drawn out again (0.5)
  no_brace     the left hand stays away from the door (the door follows the pull)
Writes one MP4 per camera, setup.png and result.json; DEMO_DEBUG=1 prints a trace every 10 steps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
VARIANTS = ("", "no_turn", "wrong_angle", "cam_in_line", "no_brace")
KI, CORR_MAX = 0.2, 0.03        # joint-feedback trim: gain per step and limit (m)


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# -- grip rotations: columns are the TCP's closing axis, its y axis and its approach axis (yam_arena convention) -----------
KEY0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])    # pointing +x, level, fingers closing along y
EDGE = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])   # pointing -y, level, fingers closing along x
DOWN = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])   # pointing down, fingers closing along y


def key_rot(turn: float, pitch: float = 0.0) -> np.ndarray:
    """The key grip with the key turned ``turn`` (clockwise seen from the robot) and the hand pitched ``pitch`` down."""
    return _ry(pitch) @ _rx(turn) @ KEY0


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

    def follow(self, points, rot, duration: float):
        """Through a list of tool points at a steady rate."""
        n = max(1, round(duration * CONTROL_HZ))
        pts = [self.pos.copy()] + [np.asarray(p, float) for p in points]
        for k in range(1, n + 1):
            f = k / n * (len(pts) - 1)
            i = min(int(f), len(pts) - 2)
            self._solve(pts[i] + (f - i) * (pts[i + 1] - pts[i]), rot)
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


# -- the plan ------------------------------------------------------------------------------------------------------------
TIP = 0.013              # the pad tips reach 13 mm past the tool point
LOAD = 0.005             # the light pull or push: the target is kept this far ahead of the measured tool point
GAVE = 0.0045            # "it gave": the tool point has moved this far ...
GAVE_S = 0.4             # ... within this time
TURN_STEPS, SETTLE_STEPS, TUG_STEPS, TUG_RAMP, EASE_STEPS = 5, 5, 18, 6, 5   # a tug cycle: turn, 0.25 s settle, 0.9 s tug, 0.25 s ease
TURN_RATE = math.radians(30.0)    # the unloaded turns
# Key angles tried for the old cylinder (hand; the plug follows about 3 degrees behind). Where its cam rests is not
# known (a trial setting, ECS_CAM_DEG in the task file; 30 degrees by default, either side): the search starts just
# off key-out and works outward, both ways in turn, 7 degrees apart, so one try falls within 3.5 degrees of any cam
# angle up to 42 degrees and inside the cam's 17 degree window. Nothing here reads the setting.
PULL_AT = tuple(math.radians(s * a) for a in (7.0, 14.0, 21.0, 28.0, 35.0, 42.0) for s in (1.0, -1.0))
# Pushing the new cylinder in: first at 14 degrees, where the cam is square on the door's face (that stop is the
# reference for centring), then as above.
PUSH_AT = tuple(math.radians(a) for a in (14.0, 29.0, 35.0, 24.0, 40.0))
# Taking the key out: its window is about 6 degrees wide round key-out; down through it in 2.5 degree steps, and back.
RAMP_STEPS = 16          # the load comes on over 0.8 s
MAX_STEP = 0.003         # and no target moves more than 3 mm a step (60 mm/s) while a part is being drawn
TAKE_UP, TAKE_UP_MAX = 1.0, 2.4   # the load is held at least this long before the turn starts, and until the
TOOK = 0.012             # the tool point has moved this far while the load came on: the part is free already
SETTLED = 0.0005         # tool point moves less than this in 0.3 s (the held door gives a little under the load)
MIDDLE = math.radians(2.0)        # turned on by this after it gave (by then the cam is some degrees inside its window)
OFF_HOME = math.radians(18.0)     # first turn, without load: well off the key-out position, where the key is captive
#                                   (the plug follows about 3 degrees behind the hand)
SWEEP_TO = math.radians(50.0)
PUSH_FROM = math.radians(10.0)    # the new key's first angle for a push
SWEEP_RATE = math.radians(8.0)    # rad/s
BACK_TO = math.radians(5.0)       # the new key is turned back to here without load (cam 25 degrees out of line)
FOUND_AT = math.radians(20.0)     # dry run only: where the script assumes the cam gives
DROOP = 0.003            # the loose cylinder's key hangs this far below the lock's axis (layout: 0.5 mm play in the case)
DROOP_NEW = 0.005        # ... and the new one's, resting 25 mm in (it leans about 5 degrees in the door's hole)
DRAW = 0.085             # the old cylinder is drawn this far out by its key
EDGE_Z = 0.262           # the left hand's height on the door edge
EDGE_IN = 0.020          # its fingertips reach this far onto the door's faces
EDGE_GRIP = 0.40         # finger opening (fraction) holding the 44 mm door
BODY_OPEN, BODY_GRIP = 0.36, 0.04     # finger opening for the 17 mm barrel: approach (34 mm) and hold (about 33 N)
BODY_AT = 0.021          # the hand takes the barrel this far outside the cylinder's middle (fingertips 3 mm short of the cam)
DOWNWARD = math.pi / 2   # hand pitch pointing straight down
STOP_IN = 0.02515        # the new cylinder's leading end is this far into the door when its cam meets the door's face
CENTRE_PAST = 0.0007     # pushed this far past the computed centre: the hand eases back about that much on letting go
                         # (MuJoCo rehearsal: 1.0 mm short without it; the cam has 1.15 mm of play either way)
WRONG_AT = math.radians(10.0)     # wrong_angle variant: the cam 20 degrees off line (18 squeezed through the window's edge)
LAG = math.radians(2.5)  # the wrist's reading this far behind its command after a pause: the key is not turning
NUDGES = (0.0008, -0.0008, 0.0016, -0.0016, 0.0024, -0.0024)   # shifts along the cylinder tried in turn (m, + is in)
NUDGE_LOAD, NUDGE_S = 0.005, 0.8   # each made with a light load, for 0.8 s at most
BACK_STEP, BACK_PAUSE = math.radians(4.0), 0.4
ENTERED = 0.004          # the hand has gone this far past where the cylinder's end meets the door: it is in the hole
SEEK_PITCH, SEEK_R = 0.0024, 0.007    # the search circles outward 2.4 mm a turn, to 7 mm at most
SEEK_SPEED, SEEK_S = 0.012, 6.0       # at 12 mm/s, for 6 s at most
SEEK_LOAD = 0.004        # the push during the search
TRAY_FOOT = (0.045, 0.014, 0.007)     # where the old cylinder's far end is set down in the tray, from the tray's centre
                                      # (Isaac batch a7ae2c69: from 28 mm the laid cylinder slid 36 mm toward the robot)
LAY = tuple(math.radians(a) for a in (90.0, 60.0, 40.0, 30.0))    # hand pitch while the cylinder is laid down
HOME_R = (0.18, -0.20, 0.26)


def plan(T) -> dict:
    """Tool-point targets (world, m) from the task file's fixed layout."""
    p = {}
    top = np.array([0.0, 0.0, T.TOP])
    bow = T.KEY_IN - T.NECK_L - T.BOW / 2                       # a key's bow centre, in its cylinder's frame (x)
    p["old_bow"] = np.array([T.SEAT[0] + bow, T.YK, T.ZK + T.BOW_Z - DROOP]) + top
    p["old_pre"] = p["old_bow"] - np.array([0.060, 0.0, 0.0])
    p["via"] = np.array([0.26, -0.22, 0.20]) + top                  # on the way there: the wrist turns before it folds
    p["old_out"] = p["old_bow"] - np.array([DRAW, 0.0, 0.0])
    # the old cylinder hangs from its key under the downward hand; its far end is set on the tray's floor and the
    # hand swings back and down about it, laying the cylinder flat
    hang = T.CYL_L + T.KEY_GAP + T.NECK_L + T.BOW / 2                       # bow centre to the cylinder's far end
    foot = np.array([T.TRAY_XY[0] + TRAY_FOOT[0], T.TRAY_XY[1] + TRAY_FOOT[1], TRAY_FOOT[2]]) + top
    p["lay"] = [foot + hang * np.array([-math.cos(a), 0.0, math.sin(a)]) for a in LAY]
    p["tray_over"] = p["lay"][0] + np.array([0.0, 0.0, 0.070])
    p["tray_up"] = p["lay"][-1] + np.array([-0.030, 0.0, 0.080])
    # the new cylinder stands on its end, key up: the hand comes down over the key and takes the barrel by its sides,
    # the key's bow between its fingers; turned level, the hand points along the cylinder
    p["new_grasp"] = np.array([T.NEW_XY[0], T.NEW_XY[1], T.CYL_L / 2 + 0.0005 + BODY_AT]) + top
    p["new_over"] = p["new_grasp"] + np.array([0.0, 0.0, 0.085])
    lead = T.CYL_L / 2 + BODY_AT                                            # tool point to the leading end
    p["ins_pre"] = np.array([T.XF - 0.008 - lead, T.YK, T.ZK]) + top
    p["ins_via"] = np.array([T.XF - 0.070 - lead, T.YK - 0.030, T.ZK + 0.020]) + top     # level before it nears the door
    p["ins_stop"] = np.array([T.XF + STOP_IN + 0.003 - lead, T.YK, T.ZK]) + top
    p["ins_face"] = np.array([T.XF - lead, T.YK, T.ZK]) + top               # the cylinder's end at the door's face
    # resting 25 mm in, on its cam, the cylinder is where the lock is, whatever the hand's grip on it was
    p["new_bow"] = np.array([T.XF + STOP_IN - T.CYL_L / 2 + bow, T.YK, T.ZK + T.BOW_Z]) + top
    p["new_bow_low"] = p["new_bow"] - np.array([0.0, 0.0, DROOP_NEW])
    p["new_home"] = np.array([T.SEAT[0] + bow, T.YK, T.ZK + T.BOW_Z]) + top
    p["edge"] = np.array([T.XF + T.DOOR_T / 2, T.EDGE - EDGE_IN + TIP, EDGE_Z]) + top
    p["edge_pre"] = p["edge"] + np.array([0.0, 0.060, 0.0])
    p["home_l"] = p["edge_pre"] + np.array([0.0, 0.0, 0.035])
    p["home_r"] = np.array(HOME_R) + top
    return p


def feel(ctx, arm, direction: float, turn0: float, turn1: float, rate: float, pitch: float = 0.0, load: float = LOAD):
    """Turn the key from ``turn0`` toward ``turn1`` at ``rate`` while keeping a light load along x (``direction`` +1
    push, -1 pull): every step the target is set LOAD ahead of where the arm's joints say the tool point is. Stops
    turning once the tool point has moved GAVE within the last GAVE_S seconds (slow creep of the held door does not
    count). Leaves the angle in ``ctx.turn``."""
    def measured_x():
        m = arm.read() if arm.read is not None else None
        return None if m is None else float(m[0][0])

    y, z = float(arm.pos[1]), float(arm.pos[2])
    ctx.fine = 70                                             # (logging only: a trace line every step for a while)
    arm.corr[0] = 0.0                                         # the load is set from the joints, not from the trim
    n0, n1 = max(1, round(TAKE_UP * CONTROL_HZ)), max(1, round(TAKE_UP_MAX * CONTROL_HZ))
    seen = []
    for k in range(n1):                                       # take up the load, gently, until door and arm have settled
        x = measured_x()
        ramp = min(1.0, (k + 1) / RAMP_STEPS)                 # (a sudden pull would knock the key over its pin)
        arm._solve(arm.pos if x is None else np.array([x + direction * load * ramp, y, z]), key_rot(turn0, pitch))
        yield
        seen.append(measured_x())
        if seen[-1] is not None and abs(seen[-1] - seen[0]) > TOOK:      # it is already coming: no need to turn
            ctx.turn, ctx.gave, ctx.held_x = turn0, True, seen[0]
            return
        if k >= n0 and (seen[-1] is None or abs(seen[-1] - seen[-7]) < SETTLED):
            break
    recent = [measured_x()] * max(1, round(GAVE_S * CONTROL_HZ))
    ctx.held_x = float(arm.pos[0]) if recent[0] is None else recent[0]     # where the load is first held
    n = max(1, round(abs(turn1 - turn0) / rate * CONTROL_HZ))
    ctx.turn, ctx.gave = turn0, False
    for k in range(1, n + 1):
        ctx.turn = turn0 + (turn1 - turn0) * k / n
        x = measured_x()
        if x is None:                                         # dry run: assume it gives at the layout's angle
            if abs(ctx.turn - turn0) >= abs(FOUND_AT - OFF_HOME):
                arm._solve(arm.pos + np.array([direction * GAVE, 0.0, 0.0]), key_rot(ctx.turn, pitch))
                ctx.gave = True
                yield
                return
            arm._solve(arm.pos, key_rot(ctx.turn, pitch))
            yield
            continue
        arm._solve(np.array([x + direction * load, y, z]), key_rot(ctx.turn, pitch))
        yield
        recent.append(measured_x())
        if direction * (recent[-1] - recent.pop(0)) > GAVE:
            ctx.gave = True
            return


def tug(ctx, arm, direction: float, angles, load: float = LOAD):
    """Find the angle at which the part comes, the way a person does it: turn the key to the next angle of
    ``angles`` with no load on it, give a light tug (or push: ``direction`` +1) along x, and if the hand has not
    moved GAVE from where it rests, let go of the load and try the next angle. Turning under a steady load wedges
    the cam's corner in the edge of the aperture before it is in line (Isaac run 7). The load is the target set ahead
    of where the arm's joints say the tool point is, brought on over TUG_RAMP steps. Leaves the angle in
    ``ctx.turn`` and, in ``ctx.held_x``, where the hand was held under the last tug that did not give."""
    def measured_x():
        m = arm.read() if arm.read is not None else None
        return None if m is None else float(m[0][0])

    y, z = float(arm.pos[1]), float(arm.pos[2])
    arm.corr[0] = 0.0
    ctx.fine = 100                                            # (logging only)
    seat = measured_x()
    ctx.held_x = float(arm.pos[0]) if seat is None else seat
    ctx.gave = False
    for i, angle in enumerate(angles):
        steps = max(TURN_STEPS, round(abs(angle - ctx.turn) / TURN_RATE * CONTROL_HZ))
        start = ctx.turn
        for k in range(1, steps + 1):                         # turn, unloaded, the hand back where the part rests
            arm._solve(arm.pos if seat is None else np.array([seat, y, z]), key_rot(start + (angle - start) * k / steps))
            yield
        ctx.turn = angle
        if seat is None:                                      # dry run: assume it comes at the second angle tried
            if i >= min(1, len(angles) - 1):
                arm._solve(arm.pos + np.array([direction * GAVE, 0.0, 0.0]), key_rot(angle))
                ctx.gave = True
                yield
                return
            continue
        for _ in range(SETTLE_STEPS):                         # let the plug catch up with the key
            arm._solve(np.array([seat, y, z]), key_rot(angle))
            yield
        for k in range(TUG_STEPS):                            # the tug
            x = measured_x()
            arm._solve(np.array([x + direction * load * min(1.0, (k + 1) / TUG_RAMP), y, z]), key_rot(angle))
            yield
            if direction * (measured_x() - seat) > GAVE:
                ctx.gave = True
                return
        ctx.held_x = measured_x()                             # held under the load: where the part's stop is
        for _ in range(EASE_STEPS):                           # no: ease off
            arm._solve(np.array([seat, y, z]), key_rot(angle))
            yield


def draw(arm, direction: float, to_x: float, rot, seconds: float, lead: float = LOAD, wobble: float = 0.0,
         tol: float = 0.001):
    """Keep a light load along x until the tool point reaches ``to_x`` (by the arm's joints) or time runs out. With
    ``wobble`` the hand also moves that far up and down (1.5 Hz), which lets a leaning cylinder's end find a hole."""
    y, z0 = float(arm.pos[1]), float(arm.pos[2])
    arm.corr[0] = 0.0
    n = max(1, round(seconds * CONTROL_HZ))
    for k in range(n):
        m = arm.read() if arm.read is not None else None
        z = z0 + wobble * math.sin(2 * math.pi * 1.5 * k / CONTROL_HZ)
        if m is None:                                         # dry run: a straight move
            x = arm.pos[0] + (to_x - arm.pos[0]) / (n - k)
        else:
            x = float(m[0][0]) + direction * lead * min(1.0, (k + 1) / RAMP_STEPS)
            x = max(x, to_x) if direction < 0 else min(x, to_x)
            x = float(np.clip(x, arm.pos[0] - MAX_STEP, arm.pos[0] + MAX_STEP))
        arm._solve(np.array([x, y, z]), rot)
        yield
        if m is not None and abs(float(arm.read()[0][0]) - to_x) < tol:
            break
    m = arm.read() if arm.read is not None else None      # end without load: the target where the joints are
    arm._solve(np.array([to_x if m is None else float(m[0][0]), y, z0]), rot)
    yield


def enter(ctx, arm, face_x: float, stop_x: float, rot):
    """Push the held cylinder into the door's hole: a light push toward ``stop_x``; if the tool point (by the arm's
    joints) has not passed ``face_x`` by ENTERED, the cylinder's end is on the door's face beside the hole, and the
    hand circles outward under a lighter push until it moves in; then on to ``stop_x``, where the cam stops it."""
    def measured_x():
        m = arm.read() if arm.read is not None else None
        return None if m is None else float(m[0][0])

    y0, z0 = float(arm.pos[1]), float(arm.pos[2])
    arm.corr[0] = 0.0
    if measured_x() is None:                                  # dry run: straight in
        yield from arm.move(np.array([stop_x, y0, z0]), rot, 2.0)
        return
    dy = dz = 0.0
    for _ in range(round(1.2 * CONTROL_HZ)):                  # straight first
        x = measured_x()
        arm._solve(np.array([min(x + 0.006, stop_x), y0, z0]), rot)
        yield
    ctx.sought = False
    if measured_x() < face_x + ENTERED:
        ctx.sought = True
        stalled, angle = measured_x(), 0.0
        for _ in range(round(SEEK_S * CONTROL_HZ)):
            r = SEEK_PITCH * angle / (2 * math.pi)
            angle += SEEK_SPEED / CONTROL_HZ / max(r, 0.0006)
            dy, dz = r * math.cos(angle), r * math.sin(angle)
            x = measured_x()
            arm._solve(np.array([min(x + SEEK_LOAD, stop_x), y0 + dy, z0 + dz]), rot)
            yield
            if measured_x() > stalled + ENTERED or r > SEEK_R:
                break
    for _ in range(round(2.5 * CONTROL_HZ)):                  # on to the stop, the hand staying where it went in
        x = measured_x()
        arm._solve(np.array([min(x + 0.010, stop_x), y0 + dy, z0 + dz]), rot)
        yield
        if abs(measured_x() - stop_x) < 0.001:
            break
    arm._solve(np.array([measured_x(), y0 + dy, z0 + dz]), rot)
    yield


def turn_back(ctx, arm, turn0: float, turn1: float):
    """Turn the key from ``turn0`` back to ``turn1`` without load, in steps of BACK_STEP with a pause after each. The
    cam can only leave the in-line position when it is between the lock's plates. If after a pause the wrist's own
    reading has fallen LAG or more behind its command (compared with before the turn), the cam is caught on a plate:
    the hand takes that step back and shifts the cylinder a little along its axis (NUDGES, in turn) before trying
    again."""
    def measured_turn():
        m = arm.read() if arm.read is not None else None
        if m is None:
            return None
        c = np.asarray(m[1])[:, 0]                             # the closing axis: (0, -cos, -sin) of the key's turn
        return math.atan2(-c[2], -c[1])

    def measured_x():
        return float(arm.read()[0][0])

    p0 = arm.pos.copy()
    start = measured_turn()
    rest = 0.0 if start is None else start - turn0            # the wrist's reading against its command before turning
    n = max(1, math.ceil(abs(turn1 - turn0) / BACK_STEP))
    pause = max(1, round(BACK_PAUSE * CONTROL_HZ))
    k, tries = 0, 0
    x_cmd = float(p0[0])
    x_start = None if start is None else measured_x()
    ctx.nudged = 0
    while k < n:
        cmd = turn0 + (turn1 - turn0) * (k + 1) / n
        for _ in range(pause):
            arm._solve(np.array([x_cmd, p0[1], p0[2]]), key_rot(cmd))
            yield
        got = measured_turn()
        if os.environ.get("DEMO_DEBUG_TURN") and got is not None:
            print(f"TURNBACK k {k} cmd {math.degrees(cmd):.1f} wrist {math.degrees(got):.1f} "
                  f"moved {1000 * (measured_x() - x_start):.1f} mm", flush=True)
        if got is not None and abs(got - cmd - rest) > LAG and tries < len(NUDGES):
            # caught: take the step back, then move the cylinder along its axis until the joints show the hand is
            # NUDGES[tries] from where the turn began (a light push or pull, as before), and try the step again
            want = x_start + NUDGES[tries]
            tries += 1
            ctx.nudged = tries
            back = turn0 + (turn1 - turn0) * k / n
            for _ in range(pause):
                arm._solve(np.array([x_cmd, p0[1], p0[2]]), key_rot(back))
                yield
            for _ in range(round(NUDGE_S * CONTROL_HZ)):
                x = measured_x()
                if abs(x - want) < 0.0002:
                    break
                x_cmd = x + math.copysign(min(NUDGE_LOAD, 4 * abs(want - x) + 0.002), want - x)
                arm._solve(np.array([x_cmd, p0[1], p0[2]]), key_rot(back))
                yield
            x_cmd = measured_x()
            continue
        k += 1
        ctx.turn = cmd


def script(task, ctx):
    T, P = task, plan(task)
    variant = os.environ.get("DEMO_VARIANT", "")
    L, R = ctx.left, ctx.right
    X = np.array([1.0, 0.0, 0.0])
    XZ = np.array([1.0, 0.0, 1.0])
    ALL = np.array([1.0, 1.0, 1.0])
    ctx.turn, ctx.gave, ctx.sought, ctx.held_x, ctx.nudged = 0.0, False, False, 0.0, 0

    def left_hold():
        return chain(L.move(P["edge_pre"], EDGE, 2.2, 0.80), L.move(P["edge"], EDGE, 1.0), L.hold(0.5, EDGE_GRIP))

    def left_away():
        return chain(L.hold(0.5, 0.80), L.move(P["edge_pre"], EDGE, 1.0), L.move(P["home_l"], EDGE, 1.2))

    # 1. left hand on the door edge; right hand to the old key
    reach = chain(R.move(P["via"], key_rot(0.0), 1.4, 0.45), R.move(P["old_pre"], key_rot(0.0), 1.4, servo=ALL),
                  R.move(P["old_bow"], key_rot(0.0), 1.0, servo=ALL), R.hold(0.3, servo=ALL), R.hold(0.5, 0.0))
    if variant == "no_brace":
        yield from reach
    else:
        yield from both(left_hold(), reach)

    if variant == "no_turn":
        yield from chain(R.move(P["old_out"], key_rot(0.0), 3.0, servo=X), R.relax(0.3), wait(2.0))
        return
    # 2. turn the key with no load on it, tug; ease off and try another angle until the hand moves out
    if variant == "wrong_angle":
        yield from chain(R.move(P["old_bow"], key_rot(WRONG_AT), 1.0), R.hold(0.5),
                         R.move(P["old_out"], key_rot(WRONG_AT), 3.0, servo=X), R.relax(0.3), wait(2.0))
        return
    ctx.turn = 0.0
    yield from tug(ctx, R, -1.0, PULL_AT)
    turn = ctx.turn
    yield from draw(R, -1.0, float(P["old_out"][0]), key_rot(turn), 4.0, lead=0.007, wobble=0.001)
    R.corr[:] = 0.0
    yield from R.move(np.array([P["old_out"][0], P["old_out"][1], P["old_bow"][2]]), key_rot(turn), 0.6, servo=XZ)
    # 3. the old cylinder into the tray
    yield from chain(R.move(P["tray_over"], key_rot(turn, LAY[0]), 2.8, servo=XZ),
                     R.move(P["lay"][0], key_rot(turn, LAY[0]), 1.0), R.relax(0.2),
                     R.move(P["lay"][1], key_rot(turn, LAY[1]), 0.7), R.move(P["lay"][2], key_rot(turn, LAY[2]), 0.6),
                     R.move(P["lay"][3], key_rot(turn, LAY[3]), 0.5), R.hold(0.5, 1.0),
                     R.move(P["tray_up"], key_rot(turn, LAY[3]), 0.8))
    # 4. the new cylinder: taken by its barrel from above, turned level, into the door until its cam stops on the
    # door's face
    yield from chain(R.move(P["new_over"], key_rot(0.0, DOWNWARD), 1.8, BODY_OPEN, servo=ALL),
                     R.move(P["new_grasp"], key_rot(0.0, DOWNWARD), 1.2, servo=ALL), R.hold(0.9, servo=ALL),
                     R.hold(0.5, BODY_GRIP),
                     R.move(P["new_over"], key_rot(0.0, DOWNWARD), 0.8, servo=ALL),
                     R.move(P["ins_via"], key_rot(0.0), 3.2, servo=ALL), R.move(P["ins_pre"], key_rot(0.0), 1.0, servo=ALL),
                     R.hold(0.3, servo=ALL))
    yield from enter(ctx, R, float(P["ins_face"][0]), float(P["ins_stop"][0]), key_rot(0.0))
    # 5. by the key: the hand opens, slides back to the bow, closes on it and lifts it level; then turns under a
    # light push until the hand moves in, centres the cylinder, turns back, and takes the key out
    yield from chain(R.hold(0.4, BODY_OPEN), R.hold(0.4, servo=ALL))          # let go; the trim settles without the load
    yield from chain(R.move(P["new_bow_low"], key_rot(0.0), 0.9, servo=ALL), R.hold(0.5, 0.0),
                     R.move(P["new_bow"], key_rot(0.0), 0.8, servo=np.array([0.0, 0.0, 1.0])),
                     R.hold(0.3, servo=np.array([0.0, 0.0, 1.0])))
    ctx.turn = 0.0
    yield from tug(ctx, R, +1.0, PUSH_AT)
    turn = ctx.turn
    # held against the door's face by its cam, the cylinder was exactly STOP_IN deep: centred is a known distance on
    centre = ctx.held_x + (T.DOOR_T / 2 + T.CYL_L / 2 - STOP_IN) + CENTRE_PAST
    yield from draw(R, +1.0, centre, key_rot(turn), 3.5, lead=0.007, wobble=0.0015)
    yield from draw(R, +1.0, centre, key_rot(turn), 1.0, lead=0.004, tol=0.0003)   # the last of the way, gently
    yield from R.hold(0.3)
    if variant == "cam_in_line":
        yield from chain(R.hold(0.5, 1.0), R.move(R.pos - 0.07 * X, key_rot(turn), 1.0), R.move(P["home_r"], key_rot(0.0), 1.5))
        yield from left_away()
        yield from wait(1.5)
        return
    yield from turn_back(ctx, R, turn, BACK_TO)
    # 6. the right hand lets go of the key and backs away along the cylinder; the left hand lets go of the door
    away = chain(R.hold(0.6, 1.0), R.move(R.pos - 0.070 * X, key_rot(ctx.turn), 1.2),
                 R.move(P["home_r"], key_rot(0.0), 1.5))
    if variant == "no_brace":
        yield from away
    else:
        yield from both(away, left_away())
    yield from wait(1.5)


# -- inter-arm clearance from the URDF collision meshes (logging only; commanded joints, no simulator state) -------------
@lru_cache(maxsize=1)
def _meshes():
    from yam_arena import YAM_URDF
    from yam_arena.urdf import rpy_matrix

    root = ET.parse(YAM_URDF).getroot()
    out: dict[str, list[np.ndarray]] = {}
    for link in root.iter("link"):
        for col in link.findall("collision"):
            mesh = col.find("geometry/mesh")
            if mesh is None:
                continue
            o = col.find("origin")
            xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = np.array([float(v) for v in o.get("rpy", "0 0 0").split()]) if o is not None else np.zeros(3)
            raw = open(str(YAM_URDF.parent / mesh.get("filename")), "rb").read()
            n = int(np.frombuffer(raw, np.uint32, 1, 80)[0])
            tri = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)["v"]
            a, b, c = (tri[:, i].astype(float) for i in range(3))
            pts = np.vstack([a, b, c, (a + b) / 2, (b + c) / 2, (a + c) / 2, (a + b + c) / 3])
            out.setdefault(link.get("name"), []).append(pts @ rpy_matrix(rpy).T + xyz)
    res = {}
    for k, v in out.items():
        pts = np.vstack(v)
        _, idx = np.unique(np.round(pts / 0.006).astype(np.int64), axis=0, return_index=True)
        res[k] = pts[np.sort(idx)]
    return res


def arm_points(q, grip, base_pos) -> dict[str, np.ndarray]:
    """World points on every collision mesh, per link, for arm joints ``q`` and finger opening ``grip`` (1 open)."""
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS, FINGER_OPEN
    from yam_arena.urdf import forward_kinematics

    qd = {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)}
    qd.update(joint7=FINGER_OPEN * float(grip), joint8=FINGER_OPEN * float(grip))
    poses = forward_kinematics(YAM_URDF, qd)
    return {link: pts @ poses[link][:3, :3].T + poses[link][:3, 3] + np.asarray(base_pos)
            for link, pts in _meshes().items() if link in poses}


def arm_gap(left, right) -> float:
    """Smallest sampled distance between any link of one arm and any link of the other (m)."""
    pa, pb = arm_points(left.q, left.grip, left.base), arm_points(right.q, right.grip, right.base)
    best = np.inf
    for a in pa.values():
        for b in pb.values():
            lo, hi = np.maximum(a.min(0), b.min(0)), np.minimum(a.max(0), b.max(0))
            if float(np.linalg.norm(np.maximum(lo - hi, 0.0))) >= best:
                continue
            for i in range(0, len(a), 256):
                d = a[i:i + 256, None, :] - b[None, :, :]
                best = min(best, float(np.sqrt((d * d).sum(-1).min())))
    return best


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.turn, self.gave, self.sought, self.held_x, self.nudged, self.fine = 0.0, False, False, 0.0, 0, 0
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

    print("PLAN " + json.dumps({k: np.round(v, 4).tolist() for k, v in plan(task).items()}, default=float), flush=True)

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def row(steps, info):
        """Logging only: reads simulator state for the trace."""
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)
        S = task._STATE
        top = np.array([0.0, 0.0, task.TOP])
        rr = np.asarray(ctx.joint_tcp("right")[1])[:, 0]           # the right hand's closing axis: its key turn
        return {"step": steps, "score": info.get("score"), **S["info"],
                "wrist_deg": round(math.degrees(math.atan2(-rr[2], -rr[1])), 1),
                "turn_deg": round(math.degrees(ctx.turn), 1), "gave": ctx.gave, "sought": ctx.sought, "nudged": ctx.nudged,
                "old": np.round(st["old_cylinder"].pos - top, 4).tolist(),
                "old_plug_deg": round(math.degrees(st["old_cylinder"].hinges["plug"]), 1),
                "new": np.round(st["new_cylinder"].pos - top, 4).tolist(),
                "new_plug_deg": round(math.degrees(st["new_cylinder"].hinges["plug"]), 1),
                "tray": np.round(st["tray"].pos - top, 4).tolist(), "screw": np.round(st["screw"].pos - top, 4).tolist(),
                "tcp_mm": {a.name[0]: np.round(1000 * (ctx.joint_tcp(a.name)[0] - top), 1).tolist()
                           for a in (ctx.left, ctx.right)},
                "tcp_err_mm": {a.name[0]: (np.round(1000 * (ctx.joint_tcp(a.name)[0] - a.pos - a.corr), 1).tolist())
                               for a in (ctx.left, ctx.right)},
                "corr_mm": {"L": np.round(1000 * ctx.left.corr, 1).tolist(),
                            "R": np.round(1000 * ctx.right.corr, 1).tolist()}}

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    best = 0.0
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        best = max(best, float(info.get("score", 0.0) or 0.0))
        snap(res.observation)
        if steps % 5 == 0:
            try:
                min_gap = min(min_gap, arm_gap(ctx.left, ctx.right))
            except Exception as exc:  # noqa: BLE001 - logging only
                if steps == 5:
                    print(f"clearance unavailable: {exc!r}", flush=True)
        if debug and getattr(ctx, "fine", 0) > 0:
            ctx.fine -= 1
            try:
                r = row(steps, info)
                print("FINE " + json.dumps({k: r[k] for k in ("step", "turn_deg", "gave", "door_deg", "old_out_mm",
                      "old_plug_deg", "wrist_deg", "new_out_mm", "new_plug_deg", "cam_off_deg", "tcp_mm", "tcp_err_mm")},
                      default=str), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"fine trace unavailable: {exc!r}", flush=True)
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
    result = {"task_score": score, "success": success, "steps": steps, "variant": variant, "best_score_seen": best,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "max_trim_mm": {"left": round(1000 * ctx.left.max_corr, 1), "right": round(1000 * ctx.right.max_corr, 1)},
              "min_arm_link_gap_mm": round(1000 * min_gap, 1) if min_gap < float("inf") else None,
              "end": S["info"], "finger_check": S["finger_check"]}
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


def dry_run(task_path: str) -> None:
    """No simulator: play the script on the IK alone and report the worst IK error and the smallest arm-to-arm gap."""
    import importlib

    from yam_arena import station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)

    class Dry:
        left = Arm("left", station.BASES["left"])
        right = Arm("right", station.BASES["right"])
        turn, gave, sought, held_x, nudged, fine = 0.0, False, False, 0.0, 0, 0

    steps, gap, worst = 0, float("inf"), (0.0, 0, "")
    for _ in script(task, Dry):
        steps += 1
        for arm in (Dry.left, Dry.right):
            from yam_arena import ik

            pe = float(np.linalg.norm(ik.tcp_pose(arm.q, arm.base)[0] - arm.pos))
            if pe > worst[0]:
                worst = (pe, steps, arm.name)
            if pe > 0.002 and steps % 5 == 0:
                print(f"DRY {steps:4d} {arm.name} IK error {1000 * pe:.1f} mm at {np.round(arm.pos - [0, 0, task.TOP], 3).tolist()}",
                      flush=True)
        if steps % 5 == 0:
            gap = min(gap, arm_gap(Dry.left, Dry.right))
        if steps % 40 == 0:
            print(f"DRY {steps:4d} L {np.round(Dry.left.pos - [0, 0, task.TOP], 3).tolist()} "
                  f"R {np.round(Dry.right.pos - [0, 0, task.TOP], 3).tolist()} q6 {Dry.right.q[5]:.2f} gap {1000 * gap:.0f} mm",
                  flush=True)
    print(f"DRY steps {steps}, worst IK error {1000 * worst[0]:.1f} mm at step {worst[1]} ({worst[2]}), "
          f"min arm gap {1000 * gap:.0f} mm", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--dry", action="store_true", help="play the script on the IK alone (no simulator)")
    a = ap.parse_args()
    if a.dry:
        dry_run(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
