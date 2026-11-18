"""Scripted full-score demo for bench-plane-reassemble, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]
    python demo.py task.py --ik          # no simulator: IK error of every planned target
    python demo.py task.py --dry         # no simulator: play the script on the IK alone

Open loop: every target comes from the task file's fixed layout (where the plane stands, where the two parts lie) and
the clock. The only feedback is the arm's own joint readings: a trim that keeps the tool point and the gripper's
orientation, as the joints report them, on the plan while the arm carries a part, pushes the iron down the frog and
pushes the cam lever (the arm is soft, about 0.25 N per mm at the fingertips, so without it the hand trails the plan
by centimetres under load). No object pose, joint state or task state steers it.

1. The right hand takes the iron assembly across its width with its fingertips, from straight above, lifts it and
   tips over 45 degrees so the iron lies at the frog's angle; it lowers it along the frog's normal over the lever-cap
   screw, 5 mm above its place, and lets go: the iron stays there. With two fingertips behind the iron's top end it
   then pushes the iron down the frog until it drops over the yoke and the yoke stops it.
2. It takes the lever cap across its width, lowers it with the keyhole's wide end over the screw head, presses it
   flat, slides it down under the head, lets go.
3. With closed fingers, pointing down, it pushes the raised cam lever over and down onto its stop, then moves away.
The left hand does nothing (the push on the cam is mostly downward; the 1.9 kg plane stays where it is).

Negative variants (DEMO_VARIANT=...):
  let_go         the iron is set down on the frog, 5 mm above its place, and let go without pushing it down: it
                 stays there, on top of the yoke, for the rest of the minute (0.25)
  askew          the iron is laid on turned 5 degrees about the lever-cap screw and let go: its corners land on the
                 cheeks and it never lies on the frog; lever cap and cam as usual (0)
  no_cap         stops after the iron (0.5)
  cam_open       lever cap hooked, cam lever left raised (0.75)
  edge_touch     the left hand first comes down on the iron's cutting edge on the batten (0)
  shove          after the cap is locked the left hand pushes the plane's knob 9 cm sideways (0.75)
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
VARIANTS = ("", "let_go", "askew", "no_cap", "cam_open", "edge_touch", "shove")
KI, CORR_MAX = 0.2, 0.03        # joint-feedback trim: gain per step and limit (m)
ROT_MAX = 0.20                  # and the limit of its turn (rad)

# -- grip rotations: columns are the TCP's closing axis, its y axis and its approach axis (yam_arena convention) -----------
FLIP = np.diag([-1.0, -1.0, 1.0])                                        # the same grip turned half a turn about its approach


def pitched(deg: float) -> np.ndarray:
    """A grip pointing ``deg`` below level (toward +x), fingers closing along y, wrist turned the way the arm reaches."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[0.0, s, c], [1.0, 0.0, 0.0], [0.0, c, -s]]) @ FLIP


LEAN = pitched(45.0)             # square to the frog's face: leaning 45 degrees toward +x, fingers closing along y
DOWN = pitched(90.0)             # straight down, fingers closing along y


def rotvec(R) -> np.ndarray:
    """The rotation vector (axis times angle) of a rotation matrix."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    a = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = float(np.linalg.norm(v))
    return np.zeros(3) if n < 1e-12 else v / n * a


def rotation(v) -> np.ndarray:
    """The rotation matrix of a rotation vector."""
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = np.asarray(v, float) / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each from where the last command left it. With ``servo`` the
    command is trimmed by the arm's own joint readings so the measured tool point, and the gripper's measured
    orientation, follow the plan under load (the wrist joints give a few degrees under a carried part)."""

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
        self.rcorr = np.zeros(3)
        self.corr_max = CORR_MAX
        self.max_corr = 0.0
        self.max_rcorr = 0.0

    def _solve(self, pos, rot, servo=None):
        from yam_arena import ik

        if servo is not None and self.read is not None:
            meas = self.read()
            if meas is not None:
                self.corr = np.clip(self.corr + KI * np.asarray(servo) * (self.pos - meas[0]), -self.corr_max,
                                    self.corr_max)
                self.max_corr = max(self.max_corr, float(np.abs(self.corr).max()))
                turn = self.rcorr + KI * rotvec(self.rot @ np.asarray(meas[1]).T)
                size = float(np.linalg.norm(turn))
                self.rcorr = turn if size <= ROT_MAX else turn * (ROT_MAX / size)
                self.max_rcorr = max(self.max_rcorr, float(np.linalg.norm(self.rcorr)))
        q, pe, _ = ik.solve(np.asarray(pos, float) + self.corr, ik.matrix_to_quat_xyzw(rotation(self.rcorr) @ rot),
                            self.q, base_pos=self.base, iters=60)
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

    def follow(self, points, rot, duration: float, servo=None):
        """Through a list of tool points at a steady rate."""
        n = max(1, round(duration * CONTROL_HZ))
        pts = [self.pos.copy()] + [np.asarray(p, float) for p in points]
        for k in range(1, n + 1):
            f = k / n * (len(pts) - 1)
            i = min(int(f), len(pts) - 2)
            self._solve(pts[i] + (f - i) * (pts[i + 1] - pts[i]), rot, servo)
            yield

    def hold(self, duration: float, grip: float | None = None, servo=None):
        yield from self.move(self.pos, self.rot, duration, grip, servo)

    def relax(self, duration: float = 0.5):
        c0, r0 = self.corr.copy(), self.rcorr.copy()
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self.corr = c0 * (1 - k / n)
            self.rcorr = r0 * (1 - k / n)
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
MM = 0.001
C45 = math.sqrt(0.5)
TIP = 0.013                         # the pad tips reach 13 mm past the tool point
PAD_HALF = 0.0075                   # closed fingertips: 15 mm across at the very tip
FINGER_OPEN_M = 0.04695             # finger travel at grip 1.0 (yam_arena.constants.FINGER_OPEN)
SQUEEZE = 0.006                     # the fingers are commanded this far inside a part's edges: 30 N a side, and centred
# Each part is held across its width by the fingertips, the gripper square to its face (mm, in the part's own frame):
IRON_GRIP = (0.0, 90.0, 8.4)        # the iron near its balance point; the pads reach 4.6 mm past its back (beside the frog)
CAP_GRIP = (0.0, 70.0, 14.0)        # the lever cap near its balance point (76 mm from the toe); the pads stop 1 mm above its underside,
#                                     because the iron under it is 0.8 mm wider than the cap
IRON_OVER = 19.0                    # the iron travels this far off the frog's face (mm): over the screw's head, under the knob's bulb
IRON_IN = 25.0                      # and comes in from this far up the slope, so its lower end never passes the front knob
IRON_UP = 5.0                       # the iron is laid on this far up the slope from its place (mm), and let go there
IRON_PRESS = 1.5                    # commanded this far into the frog's face when it is laid on
NUDGE_GAP = 34.0                    # the iron is then pushed down the frog by its top end with two fingertips this far
#                                     apart (mm): they pass either side of the lateral lever's thumb piece and the tote
NUDGE_BELOW = 4.0                   # the fingertips' ends reach this far past the iron's back
NUDGE_FROM = 5.0                    # the push starts this far up the slope from where the iron's end was laid
NUDGE_PAST = 2.5                    # and ends this far past where the yoke stops it
IRON_AIM = (0.0, 0.0, 0.0)          # calibration of the laid-on place (mm across, up the slope, off the face)
CAP_OVER = 25.0
CAP_BACK = 12.45                    # the lever cap is laid on this far up the slope: keyhole's wide end over the screw
CAP_SLIDE = 15.5                    # and slid this far down (the screw stops it after 12.45)
CAP_PRESS = 4.0                     # commanded this far into the cap iron: the cap, held by its edges, turns flat on it
CAP_AIM = (0.0, 0.0, 0.0)
CAP_SAG = 0.9                       # at its grip point the lever cap lies this much lower on its batten (it rests on its toe)
CAM_R = 30.0                        # the fingertips push the cam lever this far out from its pivot (mm)
CAM_OVER = 1.5                      # and are commanded this far past the closed stop
CAM_FREE = math.radians(45.0)       # the lever's first part of its swing is free (the cam has not met the spring plate's preload)
CAM_TRIM = 0.20                     # while pushing the lever the joint trim may add up to this (m): the arm is soft
#                                     (about 0.25 N per mm), and the cam needs 20 N and more at the fingertips
CAM_AHEAD = 0.004                   # the tool point this far past the plan means the lever has given way (m)
CAM_BLEND = math.radians(25.0)
PARK = (0.200, -0.250, 0.200)       # where the right hand waits at the end (above the bench top)
KNOB_PUSH = 0.090                   # variant shove: the knob is pushed this far sideways
LET_GO_WAIT = 38.0                  # variant let_go: the let-go iron is left alone this long (s), to the end of the minute


def grip_on(half_width_mm: float) -> float:
    """Finger command that holds a part of this half width with SQUEEZE of spring on each side."""
    return (half_width_mm * MM - SQUEEZE) / FINGER_OPEN_M


def grip_open(gap_mm: float) -> float:
    """Finger command that leaves ``gap_mm`` between the two fingertips."""
    return gap_mm / 2 * MM / FINGER_OPEN_M


def on_batten(T, start, grip_mm) -> np.ndarray:
    """Tool point for a part lying on its batten."""
    return np.array(start, float) + T.rot("Z", T.PART_YAW) @ (np.asarray(grip_mm, float) * MM)


def on_frog(T, origin_bed, grip_mm) -> np.ndarray:
    """Tool point when a part lies along the frog with its origin at (l, p, h) mm."""
    return T.frog_to_world(*(np.asarray(origin_bed, float) + np.asarray(grip_mm, float)))


def cam_points(T, n=60, over=CAM_OVER, stop=None, normals=False):
    """Tool points that carry closed fingertips (pointing straight down, 15 mm across at the tip) round the cam lever's
    swing: their lower edge stays 2 mm behind a point CAM_R out on the lever's back, the leading corner while the
    lever stands, the trailing corner once it slopes down toward the toe; then ``over`` mm past the closed stop.
    Worked out in the plane's own frame (x toward the toe) and turned into the world."""
    home = T.SCREW_P - (T.SEAT_END - T.SHANK / 2)
    pivot = T.bed_to_plane(0.0, home + T.CAM_Y, T.IRON_T + T.CAPIRON_T + T.CAM_Z)
    a0, a1 = math.pi - T.CAM_OPEN, math.pi if stop is None else math.pi - T.CAM_OPEN + stop
    out, nrm = [], []
    for k in range(n + 1):
        a = a0 + (a1 - a0) * k / n
        d = np.array([C45 * (math.sin(a) - math.cos(a)), 0.0, C45 * (math.cos(a) + math.sin(a))])   # the lever's direction
        w = math.atan2(d[2], d[0])
        normal = np.array([-math.sin(w), 0.0, math.cos(w)])
        q = pivot + CAM_R * MM * d + 0.002 * normal
        shift = PAD_HALF * max(-1.0, min(1.0, w / CAM_BLEND))
        out.append(q + np.array([-shift, 0.0, TIP]))
        nrm.append(normal)
    if stop is None:
        out.append(out[-1] - over * MM * normal)
        nrm.append(normal)
    pts = [np.array(T.PLANE_POS) + T.PLANE_ROT @ p for p in out]
    return (pts, [T.PLANE_ROT @ v for v in nrm]) if normals else pts


def push_cam(arm, T, free_s: float, load_s: float, rot):
    """Carry the fingertips round the cam lever's swing (cam_points) with the joint trim on: the trim keeps the tool
    point, as the arm's own joints report it, on the planned arc, and so builds whatever push the cam needs. The trim
    turns with the lever, so the push stays square to the lever's back; and if the joints report the tool point more
    than CAM_AHEAD past the plan (the lever gave way), the trim is halved each step so the hand does not lunge."""
    pts, nrm = cam_points(T, normals=True)
    n_free = round(len(pts) * CAM_FREE / T.CAM_OPEN)
    last = None
    for seg, seconds in ((slice(0, n_free), free_s), (slice(n_free, len(pts)), load_s)):
        way = [arm.pos.copy()] + pts[seg]
        ns = [nrm[seg][0]] + nrm[seg]
        steps = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, steps + 1):
            f = k / steps * (len(way) - 1)
            i = min(int(f), len(way) - 2)
            normal = ns[i] + (f - i) * (ns[i + 1] - ns[i])
            normal = normal / np.linalg.norm(normal)
            if last is not None:
                arm.corr = rotation(np.cross(last, normal)) @ arm.corr      # (a small turn: sine and angle agree)
            last = normal
            meas = arm.read() if arm.read is not None else None
            if meas is not None and float((meas[0] - arm.pos) @ -normal) > CAM_AHEAD:
                arm.corr = 0.5 * arm.corr
            arm._solve(way[i] + (f - i) * (way[i + 1] - way[i]), rot, np.ones(3))
            yield


def iron_rot(T, variant: str = "") -> np.ndarray:
    """The gripper's rotation while it lays the iron on: square to the frog, or (variant askew) turned 5 degrees about
    the frog's normal."""
    if variant != "askew":
        return LEAN
    n = T.PLANE_ROT @ T.R_BED[:, 2]
    return rotation(math.radians(5.0) * n) @ LEAN


def plan(T, variant: str = "") -> dict:
    """Named tool-point targets (world, m) from the task file's fixed layout: name -> (arm, grip rotation, point)."""
    up = np.array([0.0, 0.0, 1.0])
    back = -LEAN[:, 2]                                          # away from the frog along its normal
    p = {}
    g = on_batten(T, T.IRON_START, IRON_GRIP)
    p["iron_pre"] = ("right", DOWN, g + 0.07 * up)
    p["iron_grasp"] = ("right", DOWN, g)
    p["iron_lift"] = ("right", DOWN, g + 0.07 * up)
    p0 = T.ENGAGED_P + IRON_UP
    aim = np.array(IRON_AIM)
    lean = iron_rot(T, variant)
    skew = lean @ LEAN.T                                        # variant askew: a turn about the frog's normal
    screw = T.frog_to_world(0.0, T.SCREW_P, 0.0)

    def iron(dp, h):
        return screw + skew @ (on_frog(T, aim + [0.0, p0 + dp, h], IRON_GRIP) - screw)

    # straight down 25 mm up the slope from its place, 19 mm off the frog's face; down the slope at that height (over
    # the screw's head, the iron's lower end behind and below the front knob's bulb); then along the face's normal
    p["iron_high"] = ("right", lean, iron(IRON_IN, IRON_OVER) + 0.07 * up)
    p["iron_in"] = ("right", lean, iron(IRON_IN, IRON_OVER))
    p["iron_over"] = ("right", lean, iron(0.0, IRON_OVER))
    p["iron_on"] = ("right", lean, iron(0.0, -IRON_PRESS))
    p["iron_off"] = ("right", lean, p["iron_on"][2] + 0.05 * back)
    p["iron_clear"] = ("right", LEAN, p["iron_on"][2] + 0.05 * back + 0.06 * up)
    # the push on the iron's top end: the gripper square to the frog, its two fingertips beyond the iron's end, their
    # flat sides against the end's face (the iron overhangs the frog's top by 6 cm there)

    def nudge(dp, off=0.0):
        return T.frog_to_world(0.0, T.ENGAGED_P + dp + T.IRON_L + PAD_HALF / MM, (TIP - off) / MM - NUDGE_BELOW)

    p["nudge_above"] = ("right", LEAN, nudge(IRON_UP + NUDGE_FROM, -0.05))
    p["nudge_start"] = ("right", LEAN, nudge(IRON_UP + NUDGE_FROM))
    p["nudge_end"] = ("right", LEAN, nudge(-NUDGE_PAST))
    p["nudge_clear"] = ("right", LEAN, nudge(-NUDGE_PAST + 3.0, -0.05))
    g = on_batten(T, T.CAP_START, CAP_GRIP) - CAP_SAG * MM * up
    p["cap_pre"] = ("right", DOWN, g + 0.07 * up)
    p["cap_grasp"] = ("right", DOWN, g)
    p["cap_lift"] = ("right", DOWN, g + 0.07 * up)
    home = T.SCREW_P - (T.SEAT_END - T.SHANK / 2)
    hc = T.IRON_T + T.CAPIRON_T
    aim = np.array(CAP_AIM)

    def cap(dp, h):
        return on_frog(T, aim + [0.0, home + dp, hc + h], CAP_GRIP)

    p["cap_high"] = ("right", LEAN, cap(CAP_BACK, CAP_OVER) + 0.05 * up)
    p["cap_over"] = ("right", LEAN, cap(CAP_BACK, CAP_OVER))
    p["cap_on"] = ("right", LEAN, cap(CAP_BACK, -CAP_PRESS))
    p["cap_slid"] = ("right", LEAN, cap(CAP_BACK - CAP_SLIDE, -CAP_PRESS))
    p["cap_back"] = ("right", LEAN, p["cap_slid"][2] + 0.05 * back)
    p["cap_clear"] = ("right", LEAN, p["cap_slid"][2] + 0.05 * back + 0.06 * up)
    arc = cam_points(T)
    behind = T.PLANE_ROT @ np.array([-0.004, 0.0, 0.002])       # start 4 mm behind the raised lever (heel side)
    p["cam_above"] = ("right", DOWN, arc[0] + behind + 0.03 * up)
    p["cam_start"] = ("right", DOWN, arc[0] + behind)
    p["cam_end"] = ("right", DOWN, arc[-1])
    p["cam_clear"] = ("right", DOWN, arc[-1] + 0.08 * up)
    p["park"] = ("right", DOWN, np.array([PARK[0], PARK[1], T.TOP + PARK[2]]))
    edge = on_batten(T, T.IRON_START, (0.0, 2.0, 2.4))          # variant edge_touch: fingers onto the cutting edge
    p["edge_above"] = ("right", DOWN, edge + 0.08 * up)
    p["edge"] = ("right", DOWN, edge)
    knob = np.array(T.PLANE_POS) + T.PLANE_ROT @ np.array([T.KNOB_X * MM, 0.0, 0.039]) + TIP * up   # variant shove
    p["knob_above"] = ("left", DOWN, knob + np.array([0.0, 0.019 + 0.030, 0.08]))
    p["knob_beside"] = ("left", DOWN, knob + np.array([0.0, 0.019 + 0.030, 0.0]))
    p["knob_pushed"] = ("left", DOWN, knob + np.array([0.0, 0.019 + 0.012 - KNOB_PUSH, 0.0]))
    p["left_away"] = ("left", DOWN, np.array([0.25, 0.25, T.TOP + 0.20]))
    p["beside"] = ("right", DOWN, np.array([0.350, -0.060, T.TOP + 0.120]))   # 2 cm from the plane's side: not clear of it
    return p


def script(task, ctx):
    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    P = {k: v[2] for k, v in plan(task, variant).items()}
    L, R = ctx.left, ctx.right
    S = np.ones(3)
    lean = iron_rot(T, variant)
    if variant == "edge_touch":      # the right hand first closes on the iron's cutting edge, then carries on
        yield from chain(R.move(P["edge_above"], DOWN, 2.5, 1.0), R.move(P["edge"], DOWN, 1.2), R.hold(0.5, grip_on(T.IRON_HALF_W)),
                         R.hold(0.3, 1.0), R.move(P["edge_above"], DOWN, 1.0))
    # 1. the iron assembly: fingertips across its width from above, tipped to the frog's angle, down along the frog's
    #    normal over the screw, let go (it stays where it lies); then pushed down the frog by its top end until the
    #    yoke stops it
    yield from chain(R.move(P["iron_pre"], DOWN, 2.5, 1.0), R.move(P["iron_grasp"], DOWN, 1.2),
                     R.hold(0.6, grip_on(T.IRON_HALF_W)),
                     R.move(P["iron_lift"], DOWN, 1.0, servo=S), R.move(P["iron_high"], lean, 2.5, servo=S),
                     R.hold(0.6, servo=S), R.move(P["iron_in"], lean, 1.5, servo=S),
                     R.move(P["iron_over"], lean, 1.2, servo=S), R.hold(0.8, servo=S),
                     R.move(P["iron_on"], lean, 1.5), R.hold(0.3),
                     R.hold(0.5, 1.0), R.relax(0.3), R.move(P["iron_off"], lean, 0.8))
    if variant == "let_go":          # the iron is set down on the frog and let go, not slid: left there for the rest
        yield from chain(R.move(P["iron_clear"], LEAN, 0.8), R.move(P["park"], DOWN, 1.5), wait(LET_GO_WAIT))   # of the minute
        return
    if variant == "askew":           # laid on askew and let go: not pushed down
        yield from R.move(P["iron_clear"], LEAN, 0.8)
    else:
        yield from chain(R.move(P["nudge_above"], LEAN, 2.0, grip_open(NUDGE_GAP)), R.move(P["nudge_start"], LEAN, 1.0),
                         R.move(P["nudge_end"], LEAN, 3.0, servo=S), R.hold(0.3), R.relax(0.3),
                         R.move(P["nudge_clear"], LEAN, 0.8))
    if variant == "no_cap":
        yield from chain(R.move(P["park"], DOWN, 1.5), wait(1.5))
        return
    # 2. the lever cap: keyhole's wide end over the screw head, slid down under it
    yield from chain(R.move(P["cap_pre"], DOWN, 2.5, 1.0), R.move(P["cap_grasp"], DOWN, 1.2),
                     R.hold(0.6, grip_on(T.CAP_HALF_W)),
                     R.move(P["cap_lift"], DOWN, 1.0, servo=S), R.move(P["cap_high"], LEAN, 2.5, servo=S),
                     R.hold(0.6, servo=S), R.move(P["cap_over"], LEAN, 1.0, servo=S), R.hold(0.8, servo=S),
                     R.move(P["cap_on"], LEAN, 1.5), R.hold(0.3), R.move(P["cap_slid"], LEAN, 1.5), R.hold(0.3),
                     R.hold(0.5, 1.0), R.relax(0.3), R.move(P["cap_back"], LEAN, 0.8), R.move(P["cap_clear"], LEAN, 0.8))
    if variant == "cam_open":
        yield from chain(R.move(P["park"], DOWN, 1.5), wait(1.5))
        return
    # 3. closed fingers push the cam lever over and down onto its stop; the joint trim keeps the fingertips (as the
    #    arm's own joints report them) on the planned arc, which is what builds the push the cam needs
    yield from chain(R.move(P["cam_above"], DOWN, 1.5, 0.0), R.move(P["cam_start"], DOWN, 1.0))
    R.corr_max = CAM_TRIM
    yield from chain(push_cam(R, T, 1.2, 5.0, DOWN), R.hold(2.0, servo=S), R.relax(0.5))
    R.corr_max = CORR_MAX
    yield from R.move(P["cam_clear"], DOWN, 1.0)
    if variant == "shove":           # the right hand waits beside the plane while the left pushes it sideways by its knob
        yield from chain(R.move(P["beside"], DOWN, 1.2), L.move(P["knob_above"], DOWN, 2.5, 0.0),
                         L.move(P["knob_beside"], DOWN, 1.2), L.move(P["knob_pushed"], DOWN, 2.5),
                         L.move(P["knob_beside"], DOWN, 1.0), L.move(P["knob_above"], DOWN, 1.0),
                         L.move(P["left_away"], DOWN, 1.5))
    yield from chain(R.move(P["park"], DOWN, 1.5), wait(1.5))


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

    print("PLAN " + json.dumps({k: np.round(v[2], 4).tolist() for k, v in plan(task, variant).items()}), flush=True)

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def in_hand(st):
        """Logging only: where the two parts are in the right hand's frame (mm), the hand's true tool point against
        the one from its joints, and its finger joints."""
        from partkit import _np
        from yam_arena.constants import TCP_OFFSET

        robot = emb._env.unwrapped.scene["right_robot"]
        origin = _np(emb._env.unwrapped.scene.env_origins[0])
        pose = _np(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")]
        x, y, z, w = pose[3:7]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        tcp = pose[:3] + R @ np.array(TCP_OFFSET) - origin
        jp, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
        out = {"tcp_true_minus_joints_mm": np.round(1000 * (tcp - ctx.joint_tcp("right")[0]), 1).tolist(),
               "fingers_mm": [round(1000 * float(jp[jn.index(j)]), 1) for j in ("joint7", "joint8") if j in jn]}
        try:
            out["arm_torque_Nm"] = np.round(_np(robot.data.applied_torque)[0][:6], 1).tolist()
        except Exception:  # noqa: BLE001 - logging only
            pass
        for name, grip in (("iron", IRON_GRIP), ("lever_cap", CAP_GRIP)):
            p_in = R.T @ (st[name].to_world(np.array(grip) * MM) - tcp)
            out[f"{name}_in_hand_mm"] = np.round(1000 * p_in, 1).tolist()
            out[f"{name}_length_in_hand"] = np.round(R.T @ st[name].rotation()[:, 1], 3).tolist()
        return out

    def row(steps, info):
        """Logging only: reads simulator state for the trace."""
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)
        S = task._STATE
        oi, _ = task.in_bed(st["plane"], st["iron"])
        oc, _ = task.in_bed(st["plane"], st["lever_cap"])
        cap = st["lever_cap"]
        foot = cap.to_world(np.array([0.0, task.CAM_Y, 1.6]) * MM + np.array([0.0, 0.0, cap.sliders.get("leaf", 0.0)
                                                                              + cap.sliders.get("foot", 0.0)]))
        return {"step": steps, "score": info.get("score"), **S["info"],
                "foot_above_cap_iron_mm": round(float(st["iron"].to_local(foot)[2]) / MM - task.IRON_T - task.CAPIRON_T, 2),
                "cap_pitch_deg": round(math.degrees(math.asin(float((task.in_bed(st["plane"], cap)[1])[2, 1]))), 2),
                "iron_bed": np.round(oi, 2).tolist(), "cap_bed": np.round(oc, 2).tolist(),
                "leaf_mm": round(1000 * cap.sliders.get("leaf", 0.0), 2), "foot_mm": round(1000 * cap.sliders.get("foot", 0.0), 2),
                "lateral_deg": round(math.degrees(st["plane"].hinges.get("lateral", 0.0)), 2),
                "tcp_err_mm": {a.name[0]: (np.round(1000 * (ctx.joint_tcp(a.name)[0] - a.pos - a.corr), 1).tolist())
                               for a in (ctx.left, ctx.right)},
                "corr_mm": {"R": np.round(1000 * ctx.right.corr, 1).tolist()},
                "grip_cmd": round(float(ctx.right.grip), 2), **in_hand(st),
                "rot_err_deg": np.round(np.degrees(rotvec(ctx.right.rot @ ctx.joint_tcp("right")[1].T)), 2).tolist(),
                "rot_trim_deg": np.round(np.degrees(ctx.right.rcorr), 2).tolist()}

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
                min_gap = min(min_gap, arm_gap(ctx.left, ctx.right))
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
              "max_turn_trim_deg": round(math.degrees(ctx.right.max_rcorr), 2),
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
        if steps % 20 == 0:
            print(f"DRY {steps:4d} L {np.round(Dry.left.pos - [0, 0, task.TOP], 3).tolist()} "
                  f"R {np.round(Dry.right.pos - [0, 0, task.TOP], 3).tolist()} gap {1000 * gap:.0f} mm", flush=True)
    print(f"DRY steps {steps}, worst IK error {1000 * worst[0]:.1f} mm at step {worst[1]} ({worst[2]}), "
          f"min arm gap {1000 * gap:.0f} mm", flush=True)


def ik_check(task_path: str) -> None:
    """No simulator: yam_arena IK error of every planned target, with the grip the script uses there."""
    import importlib

    from yam_arena import ik, station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)
    arms = {"left": Arm("left", station.BASES["left"]), "right": Arm("right", station.BASES["right"])}
    for variant in ("",):
        for name, (side, rot, p) in plan(task, variant).items():
            arm = arms[side]
            q, pe, re = ik.solve(np.asarray(p, float), ik.matrix_to_quat_xyzw(rot), arm.q, base_pos=arm.base, iters=400)
            print(f"IK {variant or 'demo':14s} {name:12s} {side:5s} {np.round(p - [0, 0, task.TOP], 3).tolist()} "
                  f"err {1000 * pe:.1f} mm / {re:.3f} rad", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ik", action="store_true", help="only print the IK error of every planned target")
    ap.add_argument("--dry", action="store_true", help="play the script on the IK alone (no simulator)")
    a = ap.parse_args()
    if a.ik:
        ik_check(a.task)
    elif a.dry:
        dry_run(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
