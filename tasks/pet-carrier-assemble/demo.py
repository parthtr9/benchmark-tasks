"""Scripted full-score demo for pet-carrier-assemble, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record]
    python demo.py task.py --ik          # no simulator: IK error and arm-to-arm clearance along the whole plan

Open loop: every target comes from the task file's fixed layout and the clock. The only feedback is each arm's own
joint readings (a trim that keeps the measured tool point on the plan under load). No object pose or task state
steers it.

1. Slide: the right hand's closed fingertips steady the bottom shell at its sill ledge; the left hand pinches the
   top's head ledge (pads above and below it) and draws the top forward until it has clicked home.
2. Knobs: each hand pinches the grip ridge of the knob on its side and turns it a quarter turn clockwise.
3. Door: the left hand takes the door out of its stand by a bar of its upper sunburst, carries it in front of the
   doorway, pushes its four prongs into the four open hinges, stands it on the sill and keeps holding it there.
4. Hinges: the right hand's closed fingertips push the two right-hand hinge levers round to closed; the left hand
   lets go of the door and pushes the two left-hand levers round; both hands move away.

Negative variants (DEMO_VARIANT=...):
  knobs_first   the knobs are turned before the top is slid home (0: the cam bars meet the top's front wall)
  no_knobs      top slid home, knobs left unlocked, door and hinges done (0.25)
  door_open     door put into the open hinges and let go, no hinge turned (0.5)
  two_hinges    only the right-hand pair of hinges is closed (0.5)
  hinges_first  all four hinges are closed before the door is offered to them (0.5)
  held          everything done, but the left fingertips stay on the last lever (0.75)
Also DEMO_VARIANT=no_brace (the top is pulled without the right hand steadying the bottom): not a negative case,
the outcome depends on the table's friction.
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
VARIANTS = ("", "knobs_first", "no_knobs", "door_open", "two_hinges", "hinges_first", "held", "no_brace")
KI, CORR_MAX = 0.2, 0.02        # joint-feedback trim: gain per step and limit (m)
PULL_CORR_MAX = 0.045           # trim limit while the left hand draws the top over its click
BRACE_CORR_MAX = 0.06           # trim limit while the right hand steadies the bottom
CARRY_CORR_MAX = 0.035          # trim limit while the left hand carries the door (the arm sags under load)
SWEEP_CORR_MAX = 0.03           # trim limit while fingertips push a hinge lever round


def fwd(yaw: float = 0.0, pitch: float = 0.0, roll: bool = False) -> np.ndarray:
    """A forward-pointing grip (columns: closing axis, the pads' width axis, approach axis), turned ``yaw`` about
    the vertical and pitched ``pitch`` (negative: nose down). ``roll``: turned a quarter turn about its approach
    axis, so the pads are above and below."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    a = np.array([cy * cp, sy * cp, sp])
    c = np.array([-cy * sp, -sy * sp, cp]) if roll else np.array([-sy, cy, 0.0])
    return np.c_[c, np.cross(a, c), a]


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


FLIP = np.diag([-1.0, -1.0, 1.0])       # the same parallel grip turned half a turn about its approach axis


def best_flip(arm, poses):
    """np.eye(3) or FLIP: whichever turn of the grip about its approach axis gives the smaller worst IK error along
    the straight moves through ``poses`` (position, rotation) from the arm's last command, solved in steps as the
    arm will move, with the wrist roll kept off its +-120 degree limit."""
    from yam_arena import ik

    out = []
    for F in (np.eye(3), FLIP):
        q, p0, r0, worst = arm.q.copy(), arm.pos.copy(), arm.rot.copy(), 0.0
        for pos, rot in poses:
            pos, rot = np.asarray(pos, float), rot @ F
            for s in np.linspace(0.0, 1.0, 9)[1:]:
                q, pe, re = ik.solve(p0 + s * (pos - p0), ik.matrix_to_quat_xyzw(ik.slerp_matrix(r0, rot, s)), q,
                                     base_pos=arm.base, iters=60)
                worst = max(worst, pe + 0.05 * re + 0.01 * max(0.0, abs(float(q[5])) - 1.9))
            p0, r0 = pos, rot
        out.append((worst, F))
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

    def follow(self, path, duration: float, servo=None):
        """Follow ``path(s)`` -> (position, rotation) for s from 0 to 1."""
        from yam_arena import ik

        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            pos, rot = path(ik.smoothstep(k / n))
            self._solve(pos, rot, servo)
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

    def limit(self, corr_max: float):
        """Generator step that sets the trim limit when the plan reaches it."""
        self.corr_max = corr_max
        return
        yield  # pragma: no cover


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


def shift(arm, d, duration: float, grip: float | None = None, servo=None):
    """Generator: move by ``d`` from wherever the arm's command is when this step starts (not when it was built)."""
    yield from arm.move(arm.pos + np.asarray(d, float), arm.rot, duration, grip, servo)


# -- the plan ------------------------------------------------------------------------------------------------------------
PITCH = math.radians(-25.0)      # closed fingertips used as a pusher point 25 degrees nose down
TIP = 0.0128                     # the pads' tips are this far past the tool point
G_LEDGE = 0.30                   # finger opening (1 = 94 mm) coming onto the top's head ledge (4 mm thick)
G_KNOB = 0.32                    # and onto a knob's grip ridge (9 mm thick)
G_DOOR = 0.125                   # and onto the door's sunburst bar (6 mm wide, in a 12 mm slot each side)
PULL_Y, BRACE_Y = 0.035, -0.055  # where the top is pulled and the bottom steadied, from the middle
BRACE_YAW = math.radians(35.0)   # the steadying hand points in from the right
PULL_PAST = 0.001                # the top is drawn this far past home (it stops on its tabs)
KNOB_TURN = math.radians(90.0)   # no overshoot: a knob is on its stop at a quarter turn
DOOR_UP = 0.003                  # the door is carried in this far above the sill
DOOR_PAST = 0.003                # and pushed this far past where its prongs meet the barrels' back walls
DOOR_AIM_Y = float(os.environ.get("PET_DOOR_AIM_Y", "0.0"))   # sideways aim offset of the held door (0: none)
TIP_PAST_DOOR = 0.005            # the pad tips reach this far behind the door
TRAVEL_X = -0.030                # the door's mid-plane while it crosses in front of the carrier (carrier frame)
LEVER_R = 0.028                  # where along a lever the fingertips push
LEVER_END = math.radians(102.0)   # 12 degrees past the stop: on Isaac a 91 degree sweep left two hinges at 65 and 78 degrees
LEVER_YAW = {"t": math.radians(35.0), "b": math.radians(15.0)}
PARK_BACK = 0.075                # how far in front of the carrier a hand parks
DOOR_TO_LEVER_BACK = 0.05        # after letting go of the door, the left hand backs off this much more, clear of the
#                                  top's head ledge, then rises straight up to
DOOR_TO_LEVER_Z = 0.36           # this tool height above the table before it crosses to the top-left lever: the whole hand
#                                  is then at least 30 mm above the door's top edge (267 mm) and the shell, and it comes
#                                  down only over the lever (the first revision crossed at about 0.22 m and its wrist and
#                                  gripper swept low past the carrier's front left corner and the door face)


def world(T, x, y, z):
    """A carrier-frame point (mm) in the world (m)."""
    return np.array([T.XF + x / 1000, y / 1000, T.TOP + z / 1000])


def poker(roll: bool = True, yaw: float = 0.0) -> np.ndarray:
    return fwd(yaw, PITCH, roll)


def lever_rot(sg: float, where: str = "t") -> np.ndarray:
    """Closed fingertips for a lever on side ``sg``: pads above and below, nose down, turned toward the middle so
    the hand stays on its own side of the door."""
    return poker(True, -sg * LEVER_YAW[where])


def lever_tool(T, sg: float, where: str, a: float) -> np.ndarray:
    """The tool point with closed fingertips behind the lever of the hinge on side ``sg`` (+1 left), ``where`` ("t"
    top, "b" bottom), when the lever is ``a`` rad from open toward closed."""
    z = sum(T.LEVER_Z[where]) / 2 + (0.5 if where == "t" else 0.0)
    axis = world(T, T.HINGE_X, sg * T.HINGE_Y, z)
    u = np.array([-math.cos(a), -sg * math.sin(a), 0.0])          # along the lever
    b = np.array([-math.sin(a), sg * math.cos(a), 0.0])           # the side the fingertips push from
    f = min(max(a / (math.pi / 2), 0.0), 1.0)
    gap = (T.LEVER_W / 2 + 8.5 - 1.0 * f) / 1000                    # lever half width + fingertip half size
    tip_mid = axis + LEVER_R * u + gap * b
    return tip_mid - (TIP - 0.004) * lever_rot(sg, where)[:, 2]


def script(task, ctx):
    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    L, R = ctx.left, ctx.right
    up = np.array([0.0, 0.0, 1.0])
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    X, XY, XYZ, YZ = np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]), np.ones(3), np.array([0.0, 1.0, 1.0])
    ledge_z = sum(T.HEAD_Z) / 2

    # 1. slide the top home: right fingertips on the sill ledge, left hand on the head ledge
    def slide():
        grip0 = world(T, T.SLIDE - 15.0, 0.0, ledge_z) + PULL_Y * ey            # tool point with the ledge between the pads
        home = grip0 - (T.SLIDE / 1000 + PULL_PAST) * ex
        lr = fwd(roll=True)
        lr = lr @ best_flip(L, [(grip0 - 0.075 * ex, lr), (grip0, lr), (home, lr)])
        rr = poker(False, BRACE_YAW)
        touch = world(T, -T.LEDGE_D - 0.5, 0.0, sum(T.SILL_Z) / 2) + BRACE_Y * ey - TIP * rr[:, 2]
        rr = rr @ best_flip(R, [(touch - 0.04 * ex, rr), (touch, rr)])
        left = chain(L.move(grip0 - 0.075 * ex, lr, 2.0, G_LEDGE), L.move(grip0, lr, 1.7), L.hold(0.5, 0.0), L.limit(PULL_CORR_MAX),
                     L.move(home, lr, 3.0, servo=X), L.hold(0.6, servo=X), L.relax(0.4), L.limit(CORR_MAX),
                     L.hold(0.5, G_LEDGE), L.move(home - 0.07 * ex, lr, 1.0))
        if variant == "no_brace":
            yield from left
            return
        right = chain(R.move(touch - 0.05 * ex + 0.10 * up, rr, 1.3, 0.0), R.move(touch - 0.04 * ex + 0.02 * up, rr, 1.0),
                      R.move(touch - 0.002 * ex, rr, 1.3), R.limit(BRACE_CORR_MAX),
                      R.move(touch + 0.0005 * ex, rr, 0.4, servo=X), R.hold(3.8, servo=X), R.relax(0.4), R.limit(CORR_MAX),
                      R.move(touch - 0.04 * ex + 0.02 * up, rr, 0.8))
        yield from both(left, right)

    # 2. turn the two knobs, one hand each
    def knobs():
        def one(A, sg):
            p = world(T, -9.0 - 1.5, sg * T.KNOB_Y, T.KNOB_Z) - TIP * ex            # pad tips 1.5 mm off the knob's disc
            r0 = fwd()
            F = best_flip(A, [(p + 0.06 * up, r0)] + [(p, rot_x(a) @ r0) for a in np.linspace(0.0, KNOB_TURN, 4)])
            r0 = r0 @ F
            r1 = rot_x(KNOB_TURN) @ r0
            # first to a point 6 cm in front of the knob, so the hand coming from the brace or the ledge stays 6-7 cm
            # off the carrier's front face (the first revision's straight path passed 8 mm from the bottom shell)
            return chain(A.move(p - 0.06 * ex + 0.06 * up, r0, 1.5, G_KNOB), A.move(p + 0.06 * up, r0, 1.2),
                         A.move(p, r0, 1.2), A.hold(0.5, 0.0),
                         A.follow(lambda s: (p, rot_x(s * KNOB_TURN) @ r0), 2.0), A.hold(0.3), A.hold(0.5, G_KNOB),
                         A.move(p - 0.035 * ex, r1, 0.8))

        yield from both(one(L, 1.0), one(R, -1.0))

    # 3. the door: out of its stand, across the front of the carrier, prongs into the open hinges
    gy, gz = T.GRIP_BAR
    hold_off = np.array([(TIP_PAST_DOOR + T.DOOR_T / 2000) - TIP, gy / 1000, gz / 1000])      # tool point from the door's origin
    d_stand = np.array(T.DOOR_START)
    d_travel = world(T, 1000 * TRAVEL_X, 0.0, T.DOOR_REST[2]) + DOOR_UP * up
    d_in = world(T, T.DOOR_REST[0] + T.BARREL_BACK - T.PIN_R, 0.0, T.DOOR_REST[2]) + DOOR_UP * up + DOOR_PAST * ex + DOOR_AIM_Y * ey
    dr = fwd(roll=True)

    def door_in(push: bool = True):
        side = np.array([d_travel[0], d_stand[1], d_stand[2]])
        rot = dr @ best_flip(L, [(side + hold_off + 0.035 * up, dr), (d_stand + hold_off, dr), (d_travel + hold_off, dr), (d_in + hold_off, dr)])
        # come round the carrier's front corner: first to the plane the door will travel in, then in to the stand
        yield from chain(L.move(side + hold_off + 0.035 * up, rot, 2.0, G_DOOR), L.move(d_stand + hold_off - 0.04 * ex, rot, 1.5),
                         L.move(d_stand + hold_off, rot, 1.2),
                         L.hold(0.5, 0.0), L.limit(CARRY_CORR_MAX),
                         L.move(d_stand + hold_off + 0.035 * up, rot, 1.0, servo=XYZ),
                         L.move(side + hold_off + 0.035 * up, rot, 2.0, servo=XYZ),
                         L.move(np.array([d_travel[0], d_stand[1], d_travel[2]]) + hold_off, rot, 1.0, servo=XYZ),
                         L.move(d_travel + hold_off, rot, 3.5, servo=XYZ), L.hold(0.3, servo=XYZ),
                         L.move(d_in + hold_off, rot, 1.5, servo=YZ), L.hold(0.3, servo=YZ),
                         # stand the door on the sill while it is still held, so it does not hang askew from the grip
                         L.move(d_in + hold_off - DOOR_UP * up, rot, 0.5, servo=YZ), L.hold(0.2, servo=YZ))

    def door_release():
        # set the door down on the sill and take the load off the hand before the pads open, so no pad carries it away
        yield from chain(L.relax(0.4), L.limit(CORR_MAX), shift(L, -0.0005 * up, 0.2), L.hold(0.5, G_DOOR),
                         shift(L, -0.03 * ex, 0.8))

    # 4. hinge levers pushed round by closed fingertips
    def to_lever(A, sg, where, high: bool = False):
        """Bring closed fingertips beside the open lever, on its outer side. ``high``: from in front of the door, back
        off, rise straight up above the door and the shell, cross over, and come down only over the lever."""
        rot = lever_rot(sg, where)
        start = lever_tool(T, sg, where, 0.0) + 0.012 * sg * ey
        pre = start + 0.03 * sg * ey - 0.03 * ex + (0.02 if where == "t" else 0.0) * up
        rot = rot @ best_flip(A, [(pre, rot), (start, rot), (lever_tool(T, sg, where, LEVER_END), rot)])
        if high:
            back = A.pos - DOOR_TO_LEVER_BACK * ex
            over = np.array([pre[0], pre[1], T.TOP + DOOR_TO_LEVER_Z])
            yield from chain(A.move(back, A.rot, 0.8, 0.0), A.move(np.array([back[0], back[1], over[2]]), A.rot, 1.2),
                             A.move(over, rot, 1.6), A.move(pre, rot, 1.0))
        else:
            yield from A.move(pre, rot, 2.2, 0.0)
        yield from A.move(start, rot, 0.8)

    def sweep(A, sg, where, stay: bool = False, high: bool = False):
        yield from to_lever(A, sg, where, high)
        rot = A.rot
        yield from chain(A.limit(SWEEP_CORR_MAX),
                         A.follow(lambda s: (lever_tool(T, sg, where, s * LEVER_END), rot), 2.2, servo=XYZ), A.hold(0.2, servo=XYZ))
        if not stay:
            yield from chain(A.relax(0.3), A.limit(CORR_MAX), shift(A, -0.03 * ex + 0.04 * sg * ey, 0.8))

    def park(A, sg):
        p = world(T, -1000 * PARK_BACK - 25.0, sg * 170.0, 200.0)
        yield from A.move(p, A.rot, 1.5)

    # both hands start beside the carrier: draw them straight back, clear of its front corners, before anything else
    yield from both(L.move(L.pos - 0.10 * ex, L.rot, 1.0), R.move(R.pos - 0.10 * ex, R.rot, 1.0))
    if variant == "knobs_first":
        yield from knobs()
        yield from slide()
    else:
        yield from slide()
        if variant != "no_knobs":
            yield from knobs()
    if variant == "hinges_first":
        yield from both(chain(sweep(R, -1.0, "t"), sweep(R, -1.0, "b"), park(R, -1.0)), chain(sweep(L, 1.0, "t"), sweep(L, 1.0, "b")))
        yield from chain(door_in(), wait(0.5), door_release(), park(L, 1.0), wait(1.5))
        return
    yield from both(door_in(), chain(wait(8.0), to_lever(R, -1.0, "t")))
    if variant == "door_open":
        yield from chain(wait(0.5), door_release(), both(park(L, 1.0), park(R, -1.0)), wait(1.5))
        return
    yield from both(chain(sweep(R, -1.0, "t"), sweep(R, -1.0, "b"), park(R, -1.0)), L.hold(12.5, servo=YZ))
    yield from door_release()
    if variant == "two_hinges":
        yield from chain(park(L, 1.0), wait(1.5))
        return
    yield from chain(sweep(L, 1.0, "t", high=True), sweep(L, 1.0, "b", stay=variant == "held"))
    if variant == "held":
        yield from wait(3.0)
        return
    yield from chain(park(L, 1.0), wait(1.5))


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

    def measured(self, name: str):
        """Logging only: the arm's measured joints and finger opening (1 open .. 0 closed)."""
        from partkit import _np
        from yam_arena import station
        from yam_arena.constants import ARM_JOINTS, FINGER_OPEN

        art = self.emb._env.unwrapped.scene[f"{name}_robot"]
        jp, jn = _np(art.data.joint_pos)[0], list(art.joint_names)
        q = [float(jp[jn.index(j)]) for j in ARM_JOINTS]
        grip = float(np.clip(jp[jn.index("joint7")] / FINGER_OPEN, 0.0, 1.0)) if "joint7" in jn else 1.0
        return q, grip, station.BASES[name]

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 2400, record: bool = True, cam_width: int = 320,
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

    clouds = clearance.PartClouds(task.OBJECTS)      # logging only: each hand's distance to each part (HANDS lines)
    hand_min: dict = {}

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
                "top_back": round(g["top_back_mm"], 2), "top_side": round(g["top_side_mm"], 2),
                "rim_gap": r2([g["rim_gap_min_mm"], g["rim_gap_max_mm"]]), "ridge": r2(g["ridge_mm"]),
                "knob_deg": r2(g["knob_deg"]), "hinge_deg": r2(g["hinge_deg"]), "pin_off": r2(g["pin_off_mm"]),
                "door_up": round(g["door_up_mm"], 2), "flags": [S["moved"], S["dropped"]],
                "corr_mm": {"L": np.round(1000 * ctx.left.corr, 1).tolist(),
                            "R": np.round(1000 * ctx.right.corr, 1).tolist()},
                "w": {n: np.round(st[n].pos - [0, 0, task.TOP], 4).tolist() for n in ("carrier", "top", "door")},
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
        if steps % 5 == 0 or 230 <= steps <= 280 or 990 <= steps <= 1120:   # every step where the hands travel
            try:                                     # logging only: measured joints against the parts' poses
                from yam_arena.scene import object_states

                gaps = clouds.hand_gaps(object_states(emb._env, emb.task.names),
                                        {"L": ctx.measured("left"), "R": ctx.measured("right")})
                print("HANDS " + json.dumps({"step": steps, **gaps}), flush=True)
                for tag, v in gaps.items():
                    for label, d in v.items():
                        if d < hand_min.get(f"{tag}:{label}", (np.inf, 0))[0]:
                            hand_min[f"{tag}:{label}"] = (d, steps)
            except Exception as exc:  # noqa: BLE001 - logging only
                if steps == 5:
                    print(f"hand gaps unavailable: {exc!r}", flush=True)
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
              "levels": S["levels"], "base_moved": S["moved"], "part_dropped": S["dropped"],
              "finger_check": S["finger_check"],
              "hand_part_min_mm": {k: {"mm": v[0], "step": v[1]} for k, v in sorted(hand_min.items())}}
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
    """No simulator: yam_arena IK error and the arm-to-arm clearance along the whole planned path, solved step by
    step as the demo does."""
    import importlib

    from yam_arena import station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)
    import clearance

    class Dry:
        left, right = Arm("left", station.BASES["left"]), Arm("right", station.BASES["right"])

    ctx = Dry()
    n, gap_min, at = 0, float("inf"), None
    for _ in script(task, ctx):
        n += 1
        if n % 4 == 0:
            try:
                gap, pair = clearance.min_clearance(ctx.left.q, ctx.left.grip, ctx.left.base, ctx.right.q, ctx.right.grip,
                                                    ctx.right.base)
                if gap < gap_min:
                    gap_min, at = gap, (n, pair)
            except Exception as exc:  # noqa: BLE001
                if n == 4:
                    print(f"clearance unavailable: {exc!r}", flush=True)
        if n % 40 == 0:
            print(f"IK step {n:4d} L {np.round(ctx.left.pos - [0, 0, task.TOP], 3).tolist()} worst {ctx.left.worst_mm:.2f} mm | "
                  f"R {np.round(ctx.right.pos - [0, 0, task.TOP], 3).tolist()} worst {ctx.right.worst_mm:.2f} mm", flush=True)
    print(f"IK steps {n} worst left {ctx.left.worst_mm:.2f} mm right {ctx.right.worst_mm:.2f} mm; "
          f"min arm-to-arm gap {1000 * gap_min:.0f} mm at {at}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ik", action="store_true", help="only print the IK error of every planned target")
    a = ap.parse_args()
    if a.ik:
        ik_check(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
