"""On the station table: a hard pet carrier (MidWest Spree 19 in plastic-door carrier, model 1419SPR-PD) part-way
through its assembly, its door end facing the robot. The plastic top rests on the plastic bottom at the back position,
its four hooked tabs standing in the holes of the bottom's rim, 11 mm short of home. The two knobs on the bottom's front
corners are unlocked, the four door hinges are fitted and open (levers pointing out of the front), and the door stands
upright in a wooden stand to the robot's left.

The assembly instructions: "Slide the top slightly forward until it clicks into place on the plastic bottom. Turn the
knobs 90 degrees clockwise to lock the top and the bottom together. With all four door hinges in the open position,
insert the prongs of the door into the bottom and top of the carrier opening. Turn all four door hinges to the closed
position to secure the door in place."

Everything holds by contact. Slid home, the top's four tab feet are under the bottom's rim, so it cannot lift, and a
bar under each of its rim flanges has ridden over a sprung ridge (the click). A knob's quarter turn swings a cam bar up
behind the top's front wall, so the top cannot slide back; short of home the wall is over the cam bar and the knob
cannot turn. Each hinge is a three-sided barrel with a lever: open, its mouth faces forward and a door prong can enter;
closed, the mouth faces a fixed cheek on the shell and the prong is kept.

Sizes: outside 480 x 315 x 320 mm, the 182 x 194 mm doorway and 0.97 kg are retail listing figures; every other
dimension is scaled from the photos in the assembly instructions (measure the bought part).

Frames. Geometry is written in mm in the CARRIER frame: origin on the table under the middle of the bottom shell's
front face, x toward the carrier's back (away from the robot), y to the robot's left, z up. The carrier stands square
to the robot, so the carrier frame is the world frame shifted to (XF, 0, TOP). The top shell's own frame coincides with
the carrier frame when the top is home.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Hinge, Slider, finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = "Get the pet carrier ready for use: top locked onto the bottom and the door fitted and shut."

BASE_C = (0.62, 0.64, 0.67)      # bottom shell: grey plastic
TOP_C = (0.72, 0.16, 0.14)       # top shell: red plastic (1419SPR-PD)
PART_C = (0.86, 0.86, 0.84)      # knobs, hinges and door: light grey plastic
RIDGE_C = (0.45, 0.55, 0.65)
WOOD = (0.72, 0.58, 0.40)

# -- layout (world, m) ---------------------------------------------------------------------------------------------------
XF = 0.28                        # the carrier's front face
STAND_XY = (0.40, 0.31)          # the door stand (the door's middle)
STAND_H, SLOT_D = 100.0, 30.0    # stand height and the depth of its slot (mm)

# -- dimensions (mm, carrier frame) ---------------------------------------------------------------------------------------
L, HW, H = 480.0, 157.5, 320.0   # length, half width, height
WALL = 4.0
FLOOR = (12.0, 16.0)             # the bottom's floor, on four feet
SEAM = 152.0                     # where the top's rim rests on the bottom's rim
FL_OUT = 173.5                   # rim flanges reach this far from the middle
FL_BOTTOM, FL_TOP = 6.0, 4.0     # their thickness
HOLE_Y, STEM_Y = (162.0, 169.0), (163.5, 167.5)      # tab holes in the bottom's rim and the tabs' stems (|y|)
TAB_X = (90.0, 370.0)            # the front edge of each hole = where a stem stops when the top is home
STEM_L, FOOT_L, HOLE_L = 6.0, 10.0, 17.5
FOOT_Z = (139.0, 145.3)          # a tab's foot: 0.7 mm under the bottom's rim
SLIDE = 11.0                     # the top starts this far behind home
RIDGE_X, RIDGE_UP = 240.0, 1.5   # the click ridge in each side rim: crest position and height over the rim
RIDGE_RATE, RIDGE_TRAVEL = 800.0, 2.5
RAMP_IN, RAMP_OUT = math.radians(25.0), math.radians(35.0)
BAR = (RIDGE_X - 7.2, RIDGE_X - 2.44)                # the bar of the top's flange that rides over the ridge
DOOR_HW, DOORWAY_Z = 91.0, (70.0, 264.0)             # the doorway: 182 x 194 mm
LEDGE_D, LEDGE_HW = 22.0, 112.0  # the ledges below and above the doorway that carry the hinges
SILL_Z, HEAD_Z = (63.0, 67.0), (273.0, 277.0)
HINGE_X, HINGE_Y = -11.0, 86.0   # hinge axes (vertical)
BARREL_IN, BARREL_BACK, BARREL_T = 6.0, 5.0, 3.0     # barrel: half mouth width, back wall distance, wall thickness
BARREL_Z = {"b": (67.0, 87.0), "t": (251.0, 273.0)}
LEVER_Z = {"b": (46.0, 54.0), "t": (277.5, 285.5)}
LEVER_L, LEVER_W = 34.0, 9.0
CHEEK = (13.0, 16.0)             # a fixed cheek beside each barrel, toward the middle of the door (distance from the axis)
KNOB_Y, KNOB_Z, KNOB_R = 135.0, 140.0, 19.0
RIDGE_GRIP = (9.0, 36.0, 10.0)   # the knob's grip ridge: thickness, length, height off the disc
CAM_X, CAM_HALF = (4.7, 14.5), 18.0                  # the knob's cam bar, behind the front wall
KNOB_FRICTION, HINGE_FRICTION = 0.03, 0.03           # N m
SHELL_MU = 0.35                  # plastic on plastic
# door (door frame: origin at the middle of its bottom edge, mid-thickness)
DOOR_W, DOOR_H, DOOR_T = 194.0, 200.0, 10.0
DOOR_CORE = 66.0                 # the door is full height out to here; its corners are stepped back for the hinges
WING_Z = (22.0, 178.0)
PIN_R, PIN_Y = 3.0, HINGE_Y
PIN_Z = {"b": (2.0, 22.0), "t": (178.0, 198.0)}
SUN_Z = (47.0, 153.0)            # centres of the two sunbursts
GRIP_BAR = (42.0, SUN_Z[1])      # door frame (y, z): where the demo holds the door, on a horizontal sunburst bar
BAR_W = 6.0
DOOR_REST = (HINGE_X, 0.0, SILL_Z[1])                # the fitted door's origin in the carrier frame


# -- shape helpers (mm in, partkit specs in m out) -------------------------------------------------------------------------
def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _quat(R):
    """(w, x, y, z) of a rotation matrix (Shepperd's method)."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2 * math.sqrt(tr + 1.0)
        q = (s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = ((R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = 2 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s)
    else:
        s = 2 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4)
    return tuple(float(v) for v in q)


def box(centre, size, color, R=None):
    spec = (tuple(float(v) / 1000 for v in centre), tuple(float(v) / 1000 for v in size), color)
    return spec if R is None else spec + (_quat(R),)


def aabb(x0, x1, y0, y1, z0, z1, color):
    return box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)), color)


def pin(y, z0, z1, color, r=PIN_R, x=0.0):
    """A vertical prong of regular octagonal section (four slabs) inscribed in the round prong: corners at r."""
    a = r * math.cos(math.pi / 8)
    w = 2 * a * math.tan(math.pi / 8)
    return [box((x, y, (z0 + z1) / 2), (2 * a, w, z1 - z0), color, _rot_z(k * math.pi / 4)) for k in range(4)]


def _cheeks(where, color):
    """The fixed cheek beside each hinge barrel: a closed barrel's mouth faces it, so one closed hinge keeps its prong."""
    z0, z1 = BARREL_Z[where]
    return [aabb(HINGE_X - 7.0, HINGE_X + 7.0, sg * (HINGE_Y - CHEEK[1]), sg * (HINGE_Y - CHEEK[0]), z0, z1, color)
            for sg in (1.0, -1.0)]


# -- the bottom shell -----------------------------------------------------------------------------------------------------
def _bottom_boxes():
    out = [aabb(0, L, -HW, HW, FLOOR[0], FLOOR[1], BASE_C)]
    for x0 in (12.0, 390.0):
        for sg in (1.0, -1.0):
            out.append(aabb(x0, x0 + 35.0, sg * 95.0, sg * 135.0, 0.0, FLOOR[0], BASE_C))            # feet
    for sg in (1.0, -1.0):
        out.append(aabb(0, L, sg * (HW - WALL), sg * HW, FLOOR[1], SEAM, BASE_C))                   # side wall
        out.append(aabb(0, WALL, sg * DOOR_HW, sg * (HW - WALL), FLOOR[1], SEAM, BASE_C))           # front, beside the doorway
        out.append(aabb(0, L, sg * HW, sg * HOLE_Y[0], SEAM - FL_BOTTOM, SEAM, BASE_C))             # rim flange, inner strip
        out.append(aabb(0, L, sg * HOLE_Y[1], sg * FL_OUT, SEAM - FL_BOTTOM, SEAM, BASE_C))         # outer strip
        x0 = 0.0
        for xs in TAB_X:                                                                           # middle strip, with the holes
            out.append(aabb(x0, xs, sg * HOLE_Y[0], sg * HOLE_Y[1], SEAM - FL_BOTTOM, SEAM, BASE_C))
            x0 = xs + HOLE_L
        out.append(aabb(x0, L, sg * HOLE_Y[0], sg * HOLE_Y[1], SEAM - FL_BOTTOM, SEAM, BASE_C))
    out.append(aabb(L - WALL, L, -(HW - WALL), HW - WALL, FLOOR[1], SEAM, BASE_C))                  # rear wall
    out.append(aabb(0, WALL, -DOOR_HW, DOOR_HW, FLOOR[1], DOORWAY_Z[0], BASE_C))                    # front, under the doorway
    out.append(aabb(-LEDGE_D, 0, -LEDGE_HW, LEDGE_HW, SILL_Z[0], SILL_Z[1], BASE_C))                # sill ledge
    return out + _cheeks("b", BASE_C)


def _ridge(sg):
    """The click: a sprung ridge standing RIDGE_UP above the bottom's rim, a RAMP_IN slope on the side the top's bar
    comes from (the back) and a RAMP_OUT face on the side it ends up (the front)."""
    crest = np.array([RIDGE_X, SEAM + RIDGE_UP])                                 # (x, z)
    yc, wy = sg * (HW + FL_OUT) / 2, FL_OUT - HW - 3.0
    d, n = np.array([math.cos(RAMP_IN), -math.sin(RAMP_IN)]), np.array([math.sin(RAMP_IN), math.cos(RAMP_IN)])
    c1 = crest + d * 4.5 - n * 2.0
    ramp = box((c1[0], yc, c1[1]), (9.0, wy, 4.0), RIDGE_C, _rot_y(RAMP_IN))
    e, m = np.array([-math.cos(RAMP_OUT), -math.sin(RAMP_OUT)]), np.array([-math.sin(RAMP_OUT), math.cos(RAMP_OUT)])
    c2 = crest + e * 2.5 - m * 1.5
    face = box((c2[0], yc, c2[1]), (5.0, wy, 3.0), RIDGE_C, _rot_y(-RAMP_OUT))
    rest = (RIDGE_X / 1000, yc / 1000, SEAM / 1000)
    return Slider(name="ridge_l" if sg > 0 else "ridge_r", rest=rest, boxes=(ramp, face), travel=RIDGE_TRAVEL / 1000,
                  rate=RIDGE_RATE, damping=8.0, mass=0.003, axis="Z", sign=-1)


def _knob(sg):
    """A knob at angle 0 (unlocked): grip ridge upright, cam bar level. A quarter turn clockwise as the robot sees it
    (positive about the carrier's x) lays the ridge level and stands the cam bar up behind the top's front wall."""
    y = sg * KNOB_Y
    t, ln, h = RIDGE_GRIP
    a = KNOB_R * math.cos(math.pi / 8)                       # an octagon inscribed in the round disc
    w = 2 * a * math.tan(math.pi / 8)
    out = [box((-5.0, y, KNOB_Z), (8.0, 2 * a, w), PART_C, _rot_x(k * math.pi / 4)) for k in range(4)]   # disc
    out.append(aabb(-9.0 - h, -9.0, y - t / 2, y + t / 2, KNOB_Z - ln / 2, KNOB_Z + ln / 2, PART_C))           # grip ridge
    out.append(aabb(-1.0, CAM_X[0], y - 4.0, y + 4.0, KNOB_Z - 4.0, KNOB_Z + 4.0, PART_C))                     # shaft
    out.append(aabb(CAM_X[0], CAM_X[1], y - CAM_HALF, y + CAM_HALF, KNOB_Z - 3.5, KNOB_Z + 3.5, PART_C))       # cam bar
    return Hinge(name="knob_l" if sg > 0 else "knob_r", pivot=(0.0, y / 1000, KNOB_Z / 1000), boxes=tuple(out),
                 lower=0.0, upper=math.pi / 2, stiffness=0.0, damping=0.05, effort=KNOB_FRICTION, mass=0.008, axis="X")


def _hinge(sg, where):
    """A door hinge at angle 0 (open): the barrel's mouth and the lever point out of the front (-x). Closed is a
    quarter turn that points both at the middle of the door (+90 degrees about z on the left, -90 on the right)."""
    y, X = sg * HINGE_Y, HINGE_X
    z0, z1 = BARREL_Z[where]
    l0, l1 = LEVER_Z[where]
    a, b, t = BARREL_IN, BARREL_BACK, BARREL_T
    out = [aabb(X + b, X + b + t, y - a - t, y + a + t, z0, z1, PART_C),                          # back wall
           aabb(X - 9.0, X + b, y + a, y + a + t, z0, z1, PART_C), aabb(X - 9.0, X + b, y - a - t, y - a, z0, z1, PART_C),
           aabb(X - LEVER_L, X + 5.0, y - LEVER_W / 2, y + LEVER_W / 2, l0, l1, PART_C)]            # lever
    if where == "b":
        out.append(aabb(X - 4.0, X + 4.0, y - 4.0, y + 4.0, l1, z0, PART_C))                       # neck through the sill
    name = f"hinge_{where}{'l' if sg > 0 else 'r'}"
    lo, hi = (0.0, math.pi / 2) if sg > 0 else (-math.pi / 2, 0.0)
    return Hinge(name=name, pivot=(X / 1000, y / 1000, (z0 + z1) / 2000), boxes=tuple(out), lower=lo, upper=hi,
                 stiffness=0.0, damping=0.05, effort=HINGE_FRICTION, mass=0.006, axis="Z")


CARRIER_POS = (XF, 0.0, TOP + 0.0005)
CARRIER = part("carrier", pos=CARRIER_POS, mass=0.42, boxes=_bottom_boxes(), color=BASE_C, friction=SHELL_MU,
               sliders=(_ridge(1.0), _ridge(-1.0)),
               hinges=(_knob(1.0), _knob(-1.0), _hinge(1.0, "b"), _hinge(-1.0, "b")))


# -- the top shell (top frame = carrier frame when home) ------------------------------------------------------------------
def _top_boxes():
    z0 = SEAM + 0.5                                  # walls end just above the seam: the top rests on its rim flanges
    # roof: a moulded-in carry handle (a bar across a hand hole) and two fields of vent slots, 8 mm bars 8 mm apart
    r0, r1, xc, ye = H - WALL, H, L / 2, HW - 30.0
    out = [aabb(0, L, sg * ye, sg * HW, r0, r1, TOP_C) for sg in (1.0, -1.0)]
    out += [aabb(x0, x1, -ye, ye, r0, r1, TOP_C) for x0, x1 in ((0.0, 50.0), (xc - 80.0, xc - 60.0), (xc + 60.0, xc + 80.0),
                                                                 (L - 50.0, L))]
    out += [aabb(xc - 60.0, xc + 60.0, sg * 45.0, sg * ye, r0, r1, TOP_C) for sg in (1.0, -1.0)]
    out.append(aabb(xc - 60.0, xc + 60.0, -11.0, 11.0, r0, r1, TOP_C))                             # the handle bar
    for f0, f1 in ((50.0, xc - 80.0), (xc + 80.0, L - 50.0)):
        xb = f0 + 8.0
        while xb + 8.0 <= f1 - 4.0:
            out.append(aabb(xb, xb + 8.0, -ye, ye, r0, r1, TOP_C))
            xb += 16.0
    for sg in (1.0, -1.0):
        # side wall: solid bands below and above a row of vertical vent slots, 10 mm bars 10 mm apart
        y0, y1, m0, m1 = sg * (HW - WALL), sg * HW, z0 + 35.0, H - WALL - 25.0
        out.append(aabb(0, L, y0, y1, z0, m0, TOP_C))
        out.append(aabb(0, L, y0, y1, m1, H - WALL, TOP_C))
        out.append(aabb(0, 60.0, y0, y1, m0, m1, TOP_C))
        out.append(aabb(L - 60.0, L, y0, y1, m0, m1, TOP_C))
        xb = 70.0
        while xb + 10.0 <= L - 64.0:
            out.append(aabb(xb, xb + 10.0, y0, y1, m0, m1, TOP_C))
            xb += 20.0
        out.append(aabb(0, WALL, sg * DOOR_HW, sg * (HW - WALL), z0, DOORWAY_Z[1], TOP_C))          # front, beside the doorway
        for x0, x1 in ((0.0, RIDGE_X - SLIDE - 3.2), BAR, (RIDGE_X + 4.5, L)):                      # rim flange, with the bar
            out.append(aabb(x0, x1, sg * HW, sg * FL_OUT, SEAM, SEAM + FL_TOP, TOP_C))
        for xs in TAB_X:                                                                           # hooked tabs
            out.append(aabb(xs, xs + STEM_L, sg * STEM_Y[0], sg * STEM_Y[1], FOOT_Z[0], SEAM, TOP_C))
            out.append(aabb(xs - FOOT_L, xs + STEM_L, sg * STEM_Y[0], sg * STEM_Y[1], FOOT_Z[0], FOOT_Z[1], TOP_C))
    out.append(aabb(L - WALL, L, -(HW - WALL), HW - WALL, z0, H - WALL, TOP_C))                     # rear wall
    out.append(aabb(0, WALL, -(HW - WALL), HW - WALL, DOORWAY_Z[1], H - WALL, TOP_C))               # front, over the doorway
    out.append(aabb(-LEDGE_D, 0, -LEDGE_HW, LEDGE_HW, HEAD_Z[0], HEAD_Z[1], TOP_C))                 # head ledge
    return out + _cheeks("t", TOP_C)


TOP_START = (XF + SLIDE / 1000, 0.0, TOP + 0.0008)
TOP_SHELL = part("top", pos=TOP_START, mass=0.38, boxes=_top_boxes(), color=TOP_C, friction=SHELL_MU,
                 hinges=(_hinge(1.0, "t"), _hinge(-1.0, "t")))


# -- the door (door frame: origin at the middle of its bottom edge) -------------------------------------------------------
def _door_boxes():
    t = DOOR_T / 2
    c = DOOR_CORE
    out = [aabb(-t, t, -c, c, 0.0, 8.0, PART_C), aabb(-t, t, -c, c, DOOR_H - 8.0, DOOR_H, PART_C)]   # bottom and top rails
    for sg in (1.0, -1.0):
        out.append(aabb(-t, t, sg * (c - 8.0), sg * c, 8.0, DOOR_H - 8.0, PART_C))                 # inner stile
        out.append(aabb(-t, t, sg * (DOOR_W / 2 - 8.0), sg * DOOR_W / 2, WING_Z[0], WING_Z[1], PART_C))   # outer stile
        for z0 in (WING_Z[0], 96.0, WING_Z[1] - 8.0):
            out.append(aabb(-t, t, sg * c, sg * (DOOR_W / 2 - 8.0), z0, z0 + 8.0, PART_C))         # wing rails
        for w in ("b", "t"):
            out += pin(sg * PIN_Y, PIN_Z[w][0], PIN_Z[w][1], PART_C)                               # prongs
    for z0 in (84.0, 110.0):
        out.append(aabb(-t, t, -(c - 8.0), c - 8.0, z0, z0 + 6.0, PART_C))                          # the row of holes' rails
    for y in (-37.2, -12.4, 12.4, 37.2):
        out.append(aabb(-t, t, y - 2.0, y + 2.0, 90.0, 110.0, PART_C))                             # and the webs between holes
    for zc, lo, hi in ((SUN_Z[0], 8.0, 84.0), (SUN_Z[1], 116.0, DOOR_H - 8.0)):                     # two sunbursts
        out.append(aabb(-t, t, -20.0, 20.0, zc - 20.0, zc + 20.0, PART_C))                         # the perforated middle
        for k in range(12):
            a = k * math.pi / 6
            ca, sa = math.cos(a), math.sin(a)
            reach = min((c - 8.0) / abs(ca) if abs(ca) > 1e-9 else 1e9,
                        ((hi - zc) if sa > 0 else (zc - lo)) / abs(sa) if abs(sa) > 1e-9 else 1e9)
            r0, r1 = 18.0, reach + 1.0
            rm = (r0 + r1) / 2
            out.append(box((0.0, rm * ca, zc + rm * sa), (DOOR_T, r1 - r0, BAR_W), PART_C, _rot_x(a)))
    return out


DOOR_START = (STAND_XY[0], STAND_XY[1], TOP + (STAND_H - SLOT_D) / 1000 + 0.0005)
DOOR = part("door", pos=DOOR_START, mass=0.12, boxes=_door_boxes(), color=PART_C, friction=SHELL_MU)

# -- the door stand: a wooden block with a slot the door's bottom edge stands in (0.5 mm a side) ---------------------------
_SLOT = DOOR_T / 2 + 0.5
STAND_POS = (STAND_XY[0], STAND_XY[1], TOP)
STAND_BOXES = [
    aabb(-30.0, 30.0, -60.0, 60.0, 0.0, STAND_H - SLOT_D, WOOD),
    aabb(-30.0, -_SLOT, -60.0, 60.0, STAND_H - SLOT_D, STAND_H, WOOD),
    aabb(_SLOT, 30.0, -60.0, 60.0, STAND_H - SLOT_D, STAND_H, WOOD)]
# Three fixed boxes (a yam_arena Box with no mass is fixed to the world).
STAND_PARTS = [Box(f"stand_{i}", pos=tuple(float(STAND_POS[k] + c[k]) for k in range(3)), size=tuple(float(v) for v in sz),
                   color=WOOD) for i, (c, sz, _col) in enumerate(STAND_BOXES)]

OBJECTS = [*STAND_PARTS, CARRIER, TOP_SHELL, DOOR]

# -- scoring -------------------------------------------------------------------------------------------------------------
HOME_TOL = 1.0            # mm: the top is no more than this behind its forward stop
RIM_GAP = 1.5             # mm: the gap under the top's rim, at each corner
SIDE_TOL = 2.5            # mm: the top's sideways offset
RIDGE_OUT = 0.6           # mm: a click ridge counts as sprung back when within this of its rest
TURNED = 80.0             # degrees of a quarter turn that count as locked (knob) or closed (hinge)
PIN_TOL = 4.5             # mm: a prong's axis from its hinge's axis
DOOR_UP = 5.0             # mm: the door's height above (or below) its seat on the sill
TIP_OVER = math.radians(45.0)   # the bottom shell tipped onto a side or an end: 0 for good
BASE_MOVE, BASE_TILT = 0.020, math.radians(5.0)    # logged only (revision 2): moving the carrier costs nothing
RELEASE = 0.03
MOVE_TOL = 0.001          # m: nothing moves more than this over the last 20 steps (1 s)
REACH_MM = {"carrier": 240.0, "top": 240.0, "door": 100.0}   # mm from the origin to the far corner, for turns
HINGES = (("hinge_bl", 1.0, "b"), ("hinge_br", -1.0, "b"), ("hinge_tl", 1.0, "t"), ("hinge_tr", -1.0, "t"))
ENVELOPE = ((-LEDGE_D, L), (-FL_OUT, FL_OUT), (0.0, H))      # the carrier's outline, carrier frame (mm)
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), base0=None, moved=False, tipped=False, dropped=False, released=False,
                  finger_check=None, recent=[], levels=(False, False, False), geom={})


_reset_state()


def _dist_box(p, lo, hi) -> float:
    """Distance (same unit as p) from a point to an axis-aligned box; negative inside."""
    d = np.maximum(np.asarray(lo, float) - p, p - np.asarray(hi, float))
    if (d <= 0).all():
        return float(d.max())
    return float(np.linalg.norm(np.maximum(d, 0.0)))


def geometry(objects) -> dict:
    """Ruler measurements of the assembly, in the bottom shell's frame (mm, degrees)."""
    carrier, top, door = objects["carrier"], objects["top"], objects["door"]
    g = {}
    t = carrier.to_local(top.pos) * 1000
    g["top_back_mm"], g["top_side_mm"] = float(t[0]), float(abs(t[1]))
    gaps = []
    for x in (20.0, L - 20.0):
        for y in (-(HW + 8.0), HW + 8.0):
            p = carrier.to_local(top.to_world(np.array([x, y, SEAM]) / 1000)) * 1000
            gaps.append(float(p[2] - SEAM))
    g["rim_gap_max_mm"], g["rim_gap_min_mm"] = max(gaps), min(gaps)
    g["ridge_mm"] = [float(carrier.sliders.get(n, 0.0)) * 1000 for n in ("ridge_l", "ridge_r")]
    g["knob_deg"] = [math.degrees(float(carrier.hinges.get(n, 0.0))) for n in ("knob_l", "knob_r")]
    turned, offs = [], []
    for name, sg, where in HINGES:
        owner = carrier if where == "b" else top
        turned.append(sg * math.degrees(float(owner.hinges.get(name, 0.0))))
        mid = (PIN_Z[where][0] + PIN_Z[where][1]) / 2
        p = owner.to_local(door.to_world(np.array([0.0, sg * PIN_Y, mid]) / 1000)) * 1000
        offs.append(float(math.hypot(p[0] - HINGE_X, p[1] - sg * HINGE_Y)))
    g["hinge_deg"], g["pin_off_mm"] = turned, offs
    d = carrier.to_local(door.pos) * 1000
    g["door_up_mm"] = float(d[2] - DOOR_REST[2])
    return g


def _levels(g) -> tuple:
    home = (g["top_back_mm"] <= HOME_TOL and g["top_side_mm"] < SIDE_TOL and -1.0 < g["rim_gap_min_mm"]
            and g["rim_gap_max_mm"] < RIM_GAP and all(abs(r) < RIDGE_OUT for r in g["ridge_mm"]))
    locked = home and all(k >= TURNED for k in g["knob_deg"])
    shut = (locked and all(o <= PIN_TOL for o in g["pin_off_mm"]) and abs(g["door_up_mm"]) < DOOR_UP
            and all(h >= TURNED for h in g["hinge_deg"]))
    return home, locked, shut


@on_step
def _step(environment) -> None:
    """Every control step: count steps and check the grippers (score() cannot see the robot): both have let go of
    the carrier and the door."""
    from yam_arena.scene import object_states

    _STATE["step"] += 1
    try:
        st = object_states(environment, ["carrier", "door"])
        carrier, door = st["carrier"], st["door"]
        lo, hi = np.array([e[0] for e in ENVELOPE]), np.array([e[1] for e in ENVELOPE])
        dlo, dhi = np.array([-DOOR_T / 2, -DOOR_W / 2, 0.0]), np.array([DOOR_T / 2, DOOR_W / 2, DOOR_H])
        near = float("inf")
        for p in finger_points(environment):
            near = min(near, _dist_box(carrier.to_local(p) * 1000, lo, hi), _dist_box(door.to_local(p) * 1000, dlo, dhi))
        _STATE["near_mm"] = near
        _STATE["released"] = bool(near >= RELEASE * 1000)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"pet-carrier-assemble: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"], _STATE["released"] = False, False


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    carrier, top, door = objects["carrier"], objects["top"], objects["door"]
    if _STATE["base0"] is None:
        _STATE["base0"] = carrier.pos.copy()
    tilt = math.acos(max(-1.0, min(1.0, float(carrier.rotation()[2, 2]))))
    # Revision 2: moving or turning the empty carrier is not penalised (it does not stop an undamaged assembly);
    # `moved` is kept for the log only. 0 for good only if the carrier tips over or a part leaves the table.
    _STATE["moved"] |= bool(float(np.linalg.norm((carrier.pos - _STATE["base0"])[:2])) > BASE_MOVE or tilt > BASE_TILT)
    _STATE["tipped"] |= bool(tilt > TIP_OVER)
    _STATE["dropped"] |= bool(min(float(carrier.pos[2]), float(top.pos[2]), float(door.pos[2])) < TOP - 0.05)
    g = geometry(objects)
    _STATE["geom"] = g
    home, locked, shut = _levels(g)
    _STATE["levels"] = (home, locked, shut)
    recent = _STATE["recent"]
    turns = [math.radians(a) * KNOB_R / 1000 for a in g["knob_deg"]] + [math.radians(a) * LEVER_L / 1000 for a in g["hinge_deg"]]
    # whole pose: each part's position and its turn, as the travel of its far corner (rotation matrix x reach)
    turned_by = [objects[n].rotation().ravel() * REACH_MM[n] / 1000 for n in ("carrier", "top", "door")]
    # the second is counted from the release: a step with a gripper within 3 cm starts the window again
    recent.append((np.concatenate([carrier.pos, top.pos, door.pos, *turned_by, turns]), bool(_STATE["released"])))
    del recent[:-20]
    still = (len(recent) == 20 and all(free for _, free in recent)
             and all(float(np.abs(r - recent[-1][0]).max()) < MOVE_TOL for r, _ in recent))
    if _STATE["tipped"] or _STATE["dropped"]:
        result = (0.0, False)
    elif not _STATE["released"]:
        result = (0.0, False)          # revision 2: no level counts while a gripper is within 3 cm (review rule 6)
    elif shut and still:
        result = (1.0, True)
    elif shut:
        result = (0.75, False)
    elif locked:
        result = (0.5, False)
    elif home:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Top home on the bottom and both knobs locked, the door's four prongs in the four closed hinges, both grippers
    clear and everything still for the last second."""
    return _update(objects)[1]


def distance(objects) -> float:
    g = geometry(objects)
    return float(max(g["top_back_mm"], 0.0) + sum(90.0 - min(k, 90.0) for k in g["knob_deg"]) * 0.3
                 + sum(min(o, 300.0) for o in g["pin_off_mm"]) + sum(90.0 - min(h, 90.0) for h in g["hinge_deg"]) * 0.3) / 1000


def score(objects) -> float:
    """Every level only with both grippers 3 cm clear of the carrier and the door (else 0). 0.25 top slid home on the
    bottom; 0.5 both knobs locked; 0.75 door fitted (four prongs in their hinges) and all four hinges closed; 1.0 with
    everything still for 1 s. 0 for good if at any point the bottom shell tipped more than 45 degrees, or the bottom,
    the top or the door left the table."""
    return _update(objects)[0]


def demo():
    """The scripted demo needs both arms and timed pushes: run ``python demo.py task.py demo_out``."""
    return []
