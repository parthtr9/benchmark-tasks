"""On the station counter: a soft-serve freezer's cleaned door, lying on its back, with its draw valve, draw handle and
handle pin beside it. Reassemble the door the way the operator's manual says (Taylor C706, steps 14-15): the draw valve
goes down its bore (its slot facing the handle lugs) until it bottoms, the draw handle's fork goes over the bar in the
valve's slot with the handle's knuckle between the door's two lugs, and the handle pin goes through lug, knuckle and
lug. Left with the handle raised, the valve is closed; pulling the handle lifts the valve through the fork and bar.

Parts are at catalogue size: door X87683 (177 x 136 x 75 mm, 0.87 kg), draw handle X56246 (174 x 70 x 28 mm, 0.31 kg),
draw valve X56072 (160 mm long, 0.09 kg) and pin 055819. The valve's diameter and the bore follow from its O-rings
(014402, 7/8 in OD x 0.103 in). The pin's diameter and length, the lug spacing and the valve's slot and bar are
scaled from the manual's drawings: measure the bought parts.

Frames. Geometry is written in the DOOR frame (mm): origin on the valve bore's axis in the door's top face, dz up the
door (toward its top), dx from the door's front (negative, the operator side) to its back, dy across. The door lies on
its back on the counter with its top toward the robot, so world x = XD - dz, world y = -dy, world z = ZD - dx.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = ("Reassemble the freezer door: draw valve in its bore, draw handle engaged on it and pinned, "
               "handle left raised.")

STEEL = (0.74, 0.75, 0.77)
DOOR_C = (0.86, 0.87, 0.88)
DARK = (0.22, 0.23, 0.25)
BLACK = (0.08, 0.08, 0.09)
MAT_C = (0.15, 0.16, 0.17)

# -- door-frame to world ---------------------------------------------------------------------------------------------
XD, YD = float(os.environ.get("SOFTSERVE_XD", "0.24")), 0.0                 # the door's top face (dz = 0) and the bore axis, world
DOOR_BACK = 50.0                   # the door's back face, dx (it rests on the counter)
MAT_T = 0.004
ZD = TOP + MAT_T + DOOR_BACK / 1000
M = np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])   # door axes -> world axes (columns)


def _quat(R):
    """(w, x, y, z) of a rotation matrix (Shepperd's method: robust near half turns, which M is)."""
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


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def dbox(x0, x1, y0, y1, z0, z1, color, R=None, origin=(0.0, 0.0, 0.0)):
    """A box given in door-frame mm (axis-aligned there, or turned by R about its own centre), as a partkit box in the
    part frame of a part whose origin is the door-frame point ``origin`` (mm)."""
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]) - np.asarray(origin, float)
    size = (abs(x1 - x0) / 1000, abs(y1 - y0) / 1000, abs(z1 - z0) / 1000)
    Rw = M @ (np.eye(3) if R is None else R)
    return (tuple(float(v) for v in (M @ c) / 1000), size, color, tuple(float(v) for v in _quat(Rw)))


def prism(cx, cy, z0, z1, r, color, origin=(0.0, 0.0, 0.0), axis="z", sides=8):
    """A regular prism of ``sides`` faces inscribed in a round part of radius ``r`` along door ``axis`` (its corners on
    the circle, so the model is never fatter than the real part; 16 sides are under 2% thin at the flats): the union of
    sides/2 bars 2r cos(180/sides deg) long and 2r sin(180/sides deg) wide, turned in equal steps (PhysX's cylinders
    miss edge contacts)."""
    out = []
    half = math.pi / sides
    r, a = r * math.cos(half), r * math.sin(half)
    for k in range(sides // 2):
        ang = k * 2 * half
        if axis == "z":
            out.append(dbox(cx - r, cx + r, cy - a, cy + a, z0, z1, color, R=_rot_z(ang) if k else None, origin=origin))
        elif axis == "y":   # cx, cy are dx, dz; z0..z1 along dy
            out.append(dbox(cx - r, cx + r, z0, z1, cy - a, cy + a, color, R=_rot_y(ang) if k else None, origin=origin))
    return out


def ring_z(cx, cy, z0, z1, r_in, t, color, n=12, origin=(0.0, 0.0, 0.0)):
    """A bore along dz: ``n`` boxes whose inner faces lie at r_in."""
    out = []
    w = 2 * (r_in + t) * math.tan(math.pi / n) * 1.04
    for k in range(n):
        a = 2 * math.pi * k / n
        rm = r_in + t / 2
        px, py = cx + rm * math.cos(a), cy + rm * math.sin(a)
        R = _rot_z(a)
        out.append(dbox(px - t / 2, px + t / 2, py - w / 2, py + w / 2, z0, z1, color, R=None, origin=origin))
        # turn the box about its own centre: rebuild with rotation
        c = np.array([px, py, (z0 + z1) / 2]) - np.asarray(origin, float)
        out[-1] = (tuple(float(v) for v in (M @ c) / 1000), (t / 1000, w / 1000, (z1 - z0) / 1000), color,
                   tuple(float(v) for v in _quat(M @ R)))
    return out


def cone_z(cx, cy, z0, z1, r0, r1, color, n=16, t=1.6, origin=(0.0, 0.0, 0.0)):
    """A 45-degree lead-in at a bore's mouth: ``n`` boxes on a cone from radius r0 at dz z0 to r1 at dz z1."""
    out = []
    gen = math.hypot(z1 - z0, r1 - r0)
    ang = math.atan2(r1 - r0, z1 - z0)                    # the cone's half-angle from the axis
    w = 2 * (r1 + t) * math.tan(math.pi / n) * 1.08
    for k in range(n):
        a = 2 * math.pi * k / n
        rm = (r0 + r1) / 2 + t / 2 * math.cos(ang)
        zc = (z0 + z1) / 2 - t / 2 * math.sin(ang)
        c = np.array([cx + rm * math.cos(a), cy + rm * math.sin(a), zc]) - np.asarray(origin, float)
        R = _rot_z(a) @ _rot_y(ang)                       # box x: through the wall; box z: along the cone
        out.append((tuple(float(v) for v in (M @ c) / 1000), (t / 1000, w / 1000, gen / 1000), color,
                    tuple(float(v) for v in _quat(M @ R))))
    return out


def plate_with_hole_y(x0, x1, y0, y1, z0, z1, hx, hz, h, color, origin=(0.0, 0.0, 0.0)):
    """A plate across dy (dy from y0 to y1) with a square hole of side ``h`` along dy centred at (dx hx, dz hz)."""
    return [dbox(x0, hx - h / 2, y0, y1, z0, z1, color, origin=origin),
            dbox(hx + h / 2, x1, y0, y1, z0, z1, color, origin=origin),
            dbox(hx - h / 2, hx + h / 2, y0, y1, z0, hz - h / 2, color, origin=origin),
            dbox(hx - h / 2, hx + h / 2, y0, y1, hz + h / 2, z1, color, origin=origin)]


# -- dimensions (mm) -------------------------------------------------------------------------------------------------
BORE_R = 10.9                      # 0.86 in bore: the 7/8 in O-rings seal in it
CHAMFER = 10.0                     # a 10 mm long, 14 degree lead-in at the bore's mouth, as O-ring bores have
MOUTH_R = 13.4                     # the lead-in's radius at the mouth
VALVE_R = 9.5                      # 0.75 in valve body (O-ring ID 0.67 in plus the grooves; sibling X55820 is 0.75 in)
VALVE_LEN = 160.0                  # X56072: 6.3 in
SEAT = -110.0                      # the valve bottoms on a ledge 110 mm down the bore
VALVE_TOP = SEAT + VALVE_LEN       # +50: the valve's head stands 50 mm out of the bore when seated
HEAD0 = 28.0                       # the slotted head: dz 28..50 above the door's top face
SLOT_W = 10.0                      # the slot across the head, open toward the front (the lugs)
BAR_R, BAR_DX, BAR_DZ = 2.5, -3.0, 40.0
SLOT_BACK = 1.5                    # the slot's back wall: the fork's end (dx +1.0) stops 0.5 mm short of it, which is
#                                    where the knuckle's hole lines up with the lugs' holes
PIVOT_DX, PIVOT_DZ = -35.0, 40.0   # the handle pin's axis (through both lugs), door frame
LUG_GAP = 28.0                     # inner faces of the lugs, +-14 (estimate: the drawings give no scale; 2 mm side play)
LUG_T = 5.0
HOLE = 8.0                         # lug and knuckle holes (square, 8 mm): a 1/4 in pin with about 0.5 mm play a side
PIN_R = 3.175                      # 1/4 in pin (estimate: measure)
PIN_LEN = 75.0                     # straight part (estimate: measure)
PIN_ARM = 25.0                     # the bent end
KNUCKLE_W = 24.0                   # across dy: 2 mm side play between the lugs (estimate)
FORK_T = 8.0                       # the fork plate's thickness across dy: 1 mm side play in the 10 mm slot
LEVER_LEN = 150.0                  # knuckle to grip end (the welded handle is 174 mm over all)

# -- the door: body with its bore, the bottom ledge, two lugs ----------------------------------------------------------
DOOR_BOXES = [
    dbox(-30.0, -BORE_R - 2, -88.0, 88.0, -136.0, 0.0, DOOR_C),           # front of the spout block
    dbox(BORE_R + 2, DOOR_BACK, -88.0, 88.0, -136.0, 0.0, DOOR_C),        # back plate
    dbox(-BORE_R - 2, BORE_R + 2, BORE_R + 2, 88.0, -136.0, 0.0, DOOR_C),  # sides of the bore
    dbox(-BORE_R - 2, BORE_R + 2, -88.0, -BORE_R - 2, -136.0, 0.0, DOOR_C),
    dbox(-BORE_R - 2, BORE_R + 2, -BORE_R - 2, BORE_R + 2, -136.0, SEAT - 4.0, DOOR_C),   # below the ledge
] + ring_z(0.0, 0.0, SEAT - 4.0, -CHAMFER, BORE_R, 1.6, DOOR_C) + cone_z(0.0, 0.0, -CHAMFER, 0.0, BORE_R, MOUTH_R,
                                                                        DOOR_C) + [
    dbox(-BORE_R, BORE_R, -BORE_R, BORE_R, SEAT - 4.0, SEAT, DARK),      # the ledge the valve bottoms on
]
for side in (1.0, -1.0):                    # square-edged lugs (the C706 manual's Fig. 6-10 shows no chamfer)
    y0, y1 = (LUG_GAP / 2, LUG_GAP / 2 + LUG_T) if side > 0 else (-LUG_GAP / 2 - LUG_T, -LUG_GAP / 2)
    DOOR_BOXES += plate_with_hole_y(-48.0, -22.0, y0, y1, 0.0, 56.0, PIVOT_DX, PIVOT_DZ, HOLE, DOOR_C)
DOOR_ORIGIN_W = (XD, YD, ZD)
DOOR = part("door", pos=DOOR_ORIGIN_W, mass=0.87, boxes=DOOR_BOXES, color=DOOR_C, friction=0.9)

# -- the draw valve: body, slotted head with the bar across the slot (part origin: on its axis at the door's top face
#    when seated) ---------------------------------------------------------------------------------------------------
_vo = (0.0, 0.0, 0.0)
ORING_DZ = (-100.0, -62.0, -24.0)
ORING_R = float(os.environ.get("SOFTSERVE_ORING_R", "9.6"))                      # band radius (corners of the 16-sided band): 1.3-1.5 mm play in the 10.9 mm bore;
#                                    the rings' squeeze that centres the valve is modelled as springs (ORING_K)
ORING_W = 2.6


def valve_boxes(oring_r=None):
    """The draw valve's shapes (valve frame = door frame when seated). The three O-rings (014402, 0.103 in section) in
    their grooves are rigid 16-sided bands just inside the bore that keep the valve coaxial in it, as the real rings
    do; each has a 0.6 mm lead-in step on its lower side. Their sliding drag is modelled separately (DRAG_N)."""
    oring_r = ORING_R if oring_r is None else oring_r
    out = prism(0.0, 0.0, SEAT + 3.0, HEAD0, VALVE_R, STEEL, origin=_vo, sides=16)
    out += prism(0.0, 0.0, SEAT, SEAT + 3.0, VALVE_R - 2.0, STEEL, origin=_vo, sides=16)      # the chamfered bottom end
    for z in ORING_DZ:
        out += prism(0.0, 0.0, z - ORING_W / 2 + 0.6, z + ORING_W / 2, oring_r, DARK, origin=_vo, sides=16)
        out += prism(0.0, 0.0, z - ORING_W / 2, z - ORING_W / 2 + 0.6, oring_r - 0.6, DARK, origin=_vo, sides=16)
    out += [
        dbox(-VALVE_R, VALVE_R, SLOT_W / 2, VALVE_R, HEAD0, VALVE_TOP, STEEL, origin=_vo),    # the head's two cheeks
        dbox(-VALVE_R, VALVE_R, -VALVE_R, -SLOT_W / 2, HEAD0, VALVE_TOP, STEEL, origin=_vo),
        dbox(SLOT_BACK, VALVE_R, -SLOT_W / 2, SLOT_W / 2, HEAD0, VALVE_TOP, STEEL, origin=_vo),   # the slot's back wall
        dbox(BAR_DX - BAR_R, BAR_DX + BAR_R, -SLOT_W / 2, SLOT_W / 2, BAR_DZ - BAR_R, BAR_DZ + BAR_R, DARK,
             origin=_vo),                                                                       # the bar
    ]
    return out


VALVE_BOXES = valve_boxes()
# Start: in its bore, slot toward the lugs, 8 mm short of home with all three O-rings past the mouth's chamfer (as an
# operator leaves it after starting it by hand); the robot pushes it home against the rings' drag
VALVE_START_OUT = 0.008
VALVE_START = tuple(float(v) for v in (np.array([XD, YD, ZD]) + (M @ np.array([0.0, 0.0, 1.0])) * VALVE_START_OUT))
VALVE = part("valve", pos=VALVE_START, mass=0.09, boxes=VALVE_BOXES, color=STEEL, friction=0.3)   # lubricated
# Negative-variant check only (demo.py sets it for DEMO_VARIANT=reversed): the valve starts turned half a turn about
# its own axis, its slot away from the lugs.
if os.environ.get("SOFTSERVE_VALVE_REVERSED") == "1":
    VALVE.meta["quat_xyzw"] = (1.0, 0.0, 0.0, 0.0)
# Debug start only (never in a submitted run): the valve already seated, to test the handle and pin steps alone.
if os.environ.get("SOFTSERVE_VALVE_SEATED") == "1":
    VALVE = part("valve", pos=(XD, YD, ZD), mass=0.09, boxes=VALVE_BOXES, color=STEEL, friction=0.3)

# -- the draw handle: knuckle (with its pin hole), fork plate reaching the bar, lever (part origin: the pin axis) ------
_ho = (PIVOT_DX, 0.0, PIVOT_DZ)
NOTCH = 6.5                        # the fork's notch round the 5 mm bar: 0.75 mm side play (estimate: the drawings give
#                                    no scale; a cast fork that rides loose on its bar), square-ended like the drawing
NOTCH_MOUTH = NOTCH                # no lead-in: the notch is straight to the fork's end
LEAD = 3.0                         # the lead-in's length along the fork
HANDLE_BOXES = plate_with_hole_y(PIVOT_DX - 8, PIVOT_DX + 8, -KNUCKLE_W / 2, KNUCKLE_W / 2, PIVOT_DZ - 8,
                                 PIVOT_DZ + 8, PIVOT_DX, PIVOT_DZ, HOLE, DARK, origin=_ho)
FORK_END = BAR_DX + BAR_R + 1.5   # +1.0: the fork's end, 0.5 mm short of the slot's back wall when fitted


def fork_boxes(notch=None, mouth=None, lead=None):
    """The fork's two tines around the bar. With ``mouth`` wider than ``notch`` the tines' last ``lead`` mm slope out
    to it; the sloped boxes are clipped so nothing reaches past FORK_END."""
    notch = NOTCH if notch is None else notch
    mouth = NOTCH_MOUTH if mouth is None else mouth
    lead = LEAD if lead is None else lead
    straight_end = FORK_END - (lead if mouth > notch else 0.0)
    out = [dbox(PIVOT_DX + 8, straight_end, -FORK_T / 2, FORK_T / 2, BAR_DZ + notch / 2, BAR_DZ + notch / 2 + 3.5,
                DARK, origin=_ho),                                           # upper tine (toward the handle's top)
           dbox(PIVOT_DX + 8, straight_end, -FORK_T / 2, FORK_T / 2, BAR_DZ - notch / 2 - 3.5, BAR_DZ - notch / 2,
                DARK, origin=_ho)]                                           # lower tine
    if mouth > notch:
        # each tine's last ``lead`` mm is a slab whose inner face runs from the notch's edge at straight_end to the
        # mouth's edge at FORK_END (door dx, dz); the slab lies outside that face, away from the bar
        t = 3.5
        for sgn in (1.0, -1.0):
            p0 = np.array([straight_end, BAR_DZ + sgn * notch / 2])
            p1 = np.array([FORK_END, BAR_DZ + sgn * mouth / 2])
            u = (p1 - p0) / np.linalg.norm(p1 - p0)
            n = sgn * np.array([-u[1], u[0]]) if sgn > 0 else np.array([u[1], -u[0]])
            c = (p0 + p1) / 2 + n * t / 2
            length = float(np.linalg.norm(p1 - p0))
            ang = math.atan2(u[1], u[0])                                     # the face's slope in the dx-dz plane
            out.append(dbox(c[0] - length / 2, c[0] + length / 2, -FORK_T / 2, FORK_T / 2, c[1] - t / 2, c[1] + t / 2,
                            DARK, R=_rot_y(-ang), origin=_ho))
    return out


HANDLE_BOXES += fork_boxes()
HANDLE_BOXES += [
    dbox(PIVOT_DX - 6, PIVOT_DX + 6, -8.0, 8.0, PIVOT_DZ + 8, PIVOT_DZ + LEVER_LEN, DARK, origin=_ho),   # lever
    dbox(PIVOT_DX - 10, PIVOT_DX + 10, -11.0, 11.0, PIVOT_DZ + LEVER_LEN - 55, PIVOT_DZ + LEVER_LEN, BLACK,
         origin=_ho),                                                        # the black grip
]


def door_to_world(p_mm):
    return np.array(DOOR_ORIGIN_W) + M @ (np.asarray(p_mm, float) / 1000)


PIVOT_W = door_to_world((PIVOT_DX, 0.0, PIVOT_DZ))
# Start: the cleaned handle lies on the door's front face in its fitted orientation, 5 cm to the robot's left of the
# valve, its grip on the face and its knuckle and fork hanging past the door's bottom edge
FACE_Z = ZD + 0.030                                   # the door's front face (dx -30) faces up
HANDLE_START = (float(XD + 0.136 + 0.020), 0.04, float(FACE_Z + 0.0105))
HANDLE = part("handle", pos=HANDLE_START, mass=0.31, boxes=HANDLE_BOXES, color=DARK, friction=0.8)

# -- the pin: an L-shaped stainless rod, its straight part along door dy (part origin: the middle of the straight part)
_po = (PIVOT_DX, 0.0, PIVOT_DZ)
PIN_BOXES = prism(PIVOT_DX, PIVOT_DZ, -PIN_LEN / 2 + 6.0, PIN_LEN / 2, PIN_R, STEEL, origin=_po, axis="y")
PIN_BOXES += prism(PIVOT_DX, PIVOT_DZ, -PIN_LEN / 2 + 3.0, -PIN_LEN / 2 + 6.0, PIN_R - 1.0, STEEL, origin=_po, axis="y")  # tapered
PIN_BOXES += prism(PIVOT_DX, PIVOT_DZ, -PIN_LEN / 2, -PIN_LEN / 2 + 3.0, PIN_R - 2.0, STEEL, origin=_po, axis="y")        # tip
PIN_BOXES += [dbox(PIVOT_DX - PIN_R, PIVOT_DX + PIN_R, PIN_LEN / 2 - 2 * PIN_R, PIN_LEN / 2, PIVOT_DZ,
                   PIVOT_DZ + PIN_ARM, STEEL, origin=_po)]                   # the bent end, toward the door's top
# The tip is at door dy -PIN_LEN/2 (world +y): it goes in from the robot's right, moving +y.
PIN_START = (float(PIVOT_W[0]) - 0.02, -0.20, float(TOP + MAT_T + PIN_R / 1000))
PIN = part("pin", pos=PIN_START, mass=0.04, boxes=PIN_BOXES, color=STEEL, friction=0.5)

MAT = Box("mat", pos=(0.25, 0.0, TOP + MAT_T / 2), size=(0.30, 0.56, MAT_T), color=MAT_C)
OBJECTS = [MAT, DOOR, VALVE, HANDLE, PIN]

# -- scoring -----------------------------------------------------------------------------------------------------------
SEAT_TOL = 0.002          # the valve's bottom within 2 mm of the ledge
KNUCKLE_TOL = 0.003        # knuckle hole within 3 mm of the lugs' axis: the 8 mm holes round the 6.35 mm pin let a
#                            pinned knuckle sit up to 2.3 mm off (0.8 mm a side in each of two holes, diagonally)
AXIS_TOL = 0.0025          # valve axis within 2.5 mm of the bore's: the rigid O-ring bands have 1.0-1.2 mm play
SLOT_TOL = math.radians(20.0)
PIN_AXIS_TOL = 0.0015     # pin axis within 1.5 mm of the lug holes' axis
RAISED_TOL = math.radians(12.0)
# The three lubricated 7/8 x 0.103 in O-rings drag on the bore. Running friction is estimated at about 4 N a ring
# (lubricated nitrile, roughly 15 % squeeze; check it with a spring scale on the real valve), 12 N in all. It is applied
# as Coulomb drag along the bore while the valve is in it, on the valve and equal and opposite on the door, re-applied
# before every physics step, so the valve needs a firm push and the door needs holding. It never earns credit by itself.
DRAG_N = 12.0
DOOR_MOVE = 0.010
RELEASE = 0.03
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), door0=None, moved=False, released=False,
                  finger_check=None, recent=[], levels=(False, False, False, False), stick_at=None,
                  free={"valve": False, "handle": False, "pin": False},
                  slip_dir=0.0)


_reset_state()


def _local(door: PartState, p_world) -> np.ndarray:
    """A world point in the door frame (mm)."""
    return M.T @ (door.to_local(np.asarray(p_world, float)) * 1000)


def _dir_local(door: PartState, d_world) -> np.ndarray:
    return M.T @ (door.rotation().T @ np.asarray(d_world, float))


def geometry(objects) -> dict:
    """Valve, handle and pin relative to the door, in door-frame mm and radians."""
    door, valve, handle, pin = objects["door"], objects["valve"], objects["handle"], objects["pin"]
    g = {}
    vo = _local(door, valve.pos)                                  # valve origin: on its axis at dz 0 when seated
    v_ax = _dir_local(door, valve.rotation() @ (M @ np.array([0.0, 0.0, 1.0])))   # valve's dz axis
    v_front = _dir_local(door, valve.rotation() @ (M @ np.array([-1.0, 0.0, 0.0])))  # its slot's opening
    g["valve_depth_mm"] = float(-vo[2])                           # 0 when seated, + if pushed past (cannot), - short
    g["valve_axis_mm"] = float(math.hypot(vo[0], vo[1]))
    g["valve_tilt"] = float(math.acos(max(-1.0, min(1.0, v_ax[2]))))
    g["slot_angle"] = float(math.atan2(v_front[1], -v_front[0]))  # 0 when the slot faces the lugs (front, -dx)
    # the bar's centre in the handle's frame: inside the notch?
    bar_w = valve.to_world(M @ (np.array([BAR_DX, 0.0, BAR_DZ]) / 1000))
    bar_h = M.T @ (handle.to_local(bar_w) * 1000) + np.array(_ho)         # handle-frame point in door coordinates
    g["bar_in_fork"] = bool(PIVOT_DX + 8 < bar_h[0] < FORK_END and abs(bar_h[2] - BAR_DZ) < NOTCH_MOUTH / 2
                            and abs(bar_h[1]) < FORK_T / 2 + SLOT_W)
    # the knuckle hole's axis and the pin's axis against the lug holes' axis
    kn = _local(door, handle.pos)
    g["knuckle_mm"] = float(math.hypot(kn[0] - PIVOT_DX, kn[2] - PIVOT_DZ))
    lever = _dir_local(door, handle.rotation() @ (M @ np.array([0.0, 0.0, 1.0])))
    g["lever_from_raised"] = float(math.atan2(lever[0], lever[2]))  # 0 raised; + toward the back (blocked), - pulled
    po = _local(door, pin.pos)
    p_ax = _dir_local(door, pin.rotation() @ (M @ np.array([0.0, 1.0, 0.0])))
    g["pin_axis_mm"] = float(math.hypot(po[0] - PIVOT_DX, po[2] - PIVOT_DZ))
    g["pin_par"] = float(abs(p_ax[1]))
    half = PIN_LEN / 2
    lo, hi = po[1] - half * abs(p_ax[1]), po[1] + half * abs(p_ax[1])
    g["pin_span"] = (float(lo), float(hi))
    g["pin_through_lugs"] = bool(lo < -(LUG_GAP / 2 + LUG_T) and hi > LUG_GAP / 2 + LUG_T)
    return g


def _levels(g) -> tuple:
    seated = (abs(g["valve_depth_mm"]) < SEAT_TOL * 1000 and g["valve_axis_mm"] < AXIS_TOL * 1000
              and g["valve_tilt"] < math.radians(3.0) and abs(g["slot_angle"]) < SLOT_TOL)
    engaged = seated and g["bar_in_fork"] and g["knuckle_mm"] < KNUCKLE_TOL * 1000
    pinned = (engaged and g["pin_axis_mm"] < PIN_AXIS_TOL * 1000 and g["pin_par"] > 0.995 and g["pin_through_lugs"])
    raised = pinned and abs(g["lever_from_raised"]) < RAISED_TOL
    return seated, engaged, pinned, raised


ORING_K = float(os.environ.get("SOFTSERVE_ORING_K", "2000.0"))   # each O-ring's radial stiffness (N/m), estimate
ORING_C = 8.0                      # and its radial damping (N s/m)
STICK_K, STICK_C = 4000.0, 20.0    # the O-rings' stiction along the bore: spring to an anchor (N/m) and damping (N s/m)


def stick_slip(dz: float, v: float, state: dict, mass: float, dt: float) -> float:
    """The O-rings' Coulomb friction along the bore (N, on the valve) at its axial position ``dz`` (m) and speed ``v``
    (m/s), with stiction, for a physics step ``dt``. Stuck: a stiff spring to the place it stuck
    (``state["stick_at"]``) holds it up to DRAG_N. Past that it slides: the anchor is dropped, so no stretch is stored,
    and the full drag opposes the slide until the slide is slow enough to be stopped within one step; then it is
    stopped and sticks where it stopped. (Batch 3b793a45: an anchor that trailed the sliding valve by DRAG_N / STICK_K
    threw it 6 mm back out when the hand let go.)"""
    anchor = state.get("stick_at")
    if anchor is None:
        stop = mass * v / dt                                # the force that stops it within this step
        if abs(stop) > DRAG_N:                              # sliding: the full drag against the slide
            state["slip_dir"] = math.copysign(1.0, v)
            return -math.copysign(DRAG_N, v)
        state["stick_at"] = dz + v * dt                     # slow enough: stopped, it sticks where it stops
        return -stop
    f = -STICK_K * (dz - anchor) - STICK_C * v
    if abs(f) > DRAG_N:                                     # breaks away: slides, with no stored stretch
        state["stick_at"], state["slip_dir"] = None, -math.copysign(1.0, f)
        return math.copysign(DRAG_N, f)
    return f


def _oring_wrench(environment):
    """The O-rings' forces on the valve (world): Coulomb drag along the bore while the valve's bottom is past the
    bore's mouth, capped at the force that would stop the relative sliding within one physics step (it only ever
    opposes the motion), and, for each ring that is inside the bore past the mouth's chamfer, the ring's squeeze as a
    radial spring toward the bore's axis (ORING_K, ORING_C). The spring acts only across the bore: it centres the
    valve and never draws it in. Returns the valve's force and torque about its centre of mass, and the door's."""
    from partkit import _np

    base = environment.unwrapped
    door_a, valve_a = base.scene["door"], base.scene["valve"]
    org = _np(base.scene.env_origins[0])
    dp, vp = _np(door_a.data.root_link_pose_w)[0], _np(valve_a.data.root_link_pose_w)[0]
    door = PartState(pos=dp[:3] - org, quat_xyzw=dp[3:7], lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.zeros(3),
                     high=np.zeros(3))
    valve = PartState(pos=vp[:3] - org, quat_xyzw=vp[3:7], lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.zeros(3),
                      high=np.zeros(3))
    zero = np.zeros(3)
    vo = _local(door, valve.pos)
    axis = door.rotation() @ (M @ np.array([0.0, 0.0, 1.0]))
    v_lin = _np(valve_a.data.root_lin_vel_w)[0] - _np(door_a.data.root_lin_vel_w)[0]
    w_ang = _np(valve_a.data.root_ang_vel_w)[0]
    com_v = (_np(valve_a.data.root_com_pos_w)[0] if hasattr(valve_a.data, "root_com_pos_w") else vp[:3]) - org
    com_d = (_np(door_a.data.root_com_pos_w)[0] if hasattr(door_a.data, "root_com_pos_w") else dp[:3]) - org
    f_v, t_v, f_d, t_d = zero.copy(), zero.copy(), zero.copy(), zero.copy()
    if vo[2] < VALVE_LEN - VALVE_TOP and math.hypot(vo[0], vo[1]) < 3.0:
        v = float(v_lin @ axis)
        dt = float(base.physics_dt) if hasattr(base, "physics_dt") else 1.0 / 240.0
        # Coulomb friction with stiction (a cap on the sliding force alone let any small push creep the valve out:
        # batch bc42e2f8)
        f_ax = stick_slip(float(vo[2]) / 1000, v, _STATE, VALVE.mass, dt)
        f = f_ax * axis
        f_v, f_d = f_v + f, f_d - f
        # the same friction resists the valve turning in its bore: the drag acting round the rings' circumference
        # (without it, the valve's slightly off-axis mass turned it in its bore by itself)
        w = float((w_ang - _np(door_a.data.root_ang_vel_w)[0]) @ axis)
        inertia = 0.5 * VALVE.mass * (VALVE_R / 1000) ** 2
        tq = -math.copysign(min(DRAG_N * ORING_R / 1000, inertia * abs(w) / dt), w) * axis
        t_v, t_d = t_v + tq, t_d - tq
    else:                                       # out of the bore: no friction, and it sticks afresh on coming back
        _STATE["stick_at"], _STATE["slip_dir"] = None, 0.0
    v_axis_w =valve.rotation() @ (M @ np.array([0.0, 0.0, 1.0]))
    door_o = np.asarray(door.pos)
    aligned = abs(float(v_axis_w @ axis)) > math.cos(math.radians(15.0))
    for dz in ORING_DZ:
        p = np.asarray(valve.pos) + v_axis_w * (dz / 1000)              # the ring's centre (world)
        rel = p - door_o
        e = rel - axis * float(rel @ axis)                               # its offset across the bore
        # only a ring that is really in the bore: past the mouth's chamfer and within 3 mm of its axis
        if not aligned or float(rel @ axis) * 1000 > -CHAMFER or np.linalg.norm(e) > 0.003:
            continue
        vel = v_lin + np.cross(w_ang, p - com_v)
        vel_across = vel - axis * float(vel @ axis)
        f = -ORING_K * e - ORING_C * vel_across
        f_v, t_v = f_v + f, t_v + np.cross(p - com_v, f)
        f_d, t_d = f_d - f, t_d + np.cross(p - com_d, -f)
    return f_v, t_v, f_d, t_d


def _install_drag(environment) -> None:
    """Re-apply the O-rings' forces before every physics step (the scene writes external forces each step)."""
    import torch
    from partkit import _t

    scene = environment.unwrapped.scene
    if getattr(scene, "_softserve_drag", False):
        return
    orig = scene.write_data_to_sim

    def write_data_to_sim(*args, **kwargs):
        try:
            f_v, t_v, f_d, t_d = _oring_wrench(environment)
            _STATE["drag_n"] = float(np.linalg.norm(f_v))
            for name, f, t in (("valve", f_v, t_v), ("door", f_d, t_d)):
                asset = scene[name]
                dev = _t(asset.data.root_pos_w).device
                force = torch.tensor([[[*f]]], dtype=torch.float32, device=dev)
                torque = torch.tensor([[[*t]]], dtype=torch.float32, device=dev)
                asset.set_external_force_and_torque(force, torque, is_global=True)
        except Exception as exc:  # noqa: BLE001
            if _STATE.get("drag_err") is None:
                _STATE["drag_err"] = repr(exc)
                print(f"softserve-door-assemble: O-ring forces unavailable: {exc!r}", flush=True)
        return orig(*args, **kwargs)

    scene.write_data_to_sim = write_data_to_sim
    scene._softserve_drag = True


@on_step
def _step(environment) -> None:
    """Every control step: count steps, apply the O-rings' drag and check the grippers (score() cannot see the
    robot)."""
    from yam_arena.scene import object_states

    _STATE["step"] += 1
    try:
        st = object_states(environment, ["door", "valve", "handle", "pin"])
        _install_drag(environment)
        pts = finger_points(environment)
        near = min(min(np.linalg.norm(p - o.pos) for o in (st["valve"], st["handle"], st["pin"])) for p in pts)
        hpts = [st["handle"].to_world(M @ (np.array([0.0, 0.0, z]) / 1000)) for z in (0.0, 50.0, 100.0, 150.0)]
        near_h = min(np.linalg.norm(p - q) for p in pts for q in hpts)
        _STATE["released"] = bool(near >= RELEASE + 0.02 and near_h >= RELEASE)
        # a level counts only once the grippers have let go of the part it credits (review rule 6)
        _STATE["free"] = {n: all(part_gap(p, st[n], spec) >= RELEASE for p in pts)
                          for n, spec in (("valve", VALVE), ("handle", HANDLE), ("pin", PIN))}
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"softserve-door-assemble: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"], _STATE["released"] = False, False
        _STATE["free"] = {"valve": False, "handle": False, "pin": False}


_BOXES: dict = {}


def part_gap(p, st, spec) -> float:
    """A point's distance (m) from a part's surface: the nearest of its boxes."""
    from partkit import _box, _quat_matrix   # noqa: PLC0415

    if spec.name not in _BOXES:
        bx = [_box(b) for b in spec.boxes]
        _BOXES[spec.name] = (np.array([b[0] for b in bx], float), np.array([b[1] for b in bx], float) / 2,
                             np.array([_quat_matrix(b[3]).T for b in bx]))
    c, h, rt = _BOXES[spec.name]
    local = np.asarray(st.to_local(np.asarray(p, float)), float)
    d = np.abs(np.einsum("nij,nj->ni", rt, local - c)) - h
    return float(np.linalg.norm(np.maximum(d, 0.0), axis=1).min())


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    door = objects["door"]
    if _STATE["door0"] is None:
        _STATE["door0"] = door.pos.copy()
    _STATE["moved"] |= float(np.linalg.norm(door.pos - _STATE["door0"])) > DOOR_MOVE      # permanent
    g = geometry(objects)
    seated, engaged, pinned, raised = _levels(g)
    _STATE["levels"] = (seated, engaged, pinned, raised)
    free = _STATE["free"]
    held_ok = (seated and free["valve"], engaged and free["valve"] and free["handle"],
               pinned and free["valve"] and free["handle"] and free["pin"])
    recent = _STATE["recent"]
    recent.append(np.concatenate([objects[n].pos for n in ("door", "valve", "handle", "pin")]))
    del recent[:-20]
    still = len(recent) == 20 and all(float(np.abs(r - recent[-1]).max()) < 0.001 for r in recent)
    if _STATE["moved"]:
        result = (0.0, False)
    elif raised and _STATE["released"] and still:
        result = (1.0, True)
    elif held_ok[2]:
        result = (0.75, False)
    elif held_ok[1]:
        result = (0.5, False)
    elif held_ok[0]:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Valve seated with its slot to the lugs, the handle's fork over its bar, the pin through lug, knuckle and lug,
    the handle raised, both grippers clear and everything still for the last second."""
    return _update(objects)[1]


def distance(objects) -> float:
    g = geometry(objects)
    return float(abs(g["valve_depth_mm"]) + g["valve_axis_mm"] + g["knuckle_mm"] + g["pin_axis_mm"]) / 1000


def score(objects) -> float:
    """End state, each level including the ones before it: 0.25 the valve seated in its bore (bottomed within 2 mm,
    axis within 1.5 mm) with its slot within 20 degrees of facing the lugs; 0.5 the handle's knuckle between the lugs
    (within 2.5 mm of their axis) with the valve's bar inside its fork; 0.75 the pin through both lugs and the knuckle
    (within 1.5 mm of the lug axis); 1.0 the handle within 12 degrees of raised, both grippers 3 cm clear and nothing
    moving more than 1 mm for 1 second. Each level counts only while no gripper finger is within 3 cm of a part it
    credits (the valve, then the handle, then the pin). 0 for good if the door was moved over 10 mm."""
    return _update(objects)[0]
