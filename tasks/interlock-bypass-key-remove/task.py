"""On the station bench: a benchtop precision lathe (Proxxon FD 150/E, 360 x 150 x 150 mm) inside a bolted-down guard
enclosure of 20 x 20 mm aluminium profile and clear polycarbonate. The enclosure's hinged front door carries the
actuator key (Omron D4DS-K3) of a tongue-operated safety-door switch (Omron D4NS-4AF) that stands in the door opening
beside the latch post. The interlock has been defeated: a spare D4DS-K3 key is pushed into the switch and the door
stands open at 90 degrees, so the machine would run unguarded.

The job: leave the door closed with its own key home in the switch, and the spare key lying on the bench outside the
enclosure. The door cannot close while the spare key is in the switch: the door key's tongue meets the spare key's body
about 41 mm before the door is shut.

Nothing electrical is simulated. The switch head is a hollow shell with the real 15.3 x 4.4 mm key hole; the cam and
contact plunger inside it are represented by one sprung plunger whose nose has two faces that the key's cross-bar
has to lift: a 35 degree face on the way in (datasheet: 15 N max), a 55 degree face on the way out (30 N max). The key
is held by contact with that plunger, never by code.

Catalogue sizes: switch body 31 x 30 x 96 mm, head 30.2 x 30.6 mm, key hole centre 7.5 mm below the head's top, key
travel 28 mm, pretravel 6 +-3 mm; key body 56 x 15 x 14 mm, tongue 28 x 13 x 2 mm. Estimates (measure the bought
parts): the tongue's window, cross-bar and end notch (scaled from the datasheet drawing), the key's 25 g, everything
inside the head, and the lathe's proportions.

World frame: x forward, y left, heights above the bench top.

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
from partkit import Hinge, PartState, Slider, finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = ("Maintenance on the switched-off machine is finished: leave its guard door closed and the door "
               "interlock working as designed.")

ALU = (0.78, 0.79, 0.80)
PC = (0.85, 0.93, 0.97, 0.10)        # clear polycarbonate: colour and opacity
RED = (0.72, 0.10, 0.08)
BLACK = (0.07, 0.07, 0.08)
STEEL = (0.62, 0.63, 0.66)
GREEN = (0.12, 0.45, 0.25)
DARK = (0.20, 0.21, 0.23)
YELLOW = (0.92, 0.74, 0.10)

# -- layout (m; heights above the bench) ------------------------------------------------------------------------------
XF = 0.300                 # the closed door's inner face: the switch's "key insertion face"
SEAL = 0.001               # the door stops 1 mm off the enclosure's frame
EX0, EX1 = XF + SEAL, 0.580          # enclosure, front to back
EY0, EY1 = -0.215, 0.265             # enclosure, right to left
EH = 0.270                           # enclosure height
P = 0.020                            # 20 x 20 mm profile
BASE_T = 0.006                       # base plate
PANE = 0.004
HX, HY = XF, 0.100                   # the door's hinge axis (vertical), at the door's inner face on its hinge edge
DOOR_T = 0.020                       # door frame thickness
DOOR_Y0 = EY0                        # the door's latch edge
DOOR_Z0, DOOR_Z1 = 0.010, 0.250
OPEN = math.pi / 2                   # the door starts open 90 degrees

# -- Omron D4NS-4AF and D4DS-K3 (catalogue values unless marked) -----------------------------------------------------
HEAD_W, HEAD_D = 0.0302, 0.0306
BODY_W, BODY_D = 0.031, 0.030
SWITCH_H, HEAD_H = 0.096, 0.033      # HEAD_H: estimate (the red head and its neck)
HOLE_W, HOLE_H = 0.0153, 0.0044
HOLE_BELOW_TOP = 0.0075
TRAVEL = 0.028
SET_GAP = 0.0015                     # door key's shoulder to the head's face when shut (Omron: 0.5 to 3 mm)
KEY_L, KEY_H, KEY_D = 0.056, 0.015, 0.014
FLANGE_D, BOSS_L = 0.0077, 0.030
TONGUE_L, TONGUE_W, TONGUE_T = 0.028, 0.013, 0.002
WINDOW_W = 0.0075                    # estimate
WINDOW_U0, BAR_U0, BAR_U1 = 0.006, 0.019, 0.0215   # along the tongue from its root: estimate
KEY_MASS = 0.025                     # estimate

XH = XF + KEY_D + SET_GAP            # the head's front face (key hole mouth): 0.3155
YK = -0.150                          # key hole centre
ZK = 0.180                           # key hole centre, above the bench
LIP = 0.002                          # the mouth is a hole in a 2 mm front lip
CAV_HALF = 0.012                     # the cavity behind the lip, half width (estimate)
FLOOR = -HOLE_H / 2                  # guide floor, relative to ZK (level with the hole's lower edge)
ROOF = 0.0008                        # guide roof (estimate): a 3 mm channel for the 2 mm tongue
DOOR_KEY_DZ = -0.0009                # the door's key sits 0.9 mm below the hole's centre line (Omron allows +-1 mm)

# The plunger that stands for the cam (all estimates; the spring is set so the forces match the datasheet)
NOSE_Y0, NOSE_Y1 = 0.0, 0.003        # across the tongue, from YK (inside the tongue's window)
NOSE_BACK = 0.0170                   # depth of the top of the steep face behind the head's face
NOSE_DROP = 0.0001                   # the nose's edge rests this far above the floor
RAMP_H = 0.0034                      # height of both faces
RAMP_IN_DEG = 35.0                   # the face the key meets going in; the nose is one square-cornered bar, so the
RAMP_OUT_DEG = 90.0 - RAMP_IN_DEG    # face it meets coming out is at 55 degrees
PLUNGER_RATE = float(os.environ.get("IBK_RATE", 6000.0))     # N/m
PLUNGER_TRAVEL = 0.0035
FRICTION = 0.3                       # steel tongue on the head's plastic and the plunger

# The door key's tongue is set (its own adjusting screw, range 20 degrees) tangent to the door's swing where it goes
# through the mouth, so a straight tongue passes the 15.3 mm hole with 0.8 mm a side.
RHO = math.hypot(XH - HX, YK - HY)                       # hinge axis to the mouth
_TH_M = math.atan2(YK - HY, XH - HX)
_TH_MID = _TH_M + (TONGUE_L / 2 - SET_GAP) / RHO          # the tongue's middle when the door is shut
TONGUE_DIR = (-math.sin(_TH_MID), math.cos(_TH_MID))     # unit vector along the door key's tongue, door shut
TONGUE_TILT = math.atan2(TONGUE_DIR[1], TONGUE_DIR[0])   # about 6.4 degrees toward the hinge
_MID = (HX + RHO * math.cos(_TH_MID), HY + RHO * math.sin(_TH_MID))
DOOR_KEY_ROOT = (_MID[0] - TONGUE_L / 2 * TONGUE_DIR[0], _MID[1] - TONGUE_L / 2 * TONGUE_DIR[1])
KEY_RADIUS = math.hypot(DOOR_KEY_ROOT[0] - HX, DOOR_KEY_ROOT[1] - HY)    # about 0.251 m


def _qz(a):
    return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))


def _qy(a):
    return (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def wbox(x0, x1, y0, y1, z0, z1, color, quat=None):
    """A box by its world extents (z above the bench), optionally turned about its own centre."""
    c = (float((x0 + x1) / 2), float((y0 + y1) / 2), float(TOP + (z0 + z1) / 2))
    s = (float(abs(x1 - x0)), float(abs(y1 - y0)), float(abs(z1 - z0)))
    return (c, s, color) if quat is None else (c, s, color, tuple(float(v) for v in quat))


def rel(boxes, origin):
    """World boxes in the frame of a part whose origin is the world point ``origin``."""
    out = []
    for b in boxes:
        c = tuple(float(b[0][i] - origin[i]) for i in range(3))
        out.append((c, *b[1:]))
    return out


def turned(boxes, angle, pivot=(HX, HY)):
    """World boxes turned by ``angle`` about the vertical line through ``pivot``."""
    ca, sa = math.cos(angle), math.sin(angle)
    out = []
    for b in boxes:
        dx, dy = b[0][0] - pivot[0], b[0][1] - pivot[1]
        c = (float(pivot[0] + ca * dx - sa * dy), float(pivot[1] + sa * dx + ca * dy), b[0][2])
        q = _qmul(_qz(angle), b[3] if len(b) > 3 else (1.0, 0.0, 0.0, 0.0))
        out.append((c, b[1], b[2], tuple(float(v) for v in q)))
    return out


def tongue_boxes(root, direction, z, color=STEEL):
    """A D4DS key's tongue as world boxes: two prongs, the solid root and the cross-bar. ``root`` is the middle of
    the tongue where it leaves the key's body (world x, y), ``direction`` a unit vector along it, ``z`` its mid height."""
    yaw = math.atan2(direction[1], direction[0])
    side = (-direction[1], direction[0])
    prong = (TONGUE_W - WINDOW_W) / 2

    def piece(u0, u1, w0, w1):
        u, w = (u0 + u1) / 2, (w0 + w1) / 2
        cx, cy = root[0] + u * direction[0] + w * side[0], root[1] + u * direction[1] + w * side[1]
        return ((float(cx), float(cy), float(TOP + z)), (float(u1 - u0), float(w1 - w0), TONGUE_T), color, _qz(yaw))

    return [piece(0.0, TONGUE_L, WINDOW_W / 2, WINDOW_W / 2 + prong),
            piece(0.0, TONGUE_L, -WINDOW_W / 2 - prong, -WINDOW_W / 2),
            piece(0.0, WINDOW_U0, -WINDOW_W / 2, WINDOW_W / 2),
            piece(BAR_U0, BAR_U1, -WINDOW_W / 2, WINDOW_W / 2)]


def key_body_boxes(x_back, y_mid, z_mid, color=BLACK):
    """A D4DS-K3 body as world boxes, its back (mounting) face at x_back, the tongue leaving toward +x: the 7.7 mm
    flange (56 x 15, its rounded ends as two steps) and the 30 mm boss standing 6.3 mm proud."""
    return [wbox(x_back, x_back + FLANGE_D, y_mid - KEY_L / 2, y_mid + KEY_L / 2, z_mid - 0.0044, z_mid + 0.0044, color),
            wbox(x_back, x_back + FLANGE_D, y_mid - 0.0245, y_mid + 0.0245, z_mid - KEY_H / 2, z_mid + KEY_H / 2, color),
            wbox(x_back + FLANGE_D, x_back + KEY_D, y_mid - BOSS_L / 2, y_mid + BOSS_L / 2, z_mid - KEY_H / 2,
                 z_mid + KEY_H / 2, color)]


# -- the enclosure: base plate, profile frame, polycarbonate panels, the switch on its bracket --------------------------
SW_TOP = ZK + HOLE_BELOW_TOP                  # top of the switch's head
SW_BOT = SW_TOP - SWITCH_H
HEAD_BOT = SW_TOP - HEAD_H
XB = XH + HEAD_D                              # the switch's back face
POST_IN = EY0 + P                             # the latch post's inner face (y)

ENC = [wbox(EX0, EX1, EY0, EY1, 0.0, BASE_T, ALU)]
for (px, py) in ((EX0, EY0), (EX0, HY), (EX0, EY1 - P), (EX1 - P, EY0), (EX1 - P, EY1 - P)):
    ENC.append(wbox(px, px + P, py, py + P, BASE_T, EH - P, ALU))                     # posts
ENC += [wbox(EX0, EX0 + P, EY0, EY1, EH - P, EH, ALU), wbox(EX1 - P, EX1, EY0, EY1, EH - P, EH, ALU),   # top rails
        wbox(EX0 + P, EX1 - P, EY0, EY0 + P, EH - P, EH, ALU), wbox(EX0 + P, EX1 - P, EY1 - P, EY1, EH - P, EH, ALU)]
ENC += [wbox(EX1 - 0.012, EX1 - 0.012 + PANE, EY0 + P, EY1 - P, BASE_T, EH - P, PC),                    # back pane
        wbox(EX0 + P, EX1 - P, EY0 + 0.008, EY0 + 0.008 + PANE, BASE_T, EH - P, PC),                    # right pane
        wbox(EX0 + P, EX1 - P, EY1 - 0.012, EY1 - 0.012 + PANE, BASE_T, EH - P, PC),                    # left pane
        wbox(EX0 + P, EX1 - P, EY0 + P, EY1 - P, EH - 0.012, EH - 0.012 + PANE, PC),                    # roof pane
        wbox(EX0 + 0.008, EX0 + 0.008 + PANE, HY + P, EY1 - P, BASE_T, EH - P, PC)]                     # fixed front pane
# the switch's bracket: a 3 mm aluminium angle bolted to the latch post, the switch screwed to it through its back
ENC += [wbox(XB, XB + 0.003, POST_IN, YK + 0.022, SW_BOT - 0.004, SW_TOP - 0.020, ALU),
        wbox(EX0 + 0.004, XB + 0.003, POST_IN, POST_IN + 0.003, SW_BOT - 0.004, SW_TOP - 0.020, ALU)]
# the switch: black body, red head. The head is a shell round the key's path.
ENC.append(wbox(XH, XH + BODY_D, YK - BODY_W / 2, YK + BODY_W / 2, SW_BOT, HEAD_BOT, BLACK))
_hw = HEAD_W / 2
ENC += [
    wbox(XH, XB, YK - _hw, YK + _hw, HEAD_BOT, ZK + FLOOR, RED),                                 # under the key's path
    wbox(XH, XH + LIP, YK - _hw, YK + _hw, ZK + HOLE_H / 2, SW_TOP, RED),                        # lip above the hole
    wbox(XH, XH + LIP, YK - _hw, YK - HOLE_W / 2, ZK + FLOOR, ZK + HOLE_H / 2, RED),             # lip beside the hole
    wbox(XH, XH + LIP, YK + HOLE_W / 2, YK + _hw, ZK + FLOOR, ZK + HOLE_H / 2, RED),
    wbox(XH + LIP, XB, YK - _hw, YK - CAV_HALF, ZK + FLOOR, SW_TOP, RED),                        # side walls
    wbox(XH + LIP, XB, YK + CAV_HALF, YK + _hw, ZK + FLOOR, SW_TOP, RED),
    wbox(XH + TRAVEL + 0.0006, XB, YK - CAV_HALF, YK + CAV_HALF, ZK + FLOOR, SW_TOP, RED),      # back wall
    wbox(XH + LIP, XB, YK - CAV_HALF, YK + CAV_HALF, ZK + 0.0060, SW_TOP, RED),                  # head cap
]
# guide roof over the tongue, in front of and behind the plunger's nose
NOSE_APEX = NOSE_BACK - RAMP_H / math.tan(math.radians(RAMP_OUT_DEG))                         # depth of the nose's edge
NOSE_FRONT = NOSE_APEX - RAMP_H / math.tan(math.radians(RAMP_IN_DEG))                         # depth of the nose's front
ENC += [wbox(XH + LIP, XH + NOSE_FRONT - 0.0004, YK - CAV_HALF, YK + CAV_HALF, ZK + ROOF, ZK + 0.0060, RED),
        wbox(XH + NOSE_BACK + 0.0004, XH + TRAVEL + 0.0006, YK - CAV_HALF, YK + CAV_HALF, ZK + ROOF, ZK + 0.0060, RED)]


def _nose():
    """The plunger's nose: one bar of rectangular section turned RAMP_IN_DEG about y, so that its lower edge is the
    nose's edge, its underside the face a key lifts going in and its end the face a key lifts coming out."""
    a = math.radians(RAMP_IN_DEG)
    length, height = RAMP_H / math.sin(a), RAMP_H / math.cos(a)
    zb = FLOOR + NOSE_DROP                                 # the edge at rest, relative to ZK
    cd = NOSE_APEX - math.cos(a) * length / 2 + math.sin(a) * height / 2
    cz = zb + math.sin(a) * length / 2 + math.cos(a) * height / 2
    return ((float(XH + cd), float(YK + (NOSE_Y0 + NOSE_Y1) / 2), float(TOP + ZK + cz)),
            (float(length), float(NOSE_Y1 - NOSE_Y0), float(height)), STEEL, _qy(a))


PLUNGER_REST = (XH + (NOSE_FRONT + NOSE_BACK) / 2, YK + (NOSE_Y0 + NOSE_Y1) / 2, TOP + ZK + 0.003)
PLUNGER_BOXES = [_nose()]
ENC_ORIGIN = (EX0, 0.0, TOP)
PLUNGER = Slider("plunger", rest=tuple(float(PLUNGER_REST[i] - ENC_ORIGIN[i]) for i in range(3)),
                 boxes=tuple(rel(PLUNGER_BOXES, ENC_ORIGIN)), travel=PLUNGER_TRAVEL, rate=PLUNGER_RATE, damping=8.0,
                 mass=0.010, axis="Z", sign=1)
ENCLOSURE = part("enclosure", pos=ENC_ORIGIN, mass=6.0, boxes=rel(ENC, ENC_ORIGIN), sliders=(PLUNGER,), color=ALU,
                 friction=FRICTION, fixed=True, meta={"contact_offset": 0.0002})

# -- the door: profile frame, polycarbonate pane, the key on its mounting flat. Written shut, then turned open. --------
_door_key_z = ZK + DOOR_KEY_DZ
_pin = (DOOR_KEY_ROOT[0] - 0.008 * TONGUE_DIR[0], DOOR_KEY_ROOT[1] - 0.008 * TONGUE_DIR[1])   # the tongue's pivot pin
DOOR_SHUT = [
    wbox(XF - DOOR_T, XF, HY - P, HY, DOOR_Z0, DOOR_Z1, ALU),                       # hinge stile
    wbox(XF - DOOR_T, XF, DOOR_Y0, DOOR_Y0 + P, DOOR_Z0, DOOR_Z1, ALU),             # latch stile
    wbox(XF - DOOR_T, XF, DOOR_Y0 + P, HY - P, DOOR_Z1 - P, DOOR_Z1, ALU),          # top rail
    wbox(XF - DOOR_T, XF, DOOR_Y0 + P, HY - P, DOOR_Z0, DOOR_Z0 + P, ALU),          # bottom rail
    wbox(XF - 0.012, XF - 0.012 + PANE, DOOR_Y0 + P, HY - P, DOOR_Z0 + P, DOOR_Z1 - P, PC),
    wbox(XF - 0.003, XF, DOOR_Y0 + P, _pin[1] + 0.036, _door_key_z - 0.020, _door_key_z + 0.020, ALU),   # key's flat
] + key_body_boxes(XF, _pin[1], _door_key_z) + tongue_boxes(DOOR_KEY_ROOT, TONGUE_DIR, _door_key_z)
DOOR_ORIGIN = (HX, HY, TOP)
DOOR_MASS = 0.85
LEAF = Hinge("leaf", pivot=(0.0, 0.0, 0.0), boxes=tuple(rel(turned(DOOR_SHUT, -OPEN), DOOR_ORIGIN)), lower=0.0,
             upper=OPEN, stiffness=0.0, damping=1.0, effort=0.15, mass=DOOR_MASS, axis="Z")
# the fixed half of the hinges: a 4 mm pin down the hinge line, inside the door's corner
DOOR = part("door", pos=DOOR_ORIGIN, mass=0.1, boxes=rel([wbox(HX - 0.0045, HX - 0.0005, HY - 0.0045, HY - 0.0005,
                                                              DOOR_Z0 + 0.002, DOOR_Z1 - 0.002, STEEL)], DOOR_ORIGIN),
            hinges=(LEAF,), color=ALU, friction=FRICTION, fixed=True, meta={"contact_offset": 0.0002})

# -- the spare key (part origin: the middle of its tongue's root; +x along the tongue) ---------------------------------
SPARE_GAP = 0.0003                              # its boss stands 0.3 mm off the head's face
SPARE_Z = ZK + FLOOR + TONGUE_T / 2             # its tongue lies on the guide floor
SPARE_START = (float(XH - SPARE_GAP), float(YK), float(TOP + SPARE_Z))
_sp = key_body_boxes(XH - SPARE_GAP - KEY_D, YK, SPARE_Z, BLACK) + tongue_boxes((XH - SPARE_GAP, YK), (1.0, 0.0),
                                                                                  SPARE_Z)
# The body's three boxes take a gripper pad's friction on painted metal (1.0); the tongue keeps 0.3.
SPARE = part("spare_key", pos=SPARE_START, mass=KEY_MASS, boxes=rel(_sp, SPARE_START), color=BLACK,
             friction=FRICTION, meta={"contact_offset": 0.0002, "grip_boxes": (0, 1, 2), "grip_friction": 1.0})

# -- the lathe (scenery, bolted to the base plate; overall 360 x 150 x 150 mm, proportions estimated) --------------------
LX, LY = 0.445, 0.020
_l0 = BASE_T + 0.0005
LATHE_BOXES = [
    wbox(LX - 0.045, LX + 0.045, LY - 0.180, LY + 0.180, _l0, _l0 + 0.040, ALU),                 # bed
    wbox(LX - 0.060, LX + 0.060, LY + 0.080, LY + 0.180, _l0 + 0.040, _l0 + 0.150, GREEN),       # headstock
    wbox(LX + 0.045, LX + 0.090, LY + 0.090, LY + 0.170, _l0, _l0 + 0.090, DARK),                # motor
    wbox(LX - 0.025, LX + 0.025, LY - 0.180, LY - 0.130, _l0 + 0.040, _l0 + 0.120, GREEN),       # tailstock
    wbox(LX - 0.055, LX + 0.055, LY - 0.060, LY + 0.000, _l0 + 0.040, _l0 + 0.062, ALU),         # cross slide
    wbox(LX - 0.035, LX - 0.015, LY - 0.045, LY - 0.020, _l0 + 0.062, _l0 + 0.090, DARK),        # tool post
]
LATHE_CYL = [((LX, LY + 0.062, TOP + _l0 + 0.095), 0.025, 0.036, STEEL, "Y"),                    # three-jaw chuck
             ((LX, LY - 0.120, TOP + _l0 + 0.095), 0.006, 0.030, STEEL, "Y"),                    # tailstock centre
             ((LX - 0.062, LY - 0.030, TOP + _l0 + 0.050), 0.014, 0.010, BLACK, "X")]            # cross-slide handwheel
LATHE_ORIGIN = (LX, LY, TOP)
LATHE = part("lathe", pos=LATHE_ORIGIN, mass=4.5, boxes=rel(LATHE_BOXES, LATHE_ORIGIN),
             cylinders=[(tuple(float(c[0][i] - LATHE_ORIGIN[i]) for i in range(3)), *c[1:]) for c in LATHE_CYL],
             color=GREEN, friction=0.6, fixed=True)

OBJECTS = [ENCLOSURE, LATHE, DOOR, SPARE]

# -- scoring -----------------------------------------------------------------------------------------------------------
IN_SWITCH = 0.019          # a spare key this far in (within 9 mm of fully in) can hold the contacts closed
DOOR_NEAR = 0.009          # 0.75: door key within 9 mm of home (the largest pretravel on the datasheet)
DOOR_HOME = 0.002          # 1.0: within 2 mm of home (inside the smallest pretravel, 3 mm)
DOOR_ONTO = 0.045          # the door within 45 mm of shut at its key while the spare key is in the switch
RELEASE = 0.03             # grippers this far from the spare key and from the door
ON_BENCH = 0.003           # the spare key's lowest point within 3 mm of the bench top
STILL_STEPS = 20
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), violation=None, spare_was_out=False, key_free=False,
                  door_free=False, finger_check=None, recent=[], info={})


_reset_state()

# sample points of a key in its own frame: the body's corners and the tongue's tip corners
_KEY_PTS = np.array([[x, y, z] for x in (-KEY_D, 0.0) for y in (-KEY_L / 2, KEY_L / 2) for z in (-KEY_H / 2, KEY_H / 2)]
                    + [[TONGUE_L, y, 0.0] for y in (-TONGUE_W / 2, TONGUE_W / 2)])


def door_gap(door: PartState) -> float:
    """How far the door's key is from home, along its swing (m): 0 with the door shut on its stop."""
    return float(max(0.0, OPEN - door.hinges["leaf"]) * KEY_RADIUS)


def spare_depth(spare: PartState) -> float:
    """How far the spare key's tongue reaches past the head's face into the key hole (m); 0 when it is not in it."""
    best = 0.0
    for y in (-TONGUE_W / 2, 0.0, TONGUE_W / 2):
        tip = spare.to_world((TONGUE_L, y, 0.0))
        if abs(tip[1] - YK) < CAV_HALF and abs(tip[2] - (TOP + ZK)) < 0.006:
            best = max(best, float(tip[0] - XH))
    return best


def spare_on_bench(spare: PartState) -> bool:
    """The spare key lies on the bench top, wholly outside the enclosure's footprint."""
    pts = np.array([spare.to_world(p) for p in _KEY_PTS])
    low = float(pts[:, 2].min() - TOP)
    outside = bool(pts[:, 0].max() < EX0 or pts[:, 0].min() > EX1 or pts[:, 1].max() < EY0 or pts[:, 1].min() > EY1)
    return bool(outside and -0.005 < low < ON_BENCH)       # on the bench top, not on the floor beside it


def _door_distance(q: float, p_world) -> float:
    """Distance from a world point to the door's frame (a box when shut), with the door at hinge angle q."""
    a = OPEN - q                                          # undo the opening: the point in the shut door's frame
    dx, dy = p_world[0] - HX, p_world[1] - HY
    x = HX + math.cos(a) * dx - math.sin(a) * dy
    y = HY + math.sin(a) * dx + math.cos(a) * dy
    z = p_world[2] - TOP
    d = (max(XF - DOOR_T - x, 0.0, x - XF), max(DOOR_Y0 - y, 0.0, y - HY), max(DOOR_Z0 - z, 0.0, z - DOOR_Z1))
    return float(math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))


@on_step
def _step(environment) -> None:
    """Every control step: count steps and check where the grippers are (score() cannot see the robot)."""
    from yam_arena.scene import object_states

    _STATE["step"] += 1
    try:
        st = object_states(environment, ["door", "spare_key"])
        pts = finger_points(environment)
        key_pts = [st["spare_key"].to_world(p) for p in ((-KEY_D / 2, 0.0, 0.0), (-KEY_D / 2, KEY_L / 2, 0.0),
                                                         (-KEY_D / 2, -KEY_L / 2, 0.0), (TONGUE_L, 0.0, 0.0))]
        _STATE["key_free"] = bool(min(np.linalg.norm(p - k) for p in pts for k in key_pts) >= RELEASE)
        q = st["door"].hinges["leaf"]
        _STATE["door_free"] = bool(min(_door_distance(q, p) for p in pts) >= RELEASE)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"interlock-bypass-key-remove: gripper check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"], _STATE["key_free"], _STATE["door_free"] = False, False, False


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    door, spare = objects["door"], objects["spare_key"]
    depth, gap = spare_depth(spare), door_gap(door)
    if depth <= 0.0:
        _STATE["spare_was_out"] |= True
    elif depth >= IN_SWITCH:
        if _STATE["spare_was_out"]:
            _STATE["violation"] = _STATE["violation"] or "spare key put back into the switch"
        if gap < DOOR_ONTO:
            _STATE["violation"] = _STATE["violation"] or "door swung onto the spare key in the switch"
    stowed = depth <= 0.0 and spare_on_bench(spare) and _STATE["key_free"]
    recent = _STATE["recent"]
    recent.append(np.concatenate([spare.pos, [door.hinges["leaf"] * KEY_RADIUS]]))
    del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(float(np.abs(r - recent[-1]).max()) < 0.001 for r in recent)
    _STATE["info"] = {"spare_depth_mm": round(1000 * depth, 2), "door_gap_mm": round(1000 * gap, 2), "stowed": stowed,
                      "on_bench": spare_on_bench(spare), "key_free": _STATE["key_free"],
                      "door_free": _STATE["door_free"], "still": still, "violation": _STATE["violation"]}
    if _STATE["violation"] or depth > 0.0:
        result = (0.0, False)
    elif stowed and gap <= DOOR_HOME and _STATE["door_free"] and still:
        result = (1.0, True)
    elif stowed and gap <= DOOR_NEAR:
        result = (0.75, False)
    elif stowed:
        result = (0.5, False)
    else:
        result = (0.25, False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """The spare key out of the switch and lying on the bench outside the enclosure, the door shut with its own key
    within 2 mm of home, both grippers clear, everything still for the last second, and no violation."""
    return _update(objects)[1]


def distance(objects) -> float:
    return float(spare_depth(objects["spare_key"]) + door_gap(objects["door"]))


def score(objects) -> float:
    """End state, each level including the ones before it. 0.25: the spare key is out of the switch (no part of its
    tongue in the key hole). 0.5: it lies on the bench top (lowest point within 3 mm of it), wholly outside the
    enclosure's footprint, with both grippers 3 cm clear of it. 0.75: the door's own key is within 9 mm of home,
    measured along its swing. 1.0: the door's key is within 2 mm of home, both grippers are 3 cm clear of the door, and
    neither the key nor the door has moved more than 1 mm for 1 second. 0 while any part of the spare key's tongue is
    in the key hole. 0 for good if the spare key was put back (19 mm or more into the hole after it had been out) or
    the door was swung to within 45 mm of shut at its key while the spare key was 19 mm or more in the hole. A spare
    key left anywhere else (inside or on the enclosure, on the lathe, in a gripper) stays at 0.25."""
    return _update(objects)[0]
