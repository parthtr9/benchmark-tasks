"""On the station counter: a centrifugal juicer (Breville Juice Fountain Compact BJE200XL) being put back together
after washing. The motor base stands with its pulp container on it, the filter basket clicked down on the motor's
drive coupling, and its wire safety locking arm lowered toward the robot. The cover stands on a draining board to the
robot's left.

The instruction booklet's last assembly steps: place the cover over the filter basket and the pulp container, the
ridges of its rim lined up with the container's two wings, and lower it; raise the safety locking arm and lock it into
the two grooves on either side of the cover. The arm's top lies across the cover's two lugs only when everything under
it is down: a cover that is up (turned off the wings, or held up by a basket that was not clicked down) puts the lugs'
guide ribs in the arm's way.

Sizes: overall 231 x 249 x 399 mm and the 76 mm feed chute are the maker's figures; every other dimension is scaled
from the booklet's product drawing (measure the bought part). The food pusher is not part of the trial.

Frames. Geometry is written in mm in the JUICER frame: origin at the centre of the base on the counter, x toward the
juicer's front (the spout, away from the robot), y to the robot's left, z up. The juicer stands square to the robot,
so the juicer frame is the world frame shifted to (XJ, YJ, TOP).

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
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = "Put the juicer together so that its safety locking arm is closed over the cover."

WHITE = (0.90, 0.90, 0.91)
CLEAR = (0.70, 0.78, 0.84)       # the clear plastic of the pulp container and the cover
STEEL = (0.74, 0.75, 0.77)
DARK = (0.20, 0.21, 0.23)
BLADE = (0.55, 0.56, 0.60)
WOOD = (0.72, 0.58, 0.40)
RIDGE = (0.45, 0.55, 0.65)

# -- layout (world, m) ---------------------------------------------------------------------------------------------------
XJ, YJ = 0.40, 0.0                       # the juicer's axis
BOARD_T = 0.035                          # draining board under the cover
COVER_XY = (0.35, 0.235)

# -- dimensions (mm, juicer frame) ----------------------------------------------------------------------------------------
BASE_R, BASE_TOP = 85.0, 163.0           # motor base
FLOOR_TOP = 168.0                        # pulp container floor (the basket's seat)
WALL_RIN, WALL_T, WALL_TOP = 100.0, 3.0, 246.0
WING_HALF, WING_ROUT, WING_H = 10.0, 107.5, 10.0     # the container's two wings, on its sides, flush with the wall top
COUPLING_R, COUPLING_TOP = 24.0, 196.0   # motor drive coupling
PROUD = 9.0                              # the basket stands this high on the coupling's click before it is pushed down
CLICK_RATE, CLICK_TRAVEL = 1200.0, 3.5   # each of the four sprung catches in the coupling (N/m, mm)
CLICK_TIP = 2.5                          # how far each catch stands out past the basket's inner lip
LIP_RIN = 24.75                          # basket collar: lip and guide ring, 0.75 mm play on the coupling
BASKET_DISC_R, BASKET_DISC_Z = 45.0, 30.0            # cutting disc (basket frame: z from the collar's bottom)
BASKET_RIM_R, BASKET_RIM_Z = 72.0, 67.0
PIVOT_Z = 125.0                          # locking arm pivots
LEG, LEG_Y = 156.0, 112.5                # arm legs: length, and distance from the middle
WIRE_R = 3.0                             # 6 mm wire
REST_DEG = 5.0                           # the lowered arm rests with its legs 5 degrees above level
# The arm's top, in the plane at right angles to its legs: (distance back from the leg tops, y). The short runs from
# the leg tops inward are the two segments that lie in the cover's grooves; the back bar is the handle.
HOOP = [(0.0, LEG_Y), (0.0, 95.0), (62.0, 89.0), (100.0, 45.0), (100.0, -45.0), (62.0, -89.0), (0.0, -95.0),
        (0.0, -LEG_Y)]
HANDLE_S = 100.0
# cover (cover frame: origin on its axis in the plane of its skirt's bottom edge)
SKIRT_RIN, SKIRT_T, SKIRT_H = 105.0, 3.0, 10.0       # the skirt drops over the container's wall (2 mm a side)
SLOT_HALF = 11.0                         # each side of the skirt has a slot for a wing (1 mm a side)
LEDGE_Z = 10.0                           # the ledge that sits on the wall top
COVER_SEAT = WALL_TOP - LEDGE_Z          # cover origin height when seated (236)
SHOULDER = (97.0, 32.0)                  # (radius, z) where the shoulder meets the dome
CHUTE_ROUT, CHUTE_T, CHUTE_BOTTOM, CHUTE_TOP = 40.5, 2.5, 12.0, 140.0
INNER_RIN, INNER_T, INNER_BOTTOM = 70.5, 3.0, 1.0    # inner ring: 1 mm above a seated basket's rim
EAR_Y0, EAR_Y1 = 98.0, 108.0             # the two lugs on the shoulder
GROOVE_FLOOR = 41.5                      # groove floor: 0.5 mm under the arm's wire when the arm is upright
GROOVE_BACK, GROOVE_FRONT = -4.5, 4.5
RIDGE_X, RIDGE_UP = -5.5, 1.5            # sprung ridge behind each groove: crest position and height over the floor
RIDGE_RATE, RIDGE_TRAVEL = 1200.0, 2.5
ARM_TOP = PIVOT_Z + LEG                  # 281: the wire's centre height when upright
assert abs(ARM_TOP - WIRE_R - 0.5 - (COVER_SEAT + GROOVE_FLOOR)) < 1e-6


# -- shape helpers (mm in, partkit specs in m out) -------------------------------------------------------------------------
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


def ring(r_in, t, z0, z1, color, n=16, skip=()):
    """A tube wall: ``n`` boxes whose inner faces lie at r_in (flats centred on the axes)."""
    out = []
    w = 2 * (r_in + t) * math.tan(math.pi / n) * 1.03
    for k in range(n):
        if k in skip:
            continue
        a = 2 * math.pi * k / n
        rm = r_in + t / 2
        out.append(box((rm * math.cos(a), rm * math.sin(a), (z0 + z1) / 2), (t, w, z1 - z0), color, _rot_z(a)))
    return out


def cone(r0, z0, r1, z1, t, color, n=16):
    """A conical wall from radius r0 at z0 to r1 at z1: ``n`` boxes centred on that surface."""
    out = []
    gen = math.hypot(r1 - r0, z1 - z0)
    ang = math.atan2(r1 - r0, z1 - z0)
    w = 2 * max(r0, r1) * math.tan(math.pi / n) * 1.05
    for k in range(n):
        a = 2 * math.pi * k / n
        rm = (r0 + r1) / 2
        out.append(box((rm * math.cos(a), rm * math.sin(a), (z0 + z1) / 2), (t, w, gen), color,
                       _rot_z(a) @ _rot_y(ang)))
    return out


def disc(r, z0, z1, color, n=16):
    """A solid ``n``-sided prism, r across the flats' middle (PhysX's own cylinders miss edge contacts)."""
    return [box((0.0, 0.0, (z0 + z1) / 2), (2 * r, 2 * r * math.tan(math.pi / n), z1 - z0), color,
                _rot_z(2 * math.pi * k / n)) for k in range(n // 2)]


def wire(p0, p1, color, r=WIRE_R, extra=0.0, flat=None):
    """A bar of regular octagonal section from p0 to p1 (four slabs), 2r across the flats; ``flat`` is a direction
    one pair of flats faces."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    d = d / length
    ref = np.asarray(flat, float) if flat is not None else (
        np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0]))
    e1 = ref - (ref @ d) * d
    e1 /= np.linalg.norm(e1)
    R = np.c_[e1, np.cross(d, e1), d]
    c = (p0 + p1) / 2
    w = 2 * r * math.tan(math.pi / 8)
    return [box(c, (2 * r, w, length + extra), color, R @ _rot_z(k * math.pi / 4)) for k in range(4)]


# -- the locking arm's geometry ------------------------------------------------------------------------------------------
def arm_axes(alpha_deg: float):
    """Unit vectors (juicer frame) along the arm's legs and along its top (back from the leg tops), with the legs
    raised ``alpha_deg`` above level toward the robot's side (90 = upright)."""
    a = math.radians(alpha_deg)
    return np.array([-math.cos(a), 0.0, math.sin(a)]), np.array([-math.sin(a), 0.0, -math.cos(a)])


def arm_point(alpha_deg: float, s: float, y: float, leg: float = LEG) -> np.ndarray:
    """A point of the arm (juicer frame, mm): ``leg`` along the legs from the pivot, ``s`` back along the top."""
    u, w = arm_axes(alpha_deg)
    return np.array([0.0, 0.0, PIVOT_Z]) + leg * u + s * w + np.array([0.0, y, 0.0])


def _arm_boxes():
    out = []
    for sgn in (1.0, -1.0):
        out += wire(arm_point(REST_DEG, 0, sgn * LEG_Y, 0.0), arm_point(REST_DEG, 0, sgn * LEG_Y), STEEL, extra=WIRE_R)
        out += wire((0.0, sgn * (BASE_R + 1.0), PIVOT_Z), (0.0, sgn * LEG_Y, PIVOT_Z), STEEL)      # pivot stub
    leg_dir = arm_axes(REST_DEG)[0]           # the arm's top has flats facing the cover when it is upright
    for (s0, y0), (s1, y1) in zip(HOOP, HOOP[1:]):
        out += wire(arm_point(REST_DEG, s0, y0), arm_point(REST_DEG, s1, y1), STEEL, extra=WIRE_R, flat=leg_dir)
    return out


# -- the juicer: base, pulp container, coupling with its four sprung catches, locking arm ----------------------------------
JUICER_BOXES = disc(BASE_R, 0.0, BASE_TOP, WHITE)
JUICER_BOXES += disc(WALL_RIN + WALL_T, BASE_TOP, FLOOR_TOP, CLEAR)
JUICER_BOXES += ring(WALL_RIN, WALL_T, FLOOR_TOP, WALL_TOP, CLEAR)
for sgn in (1.0, -1.0):
    JUICER_BOXES.append(aabb(-WING_HALF, WING_HALF, sgn * (WALL_RIN + WALL_T), sgn * WING_ROUT, WALL_TOP - WING_H,
                             WALL_TOP, CLEAR))
    JUICER_BOXES.append(aabb(-8.0, 8.0, sgn * (BASE_R - 2.0), sgn * (BASE_R + 6.0), PIVOT_Z - 8.0, PIVOT_Z + 8.0,
                             WHITE))                                                    # pivot boss
JUICER_BOXES += disc(COUPLING_R, FLOOR_TOP, COUPLING_TOP, DARK)
JUICER_BOXES.append(aabb(WALL_RIN, WALL_RIN + 34.0, -14.0, 14.0, FLOOR_TOP + 4.0, FLOOR_TOP + 26.0, CLEAR))   # spout

CLICK_Z = FLOOR_TOP + PROUD - CLICK_TIP        # the catches' middle height
_hd = 3.6                                      # each catch is a 45-degree ridge, this high from its middle to its tip
_side = _hd * math.sqrt(2.0)


CATCHES = ("catch_front", "catch_left", "catch_back", "catch_right")   # at +x, +y, -x, -y


def _catch(k):
    """One of the coupling's four sprung catches, at 90 degrees from the next: a 45-degree ridge that slides in."""
    tip = LIP_RIN + CLICK_TIP
    sgn = 1.0 if k < 2 else -1.0
    if k % 2:                                                     # on the y axis
        centre, axis, R = (0.0, sgn * (tip - _hd), CLICK_Z), "Y", np.array(
            [[1, 0, 0], [0, math.cos(math.pi / 4), -math.sin(math.pi / 4)], [0, math.sin(math.pi / 4), math.cos(math.pi / 4)]])
        size = (10.0, _side, _side)
    else:                                                         # on the x axis
        centre, axis, R, size = (sgn * (tip - _hd), 0.0, CLICK_Z), "X", _rot_y(math.pi / 4), (_side, 10.0, _side)
    return Slider(name=CATCHES[k], rest=tuple(v / 1000 for v in centre), boxes=(box(centre, size, DARK, R),),
                  travel=CLICK_TRAVEL / 1000, rate=CLICK_RATE, damping=8.0, mass=0.01, axis=axis, sign=-1 if sgn > 0 else 1)


ARM = Hinge(name="arm", pivot=(0.0, 0.0, PIVOT_Z / 1000), boxes=tuple(_arm_boxes()), lower=0.0,
            upper=math.radians(90.0), stiffness=0.0, damping=0.02, mass=0.19, axis="Y")
JUICER_POS = (XJ, YJ, TOP)
JUICER = part("juicer", pos=JUICER_POS, mass=3.4, boxes=JUICER_BOXES, color=WHITE, friction=0.6,
              hinges=(ARM,), sliders=tuple(_catch(k) for k in range(4)))

# -- the filter basket (origin: on its axis at the bottom of its collar) --------------------------------------------------
LIP_H = PROUD - 2 * CLICK_TIP - 0.4                                     # clicked down, the catches sit 0.4 mm above the lip
BASKET_BOXES = ring(LIP_RIN, 4.75, 0.0, LIP_H, DARK)                   # the lip the catches click over
BASKET_BOXES += ring(28.0, 1.5, LIP_H, 16.0, DARK)
BASKET_BOXES += ring(LIP_RIN, 4.75, 16.0, 19.0, DARK)                  # guide ring
BASKET_BOXES += ring(28.0, 1.5, 19.0, BASKET_DISC_Z, DARK)
BASKET_BOXES += disc(BASKET_DISC_R, BASKET_DISC_Z, BASKET_DISC_Z + 2.0, BLADE)   # the cutting disc
BASKET_BOXES += cone(BASKET_DISC_R, BASKET_DISC_Z + 1.0, BASKET_RIM_R, BASKET_RIM_Z, 3.0, STEEL)
# The basket starts clicked down. Negative-variant check only (demo.py sets it for DEMO_VARIANT=basket_high): it starts
# resting on the catches, not pushed down, PROUD mm high.
BASKET_HIGH_START = os.environ.get("JUICER_BASKET_HIGH") == "1"
BASKET_START = (XJ, YJ, TOP + (FLOOR_TOP + (PROUD + 0.3 if BASKET_HIGH_START else 0.2)) / 1000)
BASKET = part("basket", pos=BASKET_START, mass=0.20, boxes=BASKET_BOXES, color=STEEL, friction=0.4)


# -- the cover (origin: on its axis in the plane of its skirt's bottom edge) ----------------------------------------------
def _cover_boxes():
    out = ring(SKIRT_RIN, SKIRT_T, 0.0, SKIRT_H, CLEAR, skip=(4, 12))         # the faces at +-y carry the wing slots
    face = (SKIRT_RIN + SKIRT_T) * math.tan(math.pi / 16) * 1.03              # half a face's width
    for sgn in (1.0, -1.0):
        y0, y1 = sgn * SKIRT_RIN, sgn * (SKIRT_RIN + SKIRT_T)
        for xs in (1.0, -1.0):
            out.append(aabb(xs * (SLOT_HALF + 4.5), xs * face, y0, y1, 0.0, SKIRT_H, CLEAR))
            out.append(aabb(xs * SLOT_HALF, xs * (SLOT_HALF + 4.5), y0, y1, 4.5, SKIRT_H, CLEAR))
            # 45-degree lead-in at the slot's mouth, from 15.5 mm at the bottom edge to 11 mm
            c = (xs * (SLOT_HALF + 2.25 + 0.7), (y0 + y1) / 2, 2.25 + 0.7)
            out.append(box(c, (6.4, SKIRT_T, 2.0), CLEAR, _rot_y(xs * math.pi / 4)))
    out += ring(94.0, SKIRT_RIN + SKIRT_T - 94.0, LEDGE_Z, LEDGE_Z + 3.0, CLEAR)          # ledge on the wall top
    out += cone(SKIRT_RIN + 1.5, LEDGE_Z + 3.0, SHOULDER[0], SHOULDER[1], 3.0, CLEAR)     # shoulder
    out += cone(SHOULDER[0] + 1.0, SHOULDER[1], CHUTE_ROUT + 1.5, 38.5, 3.0, CLEAR)       # dome
    out += ring(INNER_RIN, INNER_T, INNER_BOTTOM, 34.0, CLEAR)                           # inner ring over the basket
    out += ring(CHUTE_ROUT - CHUTE_T, CHUTE_T, CHUTE_BOTTOM, CHUTE_TOP, STEEL)            # feed chute
    for sgn in (1.0, -1.0):
        y0, y1 = sgn * EAR_Y0, sgn * EAR_Y1
        out.append(aabb(-13.0, GROOVE_BACK, y0, y1, 14.0, GROOVE_FLOOR - 2.0, CLEAR))     # behind the groove
        # a rib leading to the groove, stepped down under the wire's arc: a wire that arrives too low for it (a
        # cover that is not down) meets its end squarely
        out.append(aabb(-25.0, -13.0, y0, y1, 14.0, GROOVE_FLOOR - 3.5, CLEAR))
        out.append(aabb(-35.0, -25.0, y0, y1, 14.0, GROOVE_FLOOR - 6.0, CLEAR))
        out.append(aabb(GROOVE_BACK, GROOVE_FRONT, y0, y1, 14.0, GROOVE_FLOOR, CLEAR))    # groove floor
        out.append(aabb(GROOVE_FRONT, 11.0, y0, y1, 14.0, GROOVE_FLOOR + 5.0, CLEAR))     # stop in front of the groove
    return out


def _ridge(sgn):
    """The sprung ridge behind a groove: a 20-degree ramp on the side the arm comes from, a 55-degree face on the
    groove's side."""
    crest = np.array([RIDGE_X, GROOVE_FLOOR + RIDGE_UP])                      # (x, z)
    yc, wy = sgn * (EAR_Y0 + EAR_Y1) / 2, EAR_Y1 - EAR_Y0 - 1.0
    a = math.radians(20.0)
    d, n = np.array([-math.cos(a), -math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c1 = crest + d * 4.5 - n * 2.0
    ramp = box((c1[0], yc, c1[1]), (9.0, wy, 4.0), RIDGE, _rot_y(-a))
    b = math.radians(55.0)
    e, m = np.array([math.cos(b), -math.sin(b)]), np.array([math.sin(b), math.cos(b)])
    c2 = crest + e * 1.6 - m * 1.0
    face = box((c2[0], yc, c2[1]), (3.2, wy, 2.0), RIDGE, _rot_y(b))
    rest = (RIDGE_X / 1000, yc / 1000, GROOVE_FLOOR / 1000)
    return Slider(name="ridge_l" if sgn > 0 else "ridge_r", rest=rest, boxes=(ramp, face), travel=RIDGE_TRAVEL / 1000,
                  rate=RIDGE_RATE, damping=8.0, mass=0.003, axis="Z", sign=-1)


COVER_START = (COVER_XY[0], COVER_XY[1], TOP + BOARD_T + 0.0005)
COVER = part("cover", pos=COVER_START, mass=0.25, boxes=_cover_boxes(), color=CLEAR, friction=0.4,
             sliders=(_ridge(1.0), _ridge(-1.0)))
if os.environ.get("JUICER_COVER_TURNED") == "1":      # negative-variant check only: the cover starts a quarter turn out
    COVER.meta["quat_xyzw"] = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))

BOARD = Box("board", pos=(COVER_XY[0], COVER_XY[1], TOP + BOARD_T / 2), size=(0.25, 0.25, BOARD_T), color=WOOD)
OBJECTS = [BOARD, JUICER, BASKET, COVER]

# -- scoring -------------------------------------------------------------------------------------------------------------
BASKET_TOL = 1.0          # mm: a seated cover needs the basket no more than this above its seat
ON_OFF, ON_TILT, ON_GAP = 10.0, math.radians(10.0), 12.0   # cover resting on the container: mm off centre, tilt, mm above its seat
COVER_GAP = 2.0           # mm: the gap under the cover's ledge, all round
GROOVE_WIN = (-4.2, 2.5)  # mm: where the wire's centre may be along the groove's width
LIFT_MAX = 3.0            # mm: the wire is less than this above the groove floor
RIDGE_OUT = 1.0           # mm: a ridge counts as sprung back when within this of its rest
BASE_MOVE, BASE_TILT = 0.020, math.radians(5.0)
RELEASE = 0.03
MOVE_TOL = 0.001    # m: nothing moves more than this over the last 20 steps (1 s)
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), base0=None, moved=False, dropped=False, blade=False,
                  released=False, finger_check=None, recent=[], levels=(False, False, False), geom={})


_reset_state()


def _frame(juicer: PartState):
    return juicer.pos, juicer.rotation()


def _in_juicer(juicer: PartState, p_world) -> np.ndarray:
    """A world point in the juicer frame (mm)."""
    return juicer.to_local(np.asarray(p_world, float)) * 1000


def arm_alpha(juicer: PartState) -> float:
    """The arm's raise above level, degrees (5 lowered, 90 upright)."""
    return REST_DEG + math.degrees(float(juicer.hinges.get("arm", 0.0)))


def geometry(objects) -> dict:
    juicer, basket, cover = objects["juicer"], objects["basket"], objects["cover"]
    g = {}
    b = _in_juicer(juicer, basket.pos)
    g["basket_up_mm"] = float(b[2] - FLOOR_TOP)
    g["basket_off_mm"] = float(math.hypot(b[0], b[1]))
    bz = juicer.rotation().T @ basket.rotation()[:, 2]
    g["basket_tilt"] = float(math.acos(max(-1.0, min(1.0, bz[2]))))
    c = _in_juicer(juicer, cover.pos)
    g["cover_off_mm"] = float(math.hypot(c[0], c[1]))
    cz = juicer.rotation().T @ cover.rotation()[:, 2]
    g["cover_tilt"] = float(math.acos(max(-1.0, min(1.0, cz[2]))))
    gaps = []
    for k in range(8):
        a = 2 * math.pi * k / 8
        p = cover.to_world(np.array([101.5 * math.cos(a), 101.5 * math.sin(a), LEDGE_Z]) / 1000)
        gaps.append(float(_in_juicer(juicer, p)[2] - WALL_TOP))
    g["cover_gap_max_mm"], g["cover_gap_min_mm"] = max(gaps), min(gaps)
    alpha = arm_alpha(juicer)
    g["arm_deg"] = alpha
    wires = []
    for sgn in (1.0, -1.0):
        pj = arm_point(alpha, 0.0, sgn * (EAR_Y0 + EAR_Y1) / 2)                    # the wire's centre over the lug
        pc = cover.to_local(juicer.to_world(pj / 1000)) * 1000
        wires.append((float(pc[0]), float(pc[2] - WIRE_R - GROOVE_FLOOR)))         # along the groove's width, lift
    g["wire_x_mm"] = [w[0] for w in wires]
    g["wire_lift_mm"] = [w[1] for w in wires]
    g["ridge_mm"] = [float(cover.sliders.get(n, 0.0)) * 1000 for n in ("ridge_l", "ridge_r")]
    g["catch_mm"] = [float(juicer.sliders.get(n, 0.0)) * 1000 for n in CATCHES]
    return g


def _levels(g) -> tuple:
    on = (g["cover_off_mm"] < ON_OFF and g["cover_tilt"] < ON_TILT and -1.0 < g["cover_gap_min_mm"]
          and g["cover_gap_max_mm"] < ON_GAP)
    basket = g["basket_up_mm"] <= BASKET_TOL and g["basket_off_mm"] < 2.0 and g["basket_tilt"] < math.radians(2.0)
    cover = on and basket and g["cover_gap_max_mm"] < COVER_GAP and g["cover_off_mm"] < 3.0
    arm = cover and all(GROOVE_WIN[0] <= x <= GROOVE_WIN[1] for x in g["wire_x_mm"]) and all(
        -1.0 < lift < LIFT_MAX for lift in g["wire_lift_mm"]) and all(abs(r) < RIDGE_OUT for r in g["ridge_mm"])
    return on, cover, arm


def _dist_cyl(p, r, z0, z1) -> float:
    """Distance (mm) from a point in a part's frame to the solid cylinder of radius r between z0 and z1 on its axis."""
    dr = math.hypot(p[0], p[1]) - r
    dz = max(z0 - p[2], p[2] - z1)
    if dr <= 0 and dz <= 0:
        return max(dr, dz)
    return math.hypot(max(dr, 0.0), max(dz, 0.0))


def _dist_seg(p, a, b) -> float:
    ab = b - a
    t = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


@on_step
def _step(environment) -> None:
    """Every control step: count steps and check the grippers (score() cannot see the robot): let go of everything,
    and never inside the filter basket."""
    from yam_arena.scene import object_states

    _STATE["step"] += 1
    try:
        st = object_states(environment, ["juicer", "basket", "cover"])
        juicer, basket, cover = st["juicer"], st["basket"], st["cover"]
        alpha = arm_alpha(juicer)
        hoop = [arm_point(alpha, s, y) for s, y in HOOP]
        legs = [(arm_point(alpha, 0, y, 0.0), arm_point(alpha, 0, y)) for y in (LEG_Y, -LEG_Y)]
        near = float("inf")
        for p in finger_points(environment):
            pj, pb, pc = _in_juicer(juicer, p), basket.to_local(p) * 1000, cover.to_local(p) * 1000
            if math.hypot(pb[0], pb[1]) < BASKET_RIM_R - 6.0 and pb[2] < BASKET_RIM_Z - 3.0:
                _STATE["blade"] = True                                           # permanent: reached into the basket
            d = min(_dist_cyl(pj, WALL_RIN + WALL_T, 0.0, WALL_TOP), _dist_cyl(pb, BASKET_RIM_R + 1.5, 0.0, BASKET_RIM_Z),
                    _dist_cyl(pc, SKIRT_RIN + SKIRT_T, 0.0, GROOVE_FLOOR + 5.0),
                    _dist_cyl(pc, CHUTE_ROUT, CHUTE_BOTTOM, CHUTE_TOP))
            d = min([d] + [_dist_seg(pj, a, b) - WIRE_R for a, b in zip(hoop, hoop[1:])]
                    + [_dist_seg(pj, a, b) - WIRE_R for a, b in legs])
            near = min(near, d)
        _STATE["near_mm"] = near
        _STATE["released"] = bool(near >= RELEASE * 1000)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"juicer-arm-lock: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"], _STATE["released"] = False, False


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    juicer, basket, cover = objects["juicer"], objects["basket"], objects["cover"]
    if _STATE["base0"] is None:
        _STATE["base0"] = juicer.pos.copy()
    tilt = math.acos(max(-1.0, min(1.0, float(juicer.rotation()[2, 2]))))
    _STATE["moved"] |= bool(float(np.linalg.norm((juicer.pos - _STATE["base0"])[:2])) > BASE_MOVE or tilt > BASE_TILT)
    _STATE["dropped"] |= bool(min(float(cover.pos[2]), float(basket.pos[2])) < TOP - 0.05)       # off the counter
    g = geometry(objects)
    _STATE["geom"] = g
    cover_on, cover_seated, arm_locked = _levels(g)
    _STATE["levels"] = (cover_on, cover_seated, arm_locked)
    recent = _STATE["recent"]
    recent.append(np.concatenate([juicer.pos, basket.pos, cover.pos, [math.radians(g["arm_deg"]) * LEG / 1000]]))
    del recent[:-20]
    still = len(recent) == 20 and all(float(np.abs(r - recent[-1]).max()) < MOVE_TOL for r in recent)
    if _STATE["moved"] or _STATE["dropped"] or _STATE["blade"]:
        result = (0.0, False)
    elif arm_locked and _STATE["released"] and still:
        result = (1.0, True)
    elif arm_locked:
        result = (0.75, False)
    elif cover_seated:
        result = (0.5, False)
    elif cover_on:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Cover seated all round over the clicked-down basket, the locking arm's wire in both grooves, both grippers
    clear and everything still for the last second."""
    return _update(objects)[1]


def distance(objects) -> float:
    g = geometry(objects)
    return float(max(g["basket_up_mm"], 0.0) + max(g["cover_gap_max_mm"], 0.0) + abs(90.0 - g["arm_deg"])) / 1000


def score(objects) -> float:
    """0.25 cover resting on the container over the basket; 0.5 cover seated; 0.75 locking arm in both grooves; 1.0
    with both grippers clear and everything still for 1 s. 0 for good if the base was moved or tipped, a part left the counter or a gripper
    reached into the filter basket."""
    return _update(objects)[0]


def demo():
    """The scripted demo needs both arms and timed pushes: run ``python demo.py task.py demo_out``."""
    return []
