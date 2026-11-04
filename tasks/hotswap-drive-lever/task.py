"""On the station table, in front of the right arm: a grey open-top drive bay (146 x 86 x 42 mm, 2.5 kg) whose mouth
faces the arms, and a blue 2.5-inch drive carrier (120 x 70 x 22 mm, 300 g) resting in the bay's channel, pulled
40 mm out of its mouth. An orange cam handle (91 mm bar, 10 x 8 mm section, with a 12 x 12 x 27 mm pull tab standing
on its free end) is hinged about a vertical pin on the carrier's top, 6 mm behind its front face and 10 mm in from
its left side. The handle stands open, pointing straight back at the arms. Its short end (16 mm past the pin) is the
hook. A spring-loaded connector plunger at the carrier's rear meets the bay's back wall over the last 8 mm.

Closing the handle swings it 90 degrees counter-clockwise (seen from above) until it lies across the carrier's
front. Near the end of the carrier's travel, the hook then swings out over the bay's low left wall and behind a
tall catch post at the bay's mouth. It levers the carrier the last few millimetres in against the plunger, and the
handle latches shut. A carrier pushed in by hand and let go springs back out. If the handle is closed before the
carrier is in, the hook hits the front of the catch post and the carrier cannot go in.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Hinge, PartState, Slider, on_reset, on_step, part, set_hinge_target  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Slide the blue drive carrier fully into the grey bay with its orange handle open, then swing the "
               "handle closed across the carrier's front until it latches, so the drive stays seated.")

GREY = (0.55, 0.56, 0.58)
DARK = (0.25, 0.26, 0.28)
BLUE = (0.10, 0.30, 0.90)
ORANGE = (1.00, 0.50, 0.00)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- the bay: origin at the centre of its mouth on the table; x into the bay, y left -------------------------------
FLOOR_Z = 0.004
WALL_IN = 0.036          # inner faces of the side walls at y = +-36 mm: 1 mm clear of the 70 mm carrier each side
POST_X1 = 0.010          # the catch post spans x 0..10 mm at the mouth, y 36..44 mm, up to 42 mm
BAY_BOXES = [
    _mm(0, 146, -42, 44, 0, 4, GREY),          # floor
    _mm(0, 140, -42, -36, 4, 24, GREY),        # right wall (low: the closed handle passes over it)
    _mm(10, 140, 36, 42, 4, 24, GREY),         # left wall (low: the hook swings in over it)
    _mm(0, 10, 36, 44, 4, 42, DARK),           # catch post at the mouth, on the left
    _mm(140, 146, -42, 42, 4, 30, GREY),       # back wall (the connector side)
]
BAY_MASS = 2.5

# -- the carrier: origin at the centre of its front face, on its underside; x toward its rear ----------------------
SLED_L, SLED_W, SLED_H = 0.120, 0.070, 0.022
SLED_MASS = 0.300
PIVOT = (0.006, 0.025, 0.027)    # the handle's vertical pin: 6 mm behind the front face, 10 mm in from the left side
HOOK_R = 0.016                   # the hook reaches 16 mm past the pin
ARM_L = 0.075                    # the bar reaches 75 mm the other way
TAB_R = 0.069                    # the pull tab's centre, 69 mm from the pin
BAR_HALF_W = 0.005
CLOSED = math.pi / 2             # handle angle when it lies across the front (limit)
LATCH_AT = math.radians(85)      # it latches from here on
LATCH_TARGET = math.radians(100) # a latched handle is driven into its 90-degree stop
HINGE_K, HINGE_C = 2.0, 0.05     # N m/rad, N m s/rad

# The handle at angle 0 (open): a 10 mm wide bar along x through the pin, from 75 mm in front of it to 16 mm behind,
# 1 mm above the carrier's top, and the pull tab standing on its front end.
_px, _py = PIVOT[0] * 1000, PIVOT[1] * 1000
HANDLE = Hinge("handle", pivot=PIVOT,
               boxes=(_mm(_px - 75, _px + 16, _py - 5, _py + 5, 23, 31, ORANGE),
                      _mm(_px - 75, _px - 63, _py - 6, _py + 6, 31, 50, ORANGE)),
               lower=0.0, upper=CLOSED, stiffness=HINGE_K, damping=HINGE_C, target=0.0, mass=0.030, axis="Z")

# Connector plunger: a 24 x 10 mm block reaching 19 mm behind the carrier's rear at rest. It meets the back wall
# (x 140 mm in the bay) once the carrier's front is 1 mm inside the mouth, and bottoms out after 9 mm (front at 10 mm).
PLUNGER_TRAVEL = 0.009
PLUNGER_RATE = 800.0
PLUNGER = Slider("plunger", rest=(0.1295, 0.0, 0.011), boxes=(_mm(120, 139, -12, 12, 6, 16, DARK),),
                 travel=PLUNGER_TRAVEL, rate=PLUNGER_RATE, damping=8.0, mass=0.005, axis="X", sign=-1)

SLED_BOXES = [_mm(0, 120, -35, 35, 0, 22, BLUE)]

# -- fixed layout (world frame: x forward from the arms, y left, metres) -------------------------------------------
BAY_XY = (0.30, -0.15)
START_F = -0.040                 # the carrier's front face starts 40 mm out of the bay's mouth
BAY = part("bay", pos=(*BAY_XY, TOP), yaw=0.0, mass=BAY_MASS, boxes=BAY_BOXES, color=GREY, friction=0.6)
SLED = part("carrier", pos=(BAY_XY[0] + START_F, BAY_XY[1], TOP + FLOOR_Z), yaw=0.0, mass=SLED_MASS,
            boxes=SLED_BOXES, sliders=(PLUNGER,), hinges=(HANDLE,), color=BLUE, friction=0.3)
OBJECTS = [BAY, SLED]

# -- where things are when it works (front face position f, bay frame) ---------------------------------------------
SEAT_F = POST_X1 + BAR_HALF_W - PIVOT[0]    # 9 mm: the closed hook's face against the catch post's back
STOP_F = 0.001 + PLUNGER_TRAVEL             # 10 mm: the plunger bottoms out
SPRING_F = 0.001                            # the plunger meets the back wall
ENGAGE_F = -0.008                           # from here on, closing the handle catches the hook behind the post

# -- scoring thresholds --------------------------------------------------------------------------------------------
SEATED_F = SEAT_F - 0.002        # seated: the front face within 2 mm of its latched depth
OPEN_MAX = math.radians(30)      # "still open" while inserting
HOOKED = math.radians(45)        # the hook is behind the post
SHUT = LATCH_AT                  # closed and latched
BAY_MOVE_LIMIT = 0.020
BAY_TURN_LIMIT = 10.0
LIFT_LIMIT = 0.012               # carrier's underside more than 12 mm off the bay floor: lifted out of the channel
TILT_LIMIT = 15.0
STILL_CALLS = 20

_STATE: dict = {}
_MECH: dict = {"latched": False}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(bay0=None, yaw0=None, failed=False, engaged=False, home=False, hooked=False, recent=[])
    _MECH["latched"] = False
    if environment is not None:
        set_hinge_target(environment, "carrier", "handle", 0.0)


_reset_state()


@on_step
def _handle_latch(environment) -> None:
    """The handle turns freely (its drive holds it where it was left) until it reaches 85 degrees; then it latches
    and is held against its 90-degree stop."""
    from partkit import _part_state

    angle = _part_state(environment, SLED).hinges["handle"]
    _MECH["latched"] |= angle >= LATCH_AT
    set_hinge_target(environment, "carrier", "handle", LATCH_TARGET if _MECH["latched"] else angle)


def _yaw(p: PartState) -> float:
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def carrier_in_bay(bay: PartState, sled: PartState):
    """The carrier's front-face position f along the bay (m, 0 at the mouth, positive inside), its sideways offset,
    its underside's height above the bay floor, and its tilt relative to the bay (deg)."""
    c = bay.to_local(sled.pos)
    rel = bay.rotation().T @ sled.rotation()
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, rel[2, 2]))))
    return float(c[0]), float(c[1]), float(c[2] - FLOOR_Z), tilt


def handle_angle(sled: PartState) -> float:
    return float(sled.hinges.get("handle", 0.0))


def _update(objects) -> tuple[float, bool]:
    bay, sled = objects["bay"], objects["carrier"]
    if _STATE["bay0"] is None:
        _STATE["bay0"], _STATE["yaw0"] = bay.pos.copy(), _yaw(bay)
    moved = float(np.linalg.norm((bay.pos - _STATE["bay0"])[:2]))
    turned = abs(math.degrees((_yaw(bay) - _STATE["yaw0"] + math.pi) % (2 * math.pi) - math.pi))
    f, _, lift, tilt = carrier_in_bay(bay, sled)
    _STATE["failed"] |= moved > BAY_MOVE_LIMIT or turned > BAY_TURN_LIMIT or lift > LIFT_LIMIT or tilt > TILT_LIMIT
    if _STATE["failed"]:
        return 0.0, False
    angle = handle_angle(sled)
    _STATE["engaged"] |= f >= ENGAGE_F and angle < OPEN_MAX
    _STATE["home"] |= f >= SEATED_F
    _STATE["hooked"] |= f >= ENGAGE_F and angle >= HOOKED
    recent = _STATE["recent"]
    recent.append(f)
    del recent[:-STILL_CALLS]
    still = len(recent) == STILL_CALLS and max(recent) - min(recent) < 0.001
    if f >= SEATED_F and angle >= SHUT:
        return 1.0, bool(still)
    if _STATE["hooked"]:
        return 0.75, False
    if _STATE["home"]:
        return 0.5, False
    return (0.25 if _STATE["engaged"] else 0.0), False


def success(objects) -> bool:
    """Carrier seated (front face within 2 mm of its latched depth) with the handle shut (85 degrees or more), still
    for 1 s, the bay in place."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 carrier pushed in to where the hook can catch (front face 8 mm or less short of the mouth) with the
    handle still open (under 30 degrees); 0.5 carrier pushed home (front face 7 mm or more inside); 0.75 handle swung
    45 degrees or more with the carrier that far in (hook behind the post); 1.0 carrier seated with the handle shut at
    the end. 0 if the bay ever moved 2 cm or turned 10 degrees, or the carrier was lifted 12 mm off the bay floor or
    tilted 15 degrees."""
    return _update(objects)[0]


def distance(objects) -> float:
    """Metres still to go: the carrier's front face to its seated depth plus the pull tab's arc to the shut angle."""
    f, _, _, _ = carrier_in_bay(objects["bay"], objects["carrier"])
    return max(0.0, SEAT_F - f) + TAB_R * max(0.0, CLOSED - handle_angle(objects["carrier"]))
