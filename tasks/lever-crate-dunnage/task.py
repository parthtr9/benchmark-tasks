"""On the station table, in front of the arms: a closed blue crate (220 mm deep, 260 mm wide, 100 mm tall, 2 kg) with
a lifting pocket (50 mm wide, 35 mm deep, 16 mm tall) cut into the middle of its front bottom edge, a red pry lever
pinned on a dark grey fulcrum stand in front of the crate, and a light wooden dunnage block (120 x 36 x 14 mm, 100 g)
lying on the table to the left. The lever's flat toe sits in the crate's pocket; its slanted handle rises toward the
arms and ends in a 24 mm square knob. Pressing the knob down turns the lever on its pin, and the toe lifts the
crate's front edge by up to about 25 mm, while its back edge stays on the table. Let go, the crate's weight and the
lever's return spring push the handle back up and the crate drops back onto the table.

The robot must press the lever down with one arm, hold it, slide the block under the crate's raised front edge with
the other arm, and then let go of the lever so that the crate rests on the block. The crate is too wide to grip and
its front edge has no gap to push the block into until it is lifted.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Hinge, PartState, on_reset, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Press the red lever's handle down to lift the front of the blue crate and hold it there, slide the "
               "wooden block under the crate's raised front edge with the other arm, then let go of the lever so the "
               "crate rests on the block.")

BLUE = (0.16, 0.36, 0.70)
DARK = (0.22, 0.23, 0.25)
RED = (0.85, 0.12, 0.10)
WOOD = (0.86, 0.70, 0.45)


def _mm(x0, x1, y0, y1, z0, z1, color, rot=None):
    spec = (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                     (z1 - z0) / 1000), color)
    return spec if rot is None else (*spec, rot)


# -- the crate: origin at the middle of its front bottom edge; x into the crate (away from the arms), y left --------
CRATE_D, CRATE_W, CRATE_H = 0.220, 0.260, 0.100
CRATE_MASS = 2.0
POCKET_HALF_W, POCKET_D, POCKET_H = 0.025, 0.035, 0.016
CRATE_BOXES = [
    _mm(0, 220, -130, 130, 16, 100, BLUE),          # body above the pocket
    _mm(0, 220, 25, 130, 0, 16, BLUE),              # bottom rail left of the pocket
    _mm(0, 220, -130, -25, 0, 16, BLUE),            # bottom rail right of the pocket
    _mm(35, 220, -25, 25, 0, 16, BLUE),             # behind the pocket
]

# -- the lever stand: origin on the table under the pin; x toward the crate. The lever turns about y through the pin.
# Angle 0: the toe rests flat, 1 mm under the pocket's roof (its upper stop). Negative angles press the handle down
# and lift the toe. A return spring pulls it toward its stop. ----------------------------------------------------------
PIVOT = (0.0, 0.0, 0.010)
TOE_X = (0.004, 0.075)               # the toe runs 4..75 mm in front of the pin, 3..15 mm above the table, 32 mm wide
TOE_Z = (0.003, 0.015)
TOE_HALF_W = 0.016
CRATE_GAP = 0.045                    # the crate's front face is 45 mm in front of the pin: the toe reaches 30 mm in
HANDLE_DX, HANDLE_DZ = -0.180, 0.126  # the handle bar's centre line rises 35 degrees, to 180 mm behind the pin
BAR_T, BAR_W = 0.012, 0.016
KNOB = 0.024                          # knob 24 x 24 mm, 32 mm tall, on the handle's end
KNOB_X, KNOB_Z = -0.186, 0.136        # knob centre, relative to the pin
GRIP_Z = KNOB_Z + 0.016 - 0.004       # where the demo holds it: 4 mm below the knob's top, relative to the pin
PRESSED = -0.33                       # rad: the lower stop
SPRING_K, SPRING_TARGET = 1.5, 0.25   # N m/rad toward +0.25 rad: holds the lever on its upper stop when free
LEVER_MASS = 0.20


def _bar():
    """The handle bar: from the pin up and back to the knob, as one box rotated about y."""
    a = math.atan2(HANDLE_DZ, -HANDLE_DX)            # 35 degrees above the -x direction
    length = math.hypot(HANDLE_DX, HANDLE_DZ)
    # A box along x, turned about +y by +a: its -x end rises (rotation about y takes -x toward +z).
    quat = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)
    centre = (PIVOT[0] + HANDLE_DX / 2, 0.0, PIVOT[2] + HANDLE_DZ / 2)
    return (centre, (length, BAR_W, BAR_T), RED, quat)


LEVER = Hinge(
    "lever",
    pivot=PIVOT,
    boxes=(
        _mm(1000 * TOE_X[0], 1000 * TOE_X[1], -16, 16, 1000 * TOE_Z[0], 1000 * TOE_Z[1], RED),     # toe plate
        _mm(-6, 6, -12, 12, 4, 16, RED),                                                            # hub round the pin
        _bar(),
        _mm(1000 * (KNOB_X - KNOB / 2), 1000 * (KNOB_X + KNOB / 2), -12, 12,
            1000 * (PIVOT[2] + KNOB_Z - 0.016), 1000 * (PIVOT[2] + KNOB_Z + 0.016), DARK),          # knob
    ),
    lower=PRESSED, upper=0.0, stiffness=SPRING_K, damping=0.05, target=SPRING_TARGET, mass=LEVER_MASS, axis="Y")

STAND_BOXES = [
    _mm(-14, 14, 18, 30, 0, 22, DARK), _mm(-14, 14, -30, -18, 0, 22, DARK),     # cheeks either side of the lever
    _mm(-30, 10, 30, 48, 0, 5, DARK), _mm(-30, 10, -48, -30, 0, 5, DARK),       # feet
    _mm(16, 40, -30, 30, 0, 2, DARK),                                           # floor plate under the toe
]
STAND_MASS = 2.0

BLOCK_L, BLOCK_W, BLOCK_H = 0.120, 0.036, 0.014
BLOCK_MASS = 0.10
BLOCK_BOXES = [((0.0, 0.0, BLOCK_H / 2), (BLOCK_L, BLOCK_W, BLOCK_H), WOOD)]   # origin at the middle of its underside

# -- fixed layout (world frame: x forward from the arms, y left, metres) ---------------------------------------------
CRATE_FRONT_X = 0.40
PIN_X = CRATE_FRONT_X - CRATE_GAP
BLOCK_START = (0.28, 0.17)            # lying along x, to the left of the lever
BLOCK_Y = 0.090                       # the lane where the block goes under the crate (crate frame y)
DEPTH = 0.040                         # the demo slides the block's front end 40 mm under the crate's front face
BLOCK_GRIP_BACK = 0.020               # and holds the block 20 mm from its back end
CRATE = part("crate", pos=(CRATE_FRONT_X, 0.0, TOP), mass=CRATE_MASS, boxes=CRATE_BOXES, color=BLUE, friction=0.8)
STAND = part("lever_stand", pos=(PIN_X, 0.0, TOP), mass=STAND_MASS, boxes=STAND_BOXES, hinges=(LEVER,), color=DARK,
             friction=1.0)
BLOCK = part("block", pos=(*BLOCK_START, TOP), mass=BLOCK_MASS, boxes=BLOCK_BOXES, color=WOOD, friction=0.8)
OBJECTS = [CRATE, STAND, BLOCK]

# -- scoring thresholds ------------------------------------------------------------------------------------------------
LIFTED = 0.012           # front edge 12 mm or more off the table: the lever has been used
REST_LOW, REST_HIGH = 0.010, 0.032   # resting on the 14 mm block: front edge 10..32 mm off the table
SEAT_GAP = 0.003         # and the crate's underside within 3 mm of the block's top above the block's front end
UNDER_MIN = 0.025        # the block's front end at least 25 mm behind the crate's front face
UNDER_MAX = 0.120        # and at most its own length
LANE = CRATE_W / 2 - BLOCK_W / 2   # the block's centre line within the crate's width
RELEASED = -0.10         # lever angle above this: let go (the toe hangs clear of the pocket's roof)
FLAT_TILT = 10.0         # deg: the block lies flat
FLAT_LIFT = 0.004        # its underside within 4 mm of the table
CRATE_MOVE_LIMIT = 0.030  # the front edge's middle moved more than 3 cm across the table: dragged or shoved
CRATE_TURN_LIMIT = 10.0  # deg of yaw
CRATE_ROLL_LIMIT = 10.0  # deg of sideways tilt
TIPPED = 0.050           # front edge more than 50 mm up: tipped, not levered
STAND_MOVE_LIMIT = 0.030
STAND_TURN_LIMIT = 10.0
STILL_CALLS = 20

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(crate0=None, yaw0=None, stand0=None, stand_yaw0=None, failed=False, lifted=False, inserted=False,
                  recent=[])


_reset_state()


def _yaw(p: PartState) -> float:
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def _turn(a: float, b: float) -> float:
    return abs(math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi))


def front_height(crate: PartState) -> float:
    """Height (m) of the middle of the crate's front bottom edge above the table."""
    return float(crate.pos[2] - TOP)


def crate_roll(crate: PartState) -> float:
    """Sideways tilt (deg): the crate's y axis out of the horizontal."""
    return math.degrees(math.asin(max(-1.0, min(1.0, crate.rotation()[2, 1]))))


def block_under(crate: PartState, block: PartState) -> tuple[float, float]:
    """How far (m) the block's front end lies behind the crate's front face, and its centre line's offset from the
    crate's middle, both in the crate's frame."""
    front = block.to_world((BLOCK_L / 2, 0.0, BLOCK_H / 2))
    centre = block.to_world((0.0, 0.0, BLOCK_H / 2))
    return float(crate.to_local(front)[0]), float(crate.to_local(centre)[1])


def block_flat(block: PartState) -> bool:
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, block.rotation()[2, 2]))))
    return tilt < FLAT_TILT and float(block.pos[2] - TOP) < FLAT_LIFT


def seat_gap(crate: PartState, block: PartState) -> float:
    """Height (m) of the crate's underside above the block's top, measured over the block's front end."""
    front = block.to_world((BLOCK_L / 2, 0.0, BLOCK_H))
    local = crate.to_local(front)
    underside = crate.to_world((local[0], local[1], 0.0))
    return float(underside[2] - front[2])


def lever_angle(stand: PartState) -> float:
    return float(stand.hinges.get("lever", 0.0))


def _is_under(crate, block) -> bool:
    depth, lane = block_under(crate, block)
    return UNDER_MIN <= depth <= UNDER_MAX and abs(lane) <= LANE


def _update(objects) -> tuple[float, bool]:
    crate, stand, block = objects["crate"], objects["lever_stand"], objects["block"]
    if _STATE["crate0"] is None:
        _STATE.update(crate0=crate.pos.copy(), yaw0=_yaw(crate), stand0=stand.pos.copy(), stand_yaw0=_yaw(stand))
    h = front_height(crate)
    moved = float(np.linalg.norm((crate.pos - _STATE["crate0"])[:2]))
    _STATE["failed"] |= (moved > CRATE_MOVE_LIMIT or _turn(_yaw(crate), _STATE["yaw0"]) > CRATE_TURN_LIMIT
                         or abs(crate_roll(crate)) > CRATE_ROLL_LIMIT or h > TIPPED
                         or float(np.linalg.norm((stand.pos - _STATE["stand0"])[:2])) > STAND_MOVE_LIMIT
                         or _turn(_yaw(stand), _STATE["stand_yaw0"]) > STAND_TURN_LIMIT)
    if _STATE["failed"]:
        return 0.0, False
    under = _is_under(crate, block)
    _STATE["lifted"] |= h >= LIFTED
    _STATE["inserted"] |= under and h >= LIFTED
    resting = (under and REST_LOW <= h <= REST_HIGH and block_flat(block)
               and abs(seat_gap(crate, block)) <= SEAT_GAP)
    recent = _STATE["recent"]
    recent.append(np.r_[crate.pos, block.pos])
    del recent[:-STILL_CALLS]
    still = len(recent) == STILL_CALLS and float(np.ptp(np.array(recent), axis=0).max()) < 0.001
    if resting and lever_angle(stand) >= RELEASED:
        return 1.0, bool(still)
    if resting:
        return 0.75, False
    if _STATE["inserted"]:
        return 0.5, False
    return (0.25 if _STATE["lifted"] else 0.0), False


def success(objects) -> bool:
    """The crate rests on the block (its underside on the block's top, front edge 10-32 mm up, the block lying flat
    with its front end 25-120 mm under the crate), the lever let go, crate and block still for 20 scorer calls."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 the crate's front edge was lifted 12 mm or more; 0.5 the block was slid 25 mm or more under the lifted
    front edge; 0.75 at the end the crate rests on the block but the lever is still held down; 1.0 at the end the crate
    rests on the block with the lever let go. 0 if the crate's front edge ever moved 3 cm across the table, turned 10
    degrees, tilted 10 degrees sideways or rose more than 50 mm, or the lever stand moved 3 cm or turned 10 degrees."""
    return _update(objects)[0]


def distance(objects) -> float:
    """Metres to go: the block's front end to 40 mm under the crate's front face, plus the crate's front edge to the
    middle of its resting range."""
    crate, block = objects["crate"], objects["block"]
    goal = crate.to_world((0.040, BLOCK_Y, 0.0))
    front = block.to_world((BLOCK_L / 2, 0.0, 0.0))
    return float(np.linalg.norm((front - goal)[:2]) + abs(front_height(crate) - (REST_LOW + REST_HIGH) / 2))
