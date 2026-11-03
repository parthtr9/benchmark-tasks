"""On the station table: a grey plastic case (260 x 80 x 40 mm) lying along the arms' forward direction, its blue lid
(250 x 70 x 6 mm) set 13 mm down inside its rim and held by two slide latches, one at each end. Each latch has an
orange tab (30 mm tall) standing up from the lid near its end; sliding a tab 6 mm toward the lid's centre pulls its tongue out of a
slot in the case's end wall. The robot must release both latches at once, lift the lid out with both arms, and set it
down flat on the table to the right of the case.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, Slider, on_reset, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Grip the two orange tabs on the blue lid, slide both toward the middle of the lid at the same time to "
               "release the latches, lift the lid out of the grey case, and set it down flat on the table to the right "
               "of the case.")

GREY = (0.55, 0.56, 0.58)
BLUE = (0.10, 0.30, 0.90)
ORANGE = (1.00, 0.50, 0.00)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- the case: origin under its centre on the table; long axis x; rim at z = 40 mm ------------------------------------
RIM_Z = 0.040
CASE_BOXES = [
    _mm(-130, 130, -40, 40, 0, 6, GREY),                                            # floor
    _mm(-130, 130, 36, 40, 6, 40, GREY), _mm(-130, 130, -40, -36, 6, 40, GREY),     # long walls
    _mm(-126, 126, 32, 36, 6, 21, GREY), _mm(-126, 126, -36, -32, 6, 21, GREY),     # ledges the lid rests on
]
for _s in (1, -1):  # end walls, each with a 24 x 4 mm slot (z 23-27 mm) for its latch tongue
    _x0, _x1 = sorted((_s * 126, _s * 130))
    CASE_BOXES += [_mm(_x0, _x1, 12, 36, 6, 40, GREY), _mm(_x0, _x1, -36, -12, 6, 40, GREY),
                   _mm(_x0, _x1, -12, 12, 6, 23, GREY), _mm(_x0, _x1, -12, 12, 27, 40, GREY)]

# -- the lid, in the frame it shares with the case when seated: a plate z 21-27 mm, 13 mm below the rim -------------
LID_BOXES = [_mm(-125, 125, -35, 35, 21, 27, BLUE)]
LID_BOTTOM = 0.021
TAB_X, TAB_TOP = 0.098, 0.057     # tab centre 98 mm from the lid's centre; its top 30 mm above the lid
LATCH_TRAVEL = 0.006


def _latch(name, s):
    """A latch at end ``s`` (+1 or -1): a 4 x 30 x 30 mm tab on the lid and a 30 x 20 x 3 mm tongue reaching through
    the end wall's slot. Sliding the tab toward the lid's centre (6 mm) pulls the tongue clear of the wall. The tab is
    tall so a gripper can hold its upper part without its fingertips touching the lid."""
    tab = _mm(*sorted((s * 96, s * 100)), -15, 15, 27, 57, ORANGE)
    tongue = _mm(*sorted((s * 100, s * 130)), -10, 10, 23.5, 26.5, ORANGE)
    return Slider(name, rest=(s * TAB_X, 0.0, 0.025), boxes=(tab, tongue), travel=LATCH_TRAVEL, rate=300.0,
                  damping=10.0, axis="X", sign=-s)


LATCHES = (_latch("latch_far", 1), _latch("latch_near", -1))

# -- fixed layout (world frame: x forward from the arms, y left, metres) ----------------------------------------------
CASE_XY, CASE_YAW = (0.25, 0.08), 0.0
PLACE_DY = -0.10   # the demo sets the lid down 10 cm to the right of the case's centreline (it must clear 8 cm)
CASE = part("case", pos=(*CASE_XY, TOP), yaw=CASE_YAW, mass=1.5, boxes=CASE_BOXES, color=GREY, friction=0.6)
LID = part("lid", pos=(*CASE_XY, TOP), yaw=CASE_YAW, mass=0.120, boxes=LID_BOXES, sliders=LATCHES, color=BLUE,
           friction=0.5)
OBJECTS = [CASE, LID]

# -- scoring -----------------------------------------------------------------------------------------------------------
CASE_MOVE_LIMIT = 0.03
RELEASED = 0.004   # a latch counts as released once its tab has slid 4 mm in: its tongue is then clear of the wall
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(case0=None, one=False, both=False, lifted=False, failed=False, recent=[])


_reset_state()


def latch_travel(lid: PartState):
    """How far each tab has slid toward the lid's centre (m)."""
    return {name: abs(v) for name, v in lid.sliders.items()}


def lid_in_case(case: PartState, lid: PartState):
    """Lid origin in the case frame (0 when seated), its tilt (deg) and its height above the table (m)."""
    c = case.to_local(lid.pos)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, (case.rotation().T @ lid.rotation())[2, 2]))))
    return c, tilt, float(lid.pos[2] - TOP)


def _update(objects) -> tuple[float, bool]:
    case, lid = objects["case"], objects["lid"]
    if _STATE["case0"] is None:
        _STATE["case0"] = case.pos.copy()
    _STATE["failed"] |= float(np.linalg.norm(case.pos - _STATE["case0"])) > CASE_MOVE_LIMIT
    if _STATE["failed"]:
        return 0.0, False
    travel = latch_travel(lid)
    released = [v >= RELEASED for v in travel.values()]
    _STATE["one"] |= any(released)
    _STATE["both"] |= all(released)
    c, tilt, height = lid_in_case(case, lid)
    # Lifted clear: the lid's underside at least 10 mm above the case's rim.
    _STATE["lifted"] |= bool(_STATE["both"] and c[2] + LID_BOTTOM >= RIM_Z + 0.010)
    # Set down: flat on the table (underside within 5 mm of it, tilted under 10 degrees) and clear of the case's
    # footprint (its centre at least 8 cm to the side of the case's centreline).
    down = abs(height + LID_BOTTOM) < 0.005 and tilt < 10 and abs(c[1]) >= 0.080
    recent = _STATE["recent"]
    recent.append(c.copy())
    del recent[:-20]
    still = len(recent) == 20 and all(np.linalg.norm(p - c) < 0.001 for p in recent)
    if _STATE["lifted"] and down:
        return 1.0, bool(still)
    if _STATE["lifted"]:
        return 0.75, False
    if _STATE["both"]:
        return 0.5, False
    return (0.25 if _STATE["one"] else 0.0), False


def success(objects) -> bool:
    """Lid set down flat on the table beside the case after being lifted out, still for 1 s, case in place."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 one latch released; 0.5 both released at once; 0.75 lid lifted 1 cm clear of the rim; 1.0 lid set down
    flat beside the case. 0 if the case ever moved 3 cm."""
    return _update(objects)[0]
