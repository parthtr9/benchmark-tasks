"""On the station table: a grey plastic linear rail (290 mm long, T-shaped top) lying loose on the midline, pointing
away from the arms, its open end nearest them, an orange end stop part way along and a plain handle beyond it; and a
blue carriage block (40 x 30 x 17 mm) with a matching T-shaped channel, standing on the table to the right. The robot must slide the carriage onto the
rail's open end and push it along until it clicks past the spring-loaded plunger and sits against the end stop.

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

INSTRUCTION = ("Slide the blue carriage onto the open end of the grey rail and push it along the rail until it clicks "
               "past the plunger and stops against the orange end stop.")

GREY = (0.55, 0.56, 0.58)
BLUE = (0.10, 0.30, 0.90)
ORANGE = (1.00, 0.50, 0.00)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- the rail: origin on its underside; x runs from the open end (-60 mm) to the end stop (55-60 mm) and a plain
#    handle the other arm can hold (60-230 mm), long enough that the two hands stay well apart -------------------------
OPEN_END, STOP_FACE = -0.060, 0.055
RAIL_BOXES = [
    _mm(-60, 230, -10, 10, 0, 8, GREY),         # base bar and handle
    _mm(-60, 60, -4, 4, 8, 14, GREY),           # stem
    _mm(-60, 55, -8, 8, 14, 18, GREY),          # head
    _mm(55, 60, -10, 10, 8, 30, ORANGE),        # end stop
]
HANDLE_X = 0.215
# The detent: a spring plunger on the rail's side, with a rounded 8 mm nose, between the head and the carriage's lips.
# It stands 2 mm into the path of the carriage's side wall, which pushes it sideways into the rail to pass (under 1 N);
# it springs back out behind the carriage, trapping it against the end stop. A sideways detent pushes the carriage
# sideways, which the grip resists; one the carriage pressed down would tip it between the gripper's pads.
PLUNGER_X, PLUNGER_R = 0.009, 0.004
PLUNGER = Slider("plunger", rest=(PLUNGER_X, 0.007, 0.013),
                 cylinders=(((PLUNGER_X, 0.007, 0.013), PLUNGER_R, 0.0014, ORANGE, "Z"),),
                 travel=0.0025, rate=300.0, damping=10.0, axis="Y")
CARRIAGE_LEN = 0.040
# -- the carriage, in the frame it shares with the rail when mounted (origin on the rail's underside) ---------------
# 1 mm of side play and 2 mm above and below the rail head: the carriage can tip about 5 degrees before it binds.
CARRIAGE_BOXES = [
    _mm(-20, 20, -15, 15, 20, 26, BLUE),                                      # top plate
    _mm(-20, 20, 9, 15, 9, 20, BLUE), _mm(-20, 20, -15, -9, 9, 20, BLUE),     # walls
    _mm(-20, 20, 5, 9, 9, 12, BLUE), _mm(-20, 20, -9, -5, 9, 12, BLUE),       # lips under the rail head
]
CARRIAGE_BOTTOM = 0.009
# Trapped: the carriage's rear edge is past the plunger (9 + 4 mm) and its front edge is at the stop (55 mm).
LOCK_X = (PLUNGER_X + PLUNGER_R + CARRIAGE_LEN / 2 - 0.0005, STOP_FACE - CARRIAGE_LEN / 2 + 0.0005)

# -- fixed layout (world frame: x forward from the arms, y left, metres) ----------------------------------------------
RAIL_XY, RAIL_YAW = (0.205, 0.0), 0.0
CARRIAGE_XY, CARRIAGE_YAW = (0.15, -0.15), 0.0

RAIL = part("rail", pos=(*RAIL_XY, TOP), yaw=RAIL_YAW, mass=0.150, boxes=RAIL_BOXES, sliders=(PLUNGER,), color=GREY,
            friction=0.5)
CARRIAGE = part("carriage", pos=(*CARRIAGE_XY, TOP - CARRIAGE_BOTTOM), yaw=CARRIAGE_YAW, mass=0.040,
                boxes=CARRIAGE_BOXES, color=BLUE, friction=0.3)
OBJECTS = [RAIL, CARRIAGE]

# -- scoring -----------------------------------------------------------------------------------------------------------
RAIL_MOVE_LIMIT = 0.03
RAIL_TURN_LIMIT = 15.0
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(z0=None, rail0=None, yaw0=None, lifted=False, failed=False, recent=[])


_reset_state()


def carriage_on_rail(rail: PartState, carriage: PartState):
    """Carriage origin in the rail frame, its yaw relative to the rail (deg) and its tilt (deg)."""
    c = rail.to_local(carriage.pos)
    rel = rail.rotation().T @ carriage.rotation()
    yaw = math.degrees(math.atan2(rel[1, 0], rel[0, 0]))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, rel[2, 2]))))
    return c, yaw, tilt


def _yaw(p: PartState) -> float:
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def _update(objects) -> tuple[float, bool]:
    rail, carriage = objects["rail"], objects["carriage"]
    if _STATE["z0"] is None:
        _STATE.update(z0=float(carriage.pos[2]), rail0=rail.pos.copy(), yaw0=_yaw(rail))
    moved = float(np.linalg.norm(rail.pos - _STATE["rail0"]))
    turned = abs((math.degrees(_yaw(rail) - _STATE["yaw0"]) + 180) % 360 - 180)
    _STATE["failed"] |= moved > RAIL_MOVE_LIMIT or turned > RAIL_TURN_LIMIT
    if _STATE["failed"]:
        return 0.0, False
    c, yaw, tilt = carriage_on_rail(rail, carriage)
    # On the rail: wrapped around the head (within 2 mm of its centreline and 1.5 mm of its height, square to it)
    # with at least 10 mm of the carriage over the rail.
    on = abs(c[1]) < 0.002 and abs(c[2]) < 0.0015 and abs((yaw + 90) % 180 - 90) < 5 and tilt < 5 \
        and c[0] + CARRIAGE_LEN / 2 >= OPEN_END + 0.010
    _STATE["lifted"] |= float(carriage.pos[2]) - _STATE["z0"] > 0.03
    # Only the lift is kept once reached; the rail milestones are judged on where the carriage is now, so a carriage
    # that comes off the rail loses them.
    engaged = bool(_STATE["lifted"] and on)
    halfway = engaged and c[0] >= 0.0
    plunger_out = rail.sliders.get("plunger", 0.0) > -0.001
    locked = halfway and LOCK_X[0] <= c[0] <= LOCK_X[1] and plunger_out
    recent = _STATE["recent"]
    recent.append(c.copy())
    del recent[:-20]
    still = len(recent) == 20 and all(np.linalg.norm(p - c) < 0.001 for p in recent)
    if locked and still:
        return 1.0, True
    if halfway:
        return 0.75, False
    if engaged:
        return 0.5, False
    return (0.25 if _STATE["lifted"] else 0.0), False


def success(objects) -> bool:
    """Trapped between the plunger and the end stop, on the rail, still for 1 s (within 1 mm), rail in place."""
    return _update(objects)[1]


def score(objects) -> float:
    """At the end of the run: 1.0 clicked past the plunger against the stop and still for 1 s; 0.75 on the rail past
    its midpoint; 0.5 on the rail; 0.25 the carriage was lifted 3 cm at some point. 0 if the rail ever moved 3 cm or
    turned 15 degrees."""
    return _update(objects)[0]
