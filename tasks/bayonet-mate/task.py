"""On the station table: a grey plastic box (120 x 70 x 25 mm) on a short stand right of centre, with a round socket
near its front end whose 26 mm opening has two slots; and a blue bayonet cap (24 mm barrel, square grip head, two
orange lugs) standing on the table to the left. The robot must line the lugs up with the slots, push the cap down
against the spring under it, and turn it a quarter turn counterclockwise (seen from above) so the lugs lock under the
socket's roof.

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
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Pick up the blue cap by its square head, line its orange lugs up with the slots in the round socket on "
               "the grey box, push it down, and turn it a quarter turn counterclockwise until it stops.")

GREY = (0.55, 0.56, 0.58)
BLUE = (0.10, 0.30, 0.90)
ORANGE = (1.00, 0.50, 0.00)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- the box: socket axis 34 mm in front of its centre, a 4 mm roof (z 8.5-12.5 mm) over a chamber ------------------
S = 34
SOCKET = np.array([S / 1000, 0.0])
BOX_BOXES = [
    _mm(-60, 60, -35, 35, -12.5, -6.5, GREY),
    _mm(-60, S - 24, -35, 35, -6.5, 12.5, GREY), _mm(S + 24, 60, -35, 35, -6.5, 12.5, GREY),
    _mm(S - 24, S + 24, 24, 35, -6.5, 12.5, GREY), _mm(S - 24, S + 24, -35, -24, -6.5, 12.5, GREY),
    # Roof: a 26 mm opening plus lug slots along the box's x axis, so the lugs enter only when lined up.
    _mm(S - 24, S + 24, 13, 24, 8.5, 12.5, ORANGE), _mm(S - 24, S + 24, -24, -13, 8.5, 12.5, ORANGE),
    _mm(S + 13, S + 24, 4, 13, 8.5, 12.5, GREY), _mm(S + 13, S + 24, -13, -4, 8.5, 12.5, GREY),
    _mm(S - 24, S - 13, 4, 13, 8.5, 12.5, GREY), _mm(S - 24, S - 13, -13, -4, 8.5, 12.5, GREY),
    _mm(S + 19, S + 24, -4, 4, 8.5, 12.5, GREY), _mm(S - 24, S - 19, -4, 4, 8.5, 12.5, GREY),
    # Stops under the roof: the cap turns counterclockwise only, and stops just past 90 degrees.
    _mm(S - 8, S - 5, 13, 22, -6.5, 8.5, GREY), _mm(S + 5, S + 8, -22, -13, -6.5, 8.5, GREY),
]
ROOF_UNDERSIDE = 0.0085
# A spring plate under the socket holds a set-down cap with its lugs inside the roof slots, where they cannot turn;
# pushing the cap 4 mm down clears the roof, and after the turn the spring holds the lugs up against its underside.
PLATE_REST = 0.0055
# 5 mm thick: a thin plate lets a hard-pushed cap tunnel through it.
PLATE = Slider("plate", rest=(S / 1000, 0.0, PLATE_REST), boxes=(_mm(S - 11, S + 11, -11, 11, 0.5, 5.5, ORANGE),),
               travel=0.006, rate=1500.0, damping=20.0)
CAP_BOXES = [_mm(-10, 10, -10, 10, 10, 30, BLUE),
             _mm(12, 17, -3, 3, -8, -3, ORANGE), _mm(-17, -12, -3, 3, -8, -3, ORANGE)]
CAP_CYLINDERS = [((0.0, 0.0, 0.0), 0.012, 0.020, BLUE)]
LUG_TOP = -0.003                     # in the cap frame
SEATED_Z = ROOF_UNDERSIDE - LUG_TOP  # cap origin above the box origin with its lugs against the roof's underside

# -- fixed layout (world frame: x forward from the arms, y left, metres) ----------------------------------------------
STAND_H = 0.020
BOX_XY, BOX_YAW = (0.21, -0.05), math.radians(-45)
CAP_XY, CAP_YAW = (0.12, 0.15), math.radians(-30)

BOX = part("socket_box", pos=(*BOX_XY, TOP + STAND_H + 0.0125), yaw=BOX_YAW, mass=0.150, boxes=BOX_BOXES,
           sliders=(PLATE,), color=GREY, friction=0.6)
CAP = part("cap", pos=(*CAP_XY, TOP + 0.010), yaw=CAP_YAW, mass=0.030, boxes=CAP_BOXES, cylinders=CAP_CYLINDERS,
           color=BLUE, friction=0.4)
STAND = Box("stand", pos=(*BOX_XY, TOP + STAND_H / 2), size=(0.040, 0.050, STAND_H), yaw=BOX_YAW, color=GREY)
OBJECTS = [STAND, BOX, CAP]

# -- scoring -----------------------------------------------------------------------------------------------------------
BOX_MOVE_LIMIT = 0.03
BOX_TILT_LIMIT = math.radians(20)
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(cap_z0=None, box0=None, yaw0=None, lifted=False, inserted=False, twisted=False, failed=False,
                  recent=[])


_reset_state()


def cap_in_box(box: PartState, cap: PartState):
    """Cap origin in the box frame, its distance from the socket axis, lug angle (deg, modulo 180) and tilt (deg)."""
    c = box.to_local(cap.pos)
    rel = box.rotation().T @ cap.rotation()
    lateral = float(np.hypot(c[0] - SOCKET[0], c[1] - SOCKET[1]))
    angle = (math.degrees(math.atan2(rel[1, 0], rel[0, 0])) + 90) % 180 - 90
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, rel[2, 2]))))
    return c, lateral, angle, tilt


def _update(objects) -> tuple[float, bool]:
    box, cap = objects["socket_box"], objects["cap"]
    yaw = math.atan2(box.rotation()[1, 0], box.rotation()[0, 0])
    if _STATE["cap_z0"] is None:
        _STATE.update(cap_z0=float(cap.pos[2]), box0=box.pos.copy(), yaw0=yaw)
    moved = float(np.linalg.norm(box.pos - _STATE["box0"]))
    turned = abs((math.degrees(yaw - _STATE["yaw0"]) + 180) % 360 - 180)
    _STATE["failed"] |= (moved > BOX_MOVE_LIMIT or box.rotation()[2, 2] < math.cos(BOX_TILT_LIMIT) or turned > 20)
    if _STATE["failed"]:
        return 0.0, False
    c, lateral, angle, tilt = cap_in_box(box, cap)
    centred = lateral < 0.003 and tilt < 10
    _STATE["lifted"] |= float(cap.pos[2]) - _STATE["cap_z0"] > 0.03
    inserted = centred and c[2] <= SEATED_Z + 0.0005
    _STATE["inserted"] |= bool(_STATE["lifted"] and inserted)
    _STATE["twisted"] |= bool(_STATE["inserted"] and centred and c[2] <= SEATED_Z + 0.0015 and abs(angle) > 45)
    locked = _STATE["twisted"] and centred and abs(angle) > 80 and abs(c[2] - SEATED_Z) < 0.0015
    # Still: over the last second (20 control steps) the cap stayed within 1 mm and 2 degrees. A 30 g part's
    # instantaneous velocity is too noisy in simulation to judge rest by.
    recent = _STATE["recent"]
    recent.append((c.copy(), abs(angle)))
    del recent[:-20]
    still = len(recent) == 20 and all(np.linalg.norm(p - c) < 0.001 and abs(a - abs(angle)) < 2 for p, a in recent)
    if locked:
        return 1.0, bool(still)
    if _STATE["twisted"]:
        return 0.75, False
    if _STATE["inserted"]:
        return 0.5, False
    return (0.25 if _STATE["lifted"] else 0.0), False


def success(objects) -> bool:
    """Locked: lugs turned more than 80 degrees past the slots, cap on the socket axis, lugs against the roof's
    underside, cap still (within 1 mm and 2 degrees over the last second), box in place; held 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 cap lifted 3 cm; 0.5 lugs pushed below the roof; 0.75 turned past 45 degrees while in; 1.0 locked. 0 if the
    box ever moved 3 cm, tilted 20 degrees or turned 20 degrees."""
    return _update(objects)[0]
