"""On the station table: a light grey printed robot link (160 x 40 x 30 mm) lying loose, with an 8 mm steel shaft
sticking out sideways from a shoulder collar near one end; and a 22 mm square flanged bearing (9 mm bore) standing on
edge in a slot holder to the right. The robot must slide the bearing onto the shaft and press it along until it clicks
past the spring ball on the shaft and sits against the shoulder.

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

INSTRUCTION = ("Slide the bearing onto the end of the steel shaft that sticks out of the grey robot link and press it "
               "along the shaft until it clicks past the ball and sits against the shoulder.")

GREY = (0.55, 0.56, 0.58)
STEEL = (0.78, 0.79, 0.80)
ORANGE = (1.00, 0.50, 0.00)
BLUE = (0.10, 0.30, 0.90)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- the link: origin under its centre; the shaft (axis along +y) sticks out of its side at x = 60 mm, 15 mm up; the
#    bearing seats on the shoulder at y = 40 mm and the spring ball sits 10 mm beyond it ----------------------------
SHAFT_X, SHAFT_Z, SHAFT_R = 0.060, 0.015, 0.004
SHOULDER_Y, SHAFT_END_Y = 0.040, 0.065
LINK_BOXES = [_mm(-80, 80, -20, 20, 0, 30, GREY)]
# A 20 mm standoff collar puts the shoulder 20 mm out from the link's side, so the gripper's fingers (about 25 mm wide
# along the shaft) clear the link while the bearing seats.
LINK_CYLINDERS = [
    ((SHAFT_X, 0.030, SHAFT_Z), 0.007, 0.020, STEEL, "Y"),      # standoff collar, y 20-40 mm, its end the shoulder
    ((SHAFT_X, 0.0425, SHAFT_Z), SHAFT_R, 0.045, STEEL, "Y"),   # shaft, y 20-65 mm
]
# The detent: a spring ball (a rounded 4 mm nose) on the side of the shaft, 1 mm proud of it (0.5 mm into the bore). The bearing's bore
# pushes it sideways into the shaft to pass, and it springs out behind the bearing, trapping it against the shoulder.
# It sits on the side the gripper's fingers close along, so its push can't tip the bearing between the pads.
BALL_Y, BALL_R = 0.050, 0.002
BALL_X = SHAFT_X + SHAFT_R - BALL_R + 0.001
BALL = Slider("ball", rest=(BALL_X, BALL_Y, SHAFT_Z),
              cylinders=(((BALL_X, BALL_Y, SHAFT_Z), BALL_R, 0.003, ORANGE, "Z"),),
              travel=0.0015, rate=300.0, damping=10.0, axis="X")
# -- the bearing: origin at its bore centre, bore along its y axis; a 22 mm square, 7 mm wide, with a 9 mm bore ------
#    (0.5 mm of radial play on the 8 mm shaft: it can tip about 8 degrees before it binds)
B_HALF, BORE_HALF, B_WIDTH = 0.011, 0.0045, 0.007
BEARING_BOXES = [
    _mm(-11, 11, -3.5, 3.5, 4.5, 11, STEEL), _mm(-11, 11, -3.5, 3.5, -11, -4.5, STEEL),
    _mm(-11, -4.5, -3.5, 3.5, -4.5, 4.5, STEEL), _mm(4.5, 11, -3.5, 3.5, -4.5, 4.5, STEEL),
]
# Seated: the bearing's inner face within 1 mm of the shoulder; its outer face (centre + 3.5 mm) then lies inside the
# ball's front (50 - 2 mm), so the ball has sprung out behind it.
SEATED_Y = (SHOULDER_Y + B_WIDTH / 2, SHOULDER_Y + B_WIDTH / 2 + 0.001)

# -- fixed layout (world frame: x forward from the arms, y left, metres) ----------------------------------------------
# The link is turned 180 degrees: its shaft points toward the right arm (-y) and its far end, which the left arm
# holds, is farther from the arms (+x).
LINK_XY, LINK_YAW = (0.30, 0.06), math.pi
HOLDER_XY, HOLDER_YAW = (0.15, -0.15), math.pi
LINK = part("robot_link", pos=(*LINK_XY, TOP), yaw=LINK_YAW, mass=0.060, boxes=LINK_BOXES, cylinders=LINK_CYLINDERS,
            sliders=(BALL,), color=GREY, friction=0.5)
BEARING = part("bearing", pos=(*HOLDER_XY, TOP + B_HALF), yaw=HOLDER_YAW, mass=0.012, boxes=BEARING_BOXES, color=STEEL,
               friction=0.3)


def _holder():
    """Two fixed walls 8 mm tall that hold the bearing on edge (9 mm slot for its 7 mm width)."""
    c, s = math.cos(HOLDER_YAW), math.sin(HOLDER_YAW)
    out = []
    for i, dy in enumerate((-0.0065, 0.0065)):
        x, y = HOLDER_XY[0] - s * dy, HOLDER_XY[1] + c * dy
        out.append(Box(f"holder_{i}", pos=(x, y, TOP + 0.004), size=(0.030, 0.004, 0.008), yaw=HOLDER_YAW, color=BLUE))
    return out


OBJECTS = [*_holder(), LINK, BEARING]

# -- scoring -----------------------------------------------------------------------------------------------------------
LINK_MOVE_LIMIT = 0.03
LINK_TURN_LIMIT = 15.0
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(z0=None, link0=None, yaw0=None, lifted=False, on=False, past_half=False, failed=False, recent=[])


_reset_state()


def bearing_on_shaft(link: PartState, bearing: PartState):
    """Bearing centre in the link frame, its yaw and tilt against the link (deg; 180-degree turns are equivalent)."""
    c = link.to_local(bearing.pos)
    rel = link.rotation().T @ bearing.rotation()
    yaw = (math.degrees(math.atan2(rel[1, 0], rel[0, 0])) + 90) % 180 - 90
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, abs(rel[2, 2])))))
    return c, yaw, tilt


def _yaw(p: PartState) -> float:
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def _update(objects) -> tuple[float, bool]:
    link, bearing = objects["robot_link"], objects["bearing"]
    if _STATE["z0"] is None:
        _STATE.update(z0=float(bearing.pos[2]), link0=link.pos.copy(), yaw0=_yaw(link))
    moved = float(np.linalg.norm(link.pos - _STATE["link0"]))
    turned = abs((math.degrees(_yaw(link) - _STATE["yaw0"]) + 180) % 360 - 180)
    _STATE["failed"] |= moved > LINK_MOVE_LIMIT or turned > LINK_TURN_LIMIT
    if _STATE["failed"]:
        return 0.0, False
    c, yaw, tilt = bearing_on_shaft(link, bearing)
    # On the shaft: bore within 1 mm of the shaft's axis, square to it within 10 degrees (the bore's 0.5 mm of play lets
    # a released bearing rest up to about 8 degrees off square), with at least 3 mm of shaft inside the bore.
    on = (math.hypot(c[0] - SHAFT_X, c[2] - SHAFT_Z) < 0.001 and abs(yaw) < 10 and tilt < 10
          and SHOULDER_Y < c[1] - B_WIDTH / 2 + 0.0005 and c[1] - B_WIDTH / 2 < SHAFT_END_Y - 0.003)
    _STATE["lifted"] |= float(bearing.pos[2]) - _STATE["z0"] > 0.03
    _STATE["on"] |= bool(_STATE["lifted"] and on)
    _STATE["past_half"] |= bool(_STATE["on"] and on and c[1] <= (SHOULDER_Y + SHAFT_END_Y) / 2 + B_WIDTH / 2)
    ball_up = link.sliders.get("ball", 0.0) > -0.0005
    seated = _STATE["past_half"] and on and SEATED_Y[0] - 0.0005 <= c[1] <= SEATED_Y[1] and ball_up
    recent = _STATE["recent"]
    recent.append(c.copy())
    del recent[:-20]
    still = len(recent) == 20 and all(np.linalg.norm(p - c) < 0.001 for p in recent)
    if seated:
        return 1.0, bool(still)
    if _STATE["past_half"]:
        return 0.75, False
    if _STATE["on"]:
        return 0.5, False
    return (0.25 if _STATE["lifted"] else 0.0), False


def success(objects) -> bool:
    """Seated against the shoulder behind the ball, on the shaft, still for 1 s (within 1 mm), link in place."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 bearing lifted 3 cm; 0.5 on the shaft; 0.75 past the shaft's middle; 1.0 clicked past the ball onto the
    shoulder. 0 if the link ever moved 3 cm or turned 15 degrees."""
    return _update(objects)[0]
