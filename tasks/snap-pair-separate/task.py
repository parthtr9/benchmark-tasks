"""On the station table: a 30 cm grey-and-blue printed case lying along the arms' forward direction, made of two
halves snapped together: a tongue on the near (blue) half sits in a slot in the far (grey) half, held by a spring
plunger that sits in a notch in the tongue. The robot must hold one half in each gripper and pull them straight
apart until the tongue comes out, leaving both halves on the table at least 3 cm apart.

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

INSTRUCTION = ("Hold the grey half and the blue half of the case and pull them straight apart until they come undone, "
               "leaving both halves on the table at least 3 cm apart.")

GREY = (0.55, 0.56, 0.58)
BLUE = (0.10, 0.30, 0.90)
ORANGE = (1.00, 0.50, 0.00)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- both halves share the case frame: origin at the seam (x = 0) on the table; each half is 150 x 60 x 40 mm --------
# The far (grey) half, x 0-150 mm, has a 25 mm deep slot (16 x 10 mm) at the seam for the near half's tongue.
FAR_BOXES = [
    _mm(25, 150, -30, 30, 0, 40, GREY),
    _mm(0, 25, -30, 30, 0, 15, GREY), _mm(0, 25, -30, 30, 25, 40, GREY),
    _mm(0, 25, 8, 30, 15, 25, GREY), _mm(0, 25, -30, -8, 15, 25, GREY),
]
# The near (blue) half, x -150-0 mm, and its 25 mm tongue (14 x 9 mm), with a 2 mm notch on its left side (x 9-15 mm).
NEAR_BOXES = [
    _mm(-150, 0, -30, 30, 0, 40, BLUE),
    _mm(0, 9, -7, 7, 15.5, 24.5, BLUE), _mm(9, 15, -7, 5, 15.5, 24.5, BLUE), _mm(15, 25, -7, 7, 15.5, 24.5, BLUE),
]
# The detent: a spring plunger (a rounded 6 mm nose) in the slot's left wall, 2 mm into the slot, resting in the notch.
# Pulling the tongue out pushes it sideways into the wall (about 3 N); it pushes sideways, along the gripper's pinch
# axis, so it can't tip a half between the pads.
PLUNGER_X, PLUNGER_R = 0.012, 0.003
PLUNGER = Slider("plunger", rest=(PLUNGER_X, 0.009, 0.020),
                 cylinders=(((PLUNGER_X, 0.009, 0.020), PLUNGER_R, 0.008, ORANGE, "Z"),),
                 travel=0.0025, rate=1000.0, damping=10.0, axis="Y", sign=1)
TONGUE_LEN = 0.025
GRIP_X = 0.100      # each arm holds its half 10 cm from the seam

# -- fixed layout (world frame: x forward from the arms, y left, metres) ----------------------------------------------
CASE_XY, CASE_YAW = (0.23, 0.0), 0.0
FAR = part("far_half", pos=(*CASE_XY, TOP), yaw=CASE_YAW, mass=0.060, boxes=FAR_BOXES, sliders=(PLUNGER,), color=GREY,
           friction=0.6)
NEAR = part("near_half", pos=(*CASE_XY, TOP), yaw=CASE_YAW, mass=0.060, boxes=NEAR_BOXES, color=BLUE, friction=0.6)
OBJECTS = [FAR, NEAR]

# -- scoring -----------------------------------------------------------------------------------------------------------
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(started=False, past_detent=False, apart=False, recent=[])


_reset_state()


def separation(far: PartState, near: PartState):
    """How far the halves have been pulled apart along the case's length (m; 0 when snapped together), how far off
    that line they are (m), and the tilt between them (deg)."""
    d = far.to_local(near.pos)            # the near half's origin in the far half's frame (0 when together)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, (far.rotation().T @ near.rotation())[2, 2]))))
    return float(-d[0]), float(math.hypot(d[1], d[2])), tilt


def _flat_on_table(p: PartState) -> bool:
    return abs(float(p.pos[2]) - TOP) < 0.005 and p.rotation()[2, 2] > math.cos(math.radians(10))


def _update(objects) -> tuple[float, bool]:
    far, near = objects["far_half"], objects["near_half"]
    sep, _, _ = separation(far, near)
    _STATE["started"] |= sep >= 0.002
    _STATE["past_detent"] |= sep >= 0.006
    # Apart: the tongue fully out of the slot, plus a 5 mm margin.
    _STATE["apart"] |= sep >= TONGUE_LEN + 0.005
    # Set down: both halves flat on the table with at least 3 cm between the tongue's tip and the grey half's seam
    # face. Each half's origin is on its seam face, so that gap is their origins' distance less the tongue.
    gap = float(np.linalg.norm(far.pos - near.pos)) - TONGUE_LEN
    down = _flat_on_table(far) and _flat_on_table(near) and gap >= 0.030
    recent = _STATE["recent"]
    recent.append(np.r_[far.pos, near.pos])
    del recent[:-20]
    still = len(recent) == 20 and all(np.abs(p - recent[-1]).max() < 0.001 for p in recent)
    if _STATE["apart"] and down:
        return 1.0, bool(still)
    if _STATE["apart"]:
        return 0.75, False
    if _STATE["past_detent"]:
        return 0.5, False
    return (0.25 if _STATE["started"] else 0.0), False


def success(objects) -> bool:
    """Pulled apart and both halves flat on the table with at least 3 cm between them, still for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 pulled 2 mm apart; 0.5 pulled 6 mm apart, past the detent; 0.75 the tongue out of the slot; 1.0 both halves
    flat on the table at least 3 cm apart."""
    return _update(objects)[0]
