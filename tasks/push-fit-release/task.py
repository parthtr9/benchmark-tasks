"""On the station table: a 15 mm copper pipe comes out of a wall plate into a brass push-fit coupling, and a second
15 mm copper pipe, 30 cm long, is pushed 25 mm into the coupling's open end, its far end resting on a wooden block. A
grab tooth inside the coupling bites the pipe, so it cannot be pulled out until the orange release collet on the
coupling's mouth is pressed in and held. The robot must press and hold the collet with one arm while the other pulls
the pipe out, then lay the pipe down on the table.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, Slider, on_reset, on_step, part, set_slider_target  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Press the orange release collet on the brass fitting in and hold it there while you pull the copper "
               "pipe out of the fitting, then lay the pipe down flat on the table.")

GREY = (0.55, 0.56, 0.58)
WHITE = (0.90, 0.90, 0.88)
BRASS = (0.80, 0.62, 0.25)
COPPER = (0.72, 0.42, 0.22)
ORANGE = (1.00, 0.50, 0.00)
WOOD = (0.62, 0.47, 0.30)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- the fitting assembly: origin on the table under the coupling's mouth; y runs along the pipe axis into the
#    coupling, z up. A base plate, a wall plate at the back with a copper stub coming out of it into a brass coupling
#    (30 x 30 mm, 80 mm long) on a pedestal. The coupling's mouth has an 18 mm wide bore 25 mm deep (the insertion
#    depth); a slot in its top 5-7 mm in from the mouth holds the grab tooth.
AXIS_Z = 0.0445          # the pipe's axis resting on the bore floor (37 mm)
DEPTH = 0.025            # insertion depth: the pipe's end sits against the bore's bottom 25 mm in
FITTING_BOXES = [
    _mm(-45, 45, -5, 142, 0, 8, GREY),                                              # base plate
    _mm(-60, 60, 130, 142, 8, 110, WHITE),                                          # wall plate
    _mm(-10, 10, 10, 75, 8, 30, GREY),                                              # pedestal
    _mm(-15, 15, 0, 25, 30, 37, BRASS),                                             # bore floor
    _mm(-15, -9, 0, 25, 37, 53, BRASS), _mm(9, 15, 0, 25, 37, 53, BRASS),           # bore sides
    _mm(-15, 15, 0, 4.5, 53, 60, BRASS), _mm(-15, 15, 7.5, 25, 53, 60, BRASS),      # bore top, either side of the slot
    _mm(-15, -3.5, 4.5, 7.5, 53, 60, BRASS), _mm(3.5, 15, 4.5, 7.5, 53, 60, BRASS),  # beside the slot
    _mm(-15, 15, 25, 80, 30, 60, BRASS),                                            # the coupling's closed half
]
FITTING_CYLS = [((0.0, 0.105, 0.045), 0.0075, 0.050, COPPER, "Y")]                  # stub from the wall
# The release collet: an orange square ring (50 mm wide, 36 mm tall, 20 mm hole, 5 mm thick) 4 mm in front of the mouth. A
# spring (400 N/m) holds it out; it presses in 3.5 mm. Pressed 2.5 mm, it lifts the grab tooth clear of the pipe.
COLLET = Slider("collet", rest=(0.0, -0.0065, 0.045),
                boxes=(_mm(-25, 25, -9, -4, 27, 36, ORANGE), _mm(-25, 25, -9, -4, 54, 63, ORANGE),
                       _mm(-25, -10, -9, -4, 36, 54, ORANGE), _mm(10, 25, -9, -4, 36, 54, ORANGE)),
                travel=0.0035, rate=400.0, damping=4.0, mass=0.004, axis="Y", sign=1)
# The grab tooth: a 6 x 2.5 mm blade dropping through the slot into the pipe's grip groove (its tip 0.5 mm above
# the groove's floor, 0.75 mm of play either side). A stiff drive holds it down, so the groove's wall cannot push it up;
# it is lifted 4 mm, clear of the pipe, while the collet is pressed.
TOOTH = Slider("tooth", rest=(0.0, 0.006, 0.057),
               boxes=(_mm(-3, 3, 4.75, 7.25, 49.5, 64, GREY),), travel=0.004, rate=50000.0, damping=50.0, mass=0.002,
               axis="Z", sign=1)
COLLET_RELEASE = 0.0025
TOOTH_LIFT = 0.004

# -- the pipe: origin at its middle, axis along y, inner end at +150 mm. 15 mm copper, 300 mm long; the grab tooth's
#    groove (3 mm deep, 4 mm wide) is 17-21 mm from the inner end. A thin box inside gives the part its required box.
PIPE_R, PIPE_HALF = 0.0075, 0.150
_G0, _G1 = PIPE_HALF - 0.021, PIPE_HALF - 0.017      # the groove
PIPE_CYLS = [((0.0, (_G0 - PIPE_HALF) / 2, 0.0), PIPE_R, _G0 + PIPE_HALF, COPPER, "Y"),
             ((0.0, (_G0 + _G1) / 2, 0.0), 0.0045, _G1 - _G0, COPPER, "Y"),
             ((0.0, (_G1 + PIPE_HALF) / 2, 0.0), PIPE_R, PIPE_HALF - _G1, COPPER, "Y")]
PIPE_BOXES = [_mm(-2, 2, -140, 140, -2, 2, COPPER)]

# -- fixed layout (world frame: x forward from the arms, y left, metres). The pipe runs along y: the coupling's mouth
#    is just left of the midline in front of the left arm's reach, the pipe's free end toward the right arm.
FIT_X, MOUTH_Y = 0.28, 0.02
FITTING = part("fitting", pos=(FIT_X, MOUTH_Y, TOP), mass=5.0, boxes=FITTING_BOXES, cylinders=FITTING_CYLS,
               sliders=(COLLET, TOOTH), color=BRASS, friction=0.9)
PIPE_Y0 = MOUTH_Y + DEPTH - PIPE_HALF
PIPE = part("pipe", pos=(FIT_X, PIPE_Y0, TOP + AXIS_Z), mass=0.084, boxes=PIPE_BOXES, cylinders=PIPE_CYLS,
            color=COPPER, friction=0.5)
BLOCK = Box("block", pos=(FIT_X, PIPE_Y0 - PIPE_HALF + 0.020, TOP + 0.0185), size=(0.03, 0.02, 0.037), color=WOOD)
# Cleats (20 mm tall) screwed to the table hold the base plate (90 x 147 mm) in place.
CLEATS = [Box(f"cleat_{n}", pos=(FIT_X + dx, MOUTH_Y + dy, TOP + 0.010), size=sz, color=(0.30, 0.31, 0.33))
          for n, dx, dy, sz in (("f", 0.0, -0.011, (0.09, 0.010, 0.020)), ("b", 0.0, 0.148, (0.09, 0.010, 0.020)),
                                ("l", -0.051, 0.0685, (0.010, 0.147, 0.020)),
                                ("r", 0.051, 0.0685, (0.010, 0.147, 0.020)))]
OBJECTS = [*CLEATS, BLOCK, FITTING, PIPE]


# -- the release: the collet lifts the grab tooth --------------------------------------------------------------------
@on_step
def _release(environment) -> None:
    """While the collet is pressed 2.5 mm the grab tooth is lifted clear of the pipe; otherwise it drops back."""
    from partkit import _part_state

    pressed = abs(_part_state(environment, FITTING).sliders.get("collet", 0.0)) >= COLLET_RELEASE
    set_slider_target(environment, "fitting", "tooth", TOOTH_LIFT if pressed else 0.0)


# -- scoring -----------------------------------------------------------------------------------------------------------
MOVE_LIMIT = 0.005
PULL_CREDIT = 0.005      # pulled past the grab tooth: 0.5
PULL_OUT = 0.030         # the pipe's end 5 mm out of the mouth: 0.75
CLEAR = 0.040            # laid down with its nearest point at least 4 cm from the mouth
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(fit0=None, pressed=False, pulled=False, out=False, failed=False)
    if environment is not None:
        set_slider_target(environment, "fitting", "tooth", 0.0)


_reset_state()


def collet_pressed(fitting: PartState) -> float:
    return abs(float(fitting.sliders.get("collet", 0.0)))


def pulled(fitting: PartState, pipe: PartState) -> float:
    """How far the pipe's inner end has come out from the bottom of the bore (m)."""
    end = fitting.to_local(pipe.to_world((0.0, PIPE_HALF, 0.0)))
    return DEPTH - float(end[1])


def lying_clear(fitting: PartState, pipe: PartState) -> bool:
    """On the table (axis within 3 mm of resting height, level within 10 degrees), both ends 4 cm from the mouth."""
    axis = pipe.rotation()[:, 1]
    level = math.degrees(math.asin(min(1.0, abs(float(axis[2]))))) < 10
    mouth = fitting.to_world((0.0, 0.0, AXIS_Z))
    ends = [pipe.to_world((0.0, s * PIPE_HALF, 0.0)) for s in (-1, 1)]
    seg = ends[1] - ends[0]
    t = np.clip(np.dot(mouth - ends[0], seg) / np.dot(seg, seg), 0.0, 1.0)
    near = float(np.linalg.norm(ends[0] + t * seg - mouth))
    return bool(level and abs(float(pipe.pos[2]) - TOP - PIPE_R) < 0.003 and near >= CLEAR)


def _update(objects) -> tuple[float, bool]:
    fitting, pipe = objects["fitting"], objects["pipe"]
    if _STATE["fit0"] is None:
        _STATE["fit0"] = fitting.pos.copy()
    _STATE["failed"] |= float(np.linalg.norm(fitting.pos - _STATE["fit0"])) > MOVE_LIMIT
    if _STATE["failed"]:
        return 0.0, False
    d = pulled(fitting, pipe)
    _STATE["pressed"] |= collet_pressed(fitting) >= COLLET_RELEASE
    # Pull credit only after the collet was pressed: a pipe levered out of a locked fitting has been forced.
    _STATE["pulled"] |= _STATE["pressed"] and d >= PULL_CREDIT
    _STATE["out"] |= _STATE["pressed"] and d >= PULL_OUT
    if _STATE["out"] and lying_clear(fitting, pipe):
        return 1.0, bool(pipe.resting())
    if _STATE["out"]:
        return 0.75, False
    if _STATE["pulled"]:
        return 0.5, False
    return (0.25 if _STATE["pressed"] else 0.0), False


def success(objects) -> bool:
    """The pipe out of the fitting and lying still on the table, clear of the fitting, which has not moved."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 collet pressed 2.5 mm; 0.5 pipe pulled 5 mm (past the grab tooth); 0.75 pipe out of the fitting; 1.0 laid
    flat on the table 4 cm clear, still. 0 if the fitting moved 5 mm."""
    return _update(objects)[0]
