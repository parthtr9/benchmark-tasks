"""On the station table: an upright light grey mounting panel ahead of the right arm, with a steel top-hat DIN rail
(TS35 x 7.5) across it 6 cm above the table between two dark end clamps; and a blue 22.5 mm DIN-rail module (22.5 x 55
x 90 mm) standing in a low holder in front of the right arm, its back (with the mounting hook at the top and a spring
latch at the bottom) turned toward the panel. The robot must hook the top of the module's back over the rail's upper
lip, swing the module's bottom in against the rail and push it until the latch clicks behind the rail's lower lip.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, Slider, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Pick up the blue module, hook the top of its back over the upper edge of the metal rail on the panel, "
               "swing its bottom in against the rail and push it until its latch clicks onto the rail.")

STEEL = (0.62, 0.64, 0.66)
PANEL_GREY = (0.82, 0.82, 0.80)
DARK = (0.25, 0.25, 0.27)
BLUE = (0.10, 0.30, 0.90)
ORANGE = (1.00, 0.50, 0.00)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- the rail, in the rail frame: x = 0 on the lips' front face and +x into the panel, z = 0 at the rail's centre -------
# A TS35 x 7.5 top hat: lips 5.5 mm wide at the front, walls back to a web on the panel. The lips are 2 mm thick (1 mm
# on a real rail) so a pushed latch cannot tunnel through them.
RAIL_LEN = 220
RAIL_MM = [
    (0, 2, -RAIL_LEN / 2, RAIL_LEN / 2, 12, 17.5), (0, 2, -RAIL_LEN / 2, RAIL_LEN / 2, -17.5, -12),
    (0, 7.5, -RAIL_LEN / 2, RAIL_LEN / 2, 12, 13.5), (0, 7.5, -RAIL_LEN / 2, RAIL_LEN / 2, -13.5, -12),
    (6, 7.5, -RAIL_LEN / 2, RAIL_LEN / 2, -13.5, 13.5),
]
# End clamps stand proud of the rail, so a module can't be slid on from an end.
END_STOP_MM = [(-12, 7.5, RAIL_LEN / 2, RAIL_LEN / 2 + 10, -25, 25), (-12, 7.5, -RAIL_LEN / 2 - 10, -RAIL_LEN / 2, -25, 25)]
RAIL_H = 0.060                       # rail centre above the table
PANEL_MM = (7.5, 17.5, -140, 140, -RAIL_H * 1000, 35)
RAIL_ORIGIN = np.array([0.30, -0.13, TOP + RAIL_H])


def _fixed(name, mm, color, origin=RAIL_ORIGIN, yaw=0.0):
    (cx, cy, cz), size, _ = _mm(*mm, color)
    c, s = math.cos(yaw), math.sin(yaw)
    pos = (origin[0] + c * cx - s * cy, origin[1] + s * cx + c * cy, origin[2] + cz)
    return Box(name, pos=pos, size=size, yaw=yaw, color=color)


# -- the module, in its own frame: +x from its front to its back, origin on the back at the rail's centre height -------
MODULE_BOXES = [
    _mm(-55, 0, -11.25, 11.25, -45, 45, BLUE),        # housing
    _mm(0, 5.0, -11.25, 11.25, 17.5, 21, BLUE),       # hook bridge: rests on the top lip's edge
    _mm(3.0, 5.0, -11.25, 11.25, 14, 17.5, BLUE),     # hook tooth: drops 1 mm behind the top lip
]
# The latch slides along the module's z axis. At rest its tooth reaches 1.5 mm above the lower lip's edge, behind it:
# its square back face holds the module on the rail, and its 45 degree chamfer lets the lip push it down as the
# module swings in, until it clears the lip and the spring snaps it back up.
CHAMFER_FROM, CHAMFER_TO, CHAMFER_T = np.array([3.5, -16.0]), np.array([6.0, -18.5]), 1.5


def _chamfer():
    """The box whose top face is the tooth's chamfer, turned about the module's y axis."""
    d = CHAMFER_TO - CHAMFER_FROM
    length = float(np.linalg.norm(d))
    d /= length
    n = np.array([-d[1], d[0]])
    centre = (CHAMFER_FROM + CHAMFER_TO) / 2 - n * CHAMFER_T / 2
    pitch = math.atan2(-d[1], d[0])
    return ((centre[0] / 1000, 0.0, centre[1] / 1000), (length / 1000, 0.016, CHAMFER_T / 1000), ORANGE,
            (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0))


# No preload. Hung on its hook, the module's weight presses the latch down about 0.6 mm, short of the 1.5 mm that
# clears the lip, so it does not click by itself.
LATCH = Slider("latch", rest=(0.0, 0.0, -0.020),
               boxes=(_mm(-8, 6.0, -8, 8, -22, -18.5, ORANGE), _mm(3.0, 3.5, -8, 8, -18.5, -16.0, ORANGE), _chamfer()),
               travel=0.004, rate=800.0, damping=5.0, mass=0.010, friction=0.2)

# -- fixed layout (world frame: x forward from the arms, y left, metres) ----------------------------------------------
MODULE_YAW = math.radians(-25)
HOUSING_XY = (0.19, -0.21)            # housing centre, standing upright in its holder
HALF_HEIGHT = 0.045
_c, _s = math.cos(MODULE_YAW), math.sin(MODULE_YAW)
MODULE_ORIGIN = (HOUSING_XY[0] + 0.0275 * _c, HOUSING_XY[1] + 0.0275 * _s, TOP + HALF_HEIGHT)
MODULE = part("din_module", pos=MODULE_ORIGIN, yaw=MODULE_YAW, mass=0.080, boxes=MODULE_BOXES, sliders=(LATCH,),
              color=BLUE, friction=0.6)
# Two 12 mm blocks either side of the module's base, 1 mm clear, keep the narrow module upright until it is picked.
HOLDER = [_fixed(f"holder_{i}", mm, DARK, origin=np.array(MODULE_ORIGIN), yaw=MODULE_YAW)
          for i, mm in enumerate([(-60, 5, 12.25, 20.25, -45, -33), (-60, 5, -20.25, -12.25, -45, -33)])]
FIXTURE = ([_fixed(f"rail_{i}", mm, STEEL) for i, mm in enumerate(RAIL_MM)]
           + [_fixed(f"end_clamp_{i}", mm, DARK) for i, mm in enumerate(END_STOP_MM)]
           + [_fixed("panel", PANEL_MM, PANEL_GREY)])
OBJECTS = [*FIXTURE, *HOLDER, MODULE]

# -- scoring -----------------------------------------------------------------------------------------------------------
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(z0=None, lifted=False, hooked=False, pushed=False, recent=[], step=0, scored=None, last=(0.0, False))


@on_step
def _count_step(environment=None) -> None:
    _STATE["step"] += 1


_reset_state()


def module_in_rail(module: PartState):
    """The module origin (mm) and rotation in the rail frame (the rail is square to the world)."""
    return (module.pos - RAIL_ORIGIN) * 1000, module.rotation()


def geometry(c, rel):
    """Hooked over the top lip; back within 1.5 mm of the top lip and 1 mm of the bottom lip; square; and the x (mm)
    of the back at the bottom lip (0 on the lip, negative standing off it)."""
    hook = rel @ np.array([4.0, 0, 15.75]) + c
    back_top, back_bottom = rel @ np.array([0, 0, 15.0]) + c, rel @ np.array([0, 0, -15.0]) + c
    on_rail = abs(c[1]) <= RAIL_LEN / 2 - 11.25
    hooked = bool(on_rail and 2.0 < hook[0] < 7.5 and 13.5 < hook[2] < 19.0 and rel[2, 2] > math.cos(math.radians(30)))
    flush = bool(-1.5 <= back_top[0] <= 0.5 and -1.0 <= back_bottom[0] <= 0.5)
    square = bool(abs(rel[1, 0]) < math.sin(math.radians(3)) and abs(rel[2, 1]) < math.sin(math.radians(3)))
    return hooked, flush, square, float(back_bottom[0])


def _update(objects) -> tuple[float, bool]:
    module = objects["din_module"]
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    if _STATE["z0"] is None:
        _STATE["z0"] = float(module.pos[2])
    c, rel = module_in_rail(module)
    hooked, flush, square, bottom_x = geometry(c, rel)
    _STATE["lifted"] |= float(module.pos[2]) - _STATE["z0"] > 0.03
    _STATE["hooked"] |= bool(_STATE["lifted"] and hooked)
    _STATE["pushed"] |= bool(_STATE["hooked"] and hooked and bottom_x > -2.0)
    latched = module.sliders.get("latch", 0.0) > -0.0005
    mounted = bool(_STATE["hooked"] and hooked and flush and square and abs(c[2]) < 1.5 and latched)
    # Still: over the last second (20 control steps) the module stayed within 1 mm. A light part's instantaneous
    # velocity is too noisy in simulation to judge rest by.
    recent = _STATE["recent"]
    recent.append(c.copy())
    del recent[:-20]
    still = len(recent) == 20 and all(np.linalg.norm(p - c) < 1.0 for p in recent)
    if mounted:
        result = (1.0, bool(still))
    elif _STATE["pushed"]:
        result = (0.75, False)
    elif _STATE["hooked"]:
        result = (0.5, False)
    else:
        result = (0.25 if _STATE["lifted"] else 0.0), False
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Mounted: hooked over the top lip, back within 1.5 mm of the top lip and 1 mm of the bottom lip, square within 3
    degrees, at hanging height, latch sprung back behind the lower lip, and still for the last second; held 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 module lifted 3 cm; 0.5 hung on the rail by its hook; 0.75 then pushed to within 2 mm of the lower lip;
    1.0 mounted."""
    return _update(objects)[0]
