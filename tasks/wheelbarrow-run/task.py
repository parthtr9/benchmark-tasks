"""On the station table, straight ahead of the arms: a green mini wheelbarrow (36 cm long) standing on its wheel and
two rear legs, its two round black handle grips 28 cm apart pointing back at the arms, with four red clay bricks side
by side in its tray; and an orange tape line across the table 12 cm ahead of its wheel. The robot must lift the
barrow by both handles so it balances on its wheel, wheel it forward until the wheel is on the line, and set it down
on its legs with every brick still in the tray. Lifted by one handle it rolls over sideways about that handle.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Hinge, PartState, finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Lift the wheelbarrow by both handles, wheel it forward until its wheel is on the orange line, and set it "
               "down on its legs without spilling the bricks.")

GREEN = (0.15, 0.55, 0.25)
STEEL = (0.55, 0.57, 0.60)
BLACK = (0.08, 0.08, 0.08)
BRICK_RED = (0.62, 0.25, 0.18)
ORANGE = (1.00, 0.50, 0.00)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- the barrow, in its own frame: origin at the axle, +x toward the wheel, z up (mm) -----------------------------------
# Handle grips 28 cm apart: at 18 cm the two arms' forearms nearly met with both hands on the handles.
H = 140
BARROW_MM = [
    (-270, -40, H - 10, H + 10, 20, 40, STEEL), (-270, -40, -H - 10, -H + 10, 20, 40, STEEL),   # rails
    (-190, -50, -70, 70, 20, 26, GREEN),                                                  # tray floor
    (-56, -50, -70, 70, 26, 56, GREEN), (-190, -184, -70, 70, 26, 56, GREEN),              # front, back walls
    (-190, -50, 64, 70, 26, 56, GREEN), (-190, -50, -70, -64, 26, 56, GREEN),              # side walls
    (-180, -170, -H - 10, H + 10, 14, 20, STEEL), (-70, -60, -H - 10, H + 10, 14, 20, STEEL),   # cross bars
    (-46, -40, -H - 10, H + 10, 14, 30, STEEL),                                            # front bar
    (-40, 0, 10, 14, -4, 30, STEEL), (-40, 0, -14, -10, -4, 30, STEEL),                    # fork plates
    (-175, -165, H - 10, H + 10, -30, 20, STEEL), (-175, -165, -H - 10, -H + 10, -30, 20, STEEL),  # legs
]
WHEEL_R, WHEEL_W = 0.030, 0.016
HANDLE_Y = H / 1000                    # round grips; a single pinch can't stop the loaded barrow rolling about one
GRIP_R, GRIP_LEN, GRIP_X = 0.010, 0.070, -0.305
GRIPS = [((GRIP_X, y, 0.030), GRIP_R, GRIP_LEN, BLACK, "X") for y in (HANDLE_Y, -HANDLE_Y)]
LEG_X, LEG_BOTTOM_Z = -0.170, -0.030
TRAY_IN = (np.array([-0.184, -0.064, 0.026]), np.array([-0.056, 0.064, 0.090]))   # tray inside, generous on top
# The wheel turns on a free axle; +-6 rad (about 18 cm of rolling either way from the start) covers the 12 cm run.
WHEEL = Hinge("wheel", pivot=(0.0, 0.0, 0.0), cylinders=(((0.0, 0.0, 0.0), WHEEL_R, WHEEL_W, BLACK, "Y"),),
              lower=-6.0, upper=6.0, stiffness=0.0, damping=0.0005, mass=0.030, axis="Y")

# -- fixed layout (world frame: x forward from the arms, y left, metres) ----------------------------------------------
# Plain floats: Arena writes start poses as float32 arrays and rejects numpy float64 values.
AXLE_POS = (0.48, 0.0, TOP + WHEEL_R)
AXLE = np.array(AXLE_POS)
TRAVEL = 0.12
LINE_X = AXLE_POS[0] + TRAVEL          # the orange line, 12 cm ahead of the starting axle
LINE_HALF = 0.02
BARROW = part("wheelbarrow", pos=AXLE_POS, yaw=0.0, mass=0.350, boxes=[_mm(*b) for b in BARROW_MM],
              cylinders=GRIPS, hinges=(WHEEL,), color=GREEN, friction=0.6)
BRICK_SIZE, BRICK_MASS = (0.060, 0.030, 0.030), 0.100      # mini clay bricks
BRICKS = [Box(f"brick_{i}", pos=(AXLE_POS[0] - 0.120, y, AXLE_POS[2] + 0.026 + 0.015 + 0.0005), size=BRICK_SIZE,
              mass=BRICK_MASS, color=BRICK_RED) for i, y in enumerate((-0.045, -0.015, 0.015, 0.045))]
LINE = Box("line", pos=(LINE_X, 0.0, TOP + 0.0003), size=(0.012, 0.20, 0.0006), color=ORANGE)
OBJECTS = [LINE, BARROW, *BRICKS]

# -- scoring -----------------------------------------------------------------------------------------------------------
RELEASE_GAP = 0.03
TOUCH_GAP = 0.012
BARROW_LO, BARROW_HI = np.array([-0.340, -HANDLE_Y - 0.010, -0.030]), np.array([0.030, HANDLE_Y + 0.010, 0.056])
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(x0=None, legs_prev=None, drag=0.0, balanced=False, wheeled=False, at_line=False, tipped=False,
                  spilled=False, dragged=False, touched=False, released=False, finger_check=None, recent=[], step=0,
                  scored=None, last=(0.0, False))


_reset_state()


def _box_gap(c, b) -> float:
    (cx, cy, cz), (sx, sy, sz), _ = b
    lo, hi = np.array([cx - sx / 2, cy - sy / 2, cz - sz / 2]), np.array([cx + sx / 2, cy + sy / 2, cz + sz / 2])
    return float(np.linalg.norm(np.maximum(np.maximum(lo - c, c - hi), 0.0)))


def in_grip(c) -> bool:
    """A point (barrow frame) on or around a handle grip, where the hands belong."""
    return bool(GRIP_X - GRIP_LEN / 2 - 0.010 <= c[0] <= GRIP_X + GRIP_LEN / 2 + 0.010
                and abs(abs(c[1]) - HANDLE_Y) <= 0.030)


@on_step
def _fingers(environment) -> None:
    """Every finger pad of both arms: released when all are 3 cm outside the barrow's extent; a touch off the grips when
    any is within 12 mm of the barrow's frame, tray or legs away from the handle grips (score() cannot see the robot)."""
    from yam_arena.scene import object_states

    _STATE["step"] += 1
    try:
        barrow = object_states(environment, ["wheelbarrow"])["wheelbarrow"]
        pts = [barrow.to_local(p) for p in finger_points(environment)]
        _STATE["released"] = bool(min(np.linalg.norm(np.maximum(np.maximum(BARROW_LO - c, c - BARROW_HI), 0.0))
                                      for c in pts) >= RELEASE_GAP)
        boxes = [_mm(*b) for b in BARROW_MM]
        _STATE["touched"] |= any(not in_grip(c) and min(_box_gap(c, b) for b in boxes) < TOUCH_GAP for c in pts)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - report once; an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"wheelbarrow-run: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"], _STATE["released"] = False, False


def legs(barrow: PartState) -> list:
    return [barrow.to_world((LEG_X, y, LEG_BOTTOM_Z)) for y in (HANDLE_Y, -HANDLE_Y)]


def in_tray(barrow: PartState, brick) -> bool:
    c = barrow.to_local(brick.pos)
    return bool(np.all(c >= TRAY_IN[0]) and np.all(c <= TRAY_IN[1]))


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    barrow = objects["wheelbarrow"]
    R = barrow.rotation()
    if _STATE["x0"] is None:
        _STATE["x0"] = float(barrow.pos[0])
    feet = legs(barrow)
    heights = [float(f[2] - TOP) for f in feet]
    if _STATE["legs_prev"] is not None:
        for f, prev, h in zip(feet, _STATE["legs_prev"], heights):
            if h < 0.004:  # touching the table: count how far it slides
                _STATE["drag"] += float(np.linalg.norm((f - prev)[:2]))
    _STATE["legs_prev"] = feet
    # Violations are permanent: nothing that happens later clears them.
    _STATE["tipped"] |= abs(math.degrees(math.asin(max(-1.0, min(1.0, R[2, 1]))))) > 20
    _STATE["spilled"] |= not all(in_tray(barrow, objects[f"brick_{i}"]) for i in range(len(BRICKS)))
    _STATE["dragged"] |= _STATE["drag"] > 0.020
    failed = _STATE["tipped"] or _STATE["spilled"] or _STATE["dragged"] or _STATE["touched"]
    up = min(heights) > 0.015
    near_line = abs(float(barrow.pos[0]) - LINE_X) < LINE_HALF and abs(float(barrow.pos[1]) - AXLE[1]) < 0.03
    _STATE["balanced"] |= up
    _STATE["wheeled"] |= up and float(barrow.pos[0]) - _STATE["x0"] > 0.06
    _STATE["at_line"] |= bool(up and near_line)
    square = abs(math.degrees(math.atan2(R[1, 0], R[0, 0]))) < 10
    down = max(heights) < 0.003
    # Still: over the last second (20 control steps) the barrow stayed within 1 mm.
    recent = _STATE["recent"]
    recent.append(barrow.pos.copy())
    del recent[:-20]
    still = len(recent) == 20 and all(np.linalg.norm(p - barrow.pos) < 0.001 for p in recent)
    done = bool(down and near_line and square and _STATE["released"] and still)
    if failed:
        result = (0.0, False)
    elif done:
        result = (1.0, True)
    elif _STATE["at_line"]:
        result = (0.75, False)
    elif _STATE["wheeled"]:
        result = (0.5, False)
    else:
        result = (0.25 if _STATE["balanced"] else 0.0), False
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Set down on both legs with the axle within 2 cm of the line, square within 10 degrees, every brick in the tray,
    both arms' fingers 3 cm clear of the barrow and the barrow still for the last second, with no violation; held 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if it ever tipped 20 degrees sideways, spilled a brick, dragged a leg 2 cm, or was touched off its grips.
    Otherwise 0.25 once balanced on its wheel (both legs 15 mm up), 0.5 once wheeled 6 cm forward that way, 0.75 once at
    the line that way, 1.0 set down at the line and released."""
    return _update(objects)[0]
