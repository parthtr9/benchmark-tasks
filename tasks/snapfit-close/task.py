"""On the station table: a grey plastic case base (120 x 70 x 25 mm) on a short stand to the right of centre, with an
open pocket in its top and two orange spring latches near its front end; and its blue lid (114 x 64 x 7.5 mm) lying
on two grey rails to the left. The robot must put the lid into the pocket, front tabs over the latches, and press the
front edge down until both latches click.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, Slider, finger_points, on_reset, on_step, part, write_pose  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Put the blue lid into the pocket of the grey case with its orange tabs over the orange latches, then "
               "press the front edge of the lid down until both latches click.")

GREY = (0.55, 0.56, 0.58)
BLUE = (0.10, 0.30, 0.90)
ORANGE = (1.00, 0.50, 0.00)

# -- the case base: a 120 x 70 x 25 mm box with a 116 x 66 mm pocket, floor 5 mm above its centre --------------------
FLOOR_Z = 0.005          # pocket floor, in the base frame (origin at the base's centre)
LID_HALF = 0.00375
LATCH_XY = ((0.045, 0.022), (0.045, -0.022))
LATCH_TRAVEL = 0.004
ENGAGE = 0.0007          # a latch clicks when the lid's underside reaches within 0.7 mm of the floor above it


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


BASE_BOXES = [
    _mm(-60, 60, -35, 35, -12.5, 5, GREY),
    _mm(-60, -58, -35, 35, 5, 12.5, GREY), _mm(58, 60, -35, 35, 5, 12.5, GREY),
    _mm(-58, -30, 33, 35, 5, 12.5, GREY), _mm(-58, -30, -35, -33, 5, 12.5, GREY),
    _mm(30, 58, 33, 35, 5, 12.5, GREY), _mm(30, 58, -35, -33, 5, 12.5, GREY),
]
LATCHES = tuple(
    Slider(f"latch_{i}", rest=(x, y, FLOOR_Z + LATCH_TRAVEL), boxes=(_mm(x * 1000 - 6, x * 1000 + 6, y * 1000 - 6,
                                                                       y * 1000 + 6, (FLOOR_Z + LATCH_TRAVEL) * 1000 - 1,
                                                                       (FLOOR_Z + LATCH_TRAVEL) * 1000, ORANGE),),
           travel=LATCH_TRAVEL, rate=2500.0, damping=20.0)
    for i, (x, y) in enumerate(LATCH_XY)
)
LID_BOXES = [_mm(-57, 57, -32, 32, -3.75, 3.75, BLUE),
             _mm(39, 51, 16, 28, 3.75, 4.25, ORANGE), _mm(39, 51, -28, -16, 3.75, 4.25, ORANGE)]

# -- fixed layout (world frame: x forward from the arms, y left, metres) ----------------------------------------------
STAND_H = 0.020
BASE_XY, BASE_YAW = (0.21, -0.05), math.radians(-45)
RAILS_XY, RAILS_YAW = (0.11, 0.17), math.radians(-30)
RAIL_H = 0.015


def _yawed(xy, yaw, local):
    c, s = math.cos(yaw), math.sin(yaw)
    return (xy[0] + c * local[0] - s * local[1], xy[1] + s * local[0] + c * local[1])


BASE = part("case_base", pos=(*BASE_XY, TOP + STAND_H + 0.0125), yaw=BASE_YAW, mass=0.150, boxes=BASE_BOXES,
            sliders=LATCHES, color=GREY, friction=0.6)
LID = part("lid", pos=(*RAILS_XY, TOP + RAIL_H + LID_HALF), yaw=RAILS_YAW, mass=0.040, boxes=LID_BOXES, color=BLUE,
           friction=0.6)
STAND = Box("stand", pos=(*BASE_XY, TOP + STAND_H / 2), size=(0.040, 0.050, STAND_H), yaw=BASE_YAW, color=GREY)
RAILS = [Box(f"rail_{i}", pos=(*_yawed(RAILS_XY, RAILS_YAW, (dx, 0.0)), TOP + RAIL_H / 2), size=(0.012, 0.050, RAIL_H),
             yaw=RAILS_YAW, color=GREY) for i, dx in enumerate((-0.040, 0.040))]
OBJECTS = [STAND, *RAILS, BASE, LID]

# -- latches -----------------------------------------------------------------------------------------------------------
# A real snap-fit hook flexes and catches the lid. This rigid simulation has no compliant hook, so a latch "clicks" when
# the lid's underside above it is pressed within ENGAGE of the floor, and once both have clicked the lid is held in its
# seated pose relative to the base, written back every control step.
_STATE: dict = {}


def _reset_state(environment=None) -> None:
    _STATE.update(latched=[False, False], hold=None, lid_z0=None, lifted=False, placed=False, base0=None, failed=False,
                  recent=[], released=False, finger_check=None)


_reset_state()


def _lid_in_base(base: PartState, lid: PartState):
    rel_pos = base.to_local(lid.pos)
    rel_rot = base.rotation().T @ lid.rotation()
    return rel_pos, rel_rot


def _corner_heights(base: PartState, lid: PartState):
    """Base-frame position of the lid's underside above each latch."""
    return [base.to_local(lid.to_world((x, y, -LID_HALF))) for x, y in LATCH_XY]


def _quat_xyzw(r: np.ndarray) -> tuple:
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    if w > 1e-6:
        return ((r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w)
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1 + r[i, i] - r[j, j] - r[k, k]) * 2
    q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
    q[i], q[j], q[k] = s / 4, (r[j, i] + r[i, j]) / s, (r[k, i] + r[i, k]) / s
    return tuple(q)


@on_reset
def _on_reset(environment) -> None:
    _reset_state()


@on_step
def _on_step(environment) -> None:
    from yam_arena.scene import object_states

    states = object_states(environment, ["case_base", "lid"])
    base, lid = states["case_base"], states["lid"]
    for i, c in enumerate(_corner_heights(base, lid)):
        x, y = LATCH_XY[i]
        if abs(c[0] - x) < 0.003 and abs(c[1] - y) < 0.003 and c[2] < FLOOR_Z + ENGAGE:
            _STATE["latched"][i] = True
    if all(_STATE["latched"]) and _STATE["hold"] is None:
        rel_pos, rel_rot = _lid_in_base(base, lid)
        yaw = math.atan2(rel_rot[1, 0], rel_rot[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        _STATE["hold"] = (np.array([rel_pos[0], rel_pos[1], FLOOR_Z + LID_HALF]),
                          np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]))
    if _STATE["hold"] is not None:
        rel_pos, rel_rot = _STATE["hold"]
        write_pose(environment, "lid", base.to_world(rel_pos), _quat_xyzw(base.rotation() @ rel_rot),
                   lin_vel=base.lin_vel, ang_vel=base.ang_vel)


RELEASE_GAP = 0.03
LID_LO, LID_HI = np.array([-0.057, -0.032, -LID_HALF]), np.array([0.057, 0.032, LID_HALF + 0.0005])  # in its frame


@on_step
def _fingers(environment) -> None:
    """Released: every finger pad of both arms at least 3 cm from the lid (score() cannot see the robot)."""
    from yam_arena.scene import object_states

    try:
        lid = object_states(environment, ["lid"])["lid"]
        gaps = [np.linalg.norm(np.maximum(np.maximum(LID_LO - c, c - LID_HI), 0.0))
                for c in (lid.to_local(p) for p in finger_points(environment))]
        _STATE["released"] = bool(min(gaps) >= RELEASE_GAP)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - report once; an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"snapfit-close: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"], _STATE["released"] = False, False


# -- scoring -----------------------------------------------------------------------------------------------------------
BASE_MOVE_LIMIT = 0.03
BASE_TILT_LIMIT = math.radians(20)


def _in_pocket(base: PartState, lid: PartState) -> bool:
    rel_pos, rel_rot = _lid_in_base(base, lid)
    return bool(abs(rel_pos[0]) < 0.003 and abs(rel_pos[1]) < 0.003 and rel_rot[0, 0] > math.cos(math.radians(5))
                and rel_rot[2, 2] > math.cos(math.radians(10))
                and FLOOR_Z + LID_HALF - 0.001 < rel_pos[2] < FLOOR_Z + LID_HALF + LATCH_TRAVEL + 0.002)


def _base_ok(base: PartState) -> bool:
    if _STATE["base0"] is None:
        _STATE["base0"] = base.pos.copy()
    moved = float(np.linalg.norm(base.pos - _STATE["base0"]))
    # Once the base has been knocked out of place, the trial stays failed even if it settles back.
    _STATE["failed"] |= moved > BASE_MOVE_LIMIT or base.rotation()[2, 2] < math.cos(BASE_TILT_LIMIT)
    return not _STATE["failed"]


def _update(objects) -> tuple[float, bool]:
    base, lid = objects["case_base"], objects["lid"]
    if _STATE["lid_z0"] is None:
        _STATE["lid_z0"] = float(lid.pos[2])
    ok = _base_ok(base)
    _STATE["lifted"] |= float(lid.pos[2]) - _STATE["lid_z0"] > 0.03
    inside = _in_pocket(base, lid)
    _STATE["placed"] |= bool(_STATE["lifted"] and inside and lid.resting())
    n = sum(_STATE["latched"])
    # Still and let go: over the last second (20 control steps) the lid stayed within 1 mm with both grippers at least
    # 3 cm away.
    recent = _STATE["recent"]
    recent.append((lid.pos.copy(), _STATE["released"]))
    del recent[:-20]
    still = len(recent) == 20 and all(np.linalg.norm(p - lid.pos) < 0.001 and free for p, free in recent)
    if not ok:
        return 0.0, False
    # Full score needs the lid latched, let go and still for 1 s; a latched lid still pressed or moving scores 0.75.
    if n == 2 and inside and _STATE["placed"] and still:
        return 1.0, True
    if n >= 1 and _STATE["placed"]:
        return 0.75, False
    if _STATE["placed"]:
        return 0.5, False
    return (0.25 if _STATE["lifted"] else 0.0), False


def success(objects) -> bool:
    """Both latches clicked with the lid seated in the pocket, both grippers at least 3 cm from the lid and the lid still
    (within 1 mm) over the last second, and the base still in place; held 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 lid lifted 3 cm; 0.5 lid resting in the pocket; 0.75 one latch, or both while still pressed or moving; 1.0
    both, lid seated, let go (both grippers at least 3 cm from it) and still (within 1 mm) over the last second. 0 if
    the base ever moved 3 cm or tipped 20 degrees."""
    return _update(objects)[0]
