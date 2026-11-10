"""On the station table, in front of the arms: a weighted Bunsen burner (a 9 cm base and a 16 mm barrel, mouth 15 cm
above the table) at the middle, a brass bench gas tap on the right with a short lever ending in a square grip block,
and, on the left, a steel cup-type spark striker standing in a slotted holder. The tap's lever turns on a vertical
axis and clicks into two detents, shut and a quarter turn open. The striker is two spring arms hinged at the rear:
squeezing its grip zone from 30 mm to 8 mm drags a flint across the cup, 10 cm ahead of the grip zone, and makes a
spark. Gas flows once the tap is more than 15 degrees open and is full at 90. A spark lights the burner only if the
cup is within 2.5 cm of the mouth while the gas flows (flow of 0.3 or more). Gas that has run for more than 1.5 s
without a flame builds a cloud that flares when it lights: the run is spoilt.

The robot must light the burner with the striker, using the tap beside it, then put the striker back in its holder and
let go. Opening the gas takes a hand at the tap; the striker has to be at the mouth, ready to strike, before that.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Hinge, PartState, Slider, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]
CONTROL_HZ = 20

INSTRUCTION = ("Light the Bunsen burner with the spark striker, using the gas tap beside it, then put the striker "
               "back in its holder and let go.")

BRASS = (0.78, 0.60, 0.22)
STEEL = (0.62, 0.64, 0.68)
DARK = (0.22, 0.23, 0.25)
RED = (0.80, 0.12, 0.10)
FLAME = (1.0, 0.55, 0.10)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000),
            ((x1 - x0) / 1000, (y1 - y0) / 1000, (z1 - z0) / 1000), color)


# -- the gas tap: origin on the table under the lever's vertical axis; the lever turns about z at 7 cm ------------------
TAP_XY = (0.25, -0.15)
TAP_Z = 0.07                      # the lever's height above the table
LEVER_LEN = 0.05                  # axis to the grip block's centre
GRIP_BLOCK = (0.025, 0.025, 0.03)
OPEN_ANGLE = math.pi / 2          # 90 deg CCW seen from above
DETENT_K = 0.06                   # N m/rad toward the nearer detent: about 0.05 N m of click (1 N at the grip block)
TAP_BOXES = [
    _mm(-30, 30, -30, 30, 0, 12, BRASS),                      # base plate
]
TAP_CYLINDERS = [((0.0, 0.0, 0.036), 0.012, 0.048, BRASS)]    # the pipe body under the lever
TAP_LEVER = Hinge(
    "lever", pivot=(0.0, 0.0, TAP_Z),
    boxes=(_mm(0, 40, -6, 6, 62.5, 77.5, RED),
           _mm(1000 * (LEVER_LEN - GRIP_BLOCK[0] / 2), 1000 * (LEVER_LEN + GRIP_BLOCK[0] / 2),
               -1000 * GRIP_BLOCK[1] / 2, 1000 * GRIP_BLOCK[1] / 2,
               1000 * (TAP_Z - GRIP_BLOCK[2] / 2), 1000 * (TAP_Z + GRIP_BLOCK[2] / 2), DARK)),
    cylinders=(((0.0, 0.0, TAP_Z), 0.012, 0.015, RED),),
    lower=0.0, upper=OPEN_ANGLE, stiffness=DETENT_K, damping=0.02, target=0.0, mass=0.05, axis="Z")
TAP = part("tap", pos=(*TAP_XY, TOP), mass=3.0, boxes=TAP_BOXES, cylinders=TAP_CYLINDERS, hinges=(TAP_LEVER,),
           color=BRASS, friction=1.0)

# -- the burner ---------------------------------------------------------------------------------------------------------
BURNER_XY = (0.31, 0.12)
MOUTH_Z = 0.15
BARREL_R = 0.008
BURNER = part("burner", pos=(*BURNER_XY, TOP), mass=1.5,
              boxes=[_mm(-9, 9, -9, 9, 30, 50, BRASS)],       # the air collar
              cylinders=[((0.0, 0.0, 0.0075), 0.045, 0.015, DARK),                      # base, 9 cm across
                         ((0.0, 0.0, 0.0825), BARREL_R, MOUTH_Z - 0.015, STEEL)],      # barrel, mouth at 15 cm
              color=STEEL, friction=1.0)
MOUTH = np.array([BURNER_XY[0], BURNER_XY[1], TOP + MOUTH_Z])

# -- the striker: origin at the middle of its grip zone; x toward the cup, y left, z up -----------------------------------
HOLDER_XY = (0.22, 0.20)
GRIP_Z = 0.06                     # the grip zone's height above the table in the holder
CUP_DX = 0.10                     # the cup's centre, ahead of the grip zone
OPEN_GAP, SQUEEZED_GAP = 0.030, 0.008
PAD_T = 0.004                     # thickness of each spring arm's pad
PAD_TRAVEL = (OPEN_GAP - SQUEEZED_GAP) / 2
PAD_RATE = 200.0                  # N/m per arm: 2.2 N to close the grip zone from 30 to 8 mm
STRIKER_MASS = 0.03
STRIKER = part(
    "striker", pos=(*HOLDER_XY, TOP + GRIP_Z), mass=STRIKER_MASS, friction=1.5, color=STEEL,
    boxes=[_mm(-25, 100, -3, 3, -6, 6, STEEL),                  # the bar from the hinge to the cup
           _mm(-65, -25, -7.5, 7.5, -7.5, 7.5, DARK)],          # the rear: the hinge and the tail
    cylinders=[((CUP_DX, 0.0, 0.0), 0.010, 0.010, BRASS)],      # the flint cup, 2 cm across
    sliders=(
        Slider("pad_l", rest=(0.0, OPEN_GAP / 2 - PAD_T / 2, 0.0),
               boxes=(((0.0, OPEN_GAP / 2 - PAD_T / 2, 0.0), (0.030, PAD_T, 0.020), RED),),
               travel=PAD_TRAVEL, rate=PAD_RATE, damping=3.0, mass=0.002, axis="Y", sign=-1),
        Slider("pad_r", rest=(0.0, -OPEN_GAP / 2 + PAD_T / 2, 0.0),
               boxes=(((0.0, -OPEN_GAP / 2 + PAD_T / 2, 0.0), (0.030, PAD_T, 0.020), RED),),
               travel=PAD_TRAVEL, rate=PAD_RATE, damping=3.0, mass=0.002, axis="Y", sign=1)))
# The holder: a base plate, a floor under the striker's tail between two side walls (5 mm clear each side), and a post
# under the bar just ahead of the grip zone, so the striker lies level with its grip zone at GRIP_Z.
_TAIL_BOTTOM = 1000 * (GRIP_Z - 0.0075)
_BAR_BOTTOM = 1000 * (GRIP_Z - 0.006)
HOLDER = part("holder", pos=(HOLDER_XY[0], HOLDER_XY[1], TOP), mass=1.0, friction=1.0, color=DARK,
              boxes=[_mm(-75, 45, -35, 35, 0, 8, DARK),
                     _mm(-65, -25, -7.5, 7.5, 8, _TAIL_BOTTOM, DARK),                      # floor under the tail
                     _mm(-65, -25, 12.5, 18.5, 8, _TAIL_BOTTOM + 15, DARK),               # side walls
                     _mm(-65, -25, -18.5, -12.5, 8, _TAIL_BOTTOM + 15, DARK),
                     _mm(-10, 40, -4.5, 4.5, 8, _BAR_BOTTOM, DARK)])                      # post under the bar

# The flame: visual only. It sits above the mouth while burning and is parked under the table otherwise.
FLAME_UP = (BURNER_XY[0], BURNER_XY[1], TOP + MOUTH_Z + 0.025)
FLAME_AWAY = (BURNER_XY[0], BURNER_XY[1], TOP - 0.30)
FLAME_PART = part("flame", pos=FLAME_AWAY, mass=0.001, boxes=[((0, 0, 0), (0.014, 0.014, 0.05), FLAME)], ghost=True,
                  color=FLAME)

OBJECTS = [BURNER, TAP, STRIKER, HOLDER, FLAME_PART]

# -- mechanism and scoring thresholds ---------------------------------------------------------------------------------------
GAS_START, GAS_FULL = math.radians(15), math.radians(90)
UNLIT_LIMIT_S = 1.5               # gas running unlit for longer than this builds a cloud that flares
SPARK_FROM, SPARK_TO = 0.022, 0.010   # grip gap (m): armed once it has been >= 22 mm, a spark when it gets to <= 10 mm
SPARK_REACH = 0.025               # cup within 2.5 cm of the mouth's centre
LIGHT_FLOW = 0.3                  # flow needed for a spark to light the burner
FLAME_FLOW = 0.2                  # the flame burns while flow is at least this
OPEN_ENOUGH = math.radians(60)    # tap counts as open for the score
HOME_TOL = 0.010                  # striker's grip zone within 1 cm of its holder pose
HOME_TILT = 10.0                  # deg
DROP_FALL = 0.030                 # striker fell more than 3 cm within FALL_STEPS control steps, or left the table
FALL_STEPS = 5
TABLE_HALF = (0.55, 0.45)         # |y| beyond this, or x outside 0..0.55: off the table
CLEAR = 0.030                     # grippers at least 3 cm from the striker and the tap grip
TOUCH = 0.020                     # a gripper point this close to a part it should not touch (barrel: of its surface)
STILL_CALLS = 20
STILL_TOL = 0.001

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(unlit=0.0, flare=False, lit=False, lit_ever=False, armed=False, sparks=0, dropped=False,
                  touched=None, gap=OPEN_GAP, recent=[], zs=[], points={"left": [], "right": []}, target=0.0, flame_on=False)
    if environment is not None:
        _move_flame(environment, False)


_reset_state()


# -- geometry helpers ------------------------------------------------------------------------------------------------------
def lever_angle(tap: PartState) -> float:
    return float(tap.hinges.get("lever", 0.0))


def flow(angle: float) -> float:
    """Gas flow 0..1: nothing until the lever has turned 15 degrees, full at 90."""
    return float(min(1.0, max(0.0, (angle - GAS_START) / (GAS_FULL - GAS_START))))


def grip_gap(striker: PartState) -> float:
    """Outer width of the striker's grip zone (m): 30 mm open, 8 mm squeezed."""
    return float(OPEN_GAP + min(striker.sliders.get("pad_l", 0.0), 0.0) - max(striker.sliders.get("pad_r", 0.0), 0.0))


def cup_point(striker: PartState) -> np.ndarray:
    return np.asarray(striker.to_world((CUP_DX, 0.0, 0.0)), float)


def tap_grip_point(tap: PartState) -> np.ndarray:
    """World centre of the lever's grip block."""
    a = lever_angle(tap)
    return np.asarray(tap.to_world((LEVER_LEN * math.cos(a), LEVER_LEN * math.sin(a), TAP_Z)), float)


def striker_tilt(striker: PartState) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, striker.rotation()[2, 2]))))


def at_home(striker: PartState) -> bool:
    home = np.array([HOLDER_XY[0], HOLDER_XY[1], TOP + GRIP_Z])
    return bool(np.linalg.norm(striker.pos - home) <= HOME_TOL and striker_tilt(striker) <= HOME_TILT)


def _seg_dist(p, a, b) -> float:
    a, b, p = np.asarray(a, float), np.asarray(b, float), np.asarray(p, float)
    t = float(np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * (b - a))))


def _striker_segment(striker: PartState):
    return striker.to_world((-0.065, 0.0, 0.0)), striker.to_world((CUP_DX, 0.0, 0.0))


def _gripper_points_of(rob_q, base) -> list[np.ndarray]:
    from yam_arena import ik

    tcp, rot = ik.tcp_pose(rob_q, base)
    a = rot[:, 2]
    return [tcp + 0.013 * a, tcp, tcp - 0.04 * a, tcp - 0.08 * a]


def _gripper_points(environment) -> dict:
    """Points along each gripper: pad tips, the tool point and 4 and 8 cm up the gripper body."""
    from yam_arena import station

    out = {}
    for side in ("left", "right"):
        rob = environment.unwrapped.scene[f"{side}_robot"]
        q = np.asarray(getattr(rob.data.joint_pos, "torch", rob.data.joint_pos)[0].detach().cpu(), float)[:6]
        out[side] = _gripper_points_of(q, station.BASES[side])
    return out


def _near(points, dist, a, b=None) -> bool:
    return any((_seg_dist(p, a, b) if b is not None else float(np.linalg.norm(np.asarray(p) - a))) < dist
               for p in points)


def _touch(striker: PartState, tap: PartState, points) -> str | None:
    """Which forbidden thing a gripper is touching: the burner's barrel, or the other side's part (the tap by the
    left gripper, the striker by the right one)."""
    for side in ("left", "right"):
        for p in points.get(side, []):
            d = math.hypot(p[0] - BURNER_XY[0], p[1] - BURNER_XY[1])
            if d < BARREL_R + TOUCH / 2 and TOP + 0.015 < p[2] < TOP + MOUTH_Z + 0.02:
                return "burner barrel"
    pivot = tap.to_world((0.0, 0.0, TAP_Z))
    if _near(points.get("left", []), TOUCH, pivot, tap_grip_point(tap)):
        return "left gripper on the tap"
    a, b = _striker_segment(striker)
    if _near(points.get("right", []), TOUCH, a, b):
        return "right gripper on the striker"
    return None


# -- the mechanism: run once per control step ---------------------------------------------------------------------------------
def _advance(objects, points=None, dt: float = 1.0 / CONTROL_HZ) -> dict:
    """Gas, spark, flame, flare and the violation flags, from the parts' poses and the grippers' points."""
    s = _STATE
    striker, tap = objects["striker"], objects["tap"]
    angle = lever_angle(tap)
    f = flow(angle)
    s["target"] = OPEN_ANGLE if angle >= OPEN_ANGLE / 2 else 0.0      # the detents
    if points is not None:
        s["points"] = points
    # Spark: a squeeze from >= 22 mm down to <= 10 mm.
    gap = grip_gap(striker)
    if gap >= SPARK_FROM:
        s["armed"] = True
    spark = bool(s["armed"] and gap <= SPARK_TO)
    if spark:
        s["armed"] = False
        s["sparks"] += 1
        if (not s["lit"] and f >= LIGHT_FLOW
                and float(np.linalg.norm(cup_point(striker) - MOUTH)) <= SPARK_REACH):
            s["lit"] = s["lit_ever"] = True
    s["gap"] = gap
    # Flame: burns while the flow is 0.2 or more, goes out below that.
    if s["lit"] and f < FLAME_FLOW:
        s["lit"] = False
    # Unlit time: gas flowing with no flame.
    if f > 0.0 and not s["lit"]:
        s["unlit"] += dt
        if s["unlit"] > UNLIT_LIMIT_S:
            s["flare"] = True
    # Violations (permanent).
    zs = s["zs"]
    zs.append(float(striker.pos[2]))
    del zs[:-FALL_STEPS]
    if max(zs) - zs[-1] > DROP_FALL or not (0.0 <= striker.pos[0] <= TABLE_HALF[0]
                                            and abs(striker.pos[1]) <= TABLE_HALF[1]):
        s["dropped"] = True        # fell more than 3 cm within 0.2 s (set down by hand is slower), or left the table
    if s["touched"] is None:
        s["touched"] = _touch(striker, tap, s["points"])
    recent = s["recent"]
    recent.append(np.asarray(striker.pos, float).copy())
    del recent[:-STILL_CALLS]
    return {"spark": spark, "flow": f}


def _move_flame(environment, on: bool) -> None:
    from partkit import write_pose

    write_pose(environment, "flame", FLAME_UP if on else FLAME_AWAY, (0.0, 0.0, 0.0, 1.0))


@on_step
def _control(environment) -> None:
    """Every control step: the detent drive, the mechanism, the flame."""
    from partkit import _part_state, set_hinge_target

    objects = {p.name: _part_state(environment, p) for p in (TAP, STRIKER)}
    _advance(objects, _gripper_points(environment))
    set_hinge_target(environment, "tap", "lever", _STATE["target"])
    if _STATE["lit"] != _STATE["flame_on"]:
        _STATE["flame_on"] = _STATE["lit"]
        _move_flame(environment, _STATE["lit"])


# -- scoring -------------------------------------------------------------------------------------------------------------------
def _clear_of_grippers(striker: PartState, tap: PartState) -> bool:
    a, b = _striker_segment(striker)
    grip = tap_grip_point(tap)
    pts = _STATE["points"].get("left", []) + _STATE["points"].get("right", [])
    return not (_near(pts, CLEAR, a, b) or _near(pts, CLEAR, grip))


def _still() -> bool:
    r = _STATE["recent"]
    return len(r) == STILL_CALLS and float(np.ptp(np.array(r), axis=0).max()) < STILL_TOL


def _grade(objects) -> tuple[float, bool]:
    s = _STATE
    striker, tap = objects["striker"], objects["tap"]
    if s["flare"] or s["dropped"] or s["touched"] or not s["lit_ever"]:
        return 0.0, False
    if not s["lit"]:
        return 0.25, False
    if not (at_home(striker) and lever_angle(tap) >= OPEN_ENOUGH):
        return 0.5, False
    if not _clear_of_grippers(striker, tap) or not _still():
        return 0.75, False
    return 1.0, True


def success(objects) -> bool:
    """The burner was lit without a flare, is burning, the tap is open 60 degrees or more, the striker is back in its
    holder (grip zone within 1 cm of the holder pose, upright within 10 degrees), both grippers are 3 cm or more from
    the striker and the tap's grip block, and the striker has not moved more than 1 mm for 20 control steps (1 s)."""
    return _grade(objects)[1]


def score(objects) -> float:
    """0 if the gas ever ran more than 1.5 s unlit (flare), the striker was dropped (fell 3 cm within 0.2 s or left the table), a
    gripper touched the burner's barrel or the other side's part, or the burner was never lit. Otherwise 0.25 the
    burner was lit but the flame is out at the end; 0.5 it is burning but the striker is not in its holder or the tap
    is open less than 60 degrees; 0.75 burning, tap open 60 degrees or more and striker in its holder, but a gripper
    is within 3 cm of the striker or the tap's grip block, or they are not still; 1.0 all of that settled."""
    return _grade(objects)[0]


def distance(objects) -> float:
    """Metres to go: the cup to the mouth until the burner has been lit, then the striker to its holder."""
    striker = objects["striker"]
    if not _STATE["lit_ever"]:
        return float(np.linalg.norm(cup_point(striker) - MOUTH))
    return float(np.linalg.norm(striker.pos - np.array([HOLDER_XY[0], HOLDER_XY[1], TOP + GRIP_Z])))
