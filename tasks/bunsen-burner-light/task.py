"""On the station table, in front of the arms: a Bunsen burner for bottled gas (Eisco CH0086: 76 mm cast base, 12 mm
barrel, mouth 127 mm above the table) at the middle; to its right a bench gas tap, a brass quarter-turn ball valve on
a bench plate with a yellow lever ending in an upright black grip; to its left a single-flint spark striker (Forney
86102, 197 mm, 73 g) lying in a slotted holder. A blue 450 g butane/propane cartridge with a 29 mbar regulator stands
at the back right of the table; orange 8 mm gas hose runs on the table from the regulator to the tap's inlet and from
the tap's outlet to the burner's inlet arm.

The tap's lever turns on a vertical axis and clicks into two detents, shut (across the valve) and a quarter turn open
(along it). Squeezing the striker's grip zone from 51 mm to 29 mm drags the flint across the file in its hood, 10 cm
ahead of the grip zone, and makes a spark. Gas flows once the tap is more than 15 degrees open and is full at 90. A
spark lights the burner only if the hood is within 2.5 cm of the mouth while the gas flows (flow of 0.3 or more). Gas
that has run for more than 1.5 s without a flame exceeds the trial's unlit-gas safety limit: the run is spoilt. Gas,
spark and flame are computed in code from the lever angle, the squeeze and the hood's position; the hoses and the
cartridge are fixed, rigid shapes.

The robot must light the burner with the striker, using the tap beside it, then put the striker back in its
holder and let go.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import meshkit as mk  # noqa: E402
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
CAST = (0.16, 0.17, 0.19)
YELLOW = (0.95, 0.78, 0.08)
ORANGE = (0.95, 0.42, 0.05)
BLUE = (0.08, 0.30, 0.68)
GREY = (0.72, 0.73, 0.75)
FLAME = (1.0, 0.55, 0.10)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000),
            ((x1 - x0) / 1000, (y1 - y0) / 1000, (z1 - z0) / 1000), color)


# -- the gas tap: a quarter-turn brass ball valve lying along y on a bench plate, stem up --------------------------------
# Origin on the table under the lever's vertical axis; the lever turns about z, 7 cm above the table.
TAP_XY = (0.25, -0.15)
TAP_Z = 0.07                      # the lever's height above the table
LEVER_LEN = 0.05                  # axis to the grip's centre
GRIP_BLOCK = (0.025, 0.025, 0.03)
OPEN_ANGLE = math.pi / 2          # 90 deg CCW seen from above: shut = lever across the valve (+x), open = along it (+y)
DETENT_K = 0.06                   # N m/rad toward the nearer detent: about 0.05 N m of click (1 N at the grip)
VALVE_AXIS_Z = 0.032              # the valve body's and both hose tails' centre line above the table
VALVE_BODY = (0.012, 0.050)       # radius, length along y
TAIL = (0.005, 0.025)             # hose tail radius and length (the 8 mm bore hose pushes over it)
TAIL_TIP = VALVE_BODY[1] / 2 + TAIL[1]        # 50 mm from the axis, each way along y
TAP_BOXES = [
    _mm(-30, 30, -30, 30, 0, 12, STEEL),                      # bench plate, 60 x 60 x 12 mm
    _mm(-10, 10, -15, 15, 12, 21, STEEL),                     # saddle under the valve body
]
TAP_CYLINDERS = [
    ((0.0, 0.0, VALVE_AXIS_Z), VALVE_BODY[0], VALVE_BODY[1], BRASS, "Y"),                       # valve body
    ((0.0, 0.0, 0.052), 0.006, 0.016, BRASS),                                                    # stem, up to the lever
    ((0.0, -(VALVE_BODY[1] + TAIL[1]) / 2, VALVE_AXIS_Z), TAIL[0], TAIL[1], BRASS, "Y"),        # inlet tail (-y)
    ((0.0, (VALVE_BODY[1] + TAIL[1]) / 2, VALVE_AXIS_Z), TAIL[0], TAIL[1], BRASS, "Y"),         # outlet tail (+y)
]
TAP_LEVER = Hinge(
    "lever", pivot=(0.0, 0.0, TAP_Z),
    boxes=(_mm(0, 40, -6, 6, 62.5, 77.5, YELLOW),
           _mm(1000 * (LEVER_LEN - GRIP_BLOCK[0] / 2), 1000 * (LEVER_LEN + GRIP_BLOCK[0] / 2),
               -1000 * GRIP_BLOCK[1] / 2, 1000 * GRIP_BLOCK[1] / 2,
               1000 * (TAP_Z - GRIP_BLOCK[2] / 2), 1000 * (TAP_Z + GRIP_BLOCK[2] / 2), DARK)),
    cylinders=(((0.0, 0.0, TAP_Z), 0.012, 0.015, YELLOW),),
    lower=0.0, upper=OPEN_ANGLE, stiffness=DETENT_K, damping=0.02, target=0.0, mass=0.05, axis="Z")
TAP = part("tap", pos=(*TAP_XY, TOP), mass=3.0, boxes=TAP_BOXES, cylinders=TAP_CYLINDERS, hinges=(TAP_LEVER,),
           color=BRASS, friction=1.0)

# -- the burner: Eisco CH0086 (bottled gas), 127 mm tall, 76 mm base, 12 mm barrel, 0.23 kg ------------------------------
BURNER_XY = (0.31, 0.12)
MOUTH_Z = 0.127
BARREL_R = 0.006
BASE_R, BASE_H = 0.038, 0.012
BURNER_MASS = 0.23
INLET_Z = 0.022                   # the inlet arm's centre line above the table
INLET_TIP = 0.045                 # its tip, from the barrel's axis toward -y (toward the tap)
BURNER = part("burner", pos=(*BURNER_XY, TOP), mass=BURNER_MASS,
              boxes=[_mm(-8, 8, -8, 8, 12, 30, BRASS)],                                   # jet block at the barrel's foot
              cylinders=[((0.0, 0.0, BASE_H / 2), BASE_R, BASE_H, CAST),                 # cast base, 76 mm across
                         ((0.0, 0.0, (MOUTH_Z + BASE_H) / 2), BARREL_R, MOUTH_Z - BASE_H, STEEL),   # barrel
                         ((0.0, 0.0, 0.042), 0.009, 0.020, BRASS),                       # air collar
                         ((0.0, -(INLET_TIP + 0.008) / 2, INLET_Z), 0.004, INLET_TIP - 0.008, BRASS, "Y")],  # inlet arm
              color=STEEL, friction=1.0)
BURNER_HOME = np.array([BURNER_XY[0], BURNER_XY[1], TOP])
MOUTH = np.array([BURNER_XY[0], BURNER_XY[1], TOP + MOUTH_Z])

# -- the striker: Forney 86102 single-flint striker, 197 x 51 mm, 73 g ---------------------------------------------------
# Origin at the middle of its grip zone; x toward the hood, y left, z up. The spring frame is modelled as a rigid spine
# (rear loop, bar, hood) with the two arms' grip sections as plates on springs.
HOLDER_XY = (0.22, 0.20)
GRIP_Z = 0.06                     # the grip zone's height above the table in the holder
CUP_DX = 0.10                     # the hood's centre, ahead of the grip zone
REAR_DX = -0.085                  # the rear loop's end, behind the grip zone
OPEN_GAP, SQUEEZED_GAP = 0.051, 0.029
PAD_T = 0.004                     # thickness of each arm's grip section
PAD_TRAVEL = (OPEN_GAP - SQUEEZED_GAP) / 2
PAD_RATE = 400.0                  # N/m per arm: 4.4 N to close the grip zone from 51 to 29 mm
STRIKER_MASS = 0.073
STRIKER = part(
    "striker", pos=(*HOLDER_XY, TOP + GRIP_Z), mass=STRIKER_MASS, friction=1.5, color=STEEL,
    boxes=[_mm(-25, 100, -3, 3, -6, 6, STEEL),                          # the bar from the rear loop to the hood
           _mm(1000 * REAR_DX, -25, -8, 8, -7.5, 7.5, STEEL),           # the rear spring loop
           _mm(88, 112, -12, 12, -4, 4, BRASS)],                        # the square hood with the file, 24 mm
    # (bar 9.0 cm3 at +37.5 mm, hood 4.6 cm3 at +100 mm, loop 14.4 cm3 at -55 mm: the centre of mass is in the grip zone)
    sliders=(
        Slider("pad_l", rest=(0.0, OPEN_GAP / 2 - PAD_T / 2, 0.0),
               boxes=(((0.0, OPEN_GAP / 2 - PAD_T / 2, 0.0), (0.030, PAD_T, 0.020), STEEL),),
               travel=PAD_TRAVEL, rate=PAD_RATE, damping=3.0, mass=0.002, axis="Y", sign=-1),
        Slider("pad_r", rest=(0.0, -OPEN_GAP / 2 + PAD_T / 2, 0.0),
               boxes=(((0.0, -OPEN_GAP / 2 + PAD_T / 2, 0.0), (0.030, PAD_T, 0.020), STEEL),),
               travel=PAD_TRAVEL, rate=PAD_RATE, damping=3.0, mass=0.002, axis="Y", sign=1)))
# The holder: a base plate, a floor under the striker's rear loop between two side walls (4.5 mm clear each side), and a
# post under the bar just ahead of the grip zone, so the striker lies level with its grip zone at GRIP_Z.
_TAIL_BOTTOM = 1000 * (GRIP_Z - 0.0075)
_BAR_BOTTOM = 1000 * (GRIP_Z - 0.006)
HOLDER = part("holder", pos=(HOLDER_XY[0], HOLDER_XY[1], TOP), mass=1.0, friction=1.0, color=DARK,
              boxes=[_mm(-95, 45, -35, 35, 0, 8, DARK),
                     _mm(-80, -30, -8, 8, 8, _TAIL_BOTTOM, DARK),                      # floor under the rear loop
                     _mm(-80, -30, 12.5, 18.5, 8, _TAIL_BOTTOM + 15, DARK),               # side walls
                     _mm(-80, -30, -18.5, -12.5, 8, _TAIL_BOTTOM + 15, DARK),
                     _mm(-10, 40, -4.5, 4.5, 8, _BAR_BOTTOM, DARK)])                      # post under the bar

# -- the gas supply and the two hose runs: fixed, rigid shapes (meshkit) -------------------------------------------------
SUPPLY_XY = (0.60, -0.40)
CART_R, CART_H = 0.055, 0.140     # Campingaz CV 470 Plus: 110 mm across, 140 mm tall
REG_R, REG_Z0, REG_H = 0.030, 0.153, 0.030    # regulator body: a 60 mm disc on the cartridge's valve collar
NOZZLE_Z = REG_Z0 + REG_H / 2     # the regulator's hose nozzle, pointing back toward the arms (-x)
NOZZLE_TIP = 0.060                # its tip, from the cartridge's axis
HOSE_R = 0.007                    # 8 mm bore, 14 mm outside
HOSE_GAP = 0.003                  # the colliding hose stops this short of a loose part's tail; a sleeve covers the joint


def _spline(points, step: float = 0.004) -> list[tuple]:
    """A smooth path through ``points``, sampled about every ``step`` metres: Catmull-Rom in x and y, an eased step
    in z between one point's height and the next (so the hose never dips below a point's height and leaves each
    fitting level)."""
    p = [np.asarray(q, float) for q in points]
    p = [2 * p[0] - p[1], *p, 2 * p[-1] - p[-2]]
    out = []
    for i in range(1, len(p) - 2):
        n = max(2, int(np.linalg.norm(p[i + 1] - p[i]) / step))
        for k in range(n):
            t = k / n
            q = 0.5 * ((2 * p[i]) + (-p[i - 1] + p[i + 1]) * t
                       + (2 * p[i - 1] - 5 * p[i] + 4 * p[i + 1] - p[i + 2]) * t * t
                       + (-p[i - 1] + 3 * p[i] - 3 * p[i + 1] + p[i + 2]) * t ** 3)
            q[2] = p[i][2] + (p[i + 1][2] - p[i][2]) * t * t * (3 - 2 * t)
            out.append(q)
    out.append(p[-2])
    return [tuple(float(v) for v in q) for q in out]


def path_length(path) -> float:
    q = np.asarray(path, float)
    return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())


_FLOOR = HOSE_R + 0.001           # the centre line of a hose lying on the table
_IN_TIP = np.array([TAP_XY[0], TAP_XY[1] - TAIL_TIP, VALVE_AXIS_Z])           # the tap's inlet tail tip
_OUT_TIP = np.array([TAP_XY[0], TAP_XY[1] + TAIL_TIP, VALVE_AXIS_Z])          # the tap's outlet tail tip
_ARM_TIP = np.array([BURNER_XY[0], BURNER_XY[1] - INLET_TIP, INLET_Z])        # the burner's inlet arm tip
_NOZ_TIP = np.array([SUPPLY_XY[0] - NOZZLE_TIP, SUPPLY_XY[1], NOZZLE_Z])      # the regulator's nozzle tip
# Regulator to tap inlet: off the nozzle, down to the table in front of the cartridge, along the table outside the
# right arm's work area and up into the inlet tail. World x, y; z above the table.
SUPPLY_HOSE_PATH = _spline([
    _NOZ_TIP, (0.360, -0.400, _FLOOR), (0.315, -0.385, _FLOOR), (0.275, -0.335, _FLOOR), (0.252, -0.272, _FLOOR),
    _IN_TIP + [0.0, -HOSE_GAP, 0.0]])
# Tap outlet to burner: off the outlet tail, down to the table, across the middle and up into the inlet arm.
BURNER_HOSE_PATH = _spline([
    _OUT_TIP + [0.0, HOSE_GAP, 0.0], (0.252, -0.035, _FLOOR), (0.280, 0.000, _FLOOR), (0.308, 0.028, _FLOOR),
    _ARM_TIP + [0.0, -HOSE_GAP, 0.0]])
_ON_NOZZLE, _ON_TAIL, _ON_ARM = NOZZLE_TIP - REG_R - 0.004, TAIL[1] - 0.003, 0.025   # hose pushed over each fitting
SUPPLY_HOSE_LEN = path_length(SUPPLY_HOSE_PATH) + HOSE_GAP + _ON_NOZZLE + _ON_TAIL
BURNER_HOSE_LEN = path_length(BURNER_HOSE_PATH) + 2 * HOSE_GAP + _ON_TAIL + _ON_ARM


def _sleeve(tip, direction, length, clip_at=0.010):
    """The hose end pushed over a tail or nozzle (from ``tip`` back along ``direction``) with its worm-drive clip:
    visual only, so it never pushes on the loose part it covers."""
    tip, d = np.asarray(tip, float), np.asarray(direction, float)
    return [mk.Shape(mk.tube([tuple(tip - (HOSE_GAP + 0.001) * d), tuple(tip + length * d)], HOSE_R), color=ORANGE,
                     collider="none", roughness=0.7),
            mk.Shape(mk.tube([tuple(tip + (clip_at - 0.004) * d), tuple(tip + (clip_at + 0.004) * d)], HOSE_R + 0.002),
                     color=STEEL, collider="none", metallic=0.6)]


SUPPLY = mk.mesh_part("supply", pos=(*SUPPLY_XY, TOP), shapes=[
    # cartridge: straight wall, domed shoulder, valve collar
    mk.Shape(mk.revolve([(0.0, 0.0), (0.053, 0.0), (CART_R, 0.003), (CART_R, 0.112), (0.050, 0.127), (0.032, 0.136),
                         (0.016, CART_H), (0.0, CART_H)]), color=BLUE, metallic=0.3),
    mk.Shape(mk.revolve([(0.0, CART_H - 0.001), (0.014, CART_H - 0.001), (0.014, REG_Z0 + 0.001),
                         (0.0, REG_Z0 + 0.001)]), color=DARK),
    # regulator: body disc with a domed cover, hose nozzle toward -x, on/off lever on top
    mk.Shape(mk.revolve([(0.0, REG_Z0), (REG_R, REG_Z0), (REG_R, REG_Z0 + REG_H), (0.022, REG_Z0 + REG_H + 0.008),
                         (0.0, REG_Z0 + REG_H + 0.010)]), color=GREY, metallic=0.4),
    mk.Shape(mk.tube([(-REG_R + 0.002, 0.0, NOZZLE_Z), (-NOZZLE_TIP, 0.0, NOZZLE_Z)], 0.005), color=BRASS, metallic=0.6),
    mk.Shape(mk.box((0.030, 0.012, 0.008), (0.006, 0.0, REG_Z0 + REG_H + 0.012)), color=DARK),
])
SUPPLY_HOSE = mk.mesh_part("supply_hose", pos=(0.0, 0.0, TOP), shapes=[
    mk.Shape(mk.tube(SUPPLY_HOSE_PATH, HOSE_R), color=ORANGE, roughness=0.7),
    *_sleeve(_NOZ_TIP, (1.0, 0.0, 0.0), _ON_NOZZLE),
    *_sleeve(_IN_TIP, (0.0, 1.0, 0.0), _ON_TAIL)])
BURNER_HOSE = mk.mesh_part("burner_hose", pos=(0.0, 0.0, TOP), shapes=[
    mk.Shape(mk.tube(BURNER_HOSE_PATH, HOSE_R), color=ORANGE, roughness=0.7),
    *_sleeve(_OUT_TIP, (0.0, -1.0, 0.0), _ON_TAIL),
    *_sleeve(_ARM_TIP, (0.0, 1.0, 0.0), _ON_ARM)])

# The flame: visual only. It sits above the mouth while burning and is parked under the table otherwise.
FLAME_UP = (BURNER_XY[0], BURNER_XY[1], TOP + MOUTH_Z + 0.025)
FLAME_AWAY = (BURNER_XY[0], BURNER_XY[1], TOP - 0.30)
FLAME_PART = part("flame", pos=FLAME_AWAY, mass=0.001, boxes=[((0, 0, 0), (0.012, 0.012, 0.05), FLAME)], ghost=True,
                  color=FLAME)

OBJECTS = [BURNER, TAP, STRIKER, HOLDER, SUPPLY, SUPPLY_HOSE, BURNER_HOSE, FLAME_PART]

# -- mechanism and scoring thresholds ---------------------------------------------------------------------------------------
GAS_START, GAS_FULL = math.radians(15), math.radians(90)
UNLIT_LIMIT_S = 1.5               # the trial's unlit-gas safety limit: gas running unlit for longer than this in total spoils the run
SPARK_FROM, SPARK_TO = 0.037, 0.031   # grip gap (m): armed once it has been >= 37 mm, a spark when it gets to <= 31 mm
SPARK_REACH = 0.025               # hood within 2.5 cm of the mouth's centre
LIGHT_FLOW = 0.3                  # flow needed for a spark to light the burner
FLAME_FLOW = 0.2                  # the flame burns while flow is at least this
OPEN_ENOUGH = math.radians(60)    # tap counts as open for the score
HOME_TOL = 0.010                  # striker's grip zone within 1 cm of its holder pose
HOME_TILT = 10.0                  # deg
DROP_FALL = 0.030                 # a released fall: the striker came down more than 3 cm within FALL_STEPS control steps ...
FALL_STEPS = 5                    # ... (0.2 s) ...
HELD_DIST = 0.030                 # ... while no gripper's pad tips or tool point were within 3 cm of its centre line
TABLE_HALF = (0.55, 0.45)         # |y| beyond this, or x outside 0..0.55: off the table
KNOCK_MOVE = 0.020                # the burner's base moved more than 2 cm from its place ...
KNOCK_TILT = 15.0                 # ... or the burner tilted more than 15 degrees: knocked
CLEAR = 0.030                     # grippers at least 3 cm from the striker and the tap grip
TOUCH = 0.020                     # a gripper point this close to a part it should not touch (barrel: of its surface)
STILL_CALLS = 20
STILL_TOL = 0.001

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(unlit=0.0, over_limit=False, lit=False, lit_ever=False, armed=False, sparks=0, dropped=False,
                  knocked=False, touched=None, gap=OPEN_GAP, recent=[], zs=[], held_descent=0.0,
                  points={"left": [], "right": []}, target=0.0, flame_on=False)
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
    """Outer width of the striker's grip zone (m): 51 mm open, 29 mm squeezed."""
    return float(OPEN_GAP + min(striker.sliders.get("pad_l", 0.0), 0.0) - max(striker.sliders.get("pad_r", 0.0), 0.0))


def cup_point(striker: PartState) -> np.ndarray:
    return np.asarray(striker.to_world((CUP_DX, 0.0, 0.0)), float)


def tap_grip_point(tap: PartState) -> np.ndarray:
    """World centre of the lever's grip."""
    a = lever_angle(tap)
    return np.asarray(tap.to_world((LEVER_LEN * math.cos(a), LEVER_LEN * math.sin(a), TAP_Z)), float)


def _tilt(state) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, state.rotation()[2, 2]))))


def striker_tilt(striker: PartState) -> float:
    return _tilt(striker)


def at_home(striker: PartState) -> bool:
    home = np.array([HOLDER_XY[0], HOLDER_XY[1], TOP + GRIP_Z])
    return bool(np.linalg.norm(striker.pos - home) <= HOME_TOL and striker_tilt(striker) <= HOME_TILT)


def burner_knocked(burner) -> bool:
    """The burner's base is more than 2 cm from its place, or the burner is tilted more than 15 degrees."""
    return bool(np.linalg.norm(np.asarray(burner.pos, float)[:2] - BURNER_HOME[:2]) > KNOCK_MOVE
                or _tilt(burner) > KNOCK_TILT)


def _seg_dist(p, a, b) -> float:
    a, b, p = np.asarray(a, float), np.asarray(b, float), np.asarray(p, float)
    t = float(np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * (b - a))))


def _striker_segment(striker: PartState):
    return striker.to_world((REAR_DX, 0.0, 0.0)), striker.to_world((CUP_DX, 0.0, 0.0))


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


def held(striker: PartState, points) -> bool:
    """Whether a gripper has the striker: the pad tips or the tool point of either gripper (the first two points of
    each) are within 3 cm of the striker's centre line."""
    a, b = _striker_segment(striker)
    pads = [p for side in ("left", "right") for p in points.get(side, [])[:2]]
    return _near(pads, HELD_DIST, a, b)


def _touch(striker: PartState, tap: PartState, points) -> str | None:
    """Whether a gripper is touching the burner's barrel (hot in use). Which arm works the tap and which the striker
    is the robot's choice."""
    for side in ("left", "right"):
        for p in points.get(side, []):
            d = math.hypot(p[0] - BURNER_XY[0], p[1] - BURNER_XY[1])
            if d < BARREL_R + TOUCH / 2 and TOP + 0.015 < p[2] < TOP + MOUTH_Z + 0.02:
                return "burner barrel"
    return None


# -- the mechanism: run once per control step ---------------------------------------------------------------------------------
def _advance(objects, points=None, dt: float = 1.0 / CONTROL_HZ) -> dict:
    """Gas, spark, flame, the unlit-gas limit and the violation flags, from the parts' poses and the grippers' points."""
    s = _STATE
    striker, tap = objects["striker"], objects["tap"]
    angle = lever_angle(tap)
    f = flow(angle)
    s["target"] = OPEN_ANGLE if angle >= OPEN_ANGLE / 2 else 0.0      # the detents
    if points is not None:
        s["points"] = points
    # Spark: a squeeze from >= 37 mm down to <= 31 mm.
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
        if s["unlit"] > UNLIT_LIMIT_S + 1e-6:      # (the sum of 30 steps of 0.05 s is not exactly 1.5)
            s["over_limit"] = True
    # Violations (permanent).
    # A drop is a released fall: the striker came down more than 3 cm within 0.2 s and no gripper has it. Lowering it
    # in a gripper, however fast, is not a drop (the descent is only logged, as held_descent).
    zs = s["zs"]
    zs.append(float(striker.pos[2]))
    del zs[:-FALL_STEPS]
    descent = max(zs) - zs[-1]
    if held(striker, s["points"]):
        s["held_descent"] = max(s["held_descent"], descent)
    elif descent > DROP_FALL:
        s["dropped"] = True
    if not (0.0 <= striker.pos[0] <= TABLE_HALF[0] and abs(striker.pos[1]) <= TABLE_HALF[1]):
        s["dropped"] = True        # left the table
    burner = objects.get("burner")
    if burner is not None and burner_knocked(burner):
        s["knocked"] = True
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

    objects = {p.name: _part_state(environment, p) for p in (TAP, STRIKER, BURNER)}
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
    if s["over_limit"] or s["dropped"] or s["knocked"] or s["touched"] or not s["lit_ever"]:
        return 0.0, False
    if not s["lit"]:
        return 0.25, False
    if not (at_home(striker) and lever_angle(tap) >= OPEN_ENOUGH):
        return 0.5, False
    if not _clear_of_grippers(striker, tap) or not _still():
        return 0.75, False
    return 1.0, True


def success(objects) -> bool:
    """The burner was lit within the unlit-gas limit, is burning, the tap is open 60 degrees or more, the striker is back in its
    holder (grip zone within 1 cm of the holder pose, tilted under 10 degrees), both grippers are 3 cm or more from
    the striker and the tap's grip, the striker has not moved more than 1 mm for 20 control steps (1 s), and nothing
    was dropped, knocked or touched on the way."""
    return _grade(objects)[1]


def score(objects) -> float:
    """0 if the gas ever ran more than 1.5 s unlit in total (the trial's unlit-gas safety limit), the striker was
    dropped (a released fall: it came down more than 3 cm within 0.2 s while no gripper's pad tips or tool point were
    within 3 cm of it; or it left the table), the burner was knocked (base moved more than 2 cm, or tilted more than
    15 degrees), a gripper touched the burner's barrel, or the burner was never lit. Lowering the striker in a gripper
    is never a drop. Otherwise 0.25 the burner was lit but the flame is out at the end; 0.5 it is burning but the
    striker is not in its holder or the tap is open less than 60 degrees; 0.75 burning, tap open 60 degrees or more
    and striker in its holder, but a gripper is within 3 cm of the striker or the tap's grip, or the striker has not
    yet been still for 1 s; 1.0 all of that settled."""
    return _grade(objects)[0]


def distance(objects) -> float:
    """Metres to go: the hood to the mouth until the burner has been lit, then the striker to its holder."""
    striker = objects["striker"]
    if not _STATE["lit_ever"]:
        return float(np.linalg.norm(cup_point(striker) - MOUTH))
    return float(np.linalg.norm(striker.pos - np.array([HOLDER_XY[0], HOLDER_XY[1], TOP + GRIP_Z])))
