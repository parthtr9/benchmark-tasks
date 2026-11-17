"""On the station table: a low pine wine rack with three square cubbies in a row, fixed to the table, its open front
facing the arms. The left and right cubbies hold a full wine bottle each, neck toward the arms; the middle cubby is
empty. In front of the rack a third full 750 ml bottle (1.2 kg) lies on its side on a black rubber mat, at an angle
to the rack. The robot must put that bottle away in the empty cubby, base first (neck out), pushed in until its base
is against the rack's back panel.

A cubby is 10 cm square and the bottle 7.5 cm across, so a gripper holding the bottle's body cannot go in with it, and
a bottle let go with less than about 11 cm inside has its weight outside the cubby and tips out. A bottle that is
fully in still shows 4 cm of neck.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, _np, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]
DT = 0.05                # one control step (20 Hz)

INSTRUCTION = "Put the wine bottle that lies on the mat away in the empty space of the wine rack, neck out and pushed all the way in."

PINE = (0.80, 0.64, 0.42)
BACK = (0.62, 0.48, 0.30)
GLASS = (0.04, 0.16, 0.09)
LABEL = (0.93, 0.90, 0.80)
FOIL = (0.45, 0.05, 0.10)
RUBBER = (0.06, 0.06, 0.06)

# -- the bottle: a 750 ml Bordeaux bottle, 300 mm long and 75 mm across, 1.2 kg full. Part frame: origin on the axis at
#    the middle of its length, +x toward the base. Body 190 mm (with a paper label band), a 24 mm shoulder, a 29 mm
#    neck with a foil capsule over its last 50 mm. Its centre of mass is about 111 mm from the base.
LENGTH = 0.300
HALF = LENGTH / 2
BODY_R, NECK_R, FOIL_R = 0.0375, 0.0145, 0.0150
SHOULDER_X = -0.040      # the body ends here (190 mm from the base); the neck starts 24 mm further
NECK_X = -0.064
MASS = 1.2
COM_X = 0.038            # centre of mass in the part frame (111.5 mm from the base), from the shapes' volumes


def _bottle_cylinders():
    c = []

    def seg(x0, x1, r, color):
        c.append((((x0 + x1) / 2, 0.0, 0.0), r, abs(x1 - x0), color, "X"))

    seg(HALF, 0.110, BODY_R, GLASS)                 # base end of the body
    seg(0.110, 0.020, BODY_R + 0.0003, LABEL)       # the label band, 90 mm
    seg(0.020, SHOULDER_X, BODY_R, GLASS)
    seg(SHOULDER_X, -0.048, 0.034, GLASS)           # shoulder, three steps
    seg(-0.048, -0.056, 0.028, GLASS)
    seg(-0.056, NECK_X, 0.021, GLASS)
    seg(NECK_X, -0.100, NECK_R, GLASS)              # neck
    seg(-0.100, -HALF, FOIL_R, FOIL)                # foil capsule over the cork end
    return tuple(c)


def _bottle(name, pos, yaw):
    return part(name, pos=tuple(float(v) for v in pos), yaw=float(yaw), mass=MASS, color=GLASS, friction=0.6,
                boxes=(((0.05, 0.0, 0.0), (0.01, 0.01, 0.01), GLASS),),      # hidden inside the body
                cylinders=_bottle_cylinders())


def radius_at(x_local: float) -> float:
    """The bottle's radius at a point along its axis (part frame)."""
    if x_local >= SHOULDER_X:
        return BODY_R
    if x_local >= NECK_X:
        return 0.034
    return FOIL_R


# -- the rack (fixed): 12 mm pine boards, three cubbies 100 x 100 mm and 260 mm deep, a 6 mm back panel. World frame:
#    x away from the arms, y to the robot's left, z up (m); heights below are above the table.
BOARD = 0.012
CUBBY = 0.100
FRONT = 0.330            # the rack's front face
DEPTH = 0.260            # front face to the back panel: a seated bottle's neck stands 40 mm out
BACK_T = 0.006
COL_Y = (-(CUBBY + BOARD), 0.0, CUBBY + BOARD)          # cubby centre lines: right, middle, left
ROW_Z = (BOARD,)                                        # cubby floor (one row)
RACK_HALF_W = 1.5 * CUBBY + 2 * BOARD
RACK_H = CUBBY + 2 * BOARD
EMPTY = ((1, 0),)                                       # (column, row) of the empty cubby: the middle one
STOCKED = ((0, 0), (2, 0))


def _fixed(name, x0, x1, y0, y1, z0, z1, color):
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


def _rack():
    out = []
    for i, z in enumerate((0.0, BOARD + CUBBY)):                                      # bottom and top boards
        out.append(_fixed(f"rack_shelf_{i}", FRONT, FRONT + DEPTH, -RACK_HALF_W, RACK_HALF_W, z, z + BOARD, PINE))
    for r, z in enumerate(ROW_Z):                                                      # four uprights
        for k in range(4):
            y = -RACK_HALF_W + k * (CUBBY + BOARD)
            out.append(_fixed(f"rack_upright_{r}_{k}", FRONT, FRONT + DEPTH, y, y + BOARD, z, z + CUBBY, PINE))
    out.append(_fixed("rack_back", FRONT + DEPTH, FRONT + DEPTH + BACK_T, -RACK_HALF_W, RACK_HALF_W, 0.0, RACK_H, BACK))
    return out


def cubby_axis(col: int, row: int) -> tuple[float, float]:
    """(y, height above the table) of the axis of a bottle lying in a cubby."""
    return COL_Y[col], ROW_Z[row] + BODY_R


# -- layout ------------------------------------------------------------------------------------------------------------
MAT_T = 0.005
MAT_XY, MAT_SIZE = (0.14, -0.10), (0.36, 0.12)          # the mat's long side lies along the bottle
START_XY, START_YAW = (0.14, -0.10), -math.pi / 3       # lying 60 degrees off the rack's depth, base toward the right
START_Z = MAT_T + BODY_R + 0.0005
SEATED_X = FRONT + DEPTH - HALF                          # the origin's x of a bottle whose base is on the back panel

BOTTLE = _bottle("bottle", (*START_XY, TOP + START_Z), START_YAW)
STOCK = [_bottle(f"stocked_{i}", (SEATED_X - 0.001, cubby_axis(c, r)[0], TOP + cubby_axis(c, r)[1] + 0.0005), 0.0)
         for i, (c, r) in enumerate(STOCKED)]
MAT = Box("bar_mat", pos=(MAT_XY[0], MAT_XY[1], TOP + MAT_T / 2), size=(MAT_SIZE[0], MAT_SIZE[1], MAT_T), color=RUBBER,
          yaw=START_YAW)
OBJECTS = [*_rack(), MAT, *STOCK, BOTTLE]
STOCK_NAMES = [s.name for s in STOCK]

# -- scoring -----------------------------------------------------------------------------------------------------------
FIT = (CUBBY - 2 * BODY_R) / 2 + 0.003   # the base's centre is this close to a cubby's axis when it is inside it
IN_MIN = 0.020           # 0.25: the base at least 2 cm inside an empty cubby
REST_MIN = 0.150         # 0.5: at least half its length inside (it rests there by itself)
PROUD_MAX = 0.045        # fully in: the cork end at most 45 mm out of the rack's front face (40 mm when on the back panel)
LEVEL = 3.0              # degrees from level
RELEASE = 0.030          # every part of both hands (fingers and palm) this far from the bottle's surface: let go
DROP, DROP_STEPS = 0.020, 4      # an end of the bottle coming down more than 2 cm within 0.2 s: dropped or tipped out
MOVED = 0.010            # a stocked bottle shifted more than 1 cm: disturbed
STILL_CALLS = 20         # 1 s at 20 Hz
SETTLE_CALLS = 20        # the first 1 s: things settle, nothing is judged

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(calls=0, stock0=None, ends=[], rec=[], still=False, released=False, finger_check=None,
                  failed=False, dropped=False, disturbed=False, why=[], fit=None, tilt=0.0)


_reset_state()


def ends(b: PartState):
    """World positions of the base's centre and the cork end's centre."""
    return b.to_world((HALF, 0.0, 0.0)), b.to_world((-HALF, 0.0, 0.0))


def tilt_deg(b: PartState) -> float:
    """The bottle's axis from level, in degrees."""
    base, cork = ends(b)
    return math.degrees(math.asin(min(1.0, abs(float(base[2] - cork[2])) / LENGTH)))


def cubby_fit(b: PartState):
    """Where the bottle is relative to the empty cubbies: None if its base is in neither, else a dict with the cubby,
    ``depth`` (how far the base is past the front face, m) and ``proud`` (how far the cork end sticks out, m). Only a
    bottle lying base first (neck out) counts."""
    base, cork = ends(b)
    if base[0] <= cork[0]:
        return None
    for col, row in EMPTY:
        y, h = cubby_axis(col, row)
        if abs(base[1] - y) <= FIT and abs(base[2] - TOP - h) <= FIT and base[0] > FRONT:
            if abs(cork[1] - y) > CUBBY / 2:
                return None
            return {"cubby": (col, row), "depth": float(base[0] - FRONT), "proud": float(FRONT - cork[0])}
    return None


def surface_dist(b: PartState, point) -> float:
    """Distance from a world point to the bottle's surface (0 inside it)."""
    p = b.to_local(point)
    x = float(np.clip(p[0], -HALF, HALF))
    radial = math.hypot(p[1], p[2])
    return float(math.hypot(p[0] - x, max(0.0, radial - radius_at(x))))


def _flag(name: str, why: str) -> None:
    if not _STATE[name]:
        _STATE[name] = True
        _STATE["why"].append(why)
    _STATE["failed"] = True


def observe(st, fingers=None) -> None:
    """One control step. ``st`` maps part names to states; ``fingers`` is the list of points on both grippers' finger
    and palms, or None when unknown (then the bottle never counts as let go)."""
    S = _STATE
    b = st["bottle"]
    S["calls"] += 1
    base, cork = ends(b)
    if S["calls"] > SETTLE_CALLS:
        if S["stock0"] is None:
            S["stock0"] = {n: np.array(st[n].pos, float) for n in STOCK_NAMES}
        for n in STOCK_NAMES:
            if float(np.linalg.norm(np.asarray(st[n].pos, float) - S["stock0"][n])) > MOVED:
                _flag("disturbed", f"{n} was shifted more than 1 cm")
        hist = S["ends"]
        hist.append((float(base[2]), float(cork[2])))
        del hist[:-(DROP_STEPS + 1)]
        if max(hist[0][0] - hist[-1][0], hist[0][1] - hist[-1][1]) > DROP:
            _flag("dropped", "an end of the bottle came down more than 2 cm in 0.2 s")
    S["fit"] = cubby_fit(b)
    S["tilt"] = tilt_deg(b)
    S["released"] = bool(fingers is not None and min(surface_dist(b, f) for f in fingers) >= RELEASE)
    rec = S["rec"]
    rec.append(np.r_[np.asarray(b.pos, float), base])
    del rec[:-STILL_CALLS]
    S["still"] = bool(len(rec) == STILL_CALLS and float(np.ptp(np.array(rec), axis=0).max()) < 0.001)


def hand_points(environment) -> list:
    """Points over both grippers (world, relative to the env origin): on each finger's inner and outer face along its
    68 mm width, from 13 mm past the tool point to 81 mm behind it, and across the palm's underside. The sizes are the
    gripper's collision shapes. A hand counts as away from the bottle only when all of these are."""
    from yam_arena.constants import TCP_OFFSET

    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    pts = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        pose = _np(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")]
        jp, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
        gap = max(abs(jp[jn.index(j)]) for j in ("joint7", "joint8") if j in jn)
        r = PartState(pos=pose[:3], quat_xyzw=pose[3:7], lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.zeros(3),
                      high=np.zeros(3)).rotation()
        local = [(s * (gap + d), y, z) for s in (1.0, -1.0) for d in (0.0, 0.024) for y in (-0.034, 0.0, 0.034)
                 for z in (0.0128, -0.021, -0.051, -0.081)]
        local += [(x, y, -0.0705) for x in (-0.051, 0.0, 0.051) for y in (-0.0325, 0.0, 0.0325)]
        pts += [pose[:3] + r @ (np.array(TCP_OFFSET) + np.array(q)) - origin for q in local]
    return pts


@on_step
def _watch(environment) -> None:
    from yam_arena.scene import object_states

    st = object_states(environment, ["bottle", *STOCK_NAMES])
    try:
        fingers = hand_points(environment)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - report once; unknown grippers never count as clear
        if _STATE["finger_check"] is None:
            print(f"wine-cubby-restock: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"], fingers = False, None
    observe(st, fingers)


def _update() -> tuple[float, bool]:
    S = _STATE
    fit = S.get("fit")
    if S["failed"] or fit is None or fit["depth"] < IN_MIN:
        return 0.0, False
    level = S["tilt"] <= LEVEL
    if level and fit["proud"] <= PROUD_MAX:
        if S["released"] and S["still"]:
            return 1.0, True
        return 0.75, False
    if level and fit["depth"] >= REST_MIN and S["released"]:
        return 0.5, False
    return 0.25, False


def success(objects) -> bool:
    """The bottle lies base first in the empty cubby, level within 3 degrees, its cork end at most 45 mm out of the
    rack's front face, both grippers at least 3 cm from it, nothing moving more than 1 mm for 20 control steps (1 s),
    and it was never dropped and no stocked bottle was shifted."""
    return _update()[1]


def score(objects) -> float:
    """0 if ever an end of the bottle came down more than 2 cm within 0.2 s (dropped, or let go while its weight was
    still outside the cubby so it tipped out) or a stocked bottle was shifted more than 1 cm (never undone); 0 also
    while the bottle's base is not at least 2 cm inside the empty cubby, base first. Otherwise, by where things are:
    0.25 the base is in the empty cubby; 0.5 at least 15 cm of the bottle is inside, it is level within 3 degrees and
    both grippers are 3 cm away; 0.75 it is level and fully in (cork end at most 45 mm out of the front face) but a
    gripper is within 3 cm or it has not been still for 1 s; 1.0 fully in, level, grippers away and still for 1 s."""
    return _update()[0]


def distance(objects) -> float:
    """How far the bottle's base still is from the back panel of the empty cubby (m)."""
    base, _ = ends(objects["bottle"])
    y, h = cubby_axis(*EMPTY[0])
    target = np.array([FRONT + DEPTH, y, TOP + h])
    fit = _STATE.get("fit")
    if fit is not None:
        return float(max(0.0, DEPTH - fit["depth"]))
    return float(np.linalg.norm(base - target))
