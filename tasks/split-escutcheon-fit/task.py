"""On the station bench: a mock-up wall section with a 5/8 in copper supply stub-out coming through a rough hole, its
quarter-turn angle stop valve already on, as a plumber finds it at trim-out. A one-piece split floor and wall plate
(Sioux Chief 926-2 SnapOne, chrome-plated polypropylene) stands in the groove of a block on the bench in front of it.

A solid escutcheon can no longer go on once the valve is fitted; this one is split on one side and springs shut. The
robot spreads it open at its split, passes it over the pipe between the valve and the wall, lets it spring shut round
the pipe and pushes it flat against the wall over the rough hole.

The plate is modelled as two halves joined at the top, opposite the split, by a sprung hinge that stands in for the
ring's own flex (the polypropylene bends most where the ring is opposite its split). Maker's figures (926 series spec
sheet): ID 0.625 in (15.9 mm), OD 2.53 in (64.3 mm), height 0.45 in (11.4 mm). Wall thicknesses, the bore's sleeve,
the spring and the mass are estimates (measure the bought part).

Frames. World axes: x away from the robot (toward the wall), y to the robot's left, z up. The plate's frame has its
origin on the pipe axis in the plane of its back edge (the plane that touches the wall), the same axes as the world
when it hangs square on the pipe; its split points down (-z). Geometry in this file is written in mm.

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
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = "Fit the chrome plate round the copper pipe and push it flat against the wall so it covers the hole."

CHROME = (0.80, 0.81, 0.83)
COPPER = (0.78, 0.47, 0.30)
BRASS_CHROME = (0.70, 0.71, 0.73)
PLY = (0.83, 0.72, 0.55)
PINE = (0.86, 0.76, 0.58)
DARK = (0.12, 0.12, 0.12)
WHITE = (0.93, 0.93, 0.93)

# -- layout (world, m) ---------------------------------------------------------------------------------------------------
XW = 0.360                     # the wall's front face
PIPE_Z = TOP + 0.120           # the pipe's axis, on y = 0
STAND_X = 0.170                # the grooved block's middle, in front of the valve (its groove runs along y)

# -- dimensions (mm) -----------------------------------------------------------------------------------------------------
WALL_T, WALL_W, WALL_H = 18.0, 300.0, 250.0         # 18 mm plywood wall section on its own foot
HOLE = 30.0                                          # rough hole (square, 30 mm), centred on the pipe
PIPE_R = 15.875 / 2                                  # 5/8 in OD copper (1/2 in nominal), corners on this radius
STUB = 85.0                                          # pipe out from the wall face to the valve's inlet nut (3-1/2 in stub-out)
PIPE_BACK = 30.0                                     # the pipe goes this far into the wall
PLATE_R = 64.26 / 2                                  # 2.53 in OD (corners of the 16-sided rim on this radius)
PLATE_H = 11.43                                      # 0.45 in, back edge to face
BORE_R = 15.875 / 2 + 0.35                           # 0.625 in ID with 0.35 mm play a side at its flats (estimate)
SKIRT_T = 1.5                                        # rim wall (estimate)
FACE_T = 1.5                                         # face (estimate)
SLEEVE_T, SLEEVE_L = 2.0, 7.5                        # the bore's sleeve: wall, length from the face (estimate)
N_HALF = 8                                           # segments per half: 16 round the plate
PLATE_MASS = 0.010                                   # kg (listed 0.02 lb)
SPRING = 0.2                                         # N m/rad: about 1.6 N at the split to spread it 30 degrees (estimate)
PRELOAD = 0.05                                       # rad: the spring's target past closed (it rests shut)
SPREAD_MAX = 0.60                                    # rad: the hinge's open stop
STAND_W, STAND_D, STAND_H = 89.0, 38.0, 113.0         # a 2x4 offcut on end (y, x, z)
GROOVE_W, GROOVE_D = 13.0, 5.0                        # groove along its top for the plate's rim

PIVOT = (-PLATE_H / 2, 0.0, PLATE_R * math.cos(math.pi / (2 * N_HALF)))   # the halves' joint, at the top of the rim


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _quat(R):
    """(w, x, y, z) of a rotation matrix."""
    R = np.asarray(R, float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2 * math.sqrt(tr + 1.0)
        q = (s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = ((R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = 2 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s)
    else:
        s = 2 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4)
    return tuple(float(v) for v in q)


def box(centre, size, color, R=None):
    spec = (tuple(float(v) / 1000 for v in centre), tuple(float(v) / 1000 for v in size), color)
    return spec if R is None else spec + (_quat(R),)


def aabb(x0, x1, y0, y1, z0, z1, color):
    return box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)), color)


def radial(a, r0, r1, x0, x1, color, n=2 * N_HALF, part=0.0):
    """A box spanning radii r0..r1 (flats) on the radial line at angle ``a`` from +z toward +y, x0..x1 along the axis,
    wide enough to meet its neighbours of an ``n``-sided ring at r1. ``part`` != 0: only the half of it on the side of
    increasing (+1) or decreasing (-1) angle, for a flat that the split or the top joint divides."""
    w = 2 * r1 * math.tan(math.pi / n) * 1.02
    rm = (r0 + r1) / 2
    t = np.array([math.cos(a), -math.sin(a)])                # (y, z) along increasing angle
    c = np.array([rm * math.sin(a), rm * math.cos(a)]) + (part * w / 4) * t
    return box(((x0 + x1) / 2, c[0], c[1]), (x1 - x0, w / 2 if part else w, r1 - r0), color, _rx(-a))


def half_boxes(sign: float) -> list:
    """One half of the plate (sign +1: the +y half, -1: the -y half), in the plate frame at the closed position. The
    16-sided outline has a flat at the split (bottom) and one at the top, each divided between the halves, so the
    plate stands on a flat."""
    out = []
    flat = math.cos(math.pi / (2 * N_HALF))
    r_rim = PLATE_R * flat                                 # the rim's flats (corners on PLATE_R)
    for k in range(N_HALF + 1):
        a = sign * (math.pi / N_HALF) * k
        part = sign * (1.0 if k == 0 else -1.0 if k == N_HALF else 0.0)
        out.append(radial(a, r_rim - SKIRT_T, r_rim, -PLATE_H, 0.0, CHROME, part=part))                     # rim skirt
        out.append(radial(a, BORE_R + SLEEVE_T, r_rim - SKIRT_T, -PLATE_H, -PLATE_H + FACE_T, CHROME, part=part))  # face
        out.append(radial(a, BORE_R, BORE_R + SLEEVE_T, -PLATE_H, -PLATE_H + SLEEVE_L, CHROME, part=part))  # bore sleeve
    return out


def prism_x(x0, x1, r, color, y=0.0, z=0.0, n=16):
    """A solid n-sided prism along x, corners on radius r: n/2 slabs as wide as the flats' distance, turned about x."""
    flat = r * math.cos(math.pi / n)
    w = 2 * r * math.sin(math.pi / n) * 1.02
    return [box(((x0 + x1) / 2, y, z), (x1 - x0, w, 2 * flat), color, _rx(math.pi * k / (n // 2)))
            for k in range(n // 2)]


def prism_z(z0, z1, r, color, x=0.0, y=0.0, n=8):
    flat = r * math.cos(math.pi / n)
    w = 2 * r * math.sin(math.pi / n) * 1.02
    out = []
    for k in range(n // 2):
        a = math.pi * k / (n // 2)
        c, s = math.cos(a), math.sin(a)
        out.append(box((x, y, (z0 + z1) / 2), (2 * flat, w, z1 - z0), color, np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])))
    return out


# -- the wall section, pipe and valve (one fixed part; frame at the pipe's axis in the wall's front face) ----------------
def wall_boxes() -> list:
    zf = -(PIPE_Z - TOP) * 1000                  # the bench top in this frame
    h = HOLE / 2
    b = [aabb(0.0, WALL_T, -WALL_W / 2, -h, zf, zf + WALL_H, PLY),
         aabb(0.0, WALL_T, h, WALL_W / 2, zf, zf + WALL_H, PLY),
         aabb(0.0, WALL_T, -h, h, h, zf + WALL_H, PLY),
         aabb(0.0, WALL_T, -h, h, zf, -h, PLY),
         aabb(WALL_T + 0.5, WALL_T + 0.5 + 0.8, -h - 4, h + 4, -h - 4, h + 4, DARK),        # the stud bay behind the hole
         aabb(WALL_T, WALL_T + 90.0, -WALL_W / 2, WALL_W / 2, zf, zf + 18.0, PLY)]           # the foot, behind the wall
    b += prism_x(-STUB, PIPE_BACK, PIPE_R, COPPER)                                           # the stub-out
    # quarter-turn angle stop, 1/2 in nominal compression inlet x 3/8 in OD outlet up (proportions of a common model)
    x_nut = -STUB
    b += prism_x(x_nut - 16.0, x_nut, 14.0, BRASS_CHROME, n=8)                                # inlet nut, 1 in hex (est.)
    b += prism_x(x_nut - 44.0, x_nut - 16.0, 11.0, BRASS_CHROME)                              # body (est.)
    b += prism_z(8.0, 34.0, 7.0, BRASS_CHROME, x=x_nut - 30.0)                                # outlet riser (est.)
    b += prism_z(22.0, 32.0, 8.5, BRASS_CHROME, x=x_nut - 30.0, n=8)                          # outlet nut (est.)
    b += prism_x(x_nut - 56.0, x_nut - 44.0, 4.5, BRASS_CHROME, n=8)                          # stem (est.)
    b.append(aabb(x_nut - 64.0, x_nut - 56.0, -17.0, 17.0, -6.0, 6.0, WHITE))                 # oval handle (est.)
    return b


WALL_ORIGIN = (XW, 0.0, PIPE_Z)
WALL = part("wall", pos=WALL_ORIGIN, boxes=wall_boxes(), mass=10.0, color=PLY, friction=0.4, fixed=True)

# -- the grooved bench block the plate stands in ------------------------------------------------------------------------
def stand_boxes() -> list:
    g = GROOVE_W / 2
    return [aabb(-STAND_D / 2, -g, -STAND_W / 2, STAND_W / 2, 0.0, STAND_H, PINE),
            aabb(g, STAND_D / 2, -STAND_W / 2, STAND_W / 2, 0.0, STAND_H, PINE),
            aabb(-g, g, -STAND_W / 2, STAND_W / 2, 0.0, STAND_H - GROOVE_D, PINE)]


STAND = part("stand", pos=(STAND_X, 0.0, TOP), boxes=stand_boxes(), mass=1.0, color=PINE, friction=0.5, fixed=True)

# -- the plate ------------------------------------------------------------------------------------------------------------
_RIM_LOW = PLATE_R * math.cos(math.pi / (2 * N_HALF))            # its lowest flat sits in the groove
PLATE_START = (STAND_X + PLATE_H / 2000, 0.0, TOP + (STAND_H - GROOVE_D + _RIM_LOW + 0.3) / 1000)
HALF_B = Hinge("half_b", pivot=tuple(v / 1000 for v in PIVOT), boxes=tuple(half_boxes(-1.0)), lower=-SPREAD_MAX,
               upper=0.0, stiffness=SPRING, damping=0.002, target=PRELOAD, mass=PLATE_MASS / 2, axis="X")
PLATE = part("plate", pos=PLATE_START, boxes=half_boxes(1.0), mass=PLATE_MASS / 2, hinges=(HALF_B,), color=CHROME,
             friction=0.35, meta={"torsional_patch": 0.006, "contact_offset": 0.0002})   # polypropylene on copper; the
#                                                    bore's 0.35 mm play is under partkit's default 0.5 mm contact offset
OBJECTS = [WALL, STAND, PLATE]

# -- scoring --------------------------------------------------------------------------------------------------------------
ON_AXIS = 2.0             # mm: the bore's centre within this of the pipe's axis (the pipe is inside the bore)
SHUT = 0.025              # rad: the split closed to within 1 mm at the bore
FLAT_GAP = 1.5            # mm: the back edge within this of the wall
FLAT_TILT = math.radians(3.0)
RELEASE = 0.03            # m: grippers clear of the plate
MOVE_TOL = 0.001          # m: nothing moves more than this over the last 20 steps (1 s)
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), dropped=False, released=False, finger_check=None, recent=[],
                  levels=(False, False, False), geom={})


_reset_state()


def geometry(objects) -> dict:
    plate = objects["plate"]
    g = {}
    p = (plate.pos - np.array(WALL_ORIGIN)) * 1000          # the plate's origin from the pipe's axis at the wall face
    g["axis_off_mm"] = float(math.hypot(p[1], p[2]))
    g["wall_gap_mm"] = float(-p[0])                          # the back edge's centre in front of the wall
    ax = plate.rotation()[:, 0]
    g["tilt"] = float(math.acos(max(-1.0, min(1.0, float(ax[0])))))
    g["spread"] = float(-plate.hinges.get("half_b", 0.0))    # 0 closed, + open
    g["between"] = bool(-STUB + 1.0 < p[0] < 1.0)
    return g


def _levels(g) -> tuple:
    on = g["axis_off_mm"] < ON_AXIS and g["between"]
    shut = on and g["spread"] < SHUT
    flat = shut and g["wall_gap_mm"] < FLAT_GAP and g["tilt"] < FLAT_TILT
    return on, shut, flat


def _dist_disc(p_local_mm) -> float:
    """Distance (mm) from a point in the plate frame to the plate's solid outline (a disc of PLATE_R, PLATE_H deep)."""
    dr = math.hypot(p_local_mm[1], p_local_mm[2]) - PLATE_R
    dx = max(-PLATE_H - p_local_mm[0], p_local_mm[0])
    if dr <= 0 and dx <= 0:
        return max(dr, dx)
    return math.hypot(max(dr, 0.0), max(dx, 0.0))


@on_step
def _step(environment) -> None:
    """Every control step: count steps and check that both grippers are clear of the plate (score() cannot see the
    robot)."""
    from yam_arena.scene import object_states

    _STATE["step"] += 1
    try:
        plate = object_states(environment, ["plate"])["plate"]
        near = min(_dist_disc(plate.to_local(p) * 1000) for p in finger_points(environment))
        _STATE["near_mm"] = near
        _STATE["released"] = bool(near >= RELEASE * 1000)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"split-escutcheon-fit: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"], _STATE["released"] = False, False


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    plate = objects["plate"]
    _STATE["dropped"] |= bool(float(plate.pos[2]) < TOP - 0.05)          # off the bench
    g = geometry(objects)
    _STATE["geom"] = g
    on, shut, flat = _levels(g)
    _STATE["levels"] = (on, shut, flat)
    recent = _STATE["recent"]
    recent.append(np.concatenate([plate.pos, [g["spread"] * PLATE_R / 1000]]))
    del recent[:-20]
    still = len(recent) == 20 and all(float(np.abs(r - recent[-1]).max()) < MOVE_TOL for r in recent)
    if _STATE["dropped"]:
        result = (0.0, False)
    elif flat and _STATE["released"] and still:
        result = (1.0, True)
    elif flat:
        result = (0.75, False)
    elif shut:
        result = (0.5, False)
    elif on:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """The plate round the pipe, shut, flat against the wall, both grippers clear and still for the last second."""
    return _update(objects)[1]


def distance(objects) -> float:
    g = geometry(objects)
    return float(g["axis_off_mm"] + max(g["wall_gap_mm"], 0.0) + PLATE_R * g["spread"]) / 1000


def score(objects) -> float:
    """0.25 plate round the pipe between the valve and the wall; 0.5 also shut; 0.75 also flat against the wall; 1.0
    with both grippers clear and the plate still for 1 s. 0 for good if the plate left the bench."""
    return _update(objects)[0]


def demo():
    """The scripted demo needs both arms: run ``python demo.py task.py demo_out``."""
    return []
