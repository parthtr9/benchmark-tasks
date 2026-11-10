"""On the station table, in front of the arms: a glued-up pine face frame lying flat on the bench, 160 mm deep (x) by
300 mm wide (y). It has two 30 x 30 mm rails running left to right (the near one and the far one, 130 mm apart centre
to centre) and two 30 mm wide, 10 mm high stiles joining them near the ends, and 30 mm corner blocks at the four rail
ends. The glue has not set, so the frame is racked: the far rail is shifted 8 mm to the left of the near rail, one
diagonal is 14 mm longer than the other. The glued joints are stiff: the far rail moves relative to the near one only
under a firm push (about 2 N), and stays where it is left. A rectangle of tape, 20 mm outside the squared frame's
outline, is stuck to the bench around it.

The robot must squeeze the long diagonal, pushing the two corners at its ends toward each other with one hand each,
until the two diagonals are equal, keep the frame inside its taped outline, then let go of it. One hand alone just
slides the whole frame across the bench, because the frame is lighter than the glue joints are stiff.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Hinge, PartState, Slider, finger_points, on_reset, on_step, part, set_hinge_target, set_slider_target  # noqa: E402,E501
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("The glued-up face frame on the bench is racked out of square. Make it square, so that its two "
               "corner-to-corner diagonals are equal, keep it on its outline, then let go of it.")

PINE = (0.86, 0.70, 0.45)
BLOCK_COL = (0.72, 0.52, 0.28)
TAPE = (0.95, 0.85, 0.15)

# -- the frame (part origin: middle of the near rail at table level; x away from the arms, y left) -----------------
NEAR_X = 0.200                    # world x of the near rail's centre line
SPAN = 0.130                      # rail to rail, centre to centre
RAIL_W, RAIL_H, RAIL_L = 0.030, 0.030, 0.300
STILE_Y = 0.135                   # stile centre lines at y = +-STILE_Y
STILE_W, STILE_H, STILE_L = 0.030, 0.010, 0.098
S0 = 0.008                        # the far rail starts shifted 8 mm to the left (+y)
S_STOP = 0.015                    # the far rail can slide out to 15 mm left of square
PLAY = 0.0005                     # and in to 0.5 mm right of square (the joint's play: the blocks butt there)
THETA0 = math.asin(S0 / SPAN)     # stile angle at the start
FRIC = 2.0                        # N: the far rail's static friction on the near rail (glued joints)
SLIDE_K = 3000.0                  # N/m: stiffness of the drive that holds the far rail at its anchor (stick-slip)
FRAME_MASS = 0.30
M_ROOT, M_FAR, M_STILE = 0.09, 0.09, 0.06
MU = 0.3                          # pine on the bench
CENTRE = (NEAR_X + SPAN / 2, 0.0)  # centre of the squared frame (the taped outline)
HALF_X, HALF_Y = RAIL_W / 2 + SPAN / 2, RAIL_L / 2   # 0.080, 0.150: the squared frame's half extents
TAPE_MARGIN = 0.020               # the tape's inner edge is this far outside the squared frame
TAPE_W = 0.004


def _box(x, y, z0, z1, sx, sy, color, quat=None):
    spec = ((x, y, (z0 + z1) / 2), (sx, sy, z1 - z0), color)
    return spec if quat is None else (*spec, quat)


def _rail(x_c, y_c):
    """A rail with its two corner blocks (centre (x_c, y_c) in the part frame)."""
    mid = RAIL_L - 2 * RAIL_W
    return [_box(x_c, y_c, 0.0, RAIL_H, RAIL_W, mid, PINE),
            _box(x_c, y_c + (RAIL_L - RAIL_W) / 2, 0.0, RAIL_H, RAIL_W, RAIL_W, BLOCK_COL),
            _box(x_c, y_c - (RAIL_L - RAIL_W) / 2, 0.0, RAIL_H, RAIL_W, RAIL_W, BLOCK_COL)]


def _stile(name, side):
    pivot = (RAIL_W / 2, side * STILE_Y, 0.0)
    c, s = math.cos(THETA0), math.sin(THETA0)
    centre = (pivot[0] + c * STILE_L / 2, pivot[1] + s * STILE_L / 2)
    quat = (math.cos(THETA0 / 2), 0.0, 0.0, math.sin(THETA0 / 2))
    return Hinge(name, pivot=pivot, boxes=(_box(centre[0], centre[1], 0.0, STILE_H, STILE_L, STILE_W, PINE, quat),),
                 lower=-0.2, upper=0.2, stiffness=0.5, damping=0.02, target=0.0, mass=M_STILE, axis="Z")


STILE_L_H, STILE_R_H = _stile("stile_l", +1), _stile("stile_r", -1)

FAR = Slider("far", rest=(SPAN, S0, 0.0), boxes=tuple(_rail(SPAN, S0)), axis="Y", lower=-(S0 + PLAY),
             upper=S_STOP - S0, rate=SLIDE_K, damping=40.0, mass=M_FAR)

FRAME = part("frame", pos=(NEAR_X, 0.0, TOP), mass=M_ROOT, boxes=_rail(0.0, 0.0), sliders=(FAR,),
             hinges=(STILE_L_H, STILE_R_H), color=PINE, friction=MU)


def _tape():
    cx, cy = CENTRE
    hx, hy = HALF_X + TAPE_MARGIN + TAPE_W / 2, HALF_Y + TAPE_MARGIN + TAPE_W / 2
    z, t = TOP + 0.00025, 0.0005
    out = []
    for i, sx in enumerate((-1, 1)):
        out.append(Box(f"tape_x{i}", pos=(cx + sx * hx, cy, z), size=(TAPE_W, 2 * hy + TAPE_W, t), color=TAPE))
    for i, sy in enumerate((-1, 1)):
        out.append(Box(f"tape_y{i}", pos=(cx, cy + sy * hy, z), size=(2 * hx - TAPE_W, TAPE_W, t), color=TAPE))
    return out


OBJECTS = [FRAME, *_tape()]

# -- corners, in the part frame -------------------------------------------------------------------------------------
NEAR_R, NEAR_L = (0.0, -STILE_Y), (0.0, STILE_Y)           # near-right, near-left rail centres at the ends
FAR_X = SPAN


def corners(frame: PartState):
    """World positions of the four corner-block centres at table level: near-right, near-left, far-left, far-right."""
    s = S0 + frame.sliders.get("far", 0.0)
    return [frame.to_world(p) for p in ((0.0, -STILE_Y, 0.0), (0.0, STILE_Y, 0.0), (SPAN, s + STILE_Y, 0.0),
                                        (SPAN, s - STILE_Y, 0.0))]


def rack(frame: PartState) -> float:
    """The far rail's shift to the left of the near rail (m): 0 when square."""
    return S0 + frame.sliders.get("far", 0.0)


def diagonals(frame: PartState) -> tuple[float, float]:
    """(long diagonal near-right to far-left, short diagonal near-left to far-right) in metres, on the bench plane."""
    nr, nl, fl, fr = (c[:2] for c in corners(frame))
    return float(np.linalg.norm(fl - nr)), float(np.linalg.norm(fr - nl))


def diag_diff(frame: PartState) -> float:
    a, b = diagonals(frame)
    return abs(a - b)


# -- scoring thresholds ------------------------------------------------------------------------------------------------
GOOD, FAIR, BAD = 0.002, 0.005, 0.008     # diagonal difference (m)
LIFT_LIMIT = 0.010                        # a corner lifted this far off the bench
RELEASE_GAP = 0.030                       # every finger point at least this far from the frame
STILL_CALLS = 20
STILL_TOL = 0.001

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(anchor=0.0, failed=False, why="", released=False, finger_check=None, recent=[], frame=None, fingers=None,
                  worst_diff=0.0, peak_rack=S0, calls=0)


_reset_state()


def _inside_outline(frame: PartState) -> bool:
    """All four outer corners of the frame (at table level) inside the tape's inner edge."""
    lim_x, lim_y = HALF_X + TAPE_MARGIN, HALF_Y + TAPE_MARGIN
    s = rack(frame)
    for lx, ly in ((-RAIL_W / 2, -RAIL_L / 2), (-RAIL_W / 2, RAIL_L / 2), (SPAN + RAIL_W / 2, s - RAIL_L / 2),
                   (SPAN + RAIL_W / 2, s + RAIL_L / 2)):
        w = frame.to_world((lx, ly, 0.0))
        if abs(w[0] - CENTRE[0]) > lim_x or abs(w[1] - CENTRE[1]) > lim_y:
            return False
    return True


def _box_distance(p_local, lo, hi) -> float:
    d = np.maximum(np.maximum(np.asarray(lo) - p_local, 0.0), p_local - np.asarray(hi))
    return float(np.linalg.norm(d))


def frame_distance(frame: PartState, point) -> float:
    """Distance (m) from a world point to the frame's members (axis-aligned boxes in the part frame; the stiles are
    taken as their bounding box, which is generous)."""
    p = frame.to_local(point)
    s = rack(frame)
    t = abs(s) * 0.5 + 0.002
    boxes = [((-RAIL_W / 2, -RAIL_L / 2, 0.0), (RAIL_W / 2, RAIL_L / 2, RAIL_H)),
             ((SPAN - RAIL_W / 2, s - RAIL_L / 2, 0.0), (SPAN + RAIL_W / 2, s + RAIL_L / 2, RAIL_H))]
    for side in (1, -1):
        y0 = side * STILE_Y
        boxes.append(((RAIL_W / 2, y0 - STILE_W / 2 - t, 0.0), (SPAN - RAIL_W / 2, y0 + STILE_W / 2 + t, STILE_H)))
    return min(_box_distance(p, lo, hi) for lo, hi in boxes)


def observe(frame: PartState, fingers=None) -> None:
    """One control step of bookkeeping (called from the on_step hook): flags, released, stillness window."""
    s = _STATE
    s["frame"] = frame
    cs = corners(frame)
    if any(c[2] - TOP > LIFT_LIMIT for c in cs):
        s["failed"], s["why"] = True, s["why"] or "lifted"
    if not _inside_outline(frame):
        s["failed"], s["why"] = True, s["why"] or "outline"
    if rack(frame) >= S_STOP - 0.0005:
        s["failed"], s["why"] = True, s["why"] or "stop"
    s["worst_diff"] = max(s["worst_diff"], diag_diff(frame))
    s["peak_rack"] = max(s["peak_rack"], abs(rack(frame)))
    if fingers is not None:
        s["released"] = bool(min(frame_distance(frame, p) for p in fingers) >= RELEASE_GAP)
    recent = s["recent"]
    recent.append(np.concatenate([np.ravel(cs), [rack(frame)]]))
    del recent[:-STILL_CALLS]
    s["calls"] += 1


@on_step
def _mechanism(environment) -> None:
    """The stiles follow the far rail's slide (the four-bar linkage), then this step's bookkeeping."""
    from partkit import _part_state

    frame = _part_state(environment, FRAME)
    # Coulomb friction on the far rail: a stiff drive holds it at an anchor that is dragged along once the pull
    # exceeds FRIC (the spring force never exceeds FRIC at the moment of the update).
    p, band = frame.sliders.get("far", 0.0), FRIC / SLIDE_K
    a = _STATE["anchor"]
    a = p - band if p - a > band else (p + band if a - p > band else a)
    _STATE["anchor"] = a
    set_slider_target(environment, "frame", "far", a)
    theta = math.asin(max(-0.9, min(0.9, rack(frame) / SPAN))) - THETA0
    for name in ("stile_l", "stile_r"):
        set_hinge_target(environment, "frame", name, theta)
    try:
        fingers = [np.asarray(p, float) for p in finger_points(environment)]
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - report once; an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"frame-square-rack: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"], fingers = False, None
        _STATE["released"] = False
    observe(frame, fingers)
    _STATE["fingers"] = fingers


def _still() -> bool:
    recent = _STATE["recent"]
    return len(recent) == STILL_CALLS and float(np.ptp(np.array(recent), axis=0).max()) < STILL_TOL


def _level(frame: PartState) -> tuple[float, bool]:
    s = _STATE
    if s["failed"]:
        return 0.0, False
    d = diag_diff(frame)
    if d > BAD:
        return 0.0, False
    if d > FAIR:
        return 0.25, False
    if d <= GOOD and s["released"] and _still():
        return 1.0, True
    return 0.5, False


def success(objects) -> bool:
    """Diagonals within 2 mm of each other, the frame never lifted a corner 1 cm or left its taped outline or pushed
    the far rail to its stop, both grippers 3 cm or more from the frame, and the frame still (under 1 mm) for 20
    scorer steps (1 s)."""
    return _level(objects["frame"])[1]


def score(objects) -> float:
    """By the diagonal difference at the end: over 8 mm 0, 5 to 8 mm 0.25, 2 to 5 mm 0.5, 2 mm or less 0.5 until the
    grippers are 3 cm clear and the frame has been still for 1 s, then 1.0. 0 at once and for good if a corner is
    ever lifted more than 1 cm, the frame leaves its taped outline (20 mm margin), or the far rail is driven to its
    15 mm stop (joint broken open)."""
    return _level(objects["frame"])[0]


def distance(objects) -> float:
    """The diagonal difference (m), 0 when square."""
    return float(diag_diff(objects["frame"]))
