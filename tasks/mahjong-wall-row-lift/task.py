"""On the station table, on a mahjong mat: one player's 34 tiles for a riichi mahjong wall, face down in two loose rows
of 17, side by side. The wall section is finished when the near row lies on top of the far row, tile over tile, two
high and straight, every tile still face down. The near row has to go up as a row, in one lift: seventeen loose tiles
make a beam only while they are pressed together end to end.

Sizes and sources: Marushin Sangyo's hand-play riichi tiles ("Kokyu mahjong hai Take", urea resin), listed in a
product review at 26 x 19 x 16 mm and about 13.5 to 14 g a tile (13.7 g here). Estimates (measure the bought set):
tile-on-tile friction (0.3) and tile-on-mat friction (0.4; tilt a tile on a tile, and on the mat, until it slides).
Drawn only, from photographs: the coloured back layer (6 mm) and its rounded top edge (1 mm). The mat is an ordinary
roll-up mahjong mat, modelled as a fixed 2 mm sheet.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = ("Finish this section of the mahjong wall: put the near row of tiles on top of the far row in a single "
               "lift, leaving a straight wall two tiles high with every tile face down.")

BACK = (0.93, 0.72, 0.16)        # the tiles' backs (yellow), all that shows of a face-down tile from above
FACE = (0.96, 0.94, 0.86)        # the face layer (ivory white), seen from the side and on any tile that is turned over
MAT = (0.10, 0.36, 0.22)

# -- a tile (frame: origin at its centre; its back at +z, so +z points up while it lies face down) -------------------------
TILE_D, TILE_W, TILE_T = 0.026, 0.019, 0.016      # along x (the tile's height in play), along the row (y), thickness
TILE_MASS = 0.0137
BACK_T = 0.006                   # the coloured back layer's thickness (drawn only; estimate from photographs)
EDGE_R = 0.001                   # the back's rounded top edge, drawn as a 1 mm step (drawn only; estimate)
TILE_MU = 0.3                    # urea resin on urea resin (estimate)
N = 17
GAP = 0.0003                     # a loose row: the tiles lie this far apart
PITCH = TILE_W + GAP
ROW_L = N * TILE_W + (N - 1) * GAP   # 327.8 mm

# -- layout ----------------------------------------------------------------------------------------------------------------
MAT_T = 0.002
MAT_SIZE = (0.60, 0.69)
MAT_X = 0.36
FLOOR = TOP + MAT_T              # the mat's top
NEAR_X, FAR_X = 0.25, 0.31       # the two rows' lines: 60 mm apart, 34 mm of mat between them
ROW_Y = [float((k - (N - 1) / 2) * PITCH) for k in range(N)]
Z0 = FLOOR + TILE_T / 2 + 0.0005

MAT_PART = part("mat", pos=(MAT_X, 0.0, TOP), mass=1.0, fixed=True, color=MAT, friction=0.5,
                boxes=[((0.0, 0.0, MAT_T / 2), (MAT_SIZE[0], MAT_SIZE[1], MAT_T), MAT)])


def tile_boxes():
    """How a tile is drawn, in its own frame: the ivory face layer below, the yellow back layer on it, and the back's top
    millimetre stepped in by EDGE_R all round for its rounded edge, so the seams between neighbouring tiles show as they
    do on real tiles. These boxes are drawn only; the tile collides as one full-size block (TILE_COLLIDER)."""
    face_t = TILE_T - BACK_T
    return [((0.0, 0.0, -TILE_T / 2 + face_t / 2), (TILE_D, TILE_W, face_t), FACE),
            ((0.0, 0.0, TILE_T / 2 - EDGE_R - (BACK_T - EDGE_R) / 2), (TILE_D, TILE_W, BACK_T - EDGE_R), BACK),
            ((0.0, 0.0, TILE_T / 2 - EDGE_R / 2), (TILE_D - 2 * EDGE_R, TILE_W - 2 * EDGE_R, EDGE_R), BACK)]


# One box per tile for contact: a tile split into layered colliders gave edge contacts between neighbours' layers, and
# the pressed row buckled sideways on the mat (Isaac batch 110ea54f). A colour of None makes the box a collider only.
TILE_COLLIDER = [((0.0, 0.0, 0.0), (TILE_D, TILE_W, TILE_T), None)]


def _tile(name, x, y):
    return part(name, pos=(float(x), float(y), float(Z0)), mass=TILE_MASS, color=BACK, friction=TILE_MU,
                boxes=TILE_COLLIDER, meta={"visual_boxes": tile_boxes()})


NEAR = [_tile(f"near_{k:02d}", NEAR_X, ROW_Y[k]) for k in range(N)]     # the row to lift
FAR = [_tile(f"far_{k:02d}", FAR_X, ROW_Y[k]) for k in range(N)]        # the row it goes on
TILES = NEAR + FAR
OBJECTS = [MAT_PART, *TILES]

# -- scoring ---------------------------------------------------------------------------------------------------------------
RELEASE = 0.03            # every gripper finger point this far from every tile: let go
UP_MM = 20.0              # "in the air": a tile's underside this far above the mat (a tile lying on the far row is at 16)
ON_TOP_MM = 4.0           # "on the far row": a tile's underside within this of the far row's top (16 mm)
ALIGN_MM = 5.0            # an upper tile's centre within this of the tile beneath it, along and across the row
FAR_MOVE_MM = 5.0         # the far row's tiles stay within this of where they started
ROW_MM = 5.0              # an intact row: every tile within this of the line through its end tiles, in plan and height
GAP_MM = 3.0              # ... and no joint between neighbours open more than this
STILL_STEPS = 20
STILL_MM = 1.0
EDGE = 0.01               # a tile's centre this far past the table's edge, or below its top, is off the table
TABLE_X, TABLE_Y = (-0.05, 0.75), (-0.60, 0.60)
_CORNERS = np.array([[TILE_D / 2, TILE_W / 2, TILE_T / 2], [-TILE_D / 2, -TILE_W / 2, -TILE_T / 2]])
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(step=0, released=False, as_one=False, violation="", hist=[], free=[])


_reset_state()


def face_down(st) -> bool:
    """The tile lies flat with its back up."""
    return float((st.rotation() @ np.array([0.0, 0.0, 1.0]))[2]) > 0.9


def under(st) -> float:
    """Height of the tile's centre above the mat, less half its thickness (mm): 0 lying on the mat, 16 on a tile."""
    return 1000 * (float(st.pos[2]) - FLOOR - TILE_T / 2)


def off_table(st) -> bool:
    x, y, z = (float(v) for v in st.pos)
    return (z < TOP - EDGE or not TABLE_X[0] - EDGE < x < TABLE_X[1] + EDGE
            or not TABLE_Y[0] - EDGE < y < TABLE_Y[1] + EDGE)


def row_intact(near) -> bool:
    """The 17 tiles still form one row in order: each within ROW_MM of the straight line through the two end tiles (in
    plan and in height), neighbours no more than GAP_MM apart, all face down."""
    p = np.array([np.asarray(s.pos, float) for s in near])
    a, b = p[0], p[-1]
    length = float(np.linalg.norm(b - a))
    if length < 1e-6 or not all(face_down(s) for s in near):
        return False
    u = (b - a) / length
    off = (p - a) - np.outer((p - a) @ u, u)
    pitch = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return bool(np.linalg.norm(off, axis=1).max() <= ROW_MM / 1000
                and (pitch.max() - TILE_W) <= GAP_MM / 1000 and (pitch.min() - TILE_W) >= -GAP_MM / 1000)


def levels(objects) -> tuple:
    """(intact, on top, aligned): the lifted row is one row again, lying face down; and every one of its tiles lies on
    the far row; and every one is within ALIGN_MM of the far tile beneath it, the far row within FAR_MOVE_MM of where
    it started, all 34 face down."""
    near = [objects[s.name] for s in NEAR]
    far = [objects[s.name] for s in FAR]
    lying = all(abs(under(s)) <= ON_TOP_MM or abs(under(s) - 1000 * TILE_T) <= ON_TOP_MM for s in near)
    intact = row_intact(near) and lying
    on_top = intact and all(abs(under(s) - 1000 * TILE_T) <= ON_TOP_MM for s in near)
    far_ok = all(face_down(f) and abs(under(f)) <= ON_TOP_MM
                 and math.hypot(float(f.pos[0]) - FAR_X, float(f.pos[1]) - ROW_Y[k]) <= FAR_MOVE_MM / 1000
                 for k, f in enumerate(far))
    aligned = on_top and far_ok and stacks(near, far)
    return intact, on_top, aligned


def stacks(near, far) -> bool:
    """17 stacks of two: every far tile has its own upper tile within ALIGN_MM of it, along and across the row (which
    upper tile lies on which far tile does not matter)."""
    taken = set()
    for n in near:
        d = [max(abs(float(n.pos[0] - f.pos[0])), abs(float(n.pos[1] - f.pos[1]))) for f in far]
        k = int(np.argmin(d))
        if d[k] > ALIGN_MM / 1000 or k in taken:
            return False
        taken.add(k)
    return len(taken) == len(far)


def tile_gap(p, st) -> float:
    """A point's distance from a tile's surface."""
    local = st.to_local(np.asarray(p, float))
    d = np.maximum(np.abs(local) - np.array([TILE_D, TILE_W, TILE_T]) / 2, 0.0)
    return float(np.linalg.norm(d))


@on_step
def _step(environment) -> None:
    from partkit import _part_state   # noqa: PLC0415

    _STATE["step"] += 1
    pts = finger_points(environment)
    tiles = [_part_state(environment, s) for s in TILES]
    _STATE["released"] = all(tile_gap(p, t) > RELEASE for p in pts for t in tiles)


def _update(objects) -> tuple[float, bool]:
    near = [objects[s.name] for s in NEAR]
    tiles = [objects[s.name] for s in TILES]
    if not _STATE["violation"]:
        for spec, s in zip(TILES, tiles):
            if off_table(s):
                _STATE["violation"] = f"{spec.name} off the table"
                break
    # the one lift: at one moment all 17 are in the air together, still one row (kept once reached)
    if not _STATE["as_one"] and all(under(s) >= UP_MM for s in near) and row_intact(near):
        _STATE["as_one"] = True
    hist = _STATE["hist"]
    hist.append(np.concatenate([[s.to_world(c) for c in _CORNERS] for s in tiles]))
    del hist[:-STILL_STEPS]
    free = _STATE["free"]                       # whether the grippers were clear, step by step over the same window
    free.append(bool(_STATE["released"]))
    del free[:-STILL_STEPS]
    if _STATE["violation"]:
        return 0.0, False
    intact, on_top, aligned = levels(objects)
    ok = _STATE["as_one"] and _STATE["released"]
    s = 0.25 * (ok and intact) + 0.25 * (ok and on_top) + 0.25 * (ok and aligned)
    # still for 1 s with the grippers clear for that whole second: a wall held still and only then let go has to
    # stand a full second by itself
    still = (len(hist) == STILL_STEPS and all(free)
             and max(float(np.linalg.norm(h - hist[-1], axis=1).max()) for h in hist) <= STILL_MM / 1000)
    done = bool(ok and aligned and still)
    return (1.0 if done else float(s)), done


def _once(objects) -> tuple[float, bool]:
    """``_update`` once per control step: success() and score() are both called each step, and the stillness window must
    hold one pose per step (20 steps = 1 s), not one per call."""
    if _STATE.get("updated_at") != _STATE["step"]:
        _STATE["updated_at"], _STATE["last"] = _STATE["step"], _update(objects)
    return _STATE["last"]


def success(objects) -> bool:
    return _once(objects)[1]


def distance(objects) -> float:
    return float(np.mean([np.linalg.norm(np.asarray(objects[n.name].pos, float)
                                         - (np.asarray(objects[f.name].pos, float) + [0.0, 0.0, TILE_T]))
                          for n, f in zip(NEAR, FAR)]))


def score(objects) -> float:
    return _once(objects)[0]


def demo():
    """The scripted demonstration is in demo.py."""
    return []


if os.environ.get("MAHJONG_PRINT_LAYOUT"):
    print({"near_x": NEAR_X, "far_x": FAR_X, "row_y": ROW_Y, "z0": Z0 - TOP})
