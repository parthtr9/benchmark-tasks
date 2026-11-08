"""On the station table, in front of the arms: a plywood cutting board (200 x 320 x 12 mm) with a steel rod (8 mm
diameter, along x, 140 mm long) lying on it, and a glazed ceramic wall tile (100 x 240 x 7 mm, 0.38 kg) scored across
its middle, laid score-down over the rod. The tile is modelled as two 0.19 kg halves joined by a hinge along the score at
the bottom face. The hinge only bends ends-down (it has a hard stop at 0) and springs back flat (6 N m/rad). At 3 degrees
of ends-down bend (0.31 N m) the tile snaps: for good, the hinge then turns freely from 0 to 15 degrees. The tile starts
tipped toward +y: its +y end rests on the board, its -y end is raised.

Pressing one end of the tile only rotates the whole rigid tile on the rod (the other end rises); the tile bends only if
one end is held down while the other is pressed down too, about 3 N at each end 0.10 m from the rod.

The robot must snap the tile along the score without lifting it off the rod, then let go.

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
from partkit import Hinge, PartState, finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("The tile is scored across its middle. Snap it in two along the score over the steel rod, without "
               "lifting it off the rod, then let go.")

CREAM = (0.93, 0.90, 0.82)
CREAM2 = (0.90, 0.86, 0.76)
WOOD = (0.76, 0.58, 0.34)
STEEL = (0.62, 0.64, 0.68)

# -- layout (world frame: x forward from the arms, y left, metres) --------------------------------------------------
BOARD_X = 0.29
BOARD_SIZE = (0.20, 0.32, 0.012)
BOARD_TOP = BOARD_SIZE[2]
ROD_R = 0.004
ROD_LEN = 0.14
ROD_TOP = BOARD_TOP + 2 * ROD_R            # 0.020 above the table: the rod's top line
TILE_X_HALF, TILE_HALF, TILE_T = 0.05, 0.12, 0.007   # tile 0.10 (x) x 0.24 (y) x 7 mm; each half is 0.12 long
TILE_Y0 = 0.001                            # the score starts 1 mm toward +y of the rod's top line, so it tips to +y
PRESS_Y = 0.10                             # the demo presses 0.10 m either side of the score

# Hinge: ends-down bend. Stiffness 6 N m/rad; snaps at 3 degrees (0.31 N m, about 3.1 N at 0.10 m).
K_HINGE = 6.0
SNAP_DEG = 3.0
HINGE_MAX = math.radians(15.0)
FREE_TARGET = math.radians(9.7)            # after the snap the drive leans the halves onto the board (about 1.5 N each)
HALF_MASS = 0.19

BENCH = part("bench", pos=(BOARD_X, 0.0, TOP), mass=30.0, color=WOOD, friction=1.0,
             boxes=[((0.0, 0.0, BOARD_SIZE[2] / 2), BOARD_SIZE, WOOD)],
             cylinders=(((0.0, 0.0, BOARD_TOP + ROD_R), ROD_R, ROD_LEN, STEEL, "X"),))

FAR = Hinge("far", pivot=(0.0, 0.0, 0.0), axis="X", lower=0.0, upper=HINGE_MAX, stiffness=K_HINGE, damping=0.15,
            target=0.0, mass=HALF_MASS,
            boxes=(((0.0, -TILE_HALF / 2, TILE_T / 2), (2 * TILE_X_HALF, TILE_HALF, TILE_T), CREAM2),))
TILE = part("tile", pos=(BOARD_X, TILE_Y0, TOP + ROD_TOP + 0.0002), mass=HALF_MASS, color=CREAM, friction=0.5,
            boxes=[((0.0, TILE_HALF / 2, TILE_T / 2), (2 * TILE_X_HALF, TILE_HALF, TILE_T), CREAM)], hinges=(FAR,))
OBJECTS = [BENCH, TILE]

# -- scoring thresholds ---------------------------------------------------------------------------------------------
CHIP_RADIUS = 0.06       # a finger point this close to the score line when it snaps: chipped (a gripper opened to
                         # straddle the score; the open pads sit 47 mm either side, the points 53 mm)
OFF_ROD = 0.003          # the score more than 3 mm from the rod's top line before the snap: ragged
RELEASE_GAP = 0.03       # grippers at least 3 cm from both pieces
END_ON_BOARD = 0.004     # a piece's outer bottom edge within 4 mm of the board's top: it lies on the board
STILL_CALLS = 20
STILL_TOL = 0.001
BOARD_HALF = (BOARD_SIZE[0] / 2, BOARD_SIZE[1] / 2)

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(snapped=False, chip=False, off_rod=False, off_board=False, released=False, finger_check=None,
                  recent=[], step=0, snap_step=None, peak_off=0.0, max_hinge=0.0, fingers_at_snap=None,
                  min_snap_dist=None)


_reset_state()


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def hinge_angle(tile: PartState) -> float:
    """Ends-down bend of the tile (rad), 0 flat."""
    return float(tile.hinges.get("far", 0.0))


def tilt_degrees(tile: PartState) -> float:
    """How far the +y half is tipped up from level (deg): negative while its end is down on the board."""
    return math.degrees(math.asin(max(-1.0, min(1.0, tile.rotation()[2, 1]))))


def near_end(tile: PartState) -> np.ndarray:
    """Middle of the +y half's outer bottom edge."""
    return tile.to_world((0.0, TILE_HALF, 0.0))


def far_end(tile: PartState) -> np.ndarray:
    """Middle of the -y half's outer bottom edge."""
    return tile.to_world(_rx(hinge_angle(tile)) @ np.array([0.0, -TILE_HALF, 0.0]))


def piece_distance(tile: PartState, point, which: str) -> float:
    """Distance (m) from a world point to one half ('near' = +y half, 'far' = -y half), as an oriented box."""
    p = tile.to_local(point)
    if which == "far":
        p = _rx(-hinge_angle(tile)) @ p
        centre = np.array([0.0, -TILE_HALF / 2, TILE_T / 2])
    else:
        centre = np.array([0.0, TILE_HALF / 2, TILE_T / 2])
    half = np.array([TILE_X_HALF, TILE_HALF / 2, TILE_T / 2])
    out = np.maximum(np.abs(p - centre) - half, 0.0)
    return float(np.linalg.norm(out))


def score_line_distance(tile: PartState, point) -> float:
    """Distance (m) from a world point to the score line (the segment along the tile's x axis at its pivot)."""
    p = tile.to_local(point)
    d = np.array([max(abs(p[0]) - TILE_X_HALF, 0.0), p[1], p[2]])
    return float(np.linalg.norm(d))


def rod_offset(tile: PartState) -> float:
    """Distance (m) of the score line from the rod's top line (y 0, z at the rod's top), in the y-z plane."""
    return float(math.hypot(tile.pos[1] - 0.0, tile.pos[2] - (TOP + ROD_TOP)))


def pieces_on_board(tile: PartState) -> bool:
    for end in (near_end(tile), far_end(tile)):
        if abs(end[0] - BOARD_X) > BOARD_HALF[0] or abs(end[1]) > BOARD_HALF[1] or end[2] > TOP + BOARD_TOP + END_ON_BOARD:
            return False
    return True


def left_board(tile: PartState) -> bool:
    for end in (near_end(tile), far_end(tile), tile.pos):
        if abs(end[0] - BOARD_X) > BOARD_HALF[0] + 0.01 or abs(end[1]) > BOARD_HALF[1] + 0.01 or end[2] < TOP + 0.005:
            return True
    return False


def _fingers(environment, tile: PartState):
    pts = [np.asarray(p, float) for p in finger_points(environment)]
    return pts


def trace_line(environment, tile: PartState) -> str:
    """One line of debug: hinge, tilt, pivot offset, and the nearest finger point of each arm to each end."""
    try:
        pts = _fingers(environment, tile)
        parts = []
        for name, chunk in (("L", pts[:4]), ("R", pts[4:])):
            dn = min(piece_distance(tile, p, "near") for p in chunk)
            df = min(piece_distance(tile, p, "far") for p in chunk)
            ds = min(score_line_distance(tile, p) for p in chunk)
            lowest = min(chunk, key=lambda p: p[2])
            parts.append(f"{name}: to_near {1000 * dn:.0f} to_far {1000 * df:.0f} to_score {1000 * ds:.0f} "
                         f"low_xyz_mm {np.round(1000 * (lowest - [BOARD_X, 0, TOP]), 0).tolist()}")
        finger = " | ".join(parts)
    except Exception as exc:  # noqa: BLE001
        finger = f"fingers unavailable: {exc!r}"
    return (f"tile step {_STATE['step']} hinge_deg {math.degrees(hinge_angle(tile)):.2f} tilt_deg {tilt_degrees(tile):.2f} "
            f"rod_off_mm {1000 * rod_offset(tile):.1f} snapped {_STATE['snapped']} near_z_mm "
            f"{1000 * (near_end(tile)[2] - TOP):.1f} far_z_mm {1000 * (far_end(tile)[2] - TOP):.1f} released "
            f"{_STATE['released']} {finger}")


@on_step
def _mechanism(environment) -> None:
    """The hinge's snap (permanent), the chip / off-rod / off-board flags, the release check, and the debug trace."""
    from partkit import _part_state, set_hinge_target

    s = _STATE
    s["step"] += 1
    tile = _part_state(environment, TILE)
    theta = hinge_angle(tile)
    s["max_hinge"] = max(s["max_hinge"], theta)
    if not s["snapped"]:
        off = rod_offset(tile)
        s["peak_off"] = max(s["peak_off"], off)
        if off > OFF_ROD:
            s["off_rod"] = True
    if left_board(tile):
        s["off_board"] = True
    try:
        pts = _fingers(environment, tile)
        s["released"] = bool(min(min(piece_distance(tile, p, w) for w in ("near", "far")) for p in pts) >= RELEASE_GAP)
        s["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - an unknown grip never counts as released
        if s["finger_check"] is None:
            print(f"tile-snap-rod: finger check unavailable: {exc!r}", flush=True)
        pts, s["finger_check"], s["released"] = None, False, False
    if not s["snapped"] and theta >= math.radians(SNAP_DEG):
        s["snapped"], s["snap_step"] = True, s["step"]
        if pts is not None:
            s["min_snap_dist"] = min(score_line_distance(tile, p) for p in pts)
            if s["min_snap_dist"] < CHIP_RADIUS:
                s["chip"] = True
        set_hinge_target(environment, "tile", "far", FREE_TARGET)
    if os.environ.get("DEMO_DEBUG") and (s["step"] % 10 == 0 or s["snap_step"] == s["step"]):
        print(trace_line(environment, tile), flush=True)


def _update(objects) -> tuple[float, bool]:
    tile = objects["tile"]
    s = _STATE
    if left_board(tile):
        s["off_board"] = True
    if s["off_board"] or not s["snapped"]:
        return 0.0, False
    if s["chip"] or s["off_rod"]:
        return 0.25, False
    recent = s["recent"]
    recent.append(np.r_[tile.pos, near_end(tile), far_end(tile)])
    del recent[:-STILL_CALLS]
    still = len(recent) == STILL_CALLS and float(np.ptp(np.array(recent), axis=0).max()) < STILL_TOL
    done = bool(s["released"] and still and pieces_on_board(tile))
    return (1.0 if done else 0.5), done


def success(objects) -> bool:
    """The tile snapped (the hinge bent 3 degrees), cleanly (no finger within 6 cm of the score at the snap, the score
    never more than 3 mm from the rod's top line before it), both halves on the board, both grippers at least 3 cm
    from them, and the tile still within 1 mm for 20 scorer calls (1 s)."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 never snapped, or a piece left the board (for good); 0.25 snapped with a finger within 6 cm of the score or
    with the score ever more than 3 mm off the rod's top line (a ragged edge; kept for good); 0.5 clean snap, but a
    gripper is within 3 cm of a piece, a piece is off the board, or something still moves; 1.0 clean snap, both halves
    on the board, grippers 3 cm away, still for 1 s."""
    return _update(objects)[0]


def distance(objects) -> float:
    """Metres of press travel to go to the snap at 0.10 m from the rod (0 once snapped)."""
    if _STATE["snapped"]:
        return 0.0
    return float(max(0.0, math.radians(SNAP_DEG) - hinge_angle(objects["tile"])) * PRESS_Y)
