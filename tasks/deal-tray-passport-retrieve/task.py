"""At a ticket window: a section of counter with a recessed stainless deal tray set flush in it and the window's glass
standing across the tray's middle. A passport has been passed back and lies in the dish, on its lowest line under the
glass. It must end lying on the green mat on the customer's side of the counter.

Sizes and sources: the tray is Total Security Solutions' Recessed Currency Tray in its stock size, 16 x 10 x 1-1/2 in
(the maker's drawing and data sheet: 18 gauge stainless steel, #4 brushed finish, a 1/2 in flange all round, a body 15 x
9 in under the flange in a 15-1/4 x 9-1/4 in cut-out, and a dish that is one circular arc across the 10 in direction,
closed by flat end walls). The drawing marks the arc "R" without a value: the 191.9 mm used here follows from the 9 in
body and the 1-1/2 in height less the sheet (estimate, measure the bought tray). The passport is a 125 x 88 mm booklet
(ICAO Doc 9303 part 4, the ID-3 size). No maker or state publishes a booklet's thickness or mass: the 4.5 mm and 40 g
used here are estimates (a published booklet build, 32 pages of 90 g/m2 paper, 125 g/m2 endpapers, a polycarbonate data
page of up to 0.76 mm and a board cover, works out at 3.9 mm and 36 g before thread, glue, laminate and the chip inlay;
measure and weigh the booklet used). The counter is a 1-1/2 in laminate worktop section and the glass a section of 1/4 in
tempered glass, both cut down in size only.

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
import meshkit as mk  # noqa: E402
from partkit import finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = "Take the passport out of the tray under the window and leave it lying on the green mat."

IN = 0.0254
STEEL = (0.74, 0.75, 0.77)
LAMINATE = (0.86, 0.85, 0.82)
GREEN = (0.16, 0.55, 0.27)
NAVY = (0.07, 0.11, 0.27)
GOLD = (0.80, 0.66, 0.30)
PAPER = (0.93, 0.91, 0.84)
ALUMINIUM = (0.66, 0.67, 0.69)
GLASS = (0.80, 0.90, 0.95, 0.12)      # clear glass: colour and opacity

# -- the tray (frame: origin at its centre on the flange's top face; x toward the clerk, y along the window) ---------------
TRAY_W, TRAY_D, TRAY_H = 16 * IN, 10 * IN, 1.5 * IN     # along the window, across it, overall height (maker's drawing)
FLANGE = 0.5 * IN
SHEET = 0.00127                                         # 18 gauge stainless
BODY_HALF = TRAY_D / 2 - FLANGE                         # the body under the flange is 9 in across
DISH_HALF = BODY_HALF - SHEET                           # the dish's rim, half its chord (113.0 mm)
DISH_SAG = TRAY_H - SHEET                               # its depth below the flange's top (36.8 mm)
DISH_R = (DISH_HALF ** 2 + DISH_SAG ** 2) / (2 * DISH_SAG)   # the arc's radius (191.9 mm; derived, see the docstring)
DISH_LEN = TRAY_W - 2 * FLANGE - 2 * SHEET              # the dish's length between its end walls
CUTOUT_W, CUTOUT_D = 15.25 * IN, 9.25 * IN              # the maker's cut-out
ARC_SEGMENTS = 48


def dish_z(dx: float) -> float:
    """The dish's surface below the flange's top face (tray frame z, negative), ``dx`` across the window from its lowest line."""
    dx = min(abs(float(dx)), DISH_HALF)
    return -DISH_SAG + DISH_R - math.sqrt(DISH_R ** 2 - dx ** 2)


def dish_profile(n: int = ARC_SEGMENTS):
    """The arc as (x, z) points from the customer's rim to the clerk's (tray frame), corners on the real radius."""
    th = math.asin(DISH_HALF / DISH_R)
    return [(DISH_R * math.sin(a), -DISH_SAG + DISH_R * (1 - math.cos(a))) for a in np.linspace(-th, th, n + 1)]


def tray_meshes():
    """The tray as one solid under its real top surface: the flange, the arc dish and the body's outside, then the two
    end walls with their flanges."""
    arc = dish_profile()[::-1]                         # from the clerk's rim back to the customer's
    outline = [(-TRAY_D / 2, 0.0), (-TRAY_D / 2, -SHEET), (-BODY_HALF, -SHEET), (-BODY_HALF, -TRAY_H),
               (BODY_HALF, -TRAY_H), (BODY_HALF, -SHEET), (TRAY_D / 2, -SHEET), (TRAY_D / 2, 0.0), *arc]
    # extrude runs along z: stand the (x, z) outline up and run it along the window (y)
    body = mk.extrude(outline, DISH_LEN).rotated("X", 90).moved((0.0, DISH_LEN / 2, 0.0))
    look = dict(color=STEEL, metallic=1.0, roughness=0.35)
    shapes = [mk.Shape(body, name="dish", **look)]
    for s in (1.0, -1.0):
        wall = mk.box((2 * BODY_HALF, SHEET, TRAY_H), (0.0, s * (DISH_LEN / 2 + SHEET / 2), -TRAY_H / 2))
        lip = mk.box((TRAY_D, FLANGE + SHEET, SHEET), (0.0, s * (DISH_LEN / 2 + (FLANGE + SHEET) / 2), -SHEET / 2))
        shapes += [mk.Shape(wall, name=f"end_wall_{'l' if s > 0 else 'r'}", **look),
                   mk.Shape(lip, name=f"end_flange_{'l' if s > 0 else 'r'}", **look)]
    return shapes


# -- the passport (frame: origin at its centre; 88 mm along x, 125 mm along y, spine on the clerk's side) -------------------
BOOK_D, BOOK_W, BOOK_T = 0.088, 0.125, 0.0045
BOOK_MASS = 0.040
# Rehearsal only (not the task): the two ends of what a real booklet may measure, to check that the draw does not hinge
# on the estimate. DEAL_TRAY_BOOKLET=light is 4 mm and 36 g, DEAL_TRAY_BOOKLET=heavy 5 mm and 45 g.
_BOOKLET = os.environ.get("DEAL_TRAY_BOOKLET", "")
if _BOOKLET in ("light", "heavy"):
    BOOK_T, BOOK_MASS = (0.004, 0.036) if _BOOKLET == "light" else (0.005, 0.045)
BOOK_FRICTION = 0.5       # with the tray's 0.05 PhysX averages to 0.275 (cover on brushed steel); with the worktop 0.3
BOOK_TOP_FRICTION = 1.5   # its upper face: with the fingertips' 0.5 PhysX averages to 1.0 (rubber pad on the cover)
BOOK_TOP_LAYER = 0.001    # that face is the top millimetre of the slab (PhysX sets friction per body pair, so the two
                          # real pairs, pad on cover and cover on steel, need two materials on the booklet)
TRAY_FRICTION = 0.05
COUNTER_FRICTION = 0.1


def passport_boxes():
    lower = BOOK_T - BOOK_TOP_LAYER
    return [((0.0, 0.0, -BOOK_TOP_LAYER / 2), (BOOK_D, BOOK_W, lower), NAVY),
            ((0.0, 0.0, BOOK_T / 2 - BOOK_TOP_LAYER / 2), (BOOK_D, BOOK_W, BOOK_TOP_LAYER), NAVY)]


def passport_trim():
    """Drawn only: the page block showing at the fore-edge and the two ends, and the cover's gold crest and title."""
    return [((-0.0011, 0.0, 0.0), (BOOK_D - 0.0018, BOOK_W + 0.0004, BOOK_T - 0.0012), PAPER),
            ((0.004, 0.0, BOOK_T / 2), (0.026, 0.026, 0.0002), GOLD),
            ((-0.022, 0.0, BOOK_T / 2), (0.006, 0.060, 0.0002), GOLD),
            ((0.030, 0.0, BOOK_T / 2), (0.005, 0.050, 0.0002), GOLD)]


# -- layout (world: x forward from the arm bases, y to the robot's left, z up; heights from the table top) ------------------
GLASS_X = 0.42            # the window's glass, and the dish's lowest line under it
TRAY_Y = 0.06
COUNTER_T = 1.5 * IN      # laminate worktop
COUNTER_X = (0.10, 0.62)
COUNTER_Y = (-0.30, 0.30)
COUNTER_TOP = TOP + COUNTER_T
FLANGE_TOP = COUNTER_TOP + SHEET                 # the drop-in flange lies on the worktop
GLASS_T, GLASS_H = 0.25 * IN, 0.30               # 1/4 in tempered glass; the lower 300 mm of the window
GLASS_GAP = 0.002                                # its bottom edge above the worktop (it clears the tray's flange)
SILL = 1.0 * IN                                  # aluminium sill channel on the worktop beside the tray
MAT_X = (0.125, 0.275)
MAT_Y = (TRAY_Y - 0.10, TRAY_Y + 0.10)
X_RIM = GLASS_X - DISH_HALF                      # the dish's rim on the customer's side
X_FLANGE = GLASS_X - TRAY_D / 2                  # the flange's outer edge on the customer's side

BOOK_HALF_ANGLE = math.asin(BOOK_D / 2 / DISH_R)                     # the booklet is a chord of the arc
BOOK_REST_GAP = DISH_R - math.sqrt(DISH_R ** 2 - (BOOK_D / 2) ** 2)  # air under its middle (5.1 mm)
BOOK_REST_Z = FLANGE_TOP - DISH_SAG + BOOK_REST_GAP + BOOK_T / 2     # its centre at rest on the dish's lowest line
BOOK_POS = (GLASS_X, TRAY_Y, BOOK_REST_Z + 0.002)


def counter_meshes():
    x0, x1 = COUNTER_X
    y0, y1 = COUNTER_Y
    cx0, cx1 = GLASS_X - CUTOUT_D / 2, GLASS_X + CUTOUT_D / 2
    cy0, cy1 = TRAY_Y - CUTOUT_W / 2, TRAY_Y + CUTOUT_W / 2
    slab = mk.extrude([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], COUNTER_T,
                      holes=[[(cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1)]])
    mat = mk.box((MAT_X[1] - MAT_X[0], MAT_Y[1] - MAT_Y[0], 0.0003),
                 ((MAT_X[0] + MAT_X[1]) / 2, (MAT_Y[0] + MAT_Y[1]) / 2, COUNTER_T + 0.00015))
    return [mk.Shape(slab, color=LAMINATE, name="worktop", roughness=0.6),
            mk.Shape(mat, color=GREEN, name="mat", collider="none", density=0.0, roughness=0.8)]


def window_boxes():
    """The glass across the tray's middle and its sill channel on the worktop at each side of the tray (frame: origin on
    the worktop under the glass's middle)."""
    y0, y1 = COUNTER_Y
    out = [((0.0, (y0 + y1) / 2, GLASS_GAP + GLASS_H / 2), (GLASS_T, y1 - y0, GLASS_H), GLASS)]
    for a, b in ((y0, TRAY_Y - TRAY_W / 2 - 0.001), (TRAY_Y + TRAY_W / 2 + 0.001, y1)):
        out.append(((0.0, (a + b) / 2, 0.0005 + SILL / 2), (SILL, b - a, SILL), ALUMINIUM))
    return out


COUNTER = mk.mesh_part("counter", pos=(0.0, 0.0, TOP), shapes=counter_meshes(), friction=COUNTER_FRICTION)
TRAY = mk.mesh_part("tray", pos=(GLASS_X, TRAY_Y, FLANGE_TOP), shapes=tray_meshes(), friction=TRAY_FRICTION)
WINDOW = part("window", pos=(GLASS_X, 0.0, COUNTER_TOP), mass=6.0, boxes=window_boxes(), color=ALUMINIUM, fixed=True,
              friction=0.4)
PASSPORT = part("passport", pos=BOOK_POS, mass=BOOK_MASS, boxes=passport_boxes(), color=NAVY, friction=BOOK_FRICTION,
                meta={"visual_boxes": passport_trim(), "grip_boxes": (1,), "grip_friction": BOOK_TOP_FRICTION})
OBJECTS = [COUNTER, TRAY, WINDOW, PASSPORT]

# -- scoring ---------------------------------------------------------------------------------------------------------------
RELEASE = 0.03            # every gripper finger point this far from the passport: let go
FLAT_DEG = 10.0
REST_MM = 3.0             # its lowest point this close to the surface it lies on
OVER_RIM = 0.005          # "partly over the rim": some of it this far past the dish's rim on the customer's side
LOST_DROP = 0.010         # its centre this far below the worktop outside the tray: it has left the counter
STILL_STEPS, STILL_MM = 20, 1.0
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(step=0, released=False, violation="", hist=[])


_reset_state()

_CORNERS = np.array([(sx * BOOK_D / 2, sy * BOOK_W / 2, sz * BOOK_T / 2)
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def corners(st) -> np.ndarray:
    """The passport's eight corners (world)."""
    return np.array([st.to_world(c) for c in _CORNERS])


def tilt_deg(st) -> float:
    """The angle between the passport's faces and level (0 lying flat either way up)."""
    return math.degrees(math.acos(min(1.0, abs(float((st.rotation() @ np.array([0.0, 0.0, 1.0]))[2])))))


def gap(p, st) -> float:
    """A point's distance from the passport's surface."""
    local = st.to_local(np.asarray(p, float))
    d = np.maximum(np.abs(local) - np.array([BOOK_D, BOOK_W, BOOK_T]) / 2, 0.0)
    return float(np.linalg.norm(d))


def over_tray(xy) -> bool:
    """A point over the tray's outline (flange included)."""
    return abs(xy[0] - GLASS_X) <= TRAY_D / 2 and abs(xy[1] - TRAY_Y) <= TRAY_W / 2


def lost(st) -> bool:
    """The passport has left the counter: its centre below the worktop anywhere but in the tray."""
    return bool(st.pos[2] < COUNTER_TOP - LOST_DROP and not over_tray(st.pos[:2]))


def on_counter(st) -> bool:
    """Lying flat on the customer's side of the worktop with no part of it over the dish."""
    c = corners(st)
    return bool(c[:, 0].max() <= X_RIM and c[:, 0].min() >= COUNTER_X[0] and c[:, 1].min() >= COUNTER_Y[0]
                and c[:, 1].max() <= COUNTER_Y[1] and tilt_deg(st) <= FLAT_DEG
                and COUNTER_TOP - 0.001 <= c[:, 2].min() <= FLANGE_TOP + REST_MM / 1000)


def on_mat(st) -> bool:
    """Lying flat on the worktop with all of its outline over the green mat."""
    c = corners(st)
    return bool(on_counter(st) and c[:, 0].min() >= MAT_X[0] and c[:, 0].max() <= MAT_X[1]
                and c[:, 1].min() >= MAT_Y[0] and c[:, 1].max() <= MAT_Y[1])


def over_rim(st) -> bool:
    """Some of it past the dish's rim on the customer's side (and it has not left the counter)."""
    c = corners(st)
    return bool(c[:, 0].min() <= X_RIM - OVER_RIM and not lost(st))


def level(st) -> float:
    """Where the passport is: 0.75 on the mat, 0.5 out of the dish on the customer's worktop, 0.25 partly over the
    customer-side rim, 0 in the dish or behind the glass."""
    if on_mat(st):
        return 0.75
    if on_counter(st):
        return 0.5
    if over_rim(st):
        return 0.25
    return 0.0


@on_step
def _step(environment) -> None:
    from partkit import _part_state   # noqa: PLC0415

    _STATE["step"] += 1
    st = _part_state(environment, PASSPORT)
    _STATE["released"] = all(gap(p, st) > RELEASE for p in finger_points(environment))


def _update(objects) -> tuple[float, bool]:
    st = objects["passport"]
    if not _STATE["violation"] and lost(st):
        _STATE["violation"] = "the passport left the counter"       # permanent: never cleared
    hist = _STATE["hist"]
    hist.append((corners(st), bool(_STATE["released"])))
    del hist[:-STILL_STEPS]
    if _STATE["violation"]:
        return 0.0, False
    # at rest: for 1 s (20 steps) both grippers have been at least 3 cm from the passport and its whole pose has not
    # moved 1 mm. The release is part of the window, so a passport held still and then let go is not at rest until a
    # full second after the hand has left.
    rest = (len(hist) == STILL_STEPS and all(free for _c, free in hist)
            and max(float(np.linalg.norm(c - hist[-1][0], axis=1).max()) for c, _free in hist) <= STILL_MM / 1000)
    # every level counts only once both grippers have let go of the passport (3 cm away) ...
    if not _STATE["released"]:
        return 0.0, False
    lv = level(st)
    if lv == 0.75:
        return (1.0, True) if rest else (0.75, False)       # ... 1.0 only after 1 s released and still
    return (lv if rest else 0.0), False                     # ... and the lower levels only at rest


def _once(objects) -> tuple[float, bool]:
    """``_update`` once per control step: success() and score() are both called each step, and the stillness window must
    hold one pose per step (20 steps = 1 s), not one per call."""
    if _STATE.get("updated_at") != _STATE["step"]:
        _STATE["updated_at"], _STATE["last"] = _STATE["step"], _update(objects)
    return _STATE["last"]


def success(objects) -> bool:
    return _once(objects)[1]


def distance(objects) -> float:
    st = objects["passport"]
    centre = np.array([(MAT_X[0] + MAT_X[1]) / 2, (MAT_Y[0] + MAT_Y[1]) / 2])
    return float(np.linalg.norm(np.asarray(st.pos[:2], float) - centre))


def score(objects) -> float:
    return _once(objects)[0]


def demo():
    """The scripted demonstration is in demo.py."""
    return []
