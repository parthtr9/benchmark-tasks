"""On the station table stands a section of a hand-set concrete wall form, its outside toward the arms: 3/4 in plywood
sheathing with a double 2x4 waler nailed to it, and the ends of two 1/4 in snap ties coming out through the gap
between the two waler timbers, 16 in apart. Each tie end has a 1/2 in button head and 3 mm of slack. Two steel snap-tie
wedges (slotted, with a keyhole at the thin end) stand thick end up in two wooden holders in front of the arms. The
robot must fasten both ties: slip each wedge's keyhole over a tie head and drive the wedge down, so that its slot
catches the tie behind the head and its taper draws the tie out until the slack is gone and the wedge bears on both
waler timbers.

Everything the wedges touch is true size (plywood, waler timbers, their 5/8 in gap, ties, heads, wedges, tie spacing);
only the form's height and length are cut.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, Slider, finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = "Fasten both snap ties of the wall form tight against its double waler with the two steel wedges."

PLY = (0.80, 0.66, 0.42)
PINE = (0.87, 0.74, 0.50)
STEEL = (0.62, 0.64, 0.67)
ZINC = (0.78, 0.80, 0.82)
DARK = (0.35, 0.36, 0.38)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- the form section. Part frame: origin on the table under the middle of the waler's outside face; +x into the
#    wall (away from the arms), y along the wall, z up. The waler's outside face (where the wedges bear) is x = 0.
FORM_LEN = 0.600                 # along the wall (cut down; the tie spacing is real)
FORM_H = 0.250                   # height (cut down)
WALER_D = 0.089                  # a 2x4 on edge: 38 mm tall, 89 mm deep
PLY_T = 0.019                    # 3/4 in plywood
TIE_Z = 0.095                    # the ties' height above the table
GAP = 0.016                      # 5/8 in between the two waler timbers
TIE_Y = 0.2032                   # ties 16 in apart, 8 in either side of the middle
_HL = FORM_LEN * 500
FORM_BOXES = [
    _mm(89, 108, -_HL, _HL, 0, 250, PLY),                                        # plywood sheathing
    _mm(25, 89, -_HL, _HL, 0, 38, PINE),                                         # 2x3 sole plate, flat on the slab,
                                                                                 # set 25 mm back from the waler face
    _mm(0, 89, -_HL, _HL, 1000 * TIE_Z - 8 - 38, 1000 * TIE_Z - 8, PINE),        # lower waler timber
    _mm(0, 89, -_HL, _HL, 1000 * TIE_Z + 8, 1000 * TIE_Z + 8 + 38, PINE),        # upper waler timber
]
# A snap tie end: a 6.4 mm rod through the plywood and the waler gap with a 12.7 mm button head, 4 mm thick. It can
# be drawn 3 mm out toward the arms (the slack between the form and the tie's spreader cone inside the wall) against
# a spring of 2 N/mm; at 3 mm the cone is against the plywood and the tie is tight.
ROD_R, HEAD_R, HEAD_T = 0.0032, 0.00635, 0.004
HEAD_GAP0 = 0.0097               # slack: the head's underside is 9.7 mm outside the waler face
SLACK = 0.003                    # tight: 12.7 mm (the 1/2 in a wedge takes up on a 4-3/4 in tie end)
TIE_RATE = 2000.0
HEAD_X0 = -(HEAD_GAP0 + HEAD_T / 2)          # head centre, part frame, slack


def _tie(name: str, y: float) -> Slider:
    rod0, rod1 = -HEAD_GAP0, WALER_D + PLY_T + 0.017
    return Slider(name, rest=(HEAD_X0, y, TIE_Z),
                  cylinders=(((HEAD_X0, y, TIE_Z), HEAD_R, HEAD_T, STEEL, "X"),
                             (((rod0 + rod1) / 2, y, TIE_Z), ROD_R, rod1 - rod0, STEEL, "X")),
                  travel=SLACK, rate=TIE_RATE, damping=30.0, mass=0.06, axis="X", sign=-1)


# -- a snap-tie wedge. Part frame: origin at the middle of the thin end, on the plane of the bearing edges; z along
#    the wedge toward the thick end, x toward the waler, y across. Pressed from 3 mm steel, 60 mm wide: a flat nose
#    34 mm long that lies on the waler and carries the 26 mm keyhole (8-34 mm from the thin end), then a channel
#    whose web ramps away from the bearing plane (3 mm at the nose to 26 mm at the thick end) between two flanges
#    whose edges bear on the waler. A 10 mm slot runs up the ramp from the keyhole to 14 mm short of the thick end.
W_LEN, W_WIDTH, W_SHEET = 0.156, 0.060, 0.003
HOLE, HOLE_Z0, HOLE_Z1 = 0.026, 0.008, 0.034
NOSE = HOLE_Z1                               # the ramp starts where the keyhole ends
ENTRY_Z = 0.016                              # the head goes through the keyhole's wide part here
TIGHT_Z = 0.085                              # the head sits here (past the slot's midpoint) when the tie is tight
W_SLOPE = (HEAD_GAP0 + SLACK - W_SHEET) / (TIGHT_Z - NOSE)       # 0.19: 1 mm of draw per 5.3 mm of drive
W_ANGLE = math.atan(W_SLOPE)
W_THIN, W_THICK = W_SHEET, W_SHEET + W_SLOPE * (W_LEN - NOSE)    # 3 mm and 26.2 mm
SLOT, SLOT_Z1 = 0.010, 0.142
FUNNEL = (HOLE - SLOT) / 2                   # the keyhole narrows to the slot over its last 8 mm, at 45 degrees
W_MASS = 0.25


def wedge_depth(z: float) -> float:
    """Distance from the bearing plane to the web's outside face, ``z`` from the thin end."""
    return W_SHEET + W_SLOPE * max(0.0, z - NOSE)


def wedge_z_at(depth: float) -> float:
    """Where along the ramp the web's outside face is ``depth`` from the bearing plane."""
    return NOSE + (depth - W_SHEET) / W_SLOPE


def _wedge_boxes():
    q = (math.cos(W_ANGLE / 2), 0.0, -math.sin(W_ANGLE / 2), 0.0)       # the ramp leans back by the taper angle
    u = np.array([-math.sin(W_ANGLE), 0.0, math.cos(W_ANGLE)])          # up the ramp
    p0 = np.array([-W_SHEET / 2, 0.0, NOSE])                            # the ramp's foot (mid-sheet)
    scale = 1.0 / math.cos(W_ANGLE)

    def nose(z0, z1, y0, y1):
        return ((-W_SHEET / 2, (y0 + y1) / 2, (z0 + z1) / 2), (W_SHEET, y1 - y0, z1 - z0), ZINC)

    def ramp(z0, z1, y0, y1):
        c = p0 + u * ((z0 + z1) / 2 - NOSE) * scale + np.array([0.0, (y0 + y1) / 2, 0.0])
        return (tuple(float(v) for v in c), (W_SHEET, y1 - y0, (z1 - z0) * scale), ZINC, q)

    h, s, w = HOLE / 2, SLOT / 2, W_WIDTH / 2
    boxes = [nose(0.0, HOLE_Z0, -w, w),                                              # below the keyhole
             nose(HOLE_Z0, HOLE_Z1, -w, -h), nose(HOLE_Z0, HOLE_Z1, h, w),           # beside the keyhole
             ramp(NOSE, SLOT_Z1, -w, -s), ramp(NOSE, SLOT_Z1, s, w),                 # beside the slot
             ramp(SLOT_Z1, W_LEN, -w, w)]                                            # the thick (struck) end
    # The keyhole's shoulders: two 45 degree bars that lead the tie's rod from the wide part into the slot (a round
    # keyhole does the same). Each bar's inner face is the line from the hole's side to the slot's edge.
    bar_len, bar_w, r = FUNNEL * math.sqrt(2.0), 0.006, math.sqrt(0.5)
    for side in (-1, 1):
        mid = np.array([side * (s + FUNNEL / 2), HOLE_Z1 - FUNNEL / 2])           # middle of the edge (y, z)
        c = mid + np.array([side * r, r]) * bar_w / 2
        phi = -side * math.pi / 4
        boxes.append(((-W_SHEET / 2, float(c[0]), float(c[1])), (W_SHEET, bar_len, bar_w), ZINC,
                      (math.cos(phi / 2), math.sin(phi / 2), 0.0, 0.0)))
    n = 6
    for i in range(1, n):                                                            # the two tapered flanges
        z0, z1 = NOSE + (W_LEN - NOSE) * i / n, NOSE + (W_LEN - NOSE) * (i + 1) / n
        x0 = -(wedge_depth(z0) - W_SHEET)
        for side in (-1, 1):
            y = side * (w - W_SHEET / 2)
            boxes.append((((x0 / 2), y, (z0 + z1) / 2), (-x0, W_SHEET, z1 - z0), ZINC))
    return boxes


WEDGE_BOXES = _wedge_boxes()

# -- fixed layout (world frame: x forward from the arms, y left, metres) -----------------------------------------------
FACE_X = 0.300                   # the waler's outside face
FORM = part("form", pos=(FACE_X, 0.0, TOP), mass=4.4, boxes=FORM_BOXES,
            sliders=(_tie("tie_l", TIE_Y), _tie("tie_r", -TIE_Y)), color=PLY, friction=0.5)
# Kicker blocks (38 x 38 mm) screwed to the table hold the form's foot: a strip along the back of the plywood and a
# short block in front of each end of the sole plate, outside the ties.
KICKERS = [Box("kicker_back", pos=(FACE_X + WALER_D + PLY_T + 0.0200, 0.0, TOP + 0.019), size=(0.038, FORM_LEN, 0.038),
               color=DARK),
           Box("kicker_front_l", pos=(FACE_X + 0.0050, 0.2875, TOP + 0.019), size=(0.038, 0.025, 0.038), color=DARK),
           Box("kicker_front_r", pos=(FACE_X + 0.0050, -0.2875, TOP + 0.019), size=(0.038, 0.025, 0.038), color=DARK)]
# Each wedge stands thick end up on the table, bearing edges toward the form, leaning with its web against the
# rail of a pine stand screwed to the table straight in front of its tie: a rail 55-65 mm up carried by two end
# posts, and a 12 mm stop strip 1.5 mm behind the wedge's thin end. The wedge's weight is in its web, so it rests on
# the rail by itself; it touches the stand on one side only, and lifting it frees it at once (it is thinner lower
# down). The posts are 6 mm clear of the wedge on each side.
HOLD_X = 0.200                   # the bearing plane of a wedge standing in its stand
FOOT_H, RAIL_Z0, RAIL_Z1 = 0.012, 0.055, 0.065


def _holder(tag: str, y: float):
    rail = HOLD_X - wedge_depth(RAIL_Z1) - 0.0002      # the rail's face toward the web
    out = []
    for n, (ax0, ax1, ay0, ay1, az0, az1) in {
            "back": (HOLD_X + 0.0015, HOLD_X + 0.0115, y - 0.045, y + 0.045, 0.0, FOOT_H),
            "rail": (rail - 0.010, rail, y - 0.036, y + 0.036, RAIL_Z0, RAIL_Z1),
            "post_l": (rail - 0.010, HOLD_X - 0.001, y + 0.036, y + 0.045, 0.0, RAIL_Z1),
            "post_r": (rail - 0.010, HOLD_X - 0.001, y - 0.045, y - 0.036, 0.0, RAIL_Z1)}.items():
        out.append(Box(f"holder_{tag}_{n}", pos=((ax0 + ax1) / 2, (ay0 + ay1) / 2, TOP + (az0 + az1) / 2),
                       size=(ax1 - ax0, ay1 - ay0, az1 - az0), color=PINE))
    return out


HOLDERS = _holder("l", TIE_Y) + _holder("r", -TIE_Y)
WEDGE_Z0 = TOP + 0.0005
WEDGE_L = part("wedge_l", pos=(HOLD_X, TIE_Y, WEDGE_Z0), mass=W_MASS, boxes=WEDGE_BOXES, color=ZINC, friction=0.3)
WEDGE_R = part("wedge_r", pos=(HOLD_X, -TIE_Y, WEDGE_Z0), mass=W_MASS, boxes=WEDGE_BOXES, color=ZINC, friction=0.3)
OBJECTS = [*KICKERS, *HOLDERS, FORM, WEDGE_L, WEDGE_R]

# -- scoring -----------------------------------------------------------------------------------------------------------
MOVE_LIMIT = 0.005       # the form moved this far: 0 for the rest of the trial
DRAWN = 0.0025           # a tie counts as tight once drawn out 2.5 of its 3 mm of slack
BEAR = 0.002             # ... with the wedge's bearing edges within 2 mm of the waler face
CLEAR = 0.030            # both grippers at least 3 cm from both wedges: let go
STILL_STEPS = 20         # 1 s at 20 Hz
STILL_TOL = 0.001
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(form0=None, failed=False, recent=[], released=False, steps=0)


_reset_state()


def head_world(form: PartState, side: str) -> np.ndarray:
    """Centre of a tie's button head ("l" or "r"), world frame."""
    y = TIE_Y if side == "l" else -TIE_Y
    return form.to_world((HEAD_X0 + float(form.sliders.get(f"tie_{side}", 0.0)), y, TIE_Z))


def drawn(form: PartState, side: str) -> float:
    """How far the tie has been drawn out of its slack position (m)."""
    return -float(form.sliders.get(f"tie_{side}", 0.0))


def on_tie(form: PartState, wedge: PartState, side: str) -> bool:
    """The wedge hangs on this tie: thick end up, web toward the arms, the rod in its slot and the head outside it."""
    r = wedge.rotation()
    if r[2, 2] < 0.95 or float(r[:, 0] @ form.rotation()[:, 0]) < 0.95:
        return False
    h = wedge.to_local(head_world(form, side))
    in_slot = HOLE_Z1 + 0.004 <= h[2] <= SLOT_Z1 and abs(h[1]) <= SLOT / 2
    outside = -0.035 <= h[0] <= -wedge_depth(float(h[2])) + 0.003
    return bool(in_slot and outside)


def tight(form: PartState, wedge: PartState, side: str) -> bool:
    """On the tie, the tie drawn at least 2.5 mm, and the wedge's bearing edges on the waler face at the tie."""
    if not on_tie(form, wedge, side) or drawn(form, side) < DRAWN:
        return False
    h = wedge.to_local(head_world(form, side))
    edge = form.to_local(wedge.to_world((0.0, 0.0, float(h[2]))))
    return bool(abs(float(edge[0])) <= BEAR)


def _wedge_distance(wedge: PartState, point) -> float:
    """Distance from a world point to the wedge's bounding box (part frame: x -20..0, y +-30, z 0..156 mm)."""
    p = wedge.to_local(point)
    lo, hi = np.array([-W_THICK, -W_WIDTH / 2, 0.0]), np.array([0.0, W_WIDTH / 2, W_LEN])
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - p, p - hi))))


def observe(form: PartState, wedges, fingers) -> None:
    """Once per control step: the permanent form-moved flag, whether both grippers are clear of both wedges
    (``fingers`` None: unknown, counts as holding), and the 1 s window of wedge and tie positions."""
    s = _STATE
    if s["form0"] is None:
        s["form0"] = np.array(form.pos, float)
    if float(np.linalg.norm(np.asarray(form.pos) - s["form0"])) > MOVE_LIMIT:
        s["failed"] = True
    s["released"] = fingers is not None and all(_wedge_distance(w, f) >= CLEAR for w in wedges for f in fingers)
    row = [np.asarray(w.to_world(p), float) for w in wedges for p in ((0.0, 0.0, 0.0), (0.0, 0.0, W_LEN))]
    row.append(np.array([drawn(form, "l"), drawn(form, "r"), 0.0]))
    s["recent"].append(np.concatenate(row))
    del s["recent"][:-STILL_STEPS]
    s["steps"] += 1


@on_step
def _track(environment) -> None:
    from partkit import _part_state

    form = _part_state(environment, FORM)
    wedges = [_part_state(environment, WEDGE_L), _part_state(environment, WEDGE_R)]
    try:
        fingers = [np.asarray(p, float) for p in finger_points(environment)]
    except Exception as exc:  # noqa: BLE001 - an unreadable gripper never counts as released
        if not _STATE.get("warned"):
            print(f"form-tie-wedge: gripper check unavailable: {exc!r}", flush=True)
            _STATE["warned"] = True
        fingers = None
    observe(form, wedges, fingers)


def _still() -> bool:
    recent = _STATE["recent"]
    return len(recent) == STILL_STEPS and float(np.ptp(np.array(recent), axis=0).max()) < STILL_TOL


def _level(objects) -> tuple[float, bool]:
    form, wl, wr = objects["form"], objects["wedge_l"], objects["wedge_r"]
    if _STATE["failed"]:
        return 0.0, False
    # Either wedge may go on either tie.
    n_tight = sum(any(tight(form, w, side) for w in (wl, wr)) for side in ("l", "r"))
    n_hung = sum(any(on_tie(form, w, side) for w in (wl, wr)) for side in ("l", "r"))
    if n_tight == 2:
        still = _still()
        if still and _STATE["released"]:
            return 1.0, True
        return 0.75, False
    if n_tight == 1:
        return 0.5, False
    return (0.25 if n_hung >= 1 else 0.0), False


def success(objects) -> bool:
    """Both ties tight under their wedges, both grippers 3 cm clear of the wedges, nothing moving for 1 s, and the
    form never moved more than 5 mm."""
    return _level(objects)[1]


def score(objects) -> float:
    """1.0 both ties tight, grippers clear, still for 1 s; 0.75 both tight but a gripper is within 3 cm or something
    still moves; 0.5 one tie tight; 0.25 a wedge hangs on a tie by its slot; else 0. 0 once the form has moved 5 mm."""
    return _level(objects)[0]
