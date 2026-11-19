"""On the station bench: the foot end of a manual examination table (Midmark Ritter 204 class), the "upperwrap and
footrest shelf" sub-assembly (Midmark service parts list SF-1864, page E-18, item 9) with its upholstered top, clamped
to the bench with its foot end toward the robot. Both stirrups are stowed in it: each stirrup (020-0337-00-216) lies
folded flat along its stirrup bar (051-1003-01), and the bar is pushed all the way into the shelf through its pivot
boss (053-0387-00), so only the stirrup's hinge end shows at the front.

The job: both stirrups out, unfolded and locked in their outermost positions, nothing holding them. The user guide's
steps: "A) Pull the stirrup out, then unfold. B) Lift the stirrup slightly, then move it left or right as desired.
C) Release stirrup to lock in position." Caution: "Be sure the stirrups are locked in place before using."

Nothing is scripted; the order is held by contact:
- the folded stirrup lies inside the shelf's front window, whose top edge stops it turning up until it is drawn clear;
- the bar's tail carries a detent post that the guide bracket's leaf spring (050-5027-00 with 016-0400-00) presses up
  into the bracket's groove and notches: in the groove (all along the draw) or in a notch the bar cannot swing, and it
  can swing only with its stirrup end lifted (the post drops out); let down between notches it rests unlocked, 4 deg
  front-up, on the bracket's teeth;
- every joint has its real stop (the draw collar, the stirrup's open stop, the boss's swing stops); free joints with
  light damping; the leaf spring is a rotational spring of 12 N m/rad at the bar's lift (2,100 N/m at the post, about
  11 N on the post when locked).

The hardware of record is the bench fixture in fabrication/ (printed parts, steel bar, plywood shell; drawings and
bill of materials there), built to exactly these dimensions, so the model and the fixture are the same object. The
same mechanism on a Midmark table is the optional real object; its sizes are not published, so a lab using one measures
them (README). Dimensions: bar 3/4 x 1/4 in steel flat, 565 mm long, 0.60 kg with its T-rail; stirrup 250 mm from its roll pin to the
tip of its loop, 70 mm across the loop, 0.30 kg; the roll pin starts 20 mm in front of the shelf's face and the bar
draws 400 mm to its collar (the pin then 420 mm out); pivot boss 50 mm behind the face, holding the bar from below;
post 75 mm behind the boss, 6 mm across, 30 mm tall; groove and notches 8 mm wide and 5 mm deep; lock positions 12 deg
inward, straight, 12 deg and 24 deg outward; the bar 140 mm above the bench; shelf 670 mm wide, 580 mm deep, 220 mm
tall.

World frame: x forward, y left, heights above the bench top.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import meshkit as mk  # noqa: E402
from partkit import Hinge, Slider, finger_points, on_reset, on_step, part, set_hinge_target  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = "Set this exam table's stirrups for an exam: both out, unfolded and locked in their outermost positions."

PAINT = (0.82, 0.80, 0.74)      # Midmark's powder-coated steel
VINYL = (0.36, 0.42, 0.50)      # upholstery
CHROME = (0.78, 0.79, 0.81)
STEEL = (0.55, 0.56, 0.58)
DARK = (0.18, 0.19, 0.20)

# -- layout (m) --------------------------------------------------------------------------------------------------------
XF = 0.650                  # the shelf's front face (toward the robot)
SHELF_D, SHELF_HW, SHELF_H = 0.580, 0.335, 0.220
SKIN = 0.0015               # sheet steel
CUSHION = 0.060
SY = 0.245                  # each stirrup bar's centre line, |y|
ZB = 0.140                  # the bars' centre line above the bench
BOSS_BACK = 0.050           # the pivot boss's vertical axis behind the front face
BX = XF + BOSS_BACK
DRAW = 0.400                # travel from stowed to the draw collar
STICK = 0.030               # the bar's front end stands this far out of the face when stowed
BW, BT = 0.019, 0.00635     # bar: 3/4 x 1/4 in steel flat
TAIL_R = 0.075              # the detent post's axis behind the boss axis, fully drawn
TAIL_END = 0.085            # the bar's tail end behind the boss axis, fully drawn
POST_R, POST_H = 0.003, 0.030
NOTCH_W, NOTCH_D = 0.008, 0.005
BAR_MASS, STIRRUP_MASS = 0.60, 0.30   # the bar with its T-rail and spacers (0.54 + 0.06 kg), the stirrup
X_FRONT0 = XF - STICK                       # the bar's front end, stowed
X_TAIL0 = BX + TAIL_END + DRAW              # its tail end, stowed
X_POST0 = BX + TAIL_R + DRAW
PIVOT_BACK = 0.010                          # the roll pin 10 mm behind the bar's front end
CEIL = ZB + BT / 2 + POST_H + 0.0003        # the bracket's underside: 0.3 mm over the post's top with the bar level
# The pivot boss holds the bar only from below, so a folded stirrup lying on the bar slides over it: a T-rail hangs
# under the bar (a 1/2 x 1/8 in strip on 4 mm spacers; the wide spacer, the draw collar, meets the trunnion's lips fully
# drawn); the trunnion's lips run in the gap; everything of the boss is below the bar's underside, and the lift pin is
# 10 mm below the bar's centre line. (Revision 2: before, the boss was a cylinder round the bar that a stowed stirrup
# passed through, which only links of one articulation, which do not collide, allow.)
RAIL_GAP, RAIL_T, RAIL_W = 0.004, 0.003175, 0.0127
RAIL_FRONT = 0.065                          # the rail starts 65 mm behind the bar's front end and runs to its tail
Z_RAIL = ZB - BT / 2 - RAIL_GAP - RAIL_T    # the rail's underside
PIN_DOWN = 0.010                            # the lift pin below the bar's centre line
WIN_W, WIN_TOP, WIN_BOT = 0.090, ZB + 0.0232, Z_RAIL - 0.008   # the shelf's front window for each stirrup

# the stirrup, in its own frame: s along the arm from the roll pin, w across, h up off the bar (folded = lying on the bar)
ARM_S0, ARM_S1, ARM_W, ARM_T = 0.022, 0.150, 0.025, 0.006
LOOP_S0, LOOP_L, LOOP_W, LOOP_R = 0.150, 0.100, 0.070, 0.006
CLEVIS = ((-0.018, 0.035), (-0.016, 0.016), (-0.010, 0.012))      # (s0, s1), (w0, w1), (h0, h1) about the roll pin
UNFOLD_STOP = math.radians(100.0)           # opened 10 deg past upright, toward the user: it rests on its stop
LIFT_K, LIFT_PRE = 12.0, math.radians(4.0)  # leaf spring: N m/rad at the lift, and its preload angle past level
LIFT_LO, LIFT_HI = math.radians(-3.0), math.radians(8.0)
SWING_IN, SWING_OUT = math.radians(15.0), math.radians(27.0)       # the boss's stops, inward and outward
NOTCHES = (-12.0, 0.0, 12.0, 24.0)          # lock positions, degrees outward (0 = straight, the groove's end)
OUTER = math.radians(24.0)


def wbox(x0, x1, y0, y1, z0, z1, color, rot=None):
    """A box by its world extents (z above the bench), optionally turned about z by ``rot`` (rad) about its centre."""
    c = (float((x0 + x1) / 2), float((y0 + y1) / 2), float(TOP + (z0 + z1) / 2))
    b = (c, (float(abs(x1 - x0)), float(abs(y1 - y0)), float(abs(z1 - z0))), color)
    return b if rot is None else (*b, (math.cos(rot / 2), 0.0, 0.0, math.sin(rot / 2)))


def rel(shapes, origin):
    return [(tuple(float(s[0][i] - origin[i]) for i in range(3)), *s[1:]) for s in shapes]


def loop_path(n: int = 28) -> list[tuple[float, float]]:
    """The stirrup's teardrop loop in its (s, w) plane: pointed where it joins the arm, round at its far end."""
    pts = []
    for k in range(n):
        t = 2 * math.pi * k / n
        f = (1 - math.cos(t)) / 2
        pts.append((LOOP_S0 + LOOP_L * f, LOOP_W / 2 * math.sin(t) * math.sqrt(f) * 1.15))
    return pts


class Side:
    """One stirrup: ``sign`` +1 on the left (y > 0), -1 on the right; outward swing is toward the table's edge."""

    def __init__(self, name: str, sign: int):
        self.name, self.sign = name, sign
        self.y = sign * SY
        self.out = -sign                        # boss angle sign of an outward swing (the bar's front points -x)
        self.boss = np.array([BX, self.y, TOP + ZB])
        self.pivot0 = np.array([X_FRONT0 + PIVOT_BACK, self.y, TOP + ZB])   # roll pin, stowed

    def notch_angles(self):
        return [math.radians(a) * self.out for a in NOTCHES]

    def limits(self):
        lo, hi = sorted((self.out * SWING_OUT, -self.out * SWING_IN))
        return lo, hi


LEFT, RIGHT = Side("left", 1), Side("right", -1)
SIDES = (LEFT, RIGHT)


# -- the shelf (fixed) ---------------------------------------------------------------------------------------------------
def _shelf_boxes():
    x0, x1 = XF, XF + SHELF_D
    b = [wbox(x0, x1, SHELF_HW - SKIN, SHELF_HW, 0.0, SHELF_H, PAINT),            # side panels
         wbox(x0, x1, -SHELF_HW, -SHELF_HW + SKIN, 0.0, SHELF_H, PAINT),
         wbox(x0 - 0.004, x1, -SHELF_HW - 0.004, SHELF_HW + 0.004, SHELF_H, SHELF_H + SKIN, PAINT),   # top pan
         wbox(x0 - 0.010, x1, -SHELF_HW - 0.006, SHELF_HW + 0.006, SHELF_H + SKIN, SHELF_H + SKIN + CUSHION, VINYL),
         wbox(x0, x0 + SKIN, -SHELF_HW, SHELF_HW, 0.0, WIN_BOT, PAINT),             # front face below the windows
         wbox(x0, x0 + SKIN, -SHELF_HW, SHELF_HW, WIN_TOP, SHELF_H, PAINT)]         # ... above them
    edges = [-SHELF_HW, -SY - WIN_W / 2, -SY + WIN_W / 2, SY - WIN_W / 2, SY + WIN_W / 2, SHELF_HW]
    for ya, yb in ((edges[0], edges[1]), (edges[2], edges[3]), (edges[4], edges[5])):    # ... between them
        b.append(wbox(x0, x0 + SKIN, ya, yb, WIN_BOT, WIN_TOP, PAINT))
    b.append(wbox(x0 - 0.020, x0, -0.170, 0.170, 0.150, 0.205, PAINT))              # the footrest's front, pushed in
    b += _stand_boxes()
    b.append(wbox(x0 - 0.026, x0 - 0.020, -0.060, 0.060, 0.170, 0.190, DARK))       # its pull
    for s in SIDES:
        b += _bracket_boxes(s)
    return b


STAND_X0 = 0.915            # the bench ends at x 0.91; a support stand at bench height carries the shelf's overhang
STAND_HW, STAND_LEG, STAND_PLATE = 0.300, 0.040, 0.020
FLOOR = -0.760              # the floor below the bench top


def _stand_boxes():
    """A welded steel support stand under the shelf's rear: a 320 x 600 mm top plate at bench height on four 40 mm
    square legs with a low cross rail (a real trial needs one too: the shelf is deeper than the bench is free)."""
    x0, x1 = STAND_X0, XF + SHELF_D + 0.005
    b = [wbox(x0, x1, -STAND_HW, STAND_HW, -STAND_PLATE, 0.0, DARK)]
    for xa in (x0, x1 - STAND_LEG):
        for ya in (-STAND_HW, STAND_HW - STAND_LEG):
            b.append(wbox(xa, xa + STAND_LEG, ya, ya + STAND_LEG, FLOOR, -STAND_PLATE, DARK))
    for xa in (x0, x1 - STAND_LEG):
        b.append(wbox(xa, xa + STAND_LEG, -STAND_HW, STAND_HW, FLOOR + 0.120, FLOOR + 0.160, DARK))
    return b


def _bracket_boxes(s: Side):
    """The guide bracket over the bar's tail: a plate whose underside is the groove's ceiling, the groove's two rails
    along the draw, and teeth round the arc the post swings on, leaving a notch at each lock position."""
    y, c0, c1 = s.y, CEIL - NOTCH_D, CEIL
    out = [wbox(BX + 0.055, X_POST0 + 0.020, y - 0.045, y + 0.045, CEIL, CEIL + 0.003, STEEL)]
    rail = 0.010
    for side in (-1, 1):
        yy = y + side * (NOTCH_W / 2 + rail / 2)
        out.append(wbox(BX + TAIL_R + 0.011, X_POST0 + 0.020, yy - rail / 2, yy + rail / 2, c0, c1, STEEL))   # clear of the arc's band
    half = math.degrees(math.asin((NOTCH_W / 2) / TAIL_R))
    marks = sorted(s.notch_angles())
    lo, hi = s.limits()
    edges = [math.degrees(lo) - 4.0] + [v for a in marks for v in (math.degrees(a) - half, math.degrees(a) + half)] \
        + [math.degrees(hi) + 4.0]
    for a, b in zip(edges[0::2], edges[1::2]):
        n = max(1, math.ceil((b - a) / 5.0))
        for k in range(n):
            ta, tb = math.radians(a + (b - a) * k / n), math.radians(a + (b - a) * (k + 1) / n)
            m = (ta + tb) / 2
            cx, cy = BX + TAIL_R * math.cos(m), y + TAIL_R * math.sin(m)
            chord = 2 * TAIL_R * math.sin((tb - ta) / 2) + 0.0004
            out.append(wbox(cx - 0.008, cx + 0.008, cy - chord / 2, cy + chord / 2, c0, c1, STEEL, rot=m))
    # the support rail under the stowed bar's tail half: it holds the bar level while any of it is behind the boss
    out.append(wbox(BX + 0.100, X_TAIL0 + 0.005, y - 0.012, y + 0.012, Z_RAIL - 0.0100, Z_RAIL - 0.0003, STEEL))
    return out


SHELF_ORIGIN = (XF, 0.0, TOP)
SHELF = part("shelf", pos=SHELF_ORIGIN, mass=9.0, boxes=rel(_shelf_boxes(), SHELF_ORIGIN), color=PAINT,
             friction=0.3, fixed=True, meta={"contact_offset": 0.0005})


# -- the stirrups ----------------------------------------------------------------------------------------------------------
def _stirrup_local_boxes():
    """The stirrup's arm and its clevis round the bar's end, in its own (s, w, h) frame at the roll pin."""
    (cs0, cs1), (cw0, cw1), (ch0, ch1) = CLEVIS
    h_arm = BT / 2 + ARM_T / 2                 # lying on the bar's top face
    return [((ARM_S0 + ARM_S1) / 2, 0.0, h_arm, ARM_S1 - ARM_S0, ARM_W, ARM_T),
            ((cs0 + cs1) / 2, (cw0 + cw1) / 2, (ch0 + ch1) / 2, cs1 - cs0, cw1 - cw0, ch1 - ch0)]


def stirrup_loop_mesh():
    h = BT / 2 + LOOP_R
    return mk.tube([(s, w, h) for s, w in loop_path()], radius=LOOP_R, sides=12, closed=True)


def stirrup_loop_segments():
    """The loop as straight segments (s, w, h) pairs: its colliders (one convex hull each)."""
    h = BT / 2 + LOOP_R
    p = [(s, w, h) for s, w in loop_path()]
    return [(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]


def _build_side(s: Side):
    B = s.boss
    org = tuple(float(v) for v in B)
    yaw_lo, yaw_hi = s.limits()
    bar_boxes = [wbox(X_FRONT0, X_TAIL0, s.y - BW / 2, s.y + BW / 2, ZB - BT / 2, ZB + BT / 2, STEEL)]
    post = [((X_POST0, s.y, TOP + ZB + BT / 2 + POST_H / 2), POST_R, POST_H, STEEL, "Z")]
    # the T-rail under the bar: narrow spacers, the strip, and the wide spacer, the draw collar (it meets the
    # trunnion's lips fully drawn)
    zg0, zg1 = ZB - BT / 2 - RAIL_GAP, ZB - BT / 2
    collar = [wbox(X_FRONT0 + RAIL_FRONT, X_TAIL0, s.y - RAIL_W / 2, s.y + RAIL_W / 2, Z_RAIL, zg0, STEEL)]
    collar += [wbox(x - 0.005, x + 0.005, s.y - 0.003, s.y + 0.003, zg0, zg1, DARK)
               for x in (X_FRONT0 + RAIL_FRONT + 0.010, X_FRONT0 + 0.300, X_TAIL0 - 0.030)]
    collar.append(wbox(BX + 0.016 + DRAW, BX + 0.022 + DRAW, s.y - BW / 2, s.y + BW / 2, zg0, zg1, DARK))
    zp = ZB - PIN_DOWN
    trunnion = [wbox(BX - 0.015, BX + 0.015, s.y - 0.0135, s.y + 0.0135, Z_RAIL - 0.0043, Z_RAIL - 0.0003, DARK)]
    trunnion += [wbox(BX - 0.015, BX + 0.015, s.y + d * 0.0069, s.y + d * 0.0135, Z_RAIL - 0.0003, zg1 - 0.0005, DARK)
                 for d in (-1, 1)]
    yoke = [wbox(BX - 0.025, BX + 0.025, s.y - 0.020, s.y + 0.020, 0.112, 0.120, DARK)]
    yoke += [wbox(BX - 0.006, BX + 0.006, s.y + d * 0.014, s.y + d * 0.020, 0.120, zp + 0.0055, DARK) for d in (-1, 1)]
    yoke.append(wbox(BX + 0.025, BX + 0.090, s.y - 0.008, s.y + 0.008, 0.104, 0.112, DARK))      # the spring's arm
    roller = [((BX + 0.075, s.y, TOP + Z_RAIL - 0.011), 0.011, 0.007, STEEL, "Y")]                 # under the rail
    st_boxes = []
    for (cs, cw, ch, ls, lw, lh) in _stirrup_local_boxes():
        c = s.pivot0 + np.array([cs, cw, ch])
        st_boxes.append((tuple(float(v) for v in c), (float(ls), float(lw), float(lh)), CHROME))
    swing = Hinge(f"{s.name}_swing", pivot=(0.0, 0.0, 0.0), boxes=tuple(rel(yoke, org)),
                  cylinders=tuple(rel(roller, org)),
                  lower=float(yaw_lo), upper=float(yaw_hi), damping=0.05, mass=0.08, axis="Z")
    lift = Hinge(f"{s.name}_lift", pivot=(0.0, 0.0, -PIN_DOWN), parent=swing.name, boxes=tuple(rel(trunnion, org)), lower=float(LIFT_LO),
                 upper=float(LIFT_HI), stiffness=LIFT_K, target=float(-LIFT_PRE), damping=0.05, mass=0.02, axis="Y")
    bar = Slider(f"{s.name}_bar", rest=(0.0, 0.0, 0.0), boxes=tuple(rel(bar_boxes + collar, org)),
                 cylinders=tuple(rel(post, org)), travel=DRAW, rate=0.0, damping=1.0, mass=BAR_MASS, axis="X",
                 sign=-1, parent=lift.name)
    fold = Hinge(f"{s.name}_stirrup", pivot=tuple(float(v) for v in s.pivot0 - B), parent=bar.name,
                 boxes=tuple(rel(st_boxes, org)), lower=float(-UNFOLD_STOP), upper=0.0, damping=0.02,
                 mass=STIRRUP_MASS, axis="Y")
    housing = [wbox(BX - 0.020, BX + 0.020, s.y - 0.020, s.y + 0.020, 0.0, 0.111, DARK)]   # the boss's pedestal
    p = part(f"stirrup_{s.name}", pos=org, mass=0.2, boxes=rel(housing, org), hinges=(swing, lift, fold),
             sliders=(bar,), color=CHROME, friction=0.3, fixed=True, meta={"contact_offset": 0.0005})
    loop_shift = tuple(float(v) for v in s.pivot0 - B)
    visible = mk.Shape(stirrup_loop_mesh().moved(loop_shift), color=CHROME, collider="none", metallic=0.8,
                       roughness=0.25)
    hulls = mk.hulls([mk.tube([a, b], radius=LOOP_R, sides=10).moved(loop_shift)
                      for a, b in stirrup_loop_segments()], color=CHROME)
    mk.attach(p, [visible, *hulls], to=fold.name)
    return p


STIRRUP_L, STIRRUP_R = _build_side(LEFT), _build_side(RIGHT)
OBJECTS = [SHELF, STIRRUP_L, STIRRUP_R]


# -- kinematics (the scorer's own: the stirrups' poses follow from their joints) -----------------------------------------
def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def joints(st, s: Side) -> dict:
    """Swing (rad, + counter-clockwise from above), lift (rad, + stirrup end up), draw (m, 0 stowed) and unfold
    (rad, 0 folded, + opening) of one stirrup, from its part state."""
    return {"swing": float(st.hinges[f"{s.name}_swing"]), "lift": float(st.hinges[f"{s.name}_lift"]),
            "draw": float(-st.sliders[f"{s.name}_bar"]), "unfold": float(-st.hinges[f"{s.name}_stirrup"])}


def bar_frame(s: Side, j: dict):
    """World rotation and origin of the bar's rest frame (its stowed pose moved by the joints)."""
    R = _rz(j["swing"]) @ _ry(j["lift"])
    return R, s.boss + R @ np.array([-j["draw"], 0.0, 0.0])


def bar_point(s: Side, j: dict, p_rest) -> np.ndarray:
    R, o = bar_frame(s, j)
    return o + R @ (np.asarray(p_rest, float) - s.boss)


def stirrup_point(s: Side, j: dict, q_local) -> np.ndarray:
    """A point given in the stirrup's own (s, w, h) frame at the roll pin, in the world."""
    R, o = bar_frame(s, j)
    local = _ry(-j["unfold"]) @ np.asarray(q_local, float)
    return o + R @ (s.pivot0 - s.boss + local)


def watch_points(s: Side, j: dict) -> list[np.ndarray]:
    """Points spanning the bar and the stirrup: if none moves 1 mm, the stirrup's whole pose is still."""
    h = BT / 2 + LOOP_R
    pts = [bar_point(s, j, (x, s.y + w, TOP + ZB)) for x in (X_FRONT0, X_TAIL0) for w in (-BW / 2, BW / 2)]
    pts += [stirrup_point(s, j, q) for q in ((0.0, 0.0, 0.0), (LOOP_S0 + LOOP_L, 0.0, h),
                                             (LOOP_S0 + LOOP_L / 2, LOOP_W / 2, h), (LOOP_S0 + LOOP_L / 2, -LOOP_W / 2, h))]
    return pts


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _box_dist(p, lo, hi) -> float:
    d = np.maximum(np.maximum(np.asarray(lo) - p, 0.0), np.asarray(p) - np.asarray(hi))
    return float(np.linalg.norm(d))


def gripper_gap(pts, s: Side, j: dict) -> float:
    """Smallest distance from the grippers' pad points to this stirrup's bar or stirrup (m)."""
    R, o = bar_frame(s, j)
    best = 1.0
    lo = np.array([X_FRONT0, s.y - BW / 2, TOP + ZB - BT / 2]) - s.boss
    hi = np.array([X_TAIL0, s.y + BW / 2, TOP + ZB + BT / 2]) - s.boss
    Ru = _ry(-j["unfold"])
    segs = [(np.array(a), np.array(b)) for a, b in stirrup_loop_segments()]
    for p in pts:
        q = R.T @ (np.asarray(p, float) - o)                 # in the bar's rest frame, relative to the boss
        best = min(best, _box_dist(q, lo, hi))
        u = Ru.T @ (q - (s.pivot0 - s.boss))                 # in the stirrup's own frame
        for (cs, cw, ch, ls, lw, lh) in _stirrup_local_boxes():
            c, half = np.array([cs, cw, ch]), np.array([ls, lw, lh]) / 2
            best = min(best, _box_dist(u, c - half, c + half))
        best = min(best, min(_seg_dist(u, a, b) for a, b in segs) - LOOP_R)
    return float(best)


# -- scoring ---------------------------------------------------------------------------------------------------------------
DRAWN_TOL = 0.008           # "out": within 8 mm of the draw collar (a post can rise into a notch anywhere in its
                            # 16 mm band, so a locked bar may rest a few mm short of the collar)
OPEN_TOL = math.radians(5.0)     # "unfolded": within 5 deg of its open stop
NOTCH_TOL = math.radians(2.0)    # "locked": the swing within 2 deg of a lock position ...
LOCK_LIFT = math.radians(1.5)    # ... and the stirrup end let down (the post up in the notch; 3.8 deg on a tooth)
PAST_STOP = (math.radians(3.0), 0.003)    # a joint this far past its stop has been forced
RELEASE = 0.03
STILL_STEPS = 20
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), violation=None, finger_check=None, info={},
                  clear={s.name: False for s in SIDES}, recent={s.name: [] for s in SIDES}, gap={},
                  let_go={s.name: [] for s in SIDES})
    if environment is not None:     # the leaf spring's preload: the actuators otherwise aim at the joints' start angle
        for s in SIDES:
            set_hinge_target(environment, f"stirrup_{s.name}", f"{s.name}_lift", -LIFT_PRE)


_reset_state()


def stage(s: Side, j: dict) -> int:
    """2: out, unfolded and locked in its outermost position; 1: out and unfolded; 0 otherwise."""
    out = j["draw"] >= DRAW - DRAWN_TOL
    unfolded = j["unfold"] >= UNFOLD_STOP - OPEN_TOL
    if not (out and unfolded):
        return 0
    locked_outer = abs(j["swing"] - s.out * OUTER) <= NOTCH_TOL and j["lift"] <= LOCK_LIFT
    return 2 if locked_outer else 1


def locked(s: Side, j: dict) -> bool:
    """Down in any lock position (for the record)."""
    return j["draw"] >= DRAW - DRAWN_TOL and j["lift"] <= LOCK_LIFT and \
        min(abs(j["swing"] - a) for a in s.notch_angles()) <= NOTCH_TOL


def forced(s: Side, j: dict) -> str | None:
    lo, hi = s.limits()
    a, d = PAST_STOP
    if not (lo - a <= j["swing"] <= hi + a):
        return f"the {s.name} stirrup was swung past its stop"
    if not (LIFT_LO - a <= j["lift"] <= LIFT_HI + a):
        return f"the {s.name} stirrup bar was levered past its stop"
    if not (-d <= j["draw"] <= DRAW + d):
        return f"the {s.name} stirrup was pulled past its collar"
    if not (-a <= j["unfold"] <= UNFOLD_STOP + a):
        return f"the {s.name} stirrup was forced past its open stop"
    return None


@on_step
def _step(environment) -> None:
    """Every control step: count steps and check where the grippers are (score() cannot see the robot)."""
    _STATE["step"] += 1
    try:
        from yam_arena.scene import object_states

        st = object_states(environment, [f"stirrup_{s.name}" for s in SIDES])
        pts = finger_points(environment)
        for s in SIDES:
            gap = gripper_gap(pts, s, joints(st[f"stirrup_{s.name}"], s))
            _STATE["gap"][s.name] = gap
            _STATE["clear"][s.name] = bool(gap >= RELEASE)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"exam-stirrup-set: gripper check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"] = False
        _STATE["clear"] = {s.name: False for s in SIDES}


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    points, info = 0, {}
    for s in SIDES:
        j = joints(objects[f"stirrup_{s.name}"], s)
        why = forced(s, j)
        if why and not _STATE["violation"]:
            _STATE["violation"] = why
        recent = _STATE["recent"][s.name]
        recent.append(np.array(watch_points(s, j)))
        del recent[:-STILL_STEPS]
        still = len(recent) == STILL_STEPS and all(float(np.abs(r - recent[-1]).max()) < 0.001 for r in recent)
        let_go = _STATE["let_go"][s.name]           # released for the whole window, not only at this step
        let_go.append(bool(_STATE["clear"][s.name]))
        del let_go[:-STILL_STEPS]
        released = len(let_go) == STILL_STEPS and all(let_go)
        stg = stage(s, j)
        counted = stg if (released and still) else 0
        points += counted
        info[s.name] = {"draw_mm": round(1000 * j["draw"], 1), "unfold_deg": round(math.degrees(j["unfold"]), 1),
                        "swing_out_deg": round(math.degrees(j["swing"] * s.out), 1),
                        "lift_deg": round(math.degrees(j["lift"]), 2), "locked": locked(s, j), "stage": stg,
                        "clear": _STATE["clear"][s.name], "released_1s": released, "still": still,
                        "counted": counted}
    info["violation"] = _STATE["violation"]
    _STATE["info"] = info
    result = (0.0, False) if _STATE["violation"] else (points / 4.0, points == 4)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Both stirrups fully out, unfolded on their open stops and locked in their outermost positions, both grippers
    at least 3 cm from each stirrup and its bar, and every point of both stirrups still within 1 mm for 1 s."""
    return _update(objects)[1]


def distance(objects) -> float:
    total = 0.0
    for s in SIDES:
        j = joints(objects[f"stirrup_{s.name}"], s)
        total += max(0.0, DRAW - j["draw"]) + 0.25 * max(0.0, UNFOLD_STOP - j["unfold"]) \
            + 0.48 * abs(j["swing"] - s.out * OUTER)
    return float(total)


def score(objects) -> float:
    """Each stirrup counts only once both grippers have been at least 3 cm from it (its bar and its stirrup) and no
    point of it has moved more than 1 mm, both for the last 1 s. It earns 2 points if it is fully out (within 8 mm of
    its draw collar), unfolded (within 5 deg of its open stop) and locked in its outermost position (the swing within
    2 deg of that lock position and the stirrup end let down, the bar within 1.5 deg of level), 1 point if it is out
    and unfolded but not locked there, 0 otherwise. Score: points / 4 (0, 0.25, 0.5, 0.75, 1.0). 0 for good if a
    stirrup is ever forced 3 deg or 3 mm past one of its stops."""
    return _update(objects)[0]


def demo():
    """Waypoints are not enough for this task: see demo.py."""
    return []
