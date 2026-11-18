"""On the station table: a wired electrical junction box on a plywood wall section, ready to be closed up.

Positions in metres from the midpoint between the arm bases (x forward, y left); heights above the table top.
- Wall section (fixed): 18 mm plywood, standing upright across the table at x 0.297 to 0.315, 0.40 m wide, 0.16 m tall
  (it stops 5 mm below the box's top wall, as on a training board).
- Junction box (fixed, screwed to the wall): a 4 in square steel box 2-1/8 in deep (RACO 232 class, 102 x 102 x 54 mm,
  1.6 mm galvanised steel), its blank cover on and facing the robot; its top wall (102 x 54 mm) is level, 16.5 cm up,
  centred at x 0.27, y 0. Three 1/2 in knockouts along the top wall: at y -0.033 a cable connector holds the cable
  in use (NM cable, led over the wall section and down behind it); at y 0 the knockout is still intact; at y +0.033 the knockout has
  been taken out and left open, a 22.2 mm hole into the box.
- Parts tray (fixed): a grey plastic tray 14 x 16 cm with an 8 mm rim, on the table at x 0.17, y 0.22, holding two snap-in
  knockout seals: a 1/2 in black nylon dome seal (Heyco DP-875 class: 25.8 mm head, four sprung fingers for a 22.2 mm
  hole in a panel up to 3.2 mm) at x 0.155, y 0.20, and a 3/4 in seal (Heyco DP-1093 class: 31 mm head, fingers for a
  28.6 mm hole) at x 0.20, y 0.26.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from partkit import Hinge, finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = "This training junction box is wired and its cover is on: leave it ready to be energised."

# -- the box (RACO 232 class) --------------------------------------------------------------------------------------
BOX_W, BOX_D, WALL_T = 0.102, 0.054, 0.0016
TOPW_Z = TOP + 0.165                    # top of the box's top wall
BOX_X = 0.27                            # top wall centre (x), box centred on y 0
KO_Y = {"conn": -0.033, "intact": 0.0, "open": 0.033}
KO_R = 0.0222 / 2                       # 7/8 in hole for a 1/2 in trade size
BOARD_X0 = BOX_X + BOX_D / 2            # wall face

# -- seals (Heyco DP-875 and DP-1093 class; finger stiffness estimated: check the snap force with a spring scale) --------
SEALS = {
    "seal_half": dict(head_r=0.0258 / 2, finger_r=0.0122, hole_r=KO_R, pos=(0.155, 0.20), color=(0.08, 0.08, 0.09)),
    "seal_3q": dict(head_r=0.031 / 2, finger_r=0.0156, hole_r=0.0286 / 2, pos=(0.20, 0.26), color=(0.12, 0.12, 0.13)),
}
HEAD_T = 0.0025                         # dome head's rim thickness
DOME_H = 0.003                          # dome rises this much above the rim
FINGER_L = 0.009                        # finger length below the head
FINGER_K = float(__import__('os').environ.get('KO_FINGER_K', 0.012))  # N m/rad per finger: about 19 N to push the seal home, 12 N to pull it out (estimate)
SEAL_MASS = 0.005                       # real about 2 g; 5 g for the solver (stated in fidelity)
TRAY = dict(x=0.17, y=0.22, w=0.14, d=0.16, h=0.008)
WALL_H = 0.160                          # the wall section stops just below the box's top wall

STEEL, PLY, GREY, CABLE, COPPER = (0.62, 0.64, 0.66), (0.76, 0.63, 0.45), (0.55, 0.56, 0.58), (0.92, 0.90, 0.78), (0.72, 0.45, 0.2)


def _quat_z(a):
    return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))


def ring(cz, h, r_in, r_out, color, n=16, centre=(0.0, 0.0)):
    out = []
    t = r_out - r_in
    w = 2 * r_out * math.tan(math.pi / n) * 1.02
    for k in range(n):
        a = 2 * math.pi * k / n
        c = (centre[0] + (r_in + t / 2) * math.cos(a), centre[1] + (r_in + t / 2) * math.sin(a), cz)
        out.append((c, (t, w, h), color, _quat_z(a)))
    return out


def _fixtures():
    """The wall section, the box (top wall with its three knockouts, sides, back, cover), the connector and cable, and
    the parts tray, as one fixed part in the world frame (origin on the table at x 0, y 0)."""
    b, cyl = [], []
    zt = TOPW_Z - TOP
    # wall section
    b.append(((BOARD_X0 + 0.009, 0.0, WALL_H / 2), (0.018, 0.40, WALL_H), PLY))
    # box: sides, bottom, back is the wall, cover toward the robot
    zc = zt - BOX_W / 2
    x0, x1 = BOX_X - BOX_D / 2, BOX_X + BOX_D / 2
    for sy in (-1, 1):
        b.append(((BOX_X, sy * (BOX_W / 2 - WALL_T / 2), zc), (BOX_D, WALL_T, BOX_W), STEEL))
    b.append(((BOX_X, 0.0, zt - BOX_W + WALL_T / 2), (BOX_D, BOX_W, WALL_T), STEEL))
    b.append(((x0 + WALL_T / 2, 0.0, zc), (WALL_T, BOX_W, BOX_W), STEEL))          # blank cover
    # top wall: strips between and beside the knockouts, rings round the two open holes, the intact knockout solid
    zw = zt - WALL_T / 2
    for y0, y1 in ((-BOX_W / 2, -0.033 - KO_R), (-0.033 + KO_R, -KO_R - 0.0005), (KO_R + 0.0005, 0.033 - KO_R),
                   (0.033 + KO_R, BOX_W / 2)):
        b.append(((BOX_X, (y0 + y1) / 2, zw), (BOX_D, y1 - y0, WALL_T), STEEL))
    for y in (KO_Y["conn"], KO_Y["open"]):
        for sx in (-1, 1):                     # fill the corners round each hole up to the box's depth
            b.append(((BOX_X + sx * (KO_R + (BOX_D / 2 - KO_R) / 2), y, zw), (BOX_D / 2 - KO_R, 2 * KO_R + 0.001, WALL_T), STEEL))
        b += ring(zw, WALL_T, KO_R, KO_R + 0.006, STEEL, n=16, centre=(BOX_X, y))
    cyl.append(((BOX_X, 0.0, zw), KO_R + 0.0005, WALL_T, STEEL))                       # the intact knockout
    # connector in use: a die-cast NM connector, locknut below the wall, body and clamp above, and its cable
    yc = KO_Y["conn"]
    cyl.append(((BOX_X, yc, zt - WALL_T - 0.003), 0.016, 0.005, STEEL))               # locknut inside the box
    cyl.append(((BOX_X, yc, zt + 0.006), 0.0135, 0.012, STEEL))                       # connector body
    cyl.append(((BOX_X, yc, zt + 0.016), 0.010, 0.008, STEEL))                        # clamp
    cyl.append(((BOX_X, yc, zt + 0.032), 0.0055, 0.024, CABLE))                       # cable rising
    b.append((((BOX_X + BOARD_X0) / 2 + 0.016, yc, zt + 0.044), (BOARD_X0 - BOX_X + 0.032, 0.011, 0.011), CABLE))   # over the wall
    b.append(((BOARD_X0 + 0.0235, yc, (zt + 0.044) / 2), (0.011, 0.011, zt + 0.044), CABLE))   # and down behind it
    # parts tray
    t = TRAY
    b.append(((t["x"], t["y"], 0.002), (t["w"], t["d"], 0.004), GREY))
    for sx in (-1, 1):
        b.append(((t["x"] + sx * (t["w"] / 2 - 0.0015), t["y"], t["h"] / 2), (0.003, t["d"], t["h"]), GREY))
    for sy in (-1, 1):
        b.append(((t["x"], t["y"] + sy * (t["d"] / 2 - 0.0015), t["h"] / 2), (t["w"], 0.003, t["h"]), GREY))
    return part("fixtures", pos=(0.0, 0.0, TOP), boxes=b, cylinders=cyl, mass=10.0, color=STEEL, friction=0.5, fixed=True)


def _seal(name, spec):
    """A snap-in dome seal, frame at the centre of the head's underside: a domed head, a short neck and four fingers on
    sprung hinges. Each finger flares from the neck to its barb (at ``finger_r``) just below a panel's thickness, then
    tapers back in to its tip: pushed down, the hole's edge rides up the taper and folds the fingers in; past 1.6-3.2 mm
    of panel they spring back out under it."""
    hr, fr, col = spec["head_r"], spec["finger_r"], spec["color"]
    neck_r = spec["hole_r"] - 0.0012
    boxes = [((0.0, 0.0, HEAD_T / 2), (2 * hr * 0.72, 2 * hr * 0.72, HEAD_T), col)]
    boxes += ring(HEAD_T / 2, HEAD_T, hr * 0.7, hr, col, n=16)
    boxes.append(((0.0, 0.0, HEAD_T + DOME_H / 2), (hr * 1.0, hr * 1.0, DOME_H), col))          # the dome
    cyl = [((0.0, 0.0, -0.0015), neck_r - 0.002, 0.003, col)]
    hinges = []
    # Each finger is one bar hinged at its root under the head and leaning out at 22.5 degrees: its outer face is the
    # lead-in the hole's edge rides up (from 1.8 mm inside the hole's radius at the tip to the barb, 1.1 mm outside it
    # and 2 mm below the head), and its top end is the barb that springs out under the panel.
    beta = math.atan2(fr - (neck_r - 0.0006), 0.007)
    length = math.hypot(fr - (neck_r - 0.0006), 0.007)
    thick, width = 0.002, 0.0055
    mid_r = (fr + neck_r - 0.0006) / 2 - math.cos(beta) * thick / 2
    mid_z = -0.0055 + math.sin(beta) * thick / 2
    for k in range(4):
        a = math.pi / 2 * k
        c, s = math.cos(a), math.sin(a)
        piv = ((spec["hole_r"] - 0.0008) * c, (spec["hole_r"] - 0.0008) * s, -0.0003)
        qy = (math.cos(beta / 2), 0.0, math.sin(beta / 2), 0.0)
        qz = _quat_z(a)
        q = (qz[0] * qy[0] - qz[3] * qy[3] * 0 - 0, -qz[3] * qy[2], qz[0] * qy[2], qz[3] * qy[0])   # qz * qy
        fb = [((mid_r * c, mid_r * s, mid_z), (thick, width, length), col, q)]
        hinges.append(Hinge(f"{name}_f{k}", pivot=piv, boxes=tuple(fb), axis="X", yaw=a + math.pi / 2,
                            lower=-0.05, upper=0.75, stiffness=FINGER_K, damping=0.002, target=0.0, mass=0.0005))
    x, y = spec["pos"]
    z = TOP + 0.004 + FINGER_L + 0.0005                  # standing on its fingers' tips on the tray floor
    return part(name, pos=(float(x), float(y), float(z)), boxes=boxes, cylinders=cyl, mass=SEAL_MASS, color=col,
                friction=0.5, hinges=tuple(hinges), yaw=math.pi / 4)   # fingers at 45 deg: the pads close between them


FIXTURES = _fixtures()
SEAL_HALF = _seal("seal_half", SEALS["seal_half"])
SEAL_3Q = _seal("seal_3q", SEALS["seal_3q"])
OBJECTS = [FIXTURES, SEAL_HALF, SEAL_3Q]
OPEN_KO = np.array([BOX_X, KO_Y["open"], TOPW_Z])


# -- measurement only --------------------------------------------------------------------------------------------------
_STATE: dict = {}


def _fresh():
    return {"window": [], "released": False, "released_3q": False, "geo": {}, "steps": 0, "why": [], "dropped": False}


_STATE.update(_fresh())


def seal_geo(st) -> dict:
    p = np.asarray(st.pos, float)
    r = st.rotation()
    off = 1000 * float(math.hypot(p[0] - OPEN_KO[0], p[1] - OPEN_KO[1]))
    high = 1000 * float(p[2] - OPEN_KO[2])                # head underside above the wall's top
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, r[2, 2]))))
    return {"pos": p, "off_mm": off, "high_mm": high, "tilt": tilt}


@on_reset
def _reset(env) -> None:
    _STATE.clear()
    _STATE.update(_fresh())


@on_step
def _step(env) -> None:
    from yam_arena.scene import object_states

    st = object_states(env, ["seal_half", "seal_3q"])
    g = {k: seal_geo(v) for k, v in st.items()}
    _STATE["geo"] = g
    _STATE["steps"] += 1
    for k, v in g.items():
        if v["pos"][2] < TOP - 0.05 and not _STATE["dropped"]:
            _STATE["dropped"] = True
            _STATE["why"].append(f"{k} left the table at step {_STATE['steps']}")
    p = g["seal_half"]["pos"]
    fingers = [np.asarray(q, float) for q in finger_points(env)]
    _STATE["released"] = all(float(np.linalg.norm(q - p)) >= 0.03 + 0.013 for q in fingers)
    p3 = g["seal_3q"]["pos"]                              # same test for the 3/4 in seal: 3 cm from its head's edge
    _STATE["released_3q"] = all(float(np.linalg.norm(q - p3)) >= 0.03 + SEALS["seal_3q"]["head_r"] for q in fingers)
    win = _STATE["window"]
    win.append(p)
    if len(win) > 20:
        win.pop(0)


def _still() -> bool:
    win = _STATE["window"]
    return len(win) == 20 and max(float(np.linalg.norm(q - win[-1])) for q in win) <= 0.001


def _levels():
    g = _STATE.get("geo")
    if not g or _STATE["dropped"]:
        return 0.0, False
    h, q = g["seal_half"], g["seal_3q"]
    snapped = h["off_mm"] <= 2.0 and h["high_mm"] <= 1.0 and h["tilt"] <= 5.0
    if snapped:
        return (1.0, True) if (_STATE["released"] and _still()) else (0.75, False)
    if _STATE["released"] and h["off_mm"] <= 2.0 and h["high_mm"] <= 10.0 and h["tilt"] <= 10.0:
        return 0.5, False                                 # in the hole, not snapped, let go
    if any(free and s["off_mm"] <= 6.0 and s["high_mm"] <= 15.0
           for s, free in ((h, _STATE["released"]), (q, _STATE["released_3q"]))):
        return 0.25, False                                # a seal brought onto the open knockout and let go
    return 0.0, False


def success(objects) -> bool:
    """The 1/2 in seal snapped into the open knockout (head flat on the box within 1 mm, centred within 2 mm), both
    grippers 3 cm clear of it, nothing moved more than 1 mm over the last second."""
    return _levels()[1]


def score(objects) -> float:
    """1.0 as success(); 0.75 snapped flush but still held or moving; 0.5 the 1/2 in seal in the hole, not snapped
    (head more than 1 mm up), released; 0.25 either seal brought onto the open knockout and released; 0 otherwise, and
    for good once a seal has left the table. Released: both grippers 3 cm clear of that seal's head."""
    return _levels()[0]


def distance(objects) -> float:
    g = _STATE.get("geo")
    if not g:
        return 0.5
    h = g["seal_half"]
    return float(h["off_mm"] / 1000 + max(0.0, h["high_mm"]) / 1000)
