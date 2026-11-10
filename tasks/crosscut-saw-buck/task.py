"""On the station table, in front of the arms: a pine log (100 mm across, 160 mm long, along x) lying on a small sawbuck, with
a white chalk mark across it at x 0.30 and a 8 mm starter kerf in the mark. A two-person crosscut saw stands in the kerf:
a thin steel blade 0.50 m long and 60 mm tall that runs along y, with a vertical beech handle (30 mm across, 80 mm tall)
at each end, 0.325 m either side of the kerf, one in front of each arm. Each handle is joined to its end of the blade by a
short flexible link (80 mm of play): with the handle pulled out to the end of the link the blade is pulled along;
with the handle pushed in to the other end of the link it would push the blade, and a thin blade kinks if it is pushed.

A stroke is the blade travelling at least 8 cm to one side. Only a pull moves the blade, so one hand can make a
stroke and then has to wait: the next stroke goes the other way and only the other hand can pull it. Eight strokes cut
through the log. The blade's descent is not simulated: depth is counted per stroke and the kerf is shown widening.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import (PartState, Slider, finger_points, on_reset, on_step, part,  # noqa: E402
                     set_slider_target)
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Saw through the log at the chalk mark with the two-person crosscut saw, one hand on each handle, then "
               "let go of the saw. The thin blade can only be pulled: it kinks if either handle pushes it.")

PINE = (0.72, 0.55, 0.32)
BEECH = (0.80, 0.62, 0.40)
STEEL = (0.76, 0.78, 0.82)
CHALK = (0.97, 0.97, 0.95)
BUCK = (0.35, 0.28, 0.20)
LINK = (0.20, 0.20, 0.22)

# -- geometry: the part's origin is the kerf on the table top; x away from the arms, y left, z up ----------------
LOG_X = 0.30
KERF = 0.004                           # half width of the starter kerf (the blade is 1.5 mm thick)
LOG_R, LOG_Z = 0.05, 0.07              # log radius; its axis height (top at 0.12 m)
HALF_LEN = 0.080                       # each log half runs 76 mm from the kerf
BLADE_E = 0.25                         # the blade's ends are at +-0.25 m when it is centred
BLADE_Z, BLADE_H = 0.13, 0.06          # blade centre height and height (0.10 .. 0.16 m)
HANDLE_Y = BLADE_E + 0.075                       # handle centres at +-0.325 m when the link is at its middle
HANDLE_R, HANDLE_Z, HANDLE_H = 0.015, 0.15, 0.08

BLADE_MAX = 0.10                       # the blade can travel +-0.10 m (end stops in the kerf)
GAP_MIN, GAP_MAX, GAP_0 = 0.04, 0.12, 0.075     # handle centre to blade end: pushing / pulling / at rest
KINK_MARGIN = 0.003                    # within 3 mm of the pushing end of the link
KINK_SPEED = 0.01                      # m/s: the blade moving away from a handle pushing it faster than this kinks
STROKE_LEN = 0.08                      # one stroke: at least 8 cm of blade travel to one side
STROKES_TO_CUT = 8
CUT_PER_STROKE = 0.001                 # the kerf shown widening 1 mm a stroke

# -- thresholds ------------------------------------------------------------------------------------------------
BARE_GAP = 0.01                        # a gripper within 1 cm of the blade is pushing it bare-handed
HANDS_ON = 0.03                        # a gripper within 3 cm of a handle is on it
RELEASE_GAP = 0.03                     # "let go": every gripper 3 cm or more from the saw
HELD_DIST = 0.035                      # a handle is held when a finger point is within this of its axis
STILL_CALLS = 20
STILL_TOL = 0.001
DT = 0.05                              # control step (20 Hz)


def _mm(cx, cy, cz, sx, sy, sz, color):
    return ((cx, cy, cz), (sx, sy, sz), color)


def _log_half(sign):
    c = sign * (KERF + HALF_LEN) / 2
    return ((c, 0.0, LOG_Z), LOG_R, HALF_LEN - KERF, PINE, "X")


ROOT_BOXES = [
    _mm(0.0, 0.0, 0.010, 0.22, 0.20, 0.020, BUCK),                         # sawbuck base
    _mm(0.0, -0.058, 0.040, 0.16, 0.012, 0.040, BUCK),                      # chocks either side of the log
    _mm(0.0, 0.058, 0.040, 0.16, 0.012, 0.040, BUCK),
    _mm(-KERF - 0.003, 0.0, 0.1205, 0.004, 0.030, 0.0015, CHALK),          # chalk mark across the log, both sides of the kerf
    _mm(KERF + 0.003, 0.0, 0.1205, 0.004, 0.030, 0.0015, CHALK),
]
ROOT_CYLINDERS = [_log_half(-1)]

CUT = Slider("cut", rest=(0.0, 0.0, 0.0), cylinders=(_log_half(1),), lo=0.0, hi=0.012, rate=20000.0, damping=200.0,
             mass=0.5, axis="X")
BLADE = Slider("blade", rest=(0.0, 0.0, 0.0), boxes=(_mm(0.0, 0.0, BLADE_Z, 0.0015, 2 * BLADE_E, BLADE_H, STEEL),),
               lo=-0.12, hi=0.12, rate=20000.0, damping=200.0, mass=0.4, axis="Y")


def _handle(name, side):
    y = side * HANDLE_Y
    rod = _mm(0.0, y - side * 0.06, 0.12, 0.004, 0.12, 0.010, LINK)       # the link running in toward the blade end
    return Slider(name, rest=(0.0, y, 0.0), cylinders=(((0.0, y, HANDLE_Z), HANDLE_R, HANDLE_H, BEECH),),
                  boxes=(rod,), lo=-0.17, hi=0.17, rate=60.0, damping=3.0, mass=0.12, axis="Y")


HL, HR = _handle("hl", 1), _handle("hr", -1)

SAW = part("saw", pos=(LOG_X, 0.0, TOP), mass=20.0, boxes=ROOT_BOXES, cylinders=ROOT_CYLINDERS,
           sliders=(CUT, BLADE, HL, HR), color=PINE, friction=1.0)
OBJECTS = [SAW]

_STATE: dict = {}
_MECH: dict = {}


def _fresh_mech() -> dict:
    return {"b": 0.0, "strokes": 0, "ref": 0.0, "dir": 0, "kink": False, "bare": False, "near": True,
            "held_l": False, "held_r": False, "gap_l": GAP_0, "gap_r": GAP_0, "min_margin": 1.0, "min_margin_at": None,
            "step": 0, "kink_at": None, "bare_at": None, "pts": [], "d_blade": 1.0}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(recent=[])
    _MECH.clear()
    _MECH.update(_fresh_mech())
    if environment is not None:
        for name in ("blade", "hl", "hr", "cut"):
            set_slider_target(environment, "saw", name, 0.0)


_reset_state()


# -- the saw's rule -------------------------------------------------------------------------------------------

def couple(b: float, hl: float, hr: float, held_l: bool = True, held_r: bool = True):
    """One step of the blade. ``hl``/``hr`` are the handles' y (m); ``b`` the blade's offset. The blade is pulled by a
    handle at the far end of its link and pushed by one at the near end. Returns (new b, gap_l, gap_r, kink)."""
    lo, hi = -BLADE_MAX, BLADE_MAX
    lo = max(lo, hr + BLADE_E + GAP_MIN)                    # right handle pushing toward +y
    hi = min(hi, hl - BLADE_E - GAP_MIN)                    # left handle pushing toward -y
    if held_l:
        lo = max(lo, hl - BLADE_E - GAP_MAX)                # left handle pulling toward +y (or holding the blade back)
    if held_r:
        hi = min(hi, hr + BLADE_E + GAP_MAX)                # right handle pulling toward -y
    new = min(max(b, lo), hi) if lo <= hi else b           # both hands pulling against each other: the blade stays
    gap_l, gap_r = hl - (new + BLADE_E), (new - BLADE_E) - hr
    v = (new - b) / DT
    kink = (v < -KINK_SPEED and gap_l < GAP_MIN + KINK_MARGIN) or (v > KINK_SPEED and gap_r < GAP_MIN + KINK_MARGIN)
    return new, gap_l, gap_r, bool(kink)


def _count_stroke(m: dict, b: float, hands_on: bool) -> None:
    """Strokes alternate by themselves: a stroke needs 8 cm of blade travel away from the last extreme, and
    travel to the same side as the last stroke only extends that stroke."""
    d = b - m["ref"]
    if m["dir"] and d * m["dir"] > 0:                       # going on the same way: this is the same stroke
        m["ref"] = b
        return
    if abs(d) >= STROKE_LEN and hands_on:
        m["dir"] = 1 if d > 0 else -1
        m["ref"] = b
        m["strokes"] = min(STROKES_TO_CUT, m["strokes"] + 1)
    elif m["dir"] and d * m["dir"] < 0 and abs(d) < STROKE_LEN:
        pass


def _box_dist(p, lo, hi) -> float:
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0)))


def _handle_dist(p, y) -> float:
    horizontal = max(math.hypot(p[0], p[1] - y) - HANDLE_R, 0.0)
    z0, z1 = HANDLE_Z - HANDLE_H / 2, HANDLE_Z + HANDLE_H / 2
    return math.hypot(horizontal, max(z0 - p[2], p[2] - z1, 0.0))


def blade_box(b: float):
    return (np.array([-0.00075, b - BLADE_E, BLADE_Z - BLADE_H / 2]),
            np.array([0.00075, b + BLADE_E, BLADE_Z + BLADE_H / 2]))


@on_step
def _saw_step(environment) -> None:
    """The saw's rule (the physics only holds the blade and handles on their slides): the blade follows a handle
    pulling at the end of its link, is pushed by one at the other end, and kinks (permanently) if pushed."""
    from partkit import _part_state

    m = _MECH
    m["step"] += 1
    saw = _part_state(environment, SAW)
    q = saw.sliders
    hl, hr = HANDLE_Y + q["hl"], -HANDLE_Y + q["hr"]
    try:
        pts = [saw.to_local(p) for p in finger_points(environment)]
        ok = True
    except Exception as exc:  # noqa: BLE001 - report once; an unknown grip never counts as released
        if m["step"] <= 1:
            print(f"crosscut-saw-buck: finger check unavailable: {exc!r}", flush=True)
        pts, ok = [], False
    m["pts"] = pts
    if ok:
        lo, hi = blade_box(m["b"])
        d_hl = [_handle_dist(p, HANDLE_Y + q["hl"]) for p in pts]
        d_hr = [_handle_dist(p, -HANDLE_Y + q["hr"]) for p in pts]
        m["held_l"] = min(math.hypot(p[0], p[1] - hl) for p in pts[:4]) < HELD_DIST
        m["held_r"] = min(math.hypot(p[0], p[1] - hr) for p in pts[4:]) < HELD_DIST
        m["d_blade"] = min(_box_dist(p, lo, hi) for p in pts)
        if m["d_blade"] < BARE_GAP and not m["bare"]:
            m["bare"], m["bare_at"] = True, m["step"]
        m["near"] = bool(min(d_hl + d_hr + [m["d_blade"]]) < RELEASE_GAP)
        hands_on = bool(min(d_hl) < HANDS_ON and min(d_hr) < HANDS_ON)
    else:
        m["held_l"] = m["held_r"] = m["near"] = True
        hands_on = False

    b_old = m["b"]
    b, gap_l, gap_r, kink = couple(b_old, hl, hr, m["held_l"], m["held_r"])
    if kink and not m["kink"]:
        m["kink"], m["kink_at"] = True, m["step"]
    moving = abs(b - b_old) / DT > KINK_SPEED
    if moving:
        for g in (gap_l, gap_r):
            if g - GAP_MIN < m["min_margin"]:
                m["min_margin"], m["min_margin_at"] = g - GAP_MIN, m["step"]
    m.update(b=b, gap_l=gap_l, gap_r=gap_r)
    _count_stroke(m, b, hands_on)

    set_slider_target(environment, "saw", "blade", b)
    set_slider_target(environment, "saw", "cut", CUT_PER_STROKE * m["strokes"])
    # A handle nobody holds is dragged along by the blade (the link is pulled out); a held one stays with its hand.
    t_hl = b + BLADE_E + GAP_MAX - HANDLE_Y if (not m["held_l"] and hl > b + BLADE_E + GAP_MAX) else q["hl"]
    t_hr = b - BLADE_E - GAP_MAX + HANDLE_Y if (not m["held_r"] and hr < b - BLADE_E - GAP_MAX) else q["hr"]
    set_slider_target(environment, "saw", "hl", t_hl)
    set_slider_target(environment, "saw", "hr", t_hr)


# -- scoring --------------------------------------------------------------------------------------------------

def _still(saw: PartState) -> bool:
    cur = (saw.sliders.get("blade", 0.0), saw.sliders.get("hl", 0.0), saw.sliders.get("hr", 0.0))
    recent = _STATE["recent"]
    recent.append(cur)
    del recent[:-STILL_CALLS]
    if len(recent) < STILL_CALLS:
        return False
    arr = np.array(recent)
    return bool(np.all(arr.max(0) - arr.min(0) < STILL_TOL))


def _update(objects) -> tuple[float, bool]:
    m = _MECH
    still = _still(objects["saw"])
    if m["kink"] or m["bare"] or m["strokes"] < 2:
        return 0.0, False
    n = m["strokes"]
    if n < 5:
        return 0.25, False
    if n < STROKES_TO_CUT:
        return 0.5, False
    if not m["near"] and still:
        return 1.0, True
    return 0.75, False


def success(objects) -> bool:
    """Eight strokes (cut through), no kink and no gripper within 1 cm of the blade, every gripper 3 cm or more from the
    saw, and the saw (blade and both handles) still within 1 mm over the last 20 scorer calls."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if the blade ever kinked (a handle pushed it), a gripper was ever within 1 cm of the blade, or fewer than 2
    strokes were made; 0.25 for 2-4 strokes, 0.5 for 5-7; 0.75 for 8 strokes (cut through) but a gripper still within
    3 cm of the saw or the saw not still; 1.0 for cut through, both grippers 3 cm or more from the saw and the saw
    still (moving under 1 mm over 20 steps). A stroke is 8 cm of blade travel to one side, counted with a gripper
    within 3 cm of each handle."""
    return _update(objects)[0]


def distance(objects) -> float:
    """Strokes still to make (0.1 each), plus 0.03 while a gripper is within 3 cm of the saw."""
    return 0.1 * (STROKES_TO_CUT - _MECH["strokes"]) + (0.03 if _MECH["near"] else 0.0)
