"""On the station table, in front of the arms: the table of a manual mill. A dark grey base (200 x 240 x 40 mm) carries a
fixed plate with a scribed 45-degree line from a green start dot to a red end dot (70 mm in x and in y). Two bars cross
over the plate. The saddle bar (blue, lower) lies along y and slides along x; the cross-slide bar (orange, upper) lies
along x and slides along y. The pointer is the crosshair where the two bars cross; it is on the start dot at rest. The
saddle bar carries a feed handle (a 30 x 30 x 50 mm post) at its left end; the cross-slide bar carries one on a rib at its
right end. Turning a handle is modelled as pushing its post: the left handle feeds the pointer in +x, the right one in +y.

Each slide behaves like a lead screw: it does not back-drive and holds its place unless a handle is pushed with more than
about 1 N; above that it moves with damping (about 4 N at 3 cm/s). To follow the diagonal the two handles have to be
worked at the same time, at the same speed, in one steady pass: feeding one slide and then the other cuts a step, and
stopping for 2 s or more burns the work.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, Slider, on_reset, on_step, part, set_slider_target  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]
CONTROL_HZ = 20

INSTRUCTION = ("Move the pointer along the scribed 45-degree line from the start dot to the end dot by working both feed "
               "handles at once, in one steady pass, then let go.")

DARK = (0.25, 0.26, 0.28)
GREY = (0.55, 0.56, 0.58)
BLUE = (0.10, 0.30, 0.75)
ORANGE = (1.00, 0.50, 0.00)
BLACK = (0.04, 0.04, 0.04)
GREEN = (0.05, 0.75, 0.15)
RED = (0.85, 0.08, 0.05)
LINE = (0.95, 0.95, 0.90)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- the mill table: origin at the middle of the base on the table top; x away from the arms, y left ---------------------
BASE_XY = (0.28, 0.0)
BASE_TOP = 0.040
TRAVEL = 0.070
START = (-0.030, -0.040)              # pointer start (part frame); the end dot is START + (TRAVEL, TRAVEL)
POST = 0.030                          # handle post 30 x 30 mm, 50 mm tall
POST_H = 0.050
LEFT_POST = (START[0], 0.150, 0.054 + POST_H / 2)      # rest centre of the saddle bar's handle (part frame)
RIGHT_POST = (-0.060, -0.155, 0.064 + POST_H / 2)      # rest centre of the cross-slide bar's handle
SADDLE_Z, CROSS_Z = (0.043, 0.053), (0.054, 0.064)


def _line():
    a = math.pi / 4
    mid = (START[0] + TRAVEL / 2, START[1] + TRAVEL / 2, BASE_TOP + 0.001)
    length = TRAVEL * math.sqrt(2)
    return (mid, (length, 0.003, 0.002), LINE, (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)))


BASE_BOXES = [_mm(-100, 100, -120, 120, 0, 40, DARK), _line()]
BASE_CYLS = [((START[0], START[1], BASE_TOP + 0.001), 0.006, 0.002, GREEN),
             ((START[0] + TRAVEL, START[1] + TRAVEL, BASE_TOP + 0.001), 0.006, 0.002, RED)]

SADDLE = Slider(
    "saddle", rest=(START[0], 0.045, 0.048),
    boxes=(((START[0], 0.045, 0.048), (0.014, 0.250, 0.010), BLUE),
           ((LEFT_POST[0], LEFT_POST[1], LEFT_POST[2]), (POST, POST, POST_H), BLACK)),
    travel=TRAVEL, rate=1000.0, damping=100.0, mass=0.30, axis="X", sign=1)
CROSS = Slider(
    "cross", rest=(0.0, START[1], 0.059),
    boxes=(((0.0, START[1], 0.059), (0.140, 0.014, 0.010), ORANGE),
           ((RIGHT_POST[0], (RIGHT_POST[1] + START[1]) / 2, 0.059), (0.014, START[1] - RIGHT_POST[1] + 0.015, 0.010), ORANGE),
           ((RIGHT_POST[0], RIGHT_POST[1], RIGHT_POST[2]), (POST, POST, POST_H), BLACK)),
    travel=TRAVEL, rate=1000.0, damping=100.0, mass=0.30, axis="Y", sign=1)

MILL = part("mill", pos=(*BASE_XY, TOP), mass=15.0, boxes=BASE_BOXES, cylinders=BASE_CYLS, sliders=(SADDLE, CROSS),
            color=DARK, friction=1.0)
OBJECTS = [MILL]

# -- the lead screws ---------------------------------------------------------------------------------------------------
STICK = 0.001                         # m of drive deflection (k = 1000 N/m) before a slide gives way: 1 N breakaway

# -- scoring thresholds ------------------------------------------------------------------------------------------------
END_MIN = 0.068                       # both slides at least 68 mm: the pointer is on the end dot
E_GOOD, E_OK = 0.008, 0.015           # peak |dx - dy| for 1.0, and for 0.5
START_AT = 0.002                      # motion has started once a slide is 2 mm along
PAUSE_STEPS = 2 * CONTROL_HZ          # a stop of 2 s or longer is a pause
PAUSE_MOVE = 0.0005                   # moving under 0.5 mm in 2 s is stopped
TIME_LIMIT = 15 * CONTROL_HZ          # from the start of motion to the end dot
HALF = 0.035                          # half the line, for the 0.25 credit
CLEAR = 0.03                          # grippers at least 3 cm from both handle posts at the end
BASE_MOVE_LIMIT = 0.010
STILL_CALLS = 20
STILL_MOVE = 0.001

_STATE: dict = {}
_HOLD: dict = {"saddle": 0.0, "cross": 0.0}


def _fresh() -> dict:
    return dict(n=0, base0=None, failed=False, started=None, reached=None, peak_e=0.0, pause=False, band_ok=True,
                band_progress=0.0, clear=True, hist=[], recent=[])


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(_fresh())
    _HOLD.update(saddle=0.0, cross=0.0)
    if environment is not None:
        for name in ("saddle", "cross"):
            set_slider_target(environment, "mill", name, 0.0)


_reset_state()


def post_boxes(base_pos, dx, dy):
    """(centre, half-size) of the two handle posts in the world frame for slide positions ``dx``, ``dy``."""
    b = np.asarray(base_pos, float)
    half = np.array([POST, POST, POST_H]) / 2
    return [(b + np.array([LEFT_POST[0] + dx, LEFT_POST[1], LEFT_POST[2]]), half),
            (b + np.array([RIGHT_POST[0], RIGHT_POST[1] + dy, RIGHT_POST[2]]), half)]


def clearance(points, base_pos, dx, dy) -> float:
    """Smallest distance (m) from any gripper point to either handle post."""
    best = float("inf")
    for p in points:
        for c, h in post_boxes(base_pos, dx, dy):
            d = np.maximum(np.abs(np.asarray(p, float) - c) - h, 0.0)
            best = min(best, float(np.linalg.norm(d)))
    return best


def lead_screw(held: float, q: float, travel: float = TRAVEL) -> float:
    """New drive target for a slide at ``q`` held at ``held``: it stays put while the deflection is under STICK and
    follows (STICK behind) when pushed harder."""
    if q - held > STICK:
        held = q - STICK
    elif held - q > STICK:
        held = q + STICK
    return min(max(held, 0.0), travel)


def track(dx: float, dy: float, base_moved: float, clear: bool) -> None:
    """Update the pass log for one control step with slides at ``dx``, ``dy`` (m)."""
    s = _STATE
    s["n"] += 1
    s["failed"] |= base_moved > BASE_MOVE_LIMIT
    s["clear"] = clear
    s["hist"].append((dx, dy))
    del s["hist"][:-(PAUSE_STEPS + 1)]
    s["recent"].append((dx, dy))
    del s["recent"][:-STILL_CALLS]
    if s["started"] is None and max(dx, dy) >= START_AT:
        s["started"] = s["n"]
    if s["started"] is None:
        return
    e = abs(dx - dy)
    s["peak_e"] = max(s["peak_e"], e)
    if e > E_GOOD:
        s["band_ok"] = False
    if s["band_ok"]:
        s["band_progress"] = max(s["band_progress"], min(dx, dy))
    if s["reached"] is None and min(dx, dy) >= END_MIN:
        s["reached"] = s["n"]
    if s["reached"] is None and len(s["hist"]) > PAUSE_STEPS:
        ox, oy = s["hist"][0]
        if max(abs(dx - ox), abs(dy - oy)) < PAUSE_MOVE:
            s["pause"] = True


@on_step
def _feed(environment) -> None:
    """Hold the slides like lead screws, and log the pass."""
    from partkit import _part_state, env

    st = _part_state(environment, MILL)
    q = {k: st.sliders.get(k, 0.0) for k in ("saddle", "cross")}
    for name in ("saddle", "cross"):
        _HOLD[name] = lead_screw(_HOLD[name], q[name])
        set_slider_target(environment, "mill", name, _HOLD[name])
    if _STATE["base0"] is None:
        _STATE["base0"] = st.pos.copy()
    moved = float(np.linalg.norm((st.pos - _STATE["base0"])[:2]))
    points = _gripper_points(env()) if env() is not None else []
    track(q["saddle"], q["cross"], moved, clearance(points, st.pos, q["saddle"], q["cross"]) >= CLEAR if points else True)


def _gripper_points(environment) -> list[np.ndarray]:
    """Points along each gripper: the pad tips, the tool point and 4 and 8 cm up the gripper body."""
    from yam_arena import ik, station

    pts = []
    for side in ("left", "right"):
        rob = environment.unwrapped.scene[f"{side}_robot"]
        q = np.asarray(getattr(rob.data.joint_pos, "torch", rob.data.joint_pos)[0].detach().cpu(), float)[:6]
        tcp, rot = ik.tcp_pose(q, station.BASES[side])
        a = rot[:, 2]
        pts += [tcp + 0.013 * a, tcp, tcp - 0.04 * a, tcp - 0.08 * a]
    return pts


def slides(mill: PartState) -> tuple[float, float]:
    return float(mill.sliders.get("saddle", 0.0)), float(mill.sliders.get("cross", 0.0))


def _evaluate(mill: PartState) -> tuple[float, bool]:
    s = _STATE
    dx, dy = slides(mill)
    if s["failed"]:
        return 0.0, False
    recent = s["recent"]
    still = len(recent) == STILL_CALLS and all(max(abs(a - dx), abs(b - dy)) < STILL_MOVE for a, b in recent)
    in_time = s["reached"] is not None and s["started"] is not None and s["reached"] - s["started"] <= TIME_LIMIT
    if min(dx, dy) >= END_MIN and s["started"] is not None:
        if s["peak_e"] <= E_GOOD and not s["pause"] and in_time and s["clear"]:
            return 1.0, bool(still)
        if s["peak_e"] <= E_OK:
            return 0.5, False
    if s["band_progress"] >= HALF:
        return 0.25, False
    return 0.0, False


def success(objects) -> bool:
    """Both slides at 68 mm or more (pointer on the end dot), peak |dx - dy| of 8 mm or less all pass, no stop of 2 s,
    done within 15 s of the start of motion, grippers 3 cm clear of both handles, slides steady for 20 calls."""
    return _evaluate(objects["mill"])[1]


def score(objects) -> float:
    """1.0 the pointer is on the end dot (both slides 68 mm or more), peak |dx - dy| 8 mm or less, no 2 s stop, within
    15 s of the start, grippers 3 cm clear of both handles, steady for 1 s. 0.5 on the end dot with peak |dx - dy| 15 mm
    or less but a stop of 2 s or more, a pass over 15 s, over 8 mm off, or a gripper within 3 cm. 0.25 both slides were
    35 mm along with |dx - dy| never over 8 mm, but the pass failed after that. 0 otherwise, or if the mill moved 1 cm."""
    return _evaluate(objects["mill"])[0]


def distance(objects) -> float:
    """Metres of slide travel still to go: the larger shortfall to 70 mm, plus the difference between the slides."""
    dx, dy = slides(objects["mill"])
    return max(TRAVEL - dx, TRAVEL - dy, 0.0) + abs(dx - dy)
