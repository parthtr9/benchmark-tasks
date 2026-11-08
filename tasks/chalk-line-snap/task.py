"""On the station table, in front of the arms: a fixed sheet of plywood (200 x 440 x 12 mm) with a pencil tick on its left
and right edges, a chalk line's hook caught on the left edge at the left tick, and the chalk line's reel (a blue case
70 x 35 x 60 mm, 200 g, with a nozzle sticking out of one end) standing on a grey pad nearer the right arm. The line runs
from the hook to the reel's nozzle and pays out freely.

The robot must snap a chalk line from the left tick to the right tick: pin the reel's nozzle on the right tick with one
arm so the line is taut, pluck the line with the other arm (close a gripper on it halfway along, lift it, let it go),
then put the reel back on its pad and let go. A line plucked slack leaves only a smear; a line pinned off the tick
leaves a wrong mark that cannot be erased.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, on_reset, on_step, part, write_pose  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("The chalk line's hook is caught on the left edge of the plywood at the pencil tick. Snap a straight "
               "chalk line from that tick to the pencil tick on the right edge, then put the reel back on its pad and "
               "let go.")

GREY = (0.45, 0.46, 0.50)
DARK = (0.22, 0.23, 0.25)
BLUE = (0.15, 0.35, 0.85)
WOOD = (0.86, 0.70, 0.45)
CHALK = (0.30, 0.55, 1.00)
SMEAR = (0.70, 0.80, 0.95)
STRING = (0.95, 0.95, 0.85)

# -- fixed layout (world frame: x forward from the arms, y left, metres) -------------------------------------------
PLY_X, PLY_Y = 0.30, 0.0
PLY_L, PLY_W, PLY_T = 0.20, 0.44, 0.012          # x, y, z sizes
PLY_TOP = TOP + PLY_T
TICK_L = (0.30, 0.20)               # left tick: where the hook is caught and the line starts
TICK_R = (0.30, -0.20)              # right tick: where the nozzle has to be pinned
HOOK_SIZE = 0.015
HOOK_POS = (0.30, 0.2125, PLY_TOP + HOOK_SIZE / 2)
LINE_Z = PLY_TOP + 0.002            # the taut line lies this high above the plywood

PAD_XY, PAD_SIZE = (0.15, -0.115), (0.12, 0.11, 0.004)
PAD_TOP = TOP + PAD_SIZE[2]
REEL_START = (0.15, -0.12)          # the reel's middle on the pad
REEL_X, REEL_Y, REEL_Z = 0.070, 0.035, 0.060
NOZZLE_LEN = 0.035                  # the nozzle sticks out of the +y face, flush with the reel's underside
NOZZLE_TIP = REEL_Y / 2 + NOZZLE_LEN
REEL_MASS = 0.20
REEL_BOXES = [((0.0, 0.0, REEL_Z / 2), (REEL_X, REEL_Y, REEL_Z), BLUE),
              ((0.0, REEL_Y / 2 + NOZZLE_LEN / 2, 0.006), (0.012, NOZZLE_LEN, 0.012), DARK)]

PLYWOOD = Box("plywood", pos=(PLY_X, PLY_Y, TOP + PLY_T / 2), size=(PLY_L, PLY_W, PLY_T), color=WOOD)
HOOK = Box("hook", pos=HOOK_POS, size=(HOOK_SIZE,) * 3, color=DARK)
PAD = Box("pad", pos=(PAD_XY[0], PAD_XY[1], TOP + PAD_SIZE[2] / 2), size=PAD_SIZE, color=GREY)
REEL = part("reel", pos=(*REEL_START, PAD_TOP), mass=REEL_MASS, boxes=REEL_BOXES, color=BLUE, friction=1.0)

# -- visual markers (kinematic, no collision): the pencil ticks, the line, the chalk marks, the smear ----------------
HIDDEN = (0.30, 0.0, TOP - 0.30)
N_LINE, LINE_SEG = 10, 0.043
N_MARK, MARK_SEG, N_MARKS = 6, 0.072, 2
TICKS = [part(f"tick_{n}", pos=(*xy, PLY_TOP + 0.0004), mass=0.001, visual=True, color=DARK,
              boxes=[((0, 0, 0), (0.008, 0.008, 0.0008), DARK)]) for n, xy in (("l", TICK_L), ("r", TICK_R))]
LINE = [part(f"line_{i}", pos=HIDDEN, mass=0.001, visual=True, color=STRING,
             boxes=[((0, 0, 0), (LINE_SEG, 0.002, 0.002), STRING)]) for i in range(N_LINE)]
MARKS = [[part(f"mark_{k}_{i}", pos=HIDDEN, mass=0.001, visual=True, color=CHALK,
               boxes=[((0, 0, 0), (MARK_SEG, 0.003, 0.0008), CHALK)]) for i in range(N_MARK)] for k in range(N_MARKS)]
SMEAR_PART = part("smear", pos=HIDDEN, mass=0.001, visual=True, color=SMEAR, boxes=[((0, 0, 0), (0.05, 0.03, 0.0008), SMEAR)])
OBJECTS = [PLYWOOD, HOOK, PAD, REEL, *TICKS, *LINE, *[m for ms in MARKS for m in ms], SMEAR_PART]

# -- the line and the pluck ------------------------------------------------------------------------------------------
NOZZLE_PIN = 0.003                  # taut: the nozzle's tip within 3 mm of the plywood top
PLUCK_NEAR = 0.01                   # a gripper closes within 1 cm of the line ...
PLUCK_END = 0.06                    # ... at least 6 cm from both of its ends
PLUCK_LIFT = 0.015                  # lifts it this far, then opens
CLOSED_SPACING = 0.012              # pad gap (m) below this: closed on nothing
OPEN_SPACING = 0.030                # above this: open
HOLD_SPACING = 0.070                # a gripper closed on the reel has its pads at most this far apart
FAR = 0.03                          # grippers at least 3 cm from the reel

# -- scoring thresholds ------------------------------------------------------------------------------------------------
GOOD_ERR = 0.003                    # a taut snap's far end within 3 mm of the right tick
WRONG_ERR = 0.015                   # more than 15 mm off: a wrong line
PAD_RADIUS = 0.02                   # reel back on its pad: its middle within 2 cm
TILT_LIMIT = 10.0                   # deg
STILL_CALLS = 20                    # 1 s at 20 Hz
STILL_M = 0.001

_STATE: dict = {}
_VIS: dict = {}


def _hide_all(environment) -> None:
    from yam_arena.scene import object_states  # noqa: F401  (imported only to fail early when the sim is absent)

    _VIS["line_key"] = None
    for obj in [*LINE, *[m for ms in MARKS for m in ms], SMEAR_PART]:
        write_pose(environment, obj.name, HIDDEN, (0.0, 0.0, 0.0, 1.0))


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(snaps=[], grab={"left_robot": None, "right_robot": None}, held=None, away=None, fingers_ok=None,
                  recent=[], taut=False, was_closed={"left_robot": False, "right_robot": False})
    _VIS.update(line_key=None)
    if environment is not None:
        try:
            _hide_all(environment)
        except Exception as exc:  # noqa: BLE001 - the markers are only for the eye; scoring does not need them
            print(f"chalk-line-snap: markers unavailable: {exc!r}", flush=True)


_reset_state()


def _yaw(p: PartState) -> float:
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def nozzle_tip(reel: PartState) -> np.ndarray:
    """The nozzle tip's underside point in the world."""
    return np.asarray(reel.to_world((0.0, NOZZLE_TIP, 0.0)), float)


def pinned(reel: PartState) -> bool:
    """The nozzle's tip is on the plywood (within 3 mm of its top)."""
    t = nozzle_tip(reel)
    return bool(PLY_TOP - 0.003 <= t[2] <= PLY_TOP + NOZZLE_PIN and abs(t[0] - PLY_X) <= PLY_L / 2
                and abs(t[1] - PLY_Y) <= PLY_W / 2)


def reel_distance(reel: PartState, point) -> float:
    """Distance (m) from a world point to the reel's case or nozzle."""
    c = np.asarray(reel.to_local(point), float)
    best = 1e9
    for centre, size, _ in REEL_BOXES:
        d = np.maximum(np.abs(c - np.asarray(centre)) - np.asarray(size) / 2, 0.0)
        best = min(best, float(np.linalg.norm(d)))
    return best


def _segment_distance(p, a, b) -> tuple[float, float]:
    """Distance from point p to segment a-b, and the position along it (m from a) of the nearest point."""
    ab = b - a
    n2 = float(ab @ ab)
    t = 0.0 if n2 < 1e-12 else min(1.0, max(0.0, float((p - a) @ ab) / n2))
    return float(np.linalg.norm(p - (a + t * ab))), t * math.sqrt(n2)


def arm_info(robot, origin) -> dict:
    """One arm's finger state from its own links and joints: the pad tip point, the gap between its pads, and points on
    both pads (world, relative to the env origin)."""
    from yam_arena.constants import PAD_TIP_PAST_TCP, TCP_OFFSET

    names = list(robot.body_names)
    pose = robot.data.body_link_pose_w
    pose = getattr(pose, "torch", pose)[0].detach().cpu().numpy().astype(float)[names.index("gripper")]
    jp = robot.data.joint_pos
    jp = getattr(jp, "torch", jp)[0].detach().cpu().numpy().astype(float)
    jn = list(robot.joint_names)
    half = max(abs(jp[jn.index(j)]) for j in ("joint7", "joint8") if j in jn)
    x, y, z, w = pose[3:7]
    r = PartState(pos=pose[:3], quat_xyzw=np.array([x, y, z, w]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                  low=np.zeros(3), high=np.zeros(3)).rotation()
    pts = []
    for along in (0.0, PAD_TIP_PAST_TCP):
        centre = pose[:3] + r @ (np.array(TCP_OFFSET) + [0.0, 0.0, along]) - origin
        for side in (1.0, -1.0):
            pts.append(centre + r[:, 0] * side * (half + 0.006))
    tip = pose[:3] + r @ (np.array(TCP_OFFSET) + [0.0, 0.0, PAD_TIP_PAST_TCP]) - origin
    return dict(tip=tip, spacing=2 * half, pts=pts)


# -- the markers -----------------------------------------------------------------------------------------------------

def _quat_x_to(d) -> tuple:
    """Quaternion (x, y, z, w) turning +x onto the unit vector d."""
    d = np.asarray(d, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    axis = np.cross([1.0, 0.0, 0.0], d)
    s = float(np.linalg.norm(axis))
    c = float(d[0])
    if s < 1e-9:
        return (0.0, 0.0, 0.0, 1.0) if c > 0 else (0.0, 0.0, 1.0, 0.0)
    axis = axis / s
    half = math.atan2(s, c) / 2
    return (*(axis * math.sin(half)), math.cos(half))


def _place_along(environment, objs, poly, seg_len: float) -> None:
    """Spread ``objs`` (equal boxes) along the polyline ``poly`` (list of 3-vectors), each along its tangent."""
    poly = [np.asarray(p, float) for p in poly]
    lens = [float(np.linalg.norm(b - a)) for a, b in zip(poly, poly[1:])]
    total = sum(lens)
    n = len(objs)
    for i, obj in enumerate(objs):
        s = (i + 0.5) * total / n
        k = 0
        while k < len(lens) - 1 and s > lens[k]:
            s -= lens[k]
            k += 1
        a, b = poly[k], poly[k + 1]
        d = (b - a) / (lens[k] + 1e-12)
        write_pose(environment, obj.name, tuple(a + d * s), _quat_x_to(d))


def _draw_line(environment, reel: PartState, apex) -> None:
    """The string: straight when taut, a lazy bow when slack, a V while a gripper holds it up."""
    tip = nozzle_tip(reel)
    anchor = np.array([TICK_L[0], TICK_L[1], LINE_Z])
    key = (tuple(np.round(tip, 4)), None if apex is None else tuple(np.round(apex, 4)), bool(_STATE["taut"]))
    if key == _VIS.get("line_key"):
        return
    _VIS["line_key"] = key
    end = np.array([tip[0], tip[1], tip[2] + 0.002])
    if apex is not None:
        poly = [anchor, np.asarray(apex, float), end]
    elif _STATE["taut"]:
        poly = [anchor, end]
    else:
        poly = [anchor + (end - anchor) * f + np.array([0.025 * math.sin(math.pi * f), 0.0, 0.0])
                for f in np.linspace(0.0, 1.0, 9)]
    _place_along(environment, LINE, poly, LINE_SEG)


def _draw_mark(environment, k: int, far) -> None:
    a = np.array([TICK_L[0], TICK_L[1], PLY_TOP + 0.0006])
    b = np.array([far[0], far[1], PLY_TOP + 0.0006])
    _place_along(environment, MARKS[k], [a, b], MARK_SEG)


@on_step
def _mechanism(environment) -> None:
    """The chalk line: taut while a gripper holds the reel with its nozzle pinned on the plywood, plucked when a gripper
    closes on the line, lifts it 15 mm and opens. Reads the robot's links, so it is not available to score()."""
    try:
        from partkit import _part_state

        base = environment.unwrapped
        origin = getattr(base.scene.env_origins, "torch", base.scene.env_origins)[0].detach().cpu().numpy()
        reel = _part_state(environment, REEL)
        arms = {n: arm_info(base.scene[n], origin) for n in ("left_robot", "right_robot")}
        near = {n: any(reel_distance(reel, p) < FAR for p in a["pts"]) for n, a in arms.items()}
        _STATE["held"] = any(near[n] and arms[n]["spacing"] < HOLD_SPACING for n in arms)
        _STATE["away"] = not any(near.values())
        _STATE["taut"] = bool(_STATE["held"] and pinned(reel))
        tip = nozzle_tip(reel)
        a, b = np.array([TICK_L[0], TICK_L[1], LINE_Z]), tip
        apex = None
        for n, arm in arms.items():
            closed = arm["spacing"] < CLOSED_SPACING
            opened = arm["spacing"] > OPEN_SPACING
            g = _STATE["grab"][n]
            if g is None:
                dist, along = _segment_distance(arm["tip"], a, b)
                total = float(np.linalg.norm(b - a))
                if (closed and not _STATE["was_closed"][n] and not near[n] and dist <= PLUCK_NEAR
                        and along >= PLUCK_END and total - along >= PLUCK_END):
                    _STATE["grab"][n] = dict(z0=float(arm["tip"][2]), lift=0.0)
            else:
                if opened:
                    if g["lift"] >= PLUCK_LIFT:    # a pluck: the string lets go from at least 15 mm up
                        _pluck(environment, arm["tip"], tip, bool(_STATE["taut"]))
                    _STATE["grab"][n] = None
                elif closed:
                    g["lift"] = float(arm["tip"][2] - g["z0"])
                    apex = np.array([arm["tip"][0], arm["tip"][1], LINE_Z + max(0.0, g["lift"])])
            _STATE["was_closed"][n] = closed
        _STATE["fingers_ok"] = True
        try:
            _draw_line(environment, reel, apex)
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001 - report once, keep the episode running
        if _STATE["fingers_ok"] is None:
            print(f"chalk-line-snap: mechanism/finger check unavailable: {exc!r}", flush=True)
        _STATE["fingers_ok"] = False
        _STATE["held"] = None
        _STATE["away"] = None
        _STATE["taut"] = False


def _pluck(environment, grip_pt, tip, taut: bool) -> None:
    """A pluck let go: a taut line leaves a chalk mark from the hook to where the nozzle was pinned; a slack one a smear."""
    err = float(np.linalg.norm(np.asarray(tip[:2]) - np.asarray(TICK_R)))
    _STATE["snaps"].append(dict(taut=taut, err=err, far=(float(tip[0]), float(tip[1]))))
    try:
        if taut:
            k = sum(1 for s in _STATE["snaps"] if s["taut"]) - 1
            if k < N_MARKS:
                _draw_mark(environment, k, tip)
        else:
            write_pose(environment, SMEAR_PART.name, (float(grip_pt[0]), float(grip_pt[1]), PLY_TOP + 0.0008),
                       (0.0, 0.0, 0.0, 1.0))
    except Exception:  # noqa: BLE001
        pass


def _taut_snaps() -> list:
    return [s for s in _STATE["snaps"] if s["taut"]]


def _update(objects) -> tuple[float, bool]:
    reel = objects["reel"]
    taut = _taut_snaps()
    if any(s["err"] > WRONG_ERR for s in taut):
        return 0.0, False                                  # a wrong line was snapped: it cannot be erased
    if not taut:
        return 0.0, False                                  # no snap, or only slack smears
    if any(s["err"] > GOOD_ERR for s in taut):
        return 0.25, False
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, reel.rotation()[2, 2]))))
    on_pad = bool(np.linalg.norm(reel.pos[:2] - np.asarray(REEL_START)) <= PAD_RADIUS and tilt <= TILT_LIMIT
                  and abs(float(reel.pos[2]) - PAD_TOP) <= 0.006)
    let_go = _STATE["fingers_ok"] is not True or (not _STATE["held"] and bool(_STATE["away"]))
    recent = _STATE["recent"]
    recent.append(np.r_[reel.pos, _yaw(reel) * REEL_X / 2])
    del recent[:-STILL_CALLS]
    still = len(recent) == STILL_CALLS and float(np.ptp(np.array(recent), axis=0).max()) < STILL_M
    if on_pad and let_go:
        return 1.0, bool(still)
    return 0.5, False


def success(objects) -> bool:
    """A taut snap ended within 3 mm of the right tick (and no other taut snap more than 3 mm off), the reel is back on
    its pad (middle within 2 cm), no gripper within 3 cm of it, and it stayed still for 20 scorer calls."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 no taut snap was made (none, or only slack smears), or any taut snap ended more than 15 mm from the right tick;
    0.25 a taut snap ended 3-15 mm from the tick (even if a good one was also made); 0.5 a good taut snap (within 3 mm)
    was made but the reel is not on its pad or a gripper is still within 3 cm of it; 1.0 the same with the reel back on
    its pad and let go. Snaps are kept once made: a wrong line cannot be erased."""
    return _update(objects)[0]


def distance(objects) -> float:
    """Metres to go: how far the reel's nozzle is from the right tick until a good snap exists, then how far the reel is
    from its pad."""
    reel = objects["reel"]
    taut = _taut_snaps()
    to_pad = float(np.linalg.norm(reel.pos[:2] - np.asarray(REEL_START)))
    if taut and not any(s["err"] > GOOD_ERR for s in taut):
        return to_pad
    return to_pad + float(np.linalg.norm(nozzle_tip(reel)[:2] - np.asarray(TICK_R))) + 0.05
