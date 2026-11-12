"""On the station table, in front of the arms: a hand-crank insulation tester (a "megger") and a cable terminal block.

The tester is a steel case (12 x 12 x 12 cm, 3 kg) at x 0.30, y -0.22 with a dial on its front and a hand crank on top:
a bar 4.5 cm long on a vertical axle that ends in a free-spinning 25 mm knob (30 mm tall). The crank turns against
0.03 N m of friction (0.7 N at the knob) and has no flywheel. The tester only makes its test voltage while the crank
turns at 0.4 rev/s or more (speed averaged over 0.5 s); its needle rests at "off" otherwise.

The terminal block (fixed by weight, 3 kg) at x 0.30 carries three brass test studs (25 mm across, tops 6 cm above the
table) at y +0.06, +0.14 and +0.22, and behind each stud a toggle flag (green upright, red 40 degrees toward the far
side, 0.04 N m detent) at x 0.35. The red test prod (25 mm grip, 0.06 kg) stands tip-down in a holder cup with a flared
mouth at (0.24, +0.28). The prod lead is not simulated; the tester's earth lead is already clipped to the cable armour.
The studs are plain posts: there is nothing on them to clip a lead to.

One of the three cores has failed insulation (it reads 0.5 megohm; the other two read infinity). A core is tested when
the prod's tip rests on its stud while the crank turns fast enough for 1 s. During that time the needle shows that
core's reading. A prod lying loose on a stud does not count, and neither does cranking with the prod away: one hand
turns the crank, the other holds the prod on the stud, 25 cm or more apart.

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
from partkit import Hinge, PartState, finger_points, on_reset, on_step, part, set_hinge_target  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("One core of this cable has failed insulation. Test all three cores with the hand-crank insulation "
               "tester and its red probe before moving any flag (the tester only reads while its handle is being "
               "cranked), then push the flag beside the failed core over to red, put the probe back in its holder and "
               "let go of everything.")

STEEL = (0.45, 0.48, 0.52)
DARK = (0.16, 0.17, 0.19)
BRASS = (0.78, 0.60, 0.20)
RED = (0.82, 0.10, 0.08)
GREEN = (0.10, 0.60, 0.25)
WHITE = (0.92, 0.92, 0.90)
BLACK = (0.05, 0.05, 0.06)
WOOD = (0.55, 0.40, 0.25)

# -- the tester: origin on the table under the crank axle (part frame: x away from the arms, y left) ------------------
TESTER_XY = (0.30, -0.22)
CASE_HALF, CASE_H = 0.060, 0.120
CRANK_R = 0.045                              # axle to knob axis
BAR_Z0, BAR_Z1 = CASE_H + 0.0005, CASE_H + 0.0145
KNOB_R, KNOB_H = 0.0125, 0.030
KNOB_Z0 = 0.135                              # knob 0.135 - 0.165, centre 0.15 m above the table
KNOB_CENTRE_Z = KNOB_Z0 + KNOB_H / 2
CRANK_FRICTION = 0.03                        # N m
OMEGA_MIN = 0.4                              # rev/s, averaged over 0.5 s
OMEGA_WINDOW = 10                            # control steps (0.5 s at 20 Hz)
NEEDLE_PIVOT = (-CASE_HALF - 0.0045, 0.0, 0.050)
NEEDLE_LEN = 0.045
NEEDLE_OFF, NEEDLE_FAULT, NEEDLE_INF = -0.95, -0.55, 0.95   # rad about x; positive swings to the robot's right

CRANK = Hinge("crank", pivot=(0.0, 0.0, CASE_H),
              boxes=(((0.0, CRANK_R / 2, (BAR_Z0 + BAR_Z1) / 2), (0.014, CRANK_R + 0.014, BAR_Z1 - BAR_Z0), DARK),),
              cylinders=(((0.0, 0.0, CASE_H + 0.008), 0.012, 0.016, DARK),),
              lower=-math.inf, upper=math.inf, stiffness=0.0, damping=5.0, effort=CRANK_FRICTION, mass=0.040,
              axis="Z")
KNOB = Hinge("knob", pivot=(0.0, CRANK_R, BAR_Z1), parent="crank",
             cylinders=(((0.0, CRANK_R, KNOB_CENTRE_Z), KNOB_R, KNOB_H, BLACK),),
             lower=-math.inf, upper=math.inf, stiffness=0.0, damping=0.0005, mass=0.010, axis="Z")
NEEDLE = Hinge("needle", pivot=NEEDLE_PIVOT,
               boxes=(((NEEDLE_PIVOT[0] - 0.0015, 0.0, NEEDLE_PIVOT[2] + NEEDLE_LEN / 2), (0.003, 0.004, NEEDLE_LEN),
                       RED),),
               lower=-1.2, upper=1.2, stiffness=0.02, damping=0.004, target=NEEDLE_OFF, mass=0.003, axis="X")
TESTER_BOXES = [
    ((0.0, 0.0, CASE_H / 2), (2 * CASE_HALF, 2 * CASE_HALF, CASE_H), STEEL),
    ((-CASE_HALF - 0.0015, 0.0, 0.065), (0.003, 0.100, 0.080), WHITE),        # the dial face, on the near side
    ((-CASE_HALF - 0.0035, 0.034, 0.080), (0.001, 0.028, 0.040), RED),        # the red band, left end of the scale
]
TESTER = part("tester", pos=(*TESTER_XY, TOP), mass=3.0, boxes=TESTER_BOXES, hinges=(CRANK, KNOB, NEEDLE),
              color=STEEL, friction=1.0)

# -- the terminal block: origin on the table at the middle stud's axis -------------------------------------------------
BLOCK_XY = (0.30, 0.14)
STUD_DY = (-0.08, 0.0, 0.08)                 # studs at y +0.06, +0.14, +0.22
STUD_R, PLATE_H, STUD_TOP = 0.0125, 0.030, 0.060
FLAG_DX, PIVOT_Z = 0.05, 0.030
FLAG_BAR, FLAG_TAB = 0.050, 0.025            # lever length and tab height above the pivot
RED_ANGLE = math.radians(40)
STUD_XY = [(BLOCK_XY[0], BLOCK_XY[1] + dy) for dy in STUD_DY]
FLAG_XY = [(BLOCK_XY[0] + FLAG_DX, BLOCK_XY[1] + dy) for dy in STUD_DY]
FLAG_PIVOT_Z = TOP + PIVOT_Z
FLAG_TAB_ZS = (TOP + PIVOT_Z + FLAG_BAR, TOP + PIVOT_Z + FLAG_BAR + FLAG_TAB)
FLAG_PUSH_Z = TOP + PIVOT_Z + FLAG_BAR + 0.010       # where the demo pushes: in the tab
FLAG_DETENT_K = 0.15                         # N m/rad toward green (0) or red, flipped at the midpoint; capped at 0.06
FLAG_EFFORT = 0.06


def _flag(i: int) -> Hinge:
    y = STUD_DY[i]
    return Hinge(f"flag{i + 1}", pivot=(FLAG_DX, y, PIVOT_Z),
                 boxes=(((FLAG_DX, y, PIVOT_Z + FLAG_BAR / 2), (0.008, 0.012, FLAG_BAR), DARK),
                        ((FLAG_DX, y, PIVOT_Z + FLAG_BAR + FLAG_TAB / 2), (0.012, 0.025, FLAG_TAB), GREEN)),
                 lower=0.0, upper=RED_ANGLE, stiffness=FLAG_DETENT_K, damping=0.01, effort=FLAG_EFFORT, target=0.0,
                 mass=0.010, axis="Y")


BLOCK_BOXES = [
    ((0.03, 0.0, PLATE_H / 2), (0.14, 0.22, PLATE_H), WOOD),
    ((0.098, 0.0, PLATE_H + 0.0005), (0.008, 0.20, 0.001), RED),            # the red mark at the red end of the swing
]
BLOCK_CYLS = [((0.0, dy, PLATE_H + (STUD_TOP - PLATE_H) / 2), STUD_R, STUD_TOP - PLATE_H, BRASS) for dy in STUD_DY]
BLOCK = part("block", pos=(*BLOCK_XY, TOP), mass=3.0, boxes=BLOCK_BOXES, cylinders=BLOCK_CYLS,
             hinges=tuple(_flag(i) for i in range(3)), color=WOOD, friction=1.0)

# -- the probe and its holder -------------------------------------------------------------------------------------------
PROBE_XY = (0.24, 0.28)
TIP_R, TIP_L = 0.006, 0.030
GRIP_R, GRIP_L = 0.0125, 0.120
PROBE_L = TIP_L + GRIP_L
PROBE_Z0 = 0.010                                 # the tip rests on the holder's floor
PROBE_MASS = 0.06
PROBE_BOXES = [((0.0, 0.0, TIP_L + GRIP_L / 2), (0.004, 0.004, 0.10), RED)]
PROBE_CYLS = [((0.0, 0.0, TIP_L / 2), TIP_R, TIP_L, STEEL), ((0.0, 0.0, TIP_L + GRIP_L / 2), GRIP_R, GRIP_L, RED)]
PROBE = part("probe", pos=(*PROBE_XY, TOP + PROBE_Z0), mass=PROBE_MASS, boxes=PROBE_BOXES, cylinders=PROBE_CYLS,
             color=RED, friction=0.9)

SLEEVE_IN, WALL_T, SLEEVE_TOP = 0.014, 0.006, 0.070      # inner half width, wall, top of the straight part
FLARE_RUN, FLARE_RISE = 0.020, 0.025
FLARE_PHI = math.atan2(FLARE_RUN, FLARE_RISE)
FLARE_LEN = math.hypot(FLARE_RUN, FLARE_RISE)
_c, _s = math.cos(FLARE_PHI / 2), math.sin(FLARE_PHI / 2)
_mid_out = SLEEVE_IN + WALL_T / 2 + FLARE_RUN / 2
_mid_z = SLEEVE_TOP + FLARE_RISE / 2
HOLDER_OUT = SLEEVE_IN + WALL_T
HOLDER_BOXES = [
    ((0.0, 0.0, 0.005), (2 * HOLDER_OUT, 2 * HOLDER_OUT, 0.010), DARK),
    ((SLEEVE_IN + WALL_T / 2, 0.0, (0.010 + SLEEVE_TOP) / 2), (WALL_T, 2 * HOLDER_OUT, SLEEVE_TOP - 0.010), DARK),
    ((-SLEEVE_IN - WALL_T / 2, 0.0, (0.010 + SLEEVE_TOP) / 2), (WALL_T, 2 * HOLDER_OUT, SLEEVE_TOP - 0.010), DARK),
    ((0.0, SLEEVE_IN + WALL_T / 2, (0.010 + SLEEVE_TOP) / 2), (2 * SLEEVE_IN, WALL_T, SLEEVE_TOP - 0.010), DARK),
    ((0.0, -SLEEVE_IN - WALL_T / 2, (0.010 + SLEEVE_TOP) / 2), (2 * SLEEVE_IN, WALL_T, SLEEVE_TOP - 0.010), DARK),
    # the flared mouth: four slanted plates leaning outward
    ((_mid_out, 0.0, _mid_z), (0.004, 0.072, FLARE_LEN), DARK, (_c, 0.0, _s, 0.0)),
    ((-_mid_out, 0.0, _mid_z), (0.004, 0.072, FLARE_LEN), DARK, (_c, 0.0, -_s, 0.0)),
    ((0.0, _mid_out, _mid_z), (0.072, 0.004, FLARE_LEN), DARK, (_c, -_s, 0.0, 0.0)),
    ((0.0, -_mid_out, _mid_z), (0.072, 0.004, FLARE_LEN), DARK, (_c, _s, 0.0, 0.0)),
]
HOLDER = part("holder", pos=(*PROBE_XY, TOP), mass=1.0, boxes=HOLDER_BOXES, color=DARK, friction=1.0)
OBJECTS = [TESTER, BLOCK, PROBE, HOLDER]

# -- the cores -----------------------------------------------------------------------------------------------------------
# The faulty core is hidden and set per trial configuration: MEGGER_FAULTY=0, 1 or 2 (stud at y +0.06, +0.14 or
# +0.22) in the environment before the run; 1 by default. It reads 0.5 megohm; the other two read infinity. Nothing
# in the scene shows which one it is: only the needle does, while that core is under test.
FAULTY = int(os.environ.get("MEGGER_FAULTY", "1"))
if FAULTY not in (0, 1, 2):
    raise ValueError(f"MEGGER_FAULTY must be 0, 1 or 2, got {FAULTY}")
READING = {i: ("0.5 MOhm" if i == FAULTY else "inf") for i in range(3)}

# -- scoring thresholds ----------------------------------------------------------------------------------------------------
CONTACT_Z = 0.002           # the prod's tip within 2 mm of the stud's top
CONTACT_R = 0.0065          # and within 6.5 mm of its axis (the 6 mm tip then rests wholly on the 25 mm stud)
CONTACT_TILT = 30.0         # with the prod within 30 degrees of vertical
TEST_STEPS = 20             # 1.0 s of contact with the crank fast enough
CONTROL_HZ = 20
RED_LATCH = 0.45            # rad: a flag past this has been pushed to red (the detent flips at 0.25)
RED_NOW = 0.60              # at the end the flag is red above this
GREEN_NOW = 0.20            # and green below this
DETENT_FLIP = 0.25
HOLDER_TOL, HOLDER_TILT = 0.010, 10.0
DROP_TILT = 50.0            # the prod fell over
DROP_RADIUS = 0.05          # away from the holder, a prod lower than 4 cm above the table has been dropped
DROP_Z = 0.04
RELEASE_GAP = 0.030         # grippers at least 3 cm from the prod, the knob and the flags
STILL_STEPS = 20
STILL_TOL = 0.001
KNOB_AXIS_SPAN = 0.0125

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(theta=[], omega=0.0, volt=False, timers=[0, 0, 0], tested=[False] * 3, red_ever=[False] * 3,
                  guess=False, wrong=False, dropped=False, touched=False, contact=None, recent=[], released=None,
                  fingers=None, finger_check=None, tip=[(9.0, 9.0)] * 3, needle=NEEDLE_OFF, env_ok=None)


_reset_state()


def _tilt(p: PartState) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, p.rotation()[2, 2]))))


def crank_angle(tester: PartState) -> float:
    return float(tester.hinges.get("crank", 0.0))


def flag_angle(block: PartState, i: int) -> float:
    return float(block.hinges.get(f"flag{i + 1}", 0.0))


def tip_offsets(probe: PartState) -> list[tuple[float, float]]:
    """(horizontal distance from the stud's axis, height above the stud's top) of the prod's tip for each stud (m)."""
    tip = probe.pos
    return [(float(math.hypot(tip[0] - x, tip[1] - y)), float(tip[2] - (TOP + STUD_TOP))) for x, y in STUD_XY]


def contact_index(probe: PartState):
    """Index of the stud the prod's tip rests on (within 2 mm of its top, 6.5 mm of its axis, within 30 degrees of
    vertical), or None."""
    if _tilt(probe) > CONTACT_TILT:
        return None
    for i, (d, dz) in enumerate(tip_offsets(probe)):
        if d <= CONTACT_R and abs(dz) <= CONTACT_Z:
            return i
    return None


def knob_axis(tester: PartState):
    """World end points of the knob's axis for the current crank angle."""
    th = crank_angle(tester)
    c = tester.to_world((-CRANK_R * math.sin(th), CRANK_R * math.cos(th), KNOB_Z0))
    return c, c + np.array([0.0, 0.0, KNOB_H])


def flag_segment(block: PartState, i: int):
    """World end points of flag ``i``: from its pivot up through the lever and tab, at its current angle."""
    a = flag_angle(block, i)
    piv = block.to_world((FLAG_DX, STUD_DY[i], PIVOT_Z))
    top = block.to_world((FLAG_DX + (FLAG_BAR + FLAG_TAB) * math.sin(a), STUD_DY[i],
                          PIVOT_Z + (FLAG_BAR + FLAG_TAB) * math.cos(a)))
    return piv, top


def probe_segment(probe: PartState):
    return probe.pos, probe.to_world((0.0, 0.0, PROBE_L))


def seg_dist(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = float(np.clip((p - a).dot(ab) / max(ab.dot(ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def fingers_released(fingers, tester: PartState, block: PartState, probe: PartState) -> bool:
    """Every finger point at least 3 cm from the prod, the knob and each flag (their own radii taken off)."""
    segs = [(*probe_segment(probe), GRIP_R), (*knob_axis(tester), KNOB_R)]
    segs += [(*flag_segment(block, i), 0.0125) for i in range(3)]
    return all(seg_dist(p, a, b) - r >= RELEASE_GAP for p in fingers for a, b, r in segs)


def touching_stud(fingers) -> bool:
    for p in fingers:
        if p[2] < TOP + STUD_TOP + 0.002 and any(math.hypot(p[0] - x, p[1] - y) < STUD_R + 0.006 for x, y in STUD_XY):
            return True
    return False


def observe(states, fingers=None, environment=None) -> None:
    """One control step of bookkeeping (called from the on_step hook): crank speed, contact timers, needle, flag
    detents, permanent flags and the stillness window."""
    s = _STATE
    tester, block, probe = states["tester"], states["block"], states["probe"]
    s["theta"].append(crank_angle(tester))
    del s["theta"][:-(OMEGA_WINDOW + 1)]
    span = (len(s["theta"]) - 1) / CONTROL_HZ
    s["omega"] = abs(s["theta"][-1] - s["theta"][0]) / span / (2 * math.pi) if span > 0 else 0.0
    s["volt"] = s["omega"] >= OMEGA_MIN
    s["tip"] = tip_offsets(probe)
    c = contact_index(probe)
    s["contact"] = c
    for i in range(3):
        s["timers"][i] = s["timers"][i] + 1 if (c == i and s["volt"]) else 0
        if s["timers"][i] >= TEST_STEPS:
            s["tested"][i] = True
    # the needle shows the touched core's reading while the voltage is there, else it rests at off
    s["needle"] = (NEEDLE_FAULT if c == FAULTY else NEEDLE_INF) if (c is not None and s["volt"]) else NEEDLE_OFF
    angles = [flag_angle(block, i) for i in range(3)]
    for i, a in enumerate(angles):
        if a >= RED_LATCH and not s["red_ever"][i]:
            s["red_ever"][i] = True
            if not all(s["tested"]):
                s["guess"] = True
            if i != FAULTY:
                s["wrong"] = True
    if environment is not None:
        try:
            set_hinge_target(environment, "tester", "needle", s["needle"])
            for i, a in enumerate(angles):
                set_hinge_target(environment, "block", f"flag{i + 1}", RED_ANGLE if a > DETENT_FLIP else 0.0)
            s["env_ok"] = True
        except Exception as exc:  # noqa: BLE001 - report once, keep scoring
            if s["env_ok"] is None:
                print(f"megger-core-test: drive update unavailable: {exc!r}", flush=True)
            s["env_ok"] = False
    p = probe.pos
    away = math.hypot(p[0] - PROBE_XY[0], p[1] - PROBE_XY[1]) > DROP_RADIUS
    if _tilt(probe) > DROP_TILT or (away and p[2] < TOP + DROP_Z) or p[2] < TOP - 0.02 or abs(p[1]) > 0.6:
        s["dropped"] = True
    if fingers is not None:
        s["touched"] |= touching_stud(fingers)
        s["released"] = bool(fingers_released(fingers, tester, block, probe))
    s["recent"].append(np.r_[p, 0.05 * np.array(angles), 0.05 * s["theta"][-1]])
    del s["recent"][:-STILL_STEPS]


def _still() -> bool:
    r = _STATE["recent"]
    return len(r) == STILL_STEPS and float(np.ptp(np.array(r), axis=0).max()) < STILL_TOL


@on_step
def _mechanism(environment) -> None:
    from yam_arena.scene import object_states

    states = object_states(environment, ["tester", "block", "probe"])
    try:
        fingers = [np.asarray(p, float) for p in finger_points(environment)]
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001
        if _STATE["finger_check"] is None:
            print(f"megger-core-test: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"], fingers = False, None
    _STATE["fingers"] = fingers
    observe(states, fingers, environment)


def violated() -> bool:
    return bool(_STATE["guess"] or _STATE["wrong"] or _STATE["dropped"] or _STATE["touched"])


def probe_home(probe: PartState) -> bool:
    d = probe.pos - np.array([PROBE_XY[0], PROBE_XY[1], TOP + PROBE_Z0])
    return float(np.linalg.norm(d[:2])) <= HOLDER_TOL and abs(float(d[2])) <= HOLDER_TOL and _tilt(probe) <= HOLDER_TILT


def _level(objects) -> tuple[float, bool]:
    if violated():
        return 0.0, False
    n = sum(_STATE["tested"])
    if n == 0:
        return 0.0, False
    if n < 3:
        return 0.25, False
    block, probe = objects["block"], objects["probe"]
    if not (flag_angle(block, FAULTY) >= RED_NOW and all(flag_angle(block, i) <= GREEN_NOW for i in range(3) if i != FAULTY)):
        return 0.5, False
    # fingers unknown (check unavailable) counts as released: the check only ever lowers a score
    if probe_home(probe) and _STATE["released"] is not False:
        return (1.0, _still()) if _still() else (0.75, False)
    return 0.75, False


def success(objects) -> bool:
    """All three cores tested, only the faulty core's flag at red, the prod back in its holder (within 1 cm and 10 degrees),
    both grippers at least 3 cm from the prod, the knob and the flags, and the prod, flags and crank still (within 1 mm
    / 1 degree) over the last 20 steps (1 s). Any permanent flag is a failure."""
    return _level(objects)[1]


def score(objects) -> float:
    """0 if a flag was ever raised (a flag pushed to red before all three cores were tested, a flag pushed on a
    healthy core, the prod dropped or a gripper touching a stud) or no core was tested; 0.25 one or two cores tested;
    0.5 all three tested but the faulty core's flag is not at red; 0.75 all three tested and only the faulty core
    flagged, but the prod is not back in its holder, a gripper is within 3 cm of the prod, knob or flags, or things
    are not still; 1.0 with all of that met and still for 1 s. A test counts when the tip rests on the stud (within
    2 mm of its top and 6.5 mm of its axis, prod within 30 degrees of vertical) for 1 s while the crank turns at
    0.4 rev/s or more."""
    return _level(objects)[0]


def distance(objects) -> float:
    """Rough metres to go: 10 cm per untested core, 6 cm for the faulty flag, plus the prod's distance from home."""
    tester, block, probe = objects["tester"], objects["block"], objects["probe"]
    d = 0.10 * (3 - sum(_STATE["tested"])) + (0.0 if flag_angle(block, FAULTY) >= RED_NOW else 0.06)
    home = np.array([PROBE_XY[0], PROBE_XY[1], TOP + PROBE_Z0])
    return float(d + (np.linalg.norm(probe.pos - home) if all(_STATE["tested"]) else 0.0))
