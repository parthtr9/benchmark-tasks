"""On the station table: a lap steel guitar lying flat on a low stand, its neck (nut end) toward the left arm and its
bridge end toward the right arm. Six strings run along y over the neck; a chrome-steel tone bar (25 mm across, 120 mm
long, with a square handle on top) lies across the strings' direction on a two-post felt rest beyond the nut. Coloured dots on the edge of the neck
mark frets 3 (red), 5 (green), 7 (blue), plus plain white dots at frets 9 and 12; a white tab card lying in front of the
guitar carries a red, a green and a blue dot, in that order from the left.

A lap steel has no frets: one hand lays the steel bar across the strings over a fret marker, and the bar sets the sounding
length (the pitch); the other hand plucks the strings near the bridge. The bar alone makes no sound, and plucking with
no bar gives only the open string. The strings are physical rods only over the neck, where the bar rides on them; the
plucking zone near the bridge has painted strings that the fingers pass over, and the notes are worked out by the task
file (string 1 is tuned to E4).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Play the three notes on the tab card, in order, on the lap steel guitar's first string: for each one "
               "lay the steel tone bar across the strings over the neck dot of the card's colour and pluck the "
               "string, letting the note ring for half a second. Then put the bar back on its rest and take both "
               "hands away.")

STEEL = (0.74, 0.76, 0.80)
WOOD = (0.55, 0.36, 0.20)
DARK = (0.18, 0.18, 0.20)
FELT = (0.45, 0.10, 0.10)
WHITE = (0.93, 0.93, 0.90)
RED, GREEN, BLUE = (0.85, 0.12, 0.12), (0.10, 0.65, 0.20), (0.15, 0.30, 0.90)

# -- geometry (world frame, x forward, y left, heights above the table top; metres) -------------------------------
SCALE = 0.57                  # nut to bridge
Y_NUT = 0.36
Y_BRIDGE = Y_NUT - SCALE      # -0.21
STRING_X = [0.23 + 0.02 * k for k in range(6)]     # string 1 .. 6 (real spacing is 10-11 mm; widened for the lag)
STRING_TOP = 0.060            # the strings' top (rods 8 mm tall on the 52 mm slab)
SLAB_TOP = 0.052
NECK_Y = (0.10, 0.365)        # the rods are physical here
PLUCK_Y = (-0.17, -0.11)      # the pluck zone
PLUCK_Z = (0.054, 0.068)      # a finger tip inside this height band crosses the string plane
PLUCK_SPEED = 0.03            # m/s, least crossing speed that sounds a string
OPEN_PITCH = "E4"
BAR_R, BAR_L, BAR_MASS = 0.0125, 0.12, 0.20
# The bar has a handle on top (as handled "bullet" and Stevens-style bars do): a 30 mm square aluminium block that the
# gripper pinches; a round bar gripped at its sides squeezes out of the pads.
HANDLE_X, HANDLE_Y, HANDLE_H = 0.03, 0.03, 0.03
HANDLE_Z = BAR_R + HANDLE_H / 2 - 0.0   # handle centre above the bar's axis
HANDLE_COLOR = (0.85, 0.55, 0.15)
REST_XY = (0.275, 0.45)
SADDLE_HALF = BAR_R + 0.0005   # the rest's cheeks stand 0.5 mm off the bar's sides: it cannot roll off its place
BAR_CENTRE_Z = STRING_TOP + BAR_R   # centre height of the bar on the strings or on the rest posts
FRET = {3: 0.0, 5: 0.0, 7: 0.0, 9: 0.0, 12: 0.0}
for _f in FRET:
    FRET[_f] = Y_NUT - SCALE * (1.0 - 2.0 ** (-_f / 12.0))   # fret f's marker line (m): 0.2694, 0.2170, 0.1697 ...
CARD_ORDER = (3, 5, 7)
NOTE_NAMES = ["E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4", "C5", "C#5", "D5", "D#5", "E5"]
TOL_CENTS = 40.0
CONTROL_HZ = 20
RING_STEPS = 10               # a note must ring half a second
BAR_ON_STRING_MM = 4.0        # bar underside within this of the string plane counts as on the string
BAR_ANGLE_DEG = 20.0
END_CLEAR = 0.03              # grippers this far from bar and strings at the end
REST_TOL = 0.010
STILL_STEPS = 20
GUITAR_MOVE_LIMIT = 0.010

G0 = (0.32, 0.12)             # guitar part origin (world xy)


def _b(x0, x1, y0, y1, z0, z1, color):
    """A box from world-frame limits (x, y metres; z above the table) in the guitar's part frame."""
    return (((x0 + x1) / 2 - G0[0], (y0 + y1) / 2 - G0[1], (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color)


_GUITAR = [
    _b(0.20, 0.36, Y_BRIDGE - 0.03, Y_NUT + 0.02, 0.0, SLAB_TOP, WOOD),                    # body and neck slab
    _b(0.21, 0.35, Y_BRIDGE - 0.005, Y_BRIDGE + 0.005, SLAB_TOP, 0.056, DARK),               # bridge
    _b(0.22, 0.34, -0.185, -0.17, SLAB_TOP, 0.056, DARK),                                    # pickup
]
for _x in STRING_X:
    _GUITAR.append(_b(_x - 0.0015, _x + 0.0015, NECK_Y[0], NECK_Y[1], SLAB_TOP, STRING_TOP, STEEL))   # string rods
    _GUITAR.append(_b(_x - 0.0008, _x + 0.0008, Y_BRIDGE, NECK_Y[0], SLAB_TOP, SLAB_TOP + 0.0005, STEEL))  # painted
for _f, _c in ((3, RED), (5, GREEN), (7, BLUE), (9, WHITE), (12, WHITE)):
    _GUITAR.append(_b(0.346, 0.356, FRET[_f] - 0.005, FRET[_f] + 0.005, SLAB_TOP, SLAB_TOP + 0.0006, _c))
# the tab card, flat on the table in front of the guitar: red, green, blue from the left (+y) to the right
_GUITAR.append(_b(0.42, 0.50, -0.05, 0.05, 0.0, 0.002, WHITE))
for _k, (_f, _c) in enumerate(zip(CARD_ORDER, (RED, GREEN, BLUE))):
    _y = 0.025 - 0.025 * _k
    _GUITAR.append(_b(0.445, 0.475, _y - 0.009, _y + 0.009, 0.002, 0.0026, _c))
# the bar rest: two felt posts under the bar's ends, each with two cheeks that make a snug saddle, 0.5 mm clear of the bar each side
for _px in (REST_XY[0] - 0.04, REST_XY[0] + 0.04):
    _GUITAR.append(_b(_px - 0.01, _px + 0.01, REST_XY[1] - SADDLE_HALF, REST_XY[1] + SADDLE_HALF, 0.0, STRING_TOP, FELT))
    for _s in (-1, 1):
        _y0 = REST_XY[1] + _s * SADDLE_HALF
        _GUITAR.append(_b(_px - 0.01, _px + 0.01, min(_y0, _y0 + _s * 0.01), max(_y0, _y0 + _s * 0.01), 0.0,
                          STRING_TOP + BAR_R, FELT))
        # a 45 degree lead-in flaring out from the top of each cheek, so a bar set down a few mm off still slides in
        _a = _s * math.radians(45.0) / 2
        _c = (_px - G0[0], REST_XY[1] + _s * (SADDLE_HALF + 0.0106 + 0.0015) - G0[1], STRING_TOP + BAR_R + 0.0106 - 0.002)
        _GUITAR.append((_c, (0.02, 0.03, 0.004), FELT, (math.cos(_a), math.sin(_a), 0.0, 0.0)))

GUITAR = part("guitar", pos=(*G0, TOP), mass=6.0, boxes=_GUITAR, color=WOOD, friction=0.8)
BAR = part("bar", pos=(REST_XY[0], REST_XY[1], TOP + BAR_CENTRE_Z + 0.0005), mass=BAR_MASS,
           boxes=[((0, 0, HANDLE_Z), (HANDLE_X, HANDLE_Y, HANDLE_H), HANDLE_COLOR)],
           cylinders=[((0, 0, 0), BAR_R, BAR_L, STEEL, "X")], color=STEEL, friction=0.6)
OBJECTS = [GUITAR, BAR]


# -- the music: pitch from the bar, notes from plucks -----------------------------------------------------------------
def bar_contact(bar: PartState, x_line: float = STRING_X[0]) -> dict:
    """Where the bar meets the line x = ``x_line`` (string 1): the contact's y, the gap between the bar's underside and
    the string plane, how far along the bar the line is, the bar's angle off the x axis, and whether it is on string 1."""
    c = np.asarray(bar.pos, float) - np.array([0.0, 0.0, TOP])
    a = bar.rotation()[:, 0]
    if a[0] < 0:
        a = -a
    angle = math.degrees(math.acos(max(-1.0, min(1.0, a[0]))))
    s = (x_line - c[0]) / max(a[0], 1e-6)
    y = c[1] + s * a[1]
    z_axis = c[2] + s * a[2]
    gap = z_axis - BAR_R * math.sqrt(max(0.0, 1.0 - a[2] ** 2)) - STRING_TOP
    covers = abs(s) <= BAR_L / 2 - 0.005
    on = bool(covers and angle <= BAR_ANGLE_DEG and gap <= BAR_ON_STRING_MM / 1000 and gap >= -0.004
              and Y_BRIDGE + 0.05 < y < Y_NUT)
    return dict(y=float(y), gap=float(gap), s=float(s), angle=float(angle), on=on)


def pitch_of(y_contact: float | None) -> tuple[float, str]:
    """(semitones above the open string, name or 'out of tune') for a sounding length ending at ``y_contact`` (the
    bar's contact line; None: no bar on the string, the open string)."""
    if y_contact is None:
        return 0.0, OPEN_PITCH
    semis = 12.0 * math.log2(SCALE / max(y_contact - Y_BRIDGE, 1e-3))
    idx = int(round(semis))
    cents = 100.0 * (semis - idx)
    if 0 <= idx < len(NOTE_NAMES) and abs(cents) <= TOL_CENTS:
        return semis, NOTE_NAMES[idx]
    return semis, "out of tune"


def tolerance_mm(fret: int) -> float:
    """Half-width of a note's window along the string at ``fret``, mm (the tuner's 40 cents)."""
    length = FRET[fret] - Y_BRIDGE
    return 1000.0 * length * (2.0 ** (TOL_CENTS / 1200.0) - 1.0)


EXPECTED = [NOTE_NAMES[f] for f in CARD_ORDER]        # G4, A4, B4

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, prev={}, ring=None, played=[], failed=None, plucks=[], recent=[], guitar0=None,
                  events=[], notes_heard=[], last=None, done_ready=False)


_reset_state()


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = max(0.0, min(1.0, float((p - a) @ ab) / float(ab @ ab)))
    return float(np.linalg.norm(p - (a + t * ab)))


def grippers_clear(bar: PartState, grips: dict) -> bool:
    """Every gripper point is at least END_CLEAR from the bar and out of the string zone (x 0.22-0.34, y -0.17..0.37,
    up to 3 cm over the strings)."""
    c = np.asarray(bar.pos, float) - np.array([0.0, 0.0, TOP])
    a = bar.rotation()[:, 0]
    p0, p1 = c - a * BAR_L / 2, c + a * BAR_L / 2
    for pts in grips.values():
        for p in pts:
            q = np.asarray(p, float) - np.array([0.0, 0.0, TOP])
            if _seg_dist(q, p0, p1) - BAR_R < END_CLEAR:
                return False
            if 0.22 <= q[0] <= 0.34 and PLUCK_Y[0] <= q[1] <= Y_NUT + 0.01 and q[2] < STRING_TOP + END_CLEAR:
                return False
    return True


def bar_on_rest(bar: PartState) -> bool:
    c = np.asarray(bar.pos, float) - np.array([0.0, 0.0, TOP])
    angle = bar_contact(bar, float(bar.pos[0]))["angle"]
    return bool(math.hypot(c[0] - REST_XY[0], c[1] - REST_XY[1]) < REST_TOL and abs(c[2] - BAR_CENTRE_Z) < 0.005
                and angle < 10.0)


def _fail(why: str) -> None:
    if _STATE["failed"] is None:
        _STATE["failed"] = why
        _STATE["events"].append((_STATE["step"], "FLAG " + why))


def tick(bar: PartState, guitar: PartState, grips: dict) -> None:
    """One control step of the music rules. ``grips`` maps "left"/"right" to the gripper's points in the world: pad tip
    first, then the tool point and 4 and 8 cm up the gripper."""
    S = _STATE
    S["step"] += 1
    n = S["step"]
    if S["guitar0"] is None:
        S["guitar0"] = np.asarray(guitar.pos, float).copy()
    if float(np.linalg.norm(np.asarray(guitar.pos, float) - S["guitar0"])) > GUITAR_MOVE_LIMIT:
        _fail("guitar moved")
    bc = np.asarray(bar.pos, float) - np.array([0.0, 0.0, TOP])
    if bc[2] < 0.04:
        _fail("bar dropped")
    ct = bar_contact(bar)
    S["last"] = ct
    for side, pts in grips.items():
        tip = np.asarray(pts[0], float) - np.array([0.0, 0.0, TOP])
        if 0.20 <= tip[0] <= 0.36 and Y_BRIDGE - 0.03 <= tip[1] <= Y_NUT + 0.02 and tip[2] < 0.0535:
            _fail(f"{side} gripper pressed the guitar")
    # plucks: a gripper's pad tip crosses a string's line, in the pluck zone, in the height band, fast enough
    for side, pts in grips.items():
        tip = np.asarray(pts[0], float) - np.array([0.0, 0.0, TOP])
        prev = S["prev"].get(side)
        S["prev"][side] = tip
        if prev is None:
            continue
        dx = tip[0] - prev[0]
        if abs(dx) * CONTROL_HZ < PLUCK_SPEED:
            continue
        z = 0.5 * (tip[2] + prev[2])
        y = 0.5 * (tip[1] + prev[1])
        if not (PLUCK_Y[0] <= y <= PLUCK_Y[1] and PLUCK_Z[0] <= z <= PLUCK_Z[1]):
            continue
        for k, xs in enumerate(STRING_X):
            if (prev[0] - xs) * (tip[0] - xs) < 0 or tip[0] == xs:
                S["plucks"].append((n, side, k + 1))
                if k > 0:
                    _fail(f"string {k + 1} plucked")
                    S["events"].append((n, f"pluck string {k + 1}"))
                else:
                    yc = ct["y"] if ct["on"] else None
                    semis, name = pitch_of(yc)
                    S["ring"] = dict(start=n, name=name, fretted=yc is not None, side=side)
                    S["events"].append((n, f"pluck string 1 {side} pitch {name} ({semis:.2f} semitones, "
                                           f"{'bar y %.1f mm' % (1000 * yc) if yc is not None else 'open'})"))
    # a ringing note: a fretted one ends if the bar leaves the string inside the half second
    r = S["ring"]
    if r is not None:
        if r["fretted"] and not ct["on"]:
            S["events"].append((n, f"bar lifted: {r['name']} cut short"))
            S["ring"] = None
        elif n - r["start"] >= RING_STEPS:
            S["notes_heard"].append(r["name"])
            S["events"].append((n, f"note {r['name']} rang"))
            expected = EXPECTED[len(S["played"])] if len(S["played"]) < len(EXPECTED) else None
            if r["name"] == expected:
                S["played"].append(r["name"])
            else:
                _fail(f"wrong note {r['name']} (wanted {expected})")
            S["ring"] = None
    S["recent"].append(bc.copy())
    del S["recent"][:-STILL_STEPS]


def _bar_still() -> bool:
    rec = _STATE["recent"]
    return len(rec) == STILL_STEPS and all(np.linalg.norm(p - rec[-1]) < 0.001 for p in rec)


_GRIPS: dict = {}


def _gripper_points(environment) -> dict:
    from yam_arena import ik, station

    out = {}
    for side in ("left", "right"):
        rob = environment.unwrapped.scene[f"{side}_robot"]
        q = np.asarray(getattr(rob.data.joint_pos, "torch", rob.data.joint_pos)[0].detach().cpu(), float)[:6]
        tcp, rot = ik.tcp_pose(q, station.BASES[side])
        a = rot[:, 2]
        out[side] = [tcp + 0.013 * a, tcp, tcp - 0.04 * a, tcp - 0.08 * a]
    return out


@on_step
def _music(environment) -> None:
    from partkit import _part_state

    _GRIPS.clear()
    _GRIPS.update(_gripper_points(environment))
    tick(_part_state(environment, BAR), _part_state(environment, GUITAR), _GRIPS)


def _done(bar: PartState) -> bool:
    return bool(len(_STATE["played"]) == len(EXPECTED) and _STATE["failed"] is None and bar_on_rest(bar)
                and _GRIPS and grippers_clear(bar, _GRIPS))


def success(objects) -> bool:
    """The three notes G4, A4, B4 rang in order and nothing else sounded, the bar lies on its rest, both grippers are at
    least 3 cm from the bar and the strings, and the bar has not moved more than 1 mm for 1 s."""
    return bool(_done(objects["bar"]) and _bar_still())


def distance(objects) -> float:
    """Metres from the bar to its rest."""
    c = np.asarray(objects["bar"].pos, float) - np.array([0.0, 0.0, TOP])
    return float(np.linalg.norm(c - np.array([REST_XY[0], REST_XY[1], BAR_CENTRE_Z])))


def score(objects) -> float:
    """0 if anything wrong sounded (a note that is not the next one on the card, the open string, an out-of-tune note or
    another string), the bar fell, a gripper pressed the guitar or the guitar moved 1 cm, or if no note rang. Otherwise
    0.25 once the first note rang, 0.5 after the second, 0.75 after all three; 1.0 once the bar is also on its rest with both
    grippers 3 cm clear and the bar still for 1 s."""
    if _STATE["failed"]:
        return 0.0
    k = len(_STATE["played"])
    if k == len(EXPECTED):
        return 1.0 if success(objects) else 0.75
    return 0.25 * k
