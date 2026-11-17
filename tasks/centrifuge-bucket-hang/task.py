"""On the bench: a laboratory centrifuge (Eppendorf Centrifuge 5702, 320 x 395 x 241 mm, 20 kg) with its lid open and
its four-place swing-bucket rotor A-4-38 in the chamber. The rotor turns freely on its drive when stopped. Two round
buckets (85 ml, about 190 g each) already hang opposite each other, in the left and right places; the front and back
places are empty. Two cleaned buckets stand on drying mats beside the centrifuge. Hang them so that every place holds a
bucket that hangs on both pegs of its place and swings freely.

How the real parts work (Eppendorf's product photos of rotor 5702 740.004 and bucket 5702 722.006): each of the rotor's
four lobes ends in two steel pegs, so every place (the bay between two lobes) has two pegs facing each other on one
axis. A bucket has a wider band below its neck with two pockets on opposite sides, open downward with a flared mouth.
Lowered through the bay, the pegs enter the pockets from below and the bucket hangs from the pocket tops as a pendulum.
Lowered out of line it rides on a peg by its band instead, which is the state the operating manual warns about
("Check that all buckets are fully attached ... and can swing out freely").

Published sizes: centrifuge 320 x 395 x 241 mm and 20 kg; bucket about 190 g (weight class embossed on its side);
13.5 cm centrifuging radius with the 85 ml adapter. Peg, pocket, bay and chamber dimensions are scaled from the
photos and that radius: measure the bought parts.

Frames. The centrifuge part's origin is on the rotor axis at bench level. Rotor geometry is written in mm in the rotor
frame (origin on the axis at the peg axis' height, z up), bucket geometry in mm in the bucket frame (origin on the
bucket's axis at the height of its pocket seats, x through both pockets, z up).

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
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = ("Get the centrifuge rotor ready for a run: every place fitted with a bucket that hangs on both pegs of "
               "its place and swings freely.")

HOUSING_C = (0.84, 0.85, 0.86)
LID_C = (0.62, 0.68, 0.74)
STEEL = (0.74, 0.75, 0.77)
ALU = (0.80, 0.81, 0.83)
BLACK = (0.07, 0.07, 0.08)
DARK = (0.20, 0.21, 0.23)
MAT_C = (0.16, 0.30, 0.42)
MM = 0.001


# -- small geometry helpers (quaternions are (w, x, y, z)) ----------------------------------------------------------------
def q_axis(axis: str, a: float):
    s, c = math.sin(a / 2), math.cos(a / 2)
    return (c, s, 0.0, 0.0) if axis == "x" else (c, 0.0, s, 0.0) if axis == "y" else (c, 0.0, 0.0, s)


def q_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def box(cx, cy, cz, sx, sy, sz, color, quat=None):
    """A box in mm (centre, size), as a partkit box in metres."""
    spec = ((cx * MM, cy * MM, cz * MM), (sx * MM, sy * MM, sz * MM), color)
    return spec + (tuple(float(v) for v in quat),) if quat is not None else spec


def turn(spec, a: float):
    """A partkit box turned by ``a`` about the frame's z axis."""
    (x, y, z), size, color = spec[:3]
    c, s = math.cos(a), math.sin(a)
    quat = q_mul(q_axis("z", a), spec[3] if len(spec) > 3 else (1.0, 0.0, 0.0, 0.0))
    return ((c * x - s * y, s * x + c * y, z), size, color, tuple(float(v) for v in quat))


def lift(spec, dz: float):
    """A partkit box moved up by ``dz`` metres."""
    (x, y, z), rest = spec[0], spec[1:]
    return ((x, y, z + dz),) + tuple(rest)


def ring(r_out, wall, z0, z1, color, ks=range(12), n=12):
    """Part of an ``n``-sided ring about z: boxes whose outer faces lie ``r_out`` from the axis (mm)."""
    w = 2 * r_out * math.tan(math.pi / n) * 1.02
    out = []
    for k in ks:
        a = 2 * math.pi * k / n
        rm = r_out - wall / 2
        out.append(box(rm * math.cos(a), rm * math.sin(a), (z0 + z1) / 2, wall, w, z1 - z0, color, q_axis("z", a)))
    return out


def prism(cx, cy, cz, length, across, color, axis="x", n=8):
    """A 2n-sided prism (``n`` boxes, ``across`` over the flats) along x or y: PhysX cylinders miss edge contacts."""
    side = across * math.tan(math.pi / (2 * n)) * 1.02
    out = []
    for k in range(n):
        a = math.pi * k / n
        if axis == "x":
            out.append(box(cx, cy, cz, length, side, across, color, q_axis("x", a)))
        else:
            out.append(box(cx, cy, cz, side, length, across, color, q_axis("y", a)))
    return out


def disc(r, z0, z1, color, n=6):
    """A solid 2n-sided prism about z, ``r`` to the flats (mm)."""
    side = 2 * r * math.tan(math.pi / (2 * n)) * 1.02
    return [box(0.0, 0.0, (z0 + z1) / 2, 2 * r, side, z1 - z0, color, q_axis("z", math.pi * k / n)) for k in range(n)]


# -- dimensions (mm) ---------------------------------------------------------------------------------------------------
PEG_R = 75.0            # the peg axes' distance from the rotor axis (estimate from the 13.5 cm centrifuging radius)
PEG_D = 8.0             # peg diameter (estimate)
BAY_HALF = 31.0         # a bay is 62 mm wide between the lobes' end faces (estimate)
PEG_TIP = 24.0          # the pegs stand 7 mm proud of the end faces: 48 mm between their tips
LOBE_TOP = 7.0          # lobe rims: from 22 mm below the peg axis to 7 mm above it
LOBE_BOTTOM = -22.0
HUB_TOP = -6.0          # the hub and the lobes' webs are dished 13 mm below the rims
ROTOR_R = 95.0
BODY_R = 23.0           # bucket body: 46 mm across, 1 mm clear of each peg tip
BAND_R = 28.0           # bucket band: 56 mm across, 3 mm clear of each lobe end face
NECK_R = 26.0           # threaded neck: 52 mm across, 42 mm bore
BODY_BOTTOM = -67.0     # bucket: 104 mm tall, pocket seats 67 mm above its bottom
BAND_BOTTOM, BAND_TOP, NECK_TOP = -8.0, 9.0, 37.0
SLOT = 9.5              # pocket seat width: 0.75 mm play a side on the 8 mm peg
SEAT_TOP = PEG_D / 2    # the pocket's top rests on the peg: the bucket origin is then on the peg axis
MOUTH = 6.0             # the pocket flares at 45 degrees over its lowest 6 mm (to 21.5 mm wide)
PEG_REACH = 26.0        # where a peg sits in a seated bucket's frame, along the bucket's x

CHAMBER_R = 148.0       # chamber 296 mm across
DECK = 200.0            # chamber rim above the bench
FLOOR = 60.0            # chamber floor above the bench
PIN_Z = 150.0           # peg axis above the bench
LID_OPEN = math.radians(85.0)
ROTOR_FRICTION = 0.01   # N m: a stopped drive free-wheels (estimate; about 0.1 N on a spring scale at the rotor's rim)
ROTOR_MASS = 0.9        # anodised aluminium rotor (estimate)
BUCKET_MASS = 0.19      # weight class "90": 190 g

# -- the rotor: hub, four dished lobes with thick rims, eight pegs, nut and shaft (rotor frame) --------------------------
_lobe = [
    box(47.5, 47.5, (LOBE_BOTTOM + HUB_TOP) / 2, 33.0, 33.0, HUB_TOP - LOBE_BOTTOM, BLACK),          # web
    box(73.0, BAY_HALF + 6.0, (LOBE_BOTTOM + LOBE_TOP) / 2, 26.0, 12.0, LOBE_TOP - LOBE_BOTTOM, BLACK),  # end, +x bay
    box(BAY_HALF + 6.0, 73.0, (LOBE_BOTTOM + LOBE_TOP) / 2, 12.0, 26.0, LOBE_TOP - LOBE_BOTTOM, BLACK),  # end, +y bay
]
for _a in (33.0, 45.0, 57.0):                                                                          # the curved rim
    _r = math.radians(_a)
    _lobe.append(box(86.0 * math.cos(_r), 86.0 * math.sin(_r), (LOBE_BOTTOM + LOBE_TOP) / 2, 18.0, 24.0,
                     LOBE_TOP - LOBE_BOTTOM, BLACK, q_axis("z", _r)))
_pegs = (prism(PEG_R, (PEG_TIP + BAY_HALF + 3.0) / 2, 0.0, BAY_HALF + 3.0 - PEG_TIP, PEG_D, STEEL, axis="y")
         + prism((PEG_TIP + BAY_HALF + 3.0) / 2, PEG_R, 0.0, BAY_HALF + 3.0 - PEG_TIP, PEG_D, STEEL, axis="x"))
ROTOR_SHAPES = [box(0.0, 0.0, (LOBE_BOTTOM + HUB_TOP) / 2, 72.0, 72.0, HUB_TOP - LOBE_BOTTOM, BLACK)]   # hub: clear of a
#                                                                             bucket's neck when it swings out level
ROTOR_SHAPES += [box(0.0, 0.0, 0.0, 22.0, 22.0, 12.0, STEEL, q_axis("z", a)) for a in (0.0, math.pi / 4)]   # rotor nut
ROTOR_SHAPES += [box(0.0, 0.0, -53.0, 20.0, 20.0, 62.0, DARK, q_axis("z", a)) for a in (0.0, math.pi / 4)]  # drive shaft
for _k in range(4):
    ROTOR_SHAPES += [turn(s, _k * math.pi / 2) for s in _lobe + _pegs]


# -- a bucket (bucket frame) ---------------------------------------------------------------------------------------------
def bucket_shapes():
    out = ring(BODY_R, 3.0, BODY_BOTTOM + 4.0, BAND_TOP, ALU)                         # body wall (the pockets' floor)
    out += disc(BODY_R - 3.5, BODY_BOTTOM, BODY_BOTTOM + 4.0, ALU)                    # floor
    w = 2 * BODY_R * math.tan(math.pi / 12) * 1.02                                   # 4 mm x 45 degree bottom chamfer
    for k in range(12):
        a = 2 * math.pi * k / 12
        rm, zc = BODY_R - 2.0 - 1.06, BODY_BOTTOM + 2.0 + 1.06
        out.append(box(rm * math.cos(a), rm * math.sin(a), zc, 3.0, w, 6.2, ALU,
                       q_mul(q_axis("z", a), q_axis("y", math.pi / 4))))
    out += ring(BAND_R, 6.0, BAND_BOTTOM, BAND_TOP, ALU, ks=(2, 3, 4, 8, 9, 10))      # band away from the pockets
    out += ring(BAND_R, 6.0, BAND_BOTTOM + MOUTH, BAND_TOP, ALU, ks=(1, 5, 7, 11))    # band beside the pockets' mouths
    flat = BAND_R * math.tan(math.pi / 12) * 1.02                                    # half the band's flat width
    for s in (1.0, -1.0):                                                            # the two pockets, at +-x
        xc = s * (BAND_R - 3.0)
        out.append(box(xc, 0.0, (SEAT_TOP + BAND_TOP) / 2, 6.0, 2 * flat, BAND_TOP - SEAT_TOP, ALU))   # seat top
        for t in (1.0, -1.0):
            out.append(box(xc, t * (SLOT / 2 + flat) / 2, (BAND_BOTTOM + MOUTH + SEAT_TOP) / 2, 6.0, flat - SLOT / 2,
                           SEAT_TOP - BAND_BOTTOM - MOUTH, ALU))                                         # jaws
            out.append(box(xc, t * (SLOT / 2 + MOUTH / 2 + 1.06), BAND_BOTTOM + MOUTH / 2 + 1.06, 6.0,
                           MOUTH * math.sqrt(2.0), 3.0, ALU, q_axis("x", -t * math.pi / 4)))            # flared mouth
    out += ring(NECK_R, 5.0, BAND_TOP, NECK_TOP, ALU)                                 # threaded neck
    return out


# -- the centrifuge: housing with its chamber, open lid, and the rotor on a free joint (centrifuge frame, metres) ---------
def _housing():
    b = []
    for sx in (-0.185, 0.140):                                                       # feet
        for sy in (-0.125, 0.125):
            b.append(((sx, sy, 0.005), (0.035, 0.035, 0.010), DARK))
    b.append(((-0.0225, 0.0, 0.035), (0.395, 0.314, 0.050), HOUSING_C))             # base under the chamber
    b += [((0.0, 0.0, 0.061), (0.205, 0.205, 0.002), STEEL, q_axis("z", a)) for a in (0.0, math.pi / 4)]  # bowl floor
    b += [lift(s, 0.0) for s in ring(CHAMBER_R + 9.0, 9.0, FLOOR, DECK, STEEL, ks=range(24), n=24)]       # bowl wall
    for sy in (-1.0, 1.0):
        b.append(((0.0125, sy * 0.1585, 0.105), (0.325, 0.003, 0.190), HOUSING_C))  # side skins
        b.append(((-0.185, sy * 0.1585, 0.070), (0.070, 0.003, 0.120), HOUSING_C))
        for sx in (-1.0, 1.0):
            b.append(((sx * 0.1275, sy * 0.131, 0.130), (0.045, 0.052, 0.140), HOUSING_C))   # corners round the bowl
    b.append(((0.1625, 0.0, 0.130), (0.025, 0.314, 0.140), HOUSING_C))              # rear strip (lid hinge)
    b.append(((-0.185, 0.0, 0.095), (0.070, 0.314, 0.070), HOUSING_C))              # front block
    b.append(((-0.1675, 0.0, 0.1475), (0.035, 0.314, 0.035), HOUSING_C))
    slope = q_axis("y", -math.pi / 4)                                                # sloped control panel
    b.append(((-0.1829, 0.0, 0.1629), (0.099, 0.314, 0.006), HOUSING_C, slope))
    b.append(((-0.1864, 0.0, 0.1664), (0.030, 0.090, 0.004), BLACK, slope))         # display
    for sy in (-0.090, 0.090):
        b.append(((-0.1885, sy, 0.1685), (0.028, 0.028, 0.012), DARK, slope))       # time and speed dials
    hinge = np.array([0.160, 0.0, DECK * MM])                                        # the lid, open on its rear hinge
    along = np.array([-math.cos(LID_OPEN), 0.0, math.sin(LID_OPEN)])
    outward = np.array([math.sin(LID_OPEN), 0.0, math.cos(LID_OPEN)])
    centre = hinge + 0.165 * along + 0.0205 * outward
    b.append((tuple(float(v) for v in centre), (0.330, 0.320, 0.041), LID_C, q_axis("y", LID_OPEN)))
    return b


XC = 0.21                                          # the rotor axis, world x (the housing's front face is at x -0.01)
CENTRIFUGE_POS = (XC, 0.0, TOP)
ROTOR = Hinge("rotor", pivot=(0.0, 0.0, PIN_Z * MM), boxes=tuple(lift(s, PIN_Z * MM) for s in ROTOR_SHAPES),
              lower=-math.inf, upper=math.inf, stiffness=0.0, damping=2.0, effort=ROTOR_FRICTION, mass=ROTOR_MASS,
              axis="Z")
CENTRIFUGE = part("centrifuge", pos=CENTRIFUGE_POS, mass=20.0, boxes=_housing(), hinges=[ROTOR], color=HOUSING_C,
                  friction=0.6, fixed=True)

# -- places and start poses -----------------------------------------------------------------------------------------------
# Place k is the bay at angle k * 90 degrees from the rotor's +x (rotor angle 0: place 0 at the back, 1 on the robot's
# left, 2 at the front, 3 on its right).
PEG_Z = TOP + PIN_Z * MM


def bay_world(k: int, rotor_angle: float = 0.0) -> np.ndarray:
    a = rotor_angle + k * math.pi / 2
    return np.array([XC + PEG_R * MM * math.cos(a), PEG_R * MM * math.sin(a), PEG_Z])


MAT_T = 0.003
MAT_X, MAT_Y = 0.24, 0.245
RACK_YAW = math.radians(60.0)                      # the loose buckets' pockets are turned this much from the world x axis
_shapes = bucket_shapes()
BUCKET_REST_Z = TOP + MAT_T - BODY_BOTTOM * MM + 0.0005
HUNG = {"bucket_a": 1, "bucket_b": 3}              # start: hanging in the left and right places
BUCKET_A = part("bucket_a", pos=tuple(float(v) for v in bay_world(1) + [0, 0, 0.0005]), mass=BUCKET_MASS,
                boxes=_shapes, color=ALU, friction=0.5)
BUCKET_B = part("bucket_b", pos=tuple(float(v) for v in bay_world(3) + [0, 0, 0.0005]), mass=BUCKET_MASS,
                boxes=_shapes, color=ALU, friction=0.5)
# Negative-variant check only (demo.py sets it for DEMO_VARIANT=unaligned): bucket 1 starts a quarter turn round on its
# mat, so the same motions lower it with its pockets across the pegs.
_yaw_1 = -RACK_YAW + (math.pi / 2 if os.environ.get("CENTRIFUGE_BUCKET_QUARTER_TURN") == "1" else 0.0)
BUCKET_1 = part("bucket_1", pos=(MAT_X, -MAT_Y, BUCKET_REST_Z), yaw=_yaw_1, mass=BUCKET_MASS, boxes=_shapes,
                color=ALU, friction=0.5)           # on the mat to the robot's right
BUCKET_2 = part("bucket_2", pos=(MAT_X, MAT_Y, BUCKET_REST_Z), yaw=RACK_YAW, mass=BUCKET_MASS, boxes=_shapes,
                color=ALU, friction=0.5)           # on the mat to the robot's left
MAT_R = Box("mat_right", pos=(MAT_X, -MAT_Y, TOP + MAT_T / 2), size=(0.14, 0.14, MAT_T), color=MAT_C)
MAT_L = Box("mat_left", pos=(MAT_X, MAT_Y, TOP + MAT_T / 2), size=(0.14, 0.14, MAT_T), color=MAT_C)
BUCKETS = ("bucket_a", "bucket_b", "bucket_1", "bucket_2")
LOOSE = ("bucket_1", "bucket_2")
OBJECTS = [MAT_R, MAT_L, CENTRIFUGE, BUCKET_A, BUCKET_B, BUCKET_1, BUCKET_2]

# -- scoring ---------------------------------------------------------------------------------------------------------------
SEAT_TOL = 0.0015        # a peg within 1.5 mm of its pocket's seat, up-down and across the pocket
REACH_TOL = 0.003        # and within 3 mm of its place along the peg axis
PLUMB_TOL = math.radians(3.0)
RELEASE = 0.030          # grippers 3 cm clear of a bucket
DROP = 0.015             # a bucket's seat line 15 mm or more below the pegs, inside the chamber: dropped
MOVE = 0.010             # centrifuge moved more than 10 mm
CLEAR_ABOVE = 0.020      # grippers clear of the chamber: outside it, or 20 mm or more above its rim
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), home=None, moved=False, dropped=False,
                  released={n: False for n in BUCKETS}, clear=False, finger_check=None, recent=[], report={})


_reset_state()


def rotor_angle(objects) -> float:
    return float(objects["centrifuge"].hinges.get("rotor", 0.0))


def pegs(objects) -> list:
    """World positions of each place's two pegs (where they sit in a seated bucket's band), places 0..3."""
    c = objects["centrifuge"]
    th = rotor_angle(objects)
    out = []
    for k in range(4):
        a = th + k * math.pi / 2
        centre = np.array([PEG_R * MM * math.cos(a), PEG_R * MM * math.sin(a), PIN_Z * MM])
        t = np.array([-math.sin(a), math.cos(a), 0.0]) * PEG_REACH * MM
        out.append((c.to_world(centre + t), c.to_world(centre - t)))
    return out


def _peg_seated(bucket: PartState, peg_world) -> tuple[bool, float]:
    """Is this peg in one of the bucket's pocket seats, and on which side of the bucket (sign of x)?"""
    p = bucket.to_local(peg_world)
    ok = abs(p[1]) <= SEAT_TOL and abs(p[2]) <= SEAT_TOL and abs(abs(p[0]) - PEG_REACH * MM) <= REACH_TOL
    return bool(ok), float(np.sign(p[0]))


def hang(bucket: PartState, peg_list) -> tuple[int, int]:
    """(the place whose two pegs both sit in the bucket's pockets, or -1; the most pegs of any one place seated)."""
    place, most = -1, 0
    for k, (pa, pb) in enumerate(peg_list):
        (a_ok, a_side), (b_ok, b_side) = _peg_seated(bucket, pa), _peg_seated(bucket, pb)
        if a_ok and b_ok and a_side * b_side < 0:
            place, most = k, 2
        elif a_ok or b_ok:
            most = max(most, 1)
    return place, most


def tilt(bucket: PartState) -> float:
    """The bucket axis' angle from plumb (rad)."""
    return float(math.acos(max(-1.0, min(1.0, bucket.rotation()[2, 2]))))


def in_chamber(objects, point_world, above: float = 0.05) -> bool:
    p = objects["centrifuge"].to_local(point_world)
    return bool(math.hypot(p[0], p[1]) < CHAMBER_R * MM and p[2] < DECK * MM + above)


def status(objects) -> dict:
    """Each bucket's place, pegs seated, tilt and whether it is inside the chamber or was dropped in it."""
    peg_list = pegs(objects)
    c = objects["centrifuge"]
    out = {}
    for n in BUCKETS:
        b = objects[n]
        place, most = hang(b, peg_list)
        local = c.to_local(b.pos)
        inside = in_chamber(objects, b.pos)
        out[n] = {"place": place, "pegs": most, "tilt": tilt(b), "inside": inside,
                  "low": bool(inside and local[2] < (PIN_Z * MM - DROP))}
    return out


def _axis_distance(bucket: PartState, point_world) -> float:
    """Distance from a world point to the bucket's axis between its bottom and its rim."""
    p = bucket.to_local(point_world)
    z = min(max(p[2], BODY_BOTTOM * MM), NECK_TOP * MM)
    return float(math.sqrt(p[0] ** 2 + p[1] ** 2 + (p[2] - z) ** 2))


@on_step
def _step(environment) -> None:
    """Every control step: count steps and check the grippers (score() cannot see the robot)."""
    from yam_arena.scene import object_states

    _STATE["step"] += 1
    try:
        st = object_states(environment, ["centrifuge", *BUCKETS])
        pts = finger_points(environment)
        for n in BUCKETS:
            _STATE["released"][n] = bool(min(_axis_distance(st[n], p) for p in pts) >= NECK_R * MM + RELEASE)
        _STATE["clear"] = not any(in_chamber(st, p, above=CLEAR_ABOVE) for p in pts)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"centrifuge-bucket-hang: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"] = False
        _STATE["released"] = {n: False for n in BUCKETS}
        _STATE["clear"] = False


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    c = objects["centrifuge"]
    if _STATE["home"] is None:
        _STATE["home"] = c.pos.copy()
    _STATE["moved"] |= float(np.linalg.norm(c.pos - _STATE["home"])) > MOVE                      # permanent
    st = status(objects)
    _STATE["dropped"] |= any(s["low"] for s in st.values())                                       # permanent
    released = _STATE["released"]
    hung = {n: st[n]["place"] >= 0 and released[n] for n in BUCKETS}
    # a bucket left on the rotor without both pegs in its pockets (on one peg, or riding on the pegs by its band)
    adrift = any(st[n]["inside"] and st[n]["place"] < 0 and released[n] for n in BUCKETS)
    th = rotor_angle(objects)
    recent = _STATE["recent"]
    recent.append(np.concatenate([objects[n].pos for n in BUCKETS]
                                 + [np.array([math.cos(th), math.sin(th)]) * PEG_R * MM]))
    del recent[:-20]
    still = len(recent) == 20 and all(float(np.abs(r - recent[-1]).max()) < 0.001 for r in recent)
    places = sorted(st[n]["place"] for n in BUCKETS)
    all_hung = all(hung.values()) and places == [0, 1, 2, 3]
    plumb = all(st[n]["tilt"] <= PLUMB_TOL for n in BUCKETS)
    loose_hung = sum(1 for n in LOOSE if hung[n])
    if _STATE["moved"] or _STATE["dropped"] or adrift:
        result = (0.0, False)
    elif all_hung and plumb and _STATE["clear"] and still:
        result = (1.0, True)
    elif all_hung and plumb and _STATE["clear"]:
        result = (0.75, False)
    elif loose_hung == 2:
        result = (0.5, False)
    elif loose_hung == 1:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _STATE["report"] = {"hung": hung, "adrift": adrift, "plumb": plumb, "still": still, "places": places,
                        "rotor_deg": math.degrees(th)}
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """All four places hold a bucket hanging on both pegs within 3 degrees of plumb, both grippers clear of the
    chamber, and buckets and rotor still for the last second."""
    return _update(objects)[1]


def distance(objects) -> float:
    """Sum over the two loose buckets of the distance from their seat line to the nearest place's peg axis."""
    peg_list = pegs(objects)
    centres = [(a + b) / 2 for a, b in peg_list]
    return float(sum(min(np.linalg.norm(objects[n].pos - c) for c in centres) for n in LOOSE))


def score(objects) -> float:
    """End state. 0.25: one of the two loose buckets hangs on both pegs of a place (each peg within 1.5 mm of its
    pocket's seat) with the grippers 3 cm clear of it; 0.5: both do; 0.75: all four places hold a bucket hanging on
    both pegs, each within 3 degrees of plumb, with both grippers out of the chamber; 1.0: that state with buckets and
    rotor moving less than 1 mm for 1 second. 0 while any bucket is left on the rotor without both pegs in its pockets
    (grippers clear of it); 0 for good if a bucket was dropped into the chamber (its seats 15 mm or more below the
    pegs) or the centrifuge was moved more than 10 mm."""
    return _update(objects)[0]
