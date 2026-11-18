"""On the station bench: a Stanley Bailey No. 4 smoothing plane (12-904, 245 mm, 50 mm cutter, 1.9 kg) standing on its
sole with its iron out for honing, toe toward the robot, so the frog's face looks up at the robot. Beside it, on two
battens, lie the honed iron with its cap iron screwed on (0.27 kg) and the lever cap with its cam lever raised
(0.19 kg).

The job: put the plane back together. The iron assembly goes bevel down on the 45 degree frog over the standing
lever-cap screw and is slid down until the depth adjuster's yoke stands in the cap iron's window and the lateral lever's
boss in the iron's slot; the lever cap goes on with its keyhole over the screw head, is slid down under the head, and
its cam lever is pressed flat.

Nothing is scripted. Frog face, screw, yoke tab and cheeks are fixed shapes on the loose plane body; the lateral lever
is a free hinge; the iron assembly is one rigid part with the real slot, window and hump; the lever cap carries a cam
lever on a hinge whose eccentric pushes a spring plate (a follower and a foot joined by a preloaded spring) onto the cap
iron, so the cap pivots on the screw head and clamps. Steel on steel has friction 0.3. The iron's back on the frog's
face, and the cap iron on the yoke's tab, have 2.0: more than a real frog gives (about 0.4), so that an iron set down on
the 45 degree frog stays where it is put and engages the two adjusters only when it is slid down (see README.md).

Sourced sizes: plane length, cutter width and mass (Stanley data sheet); iron 2 in x 7-1/2 in x 3/32 in (Stanley
12-313); iron slot 0.450 in; lateral disc 7/16 in; yoke 9/64 in proud of the frog; lever cap 2 in x 4-5/16 in. Every
other size here is an estimate (measure the bought plane): see README.md.

Frames (mm in the constants, m in the shapes):
  plane   origin on the bench where the frog's face, carried down, meets it; x toward the toe, y left, z up
  bed     on the frog's face: l across, p up the slope from that line, h off the face
  iron    x across, y from the cutting edge to the top end, z from the back (bevel side) to the cap iron side
  cap     x across, y from the toe to the cam end, z up from the toe's underside

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Hinge, PartState, Slider, finger_points, on_reset, on_step, part, set_slider_target  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = ("Put the plane back together: iron and cap iron on the frog with both adjusters engaged, and the "
               "lever cap locked.")

MM = 0.001
C45 = math.sqrt(0.5)
IRON_GREY = (0.24, 0.25, 0.27)
STEEL = (0.74, 0.75, 0.77)
BRIGHT = (0.86, 0.87, 0.88)
BRASS = (0.80, 0.64, 0.24)
WOOD = (0.42, 0.24, 0.12)
PINE = (0.85, 0.72, 0.50)
CHROME = (0.80, 0.81, 0.84)
RED = (0.70, 0.10, 0.10)

# -- sizes (mm) ----------------------------------------------------------------------------------------------------------
PLANE_X = 0.220                    # the plane frame's origin on the bench (m): the mouth
PLANE_YAW = math.pi                # the plane's toe points at the robot
SOLE = (-178.0, 67.0)              # heel and toe (245 mm long)
SOLE_HALF_W = 30.15                # 60.3 mm wide (estimate)
SOLE_T = 4.0
CHEEK_IN = 26.65                   # 53.3 mm between the cheeks (estimate)
CHEEK = ((-178.0, 20.0), (-125.0, 20.0), (-85.0, 46.0), (15.0, 46.0), (45.0, 22.0), (67.0, 22.0))   # top edge (x, z), estimate
KNOB_X = 37.0
FROG_HALF_W = 24.0
FROG_P = (8.0, 131.0)              # the frog face along the slope
CHANNEL_HALF_W, CHANNEL_END, CHANNEL_DEPTH = 10.0, 55.0, 4.5    # the channel the cap iron screw's head rides in
SCREW_P = 60.0                     # lever-cap screw
SHANK, HEAD, HEAD_UNDER, HEAD_T = 7.1, 10.5, 9.6, 2.2           # 9/32 in shank; head across, its underside off the face
TAB_HALF_W, TAB_P, TAB_H = 4.0, (97.3, 101.3), 3.6              # yoke tab: 5/16 in wide (estimate), 9/64 in proud
TAB_TOP = 2.5                                                   # half its width at the tip
LAT_PIVOT_P, LAT_ARM, BOSS = 127.0, 17.0, 11.11                 # lateral lever pivot, boss below it, boss across (7/16 in)
LAT_LIMIT = math.radians(8.0)

IRON_HALF_W, IRON_L, IRON_T = 25.4, 190.5, 2.4                  # 2 in x 7-1/2 in x 3/32 in
SLOT_HALF_W, SLOT_Y = 5.715, (30.0, 128.0)                      # 0.450 in slot
BEVEL = 5.15                                                    # 25 degree bevel on a 2.4 mm iron
CAPIRON_T, CAPIRON_Y = 2.0, (1.5, 115.8)
CI_SLOT_HALF_W, CI_SLOT_Y = 6.35, (48.0, 86.0)
WINDOW_Y = (96.3, 101.1)                                        # 3/16 in window, 3-11/16 in behind the cap iron's front
CAPSCREW_Y, CAPSCREW_HEAD = 38.0, 16.5
ENGAGED_P = 0.6                    # where the iron's edge end sits on the slope with the yoke in the window:
#                                    the cutting edge is then 2.1 mm above the bench, inside the mouth

CAP_HALF_W, CAP_L = 25.0, 110.0
CAP_PLATE_Z = (1.5, 5.0)
LOBE_HALF_W, LOBE_Y = 8.0, (24.0, 40.0)                         # keyhole: the wide end
SEAT_HALF_W, SEAT_END = 4.0, 48.0                               # and the narrow end the screw seats in
CAM_Y, CAM_Z = 94.0, 12.5                                       # cam pivot
CAM_E, CAM_R = 3.229, 5.582                                     # eccentric offset and radius
CAM_OPEN = math.radians(110.0)                                  # the lever's swing from raised to its closed stop
CAM_PAST = math.radians(15.0)                                   # the eccentric's high point is this far before the stop
LEVER_LEN = 38.0
LEAF_TRAVEL = 3.8
CLAMP_N = 100.0                                                 # the spring plate's preload (N): the clamp's force
# PhysX carries that force through the cam, the follower and the foot only if those pieces are not too light and the
# lever cap gets more solver passes than the default 32 (with 8 to 20 g pieces and 32 passes the foot sank 0.7 mm into
# the cap iron and the cam closed at 5 N whatever the preload); the pieces still add up to the cap's 0.19 kg
CAP_ITERATIONS = 128
CAM_MASS, LEAF_MASS, FOOT_MASS = 0.04, 0.03, 0.03
FOOT_RATE, FOOT_PRELOAD, FOOT_TRAVEL = 2500.0, CLAMP_N / 2500.0, 0.003     # N/m, m of preload, m of travel
LEAF_RATE, LEAF_PRELOAD = 1000.0, 0.002                          # the plate's own return spring: 2 N up at rest
PLANE_MASS, IRON_MASS, CAP_MASS = 1.44, 0.27, 0.19
STEEL_FRICTION = 0.3
FROG_FRICTION = 2.0                                             # the iron's back on the frog's face, the cap iron's bridge on the yoke

# -- small geometry helpers ---------------------------------------------------------------------------------------------
R_BED = np.array([[0.0, -C45, C45], [1.0, 0.0, 0.0], [0.0, C45, C45]])    # columns: l, p and h in the plane frame


def quat(R) -> tuple:
    """(w, x, y, z) of a rotation matrix (Shepperd's method)."""
    R = np.asarray(R, float)
    t = [R[0, 0] + R[1, 1] + R[2, 2], R[0, 0], R[1, 1], R[2, 2]]
    i = int(np.argmax(t))
    if i == 0:
        q = [1 + t[0], R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    elif i == 1:
        q = [R[2, 1] - R[1, 2], 1 + 2 * R[0, 0] - t[0], R[0, 1] + R[1, 0], R[0, 2] + R[2, 0]]
    elif i == 2:
        q = [R[0, 2] - R[2, 0], R[0, 1] + R[1, 0], 1 + 2 * R[1, 1] - t[0], R[1, 2] + R[2, 1]]
    else:
        q = [R[1, 0] - R[0, 1], R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], 1 + 2 * R[2, 2] - t[0]]
    q = np.array(q) / np.linalg.norm(q)
    return tuple(float(v) for v in (q if q[0] >= 0 else -q))


def rot(axis: str, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return {"X": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "Y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "Z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis].astype(float)


EYE = np.eye(3)
Q_BED = quat(R_BED)


def obox(centre, size, color, R=EYE, base=EYE):
    """A box given in mm in a frame turned ``base`` from the part frame, itself turned ``R`` in that frame."""
    c = base @ np.asarray(centre, float) * MM
    return (tuple(float(v) for v in c), tuple(float(abs(v)) * MM for v in size), color, quat(base @ R))


def span(x0, x1, y0, y1, z0, z1, color, base=EYE):
    return obox(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color, EYE, base)


def prism(cx, cy, z0, z1, diameter, bars, color, base=EYE, axis="Z"):
    """A round of this ``diameter`` as a regular prism with 2 x ``bars`` sides whose corners lie on the circle (so it
    is never fatter than the real part), built from crossed bars about ``axis`` (z: centre (cx, cy), from z0 to z1;
    x: centre (y, z) = (cx, cy), from x = z0 to z1)."""
    across = diameter * math.cos(math.pi / (2 * bars))
    w = across * math.tan(math.pi / (2 * bars))
    out = []
    for k in range(bars):
        a = k * math.pi / bars
        if axis == "Z":
            out.append(obox((cx, cy, (z0 + z1) / 2), (across, w, z1 - z0), color, rot("Z", a), base))
        else:
            out.append(obox(((z0 + z1) / 2, cx, cy), (z1 - z0, across, w), color, rot("X", a), base))
    return out


def frustum(cx, cy, z0, z1, r0, r1, n, color, base=EYE, t=1.0):
    """The sloping side of a cone frustum (radius r0 at z0, r1 at z1) as ``n`` flat facets ``t`` thick, their edges on
    the cone."""
    a0, a1 = r0 * math.cos(math.pi / n), r1 * math.cos(math.pi / n)
    out, s = [], math.hypot(a0 - a1, z1 - z0)
    am = (a0 + a1) / 2
    for k in range(n):
        f = k * 2 * math.pi / n
        ex = np.array([-math.sin(f), math.cos(f), 0.0])
        ey = np.array([(a1 - a0) * math.cos(f), (a1 - a0) * math.sin(f), z1 - z0]) / s
        ez = np.cross(ex, ey)
        centre = np.array([cx + am * math.cos(f), cy + am * math.sin(f), (z0 + z1) / 2]) - ez * t / 2
        out.append(obox(centre, (2 * min(a0, a1) * math.tan(math.pi / n), s, t), color, np.column_stack([ex, ey, ez]), base))
    return out


def bed_to_plane(l, p, h) -> np.ndarray:   # noqa: E741
    """A point on the frog (mm) in the plane frame (m)."""
    return R_BED @ np.array([l, p, h], float) * MM


def frog_to_world(l, p, h) -> np.ndarray:   # noqa: E741
    """A point on the frog (mm) in the world (m), with the plane where it starts."""
    return np.array(PLANE_POS) + PLANE_ROT @ bed_to_plane(l, p, h)


# -- the plane -----------------------------------------------------------------------------------------------------------
def _plane_body():
    b = [span(SOLE[0], -4.3, -SOLE_HALF_W, SOLE_HALF_W, 0, SOLE_T, IRON_GREY),              # sole behind the mouth
         span(5.5, SOLE[1], -SOLE_HALF_W, SOLE_HALF_W, 0, SOLE_T, IRON_GREY)]               # and in front of it
    for s in (1, -1):
        y0, y1 = sorted((s * CHEEK_IN, s * SOLE_HALF_W))
        b.append(span(-4.3, 5.5, y0, y1, 0, SOLE_T, IRON_GREY))                             # mouth ends
        for (xa, za), (xb, zb) in zip(CHEEK, CHEEK[1:]):                                    # cheek, along its top edge
            b.append(span(xa, xb, y0, y1, SOLE_T, min(za, zb), IRON_GREY))
            if za != zb:                                                                    # a ramp: a slanted bar on top
                a = math.atan2(zb - za, xb - xa)
                R = rot("Y", -a)
                t = abs(zb - za) * math.cos(a)
                mid = np.array([(xa + xb) / 2, (y0 + y1) / 2, (za + zb) / 2]) - R[:, 2] * t / 2
                b.append(obox(mid, (math.hypot(xb - xa, zb - za), y1 - y0, t), IRON_GREY, R))
    n_iron = len(b)
    grip = []
    for z0, z1, across in ((SOLE_T, 22, 26), (22, 56, 38), (56, 60, 30)):                   # front knob
        grip += prism(KNOB_X, 0, z0, z1, across, 4, WOOD)
    grip.append(span(-176, -118, -12.5, 12.5, SOLE_T, 14, WOOD))                            # tote: foot,
    grip.append(obox((-152, 0, 38), (30, 25, 52), WOOD, rot("Y", math.radians(12))))        # grip leaning forward,
    grip.append(obox((-141, 0, 80), (24, 25, 46), WOOD, rot("Y", math.radians(28))))
    grip.append(obox((-146, 0, 104), (32, 25, 11), WOOD, rot("Y", math.radians(-10))))      # and horn
    b += grip
    for x0, x1, z in ((-40, -14, 6.5), (-70, -30, 22), (-95, -50, 42), (-100, -70, 58), (-104, -84, 74)):
        b.append(span(x0, x1, -22, 22, SOLE_T, z, IRON_GREY))                               # frog casting under the face
    b.append(span(-112, -98, -22, 22, 40, 86, IRON_GREY))
    b += prism(0, 46, -119, -113, 32, 4, BRASS, axis="X")                                   # depth adjuster wheel
    n_face = len(b)
    for s in (1, -1):                                                                       # frog face: two lands
        l0, l1 = sorted((s * CHANNEL_HALF_W, s * FROG_HALF_W))
        b.append(span(l0, l1, FROG_P[0], FROG_P[1], -8, 0, STEEL, R_BED))
    b.append(span(-CHANNEL_HALF_W, CHANNEL_HALF_W, CHANNEL_END, FROG_P[1], -8, 0, STEEL, R_BED))
    face = list(range(n_face, len(b)))                                                      # the three boxes the iron lies on
    b.append(span(-CHANNEL_HALF_W, CHANNEL_HALF_W, FROG_P[0], CHANNEL_END, -8, -CHANNEL_DEPTH, IRON_GREY, R_BED))
    # lever-cap screw: 16 sides, so its shank and head are under 2% thin at the flats in the keyhole's close fit
    b += prism(0, SCREW_P, 0, HEAD_UNDER, SHANK, 8, BRIGHT, R_BED)
    b += prism(0, SCREW_P, HEAD_UNDER, HEAD_UNDER + HEAD_T, HEAD, 8, BRIGHT, R_BED)
    b += prism(0, SCREW_P, HEAD_UNDER + HEAD_T, HEAD_UNDER + HEAD_T + 1.5, 7.5, 8, BRIGHT, R_BED)
    b += frustum(0, SCREW_P, HEAD_UNDER + HEAD_T, HEAD_UNDER + HEAD_T + 1.5, HEAD / 2, 3.75, 16, BRIGHT, R_BED)
    b.append(span(-TAB_TOP, TAB_TOP, TAB_P[0], TAB_P[1], 0, TAB_H, BRASS, R_BED))           # yoke tab, its tip rounded
    drop = TAB_HALF_W - TAB_TOP                                                             # across (45 degree flanks)
    for s in (1, -1):
        l0, l1 = sorted((s * TAB_TOP, s * TAB_HALF_W))
        b.append(span(l0, l1, TAB_P[0], TAB_P[1], 0, TAB_H - drop, BRASS, R_BED))
        R = rot("Y", s * math.pi / 4)
        mid = np.array([s * (TAB_TOP + drop / 2), sum(TAB_P) / 2, TAB_H - drop / 2]) - R[:, 2] * 0.5
        b.append(obox(mid, (drop * math.sqrt(2), TAB_P[1] - TAB_P[0], 1.0), BRASS, R, R_BED))
    return b, list(range(n_iron, n_iron + len(grip))), face


def _lateral_lever():
    boss_p = LAT_PIVOT_P - LAT_ARM
    b = [span(-4.5, 4.5, boss_p - 2, 192, -1.5, -0.1, CHROME, R_BED),
         span(-6, 6, 184, 192, -10, -0.1, CHROME, R_BED)]
    b += prism(0, boss_p, -0.1, 1.2, BOSS, 8, CHROME, R_BED)
    b += prism(0, boss_p, 1.2, 2.3, 8.4, 8, CHROME, R_BED)
    b += frustum(0, boss_p, 1.2, 2.3, BOSS / 2, 4.2, 16, CHROME, R_BED)
    return b


_PLANE_BOXES, _GRIP, _FACE = _plane_body()
# The iron's hold on the frog. PhysX averages the two materials of a contact, so a box given FACE_MATERIAL has
# FROG_FRICTION against plain steel: the frog's face against the iron's back, and the cap iron's bridge against the
# yoke's tab. Everything else stays at STEEL_FRICTION (the slot on the boss and on the screw, the lever cap on the cap
# iron, the window's end against the tab).
FACE_MATERIAL = 2 * FROG_FRICTION - STEEL_FRICTION
LATERAL = Hinge("lateral", pivot=tuple(float(v) for v in bed_to_plane(0, LAT_PIVOT_P, -0.8)),
                boxes=tuple(_lateral_lever()), lower=-LAT_LIMIT, upper=LAT_LIMIT, stiffness=0.0, damping=20.0,
                effort=0.003, mass=0.015, axis="Z", rot=Q_BED)
PLANE_POS = (PLANE_X, 0.0, float(TOP + 0.0003))
PLANE_ROT = rot("Z", PLANE_YAW)
PLANE = part("plane", pos=PLANE_POS, yaw=PLANE_YAW, mass=PLANE_MASS, boxes=_PLANE_BOXES, hinges=(LATERAL,), color=IRON_GREY,
             friction=STEEL_FRICTION, meta={"contact_offset": 0.0003, "grip_boxes": tuple(_GRIP), "grip_friction": 1.0,
                                            "box_friction": {"body": {i: FACE_MATERIAL for i in _FACE}}})

# -- the iron and cap iron -----------------------------------------------------------------------------------------------
_I = [span(-IRON_HALF_W, IRON_HALF_W, 0, BEVEL, IRON_T / 2, IRON_T, BRIGHT),                # the bevel (stepped)
      span(-IRON_HALF_W, IRON_HALF_W, BEVEL, SLOT_Y[0], 0, IRON_T, STEEL),
      span(-IRON_HALF_W, -SLOT_HALF_W, SLOT_Y[0], SLOT_Y[1], 0, IRON_T, STEEL),
      span(SLOT_HALF_W, IRON_HALF_W, SLOT_Y[0], SLOT_Y[1], 0, IRON_T, STEEL),
      span(-IRON_HALF_W, IRON_HALF_W, SLOT_Y[1], IRON_L, 0, IRON_T, STEEL)]
_Z0, _Z1 = IRON_T, IRON_T + CAPIRON_T
_hump = ((CAPIRON_Y[0], _Z0), (7.5, _Z0 + 3.6), (15.0, _Z1))                               # front edge, crown, flat
for (ya, za), (yb, zb) in zip(_hump, _hump[1:]):
    a = math.atan2(zb - za, yb - ya)
    mid = np.array([0.0, (ya + yb) / 2, (za + zb) / 2]) + rot("X", a) @ np.array([0.0, 0.0, -CAPIRON_T / 2])
    _I.append(obox(mid, (2 * IRON_HALF_W, math.hypot(yb - ya, zb - za) + 0.6, CAPIRON_T), IRON_GREY, rot("X", a)))
_I += [span(-IRON_HALF_W, IRON_HALF_W, 15.0, CI_SLOT_Y[0], _Z0, _Z1, IRON_GREY),
       span(-IRON_HALF_W, -CI_SLOT_HALF_W, CI_SLOT_Y[0], CAPIRON_Y[1], _Z0, _Z1, IRON_GREY),
       span(CI_SLOT_HALF_W, IRON_HALF_W, CI_SLOT_Y[0], CAPIRON_Y[1], _Z0, _Z1, IRON_GREY),
       span(-CI_SLOT_HALF_W, CI_SLOT_HALF_W, WINDOW_Y[1], CAPIRON_Y[1], _Z0, _Z1, IRON_GREY)]
_BRIDGE = len(_I)                                                                          # the bridge the yoke's tab carries
_I.append(span(-CI_SLOT_HALF_W, CI_SLOT_HALF_W, CI_SLOT_Y[1], WINDOW_Y[0], _Z0, _Z1, IRON_GREY))
_I += prism(0, CAPSCREW_Y, -3.0, 0.0, CAPSCREW_HEAD, 4, BRASS)                              # cap iron screw's head
_I += prism(0, CAPSCREW_Y, 0.0, _Z1, 7.9, 4, BRASS)                                         # and its shank (5/16 in)

BATTEN = (0.130, 0.038, 0.038)                 # iron batten: length, width, height (a planed 2 x 2 offcut)
CAP_BATTEN = (0.106, 0.038, 0.038)
IRON_START = (0.160, -0.150, float(TOP + BATTEN[2] + 0.0002))      # the cutting edge's middle; the top end is away from the robot
CAP_START = (0.180, -0.300, float(TOP + CAP_BATTEN[2] + 0.0002))   # the toe's middle
PART_YAW = -math.pi / 2                        # both lie with their length along x, edge and toe toward the robot
IRON = part("iron", pos=IRON_START, yaw=PART_YAW, mass=IRON_MASS, boxes=_I, color=STEEL, friction=STEEL_FRICTION,
            meta={"contact_offset": 0.0003, "box_friction": {"body": {_BRIDGE: FACE_MATERIAL}}})
IRON_BATTEN = Box("iron_batten", pos=(IRON_START[0] + 0.115, IRON_START[1], float(TOP + BATTEN[2] / 2)),
                  size=BATTEN, color=PINE)
CAP_BATTEN_BOX = Box("cap_batten", pos=(CAP_START[0] + 0.055, CAP_START[1], float(TOP + CAP_BATTEN[2] / 2)),
                     size=CAP_BATTEN, color=PINE)

# -- the lever cap -------------------------------------------------------------------------------------------------------
_PZ0, _PZ1 = CAP_PLATE_Z
_C = [span(-CAP_HALF_W, CAP_HALF_W, 0, 4, 0, 3.5, CHROME)]                                  # toe
for s in (1, -1):
    x0, x1 = sorted((s * LOBE_HALF_W, s * CAP_HALF_W))
    _C.append(span(x0, x1, 4, CAP_L, _PZ0, _PZ1, CHROME))                                   # the plate's two sides
    x0, x1 = sorted((s * SEAT_HALF_W, s * LOBE_HALF_W))
    _C.append(span(x0, x1, LOBE_Y[1], SEAT_END, _PZ0, _PZ1, CHROME))                        # beside the keyhole's narrow end
    R = rot("Z", math.radians(135.0 if s > 0 else 45.0))                                    # the keyhole's shoulder,
    mid = np.array([s * (LOBE_HALF_W + SEAT_HALF_W) / 2, LOBE_Y[1] - (LOBE_HALF_W - SEAT_HALF_W) / 2,
                    (_PZ0 + _PZ1) / 2]) + np.array([s * C45, C45, 0.0]) * 1.5                  # narrowing to the seat
    _C.append(obox(mid, ((LOBE_HALF_W - SEAT_HALF_W) * math.sqrt(2), 3.0, _PZ1 - _PZ0), CHROME, R))
    x0, x1 = sorted((s * 8.5, s * 11.0))
    _C.append(span(x0, x1, CAM_Y - 4, CAM_Y + 4, _PZ1, CAM_Z + 2.5, CHROME))                # the cam pin's two ears
_C += [span(-LOBE_HALF_W, LOBE_HALF_W, 4, LOBE_Y[0], _PZ0, _PZ1, CHROME),
       span(-LOBE_HALF_W, LOBE_HALF_W, SEAT_END, CAM_Y - 6, _PZ0, _PZ1, CHROME),
       span(-LOBE_HALF_W, LOBE_HALF_W, CAM_Y + 6, CAP_L, _PZ0, _PZ1, CHROME)]
LEAF = Slider("leaf", rest=(0.0, CAM_Y * MM, 5.8 * MM), boxes=(span(-7, 7, CAM_Y - 5, CAM_Y + 5, 4.6, 7.0, BRIGHT),),
              axis="Z", lower=-LEAF_TRAVEL * MM, upper=0.0, rate=LEAF_RATE, damping=3.0, mass=LEAF_MASS,
              target=LEAF_PRELOAD)
FOOT = Slider("foot", rest=(0.0, CAM_Y * MM, 2.9 * MM), boxes=(span(-7, 7, CAM_Y - 5, CAM_Y + 5, 1.6, 4.2, BRIGHT),),
              axis="Z", lower=0.0, upper=FOOT_TRAVEL, rate=FOOT_RATE, damping=10.0, mass=FOOT_MASS, parent="leaf",
              target=-FOOT_PRELOAD)
# the cam lever, drawn raised (angle 0); it turns CAM_OPEN about +x to its closed stop, lying toward the toe
# (its pin turns freely: raised, the lever leans back on its stop, and tipped forward on the frog its eccentric rests
# on the spring plate, whose 2 N return spring holds it up)
_A_OPEN = CAM_PAST - CAM_OPEN                               # the eccentric's direction, from straight down, when raised
_lobe = (CAM_Y + CAM_E * math.sin(_A_OPEN), CAM_Z - CAM_E * math.cos(_A_OPEN))
_arm_a = math.pi - CAM_OPEN                                 # the raised arm's direction above the cap's +y
_arm_mid = np.array([0.0, CAM_Y, CAM_Z]) + (LEVER_LEN + 4) / 2 * np.array([0.0, math.cos(_arm_a), math.sin(_arm_a)])
_LEVER = prism(_lobe[0], _lobe[1], -6, 6, 2 * CAM_R, 12, BRIGHT, axis="X")
_LEVER.append(obox(_arm_mid, (14, LEVER_LEN - 4, 3), CHROME, rot("X", _arm_a)))
CAM = Hinge("cam", pivot=(0.0, CAM_Y * MM, CAM_Z * MM), boxes=tuple(_LEVER), lower=0.0, upper=CAM_OPEN, stiffness=0.0,
            damping=0.0005, mass=CAM_MASS, axis="X")
CAP = part("lever_cap", pos=CAP_START, yaw=PART_YAW, mass=CAP_MASS - CAM_MASS - LEAF_MASS - FOOT_MASS, boxes=_C,
           sliders=(LEAF, FOOT), hinges=(CAM,), color=CHROME, friction=STEEL_FRICTION,
           meta={"contact_offset": 0.0003, "self_collide": True, "position_iterations": CAP_ITERATIONS})

OBJECTS = [PLANE, IRON, CAP, IRON_BATTEN, CAP_BATTEN_BOX]

# -- scoring -------------------------------------------------------------------------------------------------------------
ALIGNED = math.cos(math.radians(10.0))   # a part counts as the right way round within 10 degrees
ON_FROG = 3.0                            # mm: the iron's back within this of the frog's face at the screw
SEATED = 0.6                             # mm: and within this of it at the yoke and at the boss
SLACK = 0.2                              # mm allowed on every footprint test (contact give)
HOOK_BACK = 2.0                          # mm: the screw within this of the keyhole's narrow end
CAM_TOL = math.radians(3.0)
PLANE_MOVE, PLANE_TILT, PLANE_OVER = 0.020, math.radians(10.0), math.radians(45.0)
EDGE_NEAR = 0.015
RELEASE = 0.03
STILL_STEPS = 20
PLANE_BOX = ((SOLE[0], SOLE[1]), (-SOLE_HALF_W, SOLE_HALF_W), (0.0, 125.0))            # mm, plane frame
IRON_BOX = ((-IRON_HALF_W, IRON_HALF_W), (0.0, IRON_L), (-3.0, 6.5))                   # mm, iron frame
CAP_BOX = ((-CAP_HALF_W, CAP_HALF_W), (0.0, CAP_L), (0.0, 15.0))                       # mm, cap frame (lever down)
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), over=False, edge=False, plane0=None, iron_free=False,
                  cap_free=False, plane_free=False, finger_check=None, recent=[], info={})
    if environment is not None:
        try:
            set_slider_target(environment, "lever_cap", "foot", -FOOT_PRELOAD)     # the spring plate's preload
            set_slider_target(environment, "lever_cap", "leaf", LEAF_PRELOAD)
        except Exception as exc:  # noqa: BLE001
            print(f"bench-plane-reassemble: spring preload not set: {exc!r}", flush=True)


_reset_state()


def _box_distance(p, bounds) -> float:
    d = [max(lo - v, 0.0, v - hi) for v, (lo, hi) in zip(p, bounds)]
    return float(math.sqrt(sum(x * x for x in d)))


def in_bed(plane: PartState, other: PartState):
    """A part's origin (mm) and rotation in the frog's frame (l, p, h)."""
    Rb = plane.rotation() @ R_BED
    return Rb.T @ (other.pos - plane.pos) / MM, Rb.T @ other.rotation()


def iron_status(plane: PartState, iron: PartState) -> dict:
    """Where the iron assembly is on the frog. ``on_frog``: over the lever-cap screw, bevel down, edge toward the
    mouth, its back within 3 mm of the face there. ``yoke`` / ``boss``: the yoke's tab inside the cap iron's window /
    the lateral lever's boss inside the iron's slot, the iron's back within 0.6 mm of the face at that point."""
    o, R = in_bed(plane, iron)

    def local(l, p, h):   # noqa: E741
        return R.T @ (np.array([l, p, h], float) - o)

    bevel_down, edge_down = bool(R[2, 2] > ALIGNED), bool(R[1, 1] > ALIGNED)
    wrong = bool(R[2, 2] < 0 or R[1, 1] < 0)
    s = local(0, SCREW_P, 0)
    over_screw = bool(abs(s[0]) <= SLOT_HALF_W - SHANK / 2 + SLACK and SLOT_Y[0] <= s[1] <= SLOT_Y[1]
                      and abs(s[2]) <= 12.0)
    on_frog = bool(over_screw and bevel_down and edge_down and -ON_FROG <= s[2] <= 1.0)
    t = local(0, sum(TAB_P) / 2, 0)
    half = (TAB_P[1] - TAB_P[0]) / 2
    yoke = bool(on_frog and abs(t[0]) <= CI_SLOT_HALF_W - TAB_HALF_W + SLACK
                and WINDOW_Y[0] + half - SLACK <= t[1] <= WINDOW_Y[1] - half + SLACK and -SEATED <= t[2] <= 1.0)
    a = plane.hinges.get("lateral", 0.0)
    b = local(LAT_ARM * math.sin(a), LAT_PIVOT_P - LAT_ARM * math.cos(a), 0)
    boss = bool(on_frog and abs(b[0]) <= SLOT_HALF_W - BOSS / 2 + SLACK
                and SLOT_Y[0] + BOSS / 2 <= b[1] <= SLOT_Y[1] - BOSS / 2 and -SEATED <= b[2] <= 1.0)
    return {"on_frog": on_frog, "yoke": yoke, "boss": boss, "wrong_way": wrong, "over_screw": over_screw,
            "slide_mm": round(float(o[1] - ENGAGED_P), 2), "gap_screw_mm": round(float(-s[2]), 2),
            "gap_yoke_mm": round(float(-t[2]), 2), "gap_boss_mm": round(float(-b[2]), 2),
            "side_mm": round(float(o[0]), 2)}


def cap_status(plane: PartState, cap: PartState) -> dict:
    """``hooked``: the lever cap lies the right way round with the screw in the narrow end of its keyhole (within 2 mm
    of the end) and its plate under the screw's head. ``cam``: the cam lever is within 3 degrees of its closed stop."""
    o, R = in_bed(plane, cap)
    s = R.T @ (np.array([0.0, SCREW_P, HEAD_UNDER]) - o)
    right_way = bool(R[2, 2] > ALIGNED and R[1, 1] > ALIGNED)
    seat = SEAT_END - SHANK / 2
    hooked = bool(right_way and abs(s[0]) <= SEAT_HALF_W - SHANK / 2 + SLACK
                  and seat - HOOK_BACK <= s[1] <= seat + 0.5 and CAP_PLATE_Z[1] - SLACK <= s[2] <= CAP_PLATE_Z[1] + 2.5)
    cam = cap.hinges.get("cam", 0.0)
    return {"hooked": hooked, "over_iron": bool(abs(s[0]) <= CAP_HALF_W and 0 <= s[1] <= CAP_L and abs(s[2]) <= 25.0),
            "cam": bool(cam >= CAM_OPEN - CAM_TOL), "cam_deg": round(math.degrees(CAM_OPEN - cam), 1),
            "screw_in_cap_mm": [round(float(v), 2) for v in s]}


def plane_shift(plane: PartState) -> tuple[float, float, float]:
    """How far the plane has moved along the bench (m), how far it has tipped or turned in all (rad), and how far it
    has tipped off its sole alone (rad)."""
    if _STATE["plane0"] is None:
        return 0.0, 0.0, 0.0
    p0, r0 = _STATE["plane0"]
    R = plane.rotation()
    cos = (np.trace(r0.T @ R) - 1) / 2
    tip = float(r0[:, 2] @ R[:, 2])
    return (float(np.linalg.norm(plane.pos - p0)), float(math.acos(max(-1.0, min(1.0, cos)))),
            float(math.acos(max(-1.0, min(1.0, tip)))))


def cutting_edge(iron: PartState) -> np.ndarray:
    """World points along the iron's cutting edge."""
    return np.array([iron.to_world(np.array([x, 0.0, IRON_T]) * MM) for x in np.linspace(-IRON_HALF_W, IRON_HALF_W, 9)])


@on_step
def _step(environment) -> None:
    """Every control step: count steps and check where the grippers are (score() cannot see the robot)."""
    from yam_arena.scene import object_states

    _STATE["step"] += 1
    try:
        st = object_states(environment, ["plane", "iron", "lever_cap"])
        pts = finger_points(environment)
        plane, iron, cap = st["plane"], st["iron"], st["lever_cap"]
        edge = cutting_edge(iron)
        if min(np.linalg.norm(p - e) for p in pts for e in edge) < EDGE_NEAR:
            _STATE["edge"] = True                                                    # permanent
        _STATE["iron_free"] = bool(min(_box_distance(iron.to_local(p) / MM, IRON_BOX) for p in pts) >= RELEASE / MM)
        _STATE["cap_free"] = bool(min(_box_distance(cap.to_local(p) / MM, CAP_BOX) for p in pts) >= RELEASE / MM)
        _STATE["plane_free"] = bool(min(_box_distance(plane.to_local(p) / MM, PLANE_BOX) for p in pts) >= RELEASE / MM)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"bench-plane-reassemble: gripper check unavailable: {exc!r}", flush=True)
        _STATE.update(finger_check=False, iron_free=False, cap_free=False, plane_free=False)


def _pose_points(part_state: PartState) -> np.ndarray:
    """Three points fixed in a part, 100 mm apart: they stand still only if its whole pose does."""
    return np.concatenate([part_state.pos, part_state.to_world((0.1, 0.0, 0.0)), part_state.to_world((0.0, 0.1, 0.0))])


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    plane, iron, cap = objects["plane"], objects["iron"], objects["lever_cap"]
    if _STATE["plane0"] is None:
        _STATE["plane0"] = (plane.pos.copy(), plane.rotation())
    shift, tilt, tip = plane_shift(plane)
    if tip > PLANE_OVER:
        _STATE["over"] = True                                                        # permanent
    i, c = iron_status(plane, iron), cap_status(plane, cap)
    violation = "plane knocked over" if _STATE["over"] else "gripper at the cutting edge" if _STATE["edge"] else None
    wrong = bool(i["wrong_way"] and i["over_screw"] and c["over_iron"])
    placed = i["on_frog"] and _STATE["iron_free"]
    engaged = placed and i["yoke"] and i["boss"]
    hooked = engaged and c["hooked"] and _STATE["cap_free"]
    locked = hooked and c["cam"]
    in_place = shift <= PLANE_MOVE and tilt <= PLANE_TILT
    recent = _STATE["recent"]
    recent.append(np.concatenate([_pose_points(plane), _pose_points(iron), _pose_points(cap),
                                  [cap.hinges.get("cam", 0.0) * LEVER_LEN * MM]]))
    del recent[:-STILL_STEPS]
    still = len(recent) == STILL_STEPS and all(float(np.abs(r - recent[-1]).max()) < 0.001 for r in recent)
    _STATE["info"] = {**i, **c, "plane_shift_mm": round(1000 * shift, 1), "plane_tilt_deg": round(math.degrees(tilt), 1),
                      "plane_tip_deg": round(math.degrees(tip), 1),
                      "iron_free": _STATE["iron_free"], "cap_free": _STATE["cap_free"],
                      "plane_free": _STATE["plane_free"], "still": still, "wrong": wrong, "violation": violation}
    if violation or wrong:
        result = (0.0, False)
    elif locked and in_place and _STATE["plane_free"] and still:
        result = (1.0, True)
    elif hooked:
        result = (0.75, False)
    elif engaged:
        result = (0.5, False)
    elif placed:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Iron assembly on the frog with the yoke in the cap iron's window and the boss in the iron's slot, lever cap
    hooked under the screw head with its cam lever on its closed stop, the plane upright where it stood, both grippers
    clear, everything still for the last second, and no violation."""
    return _update(objects)[1]


def distance(objects) -> float:
    plane, iron, cap = objects["plane"], objects["iron"], objects["lever_cap"]
    oi, _ = in_bed(plane, iron)
    oc, _ = in_bed(plane, cap)
    cap_home = np.array([0.0, SCREW_P - (SEAT_END - SHANK / 2), IRON_T + CAPIRON_T])
    return float((np.linalg.norm(oi - [0.0, ENGAGED_P, 0.0]) + np.linalg.norm(oc - cap_home)) * MM)


def score(objects) -> float:
    """End state. 0.25: the iron assembly lies on the frog over the lever-cap screw, bevel down and cutting edge
    toward the mouth (within 10 degrees), its back within 3 mm of the frog's face at the screw, both grippers 3 cm
    clear of it. 0.5: also the yoke's tab stands inside the cap iron's window and the lateral lever's boss inside the
    iron's slot, the iron's back within 0.6 mm of the face at both (the iron then cannot slide more than 1 mm along
    the frog). 0.75: also the lever cap lies on the cap iron the right way round with the screw within 2 mm of the
    narrow end of its keyhole, its plate under the screw's head, both grippers 3 cm clear of it. 1.0: also the cam
    lever is within 3 degrees of its closed stop, the plane has moved no more than 20 mm and tipped or turned no more
    than 10 degrees, both grippers are 3 cm clear of the plane, and plane, iron and lever cap have not moved more
    than 1 mm for 1 second. 0 while the lever cap lies over an iron that is bevel up or has its cutting edge up the
    slope. 0 for good if the plane was tipped more than 45 degrees off its sole (knocked over) or a gripper came
    within 15 mm of the iron's cutting edge (blade contact)."""
    return _update(objects)[0]
