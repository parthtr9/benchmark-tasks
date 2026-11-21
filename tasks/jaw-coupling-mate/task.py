"""On the bench: a small drive, bolted down. A worm gear reducer's output shaft carries one hub of a Lovejoy L-100 jaw
coupling, set-screwed in place; a worm does not back-drive, so that hub stays where it is. In line with it, 19.5 mm
away, ends a 5/8 in driven shaft that turns freely in two UCP202-10 pillow blocks. The coupling's second hub sits on
that shaft's key, slid 41.5 mm back from its working place, which a scribed line on the shaft marks. The coupling's
Hytrel spider (six legs, open centre) stands in a slotted holder in front of the right arm. Couple the two shafts:
the spider between the jaws of both hubs and the loose hub at its line. The job stops before the loose hub's set
screw is tightened.

Real parts and published sizes (read 2026-11-17 from the maker's CAD listings and its L-Line installation guide):
  hub     Lovejoy L-100 x 5/8 in, sintered iron (68514411510): 64.52 mm across, 35.05 mm through the bore, 53.09 mm
          long with its three jaws (jaws 18.04 mm), hubs 18.03 mm apart when coupled ("G"), 0.70 kg
  spider  Lovejoy L099/100 open-centre Hytrel (55D) spider (68514411486): 64.52 mm across the legs (A), legs 10.92 mm
          thick (B), 26.16 mm centre hole (H), 15.49 mm wide (W), 27 g
  bearings UCP202-10 pillow blocks, 5/8 in bore, shaft centre 30.2 mm above the foot, foot 127 x 38 mm
The jaws are as long as the hubs' spacing, so a hub pushed home stops with its jaw tips on the other hub's face: the
catalogue spacing, and the scribed line, are where it stops.

Not published, so estimates to measure on the bought parts: the spider's ring (18.5 mm outer radius), where the jaws
start (19.5 mm from the axis), and the spider's moulded leg-edge radius. That radius is the only lead-in in the
model. The maker's photo of its Hytrel and urethane spiders shows leg edges rounded to about 1 mm; here each leg edge
carries a 0.6 mm 45-degree facet, which lies inside a 1 mm radius. The hubs' jaw tips are flat and square, as in the
maker's photos. The spider is a rigid body with 0.5 mm of play per side between each leg and the jaws beside it (the
real one is a close fit and gives a little).

Why the driven shaft ends 19.5 mm short of the fixed hub's jaws: a spider can only go into a hub's jaws along the
shaft, so there must be room for it (15.49 mm wide) between those jaws and the other shaft's end. The loose hub
therefore overhangs its shaft's end when coupled, by 19.5 mm, leaving 15.55 mm of its 35.05 mm bore on the shaft:
the shaft's own diameter, which is the engagement the maker's installation guide asks for.

Frames: the shafts' common axis is along world x, AXIS_Z above the bench on the centre line. Coupling geometry is
written in mm in an "axial" frame (u = world y, v = world z, w = along the axis, w = 0 at the fixed hub's jaw tips,
positive toward the reducer), with angles measured from u toward v.

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
from partkit import Hinge, PartState, Slider, finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = ("Couple the gear reducer to the driven shaft: the spider between the jaws of both hubs and the loose "
               "hub at the scribed line on its shaft.")

IRON = (0.42, 0.43, 0.45)
STEEL = (0.74, 0.75, 0.77)
CAST = (0.16, 0.32, 0.26)
BLUE = (0.13, 0.25, 0.47)
HYTREL = (0.86, 0.78, 0.60)
BASE_C = (0.23, 0.24, 0.26)
DARK = (0.06, 0.06, 0.07)
WOOD = (0.72, 0.58, 0.40)
MM = 0.001

# -- catalogue dimensions (mm) ---------------------------------------------------------------------------------------------
R_OD = 64.52 / 2        # hub outside radius; the spider's legs reach the same radius (A)
LTB = 35.05             # hub length through the bore
JAW = 18.03             # jaw length = the hubs' spacing G
SPIDER_W = 15.49        # W
LEG_B = 10.92           # B
HOLE_R = 26.16 / 2      # H
SHAFT_R = 15.875 / 2    # 5/8 in shafts
HUB_MASS = 0.70         # 1.56 lb listed with a 15 mm bore
SPIDER_MASS = 0.027     # 0.06 lb
# -- estimates and model choices (mm) ---------------------------------------------------------------------------------------
RING_R = 18.5           # the spider ring's outer radius (estimate from the maker's photo)
JAW_R0 = 19.5           # the jaws' inner radius (estimate)
CLEAR = 0.5             # play per side between a leg and a jaw flank (the real spider is a close fit)
LEG_FACET = 0.6         # 45-degree facet on each leg edge, inside the moulded radius of about 1 mm (estimate)
SLAB = 6.0              # thickness of the boxes that make a jaw's flanks
# -- the stand (mm) ---------------------------------------------------------------------------------------------------------
SPACE = 19.5            # from the fixed hub's jaw tips to the driven shaft's end: the spider's 15.49 mm and 2 mm a side
PARK = 41.5             # the loose hub starts this far back from its line (its jaw tips 4 mm behind the shaft's end)
PHASE0 = math.radians(30.0)   # and turned this far from where its jaws face the spider's free gaps
SHAFT_BACK = 238.0      # the driven shaft's free end, measured back from the fixed hub's jaw tips
BLOCK_W = (-125.5, -175.5)    # the two pillow blocks' centres along the axis
BLOCK_H = 30.2          # shaft centre above the pillow block's foot
SLIDE_FRICTION = 1.4    # N: the 0.70 kg hub on its shaft and key at friction 0.2
TURN_FRICTION = 0.02    # N m: two sealed insert bearings (estimate; about 2.5 N on a spring scale at the shaft's surface)

AXIS_Z = (BLOCK_H + 60.0) * MM            # the pillow blocks stand on a 60 mm base frame
X_TIPS = 0.392                            # world x of the fixed hub's jaw tips
ORIGIN = (X_TIPS, 0.0, TOP + AXIS_Z)      # the coupling parts' frame: on the axis at the fixed hub's jaw tips
LINE_W = -LTB                             # the scribed line: where the loose hub's rear face stands when coupled


# -- geometry helpers -------------------------------------------------------------------------------------------------------
def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


_AXIAL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])    # (u, v, w) -> part (x, y, z)


def quat_wxyz(R) -> tuple:
    """Rotation matrix to (w, x, y, z), Shepperd's method (safe near half turns)."""
    R = np.asarray(R, float)
    t = [R[0, 0], R[1, 1], R[2, 2], R[0, 0] + R[1, 1] + R[2, 2]]
    i = int(np.argmax(t))
    if i == 3:
        q = np.array([1.0 + t[3], R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    elif i == 0:
        q = np.array([R[2, 1] - R[1, 2], 1.0 + 2 * R[0, 0] - t[3], R[0, 1] + R[1, 0], R[0, 2] + R[2, 0]])
    elif i == 1:
        q = np.array([R[0, 2] - R[2, 0], R[0, 1] + R[1, 0], 1.0 + 2 * R[1, 1] - t[3], R[1, 2] + R[2, 1]])
    else:
        q = np.array([R[1, 0] - R[0, 1], R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], 1.0 + 2 * R[2, 2] - t[3]])
    q = q / np.linalg.norm(q)
    return tuple(float(v) for v in (q if q[0] >= 0 else -q))


def to_part(shapes, color, shift_w: float = 0.0, origin_mm=(0.0, 0.0, 0.0)):
    """Axial-frame boxes [(centre uvw, size, rotation)] in mm -> partkit boxes in metres. ``shift_w`` moves them along
    the axis; ``origin_mm`` is where the axial frame's origin sits in the part frame."""
    out = []
    for c, size, R in shapes:
        p = _AXIAL @ (np.asarray(c, float) + [0.0, 0.0, shift_w]) + np.asarray(origin_mm, float)
        out.append((tuple(float(v) * MM for v in p), tuple(float(v) * MM for v in size), color,
                    quat_wxyz(_AXIAL @ np.asarray(R, float))))
    return out


def disc(radius: float, w0: float, w1: float, n: int = 8):
    """A solid 2n-sided prism about the axis with its CORNERS on ``radius`` (n bars through the centre)."""
    a = radius * math.cos(math.pi / (2 * n))                    # apothem
    side = 2 * a * math.tan(math.pi / (2 * n))
    return [(np.array([0.0, 0.0, (w0 + w1) / 2]), np.array([2 * a, side, w1 - w0]), rz(math.pi * k / n))
            for k in range(n)]


def annulus(r_out: float, r_in: float, w0: float, w1: float, n: int = 12):
    """An n-sided ring whose outer corners are on ``r_out`` and whose inner flats are on ``r_in`` (never tighter
    than the true hole)."""
    a = r_out * math.cos(math.pi / n)
    side = 2 * a * math.tan(math.pi / n)
    out = []
    for k in range(n):
        t = 2 * math.pi * k / n
        rm = (a + r_in) / 2
        out.append((np.array([rm * math.cos(t), rm * math.sin(t), (w0 + w1) / 2]), np.array([a - r_in, side, w1 - w0]),
                    rz(t)))
    return out


def jaw(phi: float, w0: float, w1: float, clear: float = CLEAR):
    """One jaw centred at angle ``phi``: a wedge whose two flanks are parallel to the legs either side of it, made of
    two flank slabs and a filler, all inside the hub's radius. Tips flat, edges square."""
    h = LEG_B / 2 + clear
    s0 = math.sqrt(JAW_R0 ** 2 - h ** 2)
    s1 = math.sqrt(R_OD ** 2 - (h + SLAB) ** 2)
    out = []
    for sgn in (1.0, -1.0):
        a = phi + sgn * math.pi / 6
        u = np.array([math.cos(a), math.sin(a), 0.0])
        n = np.array([-math.sin(a), math.cos(a), 0.0]) * sgn          # from the jaw toward this leg
        c = (s0 + s1) / 2 * u - (h + SLAB / 2) * n
        out.append((np.array([c[0], c[1], (w0 + w1) / 2]), np.array([s1 - s0, SLAB, w1 - w0]), rz(a)))
    r0, r1, half = 23.5, 31.0, 6.0
    c = (r0 + r1) / 2 * np.array([math.cos(phi), math.sin(phi), 0.0])
    out.append((np.array([c[0], c[1], (w0 + w1) / 2]), np.array([r1 - r0, 2 * half, w1 - w0]), rz(phi)))
    return out


def hub(jaw_angles, body, jaws, clear: float = CLEAR):
    """A hub: body between ``body`` = (w0, w1), three jaws between ``jaws`` = (w0, w1)."""
    out = disc(R_OD, *body)
    for phi in jaw_angles:
        out += jaw(phi, *jaws, clear=clear)
    return out


def leg(facet: float = LEG_FACET):
    """One spider leg in its own frame (x radial, y across its thickness B, z across the width W): from inside the
    ring out to where its tip's corners are on the spider's radius, with a 45-degree facet on its four long edges."""
    r1 = math.sqrt(R_OD ** 2 - (LEG_B / 2) ** 2)
    r0 = RING_R - 1.0
    length, mid = r1 - r0, np.array([(r0 + r1) / 2, 0.0, 0.0])
    if facet <= 0:
        return [(mid, np.array([length, LEG_B, SPIDER_W]), np.eye(3))]
    out = [(mid, np.array([length, LEG_B, SPIDER_W - 2 * facet]), np.eye(3)),
           (mid, np.array([length, LEG_B - 2 * facet, SPIDER_W]), np.eye(3))]
    ex, t = np.array([1.0, 0.0, 0.0]), 1.2
    for sy in (1.0, -1.0):
        for sz in (1.0, -1.0):
            n = np.array([0.0, sy, sz]) / math.sqrt(2.0)
            face = np.array([0.0, sy * (LEG_B / 2 - facet / 2), sz * (SPIDER_W / 2 - facet / 2)])
            out.append((mid + face - n * t / 2, np.array([length, facet * math.sqrt(2.0), t]),
                        np.c_[ex, np.cross(n, ex), n]))
    return out


LEG_ANGLES = tuple(math.radians(60.0 * k) for k in range(6))            # two legs level, for the fingers
# The fixed hub stands with one jaw straight up, so the two level legs rest squarely on the jaws below them. A rigid
# spider with play settles 0.5 mm low this way (its play); with a jaw straight down it would settle 1.0 mm low, which
# a close-fitting real spider does not do.
JAWS_A = tuple(math.radians(a) for a in (90.0, 210.0, 330.0))           # the fixed hub's jaws
JAWS_B = tuple(math.radians(a) for a in (30.0, 150.0, 270.0))           # where the loose hub's jaws must end up


def spider_shapes(facet: float = LEG_FACET):
    """The spider in its own axial frame (origin at its centre): ring and six legs."""
    out = annulus(RING_R, HOLE_R, -SPIDER_W / 2, SPIDER_W / 2)
    for a in LEG_ANGLES:
        for c, size, R in leg(facet):
            out.append((rz(a) @ c, size, rz(a) @ R))
    return out


def hub_a_shapes():
    """The fixed hub in the axial frame: jaws from w 0 to JAW, body behind them."""
    return hub(JAWS_A, (JAW, JAW + LTB), (0.0, JAW))


def hub_b_shapes(phase: float = PHASE0):
    """The loose hub at its line (coupled position) in the axial frame, turned ``phase`` from meshing: body from
    w -LTB to 0, jaws from 0 to JAW."""
    return hub([a + phase for a in JAWS_B], (-LTB, 0.0), (0.0, JAW))


# -- the fixed side: gear reducer on its riser, output shaft, hub A ---------------------------------------------------------
def _m(x, y, z, sx, sy, sz, color):
    return ((x * MM, y * MM, z * MM), (sx * MM, sy * MM, sz * MM), color)


_AX = AXIS_Z / MM                                   # axis height above the bench (mm); the drive part's origin is on it
_RED_X0 = JAW + LTB + 12.0                          # reducer housing's front face (12 mm of bare shaft behind the hub)
DRIVE_BOXES = to_part(hub_a_shapes(), IRON)
DRIVE_BOXES += [
    _m(_RED_X0 + 36.5, 0.0, 0.0, 73.0, 108.0, 118.0, BLUE),                      # housing 73 x 108 x 118 mm
    _m(_RED_X0 + 36.5, 0.0, -(_AX + 59.0) / 2 - 0.0, 110.0, 130.0, _AX - 59.0, BASE_C),   # riser under it
    _m(_RED_X0 + 36.5, 74.0, 33.8, 12.7, 40.0, 12.7, STEEL),                      # input shaft stub (worm, above)
    _m(_RED_X0 + 36.5, 0.0, 62.0, 30.0, 30.0, 6.0, DARK),                          # breather plug boss
]
DRIVE_CYLS = [((float((JAW + LTB / 2 + 12.0) * MM), 0.0, 0.0), SHAFT_R * MM, float((LTB + 24.0) * MM), STEEL, "X")]
DRIVE = part("drive", pos=ORIGIN, mass=6.0, boxes=DRIVE_BOXES, cylinders=DRIVE_CYLS, color=IRON, friction=0.4,
             fixed=True, meta={"contact_offset": 0.0003})

# -- the driven side: base frame, two pillow blocks, the free shaft, the loose hub on its key -------------------------------
_BASE_X0, _BASE_X1 = BLOCK_W[1] - 19.0, BLOCK_W[0] + 19.0
DRIVEN_BOXES = [_m((_BASE_X0 + _BASE_X1) / 2, 0.0, -BLOCK_H - 30.0, _BASE_X1 - _BASE_X0, 140.0, 60.0, BASE_C)]
for _w in BLOCK_W:
    DRIVEN_BOXES += [
        _m(_w, 0.0, -BLOCK_H + 7.0, 38.0, 127.0, 14.0, CAST),                     # foot
        _m(_w, 0.0, -BLOCK_H + 14.0 + 12.0, 34.0, 62.0, 24.0, CAST),              # body under the bearing
    ]
    DRIVEN_BOXES += to_part(disc(31.0, _w - 15.5, _w + 15.5, n=6), CAST)          # bearing housing, 62 mm across
SHAFT_LEN = SHAFT_BACK - SPACE
SHAFT = Hinge("shaft", pivot=(0.0, 0.0, 0.0),
              cylinders=(((-(SHAFT_BACK + SPACE) / 2 * MM, 0.0, 0.0), SHAFT_R * MM, SHAFT_LEN * MM, STEEL, "X"),
                         ((LINE_W * MM - 0.0004, 0.0, 0.0), SHAFT_R * MM + 0.0001, 0.0008, DARK, "X")),   # scribed line
              lower=-math.inf, upper=math.inf, stiffness=0.0, damping=0.5, effort=TURN_FRICTION, mass=0.32, axis="X")
HUB_B = Slider("hub_b", rest=(-PARK * MM, 0.0, 0.0), boxes=tuple(to_part(hub_b_shapes(), IRON, shift_w=-PARK)),
               lower=-0.001, upper=(PARK + 3.0) * MM, rate=0.0, damping=400.0, effort=SLIDE_FRICTION, mass=HUB_MASS,
               axis="X", parent="shaft")
DRIVEN = part("driven", pos=ORIGIN, mass=4.0, boxes=DRIVEN_BOXES, hinges=[SHAFT], sliders=[HUB_B], color=CAST,
              friction=0.4, fixed=True, meta={"contact_offset": 0.0003})

# -- the spider and its holder -----------------------------------------------------------------------------------------------
HOLDER_XY = (0.270, -0.150)
HOLDER_FLOOR = 6.0                                   # mm of board under the spider
_tip_low = math.sqrt(R_OD ** 2 - (LEG_B / 2) ** 2) * math.sin(math.radians(60.0)) + (LEG_B / 2) * math.cos(math.radians(60.0))
SPIDER_REST_Z = TOP + (HOLDER_FLOOR + _tip_low) * MM + 0.0005     # standing on two leg tips
SLOT = SPIDER_W + 1.0
HOLDER = part("holder", pos=(HOLDER_XY[0], HOLDER_XY[1], TOP), mass=0.3, color=WOOD, friction=0.6, fixed=True,
              boxes=[_m(0.0, 0.0, HOLDER_FLOOR / 2, 60.0, 80.0, HOLDER_FLOOR, WOOD)]
              + [_m(s * (SLOT / 2 + 5.0), 0.0, HOLDER_FLOOR + 12.0, 10.0, 46.0, 24.0, WOOD) for s in (1.0, -1.0)])
SPIDER_START = (HOLDER_XY[0], HOLDER_XY[1], SPIDER_REST_Z)
SPIDER = part("spider", pos=SPIDER_START, mass=SPIDER_MASS, boxes=to_part(spider_shapes(), HYTREL), color=HYTREL,
              friction=0.5, meta={"contact_offset": 0.0003})

OBJECTS = [DRIVE, DRIVEN, HOLDER] + ([] if os.environ.get("JAW_COUPLING_NO_SPIDER") == "1" else [SPIDER])
HAS_SPIDER = SPIDER in OBJECTS

# -- scoring ------------------------------------------------------------------------------------------------------------------
SEAT_FLUSH = 0.001       # the spider's near face no more than 1 mm proud of the fixed hub's jaw tips
SEAT_RADIAL = 0.0015     # its centre within 1.5 mm of the shafts' axis
SEAT_TILT = math.radians(5.0)
ENGAGE = 0.003           # the loose hub's jaw tips 3 mm or more past the spider's near face
LINE_TOL = 0.001         # the loose hub's rear face within 1 mm of the scribed line
RELEASE = 0.030          # grippers 3 cm clear
MOVE_TOL = 0.001         # nothing moves more than 1 mm in the last second
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), spider_free=False, hub_free=False, all_free=False, finger_check=None,
                  recent=[], let_go=[], report={})


_reset_state()


def axial(point_world) -> np.ndarray:
    """A world point in the axial frame (m): (u, v, w)."""
    p = np.asarray(point_world, float) - np.array(ORIGIN)
    return np.array([p[1], p[2], p[0]])


def hub_travel(objects) -> float:
    """How far the loose hub has slid toward its line from the start (m); PARK mm when it is at the line."""
    return float(objects["driven"].sliders.get("hub_b", 0.0))


def shaft_angle(objects) -> float:
    return float(objects["driven"].hinges.get("shaft", 0.0))


def status(objects) -> dict:
    """Where the spider and the loose hub are, in the numbers the scoring text uses."""
    q = hub_travel(objects)
    tips_w = JAW * MM - (PARK * MM - q)                  # the loose hub's jaw tips along the axis
    out = {"travel_mm": q / MM, "to_line_mm": (PARK * MM - q) / MM, "tips_w_mm": tips_w / MM,
           "shaft_deg": math.degrees(shaft_angle(objects)), "seated": False, "engaged_mm": None, "spider": None}
    if "spider" in objects:
        s = objects["spider"]
        c = axial(s.pos)
        axis = s.rotation()[:, 0]                        # the spider's own axis (part x) in the world
        tilt = math.acos(min(1.0, abs(float(axis[0]))))
        near = c[2] - SPIDER_W * MM / 2                  # its face toward the driven shaft
        far = c[2] + SPIDER_W * MM / 2
        radial = float(math.hypot(c[0], c[1]))
        seated = bool(radial <= SEAT_RADIAL and tilt <= SEAT_TILT and near >= -SEAT_FLUSH and far <= JAW * MM + 0.0005)
        out.update(seated=seated, engaged_mm=(tips_w - near) / MM,
                   spider={"radial_mm": radial / MM, "near_mm": near / MM, "far_mm": far / MM,
                           "tilt_deg": math.degrees(tilt)})
    return out


def _solid_distance(point_axial, w0: float, w1: float, radius: float) -> float:
    """Distance (m) from an axial-frame point to a cylinder about the axis between w0 and w1."""
    rho = math.hypot(point_axial[0], point_axial[1])
    dw = max(w0 - point_axial[2], 0.0, point_axial[2] - w1)
    return float(math.hypot(max(rho - radius, 0.0), dw))


@on_step
def _step(environment) -> None:
    """Every control step: count steps and check the grippers (score() cannot see the robot)."""
    from yam_arena.scene import object_states

    _STATE["step"] += 1
    try:
        names = ["driven"] + (["spider"] if HAS_SPIDER else [])
        st = object_states(environment, names)
        pts = [axial(p) for p in finger_points(environment)]
        q = hub_travel(st)
        hub_w0 = (-LTB - PARK) * MM + q
        free = all(_solid_distance(p, hub_w0, hub_w0 + (LTB + JAW) * MM, R_OD * MM) >= RELEASE for p in pts)
        free = free and all(_solid_distance(p, -SHAFT_BACK * MM, -SPACE * MM, SHAFT_R * MM) >= RELEASE for p in pts)
        spider_free = False
        if HAS_SPIDER:
            sp = st["spider"]
            local = [sp.to_local(np.array(ORIGIN) + np.array([p[2], p[0], p[1]])) for p in pts]
            spider_free = all(_solid_distance((v[1], v[2], v[0]), -SPIDER_W * MM / 2, SPIDER_W * MM / 2, R_OD * MM)
                              >= RELEASE for v in local)
        _STATE["spider_free"] = bool(spider_free)
        _STATE["hub_free"] = bool(free)
        _STATE["all_free"] = bool(free and spider_free)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"jaw-coupling-mate: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"] = False
        _STATE["spider_free"] = _STATE["hub_free"] = _STATE["all_free"] = False


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    st = status(objects)
    seated = st["seated"] and _STATE["spider_free"]
    # the loose hub's levels count only once the grippers have let go of it and the shaft (review rule 6)
    engaged = (seated and _STATE["hub_free"] and st["engaged_mm"] is not None and st["engaged_mm"] * MM >= ENGAGE)
    at_line = engaged and abs(st["to_line_mm"]) * MM <= LINE_TOL
    recent = _STATE["recent"]
    th = shaft_angle(objects)
    row = [np.array([hub_travel(objects), R_OD * MM * math.cos(th), R_OD * MM * math.sin(th)])]
    if "spider" in objects:
        row += [objects["spider"].pos, objects["spider"].rotation()[:, 1] * R_OD * MM]
    recent.append(np.concatenate(row))
    del recent[:-20]
    still = len(recent) == 20 and all(float(np.abs(r - recent[-1]).max()) < MOVE_TOL for r in recent)
    # the grippers must have been clear for the whole last second, not just now (release judged over the window)
    let_go = _STATE["let_go"]
    let_go.append(bool(_STATE["all_free"]))
    del let_go[:-20]
    released = len(let_go) == 20 and all(let_go)
    if at_line and released and still:
        result = (1.0, True)
    elif at_line:
        result = (0.75, False)
    elif engaged:
        result = (0.5, False)
    elif seated:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _STATE["report"] = {**st, "spider_free": _STATE["spider_free"], "all_free": _STATE["all_free"], "released_1s": released, "still": still}
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """The spider sits between the fixed hub's jaws, the loose hub's jaws are in its free gaps with the hub's rear
    face within 1 mm of the scribed line, both grippers are 3 cm clear of spider, hub and driven shaft, and nothing has
    moved more than 1 mm for the last second."""
    return _update(objects)[1]


def distance(objects) -> float:
    """The loose hub's distance from its line plus the spider's distance from its seat between the fixed hub's jaws."""
    d = abs(PARK * MM - hub_travel(objects))
    if "spider" in objects:
        seat = np.array(ORIGIN) + np.array([(JAW - SPIDER_W / 2) * MM, 0.0, 0.0])
        d += float(np.linalg.norm(objects["spider"].pos - seat))
    else:
        d += 1.0
    return float(d)


def score(objects) -> float:
    """End state. 0.25: the spider sits between the fixed hub's jaws (its centre within 1.5 mm of the shafts' axis,
    within 5 degrees of square, its near face no more than 1 mm proud of the jaw tips) with the grippers 3 cm clear
    of it; 0.5: that, and the loose hub's jaw tips are 3 mm or more past the spider's near face, in its free gaps;
    0.75: that, and the loose hub's rear face is within 1 mm of the scribed line; 1.0: that, with both grippers 3 cm
    clear of the spider, the loose hub and the driven shaft, and nothing moving more than 1 mm for 1 second. 0 in
    every other state, including the two hubs closed on each other with the spider left out."""
    return _update(objects)[0]
