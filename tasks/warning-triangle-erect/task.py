"""On the station table: one folded highway warning triangle (FMVSS 125, Deflecto 73-0711-00 class) standing on
its weighted black base inside a yellow outline. Its three orange bars lie face to face in one pack along the base,
pointing away from the robot: the bottom bar, and the two side arms hinged at its ends and folded down along it.
The pack turns on the base about a vertical swivel between stops a quarter turn apart. Each arm swings up in the
triangle's own plane on a loose rivet and falls back unless it is held; when both are up, a round pin near the tip
of one arm snaps into a slot near the tip of the other (the bars spring apart about 4 mm to let it in) and the
triangle holds itself. The robot must leave it set up: arms joined, turned across its base, standing on the mark.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Hinge, PartState, finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]
MM = 0.001

INSTRUCTION = "Set up the warning triangle so that it stands on its base on the marked spot."

BLACK = (0.06, 0.06, 0.06)
RED = (0.80, 0.05, 0.03)
ORANGE = (1.00, 0.36, 0.00)
STEEL = (0.62, 0.63, 0.65)
YELLOW = (0.95, 0.80, 0.10)

# -- the triangle (mm). Published by the maker: 17.25 in across and 16.55 in tall set up, 1.25 in thick folded,
#    3.625 lb. Everything else is an estimate from the maker's drawing and photos: measure the bought part.
SIDE = 438.2            # outer side of the triangle (17.25 in, published)
HEIGHT = 420.4          # ground to apex, set up (16.55 in, published)
PACK = 31.8             # folded thickness = the base's width (1.25 in, published)
MASS = 1.644            # one triangle with its base (3.625 lb, published)
BAR_W = 56.0            # width of each bar (FMVSS 125 allows 2 to 3 in; estimate)
BAR_T = 8.0             # thickness of each bar (estimate: three bars and two gaps in the 31.8 mm pack)
LAYER_GAP = 3.9         # between neighbouring bars' faces (three 8 mm bars and two gaps make the 31.8 mm pack)
RED_W = 23.0            # outer red reflector strip of each bar; the rest of the width is fluorescent orange
EDGE_Z = HEIGHT - SIDE * math.sqrt(3) / 2          # the bottom bar's lower edge above the ground: 40.9
PIVOT_Z = EDGE_Z + BAR_W / 2                       # the arm rivets, on the bars' centre lines: 68.9
ARM = SIDE - BAR_W * math.sqrt(3)                  # rivet to rivet, and rivet to the apex point: 341.2
HALF = ARM / 2
TIP = 40.0              # a bar runs this far past the apex point (and the bottom bar past each rivet)
TAIL = 8.0              # an arm runs this far behind its rivet
BASE_L, BASE_H = 356.0, 30.0                        # the weighted base: 356 x 31.8 x 30 mm (length, height: estimates)
BASE_MASS, BAR_MASS, HUB_MASS = 1.30, 0.105, 0.029  # base (steel-weighted), each bar, swivel hub and feet (estimates)
FOOT_X, FOOT_GAP = 190.0, 2.0                       # the bottom bar's two corner feet end 2 mm above the ground
PIN = 8.0               # the left arm's round pin: 8 mm diameter (estimate), 7.9 mm long. The slot is sized to it.
PIN_SIDE = PIN / math.sqrt(2)   # modelled as the square inscribed in that circle (corners on the true radius: 5.66 mm)
PIN_OUT = 14.0          # the pin stands this far past the apex point of the left arm's centre line ...
PIN_AT = ARM + PIN_OUT  # ... so this far from the left arm's rivet
PIN_LEN = LAYER_GAP + 4.0                           # it enters the right arm's slot by 4 mm
SLOT_S, SLOT_Z = ARM + PIN_OUT / 2, PIN_OUT * math.sqrt(3) / 2   # where the pin sits in the right arm when the
#                         triangle is set up: 348.2 mm along the arm from its rivet, 12.1 mm toward its outer edge
SLOT_PLAY = (0.5, 14.0, 2.0, 0.5)                   # slot beyond the pin: toward the rivet, toward the tip, inward, outward
BEVEL_FROM = 334.0      # from here to its tip the right arm's outer rim is bevelled on the face toward the left arm:
BEVEL_DROP = 6.0        # that much lower at the outer edge, rising to full height 1.2 mm before the slot (a lead-in
BEVEL_LAND = 1.2        # for the pin, which then springs the bars apart by its 4 mm and drops into the slot)
FLEX_K = 20.0           # N m/rad: a bar bending out of the triangle's plane about its rivet end (170 N/m at the tip;
#                         a ribbed 56 x 8 mm ABS section is 20 to 50). With the 4 mm pin this gives a snap of about 0.5 N.
FLEX_MAX = math.radians(3.0)
BAR_MU, SNAP_MU = 0.8, 0.3   # friction of the bars' faces (0.65 against the gripper's pads, which are 0.5) and of the pin
#                              and the slot's rim against each other (moulded ABS on ABS)
RIVET_FRICTION = 0.004  # N m: the arms' loose rivets (an arm's own weight gives 0.17 N m, so it falls back)
SWIVEL_FRICTION = 0.08  # N m: the swivel hub (estimate; 0.5 N at a bar's end turns it)
TURN_STOPS = (0.0, math.radians(90.0))   # the swivel's stops: folded along the base, and a quarter turn across it
ARM_STOP = math.radians(110.0)

# layers across the pack (part y, at swivel angle 0): left arm in front (+y), right arm in the middle, bottom bar behind
Y_LEFT = (BAR_T / 2 + LAYER_GAP, BAR_T / 2 + LAYER_GAP + BAR_T)        # 5.5 .. 13.5
Y_RIGHT = (-BAR_T / 2, BAR_T / 2)                                       # -4 .. 4
Y_BOTTOM = (-BAR_T / 2 - LAYER_GAP - BAR_T, -BAR_T / 2 - LAYER_GAP)     # -13.5 .. -5.5
Z0, Z1 = EDGE_Z, EDGE_Z + BAR_W                                         # a folded bar's lower and upper edges
P_LEFT = np.array([HALF, (Y_LEFT[0] + Y_LEFT[1]) / 2, PIVOT_Z]) * MM    # the left arm's rivet (part frame)
P_RIGHT = np.array([-HALF, 0.0, PIVOT_Z]) * MM                          # the right arm's rivet
PIN_END = np.array([HALF - PIN_AT, Y_LEFT[0] - PIN_LEN, PIVOT_Z]) * MM  # centre of the pin's end face (folded)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000),
            ((x1 - x0) / 1000, (y1 - y0) / 1000, (z1 - z0) / 1000), color)


def _quat_wxyz(R) -> tuple:
    """Shepperd's method (stable near half turns)."""
    R = np.asarray(R, float)
    t = [R[0, 0] + R[1, 1] + R[2, 2], R[0, 0], R[1, 1], R[2, 2]]
    i = int(np.argmax(t))
    if i == 0:
        q = np.array([1 + t[0], R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    elif i == 1:
        q = np.array([R[2, 1] - R[1, 2], 1 + 2 * R[0, 0] - t[0], R[0, 1] + R[1, 0], R[0, 2] + R[2, 0]])
    elif i == 2:
        q = np.array([R[0, 2] - R[2, 0], R[0, 1] + R[1, 0], 1 + 2 * R[1, 1] - t[0], R[1, 2] + R[2, 1]])
    else:
        q = np.array([R[1, 0] - R[0, 1], R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], 1 + 2 * R[2, 2] - t[0]])
    q = q / np.linalg.norm(q)
    return tuple(float(v) for v in (q if q[0] >= 0 else -q))


def _rot_y(deg: float) -> tuple:
    """(w, x, y, z) of a turn about the part's y axis."""
    return (math.cos(math.radians(deg) / 2), 0.0, math.sin(math.radians(deg) / 2), 0.0)


def _rim_bevel(s0, s1, c0, c1):
    """The lead-in on the right arm's outer rim (along the arm s0..s1, across it c0..c1 from the centre line, mm): a
    thin box whose upper face slopes from BEVEL_DROP below the arm's front face at the outer edge c1 up to that face
    at c0."""
    drop, d = BEVEL_DROP, 1.7
    th = math.atan2(drop, c1 - c0)
    u = np.array([0.0, -math.sin(th), math.cos(th)])          # up the slope: toward the outer edge and away from the left arm
    n = np.array([0.0, -math.cos(th), -math.sin(th)])         # into the material
    A = np.array([0.0, Y_RIGHT[1], PIVOT_Z + c0])
    B = np.array([0.0, Y_RIGHT[1] - drop, PIVOT_Z + c1])
    centre = (A + B) / 2 + (d / 2) * n + np.array([-HALF + (s0 + s1) / 2, 0.0, 0.0])
    R = np.c_[np.array([1.0, 0.0, 0.0]), np.cross(u, [1.0, 0.0, 0.0]), u]
    return (tuple(float(v) for v in centre * MM), ((s1 - s0) * MM, d * MM, float(np.linalg.norm(B - A)) * MM), RED,
            _quat_wxyz(R))


def _bar(x0, x1, y, outer_up=True):
    """A folded bar: its outer strip red, the rest orange. ``outer_up``: the outer edge is the upper one."""
    zr = (Z1 - RED_W, Z1) if outer_up else (Z0, Z0 + RED_W)
    zo = (Z0, Z1 - RED_W) if outer_up else (Z0 + RED_W, Z1)
    return [_mm(x0, x1, *y, *zo, ORANGE), _mm(x0, x1, *y, *zr, RED)]


BASE_BOXES = [_mm(-BASE_L / 2, BASE_L / 2, -PACK / 2, PACK / 2, 0, BASE_H, BLACK)]
BOTTOM_BOXES = [*_bar(-HALF - TIP, HALF + TIP, Y_BOTTOM, outer_up=False),
                _mm(-15, 15, -15, 15, BASE_H + 0.5, EDGE_Z - 1.0, BLACK),                         # the swivel hub
                *(_mm(s * FOOT_X - 5, s * FOOT_X + 5, *Y_BOTTOM, FOOT_GAP, Z0, ORANGE) for s in (-1, 1))]   # two feet
LEFT_BOXES = [*_bar(HALF - ARM - TIP, HALF + TAIL, Y_LEFT),
              # the pin: the inscribed square, turned 30 degrees so that it lies square in the right arm's slot
              (((HALF - PIN_AT) * MM, (Y_LEFT[0] - PIN_LEN / 2) * MM, PIVOT_Z * MM),
               (PIN_SIDE * MM, PIN_LEN * MM, PIN_SIDE * MM), STEEL, _rot_y(-30.0))]
# the right arm, with its slot: along the arm s = x + HALF from the rivet, across it z - PIVOT_Z toward the outer edge
_S0, _S1 = SLOT_S - PIN / 2 - SLOT_PLAY[0], SLOT_S + PIN / 2 + SLOT_PLAY[1]
_C0, _C1 = SLOT_Z - PIN / 2 - SLOT_PLAY[2], SLOT_Z + PIN / 2 + SLOT_PLAY[3]
SLOT = (_S0, _S1, _C0, _C1)
_RIM0, _END = BEVEL_FROM, ARM + TIP
_YB = (Y_RIGHT[0], Y_RIGHT[1] - BEVEL_DROP)                        # the rim's solid back part, behind the bevel
RIGHT_BOXES = [*_bar(-HALF - TAIL, -HALF + _RIM0, Y_RIGHT),
               _mm(-HALF + _RIM0, -HALF + _END, *Y_RIGHT, Z0, PIVOT_Z + _C0, ORANGE),                    # inner side
               _mm(-HALF + _RIM0, -HALF + _S0, *Y_RIGHT, PIVOT_Z + _C0, PIVOT_Z + _C1, ORANGE),          # before the slot
               _mm(-HALF + _S1, -HALF + _END, *Y_RIGHT, PIVOT_Z + _C0, PIVOT_Z + _C1, ORANGE),           # beyond the slot
               _mm(-HALF + _RIM0, -HALF + _END, *_YB, PIVOT_Z + _C1, Z1, RED),                           # outer rim, back
               _mm(-HALF + _RIM0, -HALF + _END, _YB[1], Y_RIGHT[1], PIVOT_Z + _C1, PIVOT_Z + _C1 + BEVEL_LAND, RED),
               _rim_bevel(_RIM0, _END, _C1 + BEVEL_LAND, BAR_W / 2)]

X0 = 0.30                                   # the swivel axis, on the midline
TRIANGLE = part(
    "triangle", pos=(X0, 0.0, TOP), mass=BASE_MASS, boxes=BASE_BOXES, color=BLACK, friction=BAR_MU,
    hinges=(
        Hinge("turn", pivot=(0.0, 0.0, BASE_H * MM), boxes=tuple(BOTTOM_BOXES), lower=TURN_STOPS[0],
              upper=TURN_STOPS[1], stiffness=0.0, damping=50.0, effort=SWIVEL_FRICTION, mass=BAR_MASS + HUB_MASS,
              axis="Z"),
        Hinge("arm_l", pivot=tuple(P_LEFT), lower=0.0, upper=ARM_STOP, stiffness=0.0, damping=0.2,
              effort=RIVET_FRICTION, mass=0.003, axis="Y", parent="turn"),
        Hinge("flex_l", pivot=tuple(P_LEFT), boxes=tuple(LEFT_BOXES), lower=-FLEX_MAX, upper=FLEX_MAX,
              stiffness=FLEX_K, damping=0.3, mass=BAR_MASS - 0.003, axis="Z", parent="arm_l"),
        Hinge("arm_r", pivot=tuple(P_RIGHT), lower=-ARM_STOP, upper=0.0, stiffness=0.0, damping=0.2,
              effort=RIVET_FRICTION, mass=0.003, axis="Y", parent="turn"),
        Hinge("flex_r", pivot=tuple(P_RIGHT), boxes=tuple(RIGHT_BOXES), lower=-FLEX_MAX, upper=FLEX_MAX,
              stiffness=FLEX_K, damping=0.3, mass=BAR_MASS - 0.003, axis="Z", parent="arm_r"),
    ),
    # the pin (left arm, box 2) and the right arm's tip round the slot (boxes 2 to 7) are the moulded snap pair
    meta={"self_collide": True,
          "box_friction": {"flex_l": {2: SNAP_MU}, "flex_r": {k: SNAP_MU for k in range(2, 8)}}})

# the marked spot: a yellow outline round the base's footprint, 0.5 mm thick tape
MARK_L, MARK_W, TAPE = 0.376, 0.052, 0.006
MARKS = [Box(f"mark_{k}", pos=(X0 + dx, dy, TOP + 0.00025), size=(sx, sy, 0.0005), color=YELLOW)
         for k, (dx, dy, sx, sy) in enumerate(((0.0, MARK_W / 2, MARK_L + TAPE, TAPE), (0.0, -MARK_W / 2, MARK_L + TAPE, TAPE),
                                                (MARK_L / 2, 0.0, TAPE, MARK_W - TAPE), (-MARK_L / 2, 0.0, TAPE, MARK_W - TAPE)))]
OBJECTS = [*MARKS, TRIANGLE]

# -- kinematics of the triangle, for scoring ---------------------------------------------------------------------------------


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def angles(tri: PartState) -> dict:
    """Swivel turn, how far each arm is raised from folded, and each bar's bend out of plane (rad)."""
    h = tri.hinges
    return {"turn": float(h.get("turn", 0.0)), "left": float(h.get("arm_l", 0.0)), "right": -float(h.get("arm_r", 0.0)),
            "flex_l": float(h.get("flex_l", 0.0)), "flex_r": float(h.get("flex_r", 0.0))}


class Frames:
    """Where the triangle's bars are: maps points given in the folded part frame (m) to the world, per bar."""

    def __init__(self, tri: PartState):
        h = tri.hinges
        self.tri = tri
        self.Rs = _rz(float(h.get("turn", 0.0)))
        self.Rl = _ry(float(h.get("arm_l", 0.0))) @ _rz(float(h.get("flex_l", 0.0)))
        self.Rr = _ry(float(h.get("arm_r", 0.0))) @ _rz(float(h.get("flex_r", 0.0)))

    def base(self, q):
        return self.tri.to_world(np.asarray(q, float))

    def bottom(self, q):
        return self.tri.to_world(self.Rs @ np.asarray(q, float))

    def left(self, q):
        return self.tri.to_world(self.Rs @ (P_LEFT + self.Rl @ (np.asarray(q, float) - P_LEFT)))

    def right(self, q):
        return self.tri.to_world(self.Rs @ (P_RIGHT + self.Rr @ (np.asarray(q, float) - P_RIGHT)))

    def in_right(self, world):
        """A world point in the right arm's own folded coordinates (part frame, m)."""
        s = self.Rs.T @ self.tri.to_local(np.asarray(world, float))
        return P_RIGHT + self.Rr.T @ (s - P_RIGHT)

    def local(self, bar: str, world):
        """A world point in the folded part frame of ``bar`` ('base', 'bottom', 'left' or 'right')."""
        p = self.tri.to_local(np.asarray(world, float))
        if bar == "base":
            return p
        s = self.Rs.T @ p
        if bar == "bottom":
            return s
        if bar == "left":
            return P_LEFT + self.Rl.T @ (s - P_LEFT)
        return P_RIGHT + self.Rr.T @ (s - P_RIGHT)


def pin_in_slot(tri: PartState) -> dict:
    """Where the left arm's pin is in the right arm: along the arm and across it from its seat, and how far its end
    has entered the slot (all mm)."""
    f = Frames(tri)
    q = f.in_right(f.left(PIN_END)) / MM
    return {"along": float(q[0] + HALF - SLOT_S), "across": float(q[2] - PIVOT_Z - SLOT_Z),
            "depth": float(Y_RIGHT[1] - q[1])}


# solid boxes for the "grippers clear" check: (bar, lower corner, upper corner) in each bar's folded frame (mm)
_SOLIDS = (("base", (-BASE_L / 2, -PACK / 2, 0.0), (BASE_L / 2, PACK / 2, BASE_H)),
           ("bottom", (-HALF - TIP, Y_BOTTOM[0], FOOT_GAP), (HALF + TIP, Y_BOTTOM[1], Z1)),
           ("left", (HALF - ARM - TIP, Y_LEFT[0] - PIN_LEN, Z0), (HALF + TAIL, Y_LEFT[1], Z1)),
           ("right", (-HALF - TAIL, Y_RIGHT[0], Z0), (-HALF + ARM + TIP, Y_RIGHT[1], Z1)))


def clearance(tri: PartState, points) -> float:
    """Smallest distance (m) from any of ``points`` (world) to the base or a bar."""
    f = Frames(tri)
    best = float("inf")
    for p in points:
        for bar, lo, hi in _SOLIDS:
            q = f.local(bar, p) / MM
            d = np.maximum(np.maximum(np.array(lo) - q, q - np.array(hi)), 0.0)
            best = min(best, float(np.linalg.norm(d)) * MM)
    return best


# -- scoring ---------------------------------------------------------------------------------------------------------------
RAISED = math.radians(45.0)      # an arm counts as raised past this
SEAT_TOL = 3.0                   # mm: the pin within 3 mm of its seat in the slot, along and across the arm
SEAT_DEPTH = 2.0                 # mm: and at least 2 mm into the slot
TURN_LO, TURN_HI = math.radians(80.0), math.radians(100.0)
PLUMB = math.radians(5.0)        # the triangle's plane within 5 degrees of vertical
RELEASE = 0.030                  # grippers 3 cm clear of the triangle
MOVE = 0.050                     # the base more than 5 cm from its mark
TIPPED = math.radians(30.0)      # the base tilted more than 30 degrees: the triangle went over
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), home=None, moved=False, tipped=False, clear=False,
                  finger_check=None, recent=[], report={})


_reset_state()


def status(tri: PartState) -> dict:
    a = angles(tri)
    pin = pin_in_slot(tri)
    f = Frames(tri)
    normal = tri.rotation() @ f.Rs @ np.array([0.0, 1.0, 0.0])          # the triangle's plane normal
    return {
        "turn": a["turn"], "left": a["left"], "right": a["right"], "pin": pin,
        "joined": bool(abs(pin["along"]) <= SEAT_TOL and abs(pin["across"]) <= SEAT_TOL and pin["depth"] >= SEAT_DEPTH),
        "raised": int(a["left"] >= RAISED) + int(a["right"] >= RAISED),
        "turned": bool(TURN_LO <= abs(a["turn"]) <= TURN_HI),
        "lean": float(math.asin(min(1.0, abs(float(normal[2]))))),
        "base_tilt": float(math.acos(max(-1.0, min(1.0, float(tri.rotation()[2, 2]))))),
    }


@on_step
def _step(environment) -> None:
    """Every control step: count steps and check the grippers (score() cannot see the robot)."""
    from yam_arena.scene import object_states

    _STATE["step"] += 1
    try:
        tri = object_states(environment, ["triangle"])["triangle"]
        _STATE["clear"] = bool(clearance(tri, finger_points(environment)) >= RELEASE)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"warning-triangle-erect: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"] = False
        _STATE["clear"] = False


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    tri = objects["triangle"]
    if _STATE["home"] is None:
        _STATE["home"] = tri.pos.copy()
    st = status(tri)
    _STATE["moved"] |= float(np.linalg.norm((tri.pos - _STATE["home"])[:2])) > MOVE              # permanent
    _STATE["tipped"] |= st["base_tilt"] > TIPPED                                                  # permanent
    f = Frames(tri)
    recent = _STATE["recent"]
    recent.append(np.concatenate([tri.pos, f.left(PIN_END), f.right(P_RIGHT + [ARM * MM, 0.0, 0.0]),
                                  f.bottom([HALF * MM, 0.0, PIVOT_Z * MM])]))
    del recent[:-20]
    still = len(recent) == 20 and all(float(np.abs(r - recent[-1]).max()) < 0.001 for r in recent)
    clear = bool(_STATE["clear"])
    joined = st["joined"] and clear
    standing = joined and st["turned"] and st["lean"] <= PLUMB
    if _STATE["moved"] or _STATE["tipped"]:
        result = (0.0, False)
    elif standing and still:
        result = (1.0, True)
    elif standing:
        result = (0.75, False)
    elif joined:
        result = (0.5, False)
    elif st["raised"] >= 1 or st["turned"]:
        result = (0.25, False)
    else:
        result = (0.0, False)
    _STATE["report"] = {"joined": st["joined"], "clear": clear, "turned": st["turned"], "raised": st["raised"],
                        "still": still, "turn_deg": math.degrees(st["turn"]), "left_deg": math.degrees(st["left"]),
                        "right_deg": math.degrees(st["right"]), "pin_mm": st["pin"], "lean_deg": math.degrees(st["lean"])}
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Both arms joined at the apex (pin in the slot), the triangle turned 80 to 100 degrees across its base and
    within 5 degrees of vertical, both grippers 3 cm clear of it, and nothing moving more than 1 mm for 1 second."""
    return _update(objects)[1]


def distance(objects) -> float:
    """The pin's distance from its seat in the slot (m) plus 0.2 m per radian the swivel is short of a quarter turn."""
    st = status(objects["triangle"])
    pin = st["pin"]
    gap = math.sqrt(pin["along"] ** 2 + pin["across"] ** 2 + max(0.0, 4.0 - pin["depth"]) ** 2) * MM
    return float(gap + 0.2 * abs(math.pi / 2 - abs(st["turn"])))


def score(objects) -> float:
    """End state. 0.25: one arm is raised more than 45 degrees from folded, or the triangle is turned 80 to 100
    degrees on its base. 0.5: both arms are joined at the apex (the pin at least 2 mm into the slot and within 3 mm of
    its seat) with both grippers 3 cm clear of the triangle. 0.75: joined and clear, turned 80 to 100 degrees across
    the base, and the triangle's plane within 5 degrees of vertical. 1.0: that state with nothing moving more than
    1 mm for 1 second. 0 for good if at any point the base was moved more than 5 cm from its mark or was tilted more
    than 30 degrees (the triangle went over)."""
    return _update(objects)[0]
