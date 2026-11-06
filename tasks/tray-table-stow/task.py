"""On the station table, in front of the arms: a fixed grey seatback (30 x 300 x 240 mm, tilted 10 degrees with its top
toward the arms, as if reclined) and, hinged at its foot 5 cm up, a light wooden tray table (110 x 240 x 12 mm, 0.2 kg)
that lies flat toward the arms. A small tongue (42 x 30 mm) sticks out of the tray's free edge for a gripper to hold.
A weak spring and the lean of the seat pull the tray open, so a folded tray falls back down unless something holds it.
High on the seatback, to the left, an orange latch tab on a boss swings about the x axis: it rests flat pointing left,
and when flipped down past 45 degrees it snaps the rest of the way and hangs in front of the tray's top edge.

The robot must fold the tray up against the seatback with one arm and hold it there, flip the latch tab down over the
tray's top edge with the other arm (flipping it first would block the tray from closing), and then let go of the tray.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Hinge, PartState, on_reset, on_step, part, set_hinge_stiffness, set_hinge_target  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Fold the tray table up against the seat back and hold it there while you flip the latch tab down "
               "over its top edge, then let go.")

GREY = (0.45, 0.46, 0.50)
WOOD = (0.86, 0.70, 0.45)
ORANGE = (1.00, 0.50, 0.00)
DARK = (0.22, 0.23, 0.25)


def _mm(x0, x1, y0, y1, z0, z1, color, rot=None):
    spec = (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                     (z1 - z0) / 1000), color)
    return spec if rot is None else (*spec, rot)


# -- the seat part: origin on the table under the tray's hinge line; x toward the seatback, y left, z up. ------------
SEAT_X = 0.28                       # world x of the origin
TILT = math.radians(10.0)           # the seatback leans with its top toward the arms
HINGE_Z = 0.05                      # tray hinge height
HINGE_Y = -0.02
TRAY_L, TRAY_W, TRAY_T = 0.110, 0.240, 0.012
TONGUE_L, TONGUE_W = 0.042, 0.030
TONGUE_Y = -0.08
TRAY_MASS = 0.20
TRAY_UP = math.radians(81.5)        # the tray's upper stop (the seatback's face is a little before it)
STOWED = math.radians(80.0)

TAB_PIVOT = (-0.0415, 0.070, 0.180)  # the tab's hinge line runs along x through here
TAB_X = 0.018                        # tab width along x
TAB_L = 0.050                        # tab length from the pivot (pointing +y at rest)
TAB_T = 0.015
TAB_MASS = 0.03
TAB_DOWN = math.radians(93.0)        # lower stop of the tab (hangs a little past vertical)

SEAT_MASS = 6.0

# The seatback's front face: through the tray's back face at the stowed angle, 2 mm clear of it.
_n = np.array([math.cos(math.radians(80.0)), math.sin(math.radians(80.0))])   # tray direction toward the free end
_back = np.array([0.0, HINGE_Z]) + 0.0
_tray_dir = np.array([-math.cos(STOWED), math.sin(STOWED)])
_face_normal_to_seat = np.array([math.sin(STOWED), math.cos(STOWED)])         # the tray's upper (back) face normal
FACE_POINT = np.array([0.0, HINGE_Z]) + (TRAY_T / 2 + 0.002) * _face_normal_to_seat   # (x rel, z) on the face
_up = np.array([-math.sin(TILT), math.cos(TILT)])                             # along the seatback, upward
_into = np.array([math.cos(TILT), math.sin(TILT)])                            # into the seatback
_s0, _s1 = -0.05, 0.19                                                        # extent along the plane from FACE_POINT


def _seat_box(s0, s1, y0, y1):
    """A slab of the seatback between heights s0 and s1 along its plane and y0..y1 (m)."""
    mid = FACE_POINT + _up * (s0 + s1) / 2 + _into * 0.015
    return ((float(mid[0]), (y0 + y1) / 2, float(mid[1])), (0.030, y1 - y0, s1 - s0), GREY,
            (math.cos(-TILT / 2), 0.0, math.sin(-TILT / 2), 0.0))


# The seatback is full height only at the left (where the latch tab sits) and cut down to just above the tray's hinge
# on the right, so the gripper holding the tray tongue and the arm above it clear the seat top.
SEAT_LOW = _seat_box(_s0, 0.09, -0.15, 0.025)
SEAT_HIGH = _seat_box(_s0, 0.155, 0.025, 0.15)


def face_x(z: float) -> float:
    """x (relative to the seat origin) of the seatback's front face at height z."""
    return float(FACE_POINT[0] - math.tan(TILT) * (z - FACE_POINT[1]))


_BOSS_X1 = face_x(TAB_PIVOT[2]) + 0.012                                       # buried in the seatback
SEAT_BOXES = [
    SEAT_LOW,
    SEAT_HIGH,
    _mm(-5, 90, -150, 150, 0, 10, DARK),                                       # foot plate under the seat
    _mm(1000 * (TAB_PIVOT[0] + TAB_X / 2 + 0.0015), 1000 * _BOSS_X1, 1000 * (TAB_PIVOT[1] - 0.012),
        1000 * (TAB_PIVOT[1] + 0.012), 1000 * (TAB_PIVOT[2] - 0.012), 1000 * (TAB_PIVOT[2] + 0.012), DARK),   # boss
]

TRAY = Hinge(
    "tray",
    pivot=(0.0, HINGE_Y, HINGE_Z),
    boxes=(
        _mm(-1000 * TRAY_L, 0, 1000 * (HINGE_Y - TRAY_W / 2), 1000 * (HINGE_Y + TRAY_W / 2),
            1000 * (HINGE_Z - TRAY_T / 2), 1000 * (HINGE_Z + TRAY_T / 2), WOOD),
        _mm(-1000 * (TRAY_L + TONGUE_L) + 1, -1000 * TRAY_L + 1, 1000 * (TONGUE_Y - TONGUE_W / 2),
            1000 * (TONGUE_Y + TONGUE_W / 2), 1000 * (HINGE_Z - TRAY_T / 2), 1000 * (HINGE_Z + TRAY_T / 2), WOOD),
    ),
    lower=0.0, upper=TRAY_UP, stiffness=0.12, damping=0.03, target=0.0, mass=TRAY_MASS, axis="Y")

TAB = Hinge(
    "tab",
    pivot=TAB_PIVOT,
    boxes=(
        _mm(1000 * (TAB_PIVOT[0] - TAB_X / 2), 1000 * (TAB_PIVOT[0] + TAB_X / 2), 1000 * (TAB_PIVOT[1] - 0.0075),
            1000 * (TAB_PIVOT[1] + TAB_L), 1000 * (TAB_PIVOT[2] - TAB_T / 2), 1000 * (TAB_PIVOT[2] + TAB_T / 2),
            ORANGE),
    ),
    lower=-TAB_DOWN, upper=0.0, stiffness=0.06, damping=0.02, target=0.1, mass=TAB_MASS, axis="X")

SEAT = part("seat", pos=(SEAT_X, 0.0, TOP), mass=SEAT_MASS, boxes=SEAT_BOXES, hinges=(TRAY, TAB), color=GREY,
            friction=1.0)
OBJECTS = [SEAT]

# -- scoring thresholds ------------------------------------------------------------------------------------------------
TRAY_OK = math.radians(75.0)
TAB_OK = math.radians(70.0)
DETENT = math.radians(45.0)         # the tab snaps the rest of the way past this
FAR = 0.03                          # grippers at least 3 cm from the tray
SEAT_MOVE_LIMIT = 0.03
SEAT_TURN_LIMIT = 10.0
LOCK_K = 1.0                        # N m/rad: the latched tray's drive, far stronger than gravity and the weak open spring
STILL_CALLS = 20                    # 1 s at 20 Hz
STILL_MM = 0.001                    # the tray's free edge and the tab's tip move less than 1 mm over the window

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(pos0=None, yaw0=None, failed=False, reached=False, held=None, fingers_ok=None, tab_target=None,
                  tray_target=None, latched=False, locked=False, recent=[])
    if environment is not None:    # a new episode: the tray's drive is the weak open spring again
        try:
            set_hinge_stiffness(environment, "seat", "tray", TRAY.stiffness)
            set_hinge_target(environment, "seat", "tray", 0.0)
        except Exception:  # noqa: BLE001
            pass


_reset_state()


def tray_angle(seat: PartState) -> float:
    return float(seat.hinges.get("tray", 0.0))


def tab_angle(seat: PartState) -> float:
    """How far (rad, positive) the tab has been flipped down."""
    return float(-seat.hinges.get("tab", 0.0))


def tab_angle_signed(seat: PartState) -> float:
    return float(seat.hinges.get("tab", 0.0))


def _yaw(p: PartState) -> float:
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def tray_distance(seat: PartState, point) -> float:
    """Distance (m) from a world point to the tray's slab or tongue at the tray's current angle."""
    c = np.asarray(seat.to_local(point), float) - np.array([0.0, HINGE_Y, HINGE_Z])
    th = tray_angle(seat)
    # undo the tray's turn about +y: R_y(-th)
    cs, sn = math.cos(th), math.sin(th)
    local = np.array([cs * c[0] - sn * c[2], c[1], sn * c[0] + cs * c[2]])
    best = 1e9
    for x0, x1, y0, y1 in ((-TRAY_L, 0.0, -TRAY_W / 2, TRAY_W / 2),
                           (-TRAY_L - TONGUE_L, -TRAY_L, TONGUE_Y - HINGE_Y - TONGUE_W / 2,
                            TONGUE_Y - HINGE_Y + TONGUE_W / 2)):
        lo, hi = np.array([x0, y0, -TRAY_T / 2]), np.array([x1, y1, TRAY_T / 2])
        d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
        best = min(best, float(np.linalg.norm(d)))
    return best


def finger_points(robot, origin) -> list[np.ndarray]:
    """Points on both finger pads of one arm (world, relative to the env origin): at the TCP and at the pad tips, on
    each finger's line, 6 mm outside its pad face."""
    from yam_arena.constants import PAD_TIP_PAST_TCP, TCP_OFFSET

    names = list(robot.body_names)
    pose = robot.data.body_link_pose_w
    pose = getattr(pose, "torch", pose)[0].detach().cpu().numpy().astype(float)[names.index("gripper")]
    jp = robot.data.joint_pos
    jp = getattr(jp, "torch", jp)[0].detach().cpu().numpy().astype(float)
    jn = list(robot.joint_names)
    half_gap = max(abs(jp[jn.index(j)]) for j in ("joint7", "joint8") if j in jn)
    x, y, z, w = pose[3:7]
    r = PartState(pos=pose[:3], quat_xyzw=np.array([x, y, z, w]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                  low=np.zeros(3), high=np.zeros(3)).rotation()
    pts = []
    for along in (0.0, PAD_TIP_PAST_TCP):
        centre = pose[:3] + r @ (np.array(TCP_OFFSET) + [0.0, 0.0, along]) - origin
        for side in (1.0, -1.0):
            pts.append(centre + r[:, 0] * side * (half_gap + 0.006))
    return pts


def holding(points, seat: PartState) -> bool:
    return any(tray_distance(seat, p) < FAR for p in points)


@on_step
def _mechanism(environment) -> None:
    """Latch logic, and the robot's own state (finger positions are not available to score())."""
    try:
        from yam_arena.scene import object_states

        base = environment.unwrapped
        seat = object_states(environment, ["seat"])["seat"]
        tab, tray = tab_angle(seat), tray_angle(seat)
        # Snap-over detent: past 45 degrees the tab's spring drives it down; before, it drives it back up.
        want_tab = -TAB_DOWN if tab >= DETENT else 0.1
        if _STATE["tab_target"] != want_tab:
            set_hinge_target(environment, "seat", "tab", want_tab)
            _STATE["tab_target"] = want_tab
        # Latched: tray stowed and tab down over its edge -> the tray's spring aims at the stowed angle.
        _STATE["latched"] |= tray >= TRAY_OK and tab >= TAB_OK     # once latched, the tab keeps the tray
        if _STATE["latched"] and not _STATE["locked"]:
            # Latched: the tray is clamped at the stowed angle (a firm drive aimed at it), however the arms let go.
            set_hinge_stiffness(environment, "seat", "tray", LOCK_K)
            set_hinge_target(environment, "seat", "tray", STOWED)
            _STATE["locked"] = True
        origin = getattr(base.scene.env_origins, "torch", base.scene.env_origins)[0].detach().cpu().numpy()
        pts = [p for arm in ("left_robot", "right_robot") for p in finger_points(base.scene[arm], origin)]
        _STATE["held"] = holding(pts, seat)
        _STATE["fingers_ok"] = True
    except Exception as exc:  # noqa: BLE001 - report once, keep the episode running
        if _STATE["fingers_ok"] is None:
            print(f"tray-table-stow: mechanism/finger check unavailable: {exc!r}", flush=True)
        _STATE["fingers_ok"] = False
        _STATE["held"] = None


def _update(objects) -> tuple[float, bool]:
    seat = objects["seat"]
    if _STATE["pos0"] is None:
        _STATE.update(pos0=seat.pos.copy(), yaw0=_yaw(seat))
    moved = float(np.linalg.norm((seat.pos - _STATE["pos0"])[:2]))
    turn = abs(math.degrees((_yaw(seat) - _STATE["yaw0"] + math.pi) % (2 * math.pi) - math.pi))
    _STATE["failed"] |= moved > SEAT_MOVE_LIMIT or turn > SEAT_TURN_LIMIT
    if _STATE["failed"]:
        return 0.0, False
    tray, tab = tray_angle(seat), tab_angle(seat)
    _STATE["reached"] |= tray >= TRAY_OK
    recent = _STATE["recent"]
    recent.append(np.array([tray * TRAY_L, tab * TAB_L]))
    del recent[:-STILL_CALLS]
    still = len(recent) == STILL_CALLS and float(np.ptp(np.array(recent), axis=0).max()) < STILL_MM
    if tray >= TRAY_OK and tab >= TAB_OK:
        if not _STATE["held"]:   # fingers unknown (check unavailable) count as released: the check only lowers a score
            return 1.0, bool(still)
        return 0.75, False
    if tray >= TRAY_OK:
        return 0.5, False
    return (0.25 if _STATE["reached"] else 0.0), False


def success(objects) -> bool:
    """At the end the tray is folded to 75 degrees or more, the tab is flipped down 70 degrees or more, no gripper is
    within 3 cm of the tray, and the tray and tab were still for 20 scorer calls."""
    return _update(objects)[1]


def score(objects) -> float:
    """1.0 tray >= 75 deg and tab >= 70 deg at the end and no gripper within 3 cm of the tray; 0.75 the same but a
    gripper is still holding the tray; 0.5 tray >= 75 deg but tab < 70 deg (the arm is still holding it); 0.25 the tray
    reached 75 deg at some point but ends lower; 0 the tray never reached 75 deg (this includes flipping the tab
    first), or the seat moved 3 cm or turned 10 degrees."""
    return _update(objects)[0]


def distance(objects) -> float:
    """Metres to go: the tray's free edge to its stowed angle plus the tab's tip to its flipped angle."""
    seat = objects["seat"]
    return float(TRAY_L * max(0.0, STOWED - tray_angle(seat)) + TAB_L * max(0.0, math.radians(90.0) - tab_angle(seat)))
