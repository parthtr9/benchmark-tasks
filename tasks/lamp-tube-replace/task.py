"""On the station table: an 18-inch two-pin lamp tube resting in two low cradles, and two fixed lamp sockets further
out, 46 cm apart and facing each other. Each socket has a narrow slot from its top down to a pocket. The robot must lift the
tube with both arms, lower it between the sockets so the two pins on each end go down the slots into the pockets, and
turn the tube a quarter turn so the pins lie across the slots and lock it in.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, Slider, _np, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Lift the white lamp tube with both hands, lower it between the two grey sockets so the pins on each end "
               "go down the slots, and turn the tube a quarter turn so it locks in.")

WHITE = (0.92, 0.92, 0.90)
GREY = (0.45, 0.46, 0.48)
LIGHT = (0.70, 0.71, 0.73)
STEEL = (0.75, 0.75, 0.78)
ORANGE = (1.00, 0.50, 0.00)

# -- the tube: origin on its axis at its middle; its axis runs along the part's y. An 18-inch tube: a 25.4 mm white
#    glass tube 423 mm long between two 10 mm grey end caps, with two 2.5 mm steel pins 12.7 mm apart sticking 7 mm out
#    of each cap.
#    The pins of each pair lie along the part's z. In each cap, 12 mm above the axis (beside the pins), a spring
#    plunger with a rounded 3 mm nose stands 2 mm proud; it rides on the socket's face and clicks into a hole there
#    when the tube has been turned a quarter turn either way, so the tube stays turned.
TUBE_R = 0.0127
CAP_FACE = 0.2215        # the end caps' outer faces, either side of the middle
PIN_LEN, PIN_R, PIN_DZ = 0.007, 0.00125, 0.00635
TUBE_CYLS = (
    ((0.0, 0.0, 0.0), TUBE_R, 2 * CAP_FACE - 0.020, WHITE, "Y"),
    *(((0.0, s * (CAP_FACE - 0.005), 0.0), TUBE_R, 0.010, GREY, "Y") for s in (1, -1)),
    *(((0.0, s * (CAP_FACE + PIN_LEN / 2), dz), PIN_R, PIN_LEN, STEEL, "Y") for s in (1, -1) for dz in (PIN_DZ, -PIN_DZ)),
)
DETENT_R, DETENT_RATE = 0.012, 3000.0
DETENTS = tuple(Slider(f"detent_{tag}", rest=(0.0, s * (CAP_FACE + 0.0005), DETENT_R),
                       cylinders=(((0.0, s * (CAP_FACE + 0.0005), DETENT_R), 0.0015, 0.003, ORANGE, "Z"),),
                       travel=0.003, rate=DETENT_RATE, damping=2.0, axis="Y", sign=-s) for tag, s in (("l", 1), ("r", -1)))

# -- fixed layout (world frame: x forward from the arms, y left, metres) ----------------------------------------------
CRADLE_X, CRADLE_AXIS, CRADLE_Y = 0.30, 0.050, 0.08    # the tube starts here, its axis 5 cm above the table
SOCKET_X, SOCKET_AXIS = 0.44, 0.100                     # the sockets' pocket centres, 10 cm above the table
AXIS_Z = TOP + SOCKET_AXIS
SOCKET_FACE, POCKET_DEPTH, BACK = CAP_FACE + 0.0015, 0.010, 0.010  # the sockets' inner faces, 1.5 mm beyond the caps
POCKET, SLOT, SOCKET_W, SOCKET_TOP = 0.0085, 0.002, 0.020, SOCKET_AXIS + 0.013  # half-widths; top above the table

TUBE = part("tube", pos=(CRADLE_X, 0.0, TOP + CRADLE_AXIS), mass=0.15, boxes=(((0.0, 0.0, 0.0), (0.01, 0.01, 0.01),
            WHITE),), cylinders=TUBE_CYLS, sliders=DETENTS, color=WHITE, friction=0.8)   # the box is hidden inside the tube


def _box(name, x0, x1, y0, y1, z0, z1, color=LIGHT):
    """A fixed box from its extents (metres; z above the table)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color)


def _socket(tag: str, s: int):
    """One socket, its slot facing the tube: a 4 mm slot from the top down to a 17 mm square pocket 10 mm deep, on the
    tube's axis, and a 5 mm hole 12 mm either side of it for the plunger; a ledge under the end cap carries the tube."""
    X, a, w = SOCKET_X, SOCKET_AXIS, SOCKET_W
    f0, f1 = sorted((s * SOCKET_FACE, s * (SOCKET_FACE + POCKET_DEPTH)))
    b0, b1 = sorted((s * (SOCKET_FACE + POCKET_DEPTH), s * (SOCKET_FACE + POCKET_DEPTH + BACK)))
    l0, l1 = sorted((s * (CAP_FACE - 0.010), s * SOCKET_FACE))
    return [
        _box(f"{tag}_back", X - w, X + w, b0, b1, 0.0, SOCKET_TOP),
        _box(f"{tag}_below", X - w, X + w, f0, f1, 0.0, a - POCKET),
        *(b for side, k in (("n", -1), ("f", 1)) for b in (     # beside the pocket, around the plunger's hole
            _box(f"{tag}_side_{side}_out", *sorted((X + k * w, X + k * (DETENT_R + 0.0025))), f0, f1, a - POCKET,
                 SOCKET_TOP),
            _box(f"{tag}_side_{side}_in", *sorted((X + k * (DETENT_R - 0.0025), X + k * POCKET)), f0, f1, a - POCKET,
                 SOCKET_TOP),
            _box(f"{tag}_side_{side}_lo", *sorted((X + k * (DETENT_R - 0.0025), X + k * (DETENT_R + 0.0025))), f0, f1,
                 a - POCKET, a - 0.0025),
            _box(f"{tag}_side_{side}_hi", *sorted((X + k * (DETENT_R - 0.0025), X + k * (DETENT_R + 0.0025))), f0, f1,
                 a + 0.0025, SOCKET_TOP))),
        _box(f"{tag}_lip_n", X - POCKET, X - SLOT, f0, f1, a + POCKET, SOCKET_TOP),
        _box(f"{tag}_lip_f", X + SLOT, X + POCKET, f0, f1, a + POCKET, SOCKET_TOP),
        _box(f"{tag}_ledge", X - 0.012, X + 0.012, l0, l1, 0.0, a - TUBE_R),
    ]


def _cradle(tag: str, y: float):
    """A fixed block the tube rests on, with a lip either side to keep it from rolling."""
    X, a = CRADLE_X, CRADLE_AXIS
    return [_box(f"{tag}_base", X - 0.015, X + 0.015, y - 0.01, y + 0.01, 0.0, a - TUBE_R, ORANGE),
            _box(f"{tag}_lip_n", X - TUBE_R - 0.006, X - TUBE_R - 0.001, y - 0.01, y + 0.01, 0.0, a, ORANGE),
            _box(f"{tag}_lip_f", X + TUBE_R + 0.001, X + TUBE_R + 0.006, y - 0.01, y + 0.01, 0.0, a, ORANGE)]


OBJECTS = [*_socket("socket_l", 1), *_socket("socket_r", -1), *_cradle("cradle_l", CRADLE_Y),
           *_cradle("cradle_r", -CRADLE_Y), TUBE]

# -- scoring -----------------------------------------------------------------------------------------------------------
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(z0=None, lifted=False, seated=False, turned=False, recent=[], tcps=None)


_reset_state()


def tube_fit(tube: PartState):
    """Each end's axis offset from its socket's pocket centre in x and z (m), the middle's y offset, and how far the
    pins are turned from upright (deg, 0..90)."""
    ends = [tube.to_world((0.0, s * (CAP_FACE - 0.005), 0.0)) for s in (1, -1)]
    off = max(math.hypot(e[0] - SOCKET_X, e[2] - AXIS_Z) for e in ends)
    turn = math.degrees(math.acos(min(1.0, abs(float(tube.rotation()[2, 2])))))
    return off, float(tube.pos[1]), turn


@on_step
def _track_grippers(environment) -> None:
    """Record both tool points from the arms' joint positions, so scoring can tell whether the tube is still held."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    base = environment.unwrapped
    tcps = []
    for scene_name, arm in (("left_robot", "left"), ("right_robot", "right")):
        art = base.scene[scene_name]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        tcps.append(ik.tcp_pose(_np(art.data.joint_pos)[0][ids], station.BASES[arm])[0])
    _STATE["tcps"] = tcps


RELEASE = 0.030          # both tool points at least 3 cm from the tube's axis: let go


def released(tube: PartState) -> bool:
    """Both tool points at least 3 cm from the tube's axis (unreadable grippers count as still holding it)."""
    tcps = _STATE.get("tcps")
    if not tcps:
        return False
    a, b = tube.to_world((0.0, -CAP_FACE, 0.0)), tube.to_world((0.0, CAP_FACE, 0.0))
    seg = b - a
    for t in tcps:
        u = float(np.clip(np.dot(t - a, seg) / np.dot(seg, seg), 0.0, 1.0))
        if float(np.linalg.norm(a + u * seg - t)) < RELEASE:
            return False
    return True


def detents(tube: PartState) -> list:
    """For each end: (how far the plunger's centre is from the nearest detent hole's centre, how far its nose reaches
    past the socket's face into the hole), in metres. A detent is engaged when the nose is inside its hole."""
    out = []
    for tag, s in (("l", 1), ("r", -1)):
        pressed = abs(float(tube.sliders.get(f"detent_{tag}", 0.0)))
        c = tube.to_world((0.0, s * (CAP_FACE + 0.002 - pressed), DETENT_R))        # the nose's tip
        miss = min(math.hypot(c[0] - (SOCKET_X + k * DETENT_R), c[2] - AXIS_Z) for k in (-1, 1))
        out.append((miss, s * float(c[1]) - SOCKET_FACE))
    return out


def detents_engaged(tube: PartState) -> bool:
    """Both spring plungers' noses (3 mm) have sprung into their 5 mm holes: each reaches at least 0.2 mm past the
    socket's face (which it can only do inside a hole) with its centre within the hole (2.5 mm of its centre)."""
    return all(miss <= 0.0025 and depth >= 0.0002 for miss, depth in detents(tube))


def _update(objects) -> tuple[float, bool]:
    tube = objects["tube"]
    if _STATE["z0"] is None:
        _STATE["z0"] = float(tube.pos[2])
    off, dy, turn = tube_fit(tube)
    seated = off < 0.003 and abs(dy) < 0.003       # both ends' pins down in the pockets
    _STATE["lifted"] |= float(tube.pos[2]) > _STATE["z0"] + 0.03
    _STATE["seated"] |= seated
    _STATE["turned"] |= seated and turn >= 45
    locked = seated and turn >= 75 and detents_engaged(tube)
    recent = _STATE["recent"]
    recent.append(np.r_[tube.pos, turn / 1000])
    del recent[:-20]
    still = len(recent) == 20 and all(np.abs(p - recent[-1]).max() < 0.001 for p in recent)
    if locked and still and released(tube):
        return 1.0, True
    if _STATE["turned"]:
        return 0.75, False
    if _STATE["seated"]:
        return 0.5, False
    return (0.25 if _STATE["lifted"] else 0.0), False


def success(objects) -> bool:
    """Both ends' pins in the socket pockets, turned so both spring detents have clicked into their holes, both
    grippers off the tube, still for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.25 the tube was lifted 3 cm; 0.5 both ends' pins were put down in the sockets; 0.75 it was turned 45 degrees
    there; 1.0 both detents engaged (which needs the full quarter turn), both grippers let go, still for 1 s."""
    return _update(objects)[0]
