"""On the station table: a grey benchtop horizontal press (290 x 380 mm base plate) whose orange ram slides forward
out of a housing at the back of the plate toward a dark anvil block at the front of a shallow open-topped die; two
green palm buttons on the front corners of the base plate; and a red aluminium blank (40 x 40 x 14 mm) lying on the
table in front of the press, to the left. The press has a two-hand control: the ram only moves forward while both
buttons are held, and only if they were pushed within half a second of each other. The robot must set the blank in the
die, take both grippers out of the press, push both buttons at the same time, one with each gripper, and hold them
until the ram has closed on the blank.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, Slider, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Put the red blank into the die in front of the orange ram, take both grippers out of the press, then "
               "push the two green buttons at the same time, one with each gripper, and hold them until the ram has "
               "closed on the blank. Then let go of the buttons.")

GREY = (0.55, 0.56, 0.58)
DARK = (0.25, 0.26, 0.28)
GREEN = (0.10, 0.70, 0.20)
ORANGE = (1.00, 0.50, 0.00)
RED = (0.85, 0.12, 0.12)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- the press: origin under the centre of its base plate, on the table; x forward (toward the robot is -x) ---------
# The ram rests at the back and moves toward the robot (-x) across the die, closing on the blank against the anvil.
B = 10                    # base plate top, mm above the press origin
DIE_X, DIE_Y = -40, 0     # die centre in the press frame (mm): the seated blank's centre
BLANK = 40                # blank, mm square
BLANK_H = 14              # blank thickness, mm
NEST = 46                 # die width across y, mm (3 mm a side)
GAP = 6                   # mm between a centred blank's front face and the anvil: room to set it down
ANVIL_FACE = DIE_X - BLANK / 2 - GAP        # anvil's inner face, mm
DAYLIGHT = 80             # ram face behind the centred blank's back face at rest, mm
RAM_REST_X = DIE_X + BLANK / 2 + DAYLIGHT   # ram face at rest (x, mm)
RAM_STROKE = (DAYLIGHT + GAP + 2) / 1000    # the drive's farthest target: 2 mm past a blank pushed onto the anvil (m)
RAM_TRAVEL = (DAYLIGHT + GAP + 5) / 1000    # the joint's limit (m)
RAM_W, RAM_H = 40, 10                        # ram face, mm (across y, tall); its bottom 2 mm above the plate
BUTTON_X, BUTTON_Y = -100, 150
BUTTON_TRAVEL = 0.008

_n = NEST / 2
FRAME_BOXES = [
    _mm(-140, 150, -190, 190, 0, B, GREY),                                                  # base plate
    _mm(ANVIL_FACE - 12, ANVIL_FACE, DIE_Y - _n - 10, DIE_Y + _n + 10, B, B + 25, DARK),     # anvil
    _mm(ANVIL_FACE, RAM_REST_X, DIE_Y + _n, DIE_Y + _n + 10, B, B + 4, DARK),               # die side rails
    _mm(ANVIL_FACE, RAM_REST_X, DIE_Y - _n - 10, DIE_Y - _n, B, B + 4, DARK),
    _mm(RAM_REST_X + 40, RAM_REST_X + 90, -45, 45, B, B + 60, GREY),                         # ram housing
]
for _s in (1, -1):                                                                          # button collars, 12 mm
    _bx, _by = BUTTON_X, _s * BUTTON_Y
    FRAME_BOXES += [_mm(_bx - 31, _bx - 23, _by - 31, _by + 31, B, B + 12, DARK),
                    _mm(_bx + 23, _bx + 31, _by - 31, _by + 31, B, B + 12, DARK),
                    _mm(_bx - 23, _bx + 23, _by - 31, _by - 23, B, B + 12, DARK),
                    _mm(_bx - 23, _bx + 23, _by + 23, _by + 31, B, B + 12, DARK)]

# Slider shapes are given in the press frame, at the slider's rest position.
RAM = Slider("ram", rest=(RAM_REST_X / 1000, DIE_Y / 1000, (B + 2 + RAM_H / 2) / 1000),
             boxes=(_mm(RAM_REST_X, RAM_REST_X + 10, DIE_Y - RAM_W / 2, DIE_Y + RAM_W / 2, B + 2, B + 2 + RAM_H,
                        ORANGE),
                    _mm(RAM_REST_X + 10, RAM_REST_X + 70, -12, 12, B + 2, B + 2 + RAM_H, GREY)),
             travel=RAM_TRAVEL, rate=1500.0, damping=60.0, mass=0.150, axis="X", sign=-1)


def _button(name, s):
    """A 40 mm square green button, its top 40 mm above the base plate, 8 mm of travel on a 500 N/m spring."""
    y = s * BUTTON_Y
    return Slider(name, rest=(BUTTON_X / 1000, y / 1000, (B + 25) / 1000),
                  boxes=(_mm(BUTTON_X - 20, BUTTON_X + 20, y - 20, y + 20, B + 10, B + 40, GREEN),),
                  travel=BUTTON_TRAVEL, rate=500.0, damping=10.0, mass=0.020)


BUTTONS = (_button("button_left", 1), _button("button_right", -1))

# -- fixed layout (world frame: x forward from the arms, y left, metres) ----------------------------------------------
PRESS_XY, PRESS_YAW = (0.34, 0.0), 0.0
BLANK_XY = (0.14, 0.12)
PRESS = part("press", pos=(*PRESS_XY, TOP), yaw=PRESS_YAW, mass=2.5, boxes=FRAME_BOXES, sliders=(RAM, *BUTTONS),
             color=GREY, friction=0.8)
BLANK_OBJ = Box("blank", pos=(*BLANK_XY, TOP + BLANK_H / 2000), size=(BLANK / 1000, BLANK / 1000, BLANK_H / 1000),
                mass=0.060, color=RED)
OBJECTS = [PRESS, BLANK_OBJ]

SEAT = np.array([DIE_X / 1000, DIE_Y / 1000, (B + BLANK_H / 2) / 1000])  # the seated blank's centre, press frame
# The die zone: the space over the die between the anvil and the ram's rest face, up to 8 cm above the plate. A gripper
# in it while the ram has moved past the guard line is the accident the two-hand control exists to prevent.
ZONE_LO = np.array([(ANVIL_FACE - 12) / 1000, (DIE_Y - _n - 10) / 1000, B / 1000])
ZONE_HI = np.array([RAM_REST_X / 1000, (DIE_Y + _n + 10) / 1000, (B + 80) / 1000])
GUARD = 0.015            # ram moved more than this: closing through the hazard zone
PRESSED_RAM = (DAYLIGHT + GAP - 3) / 1000   # ram moved this far: the blank is closed against the anvil (3 mm margin)
SEAT_X = (-(GAP + 1) / 1000, 0.006)          # seated: centre from against the anvil to 6 mm behind the nominal seat
BUTTON_PRESSED = 0.005   # a button down 5 mm of its 8 mm counts as pushed
SYNC_S = 0.5             # both buttons within half a second of each other, else the control refuses
CONTROL_HZ = 20
RAM_DOWN_PER_STEP = 0.003   # closing at 6 cm/s: the full stroke takes about 1.4 s
RAM_UP_PER_STEP = 0.005
PRESS_MOVE_LIMIT = 0.010
PRESS_TURN_LIMIT = 5.0

# -- the two-hand control (runs every control step, before scoring) ---------------------------------------------------
_CTRL: dict = {}


def _reset_ctrl() -> None:
    _CTRL.update(step=0, onset={"button_left": None, "button_right": None}, enabled=False, refused=False,
                 target=0.0)


_reset_ctrl()


def _press_state(environment) -> PartState:
    from partkit import _part_state

    return _part_state(environment, PRESS)


def _set_ram_target(environment, target: float) -> None:
    import torch

    asset = environment.unwrapped.scene["press"]
    idx = list(asset.joint_names).index("ram_slide")
    ref = getattr(asset.data.joint_pos, "torch", asset.data.joint_pos)
    t = torch.full((ref.shape[0], 1), float(target), dtype=ref.dtype, device=ref.device)
    asset.actuators.target_command.set_position_index(value=t, joint_ids=[idx])


def control(sliders: dict) -> float:
    """One control step of the two-hand relay from the sliders' positions; returns the ram drive's target (m, <= 0).

    The ram descends only while both buttons are held, and only if they were pushed within SYNC_S of each other
    (anti-tie-down: a button held early, or weighted down, locks the control out until both are released). Letting
    either button go at any point sends the ram back up."""
    c = _CTRL
    c["step"] += 1
    pressed = {k: sliders.get(k, 0.0) <= -BUTTON_PRESSED for k in c["onset"]}
    for k, p in pressed.items():
        if p and c["onset"][k] is None:
            c["onset"][k] = c["step"]
        if not p:
            c["onset"][k] = None
    both = all(pressed.values())
    if not any(pressed.values()):
        c["refused"] = False
    if both and not c["enabled"] and not c["refused"]:
        if abs(c["onset"]["button_left"] - c["onset"]["button_right"]) <= round(SYNC_S * CONTROL_HZ):
            c["enabled"] = True
        else:
            c["refused"] = True
    if not both:
        c["enabled"] = False
    if c["enabled"]:
        c["target"] = max(c["target"] - RAM_DOWN_PER_STEP, -RAM_STROKE)
    else:
        c["target"] = min(c["target"] + RAM_UP_PER_STEP, 0.0)
    return c["target"]


@on_step
def _two_hand_control(environment) -> None:
    _set_ram_target(environment, control(_press_state(environment).sliders))


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


# -- scoring -----------------------------------------------------------------------------------------------------------
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(press0=None, yaw0=None, loaded=False, started=False, pressed=False, failed=None, recent=[])
    _reset_ctrl()
    if environment is not None:
        _set_ram_target(environment, 0.0)


_reset_state()


def _yaw(p: PartState) -> float:
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def _rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def blank_in_die(press: PartState, blank) -> tuple[np.ndarray, float, bool]:
    """The blank's centre in the press frame, its tilt (deg), and whether it is seated in the die: centre within 4 mm
    of the die's centre across it, along it anywhere from against the anvil to 6 mm behind the
    nominal seat, resting on the die floor (within 3 mm), tilted under 5 degrees."""
    c = press.to_local(blank.pos)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, (press.rotation().T @ _rotation(blank.quat_xyzw))[2, 2]))))
    dx = c[0] - SEAT[0]
    seated = bool(SEAT_X[0] <= dx <= SEAT_X[1] and abs(c[1] - SEAT[1]) < 0.004 and abs(c[2] - SEAT[2]) < 0.003
                  and tilt < 5)
    return c, tilt, seated


def gripper_in_zone(press: PartState, points) -> bool:
    for p in points:
        q = press.to_local(p)
        if np.all(q >= ZONE_LO) and np.all(q <= ZONE_HI):
            return True
    return False


def _update(objects, points=None) -> tuple[float, bool]:
    from partkit import env

    press, blank = objects["press"], objects["blank"]
    if _STATE["press0"] is None:
        _STATE.update(press0=press.pos.copy(), yaw0=_yaw(press))
    if points is None:
        points = _gripper_points(env()) if env() is not None else []
    ram = -press.sliders.get("ram", 0.0)           # how far the ram is down, m
    buttons = [press.sliders.get(k, 0.0) <= -BUTTON_PRESSED for k in ("button_left", "button_right")]
    c, tilt, seated = blank_in_die(press, blank)
    in_zone = gripper_in_zone(press, points)
    moved = float(np.linalg.norm(press.pos - _STATE["press0"]))
    turned = abs(math.degrees((_yaw(press) - _STATE["yaw0"] + math.pi) % (2 * math.pi) - math.pi))
    if _STATE["failed"] is None:
        if moved > PRESS_MOVE_LIMIT or turned > PRESS_TURN_LIMIT:
            _STATE["failed"] = "press moved"
        elif ram > GUARD and in_zone:
            _STATE["failed"] = "gripper in the die while the ram was down"
        elif ram > GUARD and not seated:
            _STATE["failed"] = "ram cycled without the blank seated"
    if _STATE["failed"]:
        return 0.0, False
    _STATE["loaded"] |= seated
    _STATE["started"] |= bool(seated and ram > GUARD)
    _STATE["pressed"] |= bool(seated and ram >= PRESSED_RAM)
    done = _STATE["pressed"] and seated and ram < 0.005 and not any(buttons) and not in_zone
    recent = _STATE["recent"]
    recent.append(np.r_[c, ram])
    del recent[:-CONTROL_HZ]
    still = len(recent) == CONTROL_HZ and all(np.linalg.norm(p - recent[-1]) < 0.001 for p in recent)
    if done:
        return 1.0, bool(still)
    if _STATE["pressed"]:
        return 0.75, False
    if _STATE["started"]:
        return 0.5, False
    return (0.25 if _STATE["loaded"] else 0.0), False


def success(objects) -> bool:
    """The ram came down onto the seated blank and is back up, both buttons are released, the blank is still seated,
    both grippers are out of the die, and nothing moved for 1 s; the press is in place."""
    return _update(objects)[1]


def distance(objects) -> float:
    """Metres from the blank's centre to its seat in the die."""
    press, blank = objects["press"], objects["blank"]
    return float(np.linalg.norm(press.to_local(blank.pos) - SEAT))


def score(objects) -> float:
    """0.25 blank seated in the die; 0.5 ram started down with the blank seated (both buttons pushed together);
    0.75 ram came down onto the blank; 1.0 ram back up, buttons released, blank still seated, grippers out of the die.
    0 if the press moved 1 cm or turned 5 degrees, if the ram moved down 3 cm with the blank not seated, or if a
    gripper was in the die while the ram was more than 3 cm down."""
    return _update(objects)[0]
