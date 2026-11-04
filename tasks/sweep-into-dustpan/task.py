"""On the station table: a light blue dustpan (160 mm wide, 143 mm deep, open toward the left) with a handle sticking
out to the right, a hand brush to the left of it, and five small orange wooden cubes ("crumbs") scattered between the
two. The dustpan's front lip is a thin ramp lying on the table. The robot must hold the dustpan in place by its handle
with one arm and, with the other, use the brush to sweep all five cubes up the lip into the pan. The pan is light: an
unheld pan is pushed away by the brush. A cube that a gripper finger touches does not count.

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

INSTRUCTION = ("Hold the blue dustpan still by its handle with one arm, and with the other arm use the hand brush to "
               "sweep all five orange cubes into the dustpan. Do not touch the cubes with the grippers.")

BLUE = (0.12, 0.35, 0.85)
WOOD = (0.72, 0.52, 0.30)
BRISTLE = (0.15, 0.15, 0.16)
ORANGE = (1.00, 0.45, 0.05)


def _mm(x0, x1, y0, y1, z0, z1, color, rot=None):
    spec = (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                     (z1 - z0) / 1000), color)
    return spec if rot is None else (*spec, rot)


# -- the dustpan: origin at the middle of its front lip, on the table; the pan opens toward +y (the brush side) and
# its floor runs back toward -y. Its sides and back stand on the table; the lip is a 1.5 mm plate sloped up at 7
# degrees from the table to the raised floor, so cubes ride up it. -----------------------------------------------------
PAN_HALF_W = 0.080          # outer half-width (x); the side walls are 3 mm thick
INNER_HALF_W = 0.077
LIP_DEG, LIP_L, LIP_T = 7.0, 0.032, 0.0015
LIP_GAP = 0.0003            # the lip's front edge sits 0.3 mm above the table
BACK_Y = -0.140             # the back wall's inner face
FLOOR_TOP = LIP_GAP + LIP_L * math.sin(math.radians(LIP_DEG)) + LIP_T * math.cos(math.radians(LIP_DEG))
LIP_END_Y = -LIP_L * math.cos(math.radians(LIP_DEG))
HANDLE_Y = (-0.143, -0.223)  # the handle runs 80 mm out behind the back wall
HANDLE_Z = (0.024, 0.040)    # 16 mm tall, 20 mm wide, its underside 24 mm above the table
HANDLE_GRIP_Y = -0.188       # where the demo grips it


def _lip_box():
    """A thin plate whose top face ramps from the table at y = 0 up to the pan's floor, sloping up toward -y."""
    a = math.radians(LIP_DEG)
    # Rotation about x by -a: the plate's -y end rises. Place it so its lowest edge (front, underside) is at y = 0.
    cy = -(LIP_L / 2 * math.cos(a) - LIP_T / 2 * math.sin(a))
    cz = LIP_GAP + LIP_L / 2 * math.sin(a) + LIP_T / 2 * math.cos(a)
    quat = (math.cos(-a / 2), math.sin(-a / 2), 0.0, 0.0)
    return ((0.0, cy, cz), (2 * PAN_HALF_W, LIP_L, LIP_T), BLUE, quat)


PAN_BOXES = [
    _lip_box(),
    _mm(-80, 80, -140, -31.5, 1000 * FLOOR_TOP - 2, 1000 * FLOOR_TOP, BLUE),        # floor, 2 mm thick
    _mm(-80, -77, -140, 0, 0, 30, BLUE), _mm(77, 80, -140, 0, 0, 30, BLUE),        # side walls, 30 mm tall
    _mm(-80, 80, -143, -140, 0, 40, BLUE),                                          # back wall, 40 mm tall
    _mm(-10, 10, -223, -143, 24, 40, BLUE),                                         # handle
]

# -- the hand brush: origin under the middle of its bristle block; long axis x. ------------------------------------------
BRUSH_HALF_L, BRUSH_HALF_T = 0.070, 0.0175
BRISTLE_TOP, BACK_TOP, KNOB_TOP = 0.022, 0.030, 0.052
BRUSH_BOXES = [
    _mm(-70, 70, -17.5, 17.5, 0, 22, BRISTLE),     # bristle block (rigid in simulation)
    _mm(-70, 70, -17.5, 17.5, 22, 30, WOOD),       # wooden back
    _mm(-50, 50, -10, 10, 30, 52, WOOD),           # handle bar along the top, 20 mm thick
]

CUBE = 0.015

# -- fixed layout (world frame: x forward from the arms, y left, metres) ----------------------------------------------
PAN_XY = (0.28, 0.02)
BRUSH_XY = (0.28, 0.20)
CUBES_XY = [(0.245, 0.060, 0.30), (0.270, 0.125, -0.40), (0.295, 0.085, 0.10), (0.315, 0.045, 0.55),
            (0.330, 0.110, -0.20)]   # x, y, yaw
PAN = part("dustpan", pos=(*PAN_XY, TOP), mass=0.080, boxes=PAN_BOXES, color=BLUE, friction=0.5)
BRUSH = part("brush", pos=(*BRUSH_XY, TOP + 0.0005), mass=0.120, boxes=BRUSH_BOXES, color=WOOD, friction=0.6)
CUBES = [part(f"cube_{i + 1}", pos=(x, y, TOP), yaw=yaw, mass=0.005,
              boxes=[((0.0, 0.0, CUBE / 2), (CUBE, CUBE, CUBE), ORANGE)], color=ORANGE, friction=0.5)
         for i, (x, y, yaw) in enumerate(CUBES_XY)]
CUBE_NAMES = tuple(c.name for c in CUBES)
OBJECTS = [PAN, BRUSH, *CUBES]

# -- scoring -----------------------------------------------------------------------------------------------------------
PAN_MOVE_LIMIT = 0.03    # the pan's lip may move at most 3 cm from its start
PAN_TILT_LIMIT = 10.0    # deg, for the final check
INSIDE_LIP = 0.020       # a cube counts once its centre is 2 cm past the lip's front edge
TOUCH_DIST = 0.022       # a finger pad point within 22 mm of a cube's centre counts as touching it
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(pan0=None, failed=False, touched=set(), recent=[], fingers_ok=None)


_reset_state()


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


def touched_cubes(points, cubes: dict) -> set:
    """Names of cubes whose centre is within TOUCH_DIST of any of ``points``."""
    out = set()
    for name, c in cubes.items():
        centre = c.pos + c.rotation() @ np.array([0.0, 0.0, CUBE / 2])
        if any(np.linalg.norm(p - centre) < TOUCH_DIST for p in points):
            out.add(name)
    return out


@on_step
def _watch_fingers(environment) -> None:
    """Mark every cube a gripper finger comes near (the robot's own state, not available to score())."""
    try:
        from yam_arena.scene import object_states

        base = environment.unwrapped
        origin = getattr(base.scene.env_origins, "torch", base.scene.env_origins)[0].detach().cpu().numpy()
        cubes = object_states(environment, list(CUBE_NAMES))
        pts = [p for arm in ("left_robot", "right_robot") for p in finger_points(base.scene[arm], origin)]
        _STATE["touched"] |= touched_cubes(pts, cubes)
        _STATE["fingers_ok"] = True
    except Exception as exc:  # noqa: BLE001 - report once, keep the episode running
        if _STATE["fingers_ok"] is None:
            print(f"sweep-into-dustpan: finger check unavailable: {exc!r}", flush=True)
        _STATE["fingers_ok"] = False


def in_pan(pan: PartState, cube: PartState) -> bool:
    """The cube's centre lies on the pan: at least 2 cm past the lip's front edge, between the side walls, in front of
    the back wall, and lying on the pan (its centre at least half a cube, less 3 mm, above the pan's floor and not more
    than 3 cm above it)."""
    c = pan.to_local(cube.pos + cube.rotation() @ np.array([0.0, 0.0, CUBE / 2]))
    return bool(abs(c[0]) < INNER_HALF_W and BACK_Y < c[1] < -INSIDE_LIP
                and FLOOR_TOP + CUBE / 2 - 0.003 < c[2] < FLOOR_TOP + 0.030)


def pan_tilt(pan: PartState) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, pan.rotation()[2, 2]))))


def _update(objects) -> tuple[float, bool]:
    pan = objects["dustpan"]
    if _STATE["pan0"] is None:
        _STATE["pan0"] = pan.pos.copy()
    _STATE["failed"] |= float(np.linalg.norm((pan.pos - _STATE["pan0"])[:2])) > PAN_MOVE_LIMIT
    if _STATE["failed"]:
        return 0.0, False
    inside = [n for n in CUBE_NAMES if n not in _STATE["touched"] and in_pan(pan, objects[n])]
    score = 0.2 * len(inside)
    on_table = abs(pan.pos[2] - TOP) < 0.005 and pan_tilt(pan) < PAN_TILT_LIMIT
    recent = _STATE["recent"]
    recent.append(np.array([objects[n].pos for n in (*CUBE_NAMES, "dustpan")]))
    del recent[:-20]
    still = len(recent) == 20 and all(np.abs(p - recent[-1]).max() < 0.001 for p in recent)
    if len(inside) == len(CUBE_NAMES) and not on_table:
        score = 0.8   # all in, but the pan is lifted or tipped
    return score, bool(len(inside) == len(CUBE_NAMES) and on_table and still)


def success(objects) -> bool:
    """All five cubes on the pan, none touched by a finger, the pan flat on the table, everything still for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0.2 per cube on the pan that no finger touched (0.8 at most unless the pan rests flat on the table); 0 if the
    pan's lip ever moved 3 cm from where it started."""
    return _update(objects)[0]


def distance(objects) -> float:
    """Mean distance (m) from the cubes to the middle of the pan's floor."""
    pan = objects["dustpan"]
    goal = pan.to_world((0.0, (BACK_Y - INSIDE_LIP) / 2, FLOOR_TOP + CUBE / 2))
    return float(np.mean([np.linalg.norm(objects[n].pos + [0.0, 0.0, CUBE / 2] - goal) for n in CUBE_NAMES]))
