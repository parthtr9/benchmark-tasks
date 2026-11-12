"""On the station table, in front of the arms: a glass desiccator (a pale blue jar, 0.6 kg, with a 15 cm round flange and
two carrying lugs sticking out left and right) closed by a glass lid (0.4 kg, 15 cm flange, a low dome and a 35 mm
knob) that is held on by a greased ground-glass seal and a slight vacuum. Inside, on the porcelain plate, stands a small
amber sample vial. To the right of the desiccator lies a grey pad, to the left a small white tray.

The seal holds the lid down with about 40 N, so it cannot be lifted straight off (pulling the knob up lifts the whole
desiccator), but it slides sideways on the greased flange with a pull of about 6.6 N. The desiccator itself slides on the
bench with only about 2.5 N, so a lid pushed alone drags the desiccator along with it. The lid has to be slid aside while
the other arm holds the desiccator by a lug; once it is 5 cm across, the seal is broken and the lid lifts away.

The robot must take the vial out and stand it on the tray, put the lid on the pad, and not move the desiccator.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import (PartState, finger_points, gripper_half_gaps, on_reset, on_step, part, push_body)  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Take the sample vial out of the desiccator and stand it on the white tray. Put the desiccator's lid on "
               "the grey pad, and do not move the desiccator itself.")

GLASS = (0.70, 0.84, 0.92)
GLASS_LID = (0.78, 0.90, 0.96)
DARK = (0.22, 0.23, 0.25)
AMBER = (0.80, 0.42, 0.08)
GREY = (0.50, 0.51, 0.53)
WHITE = (0.94, 0.94, 0.92)
PORCELAIN = (0.92, 0.90, 0.84)

# -- layout (world frame: x forward from the arms, y left, metres). Base and lid origins are on the desiccator's axis;
# the base's at table height, the lid's at its flange underside. ------------------------------------------------------
BASE_XY = (0.29, 0.0)
VIAL_XY = (0.29, 0.0)
PAD_XY, PAD_SIZE = (0.25, -0.21), (0.17, 0.17, 0.004)
TRAY_XY, TRAY_SIZE = (0.23, 0.17), (0.08, 0.08, 0.004)
PAD_TOP, TRAY_TOP = PAD_SIZE[2], TRAY_SIZE[2]

RIM_Z = 0.080                       # the base's flange top and the lid's underside when closed
PLATE_Z = 0.015                     # the porcelain plate's top, inside
R_FLANGE, R_WALL_OUT, R_MOUTH = 0.075, 0.070, 0.060
WALL_TOP = 0.065
LUG = (0.030, 0.055, 0.030)         # lug box: along x, y (0.06..0.115 from the axis), z (0.035..0.065)
LUG_Y = 0.0875                      # its centre from the axis
LUG_GRIP_Y = 0.100                  # where the demo grips it
LUG_Z = 0.050
LID_FLANGE_H, LID_DOME_R, LID_DOME_H = 0.008, 0.055, 0.027
KNOB_R, KNOB_H = 0.0175, 0.030
KNOB_Z = LID_FLANGE_H + LID_DOME_H + KNOB_H / 2      # 0.050 above the flange underside (0.130 above the table)
LID_TOP = LID_FLANGE_H + LID_DOME_H
VIAL_R, VIAL_H = 0.014, 0.060

BASE_MASS, LID_MASS, VIAL_MASS = 0.6, 0.4, 0.03
# Surface friction (PhysX averages a pair): lid on base about 0.15, base on the bench about 0.25.
BASE_FRICTION, LID_FRICTION, VIAL_FRICTION = 0.05, 0.25, 0.8

# -- the seal ---------------------------------------------------------------------------------------------------------
SEAL_RADIUS = 0.045                 # lid axis within this of the base axis (horizontally) ...
SEAL_GAP = 0.003                    # ... and its flange within this of the base's: sealed
SEAL_FORCE = 40.0                   # N down on the lid and up on the base while sealed


def _ring(r0, r1, z0, z1, color, n=16):
    """A polygon ring of boxes (part frame), radial thickness r1 - r0."""
    rm = (r0 + r1) / 2
    tang = 2 * rm * math.tan(math.pi / n) * 1.04
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        out.append(((rm * math.cos(a), rm * math.sin(a), (z0 + z1) / 2), (r1 - r0, tang, z1 - z0), color,
                    (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))))
    return out


def _lug_box(side):
    return ((0.0, side * LUG_Y, LUG_Z), LUG, DARK)


BASE_BOXES = (_ring(R_MOUTH, R_WALL_OUT, PLATE_Z, WALL_TOP, GLASS) + _ring(R_MOUTH, R_FLANGE, WALL_TOP, RIM_Z, GLASS)
              + [_lug_box(1), _lug_box(-1)])
BASE_CYLS = [((0.0, 0.0, PLATE_Z / 2), R_WALL_OUT, PLATE_Z, PORCELAIN)]
LID_CYLS = [((0.0, 0.0, LID_FLANGE_H / 2), R_FLANGE, LID_FLANGE_H, GLASS_LID),
            ((0.0, 0.0, LID_FLANGE_H + LID_DOME_H / 2), LID_DOME_R, LID_DOME_H, GLASS_LID),
            ((0.0, 0.0, KNOB_Z), KNOB_R, KNOB_H, DARK)]
# the lid needs one box so the part is valid; a thin flange-disc box inside the cylinder adds nothing to the shape
LID_BOXES = [((0.0, 0.0, LID_FLANGE_H / 2), (0.01, 0.01, LID_FLANGE_H), GLASS_LID)]
VIAL_CYLS = [((0.0, 0.0, VIAL_H / 2), VIAL_R, VIAL_H, AMBER)]
VIAL_BOXES = [((0.0, 0.0, VIAL_H / 2), (0.006, 0.006, VIAL_H), AMBER)]

BASE = part("base", pos=(*BASE_XY, TOP), mass=BASE_MASS, boxes=BASE_BOXES, cylinders=BASE_CYLS, color=GLASS,
            friction=BASE_FRICTION)
LID = part("lid", pos=(*BASE_XY, TOP + RIM_Z + 0.0005), mass=LID_MASS, boxes=LID_BOXES, cylinders=LID_CYLS,
           color=GLASS_LID, friction=LID_FRICTION)
VIAL = part("vial", pos=(*VIAL_XY, TOP + PLATE_Z + 0.0005), mass=VIAL_MASS, boxes=VIAL_BOXES, cylinders=VIAL_CYLS,
            color=AMBER, friction=VIAL_FRICTION)
PAD = Box("pad", pos=(*PAD_XY, TOP + PAD_TOP / 2), size=PAD_SIZE, color=GREY)
TRAY = Box("tray", pos=(*TRAY_XY, TOP + TRAY_TOP / 2), size=TRAY_SIZE, color=WHITE)
OBJECTS = [PAD, TRAY, BASE, LID, VIAL]

# -- scoring thresholds ---------------------------------------------------------------------------------------------
BASE_MOVE_LIMIT = 0.015      # m: the desiccator slid more than 15 mm over the bench ...
BASE_LIFT_LIMIT = 0.008      # ... or rose more than 8 mm
BASE_TURN_LIMIT = 10.0       # deg of yaw
VIAL_TIP_LIMIT = 30.0        # deg: the vial tipped
LID_FALL_SPEED = 0.7         # m/s downward: the lid dropped about 2.5 cm or more
LID_TILT_LIMIT = 45.0
VIAL_ON_TRAY_XY = 0.026      # vial axis within this of the tray's middle
VIAL_UPRIGHT = 10.0
LID_ON_PAD_XY = 0.030        # lid axis within this of the pad's middle
LID_FLAT = 8.0
SET_TOL = 0.003              # resting height tolerance
OFF_SEAL = 0.06              # lid axis this far from the base axis, or this high above its seat: off the desiccator
OFF_SEAL_LIFT = 0.010
RELEASE_GAP = 0.03           # m: every finger point this far from the vial, the lid and the base
OPEN_HALF_GAP = 0.030        # m: each gripper's half-opening at least this (a 60 mm gap) counts as open
STILL_CALLS = 20
STILL_MM = 0.001

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(base0=None, yaw0=None, failed=False, sealed=False, released=False, opened=False, finger_check=None,
                  force_api=None, recent=[], peak_base_move=0.0, peak_base_lift=0.0, peak_vial_tilt=0.0,
                  worst_fall=0.0, lid_off=False, gap_min=9.0)


_reset_state()


def _yaw(p: PartState) -> float:
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def tilt(p: PartState) -> float:
    """Degrees the part's z axis is off vertical."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(p.rotation()[2, 2])))))


def lid_offset(lid: PartState, base: PartState) -> float:
    """Horizontal distance (m) between the lid's axis and the base's."""
    return float(np.linalg.norm((lid.pos - base.pos)[:2]))


def lid_lift(lid: PartState, base: PartState) -> float:
    """Height (m) of the lid's flange above the base's flange top."""
    return float(lid.pos[2] - base.pos[2] - RIM_Z)


def is_sealed(lid: PartState, base: PartState) -> bool:
    return lid_offset(lid, base) < SEAL_RADIUS and lid_lift(lid, base) < SEAL_GAP


def box_gap(point, part_state: PartState) -> float:
    """Distance (m) from a world point to the part's bounding box."""
    d = np.maximum(np.maximum(part_state.low - point, point - part_state.high), 0.0)
    return float(np.linalg.norm(d))


@on_step
def _mechanism(environment) -> None:
    """The seal and the gripper checks: runs after every control step, before scoring."""
    from yam_arena.scene import object_states

    st = object_states(environment, ["base", "lid", "vial"])
    base, lid, vial = st["base"], st["lid"], st["vial"]
    sealed = is_sealed(lid, base)
    _STATE["sealed"] = sealed
    f = SEAL_FORCE if sealed else 0.0
    try:
        push_body(environment, "lid", (0.0, 0.0, -f))
        push_body(environment, "base", (0.0, 0.0, f))
        if _STATE["force_api"] is None:
            _STATE["force_api"] = True
    except Exception as exc:  # noqa: BLE001 - report once
        if _STATE["force_api"] is None:
            print(f"desiccator-lid-slide: seal force unavailable: {exc!r}", flush=True)
        _STATE["force_api"] = False
    try:
        pts = finger_points(environment)
        gap = min(box_gap(np.asarray(p), s) for p in pts for s in (vial, lid, base))
        _STATE["gap_min"] = gap
        _STATE["released"] = bool(gap >= RELEASE_GAP)
        _STATE["opened"] = bool(min(gripper_half_gaps(environment)) >= OPEN_HALF_GAP)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"desiccator-lid-slide: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"], _STATE["released"], _STATE["opened"] = False, False, False


def vial_on_tray(vial: PartState) -> bool:
    d = float(np.linalg.norm(vial.pos[:2] - np.array(TRAY_XY)))
    return d <= VIAL_ON_TRAY_XY and abs(vial.pos[2] - TOP - TRAY_TOP) <= SET_TOL and tilt(vial) <= VIAL_UPRIGHT


def lid_on_pad(lid: PartState) -> bool:
    d = float(np.linalg.norm(lid.pos[:2] - np.array(PAD_XY)))
    return d <= LID_ON_PAD_XY and abs(lid.pos[2] - TOP - PAD_TOP) <= SET_TOL and tilt(lid) <= LID_FLAT


def lid_off_seal(lid: PartState, base: PartState) -> bool:
    return lid_offset(lid, base) >= OFF_SEAL or lid_lift(lid, base) >= OFF_SEAL_LIFT


def _update(objects) -> tuple[float, bool]:
    base, lid, vial = objects["base"], objects["lid"], objects["vial"]
    if _STATE["base0"] is None:
        _STATE.update(base0=base.pos.copy(), yaw0=_yaw(base))
    move = float(np.linalg.norm((base.pos - _STATE["base0"])[:2]))
    lift = float(base.pos[2] - _STATE["base0"][2])
    turn = abs(math.degrees((_yaw(base) - _STATE["yaw0"] + math.pi) % (2 * math.pi) - math.pi))
    _STATE["peak_base_move"] = max(_STATE["peak_base_move"], move)
    _STATE["peak_base_lift"] = max(_STATE["peak_base_lift"], lift)
    _STATE["peak_vial_tilt"] = max(_STATE["peak_vial_tilt"], tilt(vial))
    _STATE["worst_fall"] = max(_STATE["worst_fall"], float(-lid.lin_vel[2]))
    _STATE["failed"] |= bool(move > BASE_MOVE_LIMIT or lift > BASE_LIFT_LIMIT or turn > BASE_TURN_LIMIT
                             or tilt(vial) > VIAL_TIP_LIMIT or float(-lid.lin_vel[2]) > LID_FALL_SPEED
                             or tilt(lid) > LID_TILT_LIMIT)
    if _STATE["failed"]:
        return 0.0, False
    released = bool(_STATE["released"])
    v_placed, l_placed = vial_on_tray(vial), lid_on_pad(lid)
    v_ok = v_placed and released
    recent = _STATE["recent"]
    recent.append((np.r_[vial.pos, lid.pos, base.pos], released and bool(_STATE["opened"])))
    del recent[:-STILL_CALLS]
    still = (len(recent) == STILL_CALLS and all(f for _, f in recent)
             and float(np.ptp(np.array([p for p, _ in recent]), axis=0).max()) < STILL_MM)
    if v_placed and l_placed:      # both where they belong: full credit once let go and still, 0.75 until then
        return (1.0, True) if still else (0.75, False)
    if v_ok:
        return 0.5, False
    if lid_off_seal(lid, base):
        return 0.25, False
    return 0.0, False


def success(objects) -> bool:
    """The vial stands upright (within 10 degrees) on the tray, the lid rests flange-down on the pad, every finger point
    of both grippers is 3 cm or more from the vial, the lid and the desiccator and both grippers are open, and the vial,
    lid and desiccator stay within 1 mm over the last 20 scorer calls (1 s)."""
    return _update(objects)[1]


def score(objects) -> float:
    """At the end: 1.0 the vial is upright on the tray and the lid rests flange-down on the pad, the grippers are open and
    3 cm or more from the vial, lid and desiccator, and everything was still for 1 s; 0.75 the same but a gripper is
    still near or open too little or something moves; 0.5 the vial is upright on the tray and let go but the lid is not
    on the pad; 0.25 the lid is off the desiccator (6 cm sideways or 1 cm up) but the vial is not on the tray; else 0.
    0 whenever, at any time: the desiccator slid more than 15 mm, rose more than 8 mm or turned more than 10 degrees, the
    vial tipped more than 30 degrees, or the lid dropped (fell faster than 0.7 m/s, about 2.5 cm) or tilted over 45
    degrees. A flag is never undone."""
    return _update(objects)[0]


def distance(objects) -> float:
    """Metres to go: the vial to the tray's middle plus the lid to the pad's middle (horizontally)."""
    vial, lid = objects["vial"], objects["lid"]
    return float(np.linalg.norm(vial.pos[:2] - np.array(TRAY_XY)) + np.linalg.norm(lid.pos[:2] - np.array(PAD_XY)))


def debug_fields() -> dict:
    """A copy of the mechanism state, for the demo's DEMO_DEBUG trace only."""
    return dict(_STATE)
