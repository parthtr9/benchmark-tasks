"""On the station table, in front of the arms: a short training barbell (aluminium bar, 25 mm across, 0.4 m long, 70 g
with its collars) lies along y in two fixed J-hooks (open cradles) at x 0.28, y = +-0.07, its centre 100 mm above the
table. On each end a rubber technique plate (140 mm across, 25 mm thick, 300 g, 50 mm bore) hangs on the 30 mm sleeve
against the bar's collar. Two fixed rubber plate holders stand on the table at (0.28, +-0.31), each with a 30 mm wide
slot, 20 mm deep, in which a plate stands on its rim.

The bar balances on the hooks only while both ends carry the same load. Take the plates off one end first and the
plate left on the other end levers that end down over its hook: the bar see-saws, the bare end flies up out of its hook
and the bar comes off the rack. The robot has to take both plates off together (each arm one plate), and stand them in
their holders.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Strip the barbell: take both plates off it and stand each one in the plate holder on its own side, "
               "leaving the bar resting in the rack's hooks. Then let go of everything.")

ALU = (0.72, 0.74, 0.78)
RUBBER = (0.12, 0.12, 0.14)
RED = (0.80, 0.14, 0.12)
DARK = (0.22, 0.23, 0.25)


def _mm(x0, x1, y0, y1, z0, z1, color, rot=None):
    spec = (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                     (z1 - z0) / 1000), color)
    return spec if rot is None else (*spec, rot)


# -- layout (world frame, metres; x forward from the arms, y left) --------------------------------------------------
RACK_X = 0.28
BAR_Z = 0.100                        # the bar's axis above the table
HOOK_Y = 0.070                       # the hooks' centre lines, +-
HOOK_HALF = 0.010                    # a hook is 20 mm thick along the bar
HOLDER_Y = 0.310                     # the holders' centre lines, +-
BAR_HALF = 0.200
BAR_MASS = 0.070

# -- the bar: origin at its middle, axis along y ----------------------------------------------------------------------
BAR_R, COLLAR_R, SLEEVE_R = 0.0125, 0.026, 0.015
COLLAR_Y = (0.100, 0.105)
BAR_CYL = [((0, 0, 0), BAR_R, 0.200, ALU, "Y")]
for _s in (1.0, -1.0):
    BAR_CYL.append(((0, _s * 0.1025, 0), COLLAR_R, 0.005, RED, "Y"))                          # collar
    BAR_CYL.append(((0, _s * (COLLAR_Y[1] + BAR_HALF) / 2, 0), SLEEVE_R, BAR_HALF - COLLAR_Y[1], ALU, "Y"))   # sleeve
BAR_BOXES = [((0, 0, 0), (0.002, 0.002, 0.002), ALU)]       # partkit wants one box; it is inside the bar

# -- a plate: origin at its centre, axis along y; a ring of 16 boxes around the 50 mm bore --------------------------------
PLATE_R, PLATE_T, BORE_R = 0.070, 0.025, 0.025
PLATE_MASS = 0.30
PLATE_Y = 0.1175                     # centre line from the bar's middle
PLATE_DROP = 0.010                   # hangs this far below the bar's axis
PLATE_Z = BAR_Z - PLATE_DROP
N_SEG = 16


def _ring():
    boxes = []
    radial = PLATE_R - BORE_R
    for k in range(N_SEG):
        phi = 2 * math.pi * k / N_SEG
        theta = -phi                                     # a box's x axis turned about +y by theta points at (cos phi, sin phi)
        rc = (PLATE_R + BORE_R) / 2
        centre = (rc * math.cos(phi), 0.0, rc * math.sin(phi))
        quat = (math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0)
        boxes.append((centre, (radial, PLATE_T, 0.026), RUBBER, quat))
    return boxes


PLATE_BOXES = _ring()

# -- a hook: origin on the table under the cradle ---------------------------------------------------------------------
CRADLE_GAP = 0.015                   # half-width of the cradle between its lips (bar r 12.5 mm + 2.5 mm)
FLOOR_Z = BAR_Z - BAR_R              # the cradle's floor
HOOK_BOXES = [
    _mm(-50, 50, -30, 30, 0, 12, DARK),                                              # foot
    _mm(-15, 15, -10, 10, 12, 1000 * FLOOR_Z, DARK),                                 # post, topped by the cradle floor
    _mm(15, 20, -10, 10, 1000 * FLOOR_Z, 1000 * (BAR_Z + 0.022), DARK),              # front lip
    _mm(-20, -15, -10, 10, 1000 * FLOOR_Z, 1000 * (BAR_Z + 0.022), DARK),            # back lip
]
HOOK_MASS = 3.0

# -- a holder: origin on the table under the slot; a 30 mm slot along x, 20 mm deep, with a flared mouth -------------------
SLOT_HALF, SLOT_DEPTH, FLOOR_T = 0.015, 0.020, 0.010
HOLDER_X_HALF = 0.040                # the plate may stand anywhere within this of the holder's middle
FLARE = math.radians(45.0)


def _flare(side):
    """A 6 mm thick strip tilted 45 degrees on top of a slot wall, widening the slot's mouth."""
    z0 = FLOOR_T + SLOT_DEPTH
    ln, t = 0.020, 0.006
    alpha = -side * FLARE            # a box's z axis rotated about x by alpha points at (-sin a, cos a) in (y, z)
    d = (math.sin(-alpha), math.cos(alpha))
    n = (math.cos(alpha), math.sin(alpha))
    cy = side * SLOT_HALF + d[0] * ln / 2 + n[0] * side * t / 2 * 1.0
    cz = z0 + d[1] * ln / 2 + n[1] * t / 2
    return ((0.0, cy, cz), (0.090, t, ln), DARK, (math.cos(alpha / 2), math.sin(alpha / 2), 0.0, 0.0))


HOLDER_BOXES = [
    _mm(-45, 45, -35, 35, 0, 1000 * FLOOR_T, DARK),                                         # base
    _mm(-45, 45, 15, 35, 1000 * FLOOR_T, 1000 * (FLOOR_T + SLOT_DEPTH), DARK),               # wall, +y
    _mm(-45, 45, -35, -15, 1000 * FLOOR_T, 1000 * (FLOOR_T + SLOT_DEPTH), DARK),             # wall, -y
    _mm(40, 45, -15, 15, 1000 * FLOOR_T, 1000 * (FLOOR_T + SLOT_DEPTH), DARK),               # end stop, front
    _mm(-45, -40, -15, 15, 1000 * FLOOR_T, 1000 * (FLOOR_T + SLOT_DEPTH), DARK),             # end stop, back
    _flare(1.0), _flare(-1.0),
]
HOLDER_MASS = 3.0

BAR = part("bar", pos=(RACK_X, 0.0, TOP + BAR_Z + 0.0003), mass=BAR_MASS, boxes=BAR_BOXES, cylinders=tuple(BAR_CYL),
           color=ALU, friction=0.5)
PLATE_L = part("plate_l", pos=(RACK_X, PLATE_Y, TOP + PLATE_Z + 0.0005), mass=PLATE_MASS, boxes=PLATE_BOXES,
               color=RUBBER, friction=0.5)
PLATE_R_ = part("plate_r", pos=(RACK_X, -PLATE_Y, TOP + PLATE_Z + 0.0005), mass=PLATE_MASS, boxes=PLATE_BOXES,
                color=RUBBER, friction=0.5)
HOOK_L = part("hook_l", pos=(RACK_X, HOOK_Y, TOP), mass=HOOK_MASS, boxes=HOOK_BOXES, color=DARK, friction=1.0)
HOOK_R = part("hook_r", pos=(RACK_X, -HOOK_Y, TOP), mass=HOOK_MASS, boxes=HOOK_BOXES, color=DARK, friction=1.0)
HOLDER_L = part("holder_l", pos=(RACK_X, HOLDER_Y, TOP), mass=HOLDER_MASS, boxes=HOLDER_BOXES, color=DARK, friction=1.0)
HOLDER_R = part("holder_r", pos=(RACK_X, -HOLDER_Y, TOP), mass=HOLDER_MASS, boxes=HOLDER_BOXES, color=DARK,
                friction=1.0)
OBJECTS = [BAR, PLATE_L, PLATE_R_, HOOK_L, HOOK_R, HOLDER_L, HOLDER_R]

# -- scoring thresholds --------------------------------------------------------------------------------------------------
END_LIFT = 0.015           # a bar end more than this above its seat: tipped
CRADLE_LIFT = 0.012        # the bar at a hook more than this above its seat, or this far along x: out of its cradle
BAR_SLIDE_LIMIT = 0.030    # the bar slid along y more than this: moved
SEATED = 0.010             # at the end the bar is within this of where it started (each hook point, any axis)
PLATE_FALL = 0.030         # a plate's lowest point fell this far while no gripper held it: dropped
OFF_BAR = 0.2125           # a plate centre this far from the bar's middle: its bore is clear of the sleeve end
HELD_DIST = 0.015          # a finger pad this close to a plate: held
RELEASE_GAP = 0.03         # grippers this far from plates and bar: let go
SLOT_Y_TOL = 0.012         # a standing plate's centre within this of its slot's middle line (y)
STAND_TILT = 12.0          # deg: the plate's axis within this of the y axis
FLOOR_LOW, FLOOR_HIGH = 0.004, 0.022   # lowest point of a standing plate above the table: on the slot floor
STILL_CALLS = 20

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(bar0=None, failed=False, rec=[], released=True, held={"plate_l": False, "plate_r": False},
                  z_ref={}, dropped=False, tipped=False, min_end_lift=0.0, finger_check=None, calls=0, why=[], fired_at=None)


_reset_state()


def _lowest(plate: PartState) -> float:
    """Height of the lowest point of the plate's disc above the table."""
    a = plate.rotation()[:, 1]
    return float(plate.pos[2] - TOP - PLATE_R * math.sqrt(max(0.0, 1.0 - a[2] ** 2)) - 0.5 * PLATE_T * abs(a[2]))


def plate_tilt(plate: PartState) -> float:
    """Degrees between the plate's axis and the y axis."""
    return math.degrees(math.acos(min(1.0, abs(float(plate.rotation()[1, 1])))))


def plate_dist(plate: PartState, point) -> float:
    """Distance from a world point to the plate's solid (the bore ignored)."""
    p = plate.to_local(point)
    return float(math.hypot(max(math.hypot(p[0], p[2]) - PLATE_R, 0.0), max(abs(p[1]) - PLATE_T / 2, 0.0)))


def bar_dist(bar: PartState, point) -> float:
    p = bar.to_local(point)
    return float(math.hypot(max(math.hypot(p[0], p[2]) - COLLAR_R, 0.0), max(abs(p[1]) - BAR_HALF, 0.0)))


def bar_points(bar: PartState) -> dict:
    """World positions of the bar's two ends and of the bar at each hook."""
    return {"end_l": bar.to_world((0, BAR_HALF, 0)), "end_r": bar.to_world((0, -BAR_HALF, 0)),
            "hook_l": bar.to_world((0, HOOK_Y, 0)), "hook_r": bar.to_world((0, -HOOK_Y, 0))}


def in_holder(plate: PartState, side: float) -> bool:
    """The plate stands on the floor of the holder at ``side`` (+1 left, -1 right): upright, on the slot floor."""
    dx = abs(float(plate.pos[0]) - RACK_X)
    dy = abs(float(plate.pos[1]) - side * HOLDER_Y)
    return bool(dx <= HOLDER_X_HALF and dy <= SLOT_Y_TOL and plate_tilt(plate) <= STAND_TILT
                and FLOOR_LOW <= _lowest(plate) <= FLOOR_HIGH)


def off_bar(plate: PartState) -> bool:
    return bool(abs(float(plate.pos[1])) >= OFF_BAR)


def plate_trace(objects) -> dict:
    """For the demo's debug trace: each plate's position along the bar, the bar's end lifts and the hook offsets."""
    bar = objects["bar"]
    pts, p0 = bar_points(bar), _STATE["bar0"]
    out = {"plate_l_y_mm": round(1000 * float(objects["plate_l"].pos[1]), 1),
           "plate_r_y_mm": round(1000 * float(objects["plate_r"].pos[1]), 1)}
    if p0 is not None:
        for k in ("end_l", "end_r", "hook_l", "hook_r"):
            out[f"{k}_dz_mm"] = round(1000 * float(pts[k][2] - p0[k][2]), 1)
        out["bar_dy_mm"] = round(1000 * float(bar.pos[1] - p0["centre"][1]), 1)
    out["tipped"], out["dropped"], out["failed"] = _STATE["tipped"], _STATE["dropped"], _STATE["failed"]
    return out


@on_step
def _fingers(environment) -> None:
    """Held / released: finger pads near each plate, and every pad 3 cm from both plates and the bar (score() cannot
    see the robot)."""
    from yam_arena.scene import object_states

    try:
        st = object_states(environment, ["bar", "plate_l", "plate_r"])
        pts = finger_points(environment)
        for n in ("plate_l", "plate_r"):
            _STATE["held"][n] = bool(min(plate_dist(st[n], p) for p in pts) < HELD_DIST)
        gaps = [min(plate_dist(st["plate_l"], p), plate_dist(st["plate_r"], p), bar_dist(st["bar"], p)) for p in pts]
        _STATE["released"] = bool(min(gaps) >= RELEASE_GAP)
        _STATE["finger_check"] = True
    except Exception as exc:  # noqa: BLE001 - report once; an unknown grip never counts as released or as held
        if _STATE["finger_check"] is None:
            print(f"barbell-strip-rack: finger check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"], _STATE["released"] = False, False
        _STATE["held"] = {"plate_l": True, "plate_r": True}


def _update(objects) -> tuple[float, bool]:
    bar, pl, pr = objects["bar"], objects["plate_l"], objects["plate_r"]
    pts = bar_points(bar)
    if _STATE["bar0"] is None:
        _STATE["bar0"] = {**pts, "centre": bar.pos.copy()}
        _STATE["z_ref"] = {"plate_l": _lowest(pl), "plate_r": _lowest(pr)}
    p0 = _STATE["bar0"]
    lifts = {k: float(pts[k][2] - p0[k][2]) for k in pts}
    tipped = (lifts["end_l"] > END_LIFT or lifts["end_r"] > END_LIFT
              or any(lifts[k] > CRADLE_LIFT or abs(float(pts[k][0] - p0[k][0])) > CRADLE_LIFT
                     for k in ("hook_l", "hook_r")))
    slid = abs(float(bar.pos[1] - p0["centre"][1])) > BAR_SLIDE_LIMIT
    dropped = False
    for name, plate in (("plate_l", pl), ("plate_r", pr)):
        low = _lowest(plate)
        if _STATE["held"][name]:
            _STATE["z_ref"][name] = low                     # while a hand has it the reference follows the plate
        else:
            _STATE["z_ref"][name] = max(_STATE["z_ref"][name], low)
            dropped |= _STATE["z_ref"][name] - low > PLATE_FALL
    _STATE["calls"] += 1
    for flag, hit, text in ((tipped, tipped, f"bar tipped: end lifts L {1000 * lifts['end_l']:.1f} R {1000 * lifts['end_r']:.1f} mm, "
                             f"hook lifts L {1000 * lifts['hook_l']:.1f} R {1000 * lifts['hook_r']:.1f} mm"),
                            (slid, slid, "bar slid along y"), (dropped, dropped, "plate dropped")):
        if hit and text.split(":")[0] not in [w.split(":")[0] for w in _STATE["why"]]:
            _STATE["why"].append(text)
            if _STATE["fired_at"] is None:
                _STATE["fired_at"] = _STATE["calls"]
    _STATE["tipped"] |= bool(tipped or slid)
    _STATE["dropped"] |= bool(dropped)
    _STATE["failed"] |= _STATE["tipped"] or _STATE["dropped"]
    if _STATE["failed"]:
        return 0.0, False
    both_off = off_bar(pl) and off_bar(pr)
    in_l, in_r = in_holder(pl, 1.0), in_holder(pr, -1.0)
    seated = all(abs(float(pts[k][2] - p0[k][2])) <= SEATED and abs(float(pts[k][1] - p0[k][1])) <= SEATED
                 for k in ("hook_l", "hook_r"))
    rec = _STATE["rec"]
    rec.append(np.r_[bar.pos, pl.pos, pr.pos])
    del rec[:-STILL_CALLS]
    still = len(rec) == STILL_CALLS and float(np.ptp(np.array(rec), axis=0).max()) < 0.001
    if both_off and in_l and in_r and seated:
        if _STATE["released"] and still:
            return 1.0, True
        return 0.75, False
    if both_off and seated:
        return 0.5, False
    if (in_l and off_bar(pl) and not off_bar(pr)) or (in_r and off_bar(pr) and not off_bar(pl)):
        return 0.25, False
    return 0.0, False


def success(objects) -> bool:
    """Both plates off the bar and standing upright in their own holders, the bar seated in both hooks (each hook point
    within 1 cm of its start), every finger pad 3 cm from both plates and the bar, and the bar and plates still within
    1 mm for 20 scorer calls (1 s), with no tip or drop ever."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if a bar end ever rose more than 15 mm above its seat, the bar ever rose 12 mm or moved 12 mm along x at a
    hook (out of its cradle), the bar ever slid 3 cm along y, or a plate ever fell 3 cm while no gripper held it (permanent).
    Otherwise: 0.25 one plate is off the bar and standing in its holder while the other is still on the bar; 0.5 both
    plates are off the bar and the bar is seated in its hooks (within 1 cm), but a plate is not standing in its
    holder; 0.75 both plates stand in their holders and the bar is seated, but a gripper is within 3 cm of a plate or
    the bar, or things are not still; 1.0 all of that, grippers clear and everything still for 1 s."""
    return _update(objects)[0]


def distance(objects) -> float:
    """Metres to go: each plate's centre to the middle of its holder's slot floor."""
    total = 0.0
    for name, side in (("plate_l", 1.0), ("plate_r", -1.0)):
        goal = np.array([RACK_X, side * HOLDER_Y, TOP + FLOOR_T + PLATE_R])
        total += float(np.linalg.norm(objects[name].pos - goal))
    return total
