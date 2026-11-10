"""On the station table: a training arm (a fixed pale-skin bar, 90 mm wide, 50 mm high, lying along the table's y axis
with a red 30 mm mark on its top near the elbow end) with a blue blood-pressure cuff wrapped round its upper part and
an aneroid dial clipped on the cuff (a needle on a scale with a green tick at 100 mmHg and a red tick at 160 mmHg).
To the arms' side of the cuff stands a pump unit: a black rubber inflation bulb with a squeeze plate on top, and, a
little further out, a bleed valve with a large black T-bar knob. In a grey dish near the arms' left side stands a
stethoscope chestpiece (a silver disc with a 28 mm stem) with a spring-loaded diaphragm face.

The cuff is pumped by pressing the squeeze plate down (a full squeeze, 12 mm or more, adds 42 mmHg; it leaks 0.2 mmHg/s
by itself). The valve knob has no return spring (it stays where it is set): it bleeds the cuff at 10 mmHg/s per
radian of turn (so 3 mmHg/s is about 17 degrees); past about 50 degrees it dumps the cuff at 40 mmHg/s. A listening
check is made for each step: the chestpiece counts as listening while its centre is within 15 mm of the red mark and
its diaphragm is pressed in at least 2.5 mm (about 1 N; its own weight only presses it 1.2 mm).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Hinge, PartState, Slider, on_reset, on_step, part, set_hinge_target  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Cylinder  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Take the training arm's blood pressure with the stethoscope and cuff: pump the cuff above 160 mmHg, "
               "let it down past 100 mmHg at 2 to 4 mmHg per second while listening over the red mark the whole way, "
               "let the cuff empty, then let go of everything.")

SKIN = (0.93, 0.76, 0.64)
BLUE = (0.15, 0.30, 0.75)
BLACK = (0.04, 0.04, 0.04)
GREY = (0.55, 0.56, 0.58)
SILVER = (0.82, 0.83, 0.86)
RED = (0.85, 0.08, 0.08)
GREEN = (0.10, 0.70, 0.20)
WHITE = (0.95, 0.95, 0.95)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- fixed layout (world frame: x forward from the arms, y left, metres) ---------------------------------------------
ARM_X, ARM_Y0, ARM_Y1 = 0.33, -0.26, 0.22         # the training arm along y
ARM_W, ARM_H = 0.090, 0.050
MARK_XY = (ARM_X, 0.10)                            # the red mark (artery) on the arm's top
MARK_R = 0.015
SKIN_Z = TOP + ARM_H
CUFF_Y = (-0.22, -0.08)
CUFF_H = 0.058
DISH_XY = (0.22, 0.20)
DISH_H = 0.010
PUMP_XY = (0.24, -0.20)          # pump unit origin (bottom centre of its base)
BULB_DY, KNOB_DY = 0.045, -0.055  # squeeze plate and valve stem, along y from the unit's origin
DIAL_XY = (ARM_X, -0.15)
DIAL_Z = TOP + CUFF_H            # the dial part sits on the cuff

ARM = Box("arm", pos=(ARM_X, (ARM_Y0 + ARM_Y1) / 2, TOP + ARM_H / 2), size=(ARM_W, ARM_Y1 - ARM_Y0, ARM_H),
          color=SKIN)
CUFF = Box("cuff", pos=(ARM_X, sum(CUFF_Y) / 2, TOP + CUFF_H / 2), size=(ARM_W + 0.016, CUFF_Y[1] - CUFF_Y[0], CUFF_H),
           color=BLUE)
MARK = Cylinder("mark", pos=(*MARK_XY, SKIN_Z - 0.0018), radius=MARK_R, height=0.0040, color=RED)
DISH = Cylinder("dish", pos=(*DISH_XY, TOP + DISH_H / 2), radius=0.040, height=DISH_H, color=GREY)

# -- the chestpiece: origin at the centre of its diaphragm face on whatever it stands on --------------------------------
FACE_TRAVEL = 0.008
FACE_RATE = 400.0                  # N/m: its own 0.05 kg presses it 1.2 mm
STEM_R, STEM_TOP = 0.014, 0.050    # a 28 mm stem, 50 mm above the face
STEM_GRIP_Z = 0.040                # the demo grips it 10 mm below its top
FACE = Slider("face", rest=(0.0, 0.0, 0.003), cylinders=(((0.0, 0.0, 0.003), 0.018, 0.006, SILVER),),
              travel=FACE_TRAVEL, rate=FACE_RATE, damping=6.0, mass=0.004, axis="Z", sign=1)
CHEST = part("chest", pos=(*DISH_XY, TOP + DISH_H), mass=0.05,
             boxes=[_mm(-1, 1, -1, 1, 20, 21, SILVER)],
             cylinders=[((0.0, 0.0, 0.013), 0.022, 0.014, SILVER), ((0.0, 0.0, 0.035), STEM_R, 0.030, BLACK)],
             sliders=(FACE,), color=SILVER, friction=1.0)

# -- the pump unit: origin at the centre of its base plate, on the table -------------------------------------------------
SQUEEZE_TRAVEL = 0.025
SQUEEZE_RATE = 90.0               # N/m: 2.2 N at the full 25 mm
PLATE_TOP = 0.078                 # squeeze plate top, above the table
KNOB_Z = 0.044                    # the demo grips the T-bar at this height above the table
KNOB_LEN, KNOB_W, KNOB_H = 0.085, 0.026, 0.016
SQUEEZE = Slider("squeeze", rest=(0.0, BULB_DY, 0.070), cylinders=(((0.0, BULB_DY, 0.070), 0.023, 0.016, BLACK),),
                 travel=SQUEEZE_TRAVEL, rate=SQUEEZE_RATE, damping=3.0, mass=0.02, axis="Z", sign=-1)
KNOB_STIFF = 0.0                  # no return spring: a friction valve stays where it is set
KNOB_MAX = 1.3
KNOB = Hinge("knob", pivot=(0.0, KNOB_DY, KNOB_Z),
             boxes=(_mm(-KNOB_LEN * 500, KNOB_LEN * 500, KNOB_DY * 1000 - KNOB_W * 500, KNOB_DY * 1000 + KNOB_W * 500,
                        (KNOB_Z - KNOB_H / 2) * 1000, (KNOB_Z + KNOB_H / 2) * 1000, BLACK),),
             lower=0.0, upper=KNOB_MAX, stiffness=KNOB_STIFF, damping=0.08, target=0.0, mass=0.05, axis="Z")
PUMP = part("pump", pos=(*PUMP_XY, TOP), mass=1.0,
            boxes=[_mm(-45, 45, -85, 75, 0, 12, GREY)],
            cylinders=[((0.0, BULB_DY, 0.037), 0.025, 0.050, BLACK),            # the bulb body under the plate
                       ((0.0, KNOB_DY, 0.024), 0.012, 0.024, GREY)],           # the valve stem under the T-bar
            sliders=(SQUEEZE,), hinges=(KNOB,), color=BLACK, friction=1.0)

# -- the dial: a needle that follows the cuff pressure ------------------------------------------------------------------
NEEDLE_A0 = 2.4                   # needle direction at 0 mmHg (rad from +x, counter-clockwise)
NEEDLE_SPAN = 4.0                 # rad over 0-300 mmHg
DIAL_R = 0.026


def _needle_angle(p_mmhg: float) -> float:
    return NEEDLE_SPAN * min(300.0, max(0.0, p_mmhg)) / 300.0


def _tick(p_mmhg: float, color):
    a = NEEDLE_A0 + _needle_angle(p_mmhg)
    r = 0.021
    return ((r * math.cos(a), r * math.sin(a), 0.014), (0.007, 0.0025, 0.002), color,
            (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)))


NEEDLE = Hinge("needle", pivot=(0.0, 0.0, 0.0165),
               boxes=(((0.011 * math.cos(NEEDLE_A0), 0.011 * math.sin(NEEDLE_A0), 0.0165), (0.022, 0.003, 0.002), RED,
                       (math.cos(NEEDLE_A0 / 2), 0.0, 0.0, math.sin(NEEDLE_A0 / 2))),),
               lower=-0.2, upper=NEEDLE_SPAN + 0.2, stiffness=0.02, damping=0.004, target=0.0, mass=0.002, axis="Z")
DIAL = part("dial", pos=(*DIAL_XY, DIAL_Z), mass=2.0,
            boxes=[_mm(-26, 26, -26, 26, 0, 12, WHITE), _tick(100.0, GREEN), _tick(160.0, RED)],
            hinges=(NEEDLE,), color=WHITE, friction=1.0)
OBJECTS = [ARM, CUFF, MARK, DISH, CHEST, PUMP, DIAL]

# -- mechanism and scoring numbers --------------------------------------------------------------------------------------
CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ
SQUEEZE_ON, SQUEEZE_OFF = 0.012, 0.004       # a squeeze counts at 12 mm down and re-arms at 4 mm (the plate rests 2 mm down)
PER_SQUEEZE, P_MAX, LEAK = 42.0, 300.0, 0.2  # mmHg, mmHg, mmHg/s
BLEED_PER_RAD = 10.0                          # mmHg/s per radian of knob turn
KNOB_DEAD = 0.05                              # rad: below this the valve is shut
KNOB_DUMP = 0.90                              # rad: past this the valve dumps the cuff
DUMP_RATE = 40.0
OVERPUMP = 260.0
P_HIGH, P_LOW = 160.0, 100.0                  # the reading phase runs from 160 down to 100
RATE_GOOD = (2.0, 4.0)
RATE_LOW = (1.0, 2.0)
RATE_HIGH = (4.0, 6.0)
MISS_LIMIT_S = 1.0
LISTEN_R = 0.015
LISTEN_PRESS = 0.0025
EMPTY_P = 20.0
APART = 0.03                                  # grippers this far from the chestpiece and the pump unit
CHEST_REST_R = 0.020
PUMP_MOVE_LIMIT = 0.030
PUMP_TURN_LIMIT = 20.0
STILL_CALLS = 20
STILL_TOL = 0.001

_STATE: dict = {}
_LOG: dict = {}


def _fresh():
    return dict(p=0.0, peak=0.0, squeezes=0, armed=True, phase="before", t_phase=None, t_total=0.0, t_dead=0.0,
                rate=None, listen=False, listen_s=0.0, miss_s=0.0, flags=set(), pump0=None, pump_yaw0=None,
                recent=[], final=False, still=False, bleed=0.0, apart=False, rest_ok=False)


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(_fresh())
    _LOG.clear()
    if environment is not None:
        set_hinge_target(environment, "dial", "needle", 0.0)


_reset_state()


def _yaw(p: PartState) -> float:
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


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


def bleed_rate(angle: float) -> float:
    """Cuff bleed (mmHg/s) at a knob turn (rad): shut under 0.05, 10 mmHg/s per radian, dump past 0.9."""
    if angle < KNOB_DEAD:
        return 0.0
    if angle > KNOB_DUMP:
        return DUMP_RATE
    return BLEED_PER_RAD * angle


def chest_listening(chest: PartState) -> bool:
    """Centre within 15 mm of the mark, diaphragm pressed in 2.5 mm or more."""
    d = float(np.hypot(chest.pos[0] - MARK_XY[0], chest.pos[1] - MARK_XY[1]))
    return bool(d <= LISTEN_R and chest.sliders.get("face", 0.0) >= LISTEN_PRESS)


def chest_at_rest(chest: PartState) -> bool:
    """Standing on the arm over the mark or in the dish, upright."""
    c = chest.rotation()[2, 2]
    on_arm = np.hypot(chest.pos[0] - MARK_XY[0], chest.pos[1] - MARK_XY[1]) <= CHEST_REST_R * 1.5 \
        and abs(chest.pos[2] - SKIN_Z) < 0.008
    in_dish = np.hypot(chest.pos[0] - DISH_XY[0], chest.pos[1] - DISH_XY[1]) <= 0.030 \
        and abs(chest.pos[2] - (TOP + DISH_H)) < 0.008
    return bool(c > 0.9 and (on_arm or in_dish))


def tick(chest: PartState, pump: PartState, points) -> None:
    """One control step of the cuff, the reading phase and the checks (runs from the on_step hook)."""
    S = _STATE
    S["t_total"] += DT
    squeeze = -pump.sliders.get("squeeze", 0.0)
    angle = pump.hinges.get("knob", 0.0)
    if S["pump0"] is None:
        S.update(pump0=pump.pos.copy(), pump_yaw0=_yaw(pump))
    S["win_sq"] = max(S.get("win_sq", 0.0), squeeze)
    # pumping: a full squeeze adds 30 mmHg
    if squeeze >= SQUEEZE_ON and S["armed"]:
        S["armed"] = False
        S["squeezes"] += 1
        S["p"] = min(P_MAX, S["p"] + PER_SQUEEZE)
        if S["phase"] != "before":
            S["flags"].add("pumped during the reading")
    elif squeeze <= SQUEEZE_OFF:
        S["armed"] = True
    # leak and bleed
    rate = bleed_rate(angle)
    S["bleed"] = rate
    S["p"] = max(0.0, S["p"] - (LEAK + rate) * DT)
    p = S["p"]
    S["peak"] = max(S["peak"], p)
    if p > OVERPUMP:
        S["flags"].add("overpumped")
    # the reading phase: from the first fall below 160 (after being above it) to the first fall below 100
    listening = chest_listening(chest)
    S["listen"] = listening
    if S["phase"] == "before" and S["peak"] > P_HIGH and p < P_HIGH:
        S["phase"], S["t_phase"], S["listen_s"], S["miss_s"] = "reading", 0.0, 0.0, 0.0
    if S["phase"] == "reading":
        S["t_phase"] += DT
        if listening:
            S["listen_s"] += DT
        else:
            S["miss_s"] += DT
        if S["miss_s"] > MISS_LIMIT_S:
            S["flags"].add("lost the sound")
        if p > P_HIGH + 1.0:
            S["flags"].add("pumped during the reading")
        if p < P_LOW:
            S["phase"] = "done"
            S["rate"] = (P_HIGH - P_LOW) / max(S["t_phase"], DT)
    # permanent flags for the chestpiece and the pump unit
    if chest.rotation()[2, 2] < 0.5 or chest.pos[2] < TOP + 0.006:
        S["flags"].add("chestpiece dropped")
    moved = float(np.linalg.norm((pump.pos - S["pump0"])[:2]))
    turned = abs(math.degrees((_yaw(pump) - S["pump_yaw0"] + math.pi) % (2 * math.pi) - math.pi))
    if moved > PUMP_MOVE_LIMIT or turned > PUMP_TURN_LIMIT:
        S["flags"].add("pump unit moved")
    # end state
    if points:
        c_pt = chest.pos + np.array([0.0, 0.0, 0.03])
        k_pt = pump.to_world((0.0, KNOB_DY, KNOB_Z))
        b_pt = pump.to_world((0.0, BULB_DY, PLATE_TOP))
        near = min(min(np.linalg.norm(q - c_pt) for q in points), min(np.linalg.norm(q - k_pt) for q in points),
                   min(np.linalg.norm(q - b_pt) for q in points))
        S["apart"] = bool(near >= APART)
    else:
        S["apart"] = False
    S["rest_ok"] = chest_at_rest(chest)
    recent = S["recent"]
    recent.append(np.r_[chest.pos, pump.pos, angle, chest.sliders.get("face", 0.0)])
    del recent[:-STILL_CALLS]
    S["still"] = bool(len(recent) == STILL_CALLS and all(np.max(np.abs(r - recent[-1])[:6]) < STILL_TOL
                                                          and abs(r[6] - recent[-1][6]) < 0.02 for r in recent)
)
    # log for the DEMO_DEBUG trace and result.json
    _LOG.update(p=round(p, 1), bleed=round(rate, 2), knob=round(angle, 3), squeeze_mm=round(1000 * squeeze, 1), squeeze_win_peak_mm=round(1000 * S["win_sq"], 1),
                squeezes=S["squeezes"], face_mm=round(1000 * chest.sliders.get("face", 0.0), 2),
                chest_off_mm=round(1000 * float(np.hypot(chest.pos[0] - MARK_XY[0], chest.pos[1] - MARK_XY[1])), 1),
                listening=listening, phase=S["phase"], t_phase=S["t_phase"], miss_s=round(S["miss_s"], 2),
                rate=S["rate"], flags=sorted(S["flags"]),
                near_mm=None if not points else round(1000 * near, 1))


@on_step
def _step(environment) -> None:
    from partkit import _part_state

    chest, pump = _part_state(environment, CHEST), _part_state(environment, PUMP)
    tick(chest, pump, _gripper_points(environment))
    set_hinge_target(environment, "dial", "needle", _needle_angle(_STATE["p"]))


def reading_quality() -> float:
    """0 no usable reading, 0.25 a listened reading at 1-2 or 4-6 mmHg/s, 1.0 a listened reading at 2-4 mmHg/s."""
    r = _STATE["rate"]
    if r is None or _STATE["flags"] or _STATE["miss_s"] > MISS_LIMIT_S:
        return 0.0
    if RATE_GOOD[0] <= r <= RATE_GOOD[1]:
        return 1.0
    if RATE_LOW[0] <= r < RATE_GOOD[0] or RATE_GOOD[1] < r <= RATE_HIGH[1]:
        return 0.25
    return 0.0


def _update(objects) -> tuple[float, bool]:
    S = _STATE
    q = reading_quality()
    if q == 0.0:
        return 0.0, False
    if q < 1.0:
        return 0.25, False
    if S["p"] < EMPTY_P and S["apart"] and S["rest_ok"]:
        return 1.0, bool(S["still"])
    return 0.5, False


def success(objects) -> bool:
    """A reading phase of 2-4 mmHg/s with the chestpiece listening all but 1 s, no flag, the cuff under 20 mmHg, both
    grippers 3 cm from the chestpiece and the pump unit, the chestpiece upright on the arm or in the dish, and nothing
    changing for 20 scorer calls (1 s)."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 on any flag (cuff above 260 mmHg, chestpiece not listening for over 1 s of the 160-to-100 fall, pumped again
    during it, chestpiece dropped, pump unit moved 3 cm or turned 20 degrees), if the cuff never got above 160, or the
    fall was faster than 6 mmHg/s or slower than 1; 0.25 a listened fall at 1-2 or 4-6 mmHg/s; 0.5 a listened fall at
    2-4 mmHg/s but the cuff is still above 20 mmHg, or a gripper is within 3 cm of the chestpiece or pump unit, or
    the chestpiece is not at rest; 1.0 all of that done and still for 1 s."""
    return _update(objects)[0]


def distance(objects) -> float:
    """mmHg left to bleed to the empty mark, as a distance-like number (zero when emptied)."""
    return float(max(0.0, _STATE["p"] - EMPTY_P)) / 1000.0
