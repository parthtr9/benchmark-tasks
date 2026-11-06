"""On the station table: a heavy grey bench vise (200 x 220 mm base, 6 kg) with a fixed jaw on its right and a grey
sliding jaw on its left, a 70 mm gap between them, and a light wooden block (40 x 50 x 90 mm, 150 g) standing on the
table to the right of it. The sliding jaw is a quick-action jaw: pushed toward the fixed one it slides freely, and a
ratchet stops it opening again, with a light preload (about 6 N) on whatever it has closed on. The jaws stand 115 mm
tall on the base; the gap has no floor at working height, a block let go there falls to the base, 15 mm off the
table.

The robot must carry the block into the gap with one arm and hold it standing upright so that its top is 25 to 45 mm
above the tops of the jaws, push the sliding jaw shut against it with the other arm, and then let go of both. The
clamp, not the hands, must keep the block up. Let go before the jaw has closed and the block drops to the base.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import PartState, Slider, on_reset, on_step, part, set_slider_target  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
TOP = TABLE_TOP[SCENE]

INSTRUCTION = ("Hold the wooden block upright between the vise jaws so its top stands 25 to 45 mm above the tops of "
               "the jaws, push the grey sliding jaw shut against it with the other arm, then let go of both.")

GREY = (0.45, 0.46, 0.48)
DARK = (0.22, 0.23, 0.25)
WOOD = (0.86, 0.70, 0.45)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                     (z1 - z0) / 1000), color)


# -- the vise: origin at the middle of the underside of its base; x right (away from the arms), y left --------------
BASE_Z = 0.015                      # base top
JAW_TOP = 0.115                     # tops of both jaws
TAB_FACE, TAB_X, TAB_TOP = 0.095, 0.035, 0.156   # the push tab's outer face, middle along the jaw, top
FIXED_FACE, SLIDE_FACE = -0.025, 0.045   # inner faces of the fixed jaw and (open) sliding jaw: a 70 mm gap
JAW_HALF_X = 0.050
TRAVEL = 0.045
VISE_MASS = 6.0
SLIDER_MASS = 0.3
VISE_BOXES = [
    _mm(-100, 100, -110, 110, 0, 15, DARK),        # base
    _mm(-50, 50, -55, -25, 15, 115, GREY),         # fixed jaw
]
JAW = Slider("jaw", rest=(0.0, 0.060, 0.0675), boxes=(_mm(-50, 50, 45, 75, 20, 115, GREY),
                                       _mm(20, 50, 75, 95, 20, 156, DARK)),   # + a push tab on the outer side
             travel=TRAVEL, rate=3000.0, damping=50.0, mass=SLIDER_MASS, axis="Y", sign=-1)

# -- the block: origin at the middle of its underside --------------------------------------------------------------
BLOCK_X, BLOCK_Y, BLOCK_Z = 0.040, 0.050, 0.090     # x across the jaws' length, y across the gap, z up
BLOCK_MASS = 0.15
BLOCK_BOXES = [((0.0, 0.0, BLOCK_Z / 2), (BLOCK_X, BLOCK_Y, BLOCK_Z), WOOD)]

# -- fixed layout (world frame: x forward from the arms, y left, metres) -------------------------------------------
VISE_XY = (0.30, -0.06)
VISE_YAW = math.pi / 2              # turned 90 degrees: the gap opens toward the right arm, the sliding jaw faces the arms
BLOCK_START = (0.22, -0.26)
BLOCK_YAW = math.pi / 2             # its 40 mm faces toward the arms' sides: the fingers close on them
VISE = part("vise", pos=(*VISE_XY, TOP), yaw=VISE_YAW, mass=VISE_MASS, boxes=VISE_BOXES, sliders=(JAW,), color=GREY, friction=1.0)
BLOCK = part("block", pos=(*BLOCK_START, TOP), yaw=BLOCK_YAW, mass=BLOCK_MASS, boxes=BLOCK_BOXES, color=WOOD, friction=0.9)
OBJECTS = [VISE, BLOCK]

# -- the ratchet -----------------------------------------------------------------------------------------------------
PRELOAD = 0.002                     # m past first contact: 6 N at 3000 N/m
CLAMP_SLACK = 0.001                 # clamped: gap within 1 mm of the block's width

# -- scoring thresholds ------------------------------------------------------------------------------------------------
BAND = (0.025, 0.045)               # block top above the jaw tops
LIFT_MIN = 0.005                    # block bottom above the base
TILT_LIMIT = 5.0                    # deg
YAW_LIMIT = 5.0
VISE_MOVE_LIMIT = 0.020
VISE_TURN_LIMIT = 5.0
STILL_CALLS = 20

_STATE: dict = {}
_MECH: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(vise0=None, yaw0=None, failed=False, recent=[])
    _MECH.update(low=0.0, contact=None)
    if environment is not None:
        set_slider_target(environment, "vise", "jaw", 0.0)


_reset_state()


def _yaw(p: PartState) -> float:
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def _wrap(a: float, period: float) -> float:
    return (a + period / 2) % period - period / 2


def jaw_pos(vise: PartState) -> float:
    """The sliding jaw's position (m): 0 open, negative as it closes."""
    return float(vise.sliders.get("jaw", 0.0))


def gap(vise: PartState) -> float:
    """The width (m) between the jaws' inner faces."""
    return SLIDE_FACE - FIXED_FACE + jaw_pos(vise)


def block_in_vise(vise: PartState, block: PartState) -> dict:
    """The block in the vise frame: its middle (x, y, z), whether it is between the jaws, its bottom's height above the
    base, its top's height above the jaw tops, its tilt and its yaw relative to the vise (deg)."""
    mid = vise.to_local(block.to_world((0.0, 0.0, BLOCK_Z / 2)))
    bottom = vise.to_local(block.to_world((0.0, 0.0, 0.0)))
    top = vise.to_local(block.to_world((0.0, 0.0, BLOCK_Z)))
    between = bool(abs(mid[0]) <= JAW_HALF_X and FIXED_FACE <= mid[1] <= SLIDE_FACE + jaw_pos(vise)
                   and bottom[2] > BASE_Z - 0.005)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, block.rotation()[2, 2]))))
    yaw = abs(math.degrees(_wrap(_yaw(block) - _yaw(vise), math.pi)))
    return dict(mid=mid, between=between, lift=float(bottom[2] - BASE_Z), over=float(top[2] - JAW_TOP), tilt=tilt,
                yaw=yaw)


def is_clamped(vise: PartState, block: PartState) -> bool:
    b = block_in_vise(vise, block)
    return b["between"] and gap(vise) <= BLOCK_Y + CLAMP_SLACK and b["lift"] >= LIFT_MIN


@on_step
def _ratchet(environment) -> None:
    """The quick-action jaw: it closes freely and never opens (its drive target follows its most closed position), and
    once it has closed on the block it keeps a 2 mm preload (about 6 N) against it."""
    from partkit import _part_state

    vise = _part_state(environment, VISE)
    block = _part_state(environment, BLOCK)
    q = jaw_pos(vise)
    _MECH["low"] = min(_MECH["low"], q)
    target = _MECH["low"]
    b = block_in_vise(vise, block)
    if b["between"] and gap(vise) <= BLOCK_Y + CLAMP_SLACK:
        if _MECH["contact"] is None:
            _MECH["contact"] = q
        target = min(target, _MECH["contact"] - PRELOAD)
    else:
        _MECH["contact"] = None
    set_slider_target(environment, "vise", "jaw", target)


def _update(objects) -> tuple[float, bool]:
    vise, block = objects["vise"], objects["block"]
    if _STATE["vise0"] is None:
        _STATE.update(vise0=vise.pos.copy(), yaw0=_yaw(vise))
    moved = float(np.linalg.norm((vise.pos - _STATE["vise0"])[:2]))
    turned = abs(math.degrees(_wrap(_yaw(vise) - _STATE["yaw0"], 2 * math.pi)))
    _STATE["failed"] |= moved > VISE_MOVE_LIMIT or turned > VISE_TURN_LIMIT
    if _STATE["failed"]:
        return 0.0, False
    b = block_in_vise(vise, block)
    recent = _STATE["recent"]
    recent.append(np.r_[b["mid"], jaw_pos(vise)])
    del recent[:-STILL_CALLS]
    still = len(recent) == STILL_CALLS and float(np.ptp(np.array(recent), axis=0).max()) < 0.001
    if not b["between"]:
        return 0.0, False
    if not is_clamped(vise, block):
        return 0.25, False
    if not BAND[0] <= b["over"] <= BAND[1]:
        return 0.5, False
    if b["tilt"] > TILT_LIMIT or b["yaw"] > YAW_LIMIT:
        return 0.75, False
    return 1.0, bool(still)


def success(objects) -> bool:
    """The block clamped between the jaws (gap within 1 mm of its width, bottom 5 mm or more above the base) with its
    top 25-45 mm above the jaw tops, standing square (tilt and yaw within 5 degrees), still for 20 scorer calls."""
    return _update(objects)[1]


def score(objects) -> float:
    """At the end: 0.25 the block is between the jaws (even dropped onto the base); 0.5 clamped (gap within 1 mm of
    its width, bottom 5 mm or more above the base) but its top outside 25-45 mm above the jaw tops; 0.75 clamped with
    its top in that band but tilted or turned more than 5 degrees; 1.0 clamped, in the band, square. 0 if the vise
    ever moved 2 cm or turned 5 degrees, or the block is not between the jaws."""
    return _update(objects)[0]


def distance(objects) -> float:
    """Metres to go: the block's top to the middle of the band, plus how far the jaw gap is above the block's width."""
    vise, block = objects["vise"], objects["block"]
    b = block_in_vise(vise, block)
    off = float(np.linalg.norm([b["mid"][0], b["mid"][1] - (FIXED_FACE + BLOCK_Y / 2)]))
    return off + abs(b["over"] - (BAND[0] + BAND[1]) / 2) + max(0.0, gap(vise) - BLOCK_Y)
