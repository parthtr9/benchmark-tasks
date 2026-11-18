"""On the station counter: two Vollrath 52642 perforated nylon flatware cylinders stacked mouth to mouth. The lower one
stands upright and holds 14 washed teaspoons standing handles-down, bowls up; the upper one is empty and sits upside down
on it, the two collars together. The spoons must end handles-up in the clean cylinder, standing on the counter, with the
old cylinder set down beside it, and no spoon touched or spilled.

Sizes and sources: cylinder 95 mm across, 143 mm high, 0.154 kg (Vollrath's listing); its 112 mm collar from the listed
4.4 in width. Collar and wall thickness (4 mm, 2 mm) are estimates from photographs (measure the bought part). Teaspoon:
Winco 0005-01 "Dots" heavyweight, listed 6-1/4 in long, 1-5/16 in wide, 0.9 lb a dozen (159 mm, 33 mm, 34 g); bowl
length and depth and the handle's section are estimates.

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
from partkit import finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = ("Leave every spoon handles-up in the clean top cylinder standing on the counter, with the other cylinder "
               "set down beside it and no spoon touched or spilled.")

NYLON = (0.93, 0.93, 0.92)
STEEL = (0.78, 0.79, 0.81)

# -- the cylinder (frame: origin on its axis at its closed bottom; the mouth at +z) ---------------------------------------
CYL_R = 0.0475            # body radius (95 mm)
CYL_H = 0.143
WALL = 0.002
COLLAR_R = 0.056          # 112 mm collar at the mouth
COLLAR_T = 0.004
BOTTOM_T = 0.003
SIDES = 16
CYL_MASS = 0.154


def _quat_z(a):
    return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))


def _ring(r_in, r_out, z0, z1, color, sides=SIDES):
    """A ring about z from ``sides`` boxes (the polygon's flats on r_out: never smaller than the real part outside)."""
    out = []
    chord = 2 * r_out * math.tan(math.pi / sides) * 1.02
    for k in range(sides):
        a = 2 * math.pi * k / sides
        rm = (r_in + r_out) / 2
        out.append(((rm * math.cos(a), rm * math.sin(a), (z0 + z1) / 2), (r_out - r_in, chord, z1 - z0), color, _quat_z(a)))
    return out


def _disc(r, z0, z1, color, sides=SIDES):
    """A polygonal disc about z (union of bars), corners on r."""
    out = []
    half = math.pi / sides
    rr, a = r * math.cos(half), r * math.sin(half)
    for k in range(sides // 2):
        ang = k * 2 * half
        out.append(((0.0, 0.0, (z0 + z1) / 2), (2 * rr, 2 * a, z1 - z0), color, _quat_z(ang)))
    return out


def cylinder_boxes():
    return (_disc(CYL_R, 0.0, BOTTOM_T, NYLON)
            + _ring(CYL_R - WALL, CYL_R, BOTTOM_T, CYL_H - COLLAR_T, NYLON)
            + _ring(CYL_R - WALL, COLLAR_R, CYL_H - COLLAR_T, CYL_H, NYLON))


# -- a teaspoon (frame: origin at its handle's end; the bowl at +z) --------------------------------------------------------
# Winco 0005-01 "Dots" heavyweight 18/0 teaspoon: listed 6-1/4 in long, 1-5/16 in wide, 0.9 lb a dozen. The bowl's length
# and depth and the handle's section are estimates from photographs.
SPOON_L = 0.159
SPOON_MASS = 0.034
BOWL_L, BOWL_W, BOWL_T = 0.048, 0.033, 0.008
HANDLE_W, HANDLE_T = 0.010, 0.0025


def spoon_boxes():
    return [((0.0, 0.0, (SPOON_L - BOWL_L) / 2), (HANDLE_T, HANDLE_W, SPOON_L - BOWL_L), STEEL),
            ((0.0, 0.0, SPOON_L - BOWL_L / 2), (BOWL_T, BOWL_W, BOWL_L), STEEL)]


# -- layout ----------------------------------------------------------------------------------------------------------------
STACK_X, STACK_Y = 0.35, 0.0
SEAM_Z = CYL_H                         # the collars meet 143 mm above the table
# The spoons stand handles-down, bowls up, in rows across x like loosely packed spoons (flat bowls cannot nest): two
# side by side in the middle rows, one in the outer rows, every bowl at least 1 mm from its neighbours and 2 mm from the wall.
SPOON_XY = [(dx, dy) for dx in (0.0, 0.009, -0.009, 0.018, -0.018) for dy in (-0.0171, 0.0171)] + \
    [(dx, 0.0) for dx in (0.027, -0.027, 0.036, -0.036)]
N_SPOONS = len(SPOON_XY)
OLD_POS = (STACK_X, STACK_Y, TOP)                                      # upright, holding the spoons
NEW_POS = (STACK_X, STACK_Y, TOP + 2 * CYL_H)                          # upside down on it (its bottom at the top)
NEW_QUAT = (1.0, 0.0, 0.0, 0.0)                                        # half a turn about x (xyzw)
if os.environ.get("FLATWARE_LAYOUT") == "pour":
    # Rehearsal only (not the task): the one-arm route after its first steps, the empty cylinder already stood upright
    # beside the full one, spaced for one hand to pour from one into the other. Checks whether one arm can score 1.0.
    OLD_POS = (0.45, STACK_Y, TOP)
    NEW_POS, NEW_QUAT = (0.30, STACK_Y, TOP), (0.0, 0.0, 0.0, 1.0)

OLD = part("old_cylinder", pos=OLD_POS, mass=CYL_MASS, boxes=cylinder_boxes(), color=NYLON, friction=0.5)
NEW = part("new_cylinder", pos=NEW_POS, mass=CYL_MASS, boxes=cylinder_boxes(), color=NYLON, friction=0.5)
NEW.meta["quat_xyzw"] = NEW_QUAT

SPOONS = []
for i, (dx, dy) in enumerate(SPOON_XY):
    pos = (OLD_POS[0] + dx, OLD_POS[1] + dy, TOP + BOTTOM_T + 0.001)
    SPOONS.append(part(f"spoon_{i}", pos=pos, mass=SPOON_MASS, boxes=spoon_boxes(), color=STEEL, friction=0.4))

OBJECTS = [OLD, NEW, *SPOONS]

# -- scoring ---------------------------------------------------------------------------------------------------------------
TOUCH = 0.005             # a gripper finger point this close to a spoon's surface touches it
RELEASE = 0.03            # both grippers' finger points this far from both cylinders: let go
STILL_STEPS = 20
STILL_MM = 1.0
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(step=0, released=False, free_new=False, violation="", hist=[])


_reset_state()


def _axis(st) -> np.ndarray:
    """A cylinder's axis toward its mouth (world)."""
    return st.rotation() @ np.array([0.0, 0.0, 1.0])


def _inside(spoon_st, cyl_st) -> bool:
    """The spoon's middle within the cylinder's body."""
    local = cyl_st.to_local(spoon_st.to_world(np.array([0.0, 0.0, SPOON_L / 2])))
    return math.hypot(local[0], local[1]) < CYL_R and -0.005 < local[2] < CYL_H + 0.02


_RIMS = np.array([(r * math.cos(a), r * math.sin(a), z) for r, z in ((CYL_R, 0.0), (COLLAR_R, CYL_H))
                  for a in np.linspace(0.0, 2 * math.pi, 32, endpoint=False)])


def low_z(cyl_st) -> float:
    """A cylinder's lowest point (world z) from its two rims: the closed bottom's edge and the collar's (partkit's
    ``low`` is a box about the part's origin, here the closed bottom, so it is wrong once the cylinder is upside down)."""
    return float(min(cyl_st.to_world(p)[2] for p in _RIMS))


def spoon_up(spoon_st) -> bool:
    """Handle up: the spoon's bowl end below its handle end."""
    return float(_axis(spoon_st)[2]) < -0.7


def levels(objects) -> tuple:
    """(moved, up, set down): every spoon inside the new cylinder with that cylinder upright; and every spoon handle up;
    and the new cylinder standing on the table with the other one set down on the table off it."""
    new, old = objects["new_cylinder"], objects["old_cylinder"]
    spoons = [objects[s.name] for s in SPOONS]
    upright = float(_axis(new)[2]) > 0.95
    moved = upright and all(_inside(s, new) for s in spoons)
    up = moved and all(spoon_up(s) for s in spoons)
    new_down = low_z(new) < TOP + 0.01
    old_off = (float(np.linalg.norm(np.asarray(old.pos[:2]) - np.asarray(new.pos[:2]))) > COLLAR_R * 2 + 0.01
               and low_z(old) < TOP + 0.01)
    return moved, up, up and new_down and old_off


@on_step
def _step(environment) -> None:
    from partkit import _part_state   # noqa: PLC0415

    _STATE["step"] += 1
    pts = finger_points(environment)
    for spec in SPOONS:
        st = _part_state(environment, spec)
        if any(spoon_gap(p, st) < TOUCH for p in pts):
            _STATE["violation"] = _STATE["violation"] or f"gripper touched {spec.name}"
    cyls = [_part_state(environment, s) for s in (OLD, NEW)]
    _STATE["released"] = all(gap(p, c) > RELEASE for p in pts for c in cyls)
    # a level counts only once the grippers have let go of what it credits (review rule 6): the new cylinder for 0.25
    # and 0.5, both cylinders for 0.75
    _STATE["free_new"] = all(gap(p, cyls[1]) > RELEASE for p in pts)


def spoon_gap(p, spoon_st) -> float:
    """A point's distance from a spoon's surface (its handle and bowl boxes)."""
    local = spoon_st.to_local(np.asarray(p, float))
    out = []
    for centre, size, _color in spoon_boxes():
        d = np.maximum(np.abs(local - np.asarray(centre)) - np.asarray(size) / 2, 0.0)
        out.append(float(np.linalg.norm(d)))
    return min(out)


def gap(p, cyl_st) -> float:
    """A point's distance from a cylinder's outside (a solid as wide as its collar, closed at both ends)."""
    local = cyl_st.to_local(np.asarray(p, float))
    dr = max(0.0, math.hypot(local[0], local[1]) - COLLAR_R)
    dz = max(0.0, -local[2], local[2] - CYL_H)
    return math.hypot(dr, dz)


def _update(objects) -> tuple[float, bool]:
    spoons = [objects[s.name] for s in SPOONS]
    if not _STATE["violation"]:
        for spec, s in zip(SPOONS, spoons):
            if not (_inside(s, objects["new_cylinder"]) or _inside(s, objects["old_cylinder"])):
                _STATE["violation"] = f"{spec.name} spilled"
                break
    hist = _STATE["hist"]
    hist.append(np.concatenate([np.asarray(objects[n].pos, float) for n in ("new_cylinder", "old_cylinder")]
                               + [np.asarray(s.pos, float) for s in spoons]).reshape(-1, 3))
    del hist[:-STILL_STEPS]
    if _STATE["violation"]:
        return 0.0, False
    moved, up, done_state = levels(objects)
    free_new, released = _STATE["free_new"], _STATE["released"]
    s = 0.25 * (moved and free_new) + 0.25 * (up and free_new) + 0.25 * (done_state and released)
    still = (len(hist) == STILL_STEPS
             and max(float(np.linalg.norm(h - hist[-1], axis=1).max()) for h in hist) <= STILL_MM / 1000)
    done = done_state and _STATE["released"] and still
    return (1.0 if done else s), done


def _once(objects) -> tuple[float, bool]:
    """``_update`` once per control step: success() and score() are both called each step, and the stillness window must
    hold one pose per step (20 steps = 1 s), not one per call."""
    if _STATE.get("updated_at") != _STATE["step"]:
        _STATE["updated_at"], _STATE["last"] = _STATE["step"], _update(objects)
    return _STATE["last"]


def success(objects) -> bool:
    return _once(objects)[1]


def distance(objects) -> float:
    return float(abs(1.0 - _axis(objects["new_cylinder"])[2]))


def score(objects) -> float:
    return _once(objects)[0]


def demo():
    """The scripted demonstration is in demo.py."""
    return []
