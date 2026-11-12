"""On the station table, in front of the left arm: a grey service panel (160 x 340 x 25 mm, 3 kg) carrying four
breakers side by side, 90 mm apart along y. Each breaker is a white body (60 x 44 x 6 mm) with a red strip across its
+x end (ON) and a green strip across its -x end (OFF), and a black toggle (16 x 20 x 32 mm) that slides along x on the
body, +-14 mm. Sliding it toward the red end switches the breaker on, toward the green end off. Breakers 1, 2 and 4
(y -135, -45 and +135 mm) start ON (toggle at +12 mm). Breaker 3 (y +45 mm) has tripped: its toggle sits midway.

A tripped breaker is trip-free. Pushed straight to the red end it springs back to the middle and stays tripped; it has
to be pushed fully to OFF (past -8 mm) first, and then to ON (past +8 mm) like any other breaker. A good breaker
switches off at -8 mm and on again at +8 mm; each toggle is held by a light spring (60 N/m: it stays where it was
pushed, and a finger brushing it does not flip it).

A grey cover plate (170 x 350 x 4 mm, 250 g) lies over the breakers, 3 mm above the toggles, on two low posts. It slides
along x on a spring (20 N/m) that pulls it shut. A small upright tab on its front edge, on the right half, is for
pushing it open. A toggle can only be reached once the cover is pushed back by about 140 mm, and it slides shut as soon
as it is let go: one arm has to hold it open while the other arm works the breaker.


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

INSTRUCTION = "On the de-energised training panel, reset the tripped breaker behind the grey cover and leave every breaker on. ON is the red end of each breaker."

GREY = (0.55, 0.56, 0.58)
WHITE = (0.93, 0.93, 0.93)
RED = (0.85, 0.08, 0.08)
GREEN = (0.10, 0.65, 0.20)
BLACK = (0.05, 0.05, 0.06)


def _mm(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2000, (y0 + y1) / 2000, (z0 + z1) / 2000), ((x1 - x0) / 1000, (y1 - y0) / 1000,
                                                                      (z1 - z0) / 1000), color)


# -- the panel: origin at the middle of its underside, on the table; x away from the arms, y left -------------------
SLAB_H = 0.025
BODY_H = 0.006
PIVOT_Z = SLAB_H + BODY_H                   # the toggles sit on the breaker bodies: 31 mm up
BREAKER_Y = (-0.135, -0.045, 0.045, 0.135)
TRIPPED = 2                                 # index of the tripped breaker (breaker 3, y +45 mm)
GOOD = tuple(i for i in range(4) if i != TRIPPED)
HANDLE_H, HANDLE_W, HANDLE_D = 0.032, 0.020, 0.016   # toggle height, width along y, depth along x
TRAVEL = 0.014                               # m: the toggle's stops, +-14 mm along x (+x is the red end, ON)
ON_POS, OFF_POS, MID_POS = 0.012, -0.012, 0.0   # drive targets of an on, an off, a tripped breaker
SWITCH_AT = 0.008                            # a toggle passing +-8 mm switches
TOG_RATE, TOG_DAMP, TOG_MASS = 60.0, 3.0, 0.02    # N/m, N s/m, kg (the pad tip can push only about 3 N: the spring must stay well under that over 20 mm)

PANEL_BOXES = [_mm(-80, 80, -170, 170, 0, 25, GREY),
               _mm(-70, -58, -145, -133, 25, 65.5, GREY), _mm(-70, -58, 133, 145, 25, 65.5, GREY)]   # posts, 0.5 mm under the cover
TOGGLES = []
for _i, _y in enumerate(BREAKER_Y):
    _y_mm = 1000 * _y
    PANEL_BOXES += [_mm(-30, 30, _y_mm - 22, _y_mm + 22, 25, 31, WHITE),
                    _mm(20, 30, _y_mm - 22, _y_mm + 22, 31, 32, RED),
                    _mm(-30, -20, _y_mm - 22, _y_mm + 22, 31, 32, GREEN)]
    # Slider shapes are given in the panel frame at the slider's rest position (the toggle's middle, x 0).
    TOGGLES.append(Slider(f"t{_i + 1}", rest=(0.0, _y, PIVOT_Z + HANDLE_H / 2),
                          boxes=(_mm(-8, 8, _y_mm - 10, _y_mm + 10, 31.5, 63.5, BLACK),),
                          travel=TRAVEL, rate=TOG_RATE, damping=TOG_DAMP, mass=TOG_MASS, axis="X", sign=1,
                          symmetric=True))
PANEL_MASS = 3.0

# -- the cover: a plate sliding along x on a spring that pulls it shut (closed: front edge at x -85 mm, 3 mm over the
# handles). Its tab (8 mm thick, 60 wide, 25 tall) stands on the front edge over the right half, at y -130 mm. ------
COVER_Z0, COVER_T = 0.066, 0.004
COVER_FRONT = -0.085                         # the front edge's x when closed (panel frame); the plate is 170 mm long
COVER_TRAVEL = 0.17
COVER_OPEN = 0.14                            # the handles are reachable once it has slid back this far
COVER_HOLD = 0.165                           # the demo holds it here
COVER_RATE, COVER_DAMP, COVER_MASS = 20.0, 5.0, 0.25
TAB_T, TAB_H, TAB_Y = 0.008, 0.025, -0.130
COVER = Slider("cover", rest=(0.0, 0.0, COVER_Z0 + COVER_T / 2),
               boxes=(_mm(-85, 85, -175, 175, 1000 * COVER_Z0, 1000 * (COVER_Z0 + COVER_T), GREY),
                      _mm(-85, -77, 1000 * TAB_Y - 30, 1000 * TAB_Y + 30, 1000 * (COVER_Z0 + COVER_T),
                          1000 * (COVER_Z0 + COVER_T + TAB_H), WHITE)),
               travel=COVER_TRAVEL, rate=COVER_RATE, damping=COVER_DAMP, mass=COVER_MASS, axis="X", sign=1)

# -- fixed layout (world frame: x forward from the arms, y left, metres) -------------------------------------------
PANEL_XY = (0.28, 0.0)
PANEL = part("panel", pos=(*PANEL_XY, TOP), mass=PANEL_MASS, boxes=PANEL_BOXES, sliders=(*TOGGLES, COVER), color=GREY,
             friction=1.0)
OBJECTS = [PANEL]

# -- scoring thresholds --------------------------------------------------------------------------------------------
TOWARD_OFF = 0.005               # breaker 3 must be pushed at least this far toward OFF (to -5 mm) at some point
PANEL_MOVE_LIMIT = 0.020
PANEL_TURN_LIMIT = 5.0
STILL_CALLS = 20
STILL_POS = 0.002

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(pos0=None, yaw0=None, failed=False, recent=[], min3=0.0, good_off_ever=False,
                  mode=["tripped" if i == TRIPPED else "on" for i in range(4)])
    if environment is not None:
        for i in range(4):
            set_slider_target(environment, "panel", f"t{i + 1}", MID_POS if i == TRIPPED else ON_POS)


_reset_state()


def _advance(angles) -> None:
    """The breakers' logic on the toggle positions (m). on -> off at -8 mm, off -> on at +8 mm. A tripped breaker
    only goes to off (at -8 mm); pushed toward ON it stays tripped. Records whether a good breaker was ever off."""
    mode = _STATE["mode"]
    for i, a in enumerate(angles):
        if mode[i] == "on" and a <= -SWITCH_AT:
            mode[i] = "off"
            _STATE["good_off_ever"] |= i in GOOD
        elif mode[i] == "off" and a >= SWITCH_AT:
            mode[i] = "on"
        elif mode[i] == "tripped" and a <= -SWITCH_AT:
            mode[i] = "off"
    _STATE["min3"] = min(_STATE["min3"], float(angles[TRIPPED]))


def _target(mode: str) -> float:
    return {"on": ON_POS, "off": OFF_POS}.get(mode, MID_POS)


@on_step
def _breakers(environment) -> None:
    """After every control step: switch the breakers whose handles passed a threshold and aim each handle's drive at
    its breaker's state (on +0.44, off -0.44, tripped 0)."""
    from partkit import _part_state

    state = _part_state(environment, PANEL)
    _advance(angles_of(state))
    for i, mode in enumerate(_STATE["mode"]):
        set_slider_target(environment, "panel", f"t{i + 1}", _target(mode))


def angles_of(panel: PartState) -> list:
    """The four toggles' positions along x (m, positive toward the red end)."""
    return [float(panel.sliders.get(f"t{i + 1}", 0.0)) for i in range(4)]


def _yaw(p: PartState) -> float:
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def _update(objects) -> tuple[float, bool]:
    panel = objects["panel"]
    if _STATE["pos0"] is None:
        _STATE["pos0"], _STATE["yaw0"] = panel.pos.copy(), _yaw(panel)
    moved = float(np.linalg.norm((panel.pos - _STATE["pos0"])[:2]))
    turned = abs(math.degrees((_yaw(panel) - _STATE["yaw0"] + math.pi) % (2 * math.pi) - math.pi))
    _STATE["failed"] |= moved > PANEL_MOVE_LIMIT or turned > PANEL_TURN_LIMIT
    angles = angles_of(panel)
    _advance(angles)
    recent = _STATE["recent"]
    recent.append(angles)
    del recent[:-STILL_CALLS]
    still = len(recent) == STILL_CALLS and bool((np.ptp(np.array(recent), axis=0) < STILL_POS).all())
    mode = _STATE["mode"]
    if _STATE["failed"] or any(mode[i] == "off" for i in GOOD) or _STATE["min3"] > -TOWARD_OFF:
        return 0.0, False
    if mode[TRIPPED] == "on":
        if _STATE["good_off_ever"]:
            return 0.75, False
        return 1.0, still
    return (0.5 if mode[TRIPPED] == "off" else 0.25), False


def success(objects) -> bool:
    """Breaker 3 on again, no good breaker ever switched off, the panel in place, every handle still for 1 s."""
    return _update(objects)[1]


def score(objects) -> float:
    """0 if the panel moved 2 cm or turned 5 degrees, a good breaker (1, 2 or 4) is off at the end, or breaker 3's
    toggle never went 5 mm toward OFF; 0.25 pushed toward OFF but not reset (still tripped); 0.5 reset to OFF but
    ends off; 0.75 breaker 3 ends ON but a good breaker was off at some point; 1.0 breaker 3 on and no good breaker
    ever off. A breaker is off once its toggle passes -8 mm and on again past +8 mm."""
    return _update(objects)[0]


def distance(objects) -> float:
    """Toggle travel (m) still to go for breaker 3: down to -8 mm if tripped, then up to +8 mm."""
    panel = objects["panel"]
    a = angles_of(panel)[TRIPPED]
    mode = _STATE["mode"][TRIPPED]
    if mode == "on":
        return 0.0
    if mode == "off":
        return max(0.0, SWITCH_AT - a)
    return max(0.0, a + SWITCH_AT) + 2 * SWITCH_AT


def cover_open(panel: PartState) -> float:
    """How far the cover has slid back from shut (m)."""
    return float(panel.sliders.get("cover", 0.0))


def handle_pos(panel: PartState, k: int) -> float:
    return angles_of(panel)[k]
