"""Bench fixture for exam-stirrup-set: printable parts, dimensioned drawings and a priced bill of materials.

    uv run --no-project --with numpy --with manifold3d python fabrication/make_fixture.py      (from the task folder)

Every dimension is read from the task file (../task.py), so the fixture and the simulated task are the same object:
the bars' height and
spacing, the boss's place and swing stops, the lift spring's rate and preload, the draw, the window, the guide
bracket's groove, notches and teeth, the support rail, the stirrup's shape and mass.

Writes, next to this file:
  stl/*.stl            printable parts (binary, millimetres; print in PETG, 4 walls, 40 % infill unless noted)
  drawings/*.svg       dimensioned drawings (millimetres)
  BOM.csv              bill of materials with prices and the total
  cut_list.csv         plywood and stock cuts
  fixture_data.json    the same as data (dimensions, pieces, bill of materials, forces, checks)
  checks.txt           the geometry checks this script ran (open stop angle, clearances)
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import pathlib
import struct
import sys

import numpy as np
import manifold3d as m3

HERE = pathlib.Path(__file__).resolve().parent
TASK_DIR = HERE.parent
sys.path.insert(0, str(TASK_DIR))                                         # partkit and meshkit
try:                                                                      # the team's shared stub of yam_arena
    sys.path.insert(0, str(TASK_DIR.parents[1] / "tests"))
    import _stub_yam  # noqa: E402
    _stub_yam.install()
except ImportError:                                                       # stand-alone: a minimal stub
    import types
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Box:
        name: str
        pos: tuple
        color: tuple = (0.7, 0.7, 0.7)
        mass: float | None = None
        yaw: float = 0.0
        size: tuple = (0.04, 0.04, 0.04)

        def _check(self) -> None:
            pass

    _pkg, _tf = types.ModuleType("yam_arena"), types.ModuleType("yam_arena.task_file")
    _pkg.__path__ = []
    _tf.Box, _tf.ObjectState, _tf.TABLE_TOP = _Box, object, {"station": 0.7475, "tabletop": 0.75}
    _pkg.task_file = _tf
    sys.modules["yam_arena"], sys.modules["yam_arena.task_file"] = _pkg, _tf
_spec = importlib.util.spec_from_file_location("stirrup_task", TASK_DIR / "task.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

MM = 1000.0
BASE_T = 18.0           # 3/4 in plywood base board (actual 18 mm)
PLY_T = 6.0             # 1/4 in plywood panels (actual about 5.5-6 mm)
CHECKS: list[str] = []


def mm(v: float) -> float:
    return round(float(v) * MM, 3)


# -- fixture dimensions, all from the task file (mm; x back from the shelf's front face, y left, z above the bench) ---
D = {
    "shelf_w": mm(2 * T.SHELF_HW), "shelf_d": mm(T.SHELF_D), "shelf_h": mm(T.SHELF_H),
    "bar_y": mm(T.SY), "bar_z": mm(T.ZB), "bar_w": mm(T.BW), "bar_t": mm(T.BT),
    "bar_len": mm(T.X_TAIL0 - T.X_FRONT0), "draw": mm(T.DRAW), "stick": mm(T.STICK),
    "boss_back": mm(T.BOSS_BACK), "pivot_back": mm(T.PIVOT_BACK),
    "post_r": mm(T.TAIL_R), "post_d": mm(2 * T.POST_R), "post_h": mm(T.POST_H),
    "notch_w": mm(T.NOTCH_W), "notch_d": mm(T.NOTCH_D), "ceil": mm(T.CEIL),
    "win_w": mm(T.WIN_W), "win_top": mm(T.WIN_TOP), "win_bot": mm(T.WIN_BOT),
    "rail_gap": mm(T.RAIL_GAP), "rail_t": mm(T.RAIL_T), "rail_w": mm(T.RAIL_W), "rail_front": mm(T.RAIL_FRONT),
    "z_rail": mm(T.Z_RAIL), "pin_down": mm(T.PIN_DOWN),
    "swing_in": math.degrees(T.SWING_IN), "swing_out": math.degrees(T.SWING_OUT),
    "lift_lo": math.degrees(T.LIFT_LO), "lift_hi": math.degrees(T.LIFT_HI),
    "lift_k": T.LIFT_K, "lift_pre": math.degrees(T.LIFT_PRE), "notches": list(T.NOTCHES),
    "unfold_stop": math.degrees(T.UNFOLD_STOP), "bar_mass": T.BAR_MASS, "stirrup_mass": T.STIRRUP_MASS,
    "arm_s0": mm(T.ARM_S0), "arm_s1": mm(T.ARM_S1), "arm_w": mm(T.ARM_W), "arm_t": mm(T.ARM_T),
    "loop_s0": mm(T.LOOP_S0), "loop_l": mm(T.LOOP_L), "loop_w": mm(T.LOOP_W), "loop_r": mm(T.LOOP_R),
}
D["post_x_on_bar"] = mm(T.X_POST0 - T.X_FRONT0)             # the post's hole from the bar's front end
D["stop_x_on_bar"] = mm(T.BX + 0.016 + T.DRAW - T.X_FRONT0)  # the wide spacer (draw stop) from the front end
D["support_x0"] = mm(T.BX + 0.100 - T.XF)
D["support_x1"] = mm(T.X_TAIL0 + 0.005 - T.XF)
D["ceil_x0"] = mm(T.BX + 0.055 - T.XF)
D["ceil_x1"] = mm(T.X_POST0 + 0.020 - T.XF)
D["groove_x0"] = mm(T.BX + T.TAIL_R + 0.011 - T.XF)
D["spring_rate"] = T.LIFT_K / T.TAIL_R ** 2 / 1000.0        # N/mm at the post's radius (roller under the rail there)
D["spring_pre"] = T.LIFT_K * T.LIFT_PRE / T.TAIL_R          # N at the roller with the bar level
D["z_pin"] = D["bar_z"] - D["pin_down"]
Z_PLATE0, Z_PLATE1 = 112.0, 120.0      # the yoke's plate (above the pedestal), as in the r2 task file
ROLLER_R, ROLLER_W = 11.0, 7.0         # the spring's roller: printed, 22 x 7 mm, turning on an M8 bolt
SPRING_FREE, SPRING_OD = 38.1, 12.7    # a 1/2 in OD, 1.5 in free length compression spring


# -- small CSG helpers ---------------------------------------------------------------------------------------------
def box(x0, x1, y0, y1, z0, z1):
    return m3.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl_z(cx, cy, z0, z1, r, n=48):
    return m3.Manifold.cylinder(z1 - z0, r, r, n).translate([cx, cy, z0])


def cyl_y(cx, cz, y0, y1, r, n=48):
    return m3.Manifold.cylinder(y1 - y0, r, r, n).rotate([-90, 0, 0]).translate([cx, y0, cz])


def cyl_x(cy, cz, x0, x1, r, n=48):
    return m3.Manifold.cylinder(x1 - x0, r, r, n).rotate([0, 90, 0]).translate([x0, cy, cz])


def union(parts):
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


def capsule(a, b, r, n=24):
    s1 = m3.Manifold.sphere(r, n).translate(list(a))
    s2 = m3.Manifold.sphere(r, n).translate(list(b))
    return m3.Manifold.batch_hull([s1, s2])


def write_stl(man: m3.Manifold, path: pathlib.Path) -> dict:
    mesh = man.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    f = np.asarray(mesh.tri_verts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"exam-stirrup-set fixture part".ljust(80, b" "))
        fh.write(struct.pack("<I", len(f)))
        for tri in f:
            p = v[tri]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            n = n / (np.linalg.norm(n) or 1.0)
            fh.write(struct.pack("<12fH", *n, *p[0], *p[1], *p[2], 0))
    lo, hi = v.min(0), v.max(0)
    return {"file": f"stl/{path.name}", "size_mm": [round(float(x), 1) for x in hi - lo],
            "volume_cm3": round(man.volume() / 1000.0, 1), "petg_g": round(man.volume() / 1000.0 * 1.27 * 0.6, 0)}


# -- printed parts ---------------------------------------------------------------------------------------------------
def pedestal():
    """Stands on the base board under the boss axis; the M8 axle bolt stands up through it; a stop pin (M4) at 20 mm."""
    h = Z_PLATE0 - 1.0 - BASE_T                                    # a 1 mm washer between it and the yoke's plate
    body = box(-25, 25, -25, 25, 0, h) + box(-35, 35, -35, 35, 0, 6)
    holes = [cyl_z(0, 0, -1, h + 1, 4.2)] + [cyl_z(sx * 28, sy * 28, -1, 7, 2.25) for sx in (-1, 1) for sy in (-1, 1)]
    holes.append(cyl_z(20.0, 0, h - 15, h + 1, 2.0))               # the stop pin's hole, toward the shelf's back
    holes.append(cyl_z(0, 0, -1, 10, 7.6, 6))                      # hex pocket for the axle bolt's head
    return body - union(holes), h


def yoke(sign: int):
    """Swings on the axle: a plate with the stop slot, two cheeks carrying the lift pins 10 mm below the bar's centre,
    and the spring tower 75 mm behind the axis. Local frame: origin on the axis at the bench, x back, y left."""
    zp = D["z_pin"]
    tower_x = D["post_r"]
    t_top = D["z_rail"] - 2 * ROLLER_R - 4.0                       # the tower's top, below the roller and plunger
    t_bot = t_top - SPRING_FREE - 6.0
    parts = [box(-25, 25, -20, 20, Z_PLATE0, Z_PLATE1)]
    for d in (-1, 1):                                              # the cheeks, outside the bar's footprint
        parts.append(box(-6, 6, d * 14, d * 20, Z_PLATE1 - 1, zp + 5.5) if d > 0 else
                     box(-6, 6, -20, -14, Z_PLATE1 - 1, zp + 5.5))
    parts.append(box(20, tower_x - 15, -9, 9, t_top, Z_PLATE1))   # the arm
    parts.append(cyl_z(tower_x, 0, t_bot, t_top, 11.0))            # the tower
    parts.append(box(20, tower_x + 6, -9, 9, t_top - 10, t_top))   # the web joining arm and tower, under the roller
    holes = [cyl_z(0, 0, Z_PLATE0 - 1, Z_PLATE1 + 1, 4.3),
             cyl_y(0, zp, -21, -13.9, 2.6), cyl_y(0, zp, 13.9, 21, 2.6),
             cyl_z(tower_x, 0, t_bot + 6, t_top + 1, SPRING_OD / 2 + 0.4),      # the spring and plunger bore
             cyl_z(tower_x, 0, t_bot - 1, t_bot + 7, 4.2)]                    # the M8 preload screw (nut trap below)
    holes.append(cyl_z(tower_x, 0, t_bot - 1, t_bot + 3.5, 7.6, 6))
    # the stop slot: the pedestal's pin at +x (angle 0); the yoke turns between the boss's stops
    side = T.LEFT if sign > 0 else T.RIGHT
    lo, hi = side.limits()
    pin_r, slot_r = 2.0, 20.0
    n = 40
    for k in range(n + 1):
        a = -(lo + (hi - lo) * k / n)                              # the pin's angle in the yoke's frame
        holes.append(cyl_z(slot_r * math.cos(a), slot_r * math.sin(a), Z_PLATE0 - 1, Z_PLATE1 + 1, pin_r + 0.25, 24))
    return union(parts) - union(holes), (t_bot, t_top)


def trunnion():
    """Turns on the lift pins; its lips run in the 4 mm gap between the bar and the T-rail, so it holds the bar both
    ways while everything of it stays below the bar's underside. Frame: on the boss axis, at the bench."""
    zr, zg1 = D["z_rail"], D["bar_z"] - D["bar_t"] / 2
    block = box(-15, 15, -13.5, 13.5, zr - 4.3, zr - 0.3)
    walls = [box(-15, 15, d0, d1, zr - 0.3, zg1 - 0.5) for d0, d1 in ((-13.5, -6.85), (6.85, 13.5))]
    lips = [box(-15, 15, d0, d1, zr + D["rail_t"] + 0.3, zg1 - 0.5) for d0, d1 in ((-6.85, -3.5), (3.5, 6.85))]
    body = union([block, *walls, *lips])
    holes = [cyl_y(0, D["z_pin"], -14, -7.2, 2.45), cyl_y(0, D["z_pin"], 7.2, 14, 2.45)]   # dowel pins pressed in
    return body - union(holes)


def plunger():
    """Slides in the tower's bore on the spring; its fork carries the printed roller on an M8 x 20 bolt."""
    body = cyl_z(0, 0, 0, 8, SPRING_OD / 2 - 0.2)
    fork = [box(-6, 6, d0, d1, 6, 8 + ROLLER_R + 6) for d0, d1 in ((-ROLLER_W / 2 - 3.5, -ROLLER_W / 2 - 0.3),
                                                                  (ROLLER_W / 2 + 0.3, ROLLER_W / 2 + 3.5))]
    hole = cyl_y(0, 8 + 4 + ROLLER_R - 4, -10, 10, 4.1)
    return union([body, *fork]) - hole


def roller():
    """The spring's roller under the T-rail: 22 mm across, 7 mm wide, on an M8 bolt (a 608 bearing fits the same
    fork if the printed one wears)."""
    return cyl_y(0, 0, 0, ROLLER_W, ROLLER_R, 64) - cyl_y(0, 0, -1, ROLLER_W + 1, 4.2, 32)


def notch_plate(sign: int):
    """The guide bracket's teeth: an arc band 5 mm deep round the post's path (radius 75 mm about the boss axis), with
    an 8 mm notch at each lock position, glued and screwed under the ceiling strip. Frame: boss axis at the origin,
    top face (against the ceiling) at z 0, extending down."""
    side = T.LEFT if sign > 0 else T.RIGHT
    lo, hi = side.limits()
    r0, r1 = D["post_r"] - 8.0, D["post_r"] + 8.0
    a0, a1 = math.degrees(lo) - 4.0, math.degrees(hi) + 4.0
    pts_out = [(r1 * math.cos(math.radians(a)), r1 * math.sin(math.radians(a))) for a in np.linspace(a0, a1, 60)]
    pts_in = [(r0 * math.cos(math.radians(a)), r0 * math.sin(math.radians(a))) for a in np.linspace(a1, a0, 60)]
    band = m3.Manifold.extrude(m3.CrossSection([pts_out + pts_in]), D["notch_d"]).translate([0, 0, -D["notch_d"]])
    cuts = []
    rim = [(rr * math.cos(math.radians(a)), rr * math.sin(math.radians(a)))
           for rr, aa in ((r0 + 0.01, np.linspace(a0, a1, 60)), (r0 - 4.0, np.linspace(a1, a0, 60))) for a in aa]
    band = band + m3.Manifold.extrude(m3.CrossSection([rim]), D["notch_d"]).translate([0, 0, -D["notch_d"]])
    for a in side.notch_angles():                                  # the inner rim (inside the post's path) stays whole
        c = box(r0, r1 + 2, -D["notch_w"] / 2, D["notch_w"] / 2, -D["notch_d"] - 1, 1)
        cuts.append(c.rotate([0, 0, math.degrees(a)]))
    for a in np.linspace(a0 + 2, a1 - 2, 4):
        cuts.append(cyl_z(D["post_r"] * math.cos(math.radians(a)), D["post_r"] * math.sin(math.radians(a)),
                          -D["notch_d"] - 1, 1, 1.6, 16))
    return band - union(cuts)


def spacer(wide: bool):
    """Between the bar and its T-rail: 4 mm tall; the wide one, at the draw collar's place, is the draw stop (it meets
    the trunnion's lips with the bar fully drawn). Drilled for an M3 countersunk screw through bar, spacer and rail."""
    w = D["bar_w"] if wide else 6.0
    return box(-5 if not wide else -3, 5 if not wide else 3, -w / 2, w / 2, 0, D["rail_gap"]) - cyl_z(0, 0, -1, 5, 1.65)


def pillar():
    """Four per side hold the ceiling strip with its underside at the groove's ceiling height."""
    h = D["ceil"] - BASE_T
    return box(-6, 6, -6, 6, 0, h) - cyl_z(0, 0, h - 12, h + 1, 1.6, 16) - cyl_z(0, 0, -1, 12, 1.6, 16)


def riser():
    """Two per side under the support rail strip (3/4 in plywood, 24 mm wide): its top at the T-rail's underside."""
    h = D["z_rail"] - 0.3 - BASE_T - BASE_T
    return box(-12, 12, -12, 12, 0, h) - cyl_z(0, 0, -1, h + 1, 2.0, 16)


def clevis():
    """The stirrup's clevis round the bar's front end, in the stirrup's (s, w, h) frame at the roll pin (s along the
    arm, h up off the bar when folded). Cheeks either side of the bar; the bridge takes the steel arm's end; the
    open stop is an M3 screw across the cheeks that lands on the bar's top face at the stop angle."""
    gap = D["bar_w"] / 2 + 0.3
    cheeks = [box(-18, 35, d * gap if d > 0 else -16, 16 if d > 0 else -gap, -10, 12) for d in (-1, 1)]
    bridge = box(12, 35, -16, 16, D["bar_t"] / 2 + 0.15, 12)
    body = union([*cheeks, bridge])
    arm_pocket = box(22, 36, -D["bar_w"] / 2 - 0.2, D["bar_w"] / 2 + 0.2, D["bar_t"] / 2, D["bar_t"] / 2 + D["bar_t"] + 0.3)
    holes = [arm_pocket, cyl_y(0, 0, -17, 17, 2.65),
             cyl_z(26, 0, D["bar_t"] / 2 - 1, 13, 2.1, 16), cyl_z(32, 0, D["bar_t"] / 2 - 1, 13, 2.1, 16)]
    s_st, h_st = STOP_SH
    holes.append(cyl_y(s_st, h_st, -17, 17, 1.65, 16))
    return body - union(holes)


def loop_points(n=120):
    return [(s * MM, w * MM) for s, w in T.loop_path(n)]


def loop_shell_half(top: bool):
    """Half of the loop's 12 mm shell round a 1/4 in steel rod bent to the loop's path; the two halves are glued on the
    rod. The junction block at the pointed end clamps both rod ends and the steel arm's end (two M4 screws) and holds
    the 60 mm steel slug that brings the stirrup to its mass."""
    pts = loop_points(120)
    h0 = D["bar_t"] / 2 + D["loop_r"]
    shell = union([capsule((pts[i][0], pts[i][1], h0), (pts[(i + 1) % len(pts)][0], pts[(i + 1) % len(pts)][1], h0),
                           D["loop_r"], 16) for i in range(len(pts))])
    rod = union([capsule((pts[i][0], pts[i][1], h0), (pts[(i + 1) % len(pts)][0], pts[(i + 1) % len(pts)][1], h0),
                         3.35, 12) for i in range(len(pts))])
    s_j = D["loop_s0"]
    junction = box(s_j - 22, s_j + 6, -12.5, 12.5, D["bar_t"] / 2, D["bar_t"] / 2 + 2 * D["loop_r"])
    arm_slot = box(s_j - 23, s_j - 8, -D["bar_w"] / 2 - 0.2, D["bar_w"] / 2 + 0.2, D["bar_t"] / 2 - 0.1,
                   D["bar_t"] / 2 + D["bar_t"] + 0.2)
    screws = [cyl_z(s_j - 18, 0, 0, 40, 2.1, 16), cyl_z(s_j - 12, 0, 0, 40, 2.1, 16)]
    whole = (shell + junction) - union([rod, arm_slot, *screws])
    cut = box(-10, 400, -60, 60, h0, 60) if not top else box(-10, 400, -60, 60, -10, h0)
    return whole - cut


def loop_former():
    """A bending former: the loop's inside edge (the rod's centre line offset inward by the rod's radius) as a 15 mm
    plate; bend the rod round it in a vise."""
    cs = m3.CrossSection([loop_points(120)], m3.FillRule.NonZero)
    if cs.area() == 0:
        cs = m3.CrossSection([loop_points(120)[::-1]], m3.FillRule.NonZero)
    return m3.Manifold.extrude(cs.offset(-3.175, m3.JoinType.Round), 15.0)


def arm_strip():
    """Glued to each edge of the steel arm (19.05 mm) to bring it to the stirrup's 25 mm."""
    return box(0, D["arm_s1"] - D["arm_s0"], 0, (D["arm_w"] - D["bar_w"]) / 2, 0, D["bar_t"])


# -- geometry checks -----------------------------------------------------------------------------------------------------
def solve_stop():
    """Where the open stop screw goes: its underside lands on the bar's top face 1 mm behind the bar's front end at
    the stop angle (the bar's front end is 10 mm in front of the roll pin)."""
    phi = math.radians(D["unfold_stop"])
    r_screw = 1.5
    b, u = -(D["pivot_back"] - 1.0), D["bar_t"] / 2 + r_screw
    s = b * math.cos(phi) + u * math.sin(phi)
    h = -b * math.sin(phi) + u * math.cos(phi)
    return s, h


STOP_SH = solve_stop()


def check_stop():
    phi_stop = math.radians(D["unfold_stop"])
    s, h = STOP_SH
    first = None
    for k in range(0, 1001):
        phi = phi_stop * k / 1000
        b = s * math.cos(phi) - h * math.sin(phi)
        u = s * math.sin(phi) + h * math.cos(phi)
        if b >= -D["pivot_back"] and u - 1.5 <= D["bar_t"] / 2 + 1e-6:
            first = math.degrees(phi)
            break
    folded_clear = h - 1.5 - D["bar_t"] / 2
    CHECKS.append(f"open stop screw at s {s:.2f} mm, h {h:.2f} mm from the roll pin: it first meets the bar's top face"
                  f" at {first:.1f} deg (stop angle {D['unfold_stop']:.0f} deg); folded it clears the bar by"
                  f" {folded_clear:.2f} mm")
    assert first is not None and abs(first - D["unfold_stop"]) < 0.5 and folded_clear > 0.5


def check_clearances():
    zg1 = D["bar_z"] - D["bar_t"] / 2
    cheek_top = D["z_pin"] + 5.5
    CHECKS.append(f"boss below the bar: cheek tops {cheek_top:.2f} mm, trunnion and lips up to {zg1 - 0.5:.2f} mm, bar"
                  f" underside {zg1:.2f} mm; the folded stirrup lies on the bar's top face at {zg1 + D['bar_t']:.2f} mm")
    assert cheek_top < zg1 and cheek_top < zg1 + D["bar_t"]
    drop = (D["boss_back"]) * math.sin(math.radians(abs(D["lift_lo"])))
    gap = D["z_rail"] - drop - D["win_bot"]
    CHECKS.append(f"T-rail over the window's lower edge with the bar at {D['lift_lo']:.0f} deg: {gap:.2f} mm")
    assert gap > 3.0
    # the tail at full lift compresses the spring further; it must stay off solid height
    extra = D["post_r"] * math.sin(math.radians(D["lift_hi"]))
    pre = D["spring_pre"] / D["spring_rate"]
    CHECKS.append(f"spring: {D['spring_rate']:.2f} N/mm, preload {D['spring_pre']:.1f} N = {pre:.1f} mm; at full lift"
                  f" {pre + extra:.1f} mm of {SPRING_FREE:.1f} mm free length")
    assert pre + extra < 0.6 * SPRING_FREE
    # the yoke's tower sweeps clear of the support rail and the pedestal
    reach = math.hypot(D["post_r"] + 11.0, 11.0)
    CHECKS.append(f"spring tower reaches {reach:.1f} mm from the axis; the support rail starts {D['support_x0'] - D['boss_back']:.1f}"
                  f" mm behind it")
    assert reach * math.cos(math.radians(0)) < D["support_x0"] - D["boss_back"]


# -- drawings ----------------------------------------------------------------------------------------------------------
class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.items = w, h, []
        self.text(10, 18, title, 13, bold=True)

    def line(self, x0, y0, x1, y1, w=1.0, dash=False, color="#222"):
        d = ' stroke-dasharray="4 3"' if dash else ""
        self.items.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{color}" '
                          f'stroke-width="{w}"{d}/>')

    def rect(self, x, y, w, h, fill="none", stroke="#222", sw=1.0, dash=False):
        d = ' stroke-dasharray="4 3"' if dash else ""
        self.items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def circle(self, cx, cy, r, fill="none", stroke="#222"):
        self.items.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}"/>')

    def poly(self, pts, fill="none", stroke="#222", sw=1.0, close=True):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        tag = "polygon" if close else "polyline"
        self.items.append(f'<{tag} points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, s, size=10, anchor="start", bold=False, color="#111"):
        wgt = ' font-weight="bold"' if bold else ""
        s = s.replace("&", "&amp;").replace("<", "&lt;")
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica,Arial,sans-serif" font-size="{size}" '
                          f'text-anchor="{anchor}"{wgt} fill="{color}">{s}</text>')

    def dim(self, x0, y0, x1, y1, label, off=12.0):
        """A dimension line between two points, offset to the left of the direction x0->x1."""
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L * off, dx / L * off
        a0, a1 = (x0 + nx, y0 + ny), (x1 + nx, y1 + ny)
        self.line(x0, y0, a0[0], a0[1], 0.5, color="#666")
        self.line(x1, y1, a1[0], a1[1], 0.5, color="#666")
        self.line(*a0, *a1, 0.7, color="#a33")
        for (px, py), sgn in ((a0, 1), (a1, -1)):
            ux, uy = dx / L * sgn, dy / L * sgn
            self.poly([(px, py), (px + 5 * ux - 2 * uy, py + 5 * uy + 2 * ux), (px + 5 * ux + 2 * uy, py + 5 * uy - 2 * ux)],
                      fill="#a33", stroke="#a33", sw=0.5)
        mx, my = (a0[0] + a1[0]) / 2 + nx * 0.5, (a0[1] + a1[1]) / 2 + ny * 0.5
        self.text(mx, my + 3, label, 9, "middle", color="#a33")

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(self.items)
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                        f'viewBox="0 0 {self.w} {self.h}"><rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>\n')


def drawing_plan(out):
    """Plan view of the whole fixture, 1:4."""
    k = 0.6
    W, Dp = D["shelf_w"], D["shelf_d"]
    ox, oy = 60, 95
    sv = Svg(int(W * k + 160), int(Dp * k + 190), "Fixture plan view (scale 0.6 px/mm; front face at the top, robot side)")

    def P(x, y):           # x back from the face, y left: drawn with y to the right as seen from the robot
        return ox + (W / 2 - y) * k, oy + x * k
    sv.rect(*P(0, W / 2), W * k, Dp * k, sw=1.5)
    sv.text(*P(-8, 0), "front face (1/4 in plywood) with two windows", 9, "middle")
    for sy in (1, -1):
        y = sy * D["bar_y"]
        x0, y0 = P(0, y + D["win_w"] / 2)
        sv.rect(x0, y0 - 3, D["win_w"] * k, 6, fill="#ddd")
        sv.circle(*P(D["boss_back"], y), 4, fill="#333")
        sv.rect(*P(D["ceil_x0"], y + 62), 124 * k, (D["ceil_x1"] - D["ceil_x0"]) * k, dash=True)
        sv.rect(*P(D["support_x0"], y + 12), 24 * k, (D["support_x1"] - D["support_x0"]) * k, fill="#eee")
        xa, ya = P(-D["stick"], y)
        xb, yb = P(D["bar_len"] - D["stick"], y)
        sv.line(xa, ya, xb, yb, 3, color="#555")
        for px, py in ((D["ceil_x0"] + 6, y + 56), (D["ceil_x0"] + 6, y - 56), (D["ceil_x1"] - 6, y + 56),
                       (D["ceil_x1"] - 6, y - 56)):
            cx, cy = P(px, py)
            sv.rect(cx - 12 * k / 2, cy - 12 * k / 2, 12 * k, 12 * k, fill="#999")
    sv.dim(*P(0, D["bar_y"]), *P(0, -D["bar_y"]), f"bars {2 * D['bar_y']:.0f} apart", -26)
    sv.dim(*P(Dp, W / 2), *P(Dp, -W / 2), f"{W:.0f}", 16)
    sv.dim(*P(0, -W / 2), *P(Dp, -W / 2), f"{Dp:.0f}", -14)
    sv.dim(*P(0, D["bar_y"] + 70), *P(D["boss_back"], D["bar_y"] + 70), f"{D['boss_back']:.0f}", 0)
    sv.text(*P(D["boss_back"] + 14, D["bar_y"] + 70), "boss axis", 9, color="#a33")
    sv.dim(*P(D["ceil_x0"], -D["bar_y"] - 70), *P(D["ceil_x1"], -D["bar_y"] - 70),
           f"ceiling strip {D['ceil_x0']:.0f}-{D['ceil_x1']:.0f}", 0)
    sv.dim(*P(D["support_x0"], D["bar_y"] - 90), *P(D["support_x1"], D["bar_y"] - 90),
           f"support rail {D['support_x0']:.0f}-{D['support_x1']:.0f}", 0)
    sv.text(10, sv.h - 30, "Grey squares: ceiling pillars (12 x 12, printed). Dashed: ceiling strip 124 wide (1/4 in plywood), "
                           "notch plate and groove rails under it.", 9)
    sv.text(10, sv.h - 16, f"Windows {D['win_w']:.0f} wide, {D['win_bot']:.1f} to {D['win_top']:.1f} above the bench. "
                           f"Base board 3/4 in plywood, {W:.0f} x {Dp:.0f}.", 9)
    sv.save(out / "drawings" / "1_plan.svg")


def drawing_section(out):
    """Section along one bar (x back from the face, z up), 2:1 around the boss, with every height."""
    k = 2.2
    x_lo, x_hi = -40.0, 190.0
    z_lo, z_hi = -22.0, 190.0
    ox, oy = 70, 40
    sv = Svg(int((x_hi - x_lo) * k + 260), int((z_hi - z_lo) * k + 110),
             "Section along a bar at the boss (scale 2.2 px/mm), bar drawn fully out")

    def P(x, z):
        return ox + (x - x_lo) * k, oy + (z_hi - z) * k
    sv.line(*P(x_lo, 0), *P(x_hi, 0), 1.5)
    sv.rect(*P(x_lo, BASE_T), (x_hi - x_lo) * k, BASE_T * k, fill="#f1e7d0")
    bx = D["boss_back"]
    zb, bt = D["bar_z"], D["bar_t"]
    ped_h = Z_PLATE0 - 1.0 - BASE_T
    sv.rect(*P(bx - 25, BASE_T + ped_h), 50 * k, ped_h * k, fill="#ccc")
    sv.rect(*P(bx - 25, Z_PLATE1), 50 * k, (Z_PLATE1 - Z_PLATE0) * k, fill="#888")
    t_top = D["z_rail"] - 2 * ROLLER_R - 4.0
    sv.rect(*P(bx + 20, Z_PLATE1), (D["post_r"] - 35) * k, (Z_PLATE1 - t_top) * k, fill="#888")
    sv.rect(*P(bx + 20, t_top), (D["post_r"] - 14) * k, 10 * k, fill="#888")
    sv.rect(*P(bx + D["post_r"] - 11, t_top), 22 * k, (SPRING_FREE + 6) * k, fill="#888")
    sv.circle(*P(bx + D["post_r"], D["z_rail"] - ROLLER_R), ROLLER_R * k, fill="#bbb")
    sv.rect(*P(bx - 15, zb - bt / 2 - 0.5), 30 * k, (zb - bt / 2 - 0.5 - (D["z_rail"] - 4.3)) * k, fill="#666")
    sv.circle(*P(bx, D["z_pin"]), 2.5 * k, fill="white")
    sv.rect(*P(x_lo, zb + bt / 2), (x_hi - x_lo) * k, bt * k, fill="#9aa")
    sv.rect(*P(x_lo + 10, D["z_rail"] + D["rail_t"]), (x_hi - x_lo - 10) * k, D["rail_t"] * k, fill="#cde")
    sv.rect(*P(bx + D["post_r"] - D["post_d"] / 2, zb + bt / 2 + D["post_h"]), D["post_d"] * k, D["post_h"] * k, fill="#555")
    sv.rect(*P(D["ceil_x0"], D["ceil"] + PLY_T), (x_hi - D["ceil_x0"]) * k, PLY_T * k, fill="#f1e7d0")
    sv.rect(*P(bx + D["post_r"] - 8, D["ceil"]), 16 * k, D["notch_d"] * k, fill="#e9a")
    sv.rect(*P(-1.5, z_hi), 3 * k, (z_hi - D["win_top"]) * k, fill="#f1e7d0")
    sv.rect(*P(-1.5, D["win_bot"]), 3 * k, (D["win_bot"] - BASE_T) * k, fill="#f1e7d0")
    xs = x_hi + 10
    labels = []
    for z, label in ((BASE_T, f"base top {BASE_T:.0f}"), (Z_PLATE0, f"yoke plate {Z_PLATE0:.0f}-{Z_PLATE1:.0f}"),
                     (D["z_pin"], f"lift pin {D['z_pin']:.1f}"), (D["z_rail"], f"T-rail underside {D['z_rail']:.2f}"),
                     (zb - bt / 2, f"bar underside {zb - bt / 2:.2f}"), (zb, f"bar centre {zb:.1f}"),
                     (zb + bt / 2, f"bar top {zb + bt / 2:.2f}"), (D["ceil"], f"groove ceiling {D['ceil']:.2f}"),
                     (D["win_bot"], f"window bottom {D['win_bot']:.2f}"), (D["win_top"], f"window top {D['win_top']:.1f}")):
        labels.append((z, label))
    labels.sort(key=lambda t: -t[0])
    y_prev = -1e9
    for z, label in labels:                                        # spread the labels, leader lines to their heights
        x0, y0 = P(x_hi, z)
        yl = max(y0, y_prev + 13)
        y_prev = yl
        sv.line(x0, y0, x0 + 14, yl, 0.5, color="#a33")
        sv.text(x0 + 17, yl + 3, label, 9, color="#a33")
    sv.dim(*P(bx, -16), *P(bx + D["post_r"], -16), f"post and roller {D['post_r']:.0f} behind the axis", 0)
    sv.dim(*P(0, -6), *P(bx, -6), f"boss axis {bx:.0f} behind the face", 0)
    sv.text(10, sv.h - 44, "Roller: printed 22 x 7 mm on an M8 bolt in the plunger's fork; spring in the tower's bore on an M8 "
                           "adjusting screw.", 9)
    sv.text(10, sv.h - 30, "The trunnion (dark) turns on two 5 mm dowel pins in the yoke's cheeks; its lips run in the 4 mm gap "
                           "between the bar and the T-rail (light blue).", 9)
    sv.text(10, sv.h - 16, f"Spring: {D['spring_rate']:.2f} N/mm, preloaded to {D['spring_pre']:.1f} N with the bar level "
                           "(M8 adjusting screw under it). Notch plate (pink) under the ceiling strip.", 9)
    sv.save(out / "drawings" / "2_section.svg")


def drawing_notches(out):
    k = 3.0
    sv = Svg(560, 470, "Notch plate (left side; the right side is its mirror), seen from below, scale 3 px/mm")
    ox, oy = 90, 260
    side = T.LEFT
    lo, hi = side.limits()
    r0, r1 = D["post_r"] - 8, D["post_r"] + 8

    def P(x, y):
        return ox + x * k, oy - y * k
    sv.circle(*P(0, 0), 3, fill="#333")
    sv.text(*P(-4, -6), "boss axis", 9)
    for r in (r0, D["post_r"], r1):
        pts = [P(r * math.cos(a), r * math.sin(a)) for a in np.linspace(lo - math.radians(4), hi + math.radians(4), 50)]
        sv.poly(pts, close=False, sw=1.0 if r != D["post_r"] else 0.5)
    for deg, a in zip(D["notches"], side.notch_angles()):
        x, y = r1 + 10, 0
        sv.line(*P(0, 0), *P((r1 + 6) * math.cos(a), (r1 + 6) * math.sin(a)), 0.5, dash=True, color="#a33")
        sv.text(*P((r1 + 14) * math.cos(a), (r1 + 14) * math.sin(a)), f"{deg:+.0f} deg", 9, color="#a33")
    sv.dim(*P(0, -2), *P(D["post_r"], -2), f"R {D['post_r']:.0f} (post path)", 18)
    sv.text(10, 420, f"Arc band {r0:.0f} to {r1:.0f} mm radius, {D['notch_d']:.0f} mm deep; notches {D['notch_w']:.0f} mm wide "
                     f"at {', '.join(f'{a:+.0f}' for a in D['notches'])} deg (outward positive);", 9)
    sv.text(10, 434, f"the band runs from 4 deg past the inward stop ({D['swing_in']:.0f} deg) to 4 deg past the outward stop "
                     f"({D['swing_out']:.0f} deg). The groove (0 deg) continues back between two 10 x 5 mm basswood rails,", 9)
    sv.text(10, 448, f"{D['notch_w']:.0f} mm apart, from {D['groove_x0']:.0f} to {D['ceil_x1']:.0f} mm behind the front face.", 9)
    sv.save(out / "drawings" / "3_notch_plate.svg")


def drawing_bar(out):
    k = 1.4
    L = D["bar_len"]
    sv = Svg(int(L * k + 120), 230, "Stirrup bar: 3/4 x 1/4 in steel flat, drilling (two, identical), scale 1.4 px/mm")
    ox, oy = 50, 90

    def P(x, y=0.0):
        return ox + x * k, oy + y * k
    sv.rect(*P(0, -D["bar_w"] / 2), L * k, D["bar_w"] * k, fill="#dde")
    marks = [(D["pivot_back"], "roll pin 5.5 hole"), (D["rail_front"] + 10, "M3 csk (narrow spacer)"),
             (300 - D["stick"] + 0.0, "M3 csk"), (D["stop_x_on_bar"] + 3, "M3 csk (wide spacer = draw stop)"),
             (L - 30, "M3 csk"), (D["post_x_on_bar"], "M6 tapped, post stud from above")]
    for i, (x, label) in enumerate(marks):
        sv.circle(*P(x), 2.5, fill="white")
        sv.line(*P(x, D["bar_w"] / 2), *P(x, D["bar_w"] / 2 + 14 + 12 * (i % 3)), 0.5, color="#a33")
        sv.text(*P(x, D["bar_w"] / 2 + 22 + 12 * (i % 3)), f"{x:.0f}: {label}", 8, "middle", color="#a33")
    sv.dim(*P(0, -D["bar_w"] / 2), *P(L, -D["bar_w"] / 2), f"{L:.0f}", 14)
    sv.text(10, 210, f"T-rail: 1/2 x 1/8 in aluminium flat, {L - D['rail_front']:.0f} mm, from {D['rail_front']:.0f} mm to the tail, "
                     f"held 4 mm under the bar by the spacers. Post: M6 stud, {D['post_h']:.0f} mm above the top face, flush below.", 9)
    sv.save(out / "drawings" / "4_bar.svg")


def drawing_stirrup(out):
    k = 2.0
    sv = Svg(1000, 320, "Stirrup (two): steel arm, printed clevis and loop shell on a bent 1/4 in rod, scale 2 px/mm")
    ox, oy = 70, 130

    def P(s, w):
        return ox + s * k, oy - w * k
    sv.rect(*P(-18, 16), 53 * k, 32 * k, fill="#ccc")
    sv.rect(*P(D["arm_s0"], D["arm_w"] / 2), (D["arm_s1"] - D["arm_s0"]) * k, D["arm_w"] * k, fill="#aab")
    pts = [P(s, w) for s, w in loop_points(80)]
    sv.poly(pts, sw=D["loop_r"] * 2 * k * 0.5, stroke="#888")
    sv.circle(*P(0, 0), 2.6 * k, fill="white")
    s_st, h_st = STOP_SH
    sv.dim(*P(-18, -20), *P(D["loop_s0"] + D["loop_l"], -20), f"{D['loop_s0'] + D['loop_l'] + 18:.0f} overall", 0)
    sv.dim(*P(0, 26), *P(D["arm_s0"], 26), f"{D['arm_s0']:.0f}", 0)
    sv.text(10, 250, f"Roll pin: M5 shoulder bolt through the clevis cheeks and the bar, {D['pivot_back']:.0f} mm behind the bar's "
                     f"front end.", 9)
    sv.text(10, 236, f"Open stop: M3 x 35 screw across the cheeks at s {s_st:.1f}, h {h_st:.1f} mm from the roll pin", 9)
    sv.text(10, 264, f"(it meets the bar's top face at {D['unfold_stop']:.0f} deg). Arm: {D['arm_s1'] - D['arm_s0']:.0f} mm offcut of "
                     f"the bar steel with printed 3 mm edge strips ({D['arm_w']:.0f} wide). Loop: {D['loop_w']:.0f} x "
                     f"{D['loop_l']:.0f} mm teardrop, 12 mm shell.", 9)
    sv.text(10, 278, f"Mass {D['stirrup_mass']:.2f} kg: arm 122 g, rod 63 g, printed parts about 47 g, slug (60 mm of bar offcut) 57 g, "
                     "screws; weigh and add washers in the junction block to reach it.", 9)
    sv.save(out / "drawings" / "5_stirrup.svg")


# -- bill of materials ---------------------------------------------------------------------------------------------------
BOM = [
    # item, specification, where (example), unit price USD, qty, price source
    ("Steel flat bar", "3/4 x 1/4 in, 36 in, plain steel (the two bars; offcuts make the arms and slugs)",
     "Home Depot, Everbilt 3/4 in x 36 in x 1/4 in plain flat bar", 15.27, 2, "listed price, 2026-11-18"),
    ("Aluminium flat bar", "1/2 x 1/8 in, 48 in (the two T-rails, 500 mm each)",
     "Home Depot or hardware store, plain aluminium flat bar", 12.00, 1, "estimate"),
    ("Steel round rod", "1/4 in, 36 in (the two loop cores, 252 mm each)", "hardware store, plain steel rod", 6.00, 1,
     "estimate"),
    ("Plywood project panel", "23/32 in sanded, 2 x 4 ft (base board 670 x 580, support rails)",
     "Home Depot, 23/32 in x 2 ft x 4 ft sanded plywood project panel", 25.04, 1, "listed price, 2026-11-18"),
    ("Plywood project panel", "1/4 in x 2 x 4 ft (front face, two sides, two ceiling strips)",
     "Home Depot, PureBond birch 1/4 in x 2 ft x 4 ft project panel", 20.89, 1, "listed price, 2026-11-18"),
    ("Plywood project panel", "1/4 in x 2 x 4 ft (the shelf's top, 670 x 580)",
     "Home Depot, PureBond birch 1/4 in x 2 ft x 4 ft project panel", 20.89, 1, "listed price, 2026-11-18"),
    ("Lumber", "2 x 4 in, 8 ft (the support stand under the overhang: two legs and a cross rail)", "any lumber yard",
     5.00, 1, "estimate"),
    ("Basswood strip", "3/16 x 3/8 in, 24 in (four groove rails)", "craft or hobby store", 1.50, 4, "estimate"),
    ("PETG filament", "1 kg spool (about 0.9 kg at 40 % infill; pedestals and risers can be 15 %)", "any", 20.00, 1, "estimate"),
    ("Compression spring", f"1/2 in OD, 1.5 in free length, {D['spring_rate'] * 5.71:.0f} lbf/in "
     f"({D['spring_rate']:.1f} N/mm) +- 15 %", "McMaster-Carr or a spring assortment", 4.00, 2, "estimate"),
    ("Fasteners", "2 M8 x 130 axle bolts with nuts and washers, 2 M8 x 20 roller bolts, 2 M8 x 25 preload screws, "
     "2 M5 shoulder bolts (roll pins), 2 M3 x 35 (open stops), 8 M3 countersunk x 16, 2 M6 x 35 set screws (posts), "
     "4 5 x 20 mm dowel pins, M4 screws, wood screws", "hardware store", 15.00, 1, "estimate"),
    ("Glue and threadlocker", "wood glue, CA glue, medium threadlocker", "hardware store", 6.00, 1, "estimate"),
]


def write_bom(out):
    total = 0.0
    with open(out / "BOM.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "specification", "example source", "unit price USD", "qty", "line USD", "price"])
        for item, spec, src, price, qty, how in BOM:
            line = price * qty
            total += line
            w.writerow([item, spec, src, f"{price:.2f}", qty, f"{line:.2f}", how])
        listed = sum(p * q for _, _, _, p, q, how in BOM if how.startswith("listed"))
        w.writerow(["TOTAL", "", "", "", "", f"{total:.2f}", f"{listed:.2f} of it from listed prices, the rest estimates"])
    return total


CUTS = [
    ("base board", "3/4 in plywood", f"{D['shelf_w']:.0f} x {D['shelf_d']:.0f}", 1),
    ("support rail strip", "3/4 in plywood", f"24 x {D['support_x1'] - D['support_x0']:.0f}", 2),
    ("front face", "1/4 in plywood", f"{D['shelf_w']:.0f} x {D['shelf_h']:.0f}, two windows {D['win_w']:.0f} wide from "
     f"{D['win_bot']:.1f} to {D['win_top']:.1f} above the bench, centred {D['bar_y']:.0f} either side", 1),
    ("side panels", "1/4 in plywood", f"{D['shelf_d']:.0f} x {D['shelf_h'] - BASE_T:.0f}", 2),
    ("ceiling strips", "1/4 in plywood", f"124 x {D['ceil_x1'] - D['ceil_x0']:.0f}", 2),
    ("top", "1/4 in plywood", f"{D['shelf_w']:.0f} x {D['shelf_d']:.0f}, on the side panels", 1),
    ("stand legs", "2 x 4 in lumber", "the bench's height less 89 mm (the rail stands on edge on them)", 2),
    ("stand rail", "2 x 4 in lumber", "600 long, on edge under the base board's rear 320 mm, screwed to the legs", 1),
    ("groove rails", "3/16 x 3/8 in basswood", f"{D['ceil_x1'] - D['groove_x0']:.0f} long", 4),
    ("bars", "3/4 x 1/4 in steel flat", f"{D['bar_len']:.0f} long, drilled as drawing 4", 2),
    ("arms", "3/4 x 1/4 in steel flat (bar offcut)", f"{D['arm_s1'] - D['arm_s0']:.0f} long, two 4.2 holes", 2),
    ("slugs", "3/4 x 1/4 in steel flat (bar offcut)", "60 long", 2),
    ("T-rails", "1/2 x 1/8 in aluminium flat", f"{D['bar_len'] - D['rail_front']:.0f} long, holes matching the bar", 2),
    ("loop cores", "1/4 in steel rod", "252 long, bent on the printed former", 2),
]


def main():
    out = HERE
    check_stop()
    check_clearances()
    parts = {}
    ped, ped_h = pedestal()
    parts["pedestal"] = (ped, 2)
    for sign, nm in ((1, "left"), (-1, "right")):
        y, (tb, tt) = yoke(sign)
        parts[f"yoke_{nm}"] = (y, 1)
        parts[f"notch_plate_{nm}"] = (notch_plate(sign), 1)
    parts["trunnion"] = (trunnion(), 2)
    parts["plunger"] = (plunger(), 2)
    parts["roller"] = (roller(), 2)
    parts["spacer_narrow"] = (spacer(False), 6)
    parts["spacer_wide_draw_stop"] = (spacer(True), 2)
    parts["ceiling_pillar"] = (pillar(), 8)
    parts["support_riser"] = (riser(), 4)
    parts["clevis"] = (clevis(), 2)
    parts["loop_shell_top"] = (loop_shell_half(True), 2)
    parts["loop_shell_bottom"] = (loop_shell_half(False), 2)
    parts["loop_former"] = (loop_former(), 1)
    parts["arm_edge_strip"] = (arm_strip(), 4)
    table = []
    grams = 0.0
    for name, (man, qty) in parts.items():
        info = write_stl(man, out / "stl" / f"{name}.stl")
        info.update(name=name, qty=qty)
        grams += info["petg_g"] * qty
        table.append(info)
        assert max(info["size_mm"]) <= 256.0, name
    with open(out / "parts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["part", "qty", "file", "size mm (x y z)", "volume cm3", "PETG g each (40 % infill est.)"])
        for t in table:
            w.writerow([t["name"], t["qty"], t["file"], " x ".join(str(v) for v in t["size_mm"]), t["volume_cm3"], t["petg_g"]])
    total = write_bom(out)
    with open(out / "cut_list.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["piece", "stock", "size mm", "qty"])
        w.writerows(CUTS)
    drawing_plan(out)
    drawing_section(out)
    drawing_notches(out)
    drawing_bar(out)
    drawing_stirrup(out)
    forces = [
        {"name": "guide spring at the post", "value": f"{D['spring_pre']:.1f} N preload, {D['spring_rate']:.2f} N/mm",
         "model": f"lift spring {D['lift_k']} N m/rad preloaded {D['lift_pre']:.0f} deg (LIFT_K, LIFT_PRE)",
         "set": "turn the M8 screw under the spring until the check below reads right",
         "check": "bar drawn and straight, stirrup off: spring scale under the post, pulled straight down: the post starts "
                  "to leave its notch at about 26 N (15 N of it the bar's front-heavy weight, 11 N the spring) and is "
                  "clear (5 mm down) at about 37 N"},
        {"name": "lift to unlock", "value": "7 N to start, 9 N at release",
         "model": "follows from the spring and the masses",
         "set": "by the spring above", "check": "stirrup unfolded, bar locked straight: spring scale on the arm 115 mm above "
                                                  "the roll pin, pulled straight up"},
        {"name": "draw", "value": "about 5 to 8 N (estimate: the post pressed on the groove's ceiling, friction 0.3)",
         "model": "contact friction 0.3", "set": "PTFE tape on the groove's ceiling if it reads high",
         "check": "spring scale on the clevis, pulled along the bar, steady"},
        {"name": "stirrup mass", "value": f"{D['stirrup_mass']:.2f} kg +- 0.02", "model": "STIRRUP_MASS",
         "set": "washers in the junction block", "check": "kitchen scale"},
        {"name": "bar assembly mass", "value": "0.60 kg (steel bar 0.54 kg, T-rail and spacers 0.06 kg)",
         "model": "BAR_MASS", "set": "-", "check": "kitchen scale"},
        {"name": "swing stops", "value": f"{D['swing_in']:.0f} deg inward, {D['swing_out']:.0f} deg outward",
         "model": "SWING_IN, SWING_OUT", "set": "printed slot in the yoke", "check": "protractor on the base board"},
        {"name": "open stop", "value": f"{D['unfold_stop']:.0f} deg", "model": "UNFOLD_STOP", "set": "M3 stop screw position",
         "check": "protractor against the bar"},
    ]
    spec = {"task": "exam-stirrup-set", "revision": "r2 fixture", "units": "mm",
            "dimensions": {k: v for k, v in D.items() if not isinstance(v, list)},
            "pieces": [{"name": t["name"], "qty": t["qty"], "how": "printed (PETG)", "file": t["file"]} for t in table]
            + [{"name": c[0], "qty": c[3], "how": f"cut from {c[1]}", "size": c[2]} for c in CUTS],
            "bom": [{"item": b[0], "spec": b[1], "source": b[2], "price_usd": b[3], "qty": b[4], "price": b[5]} for b in BOM],
            "bom_total_usd": round(total, 2), "petg_grams": round(grams), "forces": forces, "checks": CHECKS}
    (out / "fixture_data.json").write_text(json.dumps(spec, indent=2) + "\n")
    (out / "checks.txt").write_text("\n".join(CHECKS) + "\n")
    print(f"BOM total ${total:.2f}; PETG about {grams:.0f} g; {len(table)} printed part files")
    for c in CHECKS:
        print(" -", c)


if __name__ == "__main__":
    main()
