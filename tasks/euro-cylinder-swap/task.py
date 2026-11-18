"""On the station bench: a 250 x 300 mm section of a 44 mm door, hung on two butt hinges from a hinge post and standing
open, as a door does while its lock is worked on. It swings both ways (0.03 N m hinge friction). In its free edge is a
euro-profile mortice deadlock (Union L2149, 2.5 in case) holding a brass double euro cylinder (ERA 5-pin 30/30, 60 mm)
with its key in. Both keys wear a push-on rubber key cap (an identifier cap, 6 mm thick over the bow). The cylinder's
fixing screw (M5 x 60) has already been taken out and lies in the parts tray (150 mm magnetic tray) on the right of the
bench, next to the new cylinder (the same ERA 30/30 in satin nickel, key in, as supplied, standing on its end).

The job: change the cylinder. Threads are excluded: the screw stays in the tray and the job ends with the new
cylinder centred in the lock with its cam turned out of line (it cannot be drawn out).

How the lock holds the cylinder, all by contact: the cylinder's cam (sintered steel, 15 mm tip radius, 9.7 mm thick)
turns with the plug between the lock case's two side plates. At the key-out angle the cam rests 30 degrees off the
body's centre line and overhangs the plates' profile-shaped apertures by about 5 mm, so the cylinder cannot slide.
Only with the key turned about 30 degrees clockwise (within about 7 degrees) is the cam in line with the web and the
cylinder free. The cam cannot be seen: the angle is found by feel, turning the key and giving a light pull.

The key is part of the plug's hinge: with the right key in, plug and key turn as one, and a turned key cannot be
withdrawn, so pulling the turned key pulls the cylinder. The pin tumblers are not modelled and the keys stay in
their cylinders (the task never takes a key out).

Catalogue sizes: cylinder 33 x 17 x 60 mm, 0.21 kg, web 10 mm, plug axis to web bottom 24.3 mm, to the screw hole
19.5 mm; cam 15 mm radius, 6.6 mm wide, 9.7 mm thick, rest angle 30 degrees; lock case 65 x 108 x 15 mm, backset
44.5 mm, forend 25.5 x 152.5 mm, 0.58 kg; door 44 mm; tray 150 mm x 20 mm. Estimates (measure the bought parts): the
plug's 13 mm, the cam cut-out's 10.3 mm, 0.5 mm a side between cylinder and case apertures, 1 mm a side in the door's
own hole, the 0.8 mm chamfer on the cylinder's ends, the key (23.8 mm bow in a 26 x 26 x 6 mm rubber key cap, 12 g),
the cylinder's height in the case, the plates' 1.5 mm, the door section's 1.3 kg, the hinges' 0.03 N m and the plug's
0.006 N m of friction. Not modelled: the pin tumblers, the second keyway, and the bolt and levers inside the case (the
bolt is withdrawn; the cam does not reach them between the key-out and the in-line positions).

World frame: x forward, y left, heights above the bench top.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import meshkit as mk  # noqa: E402
import numpy as np  # noqa: E402
from partkit import Hinge, PartState, finger_points, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
INSTRUCTION = ("Change the door's lock cylinder for the new one on the bench, leaving the new one fitted in the lock "
               "and the old one in the tray, the door where it was; the fixing screw stays in the tray.")

BRASS = (0.80, 0.62, 0.22)
SATIN = (0.76, 0.76, 0.74)
STEEL = (0.60, 0.61, 0.64)
DARK = (0.16, 0.16, 0.17)
OAK = (0.72, 0.56, 0.36)
PINE = (0.86, 0.76, 0.56)
PLY = (0.80, 0.70, 0.52)
INOX = (0.82, 0.83, 0.85)
OFFSET = 0.0002            # contact offset of the lock parts: below the tightest fit (0.3 mm)

# -- the cylinder (frame: origin on the plug axis at mid-length, +x into the door, z up, web toward -z; m) -------------
CYL_L = 0.060
R_BARREL = 0.0085
WEB_HALF = 0.005
WEB_BOTTOM = 0.0243                 # plug axis to the web's bottom
WEB_C = WEB_BOTTOM - WEB_HALF       # plug axis to the centre of the web's round bottom
SCREW_HOLE = 0.0195                 # plug axis to the fixing-screw hole
R_PLUG = 0.0065                     # estimate
CUT_HALF = 0.00515                  # half the cam cut-out (estimate)
BRIDGE_TOP = 0.0160                 # the bridge under the cam starts this far below the axis (estimate)
CAM_R, CAM_W, CAM_T = 0.015, 0.0066, 0.0097
CAM_HUB = 0.0118                    # the hub as a square inscribed in its 16.7 mm circle
CAM_FLAT = math.sqrt(CAM_R ** 2 - (CAM_W / 2) ** 2)   # the lobe's flat end: its two corners are on the 15 mm tip radius
CAM_REST = math.radians(30.0)       # the cam's angle off the centre line with the key out (the new cylinder's, as made)
# The old cylinder's cam angle is a trial setting (review rule 14): where a fitted cylinder's cam rests depends on the
# make and on which way it was last turned, so the key has to be turned, either way, until the cam is in line.
# ECS_CAM_DEG in degrees, negative for the other side; the default 30 is the scene as first submitted.
OLD_CAM_REST = math.radians(float(os.environ.get("ECS_CAM_DEG", "30")))
TURN = CAM_REST                     # key turn (clockwise seen from the key, + about x) that brings the cam in line
CHAMFER = 0.0008                    # estimate
RING_IN = 0.00675                   # the bore, as the inner flats of the outer half's 16 staves
CYL_MASS, PLUG_MASS, PIN_MASS = 0.169, 0.039, 0.001     # 0.21 kg with 1 g more on the body (PIN_MASS: kept for the sum)
# The plug's turning friction (the pin stacks and the cam's clutch spring): about 0.006 N m (estimate)
PLUG_FRICTION = 0.006
PAD_PATCH = 0.006                   # a finger pad's contact on the barrel, as a patch radius (m): it resists twisting
SLIDE_FRICTION = 0.2                # plated brass on the case's steel and in the door's hole (estimate)

# key and keyway (key frame: origin at the shoulder in the blade's mid-plane, +x along the blade)
BLADE_L, BLADE_T = 0.024, 0.0023
BLADE_Z0, BLADE_Z1 = -0.0046, 0.0030
NECK_L = 0.0086
NECK_Z0, NECK_Z1 = -0.0075, 0.0045
BOW, BOW_T = 0.026, 0.006           # the bow in its push-on rubber key cap (the metal bow is 23.8 x 2.2 mm)
BOW_Z = -0.001                      # the bow's centre height
KEY_MASS = 0.012                    # with the cap
KEYWAY_HALF = 0.00125               # 0.1 mm a side round the blade: the key turns the plug with under 1 degree of play
KEYWAY_Z0, KEYWAY_Z1 = -0.0047, 0.0031     # 0.1 mm above and below the blade: the key cannot ride up over the pin
KEYWAY_L = 0.0245
PIN_X = 0.008                       # the modelled pin, behind the plug's face
KEY_GAP = 0.0002                    # the key's shoulder stands this far off the plug's face when home
CUT = 0.002                         # depth of the key's cut over it; both flanks at 45 degrees
PIN_LX, PIN_LY = 0.0026, 0.0020
PIN_OUT = 0.0066                    # the pin's outer end at rest, from the plug axis
PIN_TRAVEL = 0.0021
PIN_RATE = float(os.environ.get("ECS_PIN_RATE", 450.0))      # N/m: about 0.9 N at full lift
POCKET_Z = 0.0095                   # floor of the pin chamber, below the axis
POCKET_X = 0.0014                   # half its length along the cylinder


def _qx(a):
    return (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)


def _qy(a):
    return (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)


def _qz(a):
    return (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))


def _f(v):
    return tuple(float(x) for x in v)


def bx(x0, x1, y0, y1, z0, z1, color, quat=None):
    """A box by its extents in a part frame, optionally turned about its own centre."""
    c = (float((x0 + x1) / 2), float((y0 + y1) / 2), float((z0 + z1) / 2))
    s = (float(abs(x1 - x0)), float(abs(y1 - y0)), float(abs(z1 - z0)))
    return (c, s, color) if quat is None else (c, s, color, _f(quat))


def radial_box(x0, x1, theta, r0, r1, t0, t1, color, centre=(0.0, 0.0)):
    """A box lying along x from x0 to x1, spanning radii r0 to r1 from the axis (at ``centre`` in y, z) in direction
    ``theta`` (rad, from +y toward +z) and t0 to t1 across it (positive toward larger angles)."""
    rc, tc = (r0 + r1) / 2, (t0 + t1) / 2
    c, s = math.cos(theta), math.sin(theta)
    return ((float((x0 + x1) / 2), float(centre[0] + rc * c - tc * s), float(centre[1] + rc * s + tc * c)),
            (float(abs(x1 - x0)), float(abs(t1 - t0)), float(abs(r1 - r0))), color, _f(_qx(theta - math.pi / 2)))


def ring_boxes(x0, x1, apothem, thick, a0, a1, n, color, centre=(0.0, 0.0)):
    """``n`` boxes whose inner faces are the flats of a polygon of the given apothem, covering angles a0 to a1. They
    overlap each other outside the polygon; the two end boxes stop exactly at the polygon's end corners."""
    step = (a1 - a0) / n
    wide, exact = (apothem + thick) * math.tan(step / 2), apothem * math.tan(step / 2)
    return [radial_box(x0, x1, a0 + (k + 0.5) * step, apothem, apothem + thick, -(exact if k == 0 else wide),
                       exact if k == n - 1 else wide, color, centre) for k in range(n)]


def loft(rings, axis="X"):
    """A closed convex mesh through polygon ``rings`` [(position along the axis, [(u, v), ...]), ...], every ring
    with the same number of points, counter-clockwise seen from the axis's positive end."""
    n = len(rings[0][1])
    pts, faces = [], []
    for s, poly in rings:
        for u, v in poly:
            pts.append({"X": (s, u, v), "Y": (v, s, u), "Z": (u, v, s)}[axis])
    for k in range(len(rings) - 1):
        a, b = k * n, (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces += [(a + i, a + j, b + j), (a + i, b + j, b + i)]
    last = (len(rings) - 1) * n
    for i in range(1, n - 1):
        faces += [(0, i + 1, i), (last, last + i, last + i + 1)]
    mesh = mk.Mesh(pts, faces)
    return mesh if mesh.volume() > 0 else mesh.flipped()


def circle(r, n=16, phase=0.0):
    return [(r * math.cos(phase + 2 * math.pi * k / n), r * math.sin(phase + 2 * math.pi * k / n)) for k in range(n)]


def stadium(half, top, centre, shrink=0.0, n=8):
    """The web's section: a strip ``half`` wide from z = -top down to a round bottom centred at z = -centre."""
    h = half - shrink
    pts = [(h, -top - shrink), (-h, -top - shrink)]
    pts += [(h * math.cos(a), -centre + h * math.sin(a)) for a in np.linspace(math.pi, 2 * math.pi, n + 1)]
    return pts


def _cam_dir(angle, rest=CAM_REST):
    """Unit vector (y, z) of the cam with the plug turned ``angle`` from the key-out position, for a cylinder whose
    cam rests ``rest`` off the centre line with the key out."""
    a = angle - rest
    return (math.sin(a), -math.cos(a))


def cylinder_parts(name, pos, color, quat_xyzw=None, cam_rest=CAM_REST):
    """One ERA 30/30 as a loose articulation (body, plug with keyway and cam on a hinge, one bottom pin on a spring
    slider in the plug), and its meshes: the chamfered inner barrel and the web's round bottom."""
    L, cut = CYL_L / 2, CUT_HALF
    face = -L                                              # the plug's face on the key side
    px0, px1 = face + PIN_X - POCKET_X, face + PIN_X + POCKET_X
    body = []
    # outer half: the bore as 15 staves (the one over the pin chamber left out), the upper web round the chamber
    stave_out = R_BARREL * math.cos(math.pi / 16)
    wide, exact = stave_out * math.tan(math.pi / 16), RING_IN * math.tan(math.pi / 16)
    for k in range(16):
        if k == 8:
            continue
        body.append(radial_box(-L, -cut, math.pi / 2 + k * math.pi / 8, RING_IN, stave_out,
                               -(exact if k == 9 else wide), exact if k == 7 else wide, color))
    body += [bx(-L, px0, -WEB_HALF, WEB_HALF, -POCKET_Z, -RING_IN, color),
             bx(px1, -cut, -WEB_HALF, WEB_HALF, -POCKET_Z, -RING_IN, color),
             bx(px0, px1, 0.00134, WEB_HALF, -POCKET_Z, -RING_IN, color),
             bx(px0, px1, -WEB_HALF, -0.00134, -POCKET_Z, -RING_IN, color)]
    # plug (hinge, frame of the body at the key-out angle): keyway walls, floor and roof; cam hub and lobe
    kw1 = face + KEYWAY_L
    plug = [bx(face, kw1, KEYWAY_HALF, 0.0041, KEYWAY_Z0, KEYWAY_Z1, STEEL),      # every corner inside the 13 mm plug
            bx(face, kw1, -0.0041, -KEYWAY_HALF, KEYWAY_Z0, KEYWAY_Z1, STEEL),
            bx(face, kw1, -0.0045, 0.0045, KEYWAY_Z1, 0.0046, STEEL),
            bx(face, kw1, -0.0015, 0.0015, KEYWAY_Z1, 0.0063, STEEL),      # the keyway's roof, solid to the plug's surface
            bx(face, kw1, -0.0019, 0.0019, -0.0062, KEYWAY_Z0, STEEL),
            bx(kw1, -CAM_T / 2, -0.0045, 0.0045, -0.0045, 0.0045, STEEL),
            bx(-CAM_T / 2, CAM_T / 2, -CAM_HUB / 2, CAM_HUB / 2, -CAM_HUB / 2, CAM_HUB / 2, DARK),
            radial_box(-CAM_T / 2, CAM_T / 2, math.atan2(_cam_dir(0.0, cam_rest)[1], _cam_dir(0.0, cam_rest)[0]), 0.004, CAM_FLAT,
                       -CAM_W / 2, CAM_W / 2, DARK)]
    # the key, home in the plug and turning with it: with the right key in, plug and key turn as one, and a turned
    # key cannot be withdrawn, so the pin tumblers are not modelled (fidelity). Its neck and its bow in the push-on
    # rubber cap stand outside the plug's face; the blade inside the keyway is part of the plug here.
    kx = face - KEY_GAP
    plug += [bx(kx - NECK_L, kx, -BLADE_T / 2, BLADE_T / 2, NECK_Z0, NECK_Z1, color),
             bx(kx - NECK_L - BOW, kx - NECK_L, -BOW_T / 2, BOW_T / 2, BOW_Z - BOW / 2, BOW_Z + BOW / 2, DARK)]
    # the plug's turning friction as Coulomb friction on its hinge (a sprung shoe rubbing the 16-stave bore snagged on
    # a stave edge in PhysX and stopped the plug at 19 degrees: Isaac runs 8 and 9)
    hinge = Hinge("plug", pivot=(0.0, 0.0, 0.0), boxes=tuple(plug), lower=-math.inf, upper=math.inf, stiffness=0.0,
                  damping=0.5, effort=PLUG_FRICTION, mass=PLUG_MASS + KEY_MASS, axis="X")
    meta = {"contact_offset": OFFSET, "self_collide": True, "torsional_patch": PAD_PATCH}
    if quat_xyzw is not None:
        meta["quat_xyzw"] = _f(quat_xyzw)
    cyl = part(name, pos=_f(pos), mass=CYL_MASS + 0.001, boxes=body, hinges=(hinge,), color=color,
               friction=SLIDE_FRICTION, meta=meta)
    nose = L - CHAMFER
    meshes = [
        loft([(cut, circle(R_BARREL)), (nose, circle(R_BARREL)), (L, circle(R_BARREL - CHAMFER))]),      # inner barrel
        loft([(cut, stadium(WEB_HALF, 0.002, WEB_C)), (nose, stadium(WEB_HALF, 0.002, WEB_C)),
              (L, stadium(WEB_HALF, 0.002, WEB_C, CHAMFER))]),                                              # inner web
        loft([(-L, stadium(WEB_HALF, POCKET_Z, WEB_C)), (-cut, stadium(WEB_HALF, POCKET_Z, WEB_C))]),      # outer web
        loft([(-cut, stadium(WEB_HALF, BRIDGE_TOP, WEB_C)), (cut, stadium(WEB_HALF, BRIDGE_TOP, WEB_C))]),  # bridge
    ]
    mk.attach(cyl, [mk.Shape(m, color=color, collider="hull", metallic=0.6) for m in meshes])
    return cyl


# -- layout (m; heights above the bench) ------------------------------------------------------------------------------
XF = 0.300                          # the door's near face
DOOR_T, DOOR_W, DOOR_H = 0.044, 0.250, 0.300
YK, ZK = 0.030, 0.120               # the cylinder's axis
BACKSET = 0.0445
EDGE = YK + BACKSET                 # the door's free edge (the forend's face)
HX, HY = XF + DOOR_T, EDGE - DOOR_W # the hinge line: the door's far face at its hinge edge (the post stands behind)
BASE_T = 0.018
DOOR_Z0, DOOR_Z1 = BASE_T + 0.004, BASE_T + 0.004 + DOOR_H
CASE_T = 0.015
PLATE = 0.0015                      # estimate
FIT = 0.0005                        # cylinder to case aperture, a side (estimate)
WOOD_FIT = 0.001                    # cylinder to the door's own hole, a side (estimate: a 19 mm bore and a 12 mm slot)
CASE_X0 = XF + (DOOR_T - CASE_T) / 2
CASE_X1 = CASE_X0 + CASE_T
CASE_Z0 = ZK - 0.036                # the case's bottom edge (estimate: the cylinder's height in the case)
CASE_H, CASE_D = 0.108, 0.065
FOREND_W, FOREND_L, FOREND_T = 0.0255, 0.1525, 0.003
SEAT = (XF + DOOR_T / 2, YK, ZK)    # the cylinder's origin when centred in the lock
DOOR_MASS = 1.3 + 0.58              # the section and the lock
HINGE_FRICTION = 0.03               # N m: the leaf's butt hinges, as Coulomb friction (estimate; a door left
                                    # free must come to rest, not drift)
SWING_LO, SWING_HI = -math.radians(85.0), math.radians(80.0)
WIN_Y, WIN_UP, WIN_DN = 0.020, 0.020, 0.034        # the mortise round the cylinder: the open inside of the case


def wall(x0, x1, fit, n_arc, n_bot, thick, color):
    """A wall across the door's window from x0 to x1 with the euro-profile hole, in the frame of the cylinder's
    axis: an arc of boxes round the barrel, two beside the web and an arc under it."""
    r, h = R_BARREL + fit, WEB_HALF + fit
    zj = math.sqrt(r * r - h * h)
    a0 = math.atan2(-zj, h)
    out = ring_boxes(x0, x1, r, thick, a0, math.pi - a0, n_arc, color)
    out += [bx(x0, x1, h, h + thick, -WEB_C, -zj, color), bx(x0, x1, -h - thick, -h, -WEB_C, -zj, color)]
    out += ring_boxes(x0, x1, h, thick, math.pi, 2 * math.pi, n_bot, color, centre=(0.0, -WEB_C))
    return out


def _shift(boxes, d):
    return [((b[0][0] + d[0], b[0][1] + d[1], b[0][2] + d[2]), *b[1:]) for b in boxes]


def leaf_boxes():
    """The door leaf in the frame of the hinge line at bench height, door at its start angle (built from the near
    face, then moved into the hinge's frame)."""
    y0, y1 = 0.0, DOOR_W                                   # hinge edge to free edge
    yk, zk = YK - HY, ZK
    b = [bx(0.0, DOOR_T, y0, yk - WIN_Y, DOOR_Z0, DOOR_Z1, OAK),
         bx(0.0, DOOR_T, yk + WIN_Y, y1, DOOR_Z0, DOOR_Z1, OAK),
         bx(0.0, DOOR_T, yk - WIN_Y, yk + WIN_Y, zk + WIN_UP, DOOR_Z1, OAK),
         bx(0.0, DOOR_T, yk - WIN_Y, yk + WIN_Y, DOOR_Z0, zk - WIN_DN, OAK)]
    cx0, cx1 = CASE_X0 - XF, CASE_X1 - XF
    walls = (wall(0.0, cx0, WOOD_FIT, 8, 3, 0.018, OAK) + wall(cx0, cx0 + PLATE, FIT, 16, 6, 0.008, STEEL)
             + wall(cx1 - PLATE, cx1, FIT, 16, 6, 0.008, STEEL) + wall(cx1, DOOR_T, WOOD_FIT, 8, 3, 0.018, OAK))
    b += _shift(walls, (0.0, yk, zk))
    # forend on the free edge, with its screw hole and the withdrawn bolt's face; hinge knuckles on the near face
    fx = DOOR_T / 2
    fz = CASE_Z0 + CASE_H / 2
    b += [bx(fx - FOREND_W / 2, fx + FOREND_W / 2, y1 - FOREND_T, y1 + 0.0003, fz - FOREND_L / 2, fz + FOREND_L / 2, STEEL),
          bx(fx - 0.003, fx + 0.003, y1, y1 + 0.0005, zk - SCREW_HOLE - 0.003, zk - SCREW_HOLE + 0.003, DARK),
          bx(fx - 0.005, fx + 0.005, y1, y1 + 0.0005, zk + 0.030, zk + 0.056, BRASS)]
    for z in (DOOR_Z0 + 0.045, DOOR_Z1 - 0.045 - 0.075):
        b.append(bx(DOOR_T, DOOR_T + 0.0028, 0.0005, 0.010, z, z + 0.075, STEEL))
    return _shift(b, (XF - HX, 0.0, 0.0))


DOOR_ORIGIN = (HX, HY, TOP)
LEAF = Hinge("leaf", pivot=(0.0, 0.0, 0.0), boxes=tuple(leaf_boxes()), lower=SWING_LO, upper=SWING_HI, stiffness=0.0,
             damping=50.0, effort=HINGE_FRICTION, mass=DOOR_MASS, axis="Z")
# the hinge post (a 32 x 115 mm door-lining offcut) behind the door's hinge edge, on its base board, both fixed to the bench
POST_T, POST_W, POST_H = 0.032, 0.115, 0.320
BASE_X0, BASE_X1, BASE_Y0, BASE_Y1 = XF - 0.020, XF + 0.140, HY - POST_W - 0.010, EDGE + 0.030
DOOR = part("door", pos=DOOR_ORIGIN, mass=4.0, boxes=[
    bx(0.003, 0.003 + POST_T, -POST_W, 0.0, BASE_T, BASE_T + POST_H, PINE),
    bx(BASE_X0 - HX, BASE_X1 - HX, BASE_Y0 - HY, BASE_Y1 - HY, 0.0, BASE_T, PLY)],
    hinges=(LEAF,), color=OAK, friction=SLIDE_FRICTION, fixed=True,
    meta={"contact_offset": OFFSET, "grip_boxes_leaf": (0, 1, 2, 3), "grip_friction": 0.5})   # the door's faces: 0.5

# -- the two cylinders and their keys ----------------------------------------------------------------------------------
OLD_POS = (SEAT[0], SEAT[1], TOP + SEAT[2])
KEY_IN = -CYL_L / 2 - KEY_GAP                   # the key's origin in the cylinder's frame when home
OLD = cylinder_parts("old_cylinder", OLD_POS, BRASS, cam_rest=OLD_CAM_REST)
# The new cylinder stands on the bench on its inner end, key up, web toward the robot.
NEW_XY = (0.150, -0.270)            # its plug axis
NEW_POS = (NEW_XY[0], NEW_XY[1], TOP + CYL_L / 2 + 0.0005)
_LQ = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))      # its +x (the end that goes in first) pointing down
NEW = cylinder_parts("new_cylinder", NEW_POS, SATIN, quat_xyzw=_LQ)

# -- the parts tray with the fixing screw ----------------------------------------------------------------------------------
TRAY_XY = (0.285, -0.410)
TRAY_R, TRAY_BASE_R, TRAY_H, TRAY_WALL = 0.075, 0.058, 0.020, 0.0025
TRAY_MASS = 0.343
TRAY = mk.mesh_part("tray", pos=(TRAY_XY[0], TRAY_XY[1], TOP + 0.0005), mass=TRAY_MASS, friction=0.6,
                    contact_offset=0.001, shapes=[
                        mk.Shape(mk.cup(TRAY_BASE_R, TRAY_H, TRAY_WALL, top_radius=TRAY_R), color=INOX, collider="none",
                                 metallic=0.8),
                        *mk.hulls(mk.cup_pieces(TRAY_BASE_R, TRAY_H, TRAY_WALL, top_radius=TRAY_R))])
SCREW_L, SCREW_R, SCREW_MASS = 0.060, 0.0025, 0.009
SCREW_HEAD_R = 0.0047
SCREW_AT = (0.0, -0.036)            # in the tray, from its centre: on the right, lying along x, head toward the robot
SCREW = mk.mesh_part("screw", pos=(TRAY_XY[0] + SCREW_AT[0], TRAY_XY[1] + SCREW_AT[1],
                                   TOP + 0.0005 + TRAY_WALL + SCREW_HEAD_R + 0.0003),
                     mass=SCREW_MASS, friction=0.5, contact_offset=0.001, shapes=[
                         mk.Shape(loft([(-SCREW_L / 2, circle(SCREW_R, 8)), (SCREW_L / 2, circle(SCREW_R, 8))]),
                                  color=STEEL, collider="hull", metallic=0.7),
                         mk.Shape(loft([(-SCREW_L / 2, circle(SCREW_HEAD_R, 8)), (-SCREW_L / 2 + 0.003, circle(SCREW_R, 8))]),
                                  color=STEEL, collider="hull", metallic=0.7)])

OBJECTS = [DOOR, OLD, NEW, TRAY, SCREW]

# -- scoring -----------------------------------------------------------------------------------------------------------
CENTRED = 0.0015           # the new cylinder's screw hole within this of the forend's
IN_HOLE = 0.002            # ... with its axis within 2 mm of the lock's and 3 degrees of square
SQUARE = math.radians(3.0)
CAM_OUT = math.radians(15.0)   # the cam at least this far off the centre line: the cylinder cannot be drawn out
DOOR_OK = math.radians(5.0)    # the door back within this of its start angle
RELEASE = 0.03             # grippers this far from a part's sample points
FELL = 0.05                # a cylinder this far below the bench top has left the bench
STILL_STEPS = 20
_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(step=0, scored=None, last=(0.0, False), violation=None, free={}, finger_check=None, recent=[], info={})


_reset_state()


_CYL_PTS = np.array([[x, 0.0, z] for x in (-CYL_L / 2, 0.0, CYL_L / 2) for z in (0.0, -WEB_C)])
_BOW_X = -CYL_L / 2 - KEY_GAP - NECK_L - BOW / 2       # the key's bow centre along the cylinder (cylinder frame)


def part_points(cyl: PartState) -> list:
    """World sample points of a cylinder with its key: the body, and the key's bow turned with the plug."""
    a = cyl.hinges.get("plug", 0.0)
    c, s = math.cos(a), math.sin(a)
    bow = [(_BOW_X, -s * z, c * z) for z in (BOW_Z - BOW / 2, BOW_Z, BOW_Z + BOW / 2)]
    neck = [(-CYL_L / 2 - KEY_GAP - NECK_L / 2, 0.0, 0.0)]
    return [cyl.to_world(p) for p in list(_CYL_PTS) + bow + neck]


def in_door_frame(q: float, p_world) -> np.ndarray:
    """A world point in the frame of the door at its start angle (x, y, height above the bench), the door at ``q``."""
    dx, dy = p_world[0] - HX, p_world[1] - HY
    c, s = math.cos(-q), math.sin(-q)
    return np.array([HX + c * dx - s * dy, HY + s * dx + c * dy, p_world[2] - TOP])


def seat_offset(door: PartState, cyl: PartState) -> tuple[float, float, float]:
    """Where a cylinder is relative to the lock: (how far out toward the key side along the lock's axis, its axis's
    distance from the lock's axis at mid-length, the angle between the two axes)."""
    q = door.hinges["leaf"]
    p = in_door_frame(q, cyl.pos)
    axis = cyl.rotation()[:, 0]
    normal = np.array([math.cos(q), math.sin(q), 0.0])
    return float(SEAT[0] - p[0]), float(math.hypot(p[1] - SEAT[1], p[2] - SEAT[2])), float(
        math.acos(max(-1.0, min(1.0, float(axis @ normal)))))


def in_lock(door: PartState, cyl: PartState) -> bool:
    """Any part of the cylinder is inside the door's hole."""
    out, off, tilt = seat_offset(door, cyl)
    return bool(abs(out) < DOOR_T / 2 + CYL_L / 2 and off < 0.012 and tilt < math.radians(25.0))


def cam_off_line(cyl: PartState) -> float:
    """The cam's angle from the body's centre line (rad, 0 to pi)."""
    a = cyl.hinges["plug"] - TURN
    return abs(math.atan2(math.sin(a), math.cos(a)))


def in_tray(tray, pts_world) -> bool:
    """Every sample point is over the tray's floor or rim and below its rim's height plus the part."""
    w = np.asarray(pts_world, float)
    d = np.hypot(w[:, 0] - tray.pos[0], w[:, 1] - tray.pos[1])
    return bool(d.max() < TRAY_R and w[:, 2].min() - tray.pos[2] < TRAY_H + 0.010)


def _door_distance(q: float, p_world) -> float:
    p = in_door_frame(q, p_world)
    d = (max(XF - p[0], 0.0, p[0] - XF - DOOR_T), max(HY - p[1], 0.0, p[1] - EDGE), max(DOOR_Z0 - p[2], 0.0, p[2] - DOOR_Z1))
    return float(math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))


_NAMES = ["door", "old_cylinder", "new_cylinder"]


@on_step
def _step(environment) -> None:
    """Every control step: count steps and check where the grippers are (score() cannot see the robot)."""
    from yam_arena.scene import object_states

    _STATE["step"] += 1
    try:
        st = object_states(environment, _NAMES)
        pts = finger_points(environment)
        free = {}
        for name in ("old_cylinder", "new_cylinder"):
            w = part_points(st[name])
            free[name] = bool(min(np.linalg.norm(np.asarray(p) - np.asarray(k)) for p in pts for k in w) >= RELEASE)
        q = st["door"].hinges["leaf"]
        free["door"] = bool(min(_door_distance(q, p) for p in pts) >= RELEASE)
        _STATE["free"], _STATE["finger_check"] = free, True
    except Exception as exc:  # noqa: BLE001 - an unknown grip never counts as released
        if _STATE["finger_check"] is None:
            print(f"euro-cylinder-swap: gripper check unavailable: {exc!r}", flush=True)
        _STATE["finger_check"], _STATE["free"] = False, {}


def levels(objects, free) -> dict:
    """The end-state checks, each part's only once both grippers are 3 cm from it."""
    door, old, new, tray, screw = (objects[n] for n in ("door", "old_cylinder", "new_cylinder", "tray", "screw"))
    q = door.hinges["leaf"]
    old_out, _, _ = seat_offset(door, old)
    old_in = in_lock(door, old)
    # in the tray: the cylinder's body (its key, left in, may reach over the tray's rim)
    old_stowed = ((not old_in) and in_tray(tray, [old.to_world(p) for p in _CYL_PTS])
                  and free.get("old_cylinder", False))
    new_out, new_off, new_tilt = seat_offset(door, new)
    new_free = free.get("new_cylinder", False)
    centred = abs(new_out) <= CENTRED and new_off <= IN_HOLE and new_tilt <= SQUARE and new_free
    cam = cam_off_line(new)
    locked = centred and cam >= CAM_OUT
    screw_ok = in_tray(tray, [screw.to_world((-SCREW_L / 2, 0.0, 0.0)), screw.to_world((SCREW_L / 2, 0.0, 0.0))])
    door_ok = abs(q) <= DOOR_OK and free.get("door", False)
    return {"q": q, "old_out": old_out, "old_in": old_in, "old_stowed": old_stowed, "new_out": new_out,
            "new_off": new_off, "new_tilt": new_tilt, "centred": centred, "cam": cam, "locked": locked,
            "screw_ok": screw_ok, "door_ok": door_ok}


def _update(objects) -> tuple[float, bool]:
    # success() and score() both run every control step; evaluate each step once.
    if _STATE["scored"] == _STATE["step"]:
        return _STATE["last"]
    _STATE["scored"] = _STATE["step"]
    door, old, new = objects["door"], objects["old_cylinder"], objects["new_cylinder"]
    free = _STATE["free"]
    if min(old.pos[2], new.pos[2]) < TOP - FELL:
        _STATE["violation"] = _STATE["violation"] or "a cylinder left the bench"
    lv = levels(objects, free)
    q = lv["q"]
    # whole poses: each cylinder by its two ends' web points (its turn about its own axis too), its plug by the key's
    # bow, the door by its free edge, the screw by its ends
    pts = [old.to_world(p) for p in ((-CYL_L / 2, 0.0, -WEB_C), (CYL_L / 2, 0.0, -WEB_C))]
    pts += [new.to_world(p) for p in ((-CYL_L / 2, 0.0, -WEB_C), (CYL_L / 2, 0.0, -WEB_C))]
    pts += part_points(old)[-2:-1] + part_points(new)[-2:-1]
    pts += [np.array([q * DOOR_W, 0.0, 0.0])]
    pts += [objects["screw"].to_world((x, 0.0, 0.0)) for x in (-SCREW_L / 2, SCREW_L / 2)]
    # revision 2: the second counts only steps with both grippers clear of all three parts (each part's free flag
    # goes into the window), so parts held still and then let go do not score 1.0 on the first released step
    all_free = all(free.get(n, False) for n in ("old_cylinder", "new_cylinder", "door"))
    recent = _STATE["recent"]
    recent.append((np.concatenate([np.asarray(p, float) for p in pts]), all_free))
    del recent[:-STILL_STEPS]
    still = (len(recent) == STILL_STEPS and all(f for _, f in recent)
             and all(float(np.abs(r - recent[-1][0]).max()) < 0.001 for r, _ in recent))
    _STATE["info"] = {"door_deg": round(math.degrees(q), 2), "old_out_mm": round(1000 * lv["old_out"], 2),
                      "old_in_lock": lv["old_in"], "old_stowed": lv["old_stowed"],
                      "new_out_mm": round(1000 * lv["new_out"], 2), "new_off_mm": round(1000 * lv["new_off"], 2),
                      "new_tilt_deg": round(math.degrees(lv["new_tilt"]), 2), "centred": lv["centred"],
                      "cam_off_deg": round(math.degrees(lv["cam"]), 1), "locked": lv["locked"],
                      "screw_in_tray": lv["screw_ok"], "door_ok": lv["door_ok"], "free": dict(free), "still": still,
                      "violation": _STATE["violation"]}
    if _STATE["violation"] or not lv["old_stowed"]:
        result = (0.0, False)
    elif not lv["centred"]:
        result = (0.25, False)
    elif not lv["locked"]:
        result = (0.5, False)
    elif lv["door_ok"] and lv["screw_ok"] and still:
        result = (1.0, True)
    else:
        result = (0.75, False)
    _STATE["last"] = result
    return result


def success(objects) -> bool:
    """Old cylinder in the tray, new cylinder centred in the lock with its cam turned out of line, the screw in the
    tray, the door back within 5 degrees of where it started, both grippers clear, everything still for the last
    second, and no cylinder off the bench."""
    return _update(objects)[1]


def distance(objects) -> float:
    door = objects["door"]
    out, off, _ = seat_offset(door, objects["new_cylinder"])
    return float(abs(out) + off)


def score(objects) -> float:
    """End state, each level including the ones before it, and each only once both grippers are 3 cm from the parts
    it names. 0.25: the old cylinder is wholly out of the door and lies in the tray. 0.5: the new cylinder is through
    the lock case with its fixing-screw hole within 1.5 mm of the forend's screw hole (its axis within 2 mm and 3
    degrees of the lock's). 0.75: its cam is at least 15 degrees off the body's centre line (it cannot be drawn out).
    1.0: also the door is back within 5 degrees of its start angle, the screw is in the tray, and none of the parts
    has moved more than 1 mm for 1 second (whole poses) with both grippers clear of all three parts for that second.
    0 for good if a cylinder left the bench."""
    return _update(objects)[0]
