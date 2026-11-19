"""bay: the station changed for an object that stands on the floor.

The station's table is replaced by a mounting shelf on an anchored plywood stand (see ``stand_boxes``) that carries the two arm bases (and the top camera's stand) at
``MOUNT_HEIGHT`` above the floor, so the floor in front of the arms is free. Arms, grippers, control gains, action
interface and cameras are the standard ones; the top camera keeps its calibrated pose in the room.

It works by changing three constants of ``yam_arena.station`` before the embodiment builds its rig (the task file is
loaded first): the arm base positions, the table's size and place, and the table-top height that the camera stand's
foot stands on. yam_arena itself is not modified, and ``--embodiment yam_arena`` / ``--scorer task_score`` run as usual.
Demos take the bases from ``yam_arena.station.BASES`` and so follow the change.
"""

from __future__ import annotations

FLOOR_Z = -0.0125            # the station's floor in the world frame (yam_arena.station.FLOOR_Z)
MOUNT_HEIGHT = 0.65          # arm base plane above the floor (the station table puts it at 0.7625)
SHELF_DEPTH, SHELF_WIDTH, SHELF_THICK = 0.36, 1.30, 0.04
SHELF_FRONT_X = 0.05         # the shelf's front edge, ahead of the arm bases (which are at x = 0)
BASE_Z = FLOOR_Z + MOUNT_HEIGHT
SHELF_TOP = BASE_Z - 0.0025  # the arm bases stand 2.5 mm above the surface, as on the station table
# The stand under the shelf (fixed Boxes in the task file's OBJECTS): a closed plywood box anchored to the floor.
# 30 mm birch plywood back panel across the full width; two 30 mm end panels under the shelf's ends and two 30 mm webs
# under the arm bases, all full depth from the back panel to the shelf's front edge and full height from the floor to
# the shelf's underside. Six 50 x 50 x 5 mm steel angles, 100 mm long, tie it to the floor with one concrete screw
# anchor each: four on the back panel's rear face (they take the uplift when the arms reach forward) and two inside
# the end panels at their front ends (they take the uplift when the arms push down on the floor in front). Nothing of
# it is ahead of the shelf's front edge (x = 0.05), so the floor in front stays free.
PANEL_THICK = 0.03           # back panel, end panels and webs
PANEL_X = SHELF_FRONT_X - SHELF_DEPTH + PANEL_THICK / 2
PANEL_HEIGHT = SHELF_TOP - SHELF_THICK - FLOOR_Z
PANEL_Z = FLOOR_Z + PANEL_HEIGHT / 2
SIDE_X0 = SHELF_FRONT_X - SHELF_DEPTH + PANEL_THICK      # end panels and webs: from the back panel's front face ...
SIDE_X1 = SHELF_FRONT_X                                   # ... to the shelf's front edge
SIDE_Y = (SHELF_WIDTH / 2 - PANEL_THICK / 2, 0.305)       # centre planes |y|: end panels, webs (under the arm bases)
ANGLE_LEG, ANGLE_T, ANGLE_LEN = 0.05, 0.005, 0.10         # steel floor angles: legs, thickness, length
BACK_FACE_X = SHELF_FRONT_X - SHELF_DEPTH                 # the back panel's rear face (x = -0.31)
REAR_ANGLE_Y = (0.20, 0.55)                               # |y| of the rear angles' centres
REAR_ANCHOR_X = BACK_FACE_X - ANGLE_LEG / 2               # x = -0.335
END_INNER_Y = SHELF_WIDTH / 2 - PANEL_THICK               # the end panels' inner faces (|y| = 0.62)
FRONT_ANGLE_X = SHELF_FRONT_X - 0.02 - ANGLE_LEN / 2      # front angles' centre (x = -0.02), 20 mm behind the front
FRONT_ANCHOR_X = FRONT_ANGLE_X


def stand_boxes() -> list:
    """The stand as (name, centre, size): back panel, end panels, webs, and each floor angle as two legs."""
    out = [("mount_panel", (PANEL_X, 0.0, PANEL_Z), (PANEL_THICK, SHELF_WIDTH, PANEL_HEIGHT))]
    cx, lx = (SIDE_X0 + SIDE_X1) / 2, SIDE_X1 - SIDE_X0
    for kind, y in zip(("end", "web"), SIDE_Y):
        for s, tag in ((1, "l"), (-1, "r")):
            out.append((f"mount_{kind}_{tag}", (cx, s * y, PANEL_Z), (lx, PANEL_THICK, PANEL_HEIGHT)))
    floor_z, up_z = FLOOR_Z + ANGLE_T / 2, FLOOR_Z + ANGLE_LEG / 2
    for k, y in enumerate((-REAR_ANGLE_Y[1], -REAR_ANGLE_Y[0], REAR_ANGLE_Y[0], REAR_ANGLE_Y[1])):
        out.append((f"mount_angle_rear{k}_floor", (BACK_FACE_X - ANGLE_LEG / 2, y, floor_z), (ANGLE_LEG, ANGLE_LEN, ANGLE_T)))
        out.append((f"mount_angle_rear{k}_up", (BACK_FACE_X - ANGLE_T / 2, y, up_z), (ANGLE_T, ANGLE_LEN, ANGLE_LEG)))
    for s, tag in ((1, "l"), (-1, "r")):
        out.append((f"mount_angle_front_{tag}_floor", (FRONT_ANGLE_X, s * (END_INNER_Y - ANGLE_LEG / 2), floor_z),
                    (ANGLE_LEN, ANGLE_LEG, ANGLE_T)))
        out.append((f"mount_angle_front_{tag}_up", (FRONT_ANGLE_X, s * (END_INNER_Y - ANGLE_T / 2), up_z),
                    (ANGLE_LEN, ANGLE_T, ANGLE_LEG)))
    return out


def install() -> bool:
    """Apply the change (idempotent). Returns False where yam_arena's station module is not available (offline
    tests)."""
    try:
        from yam_arena import station
    except Exception:  # noqa: BLE001
        return False
    if getattr(station, "_floor_bay", False):
        return True
    station.BASES = {side: (float(p[0]), float(p[1]), BASE_Z) for side, p in station.BASES.items()}
    station.TABLE_TOP = SHELF_TOP
    station.TABLE_SIZE = (SHELF_DEPTH, SHELF_WIDTH, SHELF_THICK)
    station.TABLE_CENTER = (SHELF_FRONT_X - SHELF_DEPTH / 2, 0.0, SHELF_TOP - SHELF_THICK / 2)
    station._floor_bay = True
    return True


install()
