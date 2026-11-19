"""bay: the station changed for an object that stands on the floor.

The station's table is replaced by a mounting shelf that carries the two arm bases (and the top camera's stand) at
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
PANEL_THICK = 0.03           # the back panel that carries the shelf (a fixed Box in the task file's OBJECTS)
PANEL_X = SHELF_FRONT_X - SHELF_DEPTH + PANEL_THICK / 2
PANEL_HEIGHT = SHELF_TOP - SHELF_THICK - FLOOR_Z
PANEL_Z = FLOOR_Z + PANEL_HEIGHT / 2


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
