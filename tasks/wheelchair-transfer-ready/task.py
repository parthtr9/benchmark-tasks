"""A standard folding manual wheelchair, a Drive Medical Silver Sport 2 (18 in seat, desk arms, swing-away footrests,
item SSP218DDA-SF), stands open on the floor in front of the arms, facing them, parked for a transfer: both wheel
locks are on at the start. The station's table is replaced by a mounting shelf on an anchored plywood stand that
carries the arm bases 0.65 m above the floor (see bay.py).

The chair is at its catalogue size: 24 x 1 in rear wheels with handrims, 8 in casters on swivel forks, steel tube
frame, sling seat and back (rigid panels here), desk-length arm pads. Nothing is fixed to the world. The two rear
wheels are held by their wheel locks (the lock is drawn pushed fully forward, its shoe pressed into the tyre; in the
simulation the locked wheel's axle joint is locked, the lock's holding torque is not modelled); the casters roll and
swivel freely. Each wheel collides as convex pie slices (one centred under the axle) with a rim point every 1.5
degrees (3 on a caster): a single cylinder prim collides as a coarse prism, and the chair then hops by itself.

Each swing-away footrest (leg rest) hangs by two greased sleeves on two greased 28 mm hinge pins on brackets of the
front frame post and is held pointing forward by a spring latch: a tooth on the footrest's release lever sits in a
notch of the lower bracket's index plate. Pressing the lever's red paddle (on the outside of the hanger tube) toward
the tube withdraws the tooth by contact; the footrest then swings outward about the pins and lifts off them. Nothing
in the code sets the latch, the pins or the footrest's state: release, swing and lift are contact physics.

Beside the chair, on each side, a blue floor mat marks where the footrests go.

World frame: x forward from the arms, y left, z up; the floor is at z = -0.0125. The chair's own frame has x toward
its front (toward the arms), so the chair's left side is at negative world y, in front of the right arm.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bay  # noqa: E402,F401  (replaces the station table by the arm mounting shelf)
import meshkit as mk  # noqa: E402
import numpy as np  # noqa: E402
from partkit import Hinge, Material, PartState, on_reset, on_step, part  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
CONTROL_HZ = 20
FLOOR = bay.FLOOR_Z

INSTRUCTION = "Take both leg rests off the wheelchair and set them down on the floor mats beside it."

SILVER = (0.70, 0.71, 0.73)
BLACK = (0.06, 0.06, 0.07)
NYLON = (0.10, 0.10, 0.12)
GREY = (0.45, 0.46, 0.48)
STEEL = (0.62, 0.64, 0.68)
RED = (0.75, 0.10, 0.08)
BLUE = (0.10, 0.30, 0.65)
WOOD = (0.80, 0.62, 0.40)

# -- the chair, in its own frame: x forward, y to its left, z up, origin on the floor under the rear axle ---------------
CHAIR_X = 0.85                    # world x of the rear axle; the chair faces the arms (yaw 180 degrees)
WHEEL_R, TYRE_W, WHEEL_Y = 0.305, 0.025, 0.295        # 24 x 1 in rear wheel; tyre centre plane
RAIL_Y, TUBE_R = 0.24, 0.011      # side frames 480 mm apart (18 in seat), 22 mm tube
SEAT_Z = 0.482                    # seat rail centre; the sling's top is 0.495 m (19.5 in) above the floor
POST_X = 0.40                     # front frame post
CASTER_R, CASTER_W, CASTER_TRAIL = 0.1016, 0.025, 0.045   # 8 x 1 in caster, axle 45 mm behind the swivel
CHAIR_MASS = 16.8                 # chair without footrests (kg); with both, 18.6 kg (41 lb)
WHEEL_MASS, CASTER_MASS, FORK_MASS = 1.9, 0.35, 0.25
TYRE = Material(friction=0.9)                                      # solid polyurethane tyre
PIN = Material(friction=0.15)                                      # greased steel pins in the leg rests' sleeves
FOOTREST_FRICTION = 0.8                                            # chromed steel tube, plastic plate
CASTER_TYRE = Material(friction=0.9)
# Rolling resistance of the solid tyres, as viscous damping in the axles: about 2 N to roll the chair at 5 cm/s
WHEEL_DAMPING, CASTER_DAMPING = 1.0, 0.1         # N m s/rad
SPAWN_LIFT = 0.0003               # the chair is placed this far above the floor (it does not drop onto it)


def round_hulls(radius: float, width: float, pieces: int) -> list:
    """A wheel's collider about the y axis through the origin: ``pieces`` convex pie slices, each with a rim point
    every 360 / (30 * pieces) degrees (64 points per slice, PhysX's GPU hull limit), so the tread is round to a few
    hundredths of a millimetre (a single cylinder prim collides as a coarse prism)."""
    step = 360.0 / pieces
    half = step / 60.0           # half a rim step: the wheel stands on a flat, not on a vertex
    prof = [(0.0, -width / 2), (radius, -width / 2), (radius, width / 2), (0.0, width / 2)]
    first = 270.0 - step / 2 + half      # one slice is centred under the axle, so the floor contact is on one hull
    return [mk.revolve(prof, 30, angle=step, start=first + k * step).rotated("X", 90) for k in range(pieces)]


def world(p) -> np.ndarray:
    """A chair-frame point in the world, with the chair where the task file puts it."""
    return np.array([CHAIR_X - p[0], -p[1], FLOOR + p[2]])


# The wheel lock (push-to-lock), side plane (x, z): handle pivot A, shoe-arm pivot B, toggle link from the handle's
# lower arm (length LOCK_A) to the shoe. At dead centre the shoe is at LOCK_P, pressed LOCK_EMBED into the tyre.
LOCK_A_PIVOT = (0.335, 0.452)
LOCK_B_PIVOT = (0.320, 0.380)
LOCK_ARM = 0.026                  # handle pivot to the toggle link's pin
SHOE_R = 0.005                    # the shoe: a 10 mm steel bar across the tread
LOCK_CENTRE_EMBED = 0.0040
LOCK_CENTRE_AT = math.radians(24.0)       # where on the tyre the shoe bears, above the axle's level
LOCK_BEFORE, LOCK_PAST = math.radians(38.0), math.radians(7.0)    # handle travel before and past dead centre
LOCK_TRAVEL = LOCK_BEFORE + LOCK_PAST                               # 45 degrees, stop to stop
HANDLE_UNLOCKED = 0.0                      # the grip stands upright when unlocked (45 degrees forward when locked)
HANDLE_LEN, GRIP_FROM, GRIP_R = 0.115, 0.035, 0.011
LOCK_Y = (0.2555, 0.2610, 0.2670)         # the shoe arm's, the toggle link's and the handle's planes


def _rot_fwd(v, th):
    """Turn (x, z) by th the way the handle turns when pushed forward (about the chair's +y axis)."""
    c, s = math.cos(th), math.sin(th)
    return (v[0] * c + v[1] * s, -v[0] * s + v[1] * c)


def lock_linkage(th: float) -> dict:
    """The four-bar at handle angle th (0 unlocked .. LOCK_TRAVEL at the stop): the toggle pin, the shoe's centre
    and how far the shoe is into the tyre (m, negative: clear of it)."""
    A, B = np.array(LOCK_A_PIVOT), np.array(LOCK_B_PIVOT)
    C = np.array([0.0, WHEEL_R])
    P = C + (WHEEL_R + SHOE_R - LOCK_CENTRE_EMBED) * np.array([math.cos(LOCK_CENTRE_AT), math.sin(LOCK_CENTRE_AT)])
    b = float(np.linalg.norm(P - B))
    w = (P - A) / np.linalg.norm(P - A)
    c = float(np.linalg.norm(P - A)) - LOCK_ARM

    def at(angle, prev):
        pin = A + LOCK_ARM * np.array(_rot_fwd(w, angle))
        d = pin - B
        D = float(np.linalg.norm(d))
        x = (b * b - c * c + D * D) / (2 * D)
        h = math.sqrt(max(0.0, b * b - x * x))
        e = d / D
        n = np.array([-e[1], e[0]])
        return min((B + x * e + h * n, B + x * e - h * n), key=lambda q: float(np.linalg.norm(q - prev))), pin

    target = th - LOCK_BEFORE          # angle from dead centre
    shoe, pin = P, A + LOCK_ARM * w
    steps = max(1, int(abs(target) / math.radians(1.0)) + 1)
    for k in range(1, steps + 1):
        shoe, pin = at(target * k / steps, shoe)
    return {"pin": pin, "shoe": shoe, "embed": float(WHEEL_R + SHOE_R - np.linalg.norm(shoe - C)),
            "arm": b, "link": c}


_L0 = lock_linkage(0.0)
LOCK_PIN0 = tuple(float(v) for v in _L0["pin"])       # toggle pin, unlocked
LOCK_SHOE0 = tuple(float(v) for v in _L0["shoe"])     # shoe centre, unlocked
_u = (_L0["shoe"] - _L0["pin"]) / np.linalg.norm(_L0["shoe"] - _L0["pin"])
LINK_END0 = tuple(float(v) for v in _L0["shoe"] - 0.009 * _u)    # the drawn link stops at the shoe bar's surface
_L1 = lock_linkage(LOCK_TRAVEL)                       # the lock as it is at the start: pushed to its forward stop
LOCK_ON_PIN = tuple(float(v) for v in _L1["pin"])
LOCK_ON_SHOE = tuple(float(v) for v in _L1["shoe"])


def handle_point(th: float, along: float) -> np.ndarray:
    """A point on the handle's centre line (chair x, z), ``along`` metres from its pivot, at handle angle th."""
    lean = HANDLE_UNLOCKED + th
    return np.array(LOCK_A_PIVOT) + along * np.array([math.sin(lean), math.cos(lean)])


# The footrest mount: two hinge pins on brackets ahead of the front post; index plate with a notch on the lower one.
PIN_X = 0.45                      # the pins' axis
PIN_R, PIN_LEN = 0.004, 0.028
LOWER_SEAT_Z = 0.342              # top of the lower bracket's index plate: the lower sleeve sits on it
UPPER_ARM_TOP = 0.430
PLATE_R, NOTCH_R, NOTCH_HALF = 0.022, 0.014, 0.0047   # index plate radius, notch floor radius, notch half width
PLATE_Z = (0.326, 0.338)          # the notched layer; a 4 mm cap above it keeps the tooth from lifting out
TOOTH_Z = (0.329, 0.336)
# The footrest (chair frame, left one; the right one is its mirror image). Origin: pin axis at LOWER_SEAT_Z.
HANGER_TOP, HANGER_FOOT = (0.49, 0.46), (0.69, 0.14)          # the hanger tube's centre line (x, z), 22 mm tube
FOOTPLATE = ((0.685, 0.14, 0.090), (0.15, 0.20, 0.018))        # centre (y for the left one), size: plastic plate
SLEEVE_HOLE, SLEEVE_WALL, SLEEVE_H = 0.0105, 0.0035, 0.012
LEVER_PIVOT = (0.50, 0.267)       # the release lever's pivot (x, y), a vertical pin on the hanger's bracket
LEVER_TOOTH_X = 0.45              # the tooth, 50 mm behind the pivot
PADDLE_X, PADDLE_Z = (0.555, 0.605), (0.285, 0.345)            # the paddle, 80 mm ahead of the pivot
PADDLE_Y = (0.273, 0.277)
LEVER_TRAVEL = 0.21               # rad at the stop; the tooth clears the plate from about 0.13 rad
LEVER_RATE, LEVER_PRELOAD = 2.0, 0.10                           # N m/rad and rad: 2.5 N at the paddle at rest, 8 N pressed
GRIP_X = 0.58                     # where a gripper takes the hanger tube and the paddle together
FOOTREST_MASS = 0.9
FOOTREST_COM = (0.152, -0.028, -0.095)     # from the footrest's origin (left one)
FOOTREST_INERTIA = (0.018, 0.021, 0.008)


def hanger_z(x: float) -> float:
    (x0, z0), (x1, z1) = HANGER_TOP, HANGER_FOOT
    return z0 + (z1 - z0) * (x - x0) / (x1 - x0)


GRIP_Z = hanger_z(GRIP_X)


def _tube(points, r=TUBE_R, sides=12):
    return mk.tube([tuple(float(v) for v in p) for p in points], r, sides)


def _ring_y(radius, tube_r, y, n=48):
    """A torus about the y axis (a tyre, a handrim), centred on the origin plane y."""
    return mk.tube([(radius * math.cos(2 * math.pi * k / n), y, radius * math.sin(2 * math.pi * k / n))
                    for k in range(n)], tube_r, 10, closed=True)


def _disc_y(r_in, r_out, half_w):
    """An annulus (or disc) about the y axis, centred on the origin."""
    prof = [(r_in, -half_w), (r_out, -half_w), (r_out, half_w), (r_in, half_w)]
    return mk.revolve(prof, 48).rotated("X", 90)


def _plate_rim(sy, z0, z1):
    """The index plate's rim (an annulus from just inside NOTCH_R out to PLATE_R) as convex prisms about the origin,
    leaving a notch with parallel walls, 2 * NOTCH_HALF wide, toward the chair's outside (+y on the left side)."""
    r0, r1, h = NOTCH_R - 0.002, PLATE_R, NOTCH_HALF
    edge = math.degrees(math.asin(h / r1))
    at = lambda r, t: (r * math.sin(math.radians(t)), sy * r * math.cos(math.radians(t)))  # noqa: E731
    angles = [edge, *range(30, 331, 30), 360.0 - edge]
    pieces = []
    for a, b in zip(angles, angles[1:]):
        outer = [at(r1, a + (b - a) * k / 4) for k in range(5)]
        inner_b = (-h, sy * math.sqrt(r0 * r0 - h * h)) if b == angles[-1] else at(r0, b)
        inner_a = (h, sy * math.sqrt(r0 * r0 - h * h)) if a == angles[0] else at(r0, a)
        outline = [*outer, inner_b, inner_a]
        pieces.append(mk.extrude(outline if sy > 0 else outline[::-1], z1 - z0, z0=z0))
    return pieces


def _chair():
    boxes = [
        ((0.19, 0.0, 0.492), (0.41, 0.457, 0.006), NYLON),                       # seat sling (rigid panel)
        ((-0.036, 0.0, 0.70), (0.006, 0.457, 0.40), NYLON),                      # back sling (rigid panel)
    ]
    cylinders = []
    meshes = []       # (mesh, colour, collider)
    hinges, loops = [], []
    attach = {}
    for sy, tag in ((1, "l"), (-1, "r")):
        y = sy * RAIL_Y
        boxes += [
            ((0.098, y, 0.722), (0.254, 0.05, 0.025), BLACK),                      # desk-length arm pad (10 in)
            ((0.11, y, 0.59), (0.18, 0.003, 0.17), NYLON),                         # arm skirt guard
            ((0.0, sy * 0.262, WHEEL_R), (0.05, 0.022, 0.07), STEEL),              # axle plate
            # footrest hinge brackets on the front post: lower arm, upper arm
            (((POST_X + PIN_X) / 2, y, 0.322), (PIN_X - POST_X + 0.016, 0.016, 0.008), STEEL),
            (((POST_X + PIN_X) / 2, y, UPPER_ARM_TOP - 0.004), (PIN_X - POST_X + 0.016, 0.016, 0.008), STEEL),
            # wheel lock clamp on the side frame (carries the two pivots)
            ((0.3275, sy * 0.2525, 0.43), (0.05, 0.003, 0.11), STEEL),
        ]
        cylinders += [
            ((POST_X, y, 0.240), 0.014, 0.05, SILVER),                             # caster stem housing
            ((PIN_X, y, PLATE_Z[1] + 0.002), PLATE_R, 0.004, STEEL),               # index plate cap
            ((PIN_X, y, (PLATE_Z[0] + PLATE_Z[1]) / 2), NOTCH_R, PLATE_Z[1] - PLATE_Z[0], STEEL),   # plate core
        ]
        # the two hinge pins: greased steel, on a link welded to the frame (a joint locked at 0) so that they carry
        # their own low-friction contact material
        hinges.append(Hinge(f"pins_{tag}", pivot=(PIN_X, y, LOWER_SEAT_Z), axis="Z", lower=0.0, upper=0.0, mass=0.04,
                            cylinders=(((PIN_X, y, LOWER_SEAT_Z + PIN_LEN / 2), PIN_R, PIN_LEN, STEEL),
                                       ((PIN_X, y, UPPER_ARM_TOP + PIN_LEN / 2), PIN_R, PIN_LEN, STEEL)),
                            material=PIN))
        # index plate rim, notch toward the outside
        meshes += [(m.moved((PIN_X, y, 0.0)), STEEL, "hull") for m in _plate_rim(sy, *PLATE_Z)]
        # frame tubes: straight runs collide (convex hulls), bends are drawn only
        meshes += [
            (_tube(mk.fillet_path([(-0.03, y, SEAT_Z), (POST_X, y, SEAT_Z), (POST_X, y, 0.215)], 0.03)), SILVER, "none"),
            (_tube([(-0.03, y, SEAT_Z), (POST_X - 0.03, y, SEAT_Z)]), SILVER, "hull"),
            (_tube([(POST_X, y, SEAT_Z - 0.03), (POST_X, y, 0.265)]), SILVER, "hull"),
            (_tube(mk.fillet_path([(-0.03, y, 0.20), (-0.03, y, 0.875), (-0.10, y, 0.905)], 0.04)), SILVER, "none"),
            (_tube([(-0.03, y, 0.20), (-0.03, y, 0.84)]), SILVER, "hull"),
            (_tube([(-0.085, y, 0.8986), (-0.19, y, 0.9436)], 0.016), BLACK, "hull"),            # push handle grip
            (_tube([(-0.03, y, 0.25), (POST_X, y, 0.25)]), SILVER, "hull"),                        # lower side rail
            (_tube(mk.fillet_path([(0.02, y, SEAT_Z), (0.02, y, 0.70), (0.21, y, 0.70), (0.21, y, SEAT_Z)], 0.03),
                   0.0095), SILVER, "none"),                                                       # arm frame
            (_tube([(0.19, y, 0.25), (0.19, -sy * 0.20, 0.465)], 0.0095), SILVER, "none"),         # cross brace
            (_tube([(0.0, sy * 0.24, WHEEL_R), (0.0, sy * (WHEEL_Y + 0.02), WHEEL_R)], 0.008), STEEL, "none"),   # axle
        ]
        # rear wheel: the tyre collides as a cylinder (not drawn); tyre, rim, spokes, hub and handrim are drawn
        wy = sy * WHEEL_Y
        # the wheel lock is on: the wheel does not turn on its axle (a locked joint)
        hinges.append(Hinge(f"wheel_{tag}", pivot=(0.0, wy, WHEEL_R), axis="Y", lower=0.0, upper=0.0,
                            mass=WHEEL_MASS, damping=WHEEL_DAMPING, friction=0.05, material=TYRE))
        wheel = [(_disc_y(WHEEL_R - 0.022, WHEEL_R, TYRE_W / 2), GREY), (_disc_y(WHEEL_R - 0.04, WHEEL_R - 0.022, 0.009), BLACK),
                 (_disc_y(0.0, 0.035, 0.022), BLACK), (_ring_y(0.265, 0.008, sy * 0.030), STEEL)]
        for k in range(5):
            wheel.append((mk.box((0.27, 0.014, 0.034), (0.15, 0.0, 0.0)).rotated("Y", 72.0 * k), BLACK))
        for k in range(6):   # handrim stand-offs
            wheel.append((mk.box((0.012, 0.03, 0.012), (0.265, sy * 0.015, 0.0)).rotated("Y", 60.0 * k + 30.0), STEEL))
        attach[f"wheel_{tag}"] = [(m.moved((0.0, wy, WHEEL_R)), c, "none") for m, c in wheel]
        attach[f"wheel_{tag}"] += [(m.moved((0.0, wy, WHEEL_R)), None, "hull") for m in round_hulls(WHEEL_R, TYRE_W, 8)]
        # caster: swivel fork under the front post, wheel trailing behind the swivel
        cx = POST_X - CASTER_TRAIL
        hinges.append(Hinge(f"fork_{tag}", pivot=(POST_X, y, 0.212), axis="Z",
                            lower=None, upper=None,
                            boxes=(((POST_X - 0.01, y, 0.207), (0.05, 0.040, 0.008), SILVER),), mass=FORK_MASS,
                            damping=0.002, friction=0.02))
        attach[f"fork_{tag}"] = [
            (_tube([(POST_X - 0.01, y + s * 0.018, 0.205), (cx, y + s * 0.018, CASTER_R)], 0.006), SILVER, "none")
            for s in (1, -1)]
        hinges.append(Hinge(f"caster_{tag}", pivot=(cx, y, CASTER_R), axis="Y", lower=None, upper=None,
                            parent=f"fork_{tag}",
                            mass=CASTER_MASS, damping=CASTER_DAMPING, friction=0.01, material=CASTER_TYRE))
        attach[f"caster_{tag}"] = [(m.moved((cx, y, CASTER_R)), c, "none") for m, c in (
            (_disc_y(CASTER_R - 0.018, CASTER_R, CASTER_W / 2), GREY), (_disc_y(0.0, CASTER_R - 0.018, 0.010), BLACK))]
        attach[f"caster_{tag}"] += [(m.moved((cx, y, CASTER_R)), None, "hull")
                                        for m in round_hulls(CASTER_R, CASTER_W, 4)]
        # wheel lock, drawn on (handle pushed fully forward, shoe in the tyre): handle, toggle link, shoe arm, shoe.
        # Drawn only (no colliders): the lock is not part of the task, and the shoe must not fight the locked wheel.
        y_arm, y_link, y_handle = (sy * v for v in LOCK_Y)
        A, B = LOCK_A_PIVOT, LOCK_B_PIVOT
        pin, shoe = LOCK_ON_PIN, LOCK_ON_SHOE
        tip = handle_point(LOCK_TRAVEL, GRIP_FROM + 0.01)
        xz = lambda q, yy: (float(q[0]), yy, float(q[1]))  # noqa: E731
        meshes += [
            (_tube([xz(A, y_handle), xz(pin, y_handle)], 0.005), STEEL, "none"),
            (_tube([xz(A, y_handle), xz(tip, y_handle)], 0.006), STEEL, "none"),
            (_tube([xz(pin, y_link), xz(shoe, y_link)], 0.004), STEEL, "none"),
            (_tube([xz(B, y_arm), xz(shoe, y_arm)], 0.005), STEEL, "none"),
            (_tube([xz(shoe, sy * (LOCK_Y[0] - 0.0015)), xz(shoe, sy * (WHEEL_Y + TYRE_W / 2))], SHOE_R), STEEL, "none"),
        ]
        g0, g1 = handle_point(LOCK_TRAVEL, GRIP_FROM), handle_point(LOCK_TRAVEL, HANDLE_LEN)
        meshes.append((_tube([xz(g0, y_handle), xz(g1, y_handle)], GRIP_R), BLACK, "none"))
    chair = part("chair", pos=tuple(world((0.0, 0.0, SPAWN_LIFT))), yaw=math.pi, mass=CHAIR_MASS - 2 * (WHEEL_MASS
                 + CASTER_MASS + FORK_MASS), boxes=boxes, cylinders=cylinders, hinges=hinges, loops=loops,
                 color=SILVER, friction=0.8, self_collide=True,
                 com=(0.15, 0.0, 0.45),
                 inertia=(1.2, 0.9, 1.2), angular_damping=0.05)
    mk.attach(chair, [mk.Shape(m, color=c, collider=k, metallic=0.4 if c in (SILVER, STEEL) else 0.0)
                      for m, c, k in meshes])
    for name, shapes in attach.items():
        mk.attach(chair, [mk.Shape(m, color=c or BLACK, collider=k, visible=c is not None,
                                   metallic=0.4 if c in (SILVER, STEEL) else 0.0) for m, c, k in shapes], to=name)
    return chair


def _sleeve(z0, color=STEEL):
    """Four walls round a square hole on the pin axis (footrest frame: the axis is x = 0, y = 0)."""
    h, w = SLEEVE_HOLE / 2, SLEEVE_WALL
    zc = z0 + SLEEVE_H / 2
    return [((h + w / 2, 0.0, zc), (w, SLEEVE_HOLE + 2 * w, SLEEVE_H), color),
            ((-h - w / 2, 0.0, zc), (w, SLEEVE_HOLE + 2 * w, SLEEVE_H), color),
            ((0.0, h + w / 2, zc), (SLEEVE_HOLE, w, SLEEVE_H), color),
            ((0.0, -h - w / 2, zc), (SLEEVE_HOLE, w, SLEEVE_H), color)]


def _footrest(tag: str, sy: int):
    """A swing-away footrest in its own frame: the chair's axes, origin on the pin axis at LOWER_SEAT_Z."""
    ox, oz = PIN_X, LOWER_SEAT_Z
    rel = lambda x, y, z: (x - ox, sy * (y - RAIL_Y), z - oz)  # noqa: E731  (chair-left coordinates -> this frame)
    upper_z0 = UPPER_ARM_TOP + 0.001 - oz
    upper_x = HANGER_TOP[0] + (HANGER_TOP[1] - (UPPER_ARM_TOP + 0.007)) * (HANGER_FOOT[0] - HANGER_TOP[0]) / (
        HANGER_TOP[1] - HANGER_FOOT[1])
    lower_x = HANGER_TOP[0] + (HANGER_TOP[1] - (oz + 0.006)) * (HANGER_FOOT[0] - HANGER_TOP[0]) / (
        HANGER_TOP[1] - HANGER_FOOT[1])
    half = SLEEVE_HOLE / 2 + SLEEVE_WALL
    sleeves = Hinge("sleeves", pivot=(0.0, 0.0, 0.0), axis="Z", lower=0.0, upper=0.0, mass=0.04, material=PIN,
                    boxes=tuple(_sleeve(0.0) + _sleeve(upper_z0)))    # greased sleeves, welded to the hanger
    boxes = [
        (((half + upper_x - ox) / 2, 0.0, upper_z0 + 0.006), (upper_x - ox - half, 0.012, 0.008), STEEL),   # upper hinge plate
        (((half + lower_x - ox) / 2, 0.0, 0.006), (lower_x - ox - half, 0.012, 0.008), STEEL),              # lower hinge plate
        (rel(*FOOTPLATE[0]), FOOTPLATE[1], BLACK),                                                          # footplate
        (rel(LEVER_PIVOT[0], 0.256, 0.3405), (0.016, 0.028, 0.007), STEEL),                                 # lever bracket
    ]
    t0, t1 = rel(HANGER_TOP[0], RAIL_Y, HANGER_TOP[1]), rel(HANGER_FOOT[0], RAIL_Y, HANGER_FOOT[1])
    stem = rel(HANGER_FOOT[0], RAIL_Y, FOOTPLATE[0][2])
    lever_pivot = rel(LEVER_PIVOT[0], LEVER_PIVOT[1], 0.3325)
    zc, zh = (TOOTH_Z[0] + TOOTH_Z[1]) / 2, TOOTH_Z[1] - TOOTH_Z[0]
    pz, ph = (PADDLE_Z[0] + PADDLE_Z[1]) / 2, PADDLE_Z[1] - PADDLE_Z[0]
    lever_boxes = (
        (rel((LEVER_TOOTH_X - 0.004 + PADDLE_X[0]) / 2, 0.267, zc), (PADDLE_X[0] - LEVER_TOOTH_X + 0.004, 0.006, zh), STEEL),
        (rel(LEVER_TOOTH_X, 0.260, zc), (0.008, 0.008, zh), STEEL),                                           # the tooth
        (rel(PADDLE_X[0] + 0.004, 0.2705, zc), (0.008, 0.013, zh), STEEL),                                    # paddle arm
        (rel((PADDLE_X[0] + PADDLE_X[1]) / 2, (PADDLE_Y[0] + PADDLE_Y[1]) / 2, pz),
         (PADDLE_X[1] - PADDLE_X[0], PADDLE_Y[1] - PADDLE_Y[0], ph), RED),                                    # the paddle
    )
    lo, hi = (-LEVER_TRAVEL, 0.0) if sy > 0 else (0.0, LEVER_TRAVEL)
    lever = Hinge("lever", pivot=lever_pivot, axis="Z", lower=lo, upper=hi, boxes=lever_boxes, stiffness=LEVER_RATE,
                  damping=0.02, target=sy * LEVER_PRELOAD, mass=0.05)
    origin = world((ox, sy * RAIL_Y, oz + SPAWN_LIFT))
    com = (FOOTREST_COM[0], sy * FOOTREST_COM[1], FOOTREST_COM[2])
    p = part(f"footrest_{tag}", pos=tuple(origin), yaw=math.pi, mass=FOOTREST_MASS - 0.09, boxes=boxes, hinges=(lever, sleeves),
             color=STEEL, friction=FOOTREST_FRICTION, com=com, inertia=FOOTREST_INERTIA, angular_damping=0.3)
    heel = [rel(0.62, 0.225, 0.099), rel(0.59, 0.19, 0.16), rel(0.59, 0.09, 0.16), rel(0.62, 0.055, 0.099)]
    mk.attach(p, [
        mk.Shape(_tube([t0, t1]), color=STEEL, collider="hull", metallic=0.5),
        mk.Shape(_tube([t1, stem], 0.009), color=STEEL, collider="hull", metallic=0.5),
        mk.Shape(_tube(mk.fillet_path(heel, 0.03), 0.004), color=BLACK, collider="none"),
    ])
    return p


CHAIR = _chair()
FOOTREST_L = _footrest("l", 1)      # the chair's left footrest (negative world y, in front of the right arm)
FOOTREST_R = _footrest("r", -1)

# -- the room: the mounting shelf's anchored stand and the two floor mats -----------------------------------------------
MAT_X, MAT_Y, MAT_T = (0.05, 0.75), (0.34, 1.00), 0.002         # each mat: world x range, |y| range, thickness
STAND = [Box(name, pos=c, size=sz, color=STEEL if "angle" in name else WOOD) for name, c, sz in bay.stand_boxes()]
MATS = [Box(f"mat_{tag}", pos=((MAT_X[0] + MAT_X[1]) / 2, s * (MAT_Y[0] + MAT_Y[1]) / 2, FLOOR + MAT_T / 2),
            size=(MAT_X[1] - MAT_X[0], MAT_Y[1] - MAT_Y[0], MAT_T), color=BLUE) for tag, s in (("l", -1), ("r", 1))]

OBJECTS = [CHAIR, FOOTREST_L, FOOTREST_R, *STAND, *MATS]
FOOTRESTS = {"l": FOOTREST_L, "r": FOOTREST_R}
SIDE = {"l": 1, "r": -1}

# -- scoring thresholds -------------------------------------------------------------------------------------------------------
CHAIR_MOVE = 0.10                 # the chair dragged more than 10 cm out of its parked place (rear axle midpoint) ...
CHAIR_YAW = 10.0                  # ... or turned more than 10 degrees on the spot ...
CHAIR_TILT = 10.0                 # ... or tipped more than 10 degrees: the run scores 0 (the chair is no longer parked
                                  # for the transfer; with its wheels locked it only moves if it is dragged or tipped)
SWUNG = math.radians(45.0)        # a footrest still on its pins, turned 45 degrees or more outward: released and swung
OFF_PINS = 0.04                   # footrest's lower sleeve more than 4 cm from its pin axis (or above the pin tops)
ON_FLOOR = 0.03                   # footrest's lowest point within 3 cm of the floor
PINCH_GAP = 0.030                 # a gripper holds a footrest: its fingers closed to a half gap under 30 mm (open: 47 mm)
GRIP_NEAR = 0.030                 # ... and both pad points of one pair within 30 mm of any solid part of the footrest
                                  # (the points are 6 mm behind the pad faces: the pads within about 25 mm of it; on
                                  # the GPU a pad point's distance reached 23 mm while the hand carried a leg rest)
UNSUPPORTED_STEPS = 2             # off its pins, not held, lowest point above 3 cm for 2 control steps: dropped
SLIP = 0.045                      # ... and moving with that gripper: the footrest's point at the gripper's tool point
SLIP_STEPS = 2                    # (in the footrest's own frame) moved no more than 45 mm over the last two control
                                  # steps, at any speed. Measured on the GPU: in the demo's hands it moved at most
                                  # 30 mm in 0.1 s (when the paddle springs back out of the grip the leg rest
                                  # re-seats in the jaws), and 59-76 mm in the first 0.1 s of a release 25 cm up.
                                  # The point at the hand, not the footplate: the leg rest pivots a little in the grip.
CLEAR = 0.03                      # a set-down footrest has no gripper point within 3 cm of it
STILL_CALLS = 20
STILL_TOL = 0.001

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(chair0=None, moved=False, dropped={"l": False, "r": False}, air={"l": 0, "r": 0},
                  points={"left": [], "right": []}, recent=[], steps=0, in_hand={}, slip={}, with_hand={})
    if environment is not None:
        _set_springs(environment)


_reset_state()


# -- geometry helpers -----------------------------------------------------------------------------------------------------------
def _yaw(state) -> float:
    r = state.rotation()
    return math.atan2(r[1, 0], r[0, 0])


def _tilt(state) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, state.rotation()[2, 2]))))


def footrest_points(state: PartState, tag: str) -> dict:
    """World points of a footrest: its lower sleeve (origin), the hanger tube's ends, the footplate's centre and
    corners, and the grip zone."""
    sy = SIDE[tag]
    rel = lambda x, y, z: (x - PIN_X, sy * (y - RAIL_Y), z - LOWER_SEAT_Z)  # noqa: E731
    (cx, cy, cz), (sx, syz, sz) = FOOTPLATE
    corners = [state.to_world(rel(cx + a * sx / 2, cy + b * syz / 2, cz - sz / 2)) for a in (-1, 1) for b in (-1, 1)]
    return {"origin": np.asarray(state.pos, float),
            "top": state.to_world(rel(HANGER_TOP[0], RAIL_Y, HANGER_TOP[1])),
            "foot": state.to_world(rel(HANGER_FOOT[0], RAIL_Y, HANGER_FOOT[1])),
            "plate": state.to_world(rel(cx, cy, cz)), "corners": corners,
            "grip": state.to_world(rel(GRIP_X, RAIL_Y, GRIP_Z)),
            "sleeve_top": state.to_world(rel(PIN_X, RAIL_Y, UPPER_ARM_TOP + SLEEVE_H))}


def pin_axis(chair: PartState, tag: str) -> tuple:
    """The footrest pins' axis on the chair: the point at the lower seat and the axis direction (world)."""
    p = chair.to_world((PIN_X, SIDE[tag] * RAIL_Y, LOWER_SEAT_Z))
    return np.asarray(p, float), chair.rotation()[:, 2]


def on_pins(chair: PartState, footrest: PartState, tag: str) -> bool:
    """The footrest's lower sleeve is within 4 cm of the pin axis and no higher than the pin tops above its seat."""
    p, axis = pin_axis(chair, tag)
    d = np.asarray(footrest.pos, float) - p
    up = float(d @ axis)
    return bool(np.linalg.norm(d - up * axis) <= OFF_PINS and -0.02 <= up <= PIN_LEN)


def lowest(footrest: PartState, tag: str) -> float:
    """Height of the footrest's lowest point above the floor (m)."""
    pts = footrest_points(footrest, tag)
    zs = [p[2] for p in pts["corners"]] + [pts["top"][2] - TUBE_R, pts["foot"][2] - TUBE_R, pts["origin"][2],
                                           pts["sleeve_top"][2]]
    return float(min(zs) - FLOOR)


def on_mat(footrest: PartState, tag: str) -> bool:
    """The footrest lies or stands on one of the two mats: its lowest point is within 3 cm of the floor, and the
    hanger tube's two ends and the footplate's centre are all over the same mat."""
    pts = footrest_points(footrest, tag)
    ref = [pts["top"], pts["foot"], pts["plate"]]
    if lowest(footrest, tag) > ON_FLOOR:
        return False
    for s in (1, -1):
        if all(MAT_X[0] <= p[0] <= MAT_X[1] and MAT_Y[0] <= s * p[1] <= MAT_Y[1] for p in ref):
            return True
    return False


def _seg_dist(p, a, b) -> float:
    a, b, p = np.asarray(a, float), np.asarray(b, float), np.asarray(p, float)
    t = float(np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * (b - a))))


def gripper_gap(footrest: PartState, tag: str, points) -> float:
    """Smallest distance (m) from any gripper point to the footrest's hanger tube, release paddle or footplate."""
    pts = footrest_points(footrest, tag)
    best = float("inf")
    for side in ("left", "right"):
        for p in points.get(side, []):
            p = np.asarray(p, float)
            best = min(best, _seg_dist(p, pts["top"], pts["foot"]) - TUBE_R,
                       float(np.linalg.norm(p - pts["grip"])) - 0.04, float(np.linalg.norm(p - pts["plate"])) - 0.10)
    return best


def swing_angle(chair: PartState, footrest: PartState, tag: str) -> float:
    """How far the footrest is turned outward about the vertical relative to the chair (rad, positive outward)."""
    r = chair.rotation().T @ footrest.rotation()
    return SIDE[tag] * math.atan2(r[1, 0], r[0, 0])


def footrest_solids(tag: str) -> tuple:
    """The footrest's solid parts in its own frame, as built by ``_footrest``: tubes as (end, end, radius), boxes as
    (centre, size) (hanger tube, footplate stem, footplate, hinge plates, sleeves, lever bracket), and the release
    lever's boxes (lever arm, tooth, paddle arm, paddle) with the lever's pivot, at lever angle 0."""
    p = FOOTRESTS[tag]
    sy = SIDE[tag]
    rel = lambda x, y, z: np.array((x - PIN_X, sy * (y - RAIL_Y), z - LOWER_SEAT_Z))  # noqa: E731
    t0, t1 = rel(HANGER_TOP[0], RAIL_Y, HANGER_TOP[1]), rel(HANGER_FOOT[0], RAIL_Y, HANGER_FOOT[1])
    stem = rel(HANGER_FOOT[0], RAIL_Y, FOOTPLATE[0][2])
    tubes = [(t0, t1, TUBE_R), (t1, stem, 0.009)]
    boxes = [(np.asarray(b[0], float), np.asarray(b[1], float)) for b in p.boxes]
    lever = next(h for h in p.hinges if h.name == "lever")
    for h in p.hinges:
        if h is not lever:
            boxes += [(np.asarray(b[0], float), np.asarray(b[1], float)) for b in h.boxes]
    lever_boxes = [(np.asarray(b[0], float), np.asarray(b[1], float)) for b in lever.boxes]
    return tubes, boxes, (np.asarray(lever.pivot, float), lever_boxes)


def _box_dist(q, c, size) -> float:
    return float(np.linalg.norm(np.maximum(np.abs(q - c) - size / 2, 0.0)))


def footrest_surface_dist(footrest: PartState, tag: str, point) -> float:
    """Distance (m) from a world point to the nearest solid part of the footrest (0 inside one), with the release
    lever at its current angle about its pivot."""
    q = footrest.to_local(point)
    tubes, boxes, (pivot, lever_boxes) = footrest_solids(tag)
    best = min(max(0.0, _seg_dist(q, a, b) - r) for a, b, r in tubes)
    for c, size in boxes:
        best = min(best, _box_dist(q, c, size))
    a = float(getattr(footrest, "hinges", {}).get("lever", 0.0))
    ca, sa = math.cos(a), math.sin(a)
    d = q - pivot                                   # the point in the lever's frame (turned back by its angle)
    ql = pivot + np.array((ca * d[0] + sa * d[1], -sa * d[0] + ca * d[1], d[2]))
    for c, size in lever_boxes:
        best = min(best, _box_dist(ql, c, size))
    return best


def half_gap(g) -> float:
    """A gripper's half gap (m) from its points (as from partkit.gripper_points)."""
    return float(np.linalg.norm(np.asarray(g[0], float) - np.asarray(g[1], float))) / 2 - 0.006


def grip_point(footrest: PartState, g) -> np.ndarray:
    """Where a gripper's tool point is on the footrest: the tool point in the footrest's own frame (m)."""
    return footrest.to_local(np.asarray(g[4], float))


def held(footrest: PartState, tag: str, points, with_hand=None) -> bool:
    """A gripper holds the footrest: its fingers are closed to a half gap under 30 mm, the two pads (at the tool
    point, or at the pad tips) are both within 30 mm of the footrest's solid parts, on any part of it: hanger tube,
    footplate, release lever and paddle, bracket, hinge plate or sleeve (``points`` as from partkit.gripper_points:
    the pad points at the tool point first, then at the pad tips; each lies 6 mm behind its pad face), and the
    footrest moves with that gripper (``with_hand[side]``, kept by ``_advance``: the tool point's place on the
    footrest, in the footrest's own frame, moved no more than 45 mm over the last two control steps, i.e. the footrest
    did not slide or fall out of the hand; not given: assumed)."""
    for side in ("left", "right"):
        g = points.get(side, [])
        if len(g) < 5:
            continue
        if half_gap(g) >= PINCH_GAP or not (with_hand or {}).get(side, True):
            continue
        for pair in ((g[0], g[1]), (g[2], g[3])):
            if all(footrest_surface_dist(footrest, tag, p) <= GRIP_NEAR for p in pair):
                return True
    return False


def chair_displaced(chair: PartState, chair0) -> bool:
    """The chair has left its parked place: rear axle midpoint more than 10 cm from where it stood, turned more than
    10 degrees, or tipped more than 10 degrees."""
    pos0, yaw0 = chair0
    turn = abs((_yaw(chair) - yaw0 + math.pi) % (2 * math.pi) - math.pi)
    return bool(np.linalg.norm(np.asarray(chair.pos, float)[:2] - pos0[:2]) > CHAIR_MOVE
                or math.degrees(turn) > CHAIR_YAW or _tilt(chair) > CHAIR_TILT)


# -- bookkeeping: run once per control step -----------------------------------------------------------------------------------
def _advance(objects, points=None) -> None:
    s = _STATE
    chair = objects["chair"]
    if points is not None:
        s["points"] = points
    if s["chair0"] is None:
        s["chair0"] = (np.asarray(chair.pos, float).copy(), _yaw(chair))
    s["steps"] += 1
    s["moved"] |= chair_displaced(chair, s["chair0"])
    for tag in ("l", "r"):
        # dropped = lost its support: off its pins, its lowest point more than 3 cm above the floor, and no gripper
        # holding it (closed on it and the footrest moving with it), for two control steps in a row (a footrest let go
        # above the floor, or slipping or falling out of a hand, at any speed)
        fr = objects[f"footrest_{tag}"]
        with_hand = {}
        for side in ("left", "right"):
            g = s["points"].get(side, [])
            hist = s["in_hand"].setdefault((tag, side), [])
            if len(g) < 5:
                hist.clear()
                continue
            hist.append(grip_point(fr, g))
            del hist[:-(SLIP_STEPS + 1)]
            slip = float(np.linalg.norm(hist[-1] - hist[0]))
            s["slip"][(tag, side)] = slip
            with_hand[side] = slip <= SLIP
        s["with_hand"][tag] = with_hand
        unsupported = (not on_pins(chair, fr, tag) and lowest(fr, tag) > ON_FLOOR
                       and not held(fr, tag, s["points"], with_hand))
        s["air"][tag] = s["air"][tag] + 1 if unsupported else 0
        s["dropped"][tag] |= s["air"][tag] >= UNSUPPORTED_STEPS
    sample = np.r_[np.asarray(chair.pos, float), objects["footrest_l"].pos, objects["footrest_r"].pos]
    s["recent"].append(sample)
    del s["recent"][:-STILL_CALLS]


def _set_springs(environment) -> None:
    from partkit import set_hinge_target

    for tag, sy in SIDE.items():
        set_hinge_target(environment, f"footrest_{tag}", "lever", sy * LEVER_PRELOAD)


@on_step
def _control(environment) -> None:
    """Every control step: read the parts and the grippers, keep the latch springs' preload."""
    from partkit import _part_state, gripper_points

    objects = {p.name: _part_state(environment, p) for p in (CHAIR, FOOTREST_L, FOOTREST_R)}
    _advance(objects, gripper_points(environment))
    _set_springs(environment)


# -- scoring ---------------------------------------------------------------------------------------------------------------------
def footrest_set_down(objects, tag: str) -> bool:
    """The footrest is off its pins, resting on a mat, was never dropped, and no gripper is within 3 cm of it."""
    chair, fr = objects["chair"], objects[f"footrest_{tag}"]
    return bool(not _STATE["dropped"][tag] and not on_pins(chair, fr, tag) and on_mat(fr, tag)
                and gripper_gap(fr, tag, _STATE["points"]) >= CLEAR)


def _still() -> bool:
    r = _STATE["recent"]
    return len(r) == STILL_CALLS and float(np.ptp(np.array(r), axis=0).max()) < STILL_TOL


SWUNG_CREDIT, OFF_CREDIT, DOWN_CREDIT = 0.1, 0.2, 0.45


def footrest_credit(objects, tag: str) -> float:
    """0.45 set down on a mat; 0.2 off its pins but not set down (still held, or lying off the mats); 0.1 still on its
    pins but swung 45 degrees or more outward (latch released); 0 latched forward, or once it was dropped."""
    chair, fr = objects["chair"], objects[f"footrest_{tag}"]
    if _STATE["dropped"][tag]:
        return 0.0
    if footrest_set_down(objects, tag):
        return DOWN_CREDIT
    if not on_pins(chair, fr, tag):
        return OFF_CREDIT
    return SWUNG_CREDIT if swing_angle(chair, fr, tag) >= SWUNG else 0.0


def _grade(objects) -> tuple[float, bool]:
    s = _STATE
    if s["moved"]:
        return 0.0, False
    if not all(footrest_set_down(objects, t) for t in ("l", "r")):
        return round(sum(footrest_credit(objects, t) for t in ("l", "r")), 3), False
    if not _still():
        return 0.9, False
    return 1.0, True


def success(objects) -> bool:
    """Both footrests are set down on the mats (off their pins, resting on a mat, never dropped, no gripper within
    3 cm), the chair and both footrests have not moved more than 1 mm for 20 control steps (1 s), and the chair never
    left its parked place."""
    return _grade(objects)[1]


def score(objects) -> float:
    """0 if the chair ever left its parked place (rear axle midpoint more than 10 cm from where it stood, turned more
    than 10 degrees, or tipped more than 10 degrees). Otherwise the sum over the two footrests of: 0.45 if it is set
    down (off its pins, lowest point within 3 cm of the floor, hanger tube ends and footplate centre over one mat, no
    gripper within 3 cm, and never dropped); 0.2 if it is off its pins but not set down; 0.1 if it is still on its pins
    but swung 45 degrees or more outward; 0 if it is latched forward, or once it was dropped (off its pins, its lowest
    point more than 3 cm above the floor, and no gripper closed on any part of it with the footrest moving with it,
    for two control steps). With both set down: 0.9
    while anything has moved more than 1 mm in the last second, 1.0 once the chair and both footrests have been still
    for 1 s (20 control steps)."""
    return _grade(objects)[0]


def distance(objects) -> float:
    """Footrests still to set down (of two), as a number: 0 when both are set down."""
    return float(2 - sum(footrest_set_down(objects, t) for t in ("l", "r")))
