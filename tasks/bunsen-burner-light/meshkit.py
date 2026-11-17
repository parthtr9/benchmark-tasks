"""meshkit: procedural triangle-mesh bodies (cups, bottles, extruded profiles, bent tubes, hooks) for yam_arena task files.

Copy this file next to the task file and use it like this:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import meshkit as mk

    GLASS = (0.75, 0.85, 0.9)
    MUG = mk.mesh_part("mug", pos=(0.30, 0.0, TOP + 0.002), mass=0.30, shapes=[
        mk.Shape(mk.cup(radius=0.04, height=0.09, wall=0.004), color=GLASS),
        mk.Shape(mk.tube(mk.fillet_path([(0.038, 0, 0.07), (0.07, 0, 0.07), (0.07, 0, 0.02), (0.038, 0, 0.02)], 0.012),
                         radius=0.005), color=GLASS)])
    RAIL = mk.mesh_part("rail", pos=(0.4, 0.1, TOP), shapes=[mk.Shape(mk.extrude(PROFILE, 0.2), color=(0.6,) * 3)])
    OBJECTS = [MUG, RAIL, ...]          # mass -> loose body; no mass -> fixed

yam_arena's task-file objects are single primitives. A ``MeshPart`` is one body made of ``Shape``s, each a closed
triangle mesh in the body frame with its own look and collider:

    collider   loose body                                         fixed body
    "auto"     "sdf"                                              "exact"
    "sdf"      signed-distance field of the mesh (concave OK)     (same as "exact")
    "exact"    not allowed (PhysX has no dynamic triangle mesh)   the triangles themselves
    "hull"     one convex hull (at most 64 vertices on GPU)       same
    "decomposition"  PhysX convex decomposition (``max_hulls``)   same
    "none"     visual only                                        visual only

``visible=False`` makes a collider-only shape, so a body can show one mesh and collide with hand-made convex pieces
(``revolve_pieces``, ``cup_pieces``, ``hulls``). The body's mass, centre of mass and inertia are computed here from the
shapes' enclosed volumes (weighted by ``Shape.density``) and written explicitly, so they do not depend on the collider.

Mesh builders (numpy only; metres; body frame; all return a closed, outward-facing :class:`Mesh`):
``revolve`` (lathe a closed (r, z) profile), ``cup`` (hollow, straight or tapered), ``shell_profile`` (hollow vessel
from an outer outline), ``extrude`` (polygon with holes), ``tube`` (round wire or pipe along a path), ``fillet_path``
and ``arc`` (paths for it), ``hook``, ``box``. ``Mesh`` has ``moved``, ``rotated``, ``transformed``, ``volume``,
``bounds``, ``is_closed`` and ``is_convex``.

``attach(part, shapes, to="")`` adds mesh shapes to a partkit ``Part`` (its body, or one of its sliders or hinges), so
partkit joints can carry curved shapes.

Importing this module hooks ``yam_arena.scene`` (and the embodiment's step and reset) without modifying yam_arena, the
way partkit does; both kits can be used in one task file. ``success()/score()`` receive a :class:`MeshState` for a
MeshPart: the body origin's pose (not a bounding-box centre), velocities and world bounds.

Pure Python and numpy at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field, replace

import numpy as np

try:
    from yam_arena.task_file import Box, ObjectState
except Exception:  # noqa: BLE001 - offline (no yam_arena): minimal stand-ins so geometry and specs can be tested

    @dataclass(frozen=True)
    class Box:  # type: ignore[no-redef]
        name: str
        pos: tuple
        color: tuple = (0.7, 0.7, 0.7)
        mass: float | None = None
        yaw: float = 0.0
        size: tuple = (0.04, 0.04, 0.04)

        @property
        def free(self) -> bool:
            return self.mass is not None

        def _check(self) -> None:
            pass

    @dataclass(frozen=True)
    class ObjectState:  # type: ignore[no-redef]
        pos: np.ndarray
        quat_xyzw: np.ndarray
        lin_vel: np.ndarray
        ang_vel: np.ndarray
        low: np.ndarray
        high: np.ndarray

        def resting(self, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
            return bool(np.linalg.norm(self.lin_vel) < max_speed and np.linalg.norm(self.ang_vel) < max_spin)


__all__ = ["Mesh", "MeshPart", "MeshState", "Shape", "arc", "attach", "box", "cup", "cup_pieces", "env", "extrude",
           "fillet_path", "hook", "hulls", "install", "mass_properties", "mesh_part", "on_reset", "on_step", "revolve",
           "revolve_pieces", "shell_profile", "triangulate", "tube", "write_pose"]

Color = tuple[float, float, float]
COLLIDERS = ("auto", "sdf", "exact", "hull", "decomposition", "none")
GPU_HULL_VERTICES = 64      # PhysX's GPU convex hulls hold at most this many vertices


# -- meshes ------------------------------------------------------------------------------------------------------

class Mesh:
    """A triangle mesh: ``points`` (N, 3) in metres and ``faces`` (M, 3) vertex indices, counter-clockwise seen from
    outside. Immutable by convention: every method returns a new mesh."""

    __slots__ = ("points", "faces")

    def __init__(self, points, faces):
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if len(p) < 4 or len(f) < 4:
            raise ValueError("a mesh needs at least 4 points and 4 triangles")
        if not np.isfinite(p).all():
            raise ValueError("mesh points must be finite")
        if f.min() < 0 or f.max() >= len(p):
            raise ValueError("mesh face index outside its points")
        self.points, self.faces = p, f

    # measures
    def _tetra(self):
        t = self.points[self.faces]
        return t, np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))

    def volume(self) -> float:
        """Enclosed volume (m^3); negative if the faces point inwards."""
        return float(self._tetra()[1].sum() / 6.0)

    def centroid(self) -> np.ndarray:
        """Centre of the enclosed volume."""
        t, det = self._tetra()
        return (t.sum(axis=1) * det[:, None]).sum(axis=0) / (4.0 * det.sum())

    def second_moment(self) -> np.ndarray:
        """Integral of x x^T over the enclosed volume, about the mesh's own origin (unit density)."""
        t, det = self._tetra()
        canon = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]) / 120.0
        # each tetrahedron (origin, a, b, c): det * A canon A^T with A's columns a, b, c
        return np.einsum("n,nki,kl,nlj->ij", det, t, canon, t)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    def is_closed(self) -> bool:
        """Watertight and consistently wound: every directed edge is used once and so is its reverse."""
        f = self.faces
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        if (e[:, 0] == e[:, 1]).any():
            return False
        n = len(self.points)
        key = e[:, 0] * n + e[:, 1]
        rev = e[:, 1] * n + e[:, 0]
        uniq, counts = np.unique(key, return_counts=True)
        return bool((counts == 1).all() and np.array_equal(uniq, np.unique(rev)))

    def is_convex(self, tol: float = 1e-7) -> bool:
        """No vertex lies outside any face's plane (by more than ``tol`` m)."""
        t = self.points[self.faces]
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        length = np.linalg.norm(n, axis=1)
        keep = length > 1e-14
        n, origin = n[keep] / length[keep, None], t[keep, 0]
        used = self.points[np.unique(self.faces)]
        return bool(((used @ n.T) - np.einsum("ij,ij->i", n, origin)[None, :]).max() <= tol)

    # edits
    def moved(self, offset) -> Mesh:
        return Mesh(self.points + np.asarray(offset, float), self.faces)

    def transformed(self, rotation=None, offset=(0.0, 0.0, 0.0)) -> Mesh:
        """Rotate by a 3x3 matrix (about the origin), then translate."""
        r = np.eye(3) if rotation is None else np.asarray(rotation, float).reshape(3, 3)
        faces = self.faces if np.linalg.det(r) > 0 else self.faces[:, ::-1]
        return Mesh(self.points @ r.T + np.asarray(offset, float), faces)

    def rotated(self, axis: str, degrees: float) -> Mesh:
        """Turn about the body's "X", "Y" or "Z" axis through the origin (counter-clockwise, degrees)."""
        return self.transformed(_axis_rotation(axis, math.radians(degrees)))

    def scaled(self, factor) -> Mesh:
        s = np.broadcast_to(np.asarray(factor, float), (3,))
        faces = self.faces if np.prod(s) > 0 else self.faces[:, ::-1]
        return Mesh(self.points * s, faces)

    def flipped(self) -> Mesh:
        return Mesh(self.points, self.faces[:, ::-1])

    def __add__(self, other: Mesh) -> Mesh:
        """Both meshes in one (their volumes add; overlapping parts are counted twice)."""
        return Mesh(np.vstack([self.points, other.points]), np.vstack([self.faces, other.faces + len(self.points)]))

    def __repr__(self) -> str:
        return f"Mesh({len(self.points)} points, {len(self.faces)} triangles)"


def _axis_rotation(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if axis.upper() == "X":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis.upper() == "Y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis.upper() == "Z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"axis must be X, Y or Z, got {axis!r}")


def _area2(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _cross2(o, a, b) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _segments_cross(p, q, a, b) -> bool:
    """Segments pq and ab cross at a point interior to both (shared end points do not count)."""
    d1, d2, d3, d4 = _cross2(p, q, a), _cross2(p, q, b), _cross2(a, b, p), _cross2(a, b, q)
    return (d1 * d2 < -1e-18) and (d3 * d4 < -1e-18)


def triangulate(polygon, holes=()) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a simple polygon (2-D points in order, either winding, concave allowed) with optional holes.
    Returns ``(points, triangles)``: the outer ring (made counter-clockwise) followed by the holes (made clockwise),
    and counter-clockwise index triples into them."""
    rings = [np.asarray(polygon, float).reshape(-1, 2)] + [np.asarray(h, float).reshape(-1, 2) for h in holes]
    for i, ring in enumerate(rings):
        if len(ring) < 3:
            raise ValueError("a polygon needs at least 3 points")
        if (_area2(ring) < 0) == (i == 0):
            rings[i] = ring[::-1].copy()
    if abs(_area2(rings[0])) < 1e-14:
        raise ValueError("the polygon has no area")
    points = np.vstack(rings)
    starts = np.cumsum([0] + [len(r) for r in rings])
    loop = list(range(starts[0], starts[1]))
    edges = [(starts[i] + k, starts[i] + (k + 1) % len(r)) for i, r in enumerate(rings) for k in range(len(r))]
    # Join each hole to the outline by a two-way bridge, the hole reaching furthest in +x first.
    order = sorted(range(1, len(rings)), key=lambda i: -rings[i][:, 0].max())
    for i in order:
        ids = list(range(starts[i], starts[i + 1]))
        m = max(ids, key=lambda j: points[j, 0])
        best = None
        for pos, v in enumerate(loop):
            if any(_segments_cross(points[m], points[v], points[a], points[b]) for a, b in edges):
                continue
            d = float(np.sum((points[m] - points[v]) ** 2))
            if best is None or d < best[0]:
                best = (d, pos)
        if best is None:
            raise ValueError("a hole cannot be joined to the outline (is it inside the polygon?)")
        pos, k = best[1], ids.index(m)
        edges.append((m, loop[pos]))
        loop = loop[:pos + 1] + ids[k:] + ids[:k + 1] + loop[pos:]
    tris = []
    idx = loop
    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = points[i0], points[i1], points[i2]
            if _cross2(a, b, c) <= 1e-16:
                continue
            blocked = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = points[j]
                if (np.array_equal(p, a) or np.array_equal(p, b) or np.array_equal(p, c)):
                    continue
                if _cross2(a, b, p) > 1e-16 and _cross2(b, c, p) > 1e-16 and _cross2(c, a, p) >= -1e-16:
                    blocked = True
                    break
            if not blocked:
                tris.append((i0, i1, i2))
                idx = idx[:k] + idx[k + 1:]
                break
        else:
            raise ValueError("cannot triangulate: the polygon crosses itself or repeats points")
    if _cross2(points[idx[0]], points[idx[1]], points[idx[2]]) > 1e-16:
        tris.append(tuple(idx))
    return points, np.array(tris, dtype=np.int64)


def revolve(profile, segments: int = 48, angle: float = 360.0, start: float = 0.0) -> Mesh:
    """Lathe a closed profile about the z axis. ``profile`` is a simple polygon of (r, z) points with r >= 0 (either
    winding; points with r = 0 lie on the axis). ``segments`` flat facets per turn. With ``angle`` < 360 (degrees,
    from ``start``) the result is a sector closed by two flat faces."""
    prof = np.asarray(profile, float).reshape(-1, 2)
    if len(prof) < 3 or segments < 1:
        raise ValueError("revolve needs a profile of at least 3 points and at least 1 segment")
    if prof[:, 0].min() < -1e-12:
        raise ValueError("revolve: every profile point needs r >= 0")
    if _area2(prof) < 0:
        prof = prof[::-1].copy()
    full = angle >= 360.0 - 1e-9
    if full and segments < 3:
        raise ValueError("a full turn needs at least 3 segments")
    cols = segments if full else segments + 1
    angles = math.radians(start) + math.radians(min(angle, 360.0)) * np.arange(cols) / segments
    pts: list = []
    index: list[list[int]] = []
    for r, z in prof:
        if r < 1e-9:
            index.append([len(pts)] * cols)
            pts.append((0.0, 0.0, z))
        else:
            index.append(list(range(len(pts), len(pts) + cols)))
            pts.extend((r * math.cos(a), r * math.sin(a), z) for a in angles)
    faces: list = []

    def tri(a, b, c):
        if a != b and b != c and a != c:
            faces.append((a, b, c))

    n = len(prof)
    for k in range(n):
        k1 = (k + 1) % n
        for s in range(segments):
            s1 = (s + 1) % cols
            a, b, c, d = index[k][s], index[k1][s], index[k1][s1], index[k][s1]
            tri(a, c, b)
            tri(a, d, c)
    if not full:
        _, cap = triangulate(prof)
        for i, j, k in cap:
            tri(index[i][0], index[j][0], index[k][0])
            tri(index[k][-1], index[j][-1], index[i][-1])
    return Mesh(pts, faces)


def shell_profile(outline, wall: float, floor: float | None = None) -> list[tuple[float, float]]:
    """The closed (r, z) profile of a hollow vessel, for :func:`revolve`. ``outline`` is its outer surface from the
    bottom centre ``(0, z0)`` up to the rim; ``wall`` is the wall thickness (measured horizontally) and ``floor`` the
    bottom thickness (default: ``wall``)."""
    out = [(float(r), float(z)) for r, z in outline]
    floor = wall if floor is None else floor
    if len(out) < 2 or abs(out[0][0]) > 1e-9:
        raise ValueError("shell_profile: the outline starts on the axis, at (0, z of the bottom)")
    z_in = out[0][1] + floor
    if not z_in < out[-1][1]:
        raise ValueError("shell_profile: the floor is thicker than the vessel is tall")
    if any(out[i][1] < out[i - 1][1] for i in range(1, len(out))):
        raise ValueError("shell_profile: the outline must not come back down (z never decreases)")
    inner = []
    for i in range(len(out) - 1, 0, -1):
        (r1, z1), (r0, z0) = out[i], out[i - 1]
        if z1 > z_in:
            if r1 - wall <= 0:
                raise ValueError("shell_profile: the wall is thicker than the radius")
            inner.append((r1 - wall, z1))
        if z0 <= z_in < z1:
            r = r0 + (z_in - z0) / (z1 - z0) * (r1 - r0) - wall
            if r > 1e-9:
                inner.append((r, z_in))
            break
    inner.append((0.0, z_in))
    return out + inner


def cup(radius: float, height: float, wall: float, floor: float | None = None, top_radius: float | None = None,
        segments: int = 48) -> Mesh:
    """A hollow open-topped vessel standing on z = 0: outer ``radius`` at the bottom, ``top_radius`` at the rim
    (default: the same; larger makes a cone or funnel shape), ``wall`` and ``floor`` thick."""
    top = radius if top_radius is None else top_radius
    return revolve(shell_profile([(0.0, 0.0), (radius, 0.0), (top, height)], wall, floor), segments)


def revolve_pieces(profiles, segments: int = 16) -> list[Mesh]:
    """Convex pieces whose union is the solid of revolution of the given convex (r, z) polygons, each revolved with
    ``segments`` facets: a polygon that touches the axis becomes one piece, any other becomes one wedge per facet.
    For hand-made colliders (see :func:`hulls`); raises if a piece would not be convex."""
    out = []
    for profile in profiles:
        prof = np.asarray(profile, float).reshape(-1, 2)
        if prof[:, 0].min() < 1e-9:
            pieces = [revolve(prof, segments)]
        else:
            step = 360.0 / segments
            pieces = [revolve(prof, 1, angle=step, start=i * step) for i in range(segments)]
        for piece in pieces:
            if not piece.is_convex(1e-6):
                raise ValueError("revolve_pieces: a profile is not convex (or bulges past its edge on the axis)")
            if len(np.unique(piece.faces)) > GPU_HULL_VERTICES:
                raise ValueError(f"revolve_pieces: a piece has over {GPU_HULL_VERTICES} vertices; use fewer segments")
        out += pieces
    return out


def cup_pieces(radius: float, height: float, wall: float, floor: float | None = None,
               top_radius: float | None = None, segments: int = 16) -> list[Mesh]:
    """Convex pieces for :func:`cup` with the same arguments: the floor disc and one wall stave per facet."""
    top = radius if top_radius is None else top_radius
    floor = wall if floor is None else floor
    at_floor = radius + (top - radius) * floor / height
    return revolve_pieces([[(0.0, 0.0), (radius, 0.0), (at_floor, floor), (0.0, floor)],
                           [(at_floor - wall, floor), (at_floor, floor), (top, height), (top - wall, height)]],
                          segments)


def extrude(polygon, height: float, holes=(), z0: float = 0.0) -> Mesh:
    """A prism: the (x, y) ``polygon`` (simple, either winding, concave allowed, optional ``holes``) from ``z0`` to
    ``z0 + height``. Turn it with ``.rotated()`` for other directions."""
    if not height > 0:
        raise ValueError("extrude: height must be positive")
    pts2, tris = triangulate(polygon, holes)
    n = len(pts2)
    pts = np.vstack([np.c_[pts2, np.full(n, z0)], np.c_[pts2, np.full(n, z0 + height)]])
    faces = [(k, j, i) for i, j, k in tris] + [(i + n, j + n, k + n) for i, j, k in tris]
    start = 0
    for ring in [polygon, *holes]:
        m = len(ring)
        for k in range(m):
            i, j = start + k, start + (k + 1) % m
            faces += [(i, j, j + n), (i, j + n, i + n)]
        start += m
    return Mesh(pts, faces)


def box(size, centre=(0.0, 0.0, 0.0)) -> Mesh:
    """A box mesh (for a mesh body that also needs a plain block; partkit's boxes are cheaper colliders)."""
    sx, sy, sz = (float(v) / 2 for v in size)
    return extrude([(-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)], 2 * sz, z0=-sz).moved(centre)


def arc(centre, radius: float, start: float, end: float, n: int = 12, plane: str = "XZ") -> list[tuple]:
    """``n + 1`` points on a circular arc from angle ``start`` to ``end`` (degrees) in the given plane ("XY", "XZ" or
    "YZ"; the angle is measured from the plane's first axis towards its second)."""
    i, j = ("XYZ".index(c) for c in plane.upper())
    out = []
    for k in range(n + 1):
        a = math.radians(start + (end - start) * k / n)
        p = [float(v) for v in centre]
        p[i] += radius * math.cos(a)
        p[j] += radius * math.sin(a)
        out.append(tuple(p))
    return out


def fillet_path(points, radius: float, n: int = 8) -> list[tuple]:
    """Round every corner of a 3-D polyline with an arc of ``radius`` (``n`` segments each): the path of a bent rod."""
    p = [np.asarray(q, float) for q in points]
    out = [p[0]]
    for i in range(1, len(p) - 1):
        u, v = p[i - 1] - p[i], p[i + 1] - p[i]
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        u, v = u / lu, v / lv
        half = math.acos(max(-1.0, min(1.0, float(u @ v)))) / 2
        if half > math.pi / 2 - 1e-6 or half < 1e-6:
            out.append(p[i])
            continue
        d = radius / math.tan(half)
        if d > min(lu, lv) / 2 + 1e-12:
            raise ValueError(f"fillet_path: radius {radius} does not fit the corner at point {i}")
        bis = (u + v) / np.linalg.norm(u + v)
        c = p[i] + bis * radius / math.sin(half)
        a, b = p[i] + u * d - c, p[i] + v * d - c
        sweep = math.acos(max(-1.0, min(1.0, float(a @ b) / radius ** 2)))
        for k in range(n + 1):
            t = k / n
            w = (math.sin((1 - t) * sweep) * a + math.sin(t * sweep) * b) / math.sin(sweep)
            out.append(c + w)
    out.append(p[-1])
    return [tuple(float(x) for x in q) for q in out]


def tube(path, radius, sides: int = 12, closed: bool = False) -> Mesh:
    """A round rod along the 3-D ``path`` (points close enough to follow its bends: see :func:`fillet_path`,
    :func:`arc`), flat ends, or a ring with ``closed``. ``radius`` is one number or one per path point (a taper)."""
    p = np.asarray(path, float).reshape(-1, 3)
    m = len(p)
    if m < (3 if closed else 2) or sides < 3:
        raise ValueError("tube needs at least 2 path points (3 for a closed one) and 3 sides")
    radii = np.broadcast_to(np.asarray(radius, float), (m,))
    if radii.min() <= 0:
        raise ValueError("tube: radius must be positive")
    if closed:
        tangent = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    else:
        tangent = np.gradient(p, axis=0)
    norms = np.linalg.norm(tangent, axis=1)
    if norms.min() < 1e-12:
        raise ValueError("tube: the path repeats a point or doubles back on itself")
    tangent = tangent / norms[:, None]
    normal = np.cross(tangent[0], [0.0, 0.0, 1.0] if abs(tangent[0][2]) < 0.9 else [1.0, 0.0, 0.0])
    theta = 2 * math.pi * np.arange(sides) / sides
    pts = []
    for i in range(m):
        normal = normal - (normal @ tangent[i]) * tangent[i]
        normal = normal / np.linalg.norm(normal)
        binormal = np.cross(tangent[i], normal)
        pts.extend(p[i] + radii[i] * (math.cos(t) * normal + math.sin(t) * binormal) for t in theta)
    faces = []
    for i in range(m if closed else m - 1):
        i1 = (i + 1) % m
        for s in range(sides):
            s1 = (s + 1) % sides
            a, b, c, d = i * sides + s, i * sides + s1, i1 * sides + s1, i1 * sides + s
            faces += [(a, b, c), (a, c, d)]
    if not closed:
        first, last = len(pts), len(pts) + 1
        pts += [p[0], p[-1]]
        for s in range(sides):
            s1 = (s + 1) % sides
            faces += [(first, s1, s), (last, (m - 1) * sides + s, (m - 1) * sides + s1)]
    return Mesh(pts, faces)


def hook(shank: float = 0.06, gap: float = 0.026, tail: float = 0.012, wire: float = 0.003, toe: float = 0.0,
         sides: int = 10) -> Mesh:
    """A wire J- or S-hook in the body's x-z plane: a straight ``shank`` down from the origin, a half-circle bend
    over the top whose inside is ``gap`` wide, a ``tail`` coming back down on the +x side, and with ``toe`` > 0 a
    second, upturned half-circle of that inner width at the bottom (an S-hook). ``wire`` is the rod's radius. The top
    bend's centre is at ``(gap / 2 + wire, 0, 0)``: hang it there."""
    r = gap / 2 + wire
    path = []
    if toe > 0:
        rt = toe / 2 + wire
        path += arc((-rt, 0.0, -shank), rt, 180.0, 360.0, 10)[:-1]
    path.append((0.0, 0.0, -shank))
    path += arc((r, 0.0, 0.0), r, 180.0, 0.0, 12)
    path.append((2 * r, 0.0, -tail))
    return tube(path, wire, sides)


# -- bodies ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Shape:
    """One mesh of a body, in the body frame, with its look and its collider (see the module docstring)."""

    mesh: Mesh
    color: Color = (0.7, 0.7, 0.7)
    collider: str = "auto"
    visible: bool = True
    density: float = 1.0        # relative weight of this shape's volume in the body's mass; 0: carries no mass
    name: str = ""
    sdf_resolution: int = 0     # voxels along the longest side; 0: about 1 mm voxels, between 64 and 256
    max_hulls: int = 32         # "decomposition": the most convex pieces
    roughness: float = 0.45
    metallic: float = 0.0
    opacity: float = 1.0
    smooth_deg: float = 40.0    # edges sharper than this stay sharp in the render; 0: flat-shaded facets

    def __post_init__(self):
        if self.collider not in COLLIDERS:
            raise ValueError(f"collider must be one of {COLLIDERS}, got {self.collider!r}")
        if self.density < 0:
            raise ValueError("Shape.density must be >= 0")

    def resolved(self, loose: bool) -> str:
        """The collider this shape gets on a loose or a fixed body."""
        if self.collider == "auto":
            return "sdf" if loose else "exact"
        if self.collider == "sdf" and not loose:
            return "exact"
        return self.collider

    def resolution(self) -> int:
        if self.sdf_resolution:
            return int(self.sdf_resolution)
        lo, hi = self.mesh.bounds()
        return int(min(256, max(64, math.ceil(float((hi - lo).max()) / 0.001))))


def hulls(meshes, color: Color = (0.5, 0.5, 0.5)) -> list[Shape]:
    """Invisible, massless convex-hull colliders from convex meshes (``revolve_pieces``, ``cup_pieces``, your own).
    Put them in a body next to the visible mesh, which then takes ``collider="none"``."""
    return [Shape(m, color=color, collider="hull", visible=False, density=0.0) for m in meshes]


def mass_properties(shapes, mass: float) -> dict:
    """Mass distribution of a body of ``mass`` kg whose material fills the shapes' enclosed volumes (weighted by
    ``Shape.density``): ``com`` (body frame), ``inertia`` (3x3 about the centre of mass, body axes), ``principal``
    (its three principal moments) and ``axes_wxyz`` (the principal axes' rotation in the body frame)."""
    total, first, second = 0.0, np.zeros(3), np.zeros((3, 3))
    for s in shapes:
        if s.density <= 0:
            continue
        v = s.mesh.volume()
        if v <= 0 or not s.mesh.is_closed():
            raise ValueError(f"shape {s.name or '?'}: a mesh that carries mass must be closed and face outwards")
        total += s.density * v
        first += s.density * v * s.mesh.centroid()
        second += s.density * s.mesh.second_moment()
    if total <= 0:
        raise ValueError("no shape carries mass (every density is 0)")
    k = mass / total
    com = first / total
    second = second * k - mass * np.outer(com, com)
    inertia = np.trace(second) * np.eye(3) - second
    moments, vectors = np.linalg.eigh((inertia + inertia.T) / 2)
    if np.linalg.det(vectors) < 0:
        vectors[:, 2] = -vectors[:, 2]
    return {"com": com, "inertia": inertia, "principal": moments, "axes_wxyz": _matrix_quat_wxyz(vectors),
            "volume": sum(s.mesh.volume() for s in shapes if s.density > 0)}


def _matrix_quat_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
    t = float(np.trace(r))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(max(1e-18, 1.0 + r[i, i] - r[j, j] - r[k, k])) * 2
        v = [0.0, 0.0, 0.0]
        v[i], v[j], v[k] = 0.25 * s, (r[j, i] + r[i, j]) / s, (r[k, i] + r[i, k]) / s
        q = ((r[k, j] - r[j, k]) / s, *v)
    n = math.sqrt(sum(c * c for c in q))
    return tuple(float(c / n) for c in q)


def _quat_matrix_xyzw(q) -> np.ndarray:
    x, y, z, w = (float(c) for c in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@dataclass(frozen=True)
class MeshPart(Box):
    """One body made of mesh ``shapes``: loose with a ``mass``, fixed without. It subclasses Box only so yam_arena's
    task-file checks accept it; ``pos`` is the body origin (not a bounding-box centre) and ``size`` the shapes'
    bounding box. ``quat_xyzw`` replaces ``yaw`` when the body does not stand upright. Build it with
    :func:`mesh_part`."""

    shapes: tuple = ()
    quat_xyzw: tuple | None = None
    friction: float = 1.0
    contact_offset: float = 0.002
    angular_damping: float = 0.5
    linear_damping: float = 0.0
    solver_iterations: int = 16
    low: tuple = (0.0, 0.0, 0.0)       # the shapes' bounds in the body frame
    high: tuple = (0.0, 0.0, 0.0)
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")
        if not self.shapes:
            raise ValueError(f"{self.name}: a MeshPart needs at least one Shape")
        if self.mass is not None and not self.mass > 0:
            raise ValueError(f"{self.name}.mass must be positive or None")
        loose = self.mass is not None
        kinds = [s.resolved(loose) for s in self.shapes]
        if loose and "exact" in kinds:
            raise ValueError(f"{self.name}: a loose body cannot use the 'exact' collider; use sdf, hull or "
                             "decomposition")
        if all(k == "none" for k in kinds):
            raise ValueError(f"{self.name}: no shape has a collider")
        for s, k in zip(self.shapes, kinds):
            if k in ("sdf", "exact") and not s.mesh.is_closed():
                raise ValueError(f"{self.name}: shape {s.name or '?'} is not a closed mesh")
        if loose:
            mass_properties(self.shapes, self.mass)

    @property
    def free(self) -> bool:
        return self.mass is not None

    @property
    def rotation_xyzw(self) -> tuple[float, float, float, float]:
        if self.quat_xyzw is not None:
            return tuple(float(c) for c in self.quat_xyzw)
        return (0.0, 0.0, math.sin(self.yaw / 2), math.cos(self.yaw / 2))


def mesh_part(name: str, pos, shapes, mass: float | None = None, yaw: float = 0.0, quat_xyzw=None,
              friction: float = 1.0, **options) -> MeshPart:
    """Build a MeshPart at ``pos`` (its origin, world frame): loose with a ``mass`` (kg), fixed without."""
    shapes = tuple(Shape(s) if isinstance(s, Mesh) else s for s in shapes)
    shapes = tuple(s if s.name else replace(s, name=f"mesh_{i}") for i, s in enumerate(shapes))
    lo = np.min([s.mesh.bounds()[0] for s in shapes], axis=0)
    hi = np.max([s.mesh.bounds()[1] for s in shapes], axis=0)
    spec = MeshPart(name=name, pos=tuple(float(v) for v in pos), size=tuple(float(v) for v in (hi - lo)), mass=mass,
                    color=tuple(shapes[0].color), yaw=float(yaw),
                    quat_xyzw=None if quat_xyzw is None else tuple(float(v) for v in quat_xyzw), shapes=shapes,
                    friction=friction, low=tuple(lo.tolist()), high=tuple(hi.tolist()), **options)
    spec._check()
    return spec


@dataclass(frozen=True)
class MeshState(ObjectState):
    """The body origin's pose (``pos``, ``quat_xyzw``), velocities and the world-aligned bounds of its shapes."""

    def resting(self, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
        return bool(np.linalg.norm(self.lin_vel) < max_speed and np.linalg.norm(self.ang_vel) < max_spin)

    def rotation(self) -> np.ndarray:
        return _quat_matrix_xyzw(self.quat_xyzw)

    def to_local(self, world_point) -> np.ndarray:
        return self.rotation().T @ (np.asarray(world_point, float) - self.pos)

    def to_world(self, local_point) -> np.ndarray:
        return self.pos + self.rotation() @ np.asarray(local_point, float)


def state_at(spec: MeshPart, pos=None, quat_xyzw=None, lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)) -> MeshState:
    """The MeshState of ``spec`` at a pose (default: where the task file puts it). Also for offline scoring tests."""
    pos = np.asarray(spec.pos if pos is None else pos, float)
    quat = np.asarray(spec.rotation_xyzw if quat_xyzw is None else quat_xyzw, float)
    lo, hi = np.array(spec.low), np.array(spec.high)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    world = corners @ _quat_matrix_xyzw(quat).T + pos
    return MeshState(pos=pos, quat_xyzw=quat, lin_vel=np.asarray(lin_vel, float), ang_vel=np.asarray(ang_vel, float),
                     low=world.min(axis=0), high=world.max(axis=0))


def corner_normals(mesh: Mesh, smooth_deg: float = 40.0) -> np.ndarray:
    """One normal per face corner (3 M, 3): faces meeting at less than ``smooth_deg`` shade as one curved surface,
    sharper edges stay sharp."""
    t = mesh.points[mesh.faces]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = np.linalg.norm(fn, axis=1)
    unit = fn / np.maximum(area, 1e-30)[:, None]
    out = np.repeat(unit, 3, axis=0)
    if smooth_deg <= 0:
        return out
    limit = math.cos(math.radians(smooth_deg))
    # Weight each face by its angle at the vertex, so how a flat quad is split into triangles does not tilt the normal.
    angle = np.zeros((len(t), 3))
    for c in range(3):
        u, v = t[:, (c + 1) % 3] - t[:, c], t[:, (c + 2) % 3] - t[:, c]
        cos = np.einsum("ij,ij->i", u, v) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-30)
        angle[:, c] = np.arccos(np.clip(cos, -1.0, 1.0))
    around: dict[int, list[tuple[int, int]]] = {}
    for f, tri in enumerate(mesh.faces):
        for c, v in enumerate(tri):
            around.setdefault(int(v), []).append((f, c))
    for f, tri in enumerate(mesh.faces):
        for c, v in enumerate(tri):
            n = np.zeros(3)
            for g, gc in around[int(v)]:
                if unit[g] @ unit[f] >= limit:
                    n += angle[g, gc] * unit[g]
            length = np.linalg.norm(n)
            if length > 1e-30:
                out[3 * f + c] = n / length
    return out


# -- spawning (Isaac Sim, imported lazily) -------------------------------------------------------------------------

_SPECS: dict[str, MeshPart] = {}
_ATTACH: dict[str, list[tuple[str, tuple]]] = {}


def _author_shapes(stage, body_path: str, shapes, loose: bool, looks_path: str, physics_material: str,
                   contact_offset: float, offset=(0.0, 0.0, 0.0), prefix: str = "") -> None:
    """Author each shape as a UsdGeom.Mesh under ``body_path`` with its collider and look (pxr only)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt

    from yam_arena.station import linear_rgb

    looks: dict[tuple, str] = {}

    def look(s: Shape) -> str:
        key = (tuple(s.color), s.roughness, s.metallic, s.opacity)
        if key not in looks:
            path = f"{looks_path}/{prefix}mesh_look_{len(looks)}"
            extra = {"opacity": s.opacity} if s.opacity < 1.0 else {}
            vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(tuple(s.color)), roughness=s.roughness,
                                              metallic=s.metallic, **extra)
            vis.func(path, vis)
            looks[key] = path
        return looks[key]

    shift = np.asarray(offset, float)
    for s in shapes:
        kind = s.resolved(loose)
        path = f"{body_path}/{prefix}{s.name}"
        geom = UsdGeom.Mesh.Define(stage, path)
        pts = s.mesh.points - shift
        geom.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*(float(c) for c in p)) for p in pts]))
        geom.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(s.mesh.faces)))
        geom.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in s.mesh.faces.reshape(-1)]))
        geom.CreateSubdivisionSchemeAttr("none")
        geom.CreateDoubleSidedAttr(False)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        geom.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())]))
        if s.visible:
            normals = corner_normals(s.mesh, s.smooth_deg)
            geom.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*(float(c) for c in n)) for n in normals]))
            geom.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            bind_visual_material(path, look(s), stage=stage)
        else:
            UsdGeom.Imageable(geom.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)   # not rendered
        if kind == "none":
            continue
        prim = geom.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        approximation = {"exact": "none", "hull": "convexHull", "decomposition": "convexDecomposition",
                         "sdf": "sdf"}[kind]
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
        if kind == "sdf":
            PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim).CreateSdfResolutionAttr(s.resolution())
        elif kind == "hull":
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim).CreateHullVertexLimitAttr(GPU_HULL_VERTICES)
        elif kind == "decomposition":
            api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            api.CreateMaxConvexHullsAttr(int(s.max_hulls))
            api.CreateHullVertexLimitAttr(GPU_HULL_VERTICES)
            api.CreateShrinkWrapAttr(True)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(contact_offset)
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(path, physics_material, stage=stage)


def _spawn_mesh_part(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the part under ``prim_path``: an Xform at the pose holding one ``body`` Xform with the meshes."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    spec: MeshPart = _SPECS[cfg.part_name]
    loose = spec.mass is not None
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    phys = f"{prim_path}/Looks/physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    pm.func(phys, pm)

    main = f"{prim_path}/body"
    xform = UsdGeom.Xform.Define(stage, main)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    if loose:
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        props = mass_properties(spec.shapes, spec.mass)
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(float(spec.mass))
        api.CreateCenterOfMassAttr(Gf.Vec3f(*(float(c) for c in props["com"])))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(c) for c in props["principal"])))
        api.CreatePrincipalAxesAttr(Gf.Quatf(*props["axes_wxyz"]))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(int(spec.solver_iterations))
        rb.CreateAngularDampingAttr(float(spec.angular_damping))
        rb.CreateLinearDampingAttr(float(spec.linear_damping))
    _author_shapes(stage, main, spec.shapes, loose, f"{prim_path}/Looks", phys, spec.contact_offset)
    return root.GetPrim()


def _arena_object(spec: MeshPart):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class MeshPartSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_mesh_part)
        part_name: str = ""

    return Object(
        name=spec.name,
        object_type=ObjectType.RIGID if spec.mass is not None else ObjectType.BASE,
        spawner_cfg=MeshPartSpawnerCfg(part_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=spec.rotation_xyzw),
    )


def _t(x):
    return getattr(x, "torch", x)


def _np(x) -> np.ndarray:
    return _t(x).detach().cpu().numpy().astype(np.float64)


def _mesh_state(environment, spec: MeshPart) -> MeshState:
    if spec.mass is None:
        return state_at(spec)
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    return state_at(spec, pose[:3] - origin, pose[3:7], vel[:3], vel[3:6])


def write_pose(environment, name: str, pos, quat_xyzw, lin_vel=(0, 0, 0), ang_vel=(0, 0, 0)) -> None:
    """Teleport a loose part (env 0) to a world pose (relative to the env origin) with the given velocity."""
    import torch

    base = environment.unwrapped
    asset = base.scene[name]
    origin = _t(base.scene.env_origins[0])
    pose = torch.tensor([*pos, *quat_xyzw], dtype=origin.dtype, device=origin.device).unsqueeze(0)
    pose[:, :3] += origin
    asset.write_root_pose_to_sim_index(root_pose=pose)
    v = torch.tensor([*lin_vel, *ang_vel], dtype=origin.dtype, device=origin.device).unsqueeze(0)
    asset.write_root_velocity_to_sim_index(root_velocity=v)


# -- partkit parts with mesh shapes ----------------------------------------------------------------------------------

def attach(part, shapes, to: str = "") -> None:
    """Add mesh ``shapes`` (part frame, at rest) to a partkit ``Part``: to its body, or with ``to`` to the slider or
    hinge of that name, which then carries them. Call it in the task file after building the part. The part keeps
    partkit's mass; PhysX derives its inertia from all its colliders. Use "hull", "decomposition" or "sdf" colliders
    (a partkit part is always loose). ``Part.size`` (its reported bounds) does not grow to include the meshes."""
    shapes = tuple(Shape(s) if isinstance(s, Mesh) else s for s in shapes)
    for s in shapes:
        if s.resolved(True) == "exact":
            raise ValueError("attach: a partkit part is a loose body; use sdf, hull or decomposition")
    names = {x.name for x in (*getattr(part, "sliders", ()), *getattr(part, "hinges", ()))}
    if to and to not in names:
        raise ValueError(f"attach: {part.name} has no slider or hinge named {to!r}")
    _ATTACH.setdefault(part.name, []).append((to, shapes))
    _wrap_partkit()


def _wrap_partkit() -> None:
    """Make every loaded partkit author the attached meshes after it has authored a part."""
    for module in list(sys.modules.values()):
        # Read the module's own names only: some modules (torch.classes) answer any getattr with a proxy.
        names = getattr(module, "__dict__", None)
        if not isinstance(names, dict):
            continue
        spawn = names.get("_spawn_part")
        if (not callable(spawn) or not isinstance(names.get("Part"), type) or not isinstance(names.get("_SPECS"), dict)
                or getattr(spawn, "_meshkit", False)):
            continue

        def wrapped(prim_path, cfg, *args, _spawn=spawn, _module=module, **kwargs):
            from isaaclab.sim.utils.stage import get_current_stage

            prim = _spawn(prim_path, cfg, *args, **kwargs)
            spec = _module._SPECS[cfg.part_name]
            count = 0
            for to, shapes in _ATTACH.get(cfg.part_name, ()):
                sub = next((x for x in (*getattr(spec, "sliders", ()), *getattr(spec, "hinges", ())) if x.name == to),
                           None)
                offset = (0.0, 0.0, 0.0) if sub is None else getattr(sub, "rest", None) or getattr(sub, "pivot")
                named = tuple(s if s.name else replace(s, name=f"mesh_{i}") for i, s in enumerate(shapes))
                _author_shapes(get_current_stage(), f"{prim_path}/{to or 'body'}", named, True, f"{prim_path}/Looks",
                               f"{prim_path}/Looks/physics", 0.001, offset=offset, prefix=f"mk{count}_")
                count += 1
            return prim

        wrapped._meshkit = True
        module._spawn_part = wrapped


# -- hooking yam_arena -------------------------------------------------------------------------------------------

_STEP: list = []
_RESET: list = []
_ENV: dict = {"env": None}


def on_step(fn):
    """Register ``fn(env)`` to run after every control step, before the task file is scored."""
    _STEP.append(fn)
    return fn


def on_reset(fn):
    """Register ``fn(env)`` to run after every reset, once objects have settled."""
    _RESET.append(fn)
    return fn


def env():
    return _ENV["env"]


def _patch_scene(scene) -> None:
    if getattr(scene, "_meshkit_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, MeshPart):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        if _ATTACH:
            _wrap_partkit()
        return orig_task_object(spec)

    def object_states(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(environment, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = _mesh_state(environment, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._meshkit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_meshkit_patched", False):
        return
    base_task_step, base_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        for fn in _STEP:
            fn(self._env)
        return base_task_step(self, *a, **k)

    def reset(self, *a, **k):
        obs = base_reset(self, *a, **k)
        _ENV["env"] = self._env
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._meshkit_patched = True


class _Finder(importlib.abc.MetaPathFinder):
    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader, patch = spec.loader, self.targets[fullname]
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch(module)

        loader.exec_module = exec_module
        return spec


def install() -> None:
    """Hook yam_arena (now for modules already imported, otherwise when they are). Idempotent."""
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(type(f).__name__ == "_Finder" and type(f).__module__ == __name__ for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
