"""Inter-arm clearance from the YAM URDF collision meshes (every link incl. link4/link5 and the fingers), NumPy only.

    min_clearance(qa, ga, base_a, qb, gb, base_b) -> (distance_m, (link_a, link_b))

Each collision mesh is sampled by its vertices, edge midpoints and triangle centres (de-duplicated to a 6 mm grid), so
the distance is the sampled-surface distance (a few mm above the true one). ``held_boxes`` lets a caller add the part
each hand holds as extra sampled boxes. Used by the demo's trace (commanded joints, no simulator state) and offline.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _meshes():
    from yam_arena import YAM_URDF
    from yam_arena.urdf import rpy_matrix

    root = ET.parse(YAM_URDF).getroot()
    out: dict[str, list[np.ndarray]] = {}
    for link in root.iter("link"):
        for col in link.findall("collision"):
            mesh = col.find("geometry/mesh")
            if mesh is None:
                continue
            o = col.find("origin")
            xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = np.array([float(v) for v in o.get("rpy", "0 0 0").split()]) if o is not None else np.zeros(3)
            pts = _stl_points(str(YAM_URDF.parent / mesh.get("filename")))
            out.setdefault(link.get("name"), []).append(pts @ rpy_matrix(rpy).T + xyz)
    return {k: _dedupe(np.vstack(v)) for k, v in out.items()}


def _stl_points(path: str) -> np.ndarray:
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw, np.uint32, 1, 80)[0])
    tri = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)["v"].astype(float)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    return np.vstack([a, b, c, (a + b) / 2, (b + c) / 2, (a + c) / 2, (a + b + c) / 3])


def _dedupe(pts: np.ndarray, cell: float = 0.006) -> np.ndarray:
    _, idx = np.unique(np.round(pts / cell).astype(np.int64), axis=0, return_index=True)
    return pts[np.sort(idx)]


def arm_points(q, grip, base_pos) -> dict[str, np.ndarray]:
    """World points of every collision mesh per link for arm joints ``q`` and finger opening ``grip`` (1 open .. 0)."""
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS, FINGER_OPEN
    from yam_arena.urdf import forward_kinematics

    qd = {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)}
    qd.update(joint7=FINGER_OPEN * float(grip), joint8=FINGER_OPEN * float(grip))
    poses = forward_kinematics(YAM_URDF, qd)
    return {link: pts @ poses[link][:3, :3].T + poses[link][:3, 3] + np.asarray(base_pos)
            for link, pts in _meshes().items() if link in poses}


def box_points(centre, size, step=0.004) -> np.ndarray:
    c, s = np.asarray(centre, float), np.asarray(size, float)
    axes = [np.arange(-h, h + 1e-9, step) if h > step else np.array([-h, h]) for h in s / 2]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    face = np.any(np.isclose(np.abs(g), s / 2), axis=1)
    return g[face] + c


def _min_dist(a: np.ndarray, b: np.ndarray) -> float:
    best = np.inf
    for i in range(0, len(a), 256):
        d = a[i:i + 256, None, :] - b[None, :, :]
        best = min(best, float(np.sqrt((d * d).sum(-1).min())))
    return best


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    """Lower bound of the distance between two point sets: the gap between their bounding boxes."""
    lo, hi = np.maximum(a.min(0), b.min(0)), np.minimum(a.max(0), b.max(0))
    return float(np.linalg.norm(np.maximum(lo - hi, 0.0)))


def min_clearance(qa, ga, base_a, qb, gb, base_b, held_a=(), held_b=()):
    """Smallest distance between any link of arm A and any link of arm B, and between each arm and the part the other
    holds (``held_a``/``held_b``: point arrays of the part arm A / arm B holds). Returns (m, (name, name))."""
    pa, pb = arm_points(qa, ga, base_a), arm_points(qb, gb, base_b)
    pairs = sorted(((_gap(a, b), la, lb) for la, a in pa.items() for lb, b in pb.items()))
    best = (np.inf, ("", ""))
    for g, la, lb in pairs:
        if g >= best[0]:
            break
        d = _min_dist(pa[la], pb[lb])
        if d < best[0]:
            best = (d, (f"A.{la}", f"B.{lb}"))
    for part, arm, tag in ((held_a, pb, "B"), (held_b, pa, "A")):
        if len(part):
            part = np.asarray(part)
            for l, p in arm.items():
                if _gap(p, part) >= best[0]:
                    continue
                d = _min_dist(p, part)
                if d < best[0]:
                    best = (d, (f"{tag}.{l}", "held part of the other hand"))
    return best


# -- logging only: each hand's distance to each part of the task (the demo's TRACE prints it) -------------------------

HAND_LINKS = ("gripper", "tip_left", "tip_right", "link5")


def _rot_wxyz(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rot_axis(axis: str, a: float) -> np.ndarray:
    import math

    c, s = math.cos(a), math.sin(a)
    return {"X": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "Y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "Z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


class PartClouds:
    """Surface points (4 mm apart) of every box of each loose part's body and of each of its hinges, in the part frame
    (hinges at angle 0), kept box by box so only boxes near a hand are measured."""

    def __init__(self, specs, step: float = 0.004):
        self.items = []                                     # (label, part name, hinge or None, [box points])
        for spec in specs:
            if getattr(spec, "fixed", False) or not getattr(spec, "boxes", ()):
                continue
            groups = [(spec.name, None, spec.boxes)] + [(f"{spec.name}.{h.name}", h, h.boxes)
                                                        for h in getattr(spec, "hinges", ()) if h.boxes]
            for label, h, boxes in groups:
                clouds = []
                for b in boxes:
                    R = _rot_wxyz(b[3]) if len(b) > 3 else np.eye(3)
                    clouds.append(box_points((0, 0, 0), b[1], step) @ R.T + np.asarray(b[0], float))
                self.items.append((label, spec.name, h, clouds))

    def hand_gaps(self, states, arms, near: float = 0.05) -> dict:
        """``arms``: {"L": (q, grip, base), ...} at the measured joints. Per arm, the parts within ``near`` of its hand
        links and their smallest distance in mm, nearest first."""
        hands = {}
        for tag, (q, grip, base) in arms.items():
            pts = arm_points(q, grip, base)
            hands[tag] = np.concatenate([p for link, p in pts.items() if link in HAND_LINKS])
        out = {tag: {} for tag in hands}
        for label, name, h, clouds in self.items:
            st = states[name]
            R, t = st.rotation(), np.asarray(st.pos, float)
            if h is not None:
                piv = np.asarray(h.pivot, float)
                Rh = _rot_axis(h.axis, float(st.hinges.get(h.name, 0.0)))
            for c in clouds:
                local = c if h is None else (c - piv) @ Rh.T + piv
                w = local @ R.T + t
                for tag, hand in hands.items():
                    if _gap(hand, w) > near:
                        continue
                    d = 1000 * _min_dist(hand, w)
                    if d < out[tag].get(label, np.inf):
                        out[tag][label] = round(d, 1)
        return {tag: dict(sorted(v.items(), key=lambda kv: kv[1])) for tag, v in out.items()}
