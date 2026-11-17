"""Scripted full-score demo for interlock-bypass-key-remove, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]
    python demo.py task.py --ik          # no simulator: IK error of every planned target
    python demo.py task.py --dry         # no simulator: play the script on the IK alone

Open loop: every target comes from the task file's fixed layout (the switch, the hinge line, the door's size) and the
clock. The only feedback is each arm's own joint readings (a trim that keeps the measured tool point on the plan while
it pulls the key or presses the door). No object pose, joint state or task state steers it.

1. The right hand comes level from the front, closes on the spare key's body across its 15 mm height and pulls it
   straight out of the switch, against the detent.
2. It turns its wrist down, carries the key to the bench on the robot's right and lets it go there.
3. The left hand, fingers closed, pushes on the open door's outer face and swings it to about 7 degrees from shut, while
   the right hand waits outside the door's swing.
4. The right hand, fingers closed, presses the door's latch stile until the door's own key is home in the switch.
5. Both hands move away.

Negative variants (DEMO_VARIANT=...):
  door_first   the door is swung shut with the spare key still in the switch (0)
  key_inside   the spare key is pulled out and dropped inside the enclosure, then the door is closed (0.25)
  door_ajar    the spare key is pulled out and put on the bench; the door is left open (0.5)
  not_home     as the demo, but the door is pressed only to about 5 mm short of home (below 1)
Force check (DEMO_VARIANT=probe): no arm moves; a ramped pull on the spare key and a ramped torque on the door measure
the switch's extraction and insertion forces the way a spring scale would.
Writes one MP4 per camera, setup.png and result.json; DEMO_DEBUG=1 prints a trace every 10 steps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
VARIANTS = ("", "door_first", "key_inside", "door_ajar", "not_home", "probe")
KI, CORR_MAX = 0.2, 0.03        # joint-feedback trim: gain per step and limit (m)

# -- grip rotations: columns are the TCP's closing axis, its y axis and its approach axis (yam_arena convention) -----------
FWD = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])      # pointing +x, level, fingers closing vertically
PITCH = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])    # a quarter turn about +y: +x goes to -z
DOWN = PITCH @ FWD                                                         # the same grip pointing straight down
_c, _s = math.cos(math.radians(60.0)), math.sin(math.radians(60.0))
TILT = np.array([[_c, 0.0, _s], [0.0, 1.0, 0.0], [-_s, 0.0, _c]]) @ FWD     # pointing 60 degrees below level


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each from where the last command left it. With ``servo`` the
    command is trimmed by the arm's own joint readings so the measured tool point follows the plan under load."""

    def __init__(self, name: str, base_pos, read=None):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.pos = np.asarray(self.pos, float)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.read = read
        self.corr = np.zeros(3)
        self.corr_max = CORR_MAX
        self.max_corr = 0.0

    def _solve(self, pos, rot, servo=None):
        from yam_arena import ik

        if servo is not None and self.read is not None:
            meas = self.read()
            if meas is not None:
                self.corr = np.clip(self.corr + KI * np.asarray(servo) * (self.pos - meas[0]), -self.corr_max,
                                    self.corr_max)
                self.max_corr = max(self.max_corr, float(np.abs(self.corr).max()))
        q, pe, _ = ik.solve(np.asarray(pos, float) + self.corr, ik.matrix_to_quat_xyzw(rot), self.q,
                            base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None, servo=None):
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s), servo)
            yield

    def follow(self, points, rot, duration: float):
        """Through a list of tool points at a steady rate."""
        n = max(1, round(duration * CONTROL_HZ))
        pts = [self.pos.copy()] + [np.asarray(p, float) for p in points]
        for k in range(1, n + 1):
            f = k / n * (len(pts) - 1)
            i = min(int(f), len(pts) - 2)
            self._solve(pts[i] + (f - i) * (pts[i + 1] - pts[i]), rot)
            yield

    def hold(self, duration: float, grip: float | None = None, servo=None):
        yield from self.move(self.pos, self.rot, duration, grip, servo)

    def relax(self, duration: float = 0.5):
        c0 = self.corr.copy()
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self.corr = c0 * (1 - k / n)
            self._solve(self.pos, self.rot)
            yield


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def both(*gens):
    gens = [iter(g) for g in gens]
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        if gens:
            yield


def chain(*gens):
    for g in gens:
        yield from g


# -- the plan ------------------------------------------------------------------------------------------------------------
TIP = 0.013              # the pad tips reach 13 mm past the tool point
# The closed fingers' outline seen from above, from the URDF meshes: (distance along the approach axis from the tool
# point, half width). They taper from 68 mm wide at the carriages to 16 mm at the tips.
FINGER_PROFILE = ((0.013, 0.008), (0.000, 0.011), (-0.010, 0.015), (-0.020, 0.019), (-0.030, 0.023), (-0.040, 0.025),
                  (-0.050, 0.031), (-0.060, 0.034), (-0.144, 0.034))
FINGER_HALF = 0.034      # half the hand's greatest width
GRASP_IN = 0.010         # the fingertips enter 10 mm of the key body's 14 mm depth (they stop 4 mm off the head's face)
PRE_GRIP = 0.45          # fingers part-closed for the approach (about 42 mm apart)
PULL = 0.075             # the right hand is sent this far back: 28 mm of key travel plus what the detent takes up
BENCH_XY = (0.230, -0.330)   # where the spare key is put down
BENCH_Z = 0.058          # tool point height at release: the key's tongue tip is then about 1 cm above the bench
INSIDE = (0.325, -0.070, 0.140)   # variant key_inside: where the key is let go, inside the door opening
PUSH_R = 0.080           # the left hand pushes the door this far from its hinge line
PUSH_Z = 0.180           # height of both pushes
SWING_TO = math.radians(2.5)      # the left hand's plan ends this far from shut; its tapered fingertips leave the
#                                   door 5 degrees behind the plan, about 7 degrees from shut, its key just short of the hole
READY_FOR = math.radians(12.0)    # the right hand comes in clear of a door left as far as this from shut
PRESS_Y = -0.198         # the right hand presses the latch stile
PRESS_PAST = 0.020       # sent this far past shut (the door's stop takes it)
SHORT = 0.006            # variant not_home: sent only to this far short of shut


def door_face(T, shut_angle: float, r: float, off: float):
    """A point ``off`` outside the door's outer face, ``r`` from the hinge line, with the door ``shut_angle`` from shut."""
    a = -shut_angle
    dx, dy = -(T.DOOR_T + off), -r
    return np.array([T.HX + math.cos(a) * dx - math.sin(a) * dy, T.HY + math.sin(a) * dx + math.cos(a) * dy,
                     T.TOP + PUSH_Z])


def push_point(T, shut_angle: float, press: float = 0.002):
    """The left tool point that keeps closed fingertips (pointing +x) pressed on the door's outer face."""
    reach = max(a * math.cos(shut_angle) + w * math.sin(shut_angle) for a, w in FINGER_PROFILE)
    return door_face(T, shut_angle, PUSH_R, reach - press)


def plan(T) -> dict:
    """Tool-point targets (world, m) from the task file's fixed layout."""
    p = {}
    key_face = T.XH - T.SPARE_GAP - T.KEY_D                  # the spare key's outer face
    p["grasp"] = np.array([key_face + GRASP_IN - TIP, T.YK, T.TOP + T.SPARE_Z])
    p["pre"] = p["grasp"] - np.array([0.070, 0.0, 0.0])
    p["pulled"] = p["grasp"] - np.array([PULL, 0.0, 0.0])
    p["carry"] = np.array([BENCH_XY[0], BENCH_XY[1], T.TOP + 0.140])
    p["put"] = np.array([BENCH_XY[0], BENCH_XY[1], T.TOP + BENCH_Z])
    p["inside"] = np.array([INSIDE[0], INSIDE[1], T.TOP + INSIDE[2]])
    p["push_ready"] = door_face(T, T.OPEN, PUSH_R, FINGER_HALF + 0.035)
    p["push_start"] = push_point(T, T.OPEN)
    p["push_end"] = push_point(T, SWING_TO)
    stile_x = float(door_face(T, READY_FOR, T.HY - PRESS_Y, 0.0)[0])
    p["press_ready"] = np.array([stile_x - TIP - 0.010, PRESS_Y, T.TOP + PUSH_Z])
    p["press_home"] = np.array([T.XF - T.DOOR_T - TIP + PRESS_PAST, PRESS_Y, T.TOP + PUSH_Z])
    p["press_short"] = np.array([T.XF - T.DOOR_T - TIP - SHORT, PRESS_Y, T.TOP + PUSH_Z])
    return p


def script(task, ctx):
    T, P = task, plan(task)
    variant = os.environ.get("DEMO_VARIANT", "")
    L, R = ctx.left, ctx.right
    up = np.array([0.0, 0.0, 1.0])
    back = np.array([-1.0, 0.0, 0.0])
    if variant == "probe":
        yield from wait(60.0)
        return

    def swing(to_angle: float, seconds: float):
        angles = np.linspace(T.OPEN, to_angle, 24)[1:]
        return chain(L.move(P["push_ready"], FWD, 2.5, 0.0), L.move(P["push_start"], FWD, 0.8),
                     L.follow([push_point(T, a) for a in angles], FWD, seconds), L.hold(0.3))

    def left_away():
        return chain(L.move(L.pos + 0.05 * back + np.array([0.0, 0.03, 0.0]), FWD, 0.8),
                     L.move(np.array([0.22, 0.27, T.TOP + 0.24]), FWD, 1.5))

    if variant == "door_first":
        yield from chain(swing(-math.radians(3.0), 6.0), L.hold(1.0), wait(2.0))
        return

    # 1. the right hand takes the spare key by its body and pulls it straight out
    yield from chain(R.move(P["pre"], FWD, 2.5, PRE_GRIP), R.move(P["grasp"], FWD, 1.2), R.hold(0.6, 0.0),
                     R.move(P["pulled"], FWD, 3.5, servo=np.array([1.0, 0.0, 0.0])), R.relax(0.3))
    # 2. it puts the key down: on the bench to the right, or (variant) inside the enclosure
    if variant == "key_inside":
        # wrist turned 60 degrees down so the key slides off the open fingers (held level, it stays on the lower one)
        stow = chain(R.move(P["inside"] - np.array([0.06, 0.0, 0.0]), FWD, 1.5), R.move(P["inside"], TILT, 1.5),
                     R.hold(0.8, 1.0), R.move(P["inside"] - np.array([0.10, 0.0, -0.04]), TILT, 1.0),
                     R.move(P["carry"], TILT, 2.0))                      # out of the door's swing
    else:
        stow = chain(R.move(P["carry"], TILT, 2.5), R.move(P["put"], TILT, 1.0), R.hold(0.6, 1.0),
                     R.move(P["carry"], TILT, 0.8))
    yield from stow
    if variant == "door_ajar":
        yield from wait(2.0)
        return
    # 3. the left hand swings the door to about 7 degrees from shut; the right hand gets ready in front of the latch stile
    # (the right hand waits above where it put the key, outside the door's swing, and closes its fingers)
    yield from both(swing(SWING_TO, 4.5), R.hold(0.8, 0.0))
    yield from both(left_away(), R.move(P["press_ready"], FWD, 2.2))
    # 4. the right hand presses the door shut: the door's key goes into the switch and past the detent
    target = P["press_short"] if variant == "not_home" else P["press_home"]
    servo = None if variant == "not_home" else np.array([1.0, 0.0, 0.0])
    yield from chain(R.move(target, FWD, 3.5, servo=servo), R.hold(0.5, servo=servo), R.relax(0.3))
    # 5. both hands away
    yield from chain(R.move(R.pos + 0.06 * back, FWD, 1.0), R.move(np.array([0.22, -0.27, T.TOP + 0.24]), FWD, 1.5))
    yield from wait(1.5)


# -- inter-arm clearance from the URDF collision meshes (logging only; commanded joints, no simulator state) -------------
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
            raw = open(str(YAM_URDF.parent / mesh.get("filename")), "rb").read()
            n = int(np.frombuffer(raw, np.uint32, 1, 80)[0])
            tri = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)["v"]
            a, b, c = (tri[:, i].astype(float) for i in range(3))
            pts = np.vstack([a, b, c, (a + b) / 2, (b + c) / 2, (a + c) / 2, (a + b + c) / 3])
            out.setdefault(link.get("name"), []).append(pts @ rpy_matrix(rpy).T + xyz)
    res = {}
    for k, v in out.items():
        pts = np.vstack(v)
        _, idx = np.unique(np.round(pts / 0.006).astype(np.int64), axis=0, return_index=True)
        res[k] = pts[np.sort(idx)]
    return res


def arm_points(q, grip, base_pos) -> dict[str, np.ndarray]:
    """World points on every collision mesh, per link, for arm joints ``q`` and finger opening ``grip`` (1 open)."""
    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS, FINGER_OPEN
    from yam_arena.urdf import forward_kinematics

    qd = {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)}
    qd.update(joint7=FINGER_OPEN * float(grip), joint8=FINGER_OPEN * float(grip))
    poses = forward_kinematics(YAM_URDF, qd)
    return {link: pts @ poses[link][:3, :3].T + poses[link][:3, 3] + np.asarray(base_pos)
            for link, pts in _meshes().items() if link in poses}


def arm_gap(left, right) -> float:
    """Smallest sampled distance between any link of one arm and any link of the other (m)."""
    pa, pb = arm_points(left.q, left.grip, left.base), arm_points(right.q, right.grip, right.base)
    best = np.inf
    for a in pa.values():
        for b in pb.values():
            lo, hi = np.maximum(a.min(0), b.min(0)), np.minimum(a.max(0), b.max(0))
            if float(np.linalg.norm(np.maximum(lo - hi, 0.0))) >= best:
                continue
            for i in range(0, len(a), 256):
                d = a[i:i + 256, None, :] - b[None, :, :]
                best = min(best, float(np.sqrt((d * d).sum(-1).min())))
    return best


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"], read=lambda: self.joint_tcp("left"))
        self.right = Arm("right", station.BASES["right"], read=lambda: self.joint_tcp("right"))

    def joint_tcp(self, name: str):
        """The arm's tool point and gripper rotation from its own joint positions (forward kinematics)."""
        from partkit import _np
        from yam_arena import ik, station
        from yam_arena.constants import ARM_JOINTS

        art = self.emb._env.unwrapped.scene[f"{name}_robot"]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        pos, rot = ik.tcp_pose(_np(art.data.joint_pos)[0][ids], station.BASES[name])
        return np.asarray(pos, float), np.asarray(rot, float)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


class Probe:
    """Force check, logging only (it reads the door's hinge and the key's pose, and pushes on them directly; no arm
    moves). 1: a pull on the spare key that grows by 0.1 N a step until its cross-bar is over the plunger. 2: the door
    is brought slowly to where its key meets the plunger, then a closing torque grows by 0.015 N m a step until the key
    is past it. 3: an opening torque grows by 0.03 N m a step until the key is out again. Each force is reported at
    the key."""

    def __init__(self, emb, task):
        import torch

        self.torch, self.T = torch, task
        base = emb._env.unwrapped
        self.spare, self.door = base.scene["spare_key"], base.scene["door"]
        self.leaf = list(self.door.body_names).index("leaf")
        self.phase, self.f, self.out = "pull", 0.0, {}
        self.settle, self.last_gap = 0, None

    def _force(self, asset, force, torque, body=None):
        t = self.torch
        dev = getattr(asset.data.root_pos_w, "torch", asset.data.root_pos_w).device
        n = 1 if body is None else len(asset.body_names)
        F, Tq = t.zeros((1, n, 3), device=dev), t.zeros((1, n, 3), device=dev)
        i = 0 if body is None else body
        F[0, i], Tq[0, i] = t.tensor(force, device=dev), t.tensor(torque, device=dev)
        asset.set_external_force_and_torque(F, Tq, is_global=True)

    def _torque(self, tz):
        self._force(self.door, (0.0, 0.0, 0.0), (0.0, 0.0, float(tz)), self.leaf)

    def step(self, info: dict) -> None:
        if "spare_depth_mm" not in info:
            return
        depth, gap = info["spare_depth_mm"], info["door_gap_mm"]
        speed = 0.0 if self.last_gap is None else (self.last_gap - gap) * CONTROL_HZ      # mm/s toward shut, at the key
        self.last_gap = gap
        if self.phase == "pull":
            self.f += 0.1
            self._force(self.spare, (-self.f, 0.0, 0.0), (0.0, 0.0, 0.0))
            if depth < 21.0:                              # the cross-bar is over the plunger's nose
                self.out["spare_extraction_n"] = round(self.f, 1)
                self._force(self.spare, (-0.5, 0.0, 0.0), (0.0, 0.0, 0.0))    # keeps it sliding away, off the bench
                self.phase, self.f = "approach", 0.0
        elif self.phase == "approach":                    # 60 mm/s at the key, then 10 mm/s for the last 60 mm
            want = 60.0 if gap > 75.0 else 10.0
            self.f = min(0.4, max(0.0, self.f + 0.0004 * (want - speed)))
            self._torque(self.f)
            if gap < 14.0:
                self._torque(0.0)
                self.phase, self.settle = "pause", 0
        elif self.phase == "pause":
            self.settle += 1
            self.out["door_gap_at_contact_mm"] = gap
            if self.settle > 20:
                self.phase, self.f = "close", 0.0
        elif self.phase == "close":
            self.f += 0.015
            self._torque(self.f)
            if gap < 4.0:                                 # the cross-bar is under the plunger's edge or past it
                self.out["door_insertion_n"] = round(self.f / self.T.KEY_RADIUS, 1)
                self._torque(1.2)                         # a 5 N push the rest of the way
                self.phase, self.f, self.settle = "rest", 0.0, 0
        elif self.phase == "rest":
            self.settle += 1
            if self.settle == 10:
                self._torque(0.0)
            self.out["door_gap_at_rest_mm"] = gap
            if self.settle > 25:
                self.phase = "open"
        elif self.phase == "open":
            self.f += 0.03
            self._torque(-self.f)
            if gap > 6.0:
                self.out["door_extraction_n"] = round(self.f / self.T.KEY_RADIUS, 1)
                self._torque(0.0)
                self.phase = "done"


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant.startswith("probe_r"):                    # the force check at another spring rate (N/m)
        os.environ["IBK_RATE"] = variant[len("probe_r"):]
        os.environ["DEMO_VARIANT"] = variant = "probe"
    if variant == "probe":
        max_steps = max(max_steps, 2400)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    gen = script(task, ctx)
    frames: dict[str, list] = {}
    debug = os.environ.get("DEMO_DEBUG", "1") != "0"
    min_gap = float("inf")
    probe = Probe(emb, task) if variant == "probe" else None

    print("PLAN " + json.dumps({k: np.round(v, 4).tolist() for k, v in plan(task).items()}), flush=True)

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def row(steps, info):
        """Logging only: reads simulator state for the trace."""
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)
        S = task._STATE
        return {"step": steps, "score": info.get("score"), **S["info"],
                "plunger_mm": round(1000 * st["enclosure"].sliders["plunger"], 2),
                "door_deg": round(math.degrees(st["door"].hinges["leaf"]), 2),
                "spare": np.round(st["spare_key"].pos - [0, 0, task.TOP], 4).tolist(),
                "tcp_err_mm": {a.name[0]: (np.round(1000 * (ctx.joint_tcp(a.name)[0] - a.pos - a.corr), 1).tolist())
                               for a in (ctx.left, ctx.right)},
                "corr_mm": {"L": np.round(1000 * ctx.left.corr, 1).tolist(),
                            "R": np.round(1000 * ctx.right.corr, 1).tolist()},
                "probe": None if probe is None else {"phase": probe.phase, "f": round(probe.f, 2), **probe.out}}

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        if probe is not None:
            probe.step(task._STATE["info"])
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 5 == 0:
            try:
                min_gap = min(min_gap, arm_gap(ctx.left, ctx.right))
            except Exception as exc:  # noqa: BLE001 - logging only
                if steps == 5:
                    print(f"clearance unavailable: {exc!r}", flush=True)
        if debug and (steps % 10 == 0 or probe is not None):
            try:
                print("TRACE " + json.dumps(row(steps, info), default=str), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"trace unavailable: {exc!r}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if probe is not None and probe.phase == "done":
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    S = task._STATE
    result = {"task_score": score, "success": success, "steps": steps, "variant": variant,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "max_trim_mm": {"left": round(1000 * ctx.left.max_corr, 1), "right": round(1000 * ctx.right.max_corr, 1)},
              "min_arm_link_gap_mm": round(1000 * min_gap, 1) if min_gap < float("inf") else None,
              "end": S["info"], "finger_check": S["finger_check"],
              "probe": None if probe is None else probe.out}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)
    return result


def dry_run(task_path: str) -> None:
    """No simulator: play the script on the IK alone and report the worst IK error and the smallest arm-to-arm gap."""
    import importlib

    from yam_arena import station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)

    class Dry:
        left = Arm("left", station.BASES["left"])
        right = Arm("right", station.BASES["right"])

    steps, gap, worst = 0, float("inf"), (0.0, 0, "")
    for _ in script(task, Dry):
        steps += 1
        for arm in (Dry.left, Dry.right):
            from yam_arena import ik

            pe = float(np.linalg.norm(ik.tcp_pose(arm.q, arm.base)[0] - arm.pos))
            if pe > worst[0]:
                worst = (pe, steps, arm.name)
            if pe > 0.002 and steps % 5 == 0:
                print(f"DRY {steps:4d} {arm.name} IK error {1000 * pe:.1f} mm at {np.round(arm.pos - [0, 0, task.TOP], 3).tolist()}",
                      flush=True)
        if steps % 5 == 0:
            gap = min(gap, arm_gap(Dry.left, Dry.right))
        if steps % 20 == 0:
            print(f"DRY {steps:4d} L {np.round(Dry.left.pos - [0, 0, task.TOP], 3).tolist()} "
                  f"R {np.round(Dry.right.pos - [0, 0, task.TOP], 3).tolist()} gap {1000 * gap:.0f} mm", flush=True)
    print(f"DRY steps {steps}, worst IK error {1000 * worst[0]:.1f} mm at step {worst[1]} ({worst[2]}), "
          f"min arm gap {1000 * gap:.0f} mm", flush=True)


def ik_check(task_path: str) -> None:
    """No simulator: yam_arena IK error of every planned target, with the grip the script uses there."""
    import importlib

    from yam_arena import ik, station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)
    P = plan(task)
    arms = {"left": Arm("left", station.BASES["left"]), "right": Arm("right", station.BASES["right"])}
    use = {"push_ready": ("left", FWD), "push_start": ("left", FWD), "push_end": ("left", FWD),
           "carry": ("right", TILT), "put": ("right", TILT), "inside": ("right", TILT)}
    for name, p in P.items():
        side, rot = use.get(name, ("right", FWD))
        arm = arms[side]
        q, pe, re = ik.solve(np.asarray(p, float), ik.matrix_to_quat_xyzw(rot), arm.q, base_pos=arm.base, iters=400)
        print(f"IK {name:12s} {side:5s} {np.round(p - [0, 0, task.TOP], 3).tolist()} err {1000 * pe:.1f} mm / {re:.3f} rad",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ik", action="store_true", help="only print the IK error of every planned target")
    ap.add_argument("--dry", action="store_true", help="play the script on the IK alone (no simulator)")
    a = ap.parse_args()
    if a.ik:
        ik_check(a.task)
    elif a.dry:
        dry_run(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
