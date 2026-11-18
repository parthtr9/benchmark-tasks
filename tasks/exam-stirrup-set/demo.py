"""Scripted full-score demo for exam-stirrup-set, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]
    python demo.py task.py --ik          # no simulator: IK error of every planned target
    python demo.py task.py --dry         # no simulator: play the script on the IK alone (IK error, arm-to-arm gap)

Open loop: every target comes from the task file's fixed layout (the shelf, the stirrup bars' lines, the boss, the
draw, the stirrup's size, the lock positions) and the clock; the lift is trimmed by the arm's own joint readings. No
object pose, joint state or task state steers it.

Each arm sets the stirrup on its side, both at once:
1. pinches the stirrup's hinge end (the clevis showing at the shelf's front) with the hand pitched 45 deg and draws the
   bar out 400 mm to its collar, lets go and backs off;
2. pinches the folded stirrup's arm from straight above and carries it round its roll pin to 96 deg, the hand turning
   with it, lets go (it settles on its open stop at 100 deg) and backs off;
3. pinches the bar 50 mm behind the roll pin from straight above, lifts it (the bar tilts about the boss and the
   detent post drops out of the bracket), swings it 24 deg outward round the boss, the hand turning with it, lets it
   down so the post rises into the outermost notch, lets go and rises clear.

Negative variants (DEMO_VARIANT=...):
  one_side      only the left stirrup is set (0.5)
  no_unfold     both drawn, swung and locked, but left folded (0)
  unfold_first  the hand tries to turn each stirrup up before drawing it (the shelf's window stops it), then draws,
                swings and locks it without unfolding (0)
  no_lift       drawn and unfolded, then pushed sideways without lifting: the post in its groove stops it (0.5 in
                MuJoCo; on Isaac one of the two stirrups tips back over as its hand lets go, 0.25)
  between       drawn, unfolded, swung only 18 deg and let down between notches: it rests unlocked on the teeth (0.5)
  held          as the demo, but the right hand stays on its bar at the end (0.5)
Writes one MP4 per camera, setup.png and result.json; DEMO_DEBUG=0 silences the trace printed every 10 steps.
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

VARIANTS = ("", "one_side", "no_unfold", "unfold_first", "no_lift", "between", "held")

CONTROL_HZ = 20
KI, CORR_MAX = 0.2, 0.03        # joint-feedback trim: gain per step and limit (m)

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


def rel_move(arm, offset, duration: float, rot=None, grip: float | None = None):
    """A move by ``offset`` from wherever the arm's command is when this step starts (not when the chain is built)."""
    yield from arm.move(arm.pos + np.asarray(offset, float), arm.rot if rot is None else rot, duration, grip)


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
DRAW_PITCH = math.radians(45.0)      # drawing: the hand pitched 45 deg down, fingers closing across the clevis
CLEVIS_S = 0.010         # ... the tool point 10 mm in front of the roll pin, on the exposed part of the clevis
DRAW_OVER = 0.015        # the pull is planned 15 mm past the draw collar; the collar takes it
DRAW_SLOW = 0.020        # ... the last 20 mm before the collar slowly, so the bar meets it gently (a hard stop kicks it up)
ARM_S = 0.115            # unfolding: the stirrup's arm is pinched this far from the roll pin
UNFOLD_TO = math.radians(96.0)       # ... and carried round to here (the open stop is at 100: it settles there)
RETREAT_PITCH = math.radians(65.0)    # backing off the stirrup the hand tips down to this pitch
HAND_PITCH_MAX = math.radians(88.0)  # the hand turns with it up to here (out of reach beyond); the last degrees turn in the pads
LIFT_BY = 0.065          # lifting: the hand rises this much (5 deg of bar tilt needs 38 mm at the grip; the spring and
                         # the weight load the arm about 10 N, which the joint-reading trim takes up), swings, then
                         # comes down this far below the lifted pose
DROP_BY = 0.075
ABOVE = 0.050            # approaches come in from this far off the grip
PART_OPEN = 0.55         # fingers part-open round the 25-32 mm parts (about 52 mm)
VARIANT_SWING = {"between": math.radians(18.0), "no_lift": math.radians(8.0)}   # no_lift: a push that meets the post


def pitched(pitch: float, yaw: float = 0.0):
    """Gripper pitched ``pitch`` from straight down toward +x, turned ``yaw`` about the vertical; fingers close
    across (along y before the yaw)."""
    from yam_arena import ik  # noqa: F401 - keeps the import pattern of the other helpers

    a = (math.sin(pitch) * math.cos(yaw), math.sin(pitch) * math.sin(yaw), -math.cos(pitch))
    z = np.asarray(a, float) / np.linalg.norm(a)
    x = np.array([math.sin(yaw), -math.cos(yaw), 0.0])     # yam_arena's straight-down grip: fingers along -y
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def jstate(T, s, swing=0.0, lift=0.0, draw=0.0, unfold=0.0):
    return {"swing": swing, "lift": lift, "draw": draw, "unfold": unfold}


def clevis_tcp(T, s, draw: float):
    return T.stirrup_point(s, jstate(T, s, draw=draw), (-CLEVIS_S, 0.0, 0.001))


def arm_tcp(T, s, unfold: float, swing: float = 0.0):
    return T.stirrup_point(s, jstate(T, s, swing=swing, draw=T.DRAW, unfold=unfold), (ARM_S, 0.0, T.BT / 2 + T.ARM_T / 2))


def clevis_tcp_at(T, s, swing: float):
    return T.stirrup_point(s, jstate(T, s, swing=swing, draw=T.DRAW), (-CLEVIS_S, 0.0, 0.001))


def plan(T) -> dict:
    out = {}
    for s in T.SIDES:
        k = s.name
        out[f"{k}_clevis"] = clevis_tcp(T, s, 0.0)
        out[f"{k}_drawn"] = clevis_tcp(T, s, T.DRAW + DRAW_OVER)
        out[f"{k}_arm0"] = arm_tcp(T, s, 0.0)
        out[f"{k}_arm_open"] = arm_tcp(T, s, UNFOLD_TO)
        out[f"{k}_arm_out"] = arm_tcp(T, s, UNFOLD_TO, s.out * T.OUTER) + [0.0, 0.0, LIFT_BY]
        out[f"{k}_clevis_out"] = clevis_tcp_at(T, s, s.out * T.OUTER) + [0.0, 0.0, LIFT_BY]
    return out


def script(task, ctx):
    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    target_of = lambda s: s.out * VARIANT_SWING.get(variant, T.OUTER)   # noqa: E731

    def lift_swing_drop(arm, s, point, rot_at, lift=True):
        """Holding the stirrup (or its clevis): lift, swing round the boss to the lock position, let down."""
        up = LIFT_BY if lift else 0.0
        target = target_of(s)
        yield from arm.move(point(0.0) + [0.0, 0.0, up], rot_at(0.0), 1.2, servo=(0, 0, 1))
        n = 30
        for k in range(1, n + 1):          # round the boss, the hand turning with the bar
            a = target * k / n
            yield from arm.move(point(a) + [0.0, 0.0, up], rot_at(a), 3.0 / n)
        if lift:
            yield from arm.move(point(target) + [0.0, 0.0, up - DROP_BY], rot_at(target), 1.2)
        else:                              # the post would not let it move: ease the push off before letting go
            yield from arm.move(point(0.0), rot_at(0.0), 1.2)
        yield from arm.hold(0.5)
        yield from arm.relax(0.4)

    def draw(arm, s, keep=False):
        rot = pitched(DRAW_PITCH)
        p0, p1 = clevis_tcp(T, s, 0.0), clevis_tcp(T, s, T.DRAW + DRAW_OVER)
        back = np.array([-math.sin(DRAW_PITCH), 0.0, math.cos(DRAW_PITCH)]) * ABOVE
        yield from arm.move(p0 + back, rot, 2.0, grip=PART_OPEN)
        yield from arm.move(p0, rot, 1.0)
        yield from arm.hold(0.6, grip=0.0)
        if variant == "unfold_first":      # try to turn the stirrup up while it is stowed: the window's top stops it
            yield from arm.move(p0 + [0.0, 0.0, 0.025], rot, 1.0)
            yield from arm.move(p0, rot, 0.8)
        arm.corr_max = 0.010               # a small trim only: the collar, not the trim, ends the draw
        yield from arm.move(clevis_tcp(T, s, T.DRAW - DRAW_SLOW), rot, 4.0, servo=(1, 0, 0))
        yield from arm.move(p1, rot, 1.2, servo=(1, 0, 0))
        yield from arm.hold(0.6)
        yield from arm.relax(0.4)
        arm.corr_max = CORR_MAX
        if keep:                           # the variants that skip the unfold lift and swing it by the clevis
            yield from lift_swing_drop(arm, s, lambda a: clevis_tcp_at(T, s, a), lambda a: pitched(DRAW_PITCH, a))
            a = target_of(s)
            back = np.array([-math.sin(DRAW_PITCH) * math.cos(a), -math.sin(DRAW_PITCH) * math.sin(a),
                             math.cos(DRAW_PITCH)]) * 0.07
        yield from arm.hold(0.5, grip=PART_OPEN)
        yield from rel_move(arm, back, 1.0)

    def unfold_set(arm, s):
        p0 = arm_tcp(T, s, 0.0)
        yield from arm.move(p0 + [0.0, 0.0, ABOVE], pitched(0.0), 2.0, grip=PART_OPEN)
        yield from arm.move(p0, pitched(0.0), 1.0)
        yield from arm.hold(0.6, grip=0.0)
        n = 40
        for k in range(1, n + 1):          # carried round the roll pin, the hand turning with the stirrup
            a = UNFOLD_TO * k / n
            yield from arm.move(arm_tcp(T, s, a), pitched(min(a, HAND_PITCH_MAX)), 4.0 / n)
        yield from arm.hold(0.4)
        yield from lift_swing_drop(arm, s, lambda a: arm_tcp(T, s, UNFOLD_TO, a),
                                   lambda a: pitched(HAND_PITCH_MAX, a), lift=(variant != "no_lift"))
        if variant == "held" and s.name == "right":
            yield from arm.hold(6.0)
            return
        if variant == "no_lift":           # after the blocked push the arm is still sprung off its command: let it
            yield from arm.hold(1.5)       # settle, then take its command to where its own joints say the hand is, so
            meas = arm.read() if arm.read is not None else None    # it does not spring up into the leaning loop as
            if meas is not None:                                    # the fingers open; open wide and slide off down
                yield from arm.move(np.asarray(meas[0], float) - arm.corr, arm.rot, 0.8)   # along the stirrup, then
            yield from arm.hold(1.2, grip=1.0)      # back off without tipping the hand
            yield from rel_move(arm, [0.0, 0.0, -0.045], 1.0)
            yield from rel_move(arm, [-0.040, 0.0, 0.0], 1.0)
            yield from rel_move(arm, [-0.010, -s.sign * 0.100, -0.010], 1.2, rot=pitched(math.radians(40.0)))
            return
        yield from arm.hold(0.4, grip=PART_OPEN)
        a = target_of(s)
        along = np.array([math.cos(a), math.sin(a), 0.0])        # along the bar, toward the boss
        yield from rel_move(arm, -0.030 * along + [0.0, 0.0, -0.012], 0.8,   # off through the open fingertips,
                            rot=pitched(RETREAT_PITCH, a))                   # the hand tipping down (reach)
        yield from rel_move(arm, [-0.070, 0.0, -0.010], 1.2, rot=pitched(math.radians(40.0), a))   # back, tipping down

    def side(arm, s):
        skip = variant in ("no_unfold", "unfold_first")
        yield from draw(arm, s, keep=skip)
        if not skip:
            yield from unfold_set(arm, s)

    if variant == "one_side":
        yield from side(ctx.left, T.LEFT)
    else:
        yield from both(side(ctx.left, T.LEFT), side(ctx.right, T.RIGHT))


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



def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise SystemExit(f"unknown DEMO_VARIANT {variant!r}; one of {VARIANTS}")
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

    print("PLAN " + json.dumps({k: np.round(v, 4).tolist() for k, v in plan(task).items()}), flush=True)

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def row(steps, info):
        """Logging only: reads simulator state for the trace."""
        S = task._STATE
        return {"step": steps, "score": info.get("score"), **S["info"],
                "gap_mm": {k: round(1000 * v, 1) for k, v in S.get("gap", {}).items()},
                "grip": {a.name[0]: round(a.grip, 2) for a in (ctx.left, ctx.right)},
                "tcp_err_mm": {a.name[0]: (np.round(1000 * (ctx.joint_tcp(a.name)[0] - a.pos - a.corr), 1).tolist())
                               for a in (ctx.left, ctx.right)}}

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
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
        if debug and steps % 10 == 0:
            try:
                print("TRACE " + json.dumps(row(steps, info), default=str), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"trace unavailable: {exc!r}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    S = task._STATE
    result = {"task_score": score, "success": success, "steps": steps, "variant": variant,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "max_trim_mm": {"left": round(1000 * ctx.left.max_corr, 1), "right": round(1000 * ctx.right.max_corr, 1)},
              "min_arm_link_gap_mm": round(1000 * min_gap, 1) if min_gap < float("inf") else None,
              "end": S["info"], "finger_check": S["finger_check"]}
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
    """No simulator: yam_arena IK error of every planned target."""
    import importlib

    from yam_arena import ik, station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)
    rots = {"clevis": pitched(DRAW_PITCH), "drawn": pitched(DRAW_PITCH), "arm0": pitched(0.0),
            "arm_open": pitched(HAND_PITCH_MAX)}
    for name, p in plan(task).items():
        side, key = name.split("_", 1)
        s = task.LEFT if side == "left" else task.RIGHT
        rot = rots.get(key) if key in rots else (pitched(HAND_PITCH_MAX, s.out * task.OUTER) if key == "arm_out"
                                                 else pitched(DRAW_PITCH, s.out * task.OUTER))
        q, pe, re = ik.solve(np.asarray(p, float), ik.matrix_to_quat_xyzw(rot), np.array(station_ready()), base_pos=station.BASES[side], iters=400)
        print(f"IK {name:16s} {np.round(p - [0, 0, task.TOP], 3).tolist()} err {1000 * pe:.1f} mm / {re:.3f} rad", flush=True)


def station_ready():
    from yam_arena.constants import READY

    return READY


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
