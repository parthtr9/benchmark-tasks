"""Scripted full-score demo for the manual blood-pressure task, run in the yam_arena embodiment. Open loop.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The left arm takes the chestpiece from the dish by its stem, sets it on the red mark and presses it in about 5 mm
(2 N), and holds. Meanwhile the right arm, fingers closed, presses the bulb's squeeze plate four times, then takes the
black T-bar bleed knob and rolls its wrist 0.24 rad (the knob ends up near 0.28 in sim, about 3.0 mmHg/s), holds that for a fixed time, rolls on to 1.1 rad to
empty the cuff, lets go (the valve stays open) and lifts. The left arm then opens its fingers and lifts 6 cm. Nothing is read from the
simulation to steer: only the fixed layout, timing and the arms' own joint readings.

Negative variants (DEMO_VARIANT=...):
  one_arm    the right arm pumps and bleeds; the chestpiece stays in its dish (0)
  rest_only  the left sets the chestpiece on the mark and lets go; the right pumps and bleeds (0)
  dump       the knob is rolled to 1.1 rad at once: the cuff drops at 40 mmHg/s (0)
  slow       the knob is held at 0.11 rad commanded (about 0.15 in sim): about 1.7 mmHg/s (0.25)
  no_empty   the knob is turned back shut at about 85 mmHg: the cuff is not emptied (0.5)
  overpump   seven squeezes, above 260 mmHg (0)
DEMO_DEBUG=1 prints (and saves in result.json) a trace every 10 steps: pressure, bleed rate, knob angle, chestpiece
press and offset, flags, each arm's commanded-versus-measured lag, and the fingertips-versus-feature offset at each close.
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder.
"""


from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from functools import lru_cache
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20


def _rot_axis(axis, angle) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each starting where the last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None):
        """Generator: go to ``pos``/``rot`` (world TCP); ``grip`` 1 open .. 0 closed (None keeps it)."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def set_grip(self, grip: float, duration: float = 0.8):
        yield from self.move(self.pos, self.rot, duration, grip)

    def aim(self, pos, rot):
        """Track a pose this step (for closed-loop moves)."""
        self._solve(pos, rot)


def reachable_yaw(arm, pos, yaw):
    """The grip yaw, or the same parallel grip turned 180 degrees, whichever the arm reaches more exactly. A wrist at
    its joint limit otherwise leaves the fingers several degrees off and they twist light parts as they close."""
    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    best = None
    for y in (yaw, yaw + math.pi, yaw - math.pi):
        _, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(grip_rotation(y)), arm.q, base_pos=arm.base, iters=200)
        if best is None or pe + 0.05 * re < best[0]:
            best = (pe + 0.05 * re, y)
    return best[1]


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




VARIANTS = ("", "one_arm", "rest_only", "dump", "slow", "no_empty", "overpump")
UP = np.array([0.0, 0.0, 0.07])
THETA = {"": 0.24, "slow": 0.11, "no_empty": 0.24, "dump": 1.10, "overpump": 1.10, "one_arm": 0.24, "rest_only": 0.24}
HOLD_S = {"": 24.0, "one_arm": 24.0, "rest_only": 24.0, "slow": 38.0, "no_empty": 27.0, "dump": 5.0, "overpump": 5.0}
EVENTS: list = []                 # fingertips-versus-feature offsets at each close (logging only)
LAG = {"left": 0.0, "right": 0.0}


def script(task, ctx):
    """The demo as a generator: one yield per control step. Everything is from the fixed layout and fixed timing."""
    from yam_arena.demo import grip_rotation

    T = task
    L, R = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    squeezes = 7 if variant == "overpump" else 4
    theta = THETA[variant]
    flag = {"right_done": False}

    stem = np.array([*T.DISH_XY, T.TOP + T.DISH_H + T.STEM_GRIP_Z])
    mark_contact = np.array([*T.MARK_XY, T.SKIN_Z + T.STEM_GRIP_Z])
    bulb_xy = np.array([T.PUMP_XY[0], T.PUMP_XY[1] + T.BULB_DY])
    knob = np.array([T.PUMP_XY[0], T.PUMP_XY[1] + T.KNOB_DY, T.TOP + T.KNOB_Z])
    yaw_l = reachable_yaw(L, stem, 0.0)
    yaw_k = reachable_yaw(R, knob, 0.0)
    rot_l = grip_rotation(yaw_l)

    def left():
        if variant == "one_arm":
            while not flag["right_done"]:
                yield
            return
        yield from L.move(stem + UP, rot_l, 1.5, grip=1.0)
        yield from L.move(stem, rot_l, 1.0)
        yield from L.set_grip(0.0, 0.8)
        EVENTS.append(("close_chest", "left", stem.copy()))
        yield from wait(0.2)
        yield from L.move(stem + np.array([0, 0, 0.07]), rot_l, 0.7)
        above = np.array([*T.MARK_XY, mark_contact[2] + 0.07])
        yield from L.move(above, rot_l, 1.4)
        yield from L.move(mark_contact + np.array([0, 0, 0.012]), rot_l, 1.0)
        press = mark_contact + np.array([0, 0, -0.005])           # 5 mm into the diaphragm: about 2 N
        yield from L.move(press, rot_l, 1.5)
        if variant == "rest_only":
            yield from wait(1.0)
            yield from L.set_grip(1.0, 0.6)
            yield from L.move(L.pos + UP, L.rot, 1.0)
            while not flag["right_done"]:
                yield
            return
        while not flag["right_done"]:           # keep pressing until the right arm has let go and lifted
            yield
        yield from L.set_grip(1.0, 0.5)
        yield from L.move(L.pos + np.array([0, 0, 0.06]), L.rot, 1.0)

    def right():
        rot0 = grip_rotation(0.0)
        top = np.array([bulb_xy[0], bulb_xy[1], T.TOP + T.PLATE_TOP + 0.013])
        yield from R.move(top + np.array([0, 0, 0.10]), rot0, 1.5, grip=0.0)
        yield from R.move(top + np.array([0, 0, 0.03]), rot0, 0.6)
        for _ in range(squeezes):
            yield from R.move(top + np.array([0, 0, -0.028]), rot0, 0.5)
            yield from wait(0.1)
            yield from R.move(top + np.array([0, 0, 0.012]), rot0, 0.4)
        yield from R.move(top + np.array([0, 0, 0.08]), rot0, 0.5)
        rot_k = grip_rotation(yaw_k)
        yield from R.move(knob + np.array([0, 0, 0.07]), rot_k, 1.0, grip=1.0)
        yield from R.move(knob, rot_k, 0.8)
        yield from R.set_grip(0.0, 0.6)
        EVENTS.append(("close_knob", "right", knob.copy()))
        yield from wait(0.2)
        n = max(1, round(0.6 * CONTROL_HZ))
        for k in range(1, n + 1):                    # roll the wrist to the bleed angle
            R.aim(knob, grip_rotation(yaw_k + theta * k / n))
            yield
        for _ in range(round(HOLD_S[variant] * CONTROL_HZ)):
            R.aim(knob, grip_rotation(yaw_k + theta))
            yield
        if variant not in ("no_empty", "dump", "overpump"):
            n = max(1, round(0.4 * CONTROL_HZ))
            for k in range(1, n + 1):                # roll on to the dump position and empty the cuff
                R.aim(knob, grip_rotation(yaw_k + theta + (1.1 - theta) * k / n))
                yield
            for _ in range(round(2.6 * CONTROL_HZ)):
                R.aim(knob, grip_rotation(yaw_k + 1.1))
                yield
        if variant == "no_empty":                    # shut the valve again at about 85 mmHg
            n = max(1, round(0.4 * CONTROL_HZ))
            for k in range(1, n + 1):
                R.aim(knob, grip_rotation(yaw_k + theta * (1 - k / n)))
                yield
        yield from R.set_grip(1.0, 0.5)             # let go: the valve stays where it was set
        flag["right_done"] = True
        yield from R.move(R.pos + np.array([0, 0, 0.07]), R.rot, 1.0)

    gl, gr = left(), right()
    yield from both(gl, gr)
    yield from wait(1.2)


# -- inter-arm clearance from the URDF collision meshes (logging only) ------------------------------------------------
# Each collision piece is sampled at its vertices, edge midpoints and triangle centres and moved with the real forward
# kinematics; the gap is the smallest point-to-point distance (a few mm above the true mesh distance at most).

def _stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    n = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    tri = np.frombuffer(raw[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))["v"]
    tri = tri.astype(float)
    pts = np.concatenate([tri.reshape(-1, 3), tri.mean(1), (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2,
                          (tri[:, 2] + tri[:, 0]) / 2])
    return np.unique(np.round(pts, 5), axis=0)


@lru_cache(maxsize=1)
def _pieces():
    from yam_arena import ik
    from yam_arena.urdf import rpy_matrix

    urdf = Path(ik.YAM_URDF)
    out = []
    for link in ET.parse(urdf).getroot().iter("link"):
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()] if o is not None else [0, 0, 0]
            pts = _stl(urdf.parent / col.find("geometry").find("mesh").get("filename"))
            pts = pts @ rpy_matrix(rpy).T + xyz
            out.append((link.get("name"), pts[:: max(1, len(pts) // 150)]))
    return out


def arm_points(side: str, q6, grip: float) -> list:
    """World points (one array per collision piece) of one arm at joints ``q6`` and gripper opening ``grip``
    (1 open .. 0 closed)."""
    from yam_arena import ik, station
    from yam_arena.constants import FINGER_OPEN

    q = {f"joint{i + 1}": float(v) for i, v in enumerate(q6)}
    q["joint7"] = q["joint8"] = FINGER_OPEN * float(grip)
    poses = ik.forward_kinematics(ik.YAM_URDF, q)
    base = np.asarray(station.BASES[side], float)
    out = []
    for name, pts in _pieces():
        t = poses[name]
        out.append(pts @ t[:3, :3].T + t[:3, 3] + base)
    return out


def _bounds(parts):
    return np.array([p.min(0) for p in parts]), np.array([p.max(0) for p in parts])


def _gap(a: list, b: list) -> float:
    """Smallest point-to-point distance between two lists of point sets (pruned by their bounding boxes)."""
    lo_a, hi_a = _bounds(a)
    lo_b, hi_b = _bounds(b)
    gap = np.max(np.maximum(lo_a[:, None, :] - hi_b[None, :, :], lo_b[None, :, :] - hi_a[:, None, :]), axis=2)
    gap = np.maximum(gap, 0.0)
    best = 9.0
    for i, j in zip(*np.nonzero(gap < 0.25)):
        if gap[i, j] >= best:
            continue
        d = np.linalg.norm(a[i][:, None, :] - b[j][None, :, :], axis=2)
        best = min(best, float(d.min()))
    return best


def cylinder_points(centre, radius, height, n=24) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(n)], 1)
    pts = [ring * s + [0, 0, z] for s in (0.0, 0.5, 1.0) for z in (-height / 2, height / 2)]
    return np.concatenate(pts) + np.asarray(centre, float)


def box_points(centre, size) -> np.ndarray:
    c = np.asarray(centre, float)
    h = np.asarray(size, float) / 2
    g = np.linspace(-1, 1, 5)
    return np.array([c + h * [x, y, z] for x in g for y in g for z in g])


def clearance(left_pts, right_pts, chest_pts=None, pump_pts=None) -> tuple[float, str]:
    """Smallest gap (m) and which pair: arm-arm, right arm to the chestpiece the left arm holds, left arm to the pump
    unit the right arm works. ``chest_pts``/``pump_pts`` are lists of point sets."""
    cands = [(_gap(left_pts, right_pts), "arm-arm")]
    if chest_pts is not None:
        cands.append((_gap(right_pts, chest_pts), "right arm - held chestpiece"))
    if pump_pts is not None:
        cands.append((_gap(left_pts, pump_pts), "left arm - pump unit"))
    return min(cands)


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def tcp_actual(self, side):
        from yam_arena import ik, station

        rob = self.emb._env.unwrapped.scene[f"{side}_robot"]
        q = np.asarray(getattr(rob.data.joint_pos, "torch", rob.data.joint_pos)[0].detach().cpu(), float)[:6]
        return ik.tcp_pose(q, station.BASES[side])[0]

    def clearance(self, T, chest_held):
        """Smallest gap (m) between the two arms' collision meshes, and from the right arm to the chestpiece the left arm
        holds and from the left arm to the pump unit; commanded joints (logging only)."""
        C = sys.modules[__name__]

        lp = C.arm_points("left", self.left.q, self.left.grip)
        rp = C.arm_points("right", self.right.q, self.right.grip)
        chest = None
        if chest_held:
            o = self.left.pos - np.array([0.0, 0.0, T.STEM_GRIP_Z])
            chest = [C.cylinder_points(o + [0, 0, 0.013], 0.022, 0.014), C.cylinder_points(o + [0, 0, 0.035], 0.014, 0.03)]
        pump = [C.box_points((T.PUMP_XY[0], T.PUMP_XY[1], T.TOP + 0.006), (0.09, 0.16, 0.012)),
                C.cylinder_points((T.PUMP_XY[0], T.PUMP_XY[1] + T.BULB_DY, T.TOP + 0.047), 0.025, 0.07),
                C.box_points((T.PUMP_XY[0], T.PUMP_XY[1] + T.KNOB_DY, T.TOP + T.KNOB_Z), (0.085, 0.026, 0.016))]
        return C.clearance(lp, rp, chest, pump)

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
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    gen = script(task, ctx)
    frames: dict[str, list] = {}
    debug = bool(os.environ.get("DEMO_DEBUG"))
    trace: list = []
    closes: list = []

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    lag_max = {"left": 0.0, "right": 0.0}
    clear = {"mm": None, "step": None, "pair": None}
    variant = os.environ.get("DEMO_VARIANT", "")
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        lag = {}
        for side, arm in (("left", ctx.left), ("right", ctx.right)):
            lag[side] = round(1000 * float(np.linalg.norm(ctx.tcp_actual(side) - arm.pos)), 1)
            lag_max[side] = max(lag_max[side], lag[side])
        if steps % 5 == 0:                           # inter-arm clearance from the collision meshes (commanded joints)
            try:
                held = variant != "one_arm" and ctx.left.grip < 0.5 and ctx.left.pos[2] > 0.0
                gap, pair = ctx.clearance(task, held)
                if clear["mm"] is None or 1000 * gap < clear["mm"]:
                    clear.update(mm=round(1000 * gap, 1), step=steps, pair=pair)
            except Exception as exc:                  # logging only
                clear["error"] = str(exc)[:80]
        while EVENTS:                                # fingertips versus the grip feature at each close
            name, side, feat = EVENTS.pop(0)
            closes.append({"event": name, "step": steps, "tcp_minus_feature_mm": np.round(
                1000 * (ctx.tcp_actual(side) - feat), 1).tolist()})
            print("CLOSE", closes[-1], flush=True)
        if steps % 100 == 0 or (debug and steps % 10 == 0):
            msg = f"step {steps} score {info.get('score')}"
            if debug:
                row = dict(task._LOG)
                task._STATE["win_sq"] = 0.0
                row.update(step=steps, lag_mm=lag, min_inter_arm_mm=clear["mm"], min_inter_arm_at=[clear["step"], clear["pair"]])
                row["t_phase"] = None if row.get("t_phase") is None else round(row["t_phase"], 2)
                row["rate"] = None if row.get("rate") is None else round(row["rate"], 2)
                trace.append(row)
                msg += " DEBUG " + json.dumps(row)
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    S = task._STATE
    result = {"task_score": score, "success": success, "steps": steps,
              "reading_rate_mmhg_s": S["rate"], "flags": sorted(S["flags"]), "peak_mmhg": round(S["peak"], 1),
              "final_mmhg": round(S["p"], 1), "squeezes": S["squeezes"],
              "lag_max_mm": lag_max, "closes": closes, "min_inter_arm": clear,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if debug:
        result["debug_trace"] = trace
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "debug_trace"}, default=str), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
