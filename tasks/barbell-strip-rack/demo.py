"""Scripted full-score demo for the barbell-strip task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Open loop from the fixed layout: each arm comes down over the top rim of its own plate (fingers open 45 mm, 10 mm clear
of each face), closes, lifts 10 mm so the bore floats on the sleeve, and the two slide their plates outward along the
bar at 2.5 cm/s until the plates clear the sleeve ends. They lift, carry the plates to their holders, stand them in the
slots, open and lift away. Both arms run in lockstep; no object pose is read.

Negative variants (DEMO_VARIANT=...):
  one_arm    only the left arm strips its plate; the right plate stays on the bar: it tips (0)
  late_right the right arm starts 10.5 s late, so the left plate is off first: the bar tips (0)
  on_table   both plates are set on the table beside the holders, not in them (0.5)
  no_release both plates go in the holders but the hands do not open (0.75)
  skew       the right arm runs 0.6 s behind the left throughout: tolerated, scores 1.0 like the plain demo
DEMO_DEBUG=1 prints a trace every 10 steps: per-arm lag (commanded vs actual TCP), each plate's position along the bar,
the bar ends' and hooks' lift, and the minimum distance between one arm (with the plate it holds) and the other arm.
Writes one MP4 per camera, setup.png and result.json to the output folder.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
VARIANTS = ("", "one_arm", "late_right", "on_table", "no_release", "skew")

RACK_X, PLATE_Y = 0.28, 0.1175
HOLDER_Y = 0.31
GRIP_ABOVE_CENTRE = 0.058        # the TCP sits this far above a held plate's centre (pad tips 13 mm below it)
PLATE_R = 0.07
PLATE_CENTRE = 0.090             # a hanging plate's centre above the table
OPEN_45MM = 0.48                 # grip value: pads about 45 mm apart (94 mm at 1.0)
LIFT = 0.010
SLIDE = 0.110
SLIDE_SPEED = 0.025
HOLDER_FLOOR = 0.010


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
        self.stage, self.placed_at = 0, None   # 0 plate on the bar, 1 holding it, 2 placed

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


def reachable_yaw(arm, pos, yaw):
    from yam_arena import ik
    from yam_arena.demo import grip_rotation

    best = None
    for y in (yaw, yaw + math.pi, yaw - math.pi):
        _, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(grip_rotation(y)), arm.q, base_pos=arm.base, iters=200)
        if best is None or pe + 0.05 * re < best[0]:
            best = (pe + 0.05 * re, y)
    return best[1]


def strip(arm, side, T, variant, delay=0.0):
    """One arm strips its plate (side +1 left, -1 right): grip, float, slide off, carry to the holder, stand it, let go."""
    from yam_arena.demo import grip_rotation

    top = T.TOP
    tcp_z = top + PLATE_CENTRE + GRIP_ABOVE_CENTRE
    up = np.array([0.0, 0.0, 0.06])
    grip = np.array([RACK_X, side * PLATE_Y, tcp_z])
    rot = grip_rotation(reachable_yaw(arm, grip, 0.0))
    if delay:
        yield from wait(delay)
    # 1. above the plate's top rim, fingers open 45 mm, then down and close.
    yield from arm.move(grip + up, rot, 2.5, grip=OPEN_45MM)
    yield from arm.move(grip, rot, 1.5)
    yield from arm.set_grip(0.0, 1.0)
    arm.stage = 1
    yield from wait(0.5)
    # 2. lift 10 mm (the bore floats on the sleeve), slide outward until the plate is clear of the sleeve end.
    p = grip + np.array([0.0, 0.0, LIFT])
    yield from arm.move(p, rot, 1.0)
    yield from wait(0.5)
    p = p + np.array([0.0, side * SLIDE, 0.0])
    yield from arm.move(p, rot, SLIDE / SLIDE_SPEED)
    # 3. lift 4 cm, out to the holder (or to the table beside it), lower, let go, lift away.
    p = p + np.array([0.0, 0.0, 0.04])
    yield from arm.move(p, rot, 1.0)
    y_set = side * (0.40 if variant == "on_table" else HOLDER_Y)
    floor = 0.0 if variant == "on_table" else HOLDER_FLOOR
    p = np.array([RACK_X, y_set, p[2]])
    yield from arm.move(p, rot, 2.5)
    low = np.array([RACK_X, y_set, top + floor + PLATE_R + GRIP_ABOVE_CENTRE - 0.001])
    yield from arm.move(low, rot, 2.0)
    yield from wait(0.5)
    arm.placed_at = np.array([RACK_X, y_set, low[2] - GRIP_ABOVE_CENTRE])
    if variant == "no_release":
        arm.stage = 2
        yield from wait(3.0)
        return
    yield from arm.set_grip(1.0, 0.8)
    arm.stage = 2
    yield from arm.move(low + np.array([0.0, 0.0, 0.07]), rot, 1.5)
    yield from wait(1.0)


def script(task, ctx):
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    L, R = ctx.left, ctx.right
    left = strip(L, 1.0, task, variant)
    if variant == "one_arm":
        yield from left
        return
    delay = {"late_right": 10.5, "skew": 0.6}.get(variant, 0.0)
    yield from both(left, strip(R, -1.0, task, variant, delay))


# -- clearance: every link hull of one arm against the other arm and the plate it holds --------------------------------

class Clearance:
    """Minimum signed distance (m) between the left and right arms' collision hulls, and between each arm and the plate
    the other one holds, along the commanded path. Tracks where the minimum occurs."""

    def __init__(self, task_top: float):
        from yam_arena import analysis as A

        A.link_hulls = functools.lru_cache(None)(A.link_hulls)
        self.A, self.top = A, task_top
        self.min, self.at, self.pair, self.step = math.inf, None, None, 0

    def _hulls(self, cmd, base):
        from yam_arena.constants import ARM_JOINTS, FINGER_OPEN

        A = self.A
        q = {n: float(v) for n, v in zip(ARM_JOINTS, cmd[:6])}
        f = float(cmd[6]) * FINGER_OPEN
        q["joint7"] = f
        q["joint8"] = f
        return A.world_hulls(A.fk_link_poses_world(q, base))

    @staticmethod
    def _disc(centre, n=48):
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return np.c_[centre[0] + PLATE_R * np.cos(a), np.full(n, centre[1]), centre[2] + PLATE_R * np.sin(a)]

    def _plate_pts(self, arm, side):
        if arm.stage == 1:
            c = arm.pos - np.array([0.0, 0.0, GRIP_ABOVE_CENTRE])
        elif arm.stage == 2:
            c = arm.placed_at
        else:
            c = np.array([RACK_X, side * PLATE_Y, self.top + PLATE_CENTRE])
        return self._disc(c)

    def update(self, left, right) -> float:
        from yam_arena import station

        A = self.A
        self.step += 1
        hl = self._hulls(left.cmd, station.BASES["left"])
        hr = self._hulls(right.cmd, station.BASES["right"])
        best, pair = math.inf, None
        box_l = [(h.vertices.min(0), h.vertices.max(0)) for _, h in hl]
        box_r = [(h.vertices.min(0), h.vertices.max(0)) for _, h in hr]
        for (na, a), (alo, ahi) in zip(hl, box_l):
            for (nb, b), (blo, bhi) in zip(hr, box_r):
                if np.maximum(blo - ahi, alo - bhi).max() > 0.06:
                    continue
                s = min(A._signed(a.vertices, b).min(), A._signed(b.vertices, a).min())
                if s < best:
                    best, pair = float(s), (f"L.{na}", f"R.{nb}")
        for pts, hulls, tag in ((self._plate_pts(left, 1.0), hr, "L plate vs R."), (self._plate_pts(right, -1.0), hl,
                                                                                    "R plate vs L.")):
            for nb, b in hulls:
                s = float(A._signed(pts, b).min())
                if s < best:
                    best, pair = s, (tag + nb, "")
        if best < self.min:
            self.min, self.at, self.pair = best, self.step, pair
        return best

    def summary(self) -> dict:
        return {"min_clearance_mm": round(1000 * self.min, 1), "min_clearance_step": self.at,
                "min_clearance_pair": list(self.pair) if self.pair else None}


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])



def _lags(ctx) -> dict:
    """Debug only: distance (mm) from each arm's commanded TCP to where its gripper actually is."""
    from yam_arena.constants import TCP_OFFSET
    from yam_arena.scene import object_states  # noqa: F401 - same import context as the states reader
    from partkit import PartState, _np

    base = ctx.emb._env.unwrapped
    origin = _np(base.scene.env_origins[0])
    out = {}
    for name, arm in (("L", ctx.left), ("R", ctx.right)):
        robot = base.scene[f"{'left' if name == 'L' else 'right'}_robot"]
        pose = _np(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")]
        rot = PartState(pos=pose[:3], quat_xyzw=pose[3:7], lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.zeros(3),
                        high=np.zeros(3)).rotation()
        tcp = pose[:3] + rot @ np.array(TCP_OFFSET) - origin
        out[name] = round(1000 * float(np.linalg.norm(tcp - arm.pos)), 1)
    return out


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 4.0) -> dict:
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
    clearance = Clearance(task.TOP)
    frames: dict[str, list] = {}
    debug = bool(os.environ.get("DEMO_DEBUG"))
    max_lag = {"L": 0.0, "R": 0.0}
    trace: list = []
    bar = {"max_tilt_deg": 0.0, "max_end_lift_mm": {"end_l": 0.0, "end_r": 0.0}, "max_hook_lift_mm": {"hook_l": 0.0, "hook_r": 0.0},
           "max_plate_y_mm": {"plate_l": 0.0, "plate_r": 0.0}}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

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
        try:
            clearance.update(ctx.left, ctx.right)
        except Exception as exc:  # noqa: BLE001 - logging only
            if steps == 1:
                print(f"clearance unavailable: {exc!r}", flush=True)
        try:
            lag = _lags(ctx)
            for k in lag:
                max_lag[k] = max(max_lag[k], lag[k])
        except Exception as exc:  # noqa: BLE001 - logging only
            lag = {"err": repr(exc)}
            max_lag["err"] = repr(exc)
        try:
            st = ctx.states()
            row_t = task.plate_trace(st)
            tilt = math.degrees(math.asin(max(-1.0, min(1.0, float(st["bar"].rotation()[2, 1])))))
            bar["max_tilt_deg"] = max(bar["max_tilt_deg"], abs(tilt))
            for k in bar["max_end_lift_mm"]:
                bar["max_end_lift_mm"][k] = max(bar["max_end_lift_mm"][k], row_t.get(f"{k}_dz_mm", 0.0))
            for k in bar["max_hook_lift_mm"]:
                bar["max_hook_lift_mm"][k] = max(bar["max_hook_lift_mm"][k], row_t.get(f"{k}_dz_mm", 0.0))
            for k in bar["max_plate_y_mm"]:
                bar["max_plate_y_mm"][k] = max(bar["max_plate_y_mm"][k], abs(row_t[f"{k}_y_mm"]))
        except Exception as exc:  # noqa: BLE001
            row_t = {"err": repr(exc)}
        if steps % 100 == 0 or (debug and steps % 10 == 0):
            msg = f"step {steps} score {info.get('score')}"
            if debug:
                row = {"step": steps, "lag_mm": lag, **row_t,
                       "clearance_mm": round(1000 * clearance.min, 1), "clearance_at": clearance.at}
                trace.append(row)
                msg += " TRACE " + json.dumps(row)
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "variant": os.environ.get("DEMO_VARIANT", ""),
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "max_lag_mm": max_lag, **clearance.summary()}
    try:
        st = ctx.states()
        result["bar"] = {**bar, "final_pos_mm": [round(1000 * float(v), 1) for v in st["bar"].pos],
                         "final_quat_xyzw": [round(float(v), 4) for v in st["bar"].quat_xyzw],
                         "final_tilt_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, float(st["bar"].rotation()[2, 1]))))), 2)}
        result["flags"] = {"tipped": task._STATE["tipped"], "dropped": task._STATE["dropped"], "why": list(task._STATE["why"]),
                           "fired_at_scorer_call": task._STATE["fired_at"]}
    except Exception as exc:  # noqa: BLE001
        result["bar_error"] = repr(exc)
    if debug:
        result["trace"] = trace
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
