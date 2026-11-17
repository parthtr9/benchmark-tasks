"""Scripted full-score demo for juicer-arm-lock, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Open loop: every target comes from the task file's fixed layout and the clock. The only feedback is each arm's own
joint readings (a trim that keeps the measured tool point on the plan under load). No object pose or task state
steers it.

1. The left hand takes the cover by its feed chute, carries it over the juicer and lowers it onto the pulp container
   (its skirt's slots over the container's wings), lets go and moves away.
2. The right hand, turned a quarter turn, takes the locking arm's handle bar between its pads, swings it up along
   its arc and over the cover, pushes the wire over the two ridges into the grooves, lets go and moves away.

Negative variants (DEMO_VARIANT=...):
  cover_only    cover seated, locking arm left down (0.5)
  arm_only      no cover; the arm is raised and let go (0: it falls back)
  basket_high   the basket starts not clicked down; cover placed as usual, arm raised (at most 0.25: the cover rides
                high on the basket and its ribs stop the arm short of the grooves)
  turned        the cover starts a quarter turn out and is set down that way, on the wings; arm raised (0.25)
  short         the arm is raised to 84 degrees, short of the ridges, and let go (0.5)
  held          everything done but the right hand never lets go of the handle (0.75)
Writes one MP4 per camera, setup.png and result.json; DEMO_DEBUG=1 prints a trace every 10 steps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
VARIANTS = ("", "cover_only", "arm_only", "basket_high", "turned", "short", "held")
KI, CORR_MAX = 0.2, 0.02        # joint-feedback trim: gain per step and limit (m)
CARRY_CORR_MAX = 0.035          # trim limit while the left hand carries the cover (the arm sags under its 0.25 kg)
PUSH_CORR_MAX = 0.02            # trim limit (forward only) while the right hand pushes the arm's wire over the ridges


def fwd(yaw: float = 0.0, pitch: float = 0.0, roll: bool = False) -> np.ndarray:
    """A forward-pointing grip (columns: closing axis, the pads' width axis, approach axis), turned ``yaw`` about
    the vertical and pitched ``pitch`` (negative: nose down). ``roll``: turned a quarter turn about its approach
    axis, so the pads are above and below."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    a = np.array([cy * cp, sy * cp, sp])
    c = np.array([-cy * sp, -sy * sp, cp]) if roll else np.array([-sy, cy, 0.0])
    return np.c_[c, np.cross(a, c), a]


FLIP = np.diag([-1.0, -1.0, 1.0])       # the same parallel grip turned half a turn about its approach axis


def best_grip(arm, positions, rot):
    """The grip (rot or rot turned half a turn about its approach) whose worst IK error over ``positions``, solved one
    after another from the arm's last command, is smaller, with the wrist roll kept off its +-120 degree limit."""
    from yam_arena import ik

    positions = [positions] if np.ndim(positions) == 1 else positions
    out = []
    for R in (rot, rot @ FLIP):
        q, worst = arm.q.copy(), 0.0
        for pos in positions:
            q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(R), q, base_pos=arm.base, iters=200)
            worst = max(worst, pe + 0.05 * re + 0.01 * max(0.0, abs(float(q[5])) - 1.9))
        out.append((worst, R))
    return min(out, key=lambda t: t[0])[1]


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


def shift(arm, d, rot, duration: float, grip: float | None = None, servo=None):
    """Generator: move by ``d`` from wherever the arm's command is when this step starts (not when it was built)."""
    yield from arm.move(arm.pos + np.asarray(d, float), rot, duration, grip, servo)


def stay(arm, rot, duration: float, grip: float | None = None):
    """Generator: stay where the arm is when this step starts, turning to ``rot`` and setting the grip."""
    yield from arm.move(arm.pos, rot, duration, grip)


# -- the plan ------------------------------------------------------------------------------------------------------------
CHUTE_GRIP = 0.118               # the left hand's tool point above the cover's skirt edge (pads on the chute's top third)
CARRY_UP = 0.031                 # the cover is carried this far above its seated height (wings and wall top cleared)
SEAT_UP = 0.002                  # and let go this far above its seat
G_HANDLE = 0.25                  # finger opening (1 = 94 mm) when the right hand comes onto the handle bar
BAR_AHEAD = 0.006                # the handle bar sits this far ahead of the tool point between the pads
HANDLE_Y = -0.020                # where on the handle bar, from the juicer's middle
G_LET_GO = 0.14                  # finger opening to let the handle bar go: 3.5 mm a side, so the lower finger stays off the cover
CLAMP_PITCH = math.radians(-25.0)  # nose down at the lowered handle, level from 75 degrees of raise
PUSH_DEG = 90.5                  # the arm is swung to here (where the grooves' stops hold it) ...
PUSH_PAST = 0.012                # ... and the handle is then pushed this much further forward, level, against those stops
SHORT_DEG = 84.0


def clamp_pose(T, alpha_deg: float):
    """The right hand's tool point and rotation with the handle bar between its pads at ``alpha_deg`` of raise."""
    pitch = CLAMP_PITCH * max(0.0, 1.0 - (alpha_deg - T.REST_DEG) / 70.0)
    R = fwd(0.0, pitch, roll=True)
    bar = np.array(T.JUICER_POS) + T.arm_point(alpha_deg, T.HANDLE_S, 0.0) / 1000 + np.array([0.0, HANDLE_Y, 0.0])
    return bar - BAR_AHEAD * R[:, 2], R


def plan(T) -> dict:
    """Tool-point targets (world, m) from the task file's fixed layout."""
    p = {}
    J = np.array(T.JUICER_POS)
    c0 = np.array(T.COVER_START)
    p["cover_pick"] = c0 + np.array([0.0, 0.0, CHUTE_GRIP])
    seat = J + np.array([0.0, 0.0, T.COVER_SEAT / 1000 + CHUTE_GRIP])
    p["cover_over"] = seat + np.array([0.0, 0.0, CARRY_UP])
    p["cover_seat"] = seat + np.array([0.0, 0.0, SEAT_UP])
    p["handle_rest"], _ = clamp_pose(T, T.REST_DEG)
    p["handle_up"], _ = clamp_pose(T, 90.0)
    return p


def script(task, ctx):
    T, P = task, plan(task)
    variant = os.environ.get("DEMO_VARIANT", "")
    L, R = ctx.left, ctx.right
    up = np.array([0.0, 0.0, 1.0])
    # 1. the left hand takes the cover by its chute and sets it on the container; the right hand goes to the handle
    c_rot = best_grip(L, P["cover_seat"], fwd())
    h0, hr0 = clamp_pose(T, T.REST_DEG)
    h_rot_flip = best_grip(R, h0, hr0)
    flip = not np.allclose(h_rot_flip, hr0)

    def hpose(alpha):
        p, rot = clamp_pose(T, alpha)
        return p, (rot @ FLIP if flip else rot)

    pre_p, pre_r = hpose(T.REST_DEG)
    # round the juicer's back, not through it: out to the right and back first, then down to the lowered handle
    clear = np.array([0.20, pre_p[1] - 0.14, T.TOP + 0.30])
    to_handle = chain(R.move(clear, pre_r, 2.0, G_HANDLE), R.move(pre_p - 0.06 * pre_r[:, 2] + 0.03 * up, pre_r, 2.0),
                      R.move(pre_p - 0.06 * pre_r[:, 2], pre_r, 0.8), R.move(pre_p, pre_r, 1.2))
    if variant != "arm_only":
        servo = np.ones(3)
        L.corr_max = CARRY_CORR_MAX
        seat = P["cover_seat"]
        cover = chain(L.move(P["cover_pick"] + 0.11 * up, c_rot, 2.5, 1.0), L.move(P["cover_pick"], c_rot, 1.5),
                      L.move(P["cover_pick"], c_rot, 0.6, 0.0), L.move(P["cover_pick"] + 0.03 * up, c_rot, 0.8, servo=servo),
                      L.move(np.array([P["cover_pick"][0], P["cover_pick"][1], P["cover_over"][2]]), c_rot, 1.5, servo=servo),
                      L.move(P["cover_over"], c_rot, 3.0, servo=servo), L.hold(0.3, servo=servo),
                      L.move(seat, c_rot, 1.8, servo=servo), L.hold(0.3),
                      L.move(seat, c_rot, 0.6, 1.0), L.relax(0.3),
                      L.move(seat - np.array([0.11, 0.0, 0.0]), c_rot, 1.0),
                      L.move(np.array([0.22, 0.24, T.TOP + 0.33]), c_rot, 1.5))
        yield from both(cover, to_handle)
        L.corr_max = CORR_MAX
    else:
        yield from both(L.move(np.array([0.22, 0.24, T.TOP + 0.30]), c_rot, 2.0), to_handle)
    if variant == "cover_only":
        yield from chain(R.move(pre_p - 0.06 * pre_r[:, 2] + 0.03 * up, pre_r, 1.0), wait(2.0))
        return
    # 2. the right hand closes on the handle bar and swings the arm up its arc into the grooves
    yield from R.move(pre_p, pre_r, 0.6, 0.0)
    end = SHORT_DEG if variant == "short" else PUSH_DEG
    servo = np.array([1.0, 0.0, 1.0])
    alpha, step = T.REST_DEG, 5.0
    while alpha < end - 1e-6:
        nxt = min(alpha + step, end)
        p, rot = hpose(nxt)
        if nxt > 84.0:                     # the last degrees, over the ridges: push forward only, with a small trim
            servo, R.corr_max = np.array([1.0, 0.0, 0.0]), PUSH_CORR_MAX
        yield from R.move(p, rot, 0.22 * (nxt - alpha) / step + (0.4 if nxt > 84.0 else 0.0), servo=servo)
        alpha = nxt
    if variant != "short":
        yield from R.move(R.pos + np.array([PUSH_PAST, 0.0, 0.0]), R.rot, 0.8, servo=servo)
    yield from R.hold(0.5, servo=servo)
    R.corr_max = CORR_MAX
    if variant == "held":
        yield from wait(3.0)
        return
    # let go: open just clear of the bar, draw straight back off it at the same height, then open and move away
    p, rot = hpose(end)
    back = R.pos - 0.045 * rot[:, 2]
    park = np.array([max(back[0] - 0.05, 0.20), back[1], back[2] + 0.05])
    yield from chain(R.relax(0.3), R.move(R.pos, rot, 0.5, G_LET_GO), R.move(back, rot, 1.0),
                     R.move(back, rot, 0.4, 0.6), R.move(park, rot, 1.0))
    yield from wait(1.5)


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
    if variant == "turned":
        os.environ["JUICER_COVER_TURNED"] = "1"
    if variant == "basket_high":
        os.environ["JUICER_BASKET_HIGH"] = "1"
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
    import clearance

    print("PLAN " + json.dumps({k: np.round(v - [0, 0, task.TOP], 4).tolist() for k, v in plan(task).items()}), flush=True)

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    last_gap = {"mm": None}

    def row(steps, info):
        """Logging only: reads simulator state for the trace."""
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)
        g = task.geometry(st)
        S = task._STATE
        r2 = lambda v: [round(float(x), 2) for x in v]  # noqa: E731
        return {"step": steps, "score": info.get("score"), "levels": S["levels"], "released": S["released"],
                "near_mm": round(float(S.get("near_mm", -1.0)), 1), "gap_mm": last_gap["mm"],
                "basket_up": round(g["basket_up_mm"], 2), "basket_tilt_deg": round(math.degrees(g["basket_tilt"]), 2),
                "catch": r2(g["catch_mm"]), "cover_gap": r2([g["cover_gap_min_mm"], g["cover_gap_max_mm"]]),
                "cover_off": round(g["cover_off_mm"], 2), "arm_deg": round(g["arm_deg"], 2),
                "wire_x": r2(g["wire_x_mm"]), "wire_lift": r2(g["wire_lift_mm"]), "ridge": r2(g["ridge_mm"]),
                "flags": [S["moved"], S["dropped"], S["blade"]],
                "corr_mm": {"L": np.round(1000 * ctx.left.corr, 1).tolist(),
                            "R": np.round(1000 * ctx.right.corr, 1).tolist()},
                "w": {n: np.round(st[n].pos - [0, 0, task.TOP], 4).tolist() for n in ("juicer", "basket", "cover")},
                "tcp": {"L_cmd": np.round(ctx.left.pos - [0, 0, task.TOP], 3).tolist(),
                        "L_meas": np.round(ctx.joint_tcp("left")[0] - [0, 0, task.TOP], 3).tolist(),
                        "R_cmd": np.round(ctx.right.pos - [0, 0, task.TOP], 3).tolist(),
                        "R_meas": np.round(ctx.joint_tcp("right")[0] - [0, 0, task.TOP], 3).tolist(),
                        "grip": [round(float(ctx.left.grip), 2), round(float(ctx.right.grip), 2)]}}

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    if debug:
        try:
            print("TRACE " + json.dumps(row(0, {}), default=str), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"trace unavailable: {exc!r}", flush=True)
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
                gap, _ = clearance.min_clearance(ctx.left.q, ctx.left.grip, ctx.left.base, ctx.right.q,
                                                 ctx.right.grip, ctx.right.base)
                min_gap = min(min_gap, gap)
                last_gap["mm"] = round(1000 * gap, 1)
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
              "levels": S["levels"], "base_moved": S["moved"], "part_dropped": S["dropped"],
              "reached_into_basket": S["blade"], "finger_check": S["finger_check"]}
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


def ik_check(task_path: str) -> None:
    """No simulator: yam_arena IK error along the whole planned path, solved step by step as the demo does."""
    import importlib

    from yam_arena import station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)

    class Dry:
        left, right = Arm("left", station.BASES["left"]), Arm("right", station.BASES["right"])

    ctx = Dry()
    n = 0
    for _ in script(task, ctx):
        n += 1
        if n % 20 == 0:
            print(f"IK step {n:4d} L {np.round(ctx.left.pos - [0, 0, task.TOP], 3).tolist()} worst {ctx.left.worst_mm:.1f} mm | "
                  f"R {np.round(ctx.right.pos - [0, 0, task.TOP], 3).tolist()} worst {ctx.right.worst_mm:.1f} mm", flush=True)
    print(f"IK steps {n} worst left {ctx.left.worst_mm:.2f} mm right {ctx.right.worst_mm:.2f} mm", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ik", action="store_true", help="only print the IK error of every planned target")
    a = ap.parse_args()
    if a.ik:
        ik_check(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
