"""Scripted full-score demo for the crosscut-saw task, run in the yam_arena embodiment (open loop).

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Both arms close on their handle (fingers across x). Eight strokes: the left arm pulls the blade 10 cm toward +y while
the right arm follows 0.1 s behind with its link at the middle of its play (so its link never pushes), then the right
arm pulls it 20 cm back with the left following, and so on. Each leader first takes up its slack during the pause. Then
both open and lift away. All hand paths come from the fixed layout and timing; nothing is read from the simulation.

Negative variants (DEMO_VARIANT=...):
  one_arm       the left arm pulls one stroke and then pushes the blade back (the right arm never touches its handle): 0
  one_arm_pull  the left arm pulls once and stops (right arm idle): 0
  anchor        the right arm holds its handle still while the left pulls: the blade cannot travel 8 cm: 0
  push_stroke   the follower leads, its link pushed in to the pushing end: kink: 0
  short         four strokes: 0.25
  no_release    eight strokes but both hands stay closed on the handles: 0.75
DEMO_DEBUG=1 prints (and result.json keeps) a trace: per-arm lag, both link gaps, blade position, strokes, kink,
smallest link margin and the smallest arm-to-arm distance from the URDF collision hulls.
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


STROKE_S = 2.0          # one stroke
PAUSE_S = 0.5           # slack swap between strokes
FOLLOW_S = 0.1          # the follower trails the blade by this long
GRIP_Z = 0.155          # TCP height above the table top (handle spans 0.11 .. 0.19)
VARIANTS = ("", "one_arm", "one_arm_pull", "anchor", "push_stroke", "short", "no_release")


def _smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def hand_plan(T, variant: str = "") -> list:
    """(y_left, y_right) of the two hands for every control step of the stroking phase, from the layout alone."""
    E, G0, GMAX, GMIN = T.BLADE_E, T.GAP_0, T.GAP_MAX, T.GAP_MIN
    n_strokes = 4 if variant == "short" else 1 if variant == "one_arm_pull" else 2 if variant == "one_arm" else 8
    if variant == "anchor":
        n_strokes = 1
    plan = []
    b = 0.0
    gl, gr = G0, G0                                   # current commanded gaps (handle centre to blade end)
    hold = lambda n, yl, yr: plan.extend([(yl, yr)] * max(1, round(n * CONTROL_HZ)))
    for k in range(n_strokes):
        left_leads = k % 2 == 0
        b1 = T.BLADE_MAX if left_leads else -T.BLADE_MAX
        if variant == "one_arm" and k == 1:
            b1 = 0.0                                  # the left arm pushes the blade back toward its start
        # pause: the leader takes up its slack (to just past taut), the follower eases to the middle of its play
        lead_g, fol_g = GMAX + 0.004, G0
        if variant == "push_stroke":
            fol_g = GMIN - 0.010
        g_l0, g_r0 = gl, gr
        goal_l, goal_r = (lead_g, fol_g) if left_leads else (fol_g, lead_g)
        if variant == "one_arm" and k == 1:
            goal_l, goal_r = GMIN - 0.010, gr         # pushing: the left link goes in to its pushing end
        n = round(PAUSE_S * CONTROL_HZ)
        for i in range(1, n + 1):
            s = _smooth(i / n)
            plan.append((b + E + g_l0 + s * (goal_l - g_l0), b - E - (g_r0 + s * (goal_r - g_r0))))
        gl, gr = goal_l, goal_r
        # stroke: the follower's command is the leader's, FOLLOW_S later
        b0 = b
        B = lambda t: b0 + (b1 - b0) * _smooth(t / STROKE_S)
        n = round((STROKE_S + FOLLOW_S) * CONTROL_HZ)
        for i in range(1, n + 1):
            t = i / CONTROL_HZ
            lead, fol = B(t), (B(t) if variant == "push_stroke" else B(t - FOLLOW_S))
            if left_leads or (variant == "one_arm" and k == 1):
                plan.append((lead + E + gl, fol - E - gr))
            else:
                plan.append((fol + E + gl, lead - E - gr))
        b = b1
    hold(0.5, *plan[-1])
    return plan


def hand_pos(T, y):
    return np.array([T.LOG_X, y, T.TOP + GRIP_Z])


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    T = task
    L, R = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    up = np.array([0.0, 0.0, 0.05])
    right_idle = variant in ("one_arm", "one_arm_pull")
    yl0, yr0 = T.HANDLE_Y, -T.HANDLE_Y
    yawL = reachable_yaw(L, hand_pos(T, yl0), math.pi / 2)
    yawR = reachable_yaw(R, hand_pos(T, yr0), math.pi / 2)
    rotL, rotR = grip_rotation(yawL), grip_rotation(yawR)

    # 1. Over both handles, down onto them, close. Fingers straddle each post across x (94 mm open, post 30 mm).
    reach = [L.move(hand_pos(T, yl0) + up, rotL, 3.0, grip=1.0)]
    if not right_idle:
        reach.append(R.move(hand_pos(T, yr0) + up, rotR, 3.0, grip=1.0))
    yield from both(*reach)
    down = [L.move(hand_pos(T, yl0), rotL, 1.5)]
    if not right_idle:
        down.append(R.move(hand_pos(T, yr0), rotR, 1.5))
    yield from both(*down)
    close = [L.set_grip(0.0, 1.0)]
    if not right_idle:
        close.append(R.set_grip(0.0, 1.0))
    yield from both(*close)
    yield from wait(0.5)
    ctx.mark("strokes")

    # 2. Strokes.
    for yl, yr in hand_plan(T, variant):
        L.aim(hand_pos(T, yl), rotL)
        if not right_idle and variant != "anchor":
            R.aim(hand_pos(T, yr), rotR)
        yield
    ctx.mark("strokes_end")

    # 3. Let go and lift away.
    if variant != "no_release":
        opens = [L.set_grip(1.0, 0.6)]
        if not right_idle:
            opens.append(R.set_grip(1.0, 0.6))
        yield from both(*opens)
        lifts = [L.move(L.pos + np.array([0.0, 0.0, 0.05]), L.rot, 1.0)]
        if not right_idle:
            lifts.append(R.move(R.pos + np.array([0.0, 0.0, 0.05]), R.rot, 1.0))
        yield from both(*lifts)
    yield from wait(1.5)


# -- offline / logged arm clearance ----------------------------------------------------------------------------

def arm_clearance(steps, T) -> dict:
    """Smallest distance (m) between any collision hull of one arm and the other arm or the handle / blade it holds, over
    the commanded path. ``steps`` is a list of (step, left cmd, right cmd, blade offset). Needs yam_arena.analysis."""
    from yam_arena import analysis as A
    from yam_arena import station
    from yam_arena.constants import ARM_JOINTS, FINGER_OPEN

    def hulls(cmd, base):
        q = {n: float(v) for n, v in zip(ARM_JOINTS, cmd[:6])}
        f = float(cmd[6]) * FINGER_OPEN
        q["joint7"], q["joint8"] = f, f
        return A.world_hulls(A.fk_link_poses_world(q, base))

    def aabb(hs):
        return np.array([h.vertices.min(0) for _, h in hs]), np.array([h.vertices.max(0) for _, h in hs])

    best = {"mm": 1e9, "step": None, "pair": None}
    for step, lc, rc, b, yl, yr in steps:
        hl, hr = hulls(lc, station.BASES["left"]), hulls(rc, station.BASES["right"])
        llo, lhi = aabb(hl)
        rlo, rhi = aabb(hr)
        gap = np.maximum(rlo[None] - lhi[:, None], llo[:, None] - rhi[None]).max(-1)
        for i, j in zip(*np.where(gap < 0.35)):
            a, c = hl[i], hr[j]
            s = min(A._signed(a[1].vertices, c[1]).min(), A._signed(c[1].vertices, a[1]).min())
            if s * 1000 < best["mm"]:
                best = {"mm": s * 1000, "step": step, "pair": f"left {a[0]} / right {c[0]}"}
        # each arm against the saw part the other arm holds: its handle (box around the post and link) and the blade
        z0 = T.TOP
        parts = {"right": (hl, [(T.LOG_X - 0.015, yr - 0.015, z0 + 0.11), (T.LOG_X + 0.015, yr + 0.015, z0 + 0.19)]),
                 "left": (hr, [(T.LOG_X - 0.015, yl - 0.015, z0 + 0.11), (T.LOG_X + 0.015, yl + 0.015, z0 + 0.19)])}
        for holder, (other_hulls, (lo, hi)) in parts.items():
            lo, hi = np.array(lo), np.array(hi)
            for name, h in other_hulls:
                v = h.vertices
                d = np.linalg.norm(np.maximum(np.maximum(lo - v, v - hi), 0.0), axis=1).min()
                if d * 1000 < best["mm"] and d > 0:
                    best = {"mm": d * 1000, "step": step, "pair": f"{'left' if holder == 'right' else 'right'} {name} / "
                                                                   f"{holder} handle"}
    best["mm"] = round(float(best["mm"]), 1)
    return best


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.marks = {}
        self.step = 0

    def mark(self, name):
        self.marks[name] = self.step

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


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
    frames: dict[str, list] = {}
    debug = bool(os.environ.get("DEMO_DEBUG"))
    trace: list = []
    path_steps: list = []
    lag = {"left": 0.0, "right": 0.0}

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
        ctx.step = steps
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        M = task._MECH
        pts = M.get("pts") or []
        row = {"step": steps, "score": info.get("score"), "b_mm": round(1000 * M["b"], 1),
               "gap_l_mm": round(1000 * M["gap_l"], 1), "gap_r_mm": round(1000 * M["gap_r"], 1),
               "strokes": M["strokes"], "kink": M["kink"], "bare": M["bare"], "held": [M["held_l"], M["held_r"]],
               "margin_mm": round(1000 * M["min_margin"], 1), "d_blade_mm": round(1000 * min(M["d_blade"], 1.0), 1)}
        if len(pts) == 8:
            for k, (name, arm, sl) in enumerate((("left", ctx.left, slice(0, 2)), ("right", ctx.right, slice(4, 6)))):
                tcp = np.mean(pts[sl], axis=0)
                err = arm.pos - np.array([task.LOG_X, 0.0, task.TOP]) - tcp
                lag[name] = max(lag[name], float(np.linalg.norm(err)))
                row[f"lag_{name}_mm"] = round(1000 * float(np.linalg.norm(err)), 1)
        path_steps.append((steps, ctx.left.cmd.copy(), ctx.right.cmd.copy(), M["b"], float(ctx.left.pos[1]),
                           float(ctx.right.pos[1])))
        if debug and steps % 10 == 0:
            trace.append(row)
            print("TRACE " + json.dumps(row), flush=True)
        elif steps % 100 == 0:
            print(f"step {steps} score {info.get('score')}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    M = task._MECH
    result = {"task_score": score, "success": success, "steps": steps, "variant": os.environ.get("DEMO_VARIANT", ""),
              "strokes": M["strokes"], "kink": M["kink"], "kink_step": M["kink_at"], "bare": M["bare"],
              "min_link_margin_mm": round(1000 * M["min_margin"], 1), "min_link_margin_step": M["min_margin_at"],
              "max_lag_mm": {k: round(1000 * v, 1) for k, v in lag.items()},
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    try:
        result["min_arm_clearance"] = arm_clearance(path_steps[::2], task)
        print("CLEARANCE " + json.dumps(result["min_arm_clearance"]), flush=True)
    except Exception as exc:  # noqa: BLE001
        result["min_arm_clearance"] = f"unavailable: {exc!r}"
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
