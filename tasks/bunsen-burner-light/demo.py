"""Scripted full-score demo for the Bunsen-burner task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Open loop on the fixed layout and on timing only (no part pose or joint is read to steer). The left arm grips the
striker in its holder (the spring arms held 24 mm apart, no spark), lifts it and brings its cup 15 mm above the burner's
mouth. The right arm then takes the tap's grip block and turns it a quarter turn counter-clockwise over 1 s. From 0.6 s
into the turn the left hand squeezes the striker (to 8 mm) and lets it go three times, 0.4 s apart: the flow is past 0.3
by then, so the first squeeze lights the burner. The right arm lets go of the tap (it stays open in its detent) and
lifts away; the left arm lifts the striker off the mouth, sets it back in its holder, lets go and lifts away.

Negative variants (DEMO_VARIANT=...):
  one_arm       the right arm opens the tap and lets go; nothing is brought to the mouth (0: flare)
  gas_first     the tap is opened first, then the striker is fetched (about 5 s of gas unlit) (0: flare)
  cold_strike   the striker is squeezed at the mouth with the tap shut (0: never lit)
  no_return     lit and burning, but the striker is parked on the bench instead of its holder (0.5)
DEMO_DEBUG=1 prints a trace every 10 steps (the minimum arm-to-arm clearance on the URDF collision meshes among them).
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder.
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


HOLD_G = 0.256          # finger opening for a 24 mm grip zone (finger travel 2 x 47 mm over g in 0..1)
SQUEEZE_G = 0.07        # squeezes the zone to its 8 mm stop
OPEN_G = 0.43           # 40 mm: let go of the striker
SWING_S = 1.0           # the tap's quarter turn
VARIANTS = ("", "one_arm", "gas_first", "cold_strike", "no_return")


def smooth(x):
    return x * x * (3 - 2 * x)


class Hands:
    """What the clearance trace needs to know: which hand holds what, and the tap lever's commanded angle."""

    left_holds = False
    right_holds = False
    theta = 0.0
    lever_on = False


def tap_tcp(T, theta):
    return np.array([T.TAP_XY[0] + T.LEVER_LEN * math.cos(theta), T.TAP_XY[1] + T.LEVER_LEN * math.sin(theta),
                     T.TOP + T.TAP_Z + 0.007])


def swing(T, arm, yaw0):
    """The right hand carries the grip block round the lever's axis, the fingers turning with it."""
    from yam_arena.demo import grip_rotation

    n = round(SWING_S * CONTROL_HZ)
    for k in range(1, n + 1):
        th = T.OPEN_ANGLE * smooth(k / n)
        Hands.theta = th
        arm.aim(tap_tcp(T, th), grip_rotation(yaw0 + th))
        yield


def squeezes(L, rot, n=3, period=0.4):
    """Squeeze the striker to its stop and let it back to the hold, ``n`` times, ``period`` s apart."""
    for _ in range(n):
        yield from L.move(L.pos, rot, 0.15, grip=SQUEEZE_G)
        yield from wait(0.1)
        yield from L.move(L.pos, rot, 0.1, grip=HOLD_G)
        yield from wait(period - 0.35)


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    T = task
    L, R = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    Hands.left_holds = Hands.right_holds = False
    Hands.theta = 0.0
    up = lambda h: np.array([0, 0, h])
    # Fixed layout: the striker's grip zone in its holder; the cup is CUP_DX ahead of the hand.
    s_origin = np.array([T.HOLDER_XY[0], T.HOLDER_XY[1], T.TOP + T.GRIP_Z])
    tool = np.array([0, 0, 0.008])                       # the pads reach 13 mm below the tool point
    g_s = s_origin + tool
    mouth = np.array([T.BURNER_XY[0], T.BURNER_XY[1], T.TOP + T.MOUTH_Z])
    at_mouth = mouth + np.array([-T.CUP_DX, 0, 0.015]) + tool
    yaw_l = reachable_yaw(L, g_s, 0.0)
    rot_l = grip_rotation(yaw_l)
    g_t = tap_tcp(T, 0.0)
    yaw_r = reachable_yaw(R, g_t, 0.0)
    rot_r = grip_rotation(yaw_r)

    def left_fetch():
        yield from L.move(g_s + up(0.06), rot_l, 2.5, grip=1.0)
        yield from L.move(g_s, rot_l, 1.5)
        yield from L.set_grip(HOLD_G, 1.0)
        Hands.left_holds = True
        yield from L.move(g_s + up(0.06), rot_l, 1.0)
        yield from L.move(at_mouth + up(0.03), rot_l, 2.0)
        yield from L.move(at_mouth, rot_l, 1.0)
        yield from wait(0.3)

    def right_grab():
        yield from R.move(g_t + up(0.06), rot_r, 2.5, grip=1.0)
        yield from R.move(g_t, rot_r, 1.5)
        yield from R.set_grip(0.0, 1.0)
        Hands.right_holds = True

    def right_leave():
        yield from R.set_grip(1.0, 0.6)
        Hands.right_holds = False
        yield from R.move(R.pos + up(0.08), R.rot, 1.0)

    def left_return():
        home_above = g_s + up(0.03)
        yield from L.move(L.pos + up(0.04), rot_l, 0.8)
        yield from L.move(home_above + np.array([0, 0, 0.0]), rot_l, 2.0)
        yield from L.move(g_s + up(0.0015), rot_l, 1.5)
        yield from L.set_grip(OPEN_G, 0.8)
        Hands.left_holds = False
        yield from L.move(L.pos + up(0.08), rot_l, 1.0)

    def left_park():
        park = np.array([0.30, 0.26, T.TOP + 0.0105 + 0.008])
        yield from L.move(L.pos + up(0.04), rot_l, 0.8)
        yield from L.move(park + up(0.03), rot_l, 2.0)
        yield from L.move(park + up(0.0015), rot_l, 1.5)
        yield from L.set_grip(OPEN_G, 0.8)
        Hands.left_holds = False
        yield from L.move(L.pos + up(0.08), rot_l, 1.0)

    if variant == "one_arm":
        yield from right_grab()
        yield from swing(T, R, yaw_r)
        yield from right_leave()
        yield from wait(3.0)
        return
    if variant == "gas_first":
        yield from right_grab()
        yield from swing(T, R, yaw_r)
        yield from right_leave()
        yield from left_fetch()
        yield from squeezes(L, rot_l)
        yield from left_return()
        yield from wait(1.5)
        return

    # 1. Both hands to their parts; the left one brings the striker to the mouth while the right one takes the tap.
    yield from both(left_fetch(), right_grab())
    # 2. Right turns the tap; from 0.6 s into the turn the left squeezes the striker three times.
    if variant == "cold_strike":
        yield from squeezes(L, rot_l)
        yield from wait(0.5)
    else:
        yield from both(swing(T, R, yaw_r), _after(0.6, squeezes(L, rot_l)))
        yield from wait(0.3)
        yield from both(right_leave(), left_park() if variant == "no_return" else left_return())
    yield from wait(1.5)


def _after(seconds, gen):
    yield from wait(seconds)
    yield from gen


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def held_points(ctx, task):
    """Point sets of what each hand holds, for the clearance check: the striker (bar, tail and cup, held level at the
    left tool point) and the tap's lever (bar and grip block at the commanded angle)."""
    import clearance

    T = task
    held_l, held_r = [], []
    if Hands.left_holds:
        c = ctx.left.pos - np.array([0, 0, 0.008])
        held_l = np.vstack([clearance.box_points(c + [0.0375, 0, 0], (0.19, 0.016, 0.016)),
                            clearance.box_points(c + [T.CUP_DX, 0, 0], (0.02, 0.02, 0.01))])
    if Hands.right_holds:
        a, r = Hands.theta, T.LEVER_LEN
        centre = np.array([T.TAP_XY[0] + r * math.cos(a), T.TAP_XY[1] + r * math.sin(a), T.TOP + T.TAP_Z])
        held_r = clearance.box_points(centre, (0.03, 0.03, 0.03))
    return held_l, held_r


def arm_clearance(ctx, task):
    """Smallest distance (m) between any collision mesh of one arm and the other arm or the part the other holds,
    at the commanded joints (URDF meshes, see clearance.py)."""
    import clearance

    held_l, held_r = held_points(ctx, task)
    left, right = ctx.left, ctx.right
    return clearance.min_clearance(left.q, left.grip, left.base, right.q, right.grip, right.base, held_a=held_l,
                                   held_b=held_r)


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 4.0) -> dict:
    from inspect_robots import Scene as InspectScene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(InspectScene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    gen = script(task, ctx)

    def states():                      # logging only: the demo itself never reads a part
        from yam_arena.scene import object_states

        return object_states(emb._env, emb.task.names)

    frames: dict[str, list] = {}
    debug = bool(os.environ.get("DEMO_DEBUG"))
    min_clear = (float("inf"), ("", ""), -1)
    trace = []

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
        if steps % 5 == 0:
            try:
                d, names = arm_clearance(ctx, task)
                if d < min_clear[0]:
                    min_clear = (d, names, steps)
            except Exception as exc:  # noqa: BLE001
                d = None
                print(f"clearance check failed: {exc!r}", flush=True)
        if steps % 100 == 0 or (debug and steps % 10 == 0):
            msg = f"step {steps} score {info.get('score')}"
            if debug:
                S = task._STATE
                st = states()
                msg += (f" tap_deg {math.degrees(task.lever_angle(st['tap'])):.1f} gap_mm {1000 * S['gap']:.1f}"
                        f" lit {S['lit']} unlit_s {S['unlit']:.2f} flare {S['flare']} sparks {S['sparks']}"
                        f" touched {S['touched']} dropped {S['dropped']}"
                        f" cup_mouth_mm {1000 * float(np.linalg.norm(task.cup_point(st['striker']) - task.MOUTH)):.1f}"
                        f" striker_home {task.at_home(st['striker'])}"
                        f" L_mm {np.round(1000 * ctx.left.pos, 1)} R_mm {np.round(1000 * ctx.right.pos, 1)}"
                        f" min_clear_mm {round(1000 * min_clear[0], 1)}")
                trace.append(msg)
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "min_arm_clearance_mm": round(1000 * min_clear[0], 1), "min_arm_clearance_pair": list(min_clear[1]),
              "min_arm_clearance_step": min_clear[2]}
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
