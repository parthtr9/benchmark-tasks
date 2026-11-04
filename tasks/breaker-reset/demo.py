"""Scripted full-score demo for the breaker-reset task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The right arm closes its fingers and pushes the spring-loaded cover's tab back until the breakers are uncovered, then
keeps its fingers against the tab, holding the cover open. The left arm closes its fingers and pushes the tripped
breaker's (breaker 3) handle by its upper part, from the +x
(red) side toward the -x (green) end, tracking the measured toggle position each step until the breaker is off. It backs
off, lifts over the handle, comes down on the -x side and pushes the handle the other way, through the middle, up to
the red end: the breaker is on. Then it lifts away and the right arm lets the cover slide shut.

Negative variants (DEMO_VARIANT=...):
  on_only         push breaker 3 straight toward ON only: the trip-free mechanism springs it back (0)
  off_only        push breaker 3 to OFF and leave it there (0.5)
  cycle_neighbour switch breaker 4 off and back on first, then reset breaker 3 correctly (0.75)
  wrong_off       switch breaker 4 off and leave it, then reset breaker 3 (0)
  one_arm         the right arm opens the cover and lets go (it slides shut); the left arm goes through the
                  motions over the closed cover: nothing is reset (0)
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


FINGER_HALF = 0.002     # half the closed fingertips' extent along x at the pad tip (the pad is wider higher up)
HANDLE_HALF = 0.008
PUSH_Z = 0.027          # where on the toggle the pad tips push, above the body top (the palm sits about 4.5 mm under the TCP: keep it clear of the toggle top)
# Settings tried in turn when a push does not take (pad tip half-width along x, push height above the body top).
ATTEMPTS = [(0.002, 0.027), (0.004, 0.027), (0.008, 0.026), (0.010, 0.024)]
RATE, LEAD = 0.08, 0.020  # m/s of the command along x, and how far (m) it may run ahead of the toggle (more lead, more push)
TAB_PUSH_Z = 0.014      # where on the cover's tab the fingers push, above the plate's top


def contact(panel, task, k, pos, side, off, z_extra=0.0):
    """TCP for pushing breaker ``k``'s toggle at ``pos`` (m along x from its middle): ``side`` +1 is the +x face, -1
    the -x face; ``off`` metres clear of the face. The TCP rides 13 mm above the pad tips."""
    local = (pos + side * (HANDLE_HALF + FINGER_HALF + off), task.BREAKER_Y[k], task.PIVOT_Z + PUSH_Z + 0.013 + z_extra)
    return np.asarray(panel.to_world(local), float)


def tab_point(panel, task, slide, off, z_extra=0.0):
    """TCP for pushing the cover's tab from the front (-x) with the cover slid back ``slide`` m."""
    local = (task.COVER_FRONT + slide - FINGER_HALF - off, task.TAB_Y,
             task.COVER_Z0 + task.COVER_T + TAB_PUSH_Z + 0.013 + z_extra)
    return np.asarray(panel.to_world(local), float)


def open_cover(task, ctx, arm, yaw0):
    """Generator (right arm): push the cover's tab back to COVER_HOLD, closed loop on the measured slide."""
    from yam_arena.demo import grip_rotation

    rot = grip_rotation(yaw0)
    panel = ctx.states()["panel"]
    yield from arm.move(tab_point(panel, task, 0.0, 0.02, 0.045), rot, 2.0, grip=0.0)
    yield from arm.move(tab_point(panel, task, 0.0, 0.02), rot, 1.0)
    yield from arm.move(tab_point(panel, task, 0.0, 0.0), rot, 0.8)
    cmd = 0.0
    for _ in range(int(8 * CONTROL_HZ)):
        s = task.cover_open(ctx.states()["panel"])
        cmd = min(cmd + 0.10 / CONTROL_HZ, task.COVER_HOLD, s + 0.03)
        arm.aim(tab_point(panel, task, cmd, 0.0), rot)
        yield
        if cmd >= task.COVER_HOLD - 1e-6 and s > task.COVER_HOLD - 0.015:
            break


def hold_cover(task, ctx, arm, yaw0):
    """One step of holding the cover open (call every step): the fingers stay at the hold position."""
    from yam_arena.demo import grip_rotation

    arm.aim(tab_point(ctx.states()["panel"], task, task.COVER_HOLD + 0.003, 0.0), grip_rotation(yaw0))


def release_cover(task, ctx, arm, yaw0):
    """Back away from the tab and lift: the cover slides shut."""
    from yam_arena.demo import grip_rotation

    rot = grip_rotation(yaw0)
    panel = ctx.states()["panel"]
    yield from arm.move(tab_point(panel, task, task.COVER_HOLD, 0.03), rot, 0.8)
    yield from arm.move(tab_point(panel, task, task.COVER_HOLD, 0.03, 0.06), rot, 0.8)


def push(task, ctx, arm, k, side, goal, yaw0, hold=None, skim=0.0):
    """Generator: approach the handle of breaker ``k`` from ``side``, push it (closed loop) to ``goal`` m, and back
    away along the same side, ending clear and above. ``hold`` is called every step (the other arm holds the cover);
    ``skim`` raises the fingers that far over the handle (the cover is shut: they ride over it)."""
    from yam_arena.demo import grip_rotation

    def step_other():
        if hold:
            hold()

    def go(pos, duration):
        for _ in arm.move(pos, rot, duration, **({"grip": 0.0} if first[0] else {})):
            first[0] = False
            step_other()
            yield

    rot = grip_rotation(yaw0)
    first = [True]
    panel = ctx.states()["panel"]
    a0 = task.handle_pos(panel, k)
    clear = 0.02
    yield from go(contact(panel, task, k, a0, side, clear, 0.045 + skim), 1.5)
    yield from go(contact(panel, task, k, a0, side, clear, skim), 1.0)
    yield from go(contact(panel, task, k, a0, side, 0.0, skim), 0.8)
    cmd = a0
    quiet = 0
    for _ in range(int(6 * CONTROL_HZ)):
        panel = ctx.states()["panel"]
        a = task.handle_pos(panel, k)
        quiet = quiet + 1 if abs(cmd - a) > 0.018 else 0   # the command is well ahead of a handle that is not moving
        if quiet > 25:
            break
        step = RATE / CONTROL_HZ
        cmd = min(cmd + step, goal, a + LEAD) if goal > cmd else max(cmd - step, goal, a - LEAD)
        arm.aim(contact(panel, task, k, cmd, side, 0.0, skim), rot)
        step_other()
        yield
        if abs(cmd - goal) < 1e-6 and abs(a - goal) < 0.003:
            break
    for _ in range(int(0.4 * CONTROL_HZ)):
        step_other()
        yield
    panel = ctx.states()["panel"]
    a = task.handle_pos(panel, k)
    yield from go(contact(panel, task, k, a, side, clear, skim), 0.8)
    yield from go(contact(panel, task, k, a, side, clear, 0.045 + skim), 0.8)


def try_push(task, ctx, arm, k, side, goal, yaw0, want, hold=None, skim=0.0, fixed=None):
    """Push breaker ``k``'s handle until its state is ``want``; if a push does not take, try the next setting."""
    global FINGER_HALF, PUSH_Z
    settings = [fixed] if fixed else ATTEMPTS
    for fh, z in settings:
        FINGER_HALF, PUSH_Z = fh, z
        yield from push(task, ctx, arm, k, side, goal, yaw0, hold, skim)
        if task._STATE["mode"][k] == want or skim:
            break


def script(task, ctx):
    """The demo as a generator: one yield per control step. The right arm holds the cover open while the left arm
    works the breakers."""
    from yam_arena.demo import grip_rotation

    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    fixed = None
    if variant.startswith("probe"):          # probe_<half-width mm>_<height mm>: one fixed setting, no retries
        os.environ["DEMO_DEBUG"] = "1"
        fh, z = (int(t.lstrip("fz")) for t in variant.split("_")[1:3])
        fixed, variant = (fh / 1000, z / 1000), ""
    elif variant not in ("", "on_only", "off_only", "cycle_neighbour", "wrong_off", "one_arm"):
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    panel = ctx.states()["panel"]
    L, R = ctx.left, ctx.right
    k3 = T.TRIPPED
    yaw_l = reachable_yaw(L, contact(panel, T, k3, 0.0, 1, 0.0), 0.0)
    yaw_r = reachable_yaw(R, tab_point(panel, T, 0.0, 0.0), 0.0)

    # 1. The right arm pushes the cover open.
    yield from open_cover(T, ctx, R, yaw_r)
    hold = lambda: hold_cover(T, ctx, R, yaw_r)
    if variant == "one_arm":
        # 2. It lets go: the cover slides shut. The left arm then rides over the cover, where the handles were.
        yield from release_cover(T, ctx, R, yaw_r)
        yield from wait(1.0)
        hold, skim = None, 0.03
    else:
        skim = 0.0

    def cycle(k):
        yield from try_push(T, ctx, L, k, 1, -0.013, yaw_l, "off", hold, skim, fixed)
        yield from try_push(T, ctx, L, k, -1, 0.013, yaw_l, "on", hold, skim, fixed)

    # 2. The left arm works the breakers.
    if variant == "cycle_neighbour":
        yield from cycle(3)
    if variant == "wrong_off":
        yield from try_push(T, ctx, L, 3, 1, -0.013, yaw_l, "off", hold)
    if variant == "on_only":
        yield from try_push(T, ctx, L, k3, -1, 0.013, yaw_l, "on", hold)
    elif variant == "off_only":
        yield from try_push(T, ctx, L, k3, 1, -0.013, yaw_l, "off", hold)
    else:
        yield from cycle(k3)
    for _ in wait(0.5):
        if hold:
            hold()
        yield
    # 3. Let go of the cover.
    if hold:
        yield from release_cover(T, ctx, R, yaw_r)
    yield from wait(2.0)


def _yaw_of(p):
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


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
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % 10 == 0):
            msg = f"step {steps} score {info.get('score')}"
            if os.environ.get("DEMO_DEBUG"):
                st = ctx.states()
                msg += (f" cover_mm {1000 * task.cover_open(st['panel']):.1f} toggles_mm {np.round(1000 * np.array(task.angles_of(st['panel'])), 1).tolist()} mode {task._STATE['mode']}"
                        f" good_off {task._STATE['good_off_ever']} min3 {task._STATE['min3']:.2f}"
                        f" panel_mm {np.round(1000 * (st['panel'].pos - task._STATE['pos0']), 1) if task._STATE['pos0'] is not None else None}"
                        f" L_cmd_mm {np.round(1000 * ctx.left.pos, 1)} R_cmd_mm {np.round(1000 * ctx.right.pos, 1)}")
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
