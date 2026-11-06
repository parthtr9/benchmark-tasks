"""Scripted full-score demo for the tray-table task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The right arm slides its open fingers along the tray's plane onto the tongue at the free edge, closes, and swings the
tray up along its arc, tracking the measured tray angle each step (the gripper pitches with the tray) until it stands
against the seatback at about 80 degrees. Holding it there, the left arm, fingers closed, presses the latch tab's tip
down round its hinge line, tracking the measured tab angle, until the detent snaps it down in front of the tray's
top edge. The right arm then opens, both arms back out 15 cm along their approach and lift away.

Negative variants (DEMO_VARIANT=...):
  one_arm          the right arm folds the tray up and lets go; nobody flips the tab (0.25)
  tab_first        the tab is flipped down first, then the tray is pushed into it and stops at about 65 degrees (0)
  hold_no_latch    the right arm folds the tray up and keeps holding it; the tab is never flipped (0.5)
DEMO_DEBUG=1 prints a trace every 10 steps (angles, hand positions, distance between the hands).
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



R_TCP = 0.128          # the right TCP's distance from the tray's hinge line, along the tray (pads span the tongue)
DELTA = math.radians(15.0)   # the right tool points this far below the tray's plane
TAB_REACH = 0.040      # the left pad presses the tab this far from its hinge line
TAB_CLEAR = 0.003      # pad tip clearance above the tab at rest


def _ry(a):
    return _rot_axis((0, 1, 0), a)


def _rx(a):
    return _rot_axis((1, 0, 0), a)


def tray_point(seat, task, theta, r=R_TCP):
    """World TCP position on the tongue at tray angle ``theta`` (the tongue's middle, ``r`` from the hinge line)."""
    return np.asarray(seat.to_world((-r * math.cos(theta), task.TONGUE_Y, task.HINGE_Z + r * math.sin(theta))), float)


def tray_rot(seat, theta, yaw0):
    """Gripper rotation on the tongue: approaching along the tray toward the hinge (pointing DELTA below its plane),
    turning with the tray about the hinge line."""
    from yam_arena.demo import grip_rotation

    rs = seat.rotation()
    return rs @ _ry(theta - math.pi / 2 + DELTA) @ rs.T @ grip_rotation(yaw0)


def tab_point(seat, task, phi, clear=TAB_CLEAR):
    """World TCP position pressing the tab at tab angle ``phi`` (rad, negative = down)."""
    px, py, pz = task.TAB_PIVOT
    off = _rx(phi) @ np.array([0.0, TAB_REACH, task.TAB_T / 2 + 0.013 + clear])
    return np.asarray(seat.to_world((px + off[0], py + off[1], pz + off[2])), float)


def tab_rot(seat, phi, yaw0):
    from yam_arena.demo import grip_rotation

    rs = seat.rotation()
    return rs @ _rx(phi) @ rs.T @ grip_rotation(yaw0)


class Swing:
    """Open loop on one arm: each step the TCP goes to the hinged part's place at a commanded angle that moves toward
    ``goal`` at ``rate`` rad/s (the part's place comes from the task file's fixed layout), so the fingers push the
    part round its hinge. Nothing is read back from the part."""

    def __init__(self, arm, pose, rate):
        self.arm, self.pose, self.rate = arm, pose, rate
        self.goal, self.cmd, self.on = 0.0, 0.0, False

    def track(self):
        if not self.on:
            return
        step = self.rate / CONTROL_HZ
        self.cmd = max(self.cmd - step, self.goal) if self.goal < self.cmd else min(self.cmd + step, self.goal)
        pos, rot = self.pose(self.cmd)
        self.arm.aim(pos, rot)

    def wrap(self, gen):
        for _ in gen:
            self.track()
            yield

    def go(self, goal):
        """Run until the command has reached ``goal``."""
        self.goal = goal
        while abs(self.cmd - goal) > 1e-6:
            self.track()
            yield


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    T = task
    L, R = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in ("", "one_arm", "tab_first", "hold_no_latch"):
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    from partkit import PartState

    pos0 = np.array([T.SEAT_X, 0.0, T.TOP])        # the seat's fixed place from the task file
    seat = PartState(pos=pos0, quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                     low=pos0, high=pos0)
    up = np.array([0, 0, 0.06])

    p_r0 = tray_point(seat, T, 0.0)
    yaw_r = reachable_yaw(R, p_r0, 0.0)
    rot_r0 = tray_rot(seat, 0.0, yaw_r)
    pre_r = tray_point(seat, T, 0.0, r=R_TCP + 0.08) + np.array([0, 0, 0.012])
    p_l0 = tab_point(seat, T, 0.0)
    yaw_l = reachable_yaw(L, p_l0, 0.0)
    rot_l0 = tab_rot(seat, 0.0, yaw_l)

    tray = Swing(R, lambda a: (tray_point(seat, T, a), tray_rot(seat, a, yaw_r)), rate=0.35)
    tab = Swing(L, lambda a: (tab_point(seat, T, a, 0.0), tab_rot(seat, a, yaw_l)), rate=0.6)

    def both_track(gen):
        for _ in gen:
            tray.track()
            tab.track()
            yield

    def flip_tab():
        # Left arm: above the tab, down onto it (fingers closed), then press it round its hinge line.
        yield from both_track(L.move(p_l0 + up, rot_l0, 2.5, grip=0.0))
        yield from both_track(L.move(p_l0, rot_l0, 1.2))
        tab.on = True
        yield from tab.go(-T.TAB_DOWN + 0.05)
        yield from both_track(wait(0.5))

    def fold_tray():
        yield from R.move(pre_r, rot_r0, 2.5, grip=1.0)
        yield from R.move(p_r0, rot_r0, 1.5)
        yield from R.set_grip(0.0, 1.0)
        tray.on = True
        yield from tray.go(math.radians(68.0 if variant == "tab_first" else 80.0))   # the blocked tray is not forced
        yield from tray.wrap(wait(0.5))

    if variant == "tab_first":
        yield from flip_tab()
        yield from fold_tray()
    elif variant == "one_arm":
        yield from fold_tray()
    elif variant == "hold_no_latch":
        yield from fold_tray()
        yield from tray.wrap(wait(1.0))
        return                                 # keeps holding until the run ends
    else:
        yield from fold_tray()
        yield from flip_tab()

    # Let go of the tray, then back each hand out 15 cm along its approach before lifting.
    tray.on = False
    yield from both_track(R.set_grip(1.0, 0.8))
    yield from both_track(wait(0.5))
    back_r = R.pos - 0.15 * R.rot[:, 2]
    yield from both_track(R.move(back_r, R.rot, 1.5))
    yield from both_track(R.move(back_r + up, R.rot, 1.0))
    tab.on = False
    if variant != "one_arm":
        back_l = L.pos - 0.15 * L.rot[:, 2]
        yield from L.move(back_l, L.rot, 1.5)
        yield from L.move(back_l + up, L.rot, 1.0)
    yield from wait(1.0)


def _yaw_of(p):
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.min_hands = 9.0

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
                seat = st["seat"]
                dist = float(np.linalg.norm(ctx.left.pos - ctx.right.pos))
                ctx.min_hands = min(ctx.min_hands, dist)
                msg += (f" tray_deg {math.degrees(task.tray_angle(seat)):.1f}"
                        f" tab_deg {math.degrees(task.tab_angle(seat)):.1f} hands_mm {1000 * dist:.0f}"
                        f" held {task._STATE['held']} failed {task._STATE['failed']}"
                        f" L_mm {np.round(1000 * ctx.left.pos, 1)} R_mm {np.round(1000 * ctx.right.pos, 1)}")
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps,
              "min_hands_mm": round(1000 * ctx.min_hands, 1), "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
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
