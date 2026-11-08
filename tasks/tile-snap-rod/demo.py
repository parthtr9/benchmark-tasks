"""Scripted full-score demo for the tile task, run in the yam_arena embodiment. Open loop: it uses only the fixed layout,
timing and the arms' own commands, and never reads the tile, the rod or any task state.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Both arms close their fingers and move over the press points (0.29, +-0.10). The left arm lowers at 2 cm/s onto the +y
end (already down on the board) and holds; then the right arm lowers the same way on the raised -y end, past the point
where the tile snaps (the board stops the tile, the arm overdrives a few mm). Hold 0.5 s, lift both 6 cm, wait.

Negative variants (DEMO_VARIANT=...):
  one_arm      only the right arm presses the raised end (the tile rocks onto it)
  left_only    only the left arm presses the +y end (nothing bends)
  straddle     only the right gripper, opened 6 cm, presses both sides of the score (chip if it snaps)
  no_release   as the full demo, but both grippers stay down on the tile
  short_press  the right arm stops 3 mm above the snap point
DEMO_DEBUG=1 prints a trace (also written to debug_trace.txt in the output folder, and kept in result.json under
"trace" whether or not DEMO_DEBUG is set): each arm's commanded vs measured fingertip height (its lag), the tile's
hinge angle and tilt, and each arm's fingertips against the press zones.
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



TOP = 0.7475                          # station table top
BOARD_X = 0.29
PRESS_Y = 0.10
PAD_PAST = 0.013                      # pad tips are 13 mm below the tool point
HOVER_PAD_Z = 0.065                   # pad-tip height above the table while travelling
LEFT_PAD_Z = 0.011                    # left pad tips: tile top on the board is 0.019, so 8 mm of overdrive
RIGHT_PAD_Z = 0.014                   # right pad tips: the tile snaps with its end top at about 0.028
SHORT_PAD_Z = 0.031                   # short_press: stops above the snap point
DESCENT = 0.02                        # m/s
LIFT = 0.06
STRADDLE_GRIP = 0.64                  # 60 mm between the pads (94 mm fully open)
VARIANTS = ("", "one_arm", "left_only", "straddle", "no_release", "short_press")


def script(task, ctx):
    from yam_arena.demo import grip_rotation

    L, R_ = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    arms = {1: L, -1: R_}
    if variant in ("one_arm", "straddle"):
        arms = {-1: R_}
    if variant == "left_only":
        arms = {1: L}

    def xy(side):
        return np.array([BOARD_X, 0.0 if variant == "straddle" else side * PRESS_Y])

    def pose(side, pad_z):
        return np.array([*xy(side), TOP + pad_z + PAD_PAST])

    closed = STRADDLE_GRIP if variant == "straddle" else 0.0
    rots = {s: grip_rotation(reachable_yaw(a, pose(s, HOVER_PAD_Z), 0.0)) for s, a in arms.items()}
    final = {1: LEFT_PAD_Z, -1: SHORT_PAD_Z if variant == "short_press" else RIGHT_PAD_Z}

    yield from wait(1.0)                                       # let the tile settle onto its +y end
    yield from both(*[a.move(pose(s, HOVER_PAD_Z), rots[s], 2.5, grip=closed) for s, a in arms.items()])
    ctx.mark("hover")
    for side in (1, -1):                                       # left first, then right; each at 2 cm/s
        if side in arms:
            dur = (HOVER_PAD_Z - final[side]) / DESCENT
            yield from arms[side].move(pose(side, final[side]), rots[side], dur)
            ctx.mark("left_down" if side == 1 else "right_down")
            yield from wait(0.5)
    yield from wait(0.5)
    if variant != "no_release":
        yield from both(*[a.move(a.pos + np.array([0, 0, LIFT]), a.rot, 1.5) for a in arms.values()])
    yield from wait(2.5)


def _yaw_of(p):
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.step, self.marks = 0, {}

    def mark(self, name):
        self.marks[name] = self.step

    def measured_tcp(self, name):
        """Where the arm's tool point really is, by forward kinematics of its measured joints (logging only)."""
        from yam_arena import ik

        robot = self.emb._env.unwrapped.scene[f"{name}_robot"]
        jp = getattr(robot.data.joint_pos, "torch", robot.data.joint_pos)[0].detach().cpu().numpy()
        names = list(robot.joint_names)
        q = np.array([jp[names.index(f"joint{i}")] for i in range(1, 7)], float)
        pos, _ = ik.tcp_pose(q, (self.left if name == "left" else self.right).base)
        return np.asarray(pos, float)

    def trace(self, task):
        """One debug line: commanded vs measured pad-tip height and lag of each arm, plus the task's own tile line.
        Reads simulated state; never used for control."""
        parts = []
        for name, arm in (("L", self.left), ("R", self.right)):
            m = self.measured_tcp("left" if name == "L" else "right")
            lag = 1000 * (arm.pos - m)
            parts.append(f"{name} cmd_pad_z_mm {1000 * (arm.pos[2] - TOP - PAD_PAST):.1f} meas_pad_z_mm "
                         f"{1000 * (m[2] - TOP - PAD_PAST):.1f} lag_xyz_mm {np.round(lag, 1).tolist()} grip {arm.grip:.2f}")
        from partkit import _part_state

        tile = _part_state(self.emb._env, task.TILE)
        return " | ".join(parts) + " || " + task.trace_line(self.emb._env, tile)

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
    trace: list[str] = []

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
        ctx.step = steps
        debug = bool(os.environ.get("DEMO_DEBUG"))
        if steps % 100 == 0 or steps % 10 == 0:
            try:
                line = f"step {steps} score {info.get('score')} :: {ctx.trace(task)}"
            except Exception as exc:  # noqa: BLE001 - the trace must never stop the demo
                line = f"step {steps} score {info.get('score')} :: trace failed {exc!r}"
            trace.append(line)
            if debug:
                print(line, flush=True)
                print(line, file=sys.stderr, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "marks": ctx.marks, "snap_step": task._STATE.get("snap_step"), "chip": task._STATE.get("chip"),
              "off_rod": task._STATE.get("off_rod"), "peak_rod_off_mm": round(1000 * task._STATE.get("peak_off", 0.0), 2),
              "max_hinge_deg": round(math.degrees(task._STATE.get("max_hinge", 0.0)), 2), "trace": trace}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "debug_trace.txt").write_text("\n".join(trace) + "\n")
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
