"""Scripted full-score demo for the DIN rail task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The right arm grips the module's housing near its top and lifts it out of its holder. It carries it in front of the
rail with its bottom swung 12 degrees away from the panel, lifts the hook 12 mm above the rail's upper lip, moves it
over the lip, lowers it onto the lip and swings the module to 5 degrees, and lets go: the module hangs on its hook with
its latch resting on the lower lip. Then, with closed fingers and closed loop on the measured module pose, it pushes the
housing's front low down until the latch clicks behind the lower lip, backs off and withdraws.
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder.

DEMO_VARIANT=no_push hooks the module and lets go without pushing; DEMO_VARIANT=no_hook carries it level to the rail,
without lifting the hook over the lip, and pushes. Both must score low.
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
TILT = math.radians(12)            # the module's bottom swung away from the panel while hooking
HOOK_LIFT_MM = 12.0                # hook above its hanging height while moving over the lip
GRASP = np.array([-0.0275, 0.0, 0.036])   # tool point in the module frame: housing centre, 9 mm below its top
PIVOT = np.array([0.0035, 0.0, 0.0175])   # where the hook bridge bears on the upper lip, module frame
PUSH_Z = -0.026                    # height on the housing's front face where the closed fingers push (module frame)


def _rot_y(a) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


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


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def _grip_yaw(task, base) -> float:
    """0 or pi: the turn of the straight-down grip (fingers across the housing either way) that IK reaches best at
    the hooking poses. Joint 6's limits rule one of them out in parts of the workspace."""
    from yam_arena import ik
    from yam_arena.constants import READY
    from yam_arena.demo import grip_rotation

    worst = {}
    for flip in (0.0, math.pi):
        errors = []
        for theta, d in ((TILT, (-0.018, 0, 0.016)), (TILT, (0, 0, 0)), (math.radians(5), (0, 0, 0))):
            p, R = hooked_pose(task, theta, d)
            rot = R @ grip_rotation(flip)
            _, pe, _ = ik.solve(p + R @ GRASP, ik.matrix_to_quat_xyzw(rot), np.array(READY, float), base_pos=base,
                                iters=200)
            errors.append(pe)
        worst[flip] = max(errors)
    return min(worst, key=worst.get)


def hooked_pose(task, theta, d):
    """Module origin and rotation with the hook's bearing point at its hanging position plus ``d`` (m, rail frame) and
    the module's bottom swung ``theta`` away from the panel. The rail is square to the world."""
    R = _rot_y(theta)
    pivot = task.RAIL_ORIGIN + PIVOT + np.asarray(d, float)
    return pivot - R @ PIVOT, R


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    A = ctx.right
    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    flip = _grip_yaw(T, A.base)
    m0 = ctx.states()["din_module"]

    def tool(p, R):
        return p + R @ GRASP, R @ grip_rotation(flip)

    # 1. Grip the housing near its top and lift it out of the holder.
    grasp = tool(m0.pos, m0.rotation())
    yield from A.move(grasp[0] + [0, 0, 0.07], grasp[1], 2.5, grip=1.0)
    yield from A.move(*grasp, 1.5)
    yield from A.set_grip(0.0, 1.0)
    yield from wait(0.3)
    yield from A.move(grasp[0] + [0, 0, 0.08], grasp[1], 1.5)

    # 2. In front of the rail, bottom swung out; hook lifted over the upper lip, moved behind it and lowered onto it.
    theta = 0.0 if variant == "no_hook" else TILT
    lift = 0.0 if variant == "no_hook" else HOOK_LIFT_MM / 1000
    yield from A.move(*tool(*hooked_pose(T, theta, (-0.018, 0, lift + 0.004))), 3.0)
    yield from A.move(*tool(*hooked_pose(T, theta, (-0.018, 0, lift))), 1.0)
    yield from A.move(*tool(*hooked_pose(T, theta, (0, 0, lift))), 2.0)
    yield from A.move(*tool(*hooked_pose(T, theta, (0, 0, -0.0005))), 1.5)

    # 3. Swing the bottom in to 5 degrees and let go: it hangs on the hook, latch resting on the lower lip.
    yield from A.move(*tool(*hooked_pose(T, min(theta, math.radians(5)), (0, 0, -0.0005))), 1.5)
    yield from A.set_grip(1.0, 0.8)
    yield from A.move(A.pos + [-0.03, 0, 0.02], A.rot, 1.0)
    if variant == "no_push":
        yield from A.move(A.pos + [-0.03, 0, 0.05], A.rot, 1.5)
        return

    # 4. Closed fingers to the housing's front, low down; push along the rail's normal until the latch clicks.
    down = grip_rotation(flip)
    yield from A.set_grip(0.0, 0.6)
    m = ctx.states()["din_module"]
    face = m.to_world((-0.055, 0.0, PUSH_Z))
    start = face + np.array([-0.025, 0.0, 0.008])  # tips 5 mm below the push point; pads 2.5 cm off the face
    yield from A.move(start + [0, 0, 0.03], down, 1.5)
    yield from A.move(start, down, 1.0)
    x = start[0]
    for _ in range(int(5.0 * CONTROL_HZ)):
        m = ctx.states()["din_module"]
        c, rel = T.module_in_rail(m)
        *_, bottom_x = T.geometry(c, rel)
        if bottom_x > -0.3 and m.sliders.get("latch", 0.0) > -0.0005:
            break
        x += 0.010 / CONTROL_HZ  # 10 mm/s
        A.aim(np.array([x, start[1], start[2]]), down)
        yield

    # 5. Back off as soon as it clicks (pushing on rocks the latched module about its hook), then withdraw.
    yield from A.move(A.pos + [-0.008, 0, 0], down, 0.5)
    yield from A.move(A.pos + [-0.04, 0, 0.05], down, 1.5)


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
                m = ctx.states()["din_module"]
                c, rel = task.module_in_rail(m)
                hooked, flush, square, bottom_x = task.geometry(c, rel)
                pitch = math.degrees(math.atan2(rel[0, 2], rel[2, 2]))
                msg += (f" module_mm ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) pitch {pitch:.1f} latch_mm "
                        f"{1000 * m.sliders.get('latch', 0.0):.2f} bottom_gap_mm {-bottom_x:.2f} hooked {hooked} "
                        f"flush {flush} square {square} stages lifted {task._STATE['lifted']} hooked "
                        f"{task._STATE['hooked']} pushed {task._STATE['pushed']}")
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "variant": os.environ.get("DEMO_VARIANT", ""),
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
