"""Scripted full-score demo for the bayonet cap task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The right arm grips the cap's square head, lifts it and parks it ahead of the box with its lugs already turned to
the slots. The left arm grips the box's long sides behind the socket. The right arm lowers the cap into the slots and,
closed loop on the measured cap pose, pushes it 2 mm past the roof's underside against the spring, turns it 92
degrees counterclockwise, eases off so the spring seats the lugs under the roof, and lets go.
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


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    L, R = ctx.left, ctx.right
    T = task
    s0 = ctx.states()
    cap0, box = s0["cap"], s0["socket_box"]
    box_yaw = T.BOX_YAW
    pinch_local = grip_rotation(0.0).T @ np.array([0.0, 1.0, 0.0])
    grip_z = 0.030  # tool point above the cap origin: the pads close on the square head, clear of the barrel

    # 1. Right arm grips the cap's head and lifts it.
    grasp = cap0.pos + np.array([0, 0, grip_z])
    yield from R.move(grasp + [0, 0, 0.07], grip_rotation(T.CAP_YAW), 2.5, grip=1.0)
    yield from R.move(grasp, grip_rotation(T.CAP_YAW), 1.5)
    yield from R.set_grip(0.0, 1.0)
    yield from wait(0.3)
    yield from R.move(grasp + [0, 0, 0.07], grip_rotation(T.CAP_YAW), 1.5)

    # 2. Park the cap ahead of the box, lugs already turned to the slots, while the left arm braces.
    yield from R.move(box.to_world((0.105, 0.0, 0.10 + grip_z)), grip_rotation(box_yaw), 3.0)
    if not os.environ.get("DEMO_NO_BRACE"):
        brace_rot = grip_rotation(box_yaw) @ _rot_axis(pinch_local, math.radians(-35))
        brace = box.to_world((-0.035, 0.0, 0.0))
        yield from L.move(brace + [0, 0, 0.09], brace_rot, 3.0, grip=1.0)
        yield from L.move(brace, brace_rot, 1.5)
        yield from L.set_grip(0.0, 1.2)
        yield from wait(0.3)

    # 3. Over the socket, then down until the lugs sit in the slots.
    sock = (T.SOCKET[0], T.SOCKET[1])
    yield from R.move(box.to_world((*sock, 0.07 + grip_z)), grip_rotation(box_yaw), 2.0)
    yield from R.move(box.to_world((*sock, 0.022 + grip_z)), grip_rotation(box_yaw), 2.0)

    # 4. Closed loop on the measured cap: push it 2 mm past the roof's underside, turn it 92 degrees, hold.
    state = {"xy": np.zeros(2), "z": 0.022, "twist": 0.0}

    def servo(height, twist):
        st = ctx.states()
        b, c = st["socket_box"], st["cap"]
        local, _, angle, _ = T.cap_in_box(b, c)
        rel = b.rotation().T @ c.rotation()
        angle = math.atan2(rel[1, 0], rel[0, 0])
        dt = 1.0 / CONTROL_HZ
        state["xy"] -= 2.0 * dt * (local[:2] - T.SOCKET)
        # At most 5 mm/s down: a faster push rams the cap into the spring plate instead of compressing it.
        state["z"] += float(np.clip(2.0 * dt * (height - local[2]), -0.005 * dt, 0.005 * dt))
        state["twist"] = float(np.clip(state["twist"] + 2.0 * dt * (twist - angle), twist - 0.45, twist + 0.45))
        yaw = math.atan2(b.rotation()[1, 0], b.rotation()[0, 0])
        R.aim(b.to_world((*(T.SOCKET + state["xy"]), state["z"] + grip_z)), grip_rotation(yaw + state["twist"]))
        return local[2]

    # DEMO_VARIANT=no_push turns the cap without pushing it; DEMO_VARIANT=no_twist pushes and lets go unturned.
    variant = os.environ.get("DEMO_VARIANT", "")
    pressed = T.SEATED_Z + 0.004 if variant == "no_push" else T.SEATED_Z - 0.002
    turn = 0.0 if variant == "no_twist" else math.radians(92)
    for _ in range(int(6.0 * CONTROL_HZ)):
        if servo(pressed, 0.0) <= pressed + 0.0003:
            break
        yield
    n = int(4.0 * CONTROL_HZ)
    for i in range(1, n + 1):
        u = i / n
        servo(pressed, turn * u * u * (3 - 2 * u))
        yield
    for _ in range(int(0.8 * CONTROL_HZ)):
        servo(pressed, turn)
        yield

    # 5. Ease off so the spring seats the lugs against the roof, let go, and lift away.
    for _ in range(int(1.5 * CONTROL_HZ)):
        servo(max(T.SEATED_Z, pressed), turn)
        yield
    yield from R.set_grip(1.0, 1.0)
    yield from R.move(R.pos + [0, 0, 0.07], R.rot, 1.5)
    if not os.environ.get("DEMO_NO_BRACE"):
        yield from L.set_grip(1.0, 0.8)
        yield from L.move(L.pos + [0, 0, 0.08], L.rot, 1.5)


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
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % 20 == 0 and steps > 380):
            msg = f"step {steps} score {info.get('score')}"
            if os.environ.get("DEMO_DEBUG"):
                st = ctx.states()
                b, c = st["socket_box"], st["cap"]
                local, lateral, angle, tilt = task.cap_in_box(b, c)
                yaw = math.degrees(math.atan2(b.rotation()[1, 0], b.rotation()[0, 0]))
                msg += (f" box_moved_mm {1000 * np.linalg.norm(b.pos - task._STATE['box0']):.1f} box_yaw {yaw:.1f}"
                        f" cap_z_mm {1000 * local[2]:.1f} lateral_mm {1000 * lateral:.1f} angle {angle:.0f}"
                        f" tilt {tilt:.0f} plate_mm {1000 * b.sliders.get('plate', 0):.1f} failed {task._STATE['failed']}"
                        f" box_tilt {math.degrees(math.acos(min(1.0, b.rotation()[2, 2]))):.1f} resting {c.resting()}"
                        f" spin {np.linalg.norm(c.ang_vel):.2f} speed {np.linalg.norm(c.lin_vel):.3f}")
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    if os.environ.get("DEMO_DEBUG"):
        c = ctx.states()["cap"]
        print("cap lin_vel", np.round(c.lin_vel, 4), "ang_vel", np.round(c.ang_vel, 3), "resting", c.resting(),
              "ended", ended, flush=True)
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
