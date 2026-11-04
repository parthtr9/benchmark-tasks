"""Scripted full-score demo for the two-hand press task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The left arm picks the blank up by its top edges, sets it in the die and backs out of the press. Both arms then come
down onto their buttons in the same move, so the buttons go down within a step of each other, hold them for 3 s while
the ram comes down onto the blank, and let go together. Writes one MP4 per camera, setup.png (the first top-camera
frame) and result.json to the output folder.

Negative variants (each must score below 1):
    DEMO_VARIANT=one_button   only the right button is pushed: the ram stays up (0.25)
    DEMO_VARIANT=late         the left button is pushed 1 s after the right: the control refuses (0.25)
    DEMO_VARIANT=no_blank     both buttons pushed with the die empty: the ram cycles on nothing (0)
    DEMO_VARIANT=weighted     the blank is set on the left button to hold it down and the right gripper pushes its
                              button while the left gripper waits in the die: the control refuses a button held
                              before the other (anti-tie-down), the ram stays up, and the die is empty (0)
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


# Tool point heights (above the table): the pad tips reach 13 mm past the tool point.
PAD_PAST_TCP = 0.013


def plan(task):
    """The fixed points of the demo, in the world frame."""
    T = task
    top = T.TOP
    press = np.array([*T.PRESS_XY, top])
    blank_top = top + T.BLANK_H / 1000
    seat = press + T.SEAT                                   # the seated blank's centre
    seat_top = seat[2] + T.BLANK_H / 2000
    grip_depth = 0.006                                      # the pads hold the blank's top 6 mm
    button_top = top + (T.B + 40) / 1000                    # the buttons stand 40 mm above the base plate
    push = T.BUTTON_TRAVEL + 0.004                          # command 4 mm past the 8 mm stop: the arm is compliant
    return dict(
        pick=np.array([*T.BLANK_XY, blank_top - grip_depth + PAD_PAST_TCP]),
        # 2 mm behind the nominal seat: a short set-down stays clear of the anvil and still lies in the die.
        place=np.array([seat[0] + 0.002, seat[1], seat_top + 0.001 - grip_depth + PAD_PAST_TCP]),
        button_left=np.array([press[0] + T.BUTTON_X / 1000, press[1] + T.BUTTON_Y / 1000, button_top]),
        button_right=np.array([press[0] + T.BUTTON_X / 1000, press[1] - T.BUTTON_Y / 1000, button_top]),
        push=push,
        clear=np.array([press[0] + T.BUTTON_X / 1000 - 0.04, 0.0, top + 0.16]),  # in front of the die, high
    )


def _grip_offset(task, ctx, arm) -> np.ndarray:
    """How far (x, y) the tool point is from the held blank's centre, to add to a place target. Zero offline."""
    states = getattr(ctx, "states", None)
    if states is None:
        return np.zeros(3)
    blank = states()["blank"]
    d = arm.pos - np.asarray(blank.pos, float)
    return np.array([d[0], d[1], 0.0])


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    L, R = ctx.left, ctx.right
    P = plan(task)
    rot = grip_rotation(0.0)
    variant = os.environ.get("DEMO_VARIANT", "")
    lift = np.array([0, 0, 0.10])
    pad = PAD_PAST_TCP
    # Pushing a button: the closed pads' tips go ``push`` below the button's top.
    bl_hover, br_hover = P["button_left"] + [0, 0, 0.06], P["button_right"] + [0, 0, 0.06]
    bl_down = P["button_left"] + [0, 0, pad - P["push"]]
    br_down = P["button_right"] + [0, 0, pad - P["push"]]

    # 1. Left arm: pick the blank up by its top edges and set it in the die (or, weighted: on the left button).
    if variant != "no_blank":
        # Pinch across y, the die's tight direction (3 mm a side between the rails): the blank then cannot slide
        # sideways in the pads, and any slip along the pads (x) has the die's length to spare.
        brot = grip_rotation(math.pi / 2)
        yield from L.move(P["pick"] + lift, brot, 2.5, grip=1.0)
        yield from L.move(P["pick"], brot, 1.5)
        yield from L.set_grip(0.0, 1.0)
        yield from wait(0.3)
        yield from L.move(P["pick"] + lift, brot, 1.5)
        if variant == "weighted":
            on_button = P["button_left"] + [0, 0, task.BLANK_H / 1000 - 0.006 + pad]
            yield from L.move(on_button + [0, 0, 0.06], brot, 2.5)
            yield from L.move(on_button, brot, 1.5)
        else:
            # Closed loop: the blank rarely sits centred between the pads, so aim the place by where it really is.
            place = P["place"] + _grip_offset(task, ctx, L)
            yield from L.move(place + lift, brot, 3.0)
            yield from wait(0.5)
            place = P["place"] + _grip_offset(task, ctx, L)
            yield from L.move(place + [0, 0, 0.015], brot, 1.5)
            yield from wait(0.4)
            # Re-aim just above the rails, where the blank has stopped swinging, then set it down slowly.
            place = P["place"] + _grip_offset(task, ctx, L)
            yield from L.move(place + [0, 0, 0.015], brot, 0.5)
            yield from L.move(place, brot, 1.5)
        yield from L.set_grip(1.0, 1.0)
        yield from wait(0.3)
        yield from L.move(L.pos + lift, brot, 1.5)

    # 2. Both arms over their buttons, fingers closed.
    if variant == "weighted":
        # The left gripper reaches back into the die while the right pushes its button: the ram must not come down.
        in_die = P["place"] + [0, 0, 0.02]
        yield from both(L.move(in_die + lift, rot, 2.5, grip=0.0), R.move(br_hover, rot, 2.5, grip=0.0))
        yield from both(L.move(in_die, rot, 2.0), R.move(br_down, rot, 1.5))
        yield from wait(4.0)
        yield from both(L.move(in_die + lift, rot, 1.5), R.move(br_hover, rot, 1.5))
        yield from wait(1.5)
        return
    yield from both(L.move(bl_hover, rot, 2.5, grip=0.0), R.move(br_hover, rot, 2.5, grip=0.0))

    # 3. Push: both buttons go down in the same move (one_button: only the right; late: the left 1 s after).
    if variant == "one_button":
        yield from R.move(br_down, rot, 1.5)
    elif variant == "late":
        yield from R.move(br_down, rot, 1.5)
        yield from wait(1.0)
        yield from L.move(bl_down, rot, 1.5)
    else:
        yield from both(L.move(bl_down, rot, 1.5), R.move(br_down, rot, 1.5))
    yield from wait(3.5)   # the ram takes about 2.1 s to come down onto the blank

    # 4. Let go together; the ram returns.
    yield from both(L.move(bl_hover, rot, 1.5), R.move(br_hover, rot, 1.5))
    yield from wait(3.0)   # the ram returns in about 1.6 s


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
                c, tilt, seated = task.blank_in_die(st["press"], st["blank"])
                sl = {k: round(1000 * v, 1) for k, v in st["press"].sliders.items()}
                msg += (f" blank_mm {np.round(1000 * c, 1)} tilt {tilt:.1f} seated {seated} sliders_mm {sl}"
                        f" ctrl {dict((k, task._CTRL[k]) for k in ('enabled', 'refused', 'target'))}"
                        f" L_mm {np.round(1000 * ctx.left.pos, 1)} failed {task._STATE['failed']}")
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "failed": task._STATE.get("failed"),
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
