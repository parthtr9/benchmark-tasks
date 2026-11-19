"""Scripted full-score demo for the wheelchair leg-rest task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Open loop on the fixed layout and on timing only: every target comes from the task file's dimensions (where the chair
stands, where its hinge pins and leg rests are), never from a part's pose or joint. The wheel locks are on from the
start.

Each arm works the leg rest in front of it, both at once. The open gripper comes down over the hanger tube and the
release lever's paddle, 13 cm ahead of the hinge pins, and closes on both: the squeeze presses the paddle toward the
tube and withdraws the latch tooth. The hand carries the grip a quarter turn outward about the pins' axis, lifts 15 cm
(the pins are 28 mm long; the hand sags under the load), moves 6 cm straight out, carries the leg rest over the floor
mat, lowers it until the footplate is on the floor, opens and rises; then both arms go home.

Negative variants (DEMO_VARIANT=...):
  one_footrest  only the chair's right leg rest (in front of the left arm) is taken off and set down
  swing_only    each leg rest is released and swung out a quarter turn on its pins, then let go
  no_lever      each hand takes the hanger tube below the paddle (the latch stays in) and tries the same swing and lift
  drop          as the demo, but each leg rest is raised 10 cm over the mat and let go about 25 cm above the floor
  held          as the demo, but the hands keep hold of the leg rests on the floor and stay there
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json (with a trace of the parts, read for
logging only) to the output folder.
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
VARIANTS = ("", "one_footrest", "swing_only", "no_lever", "drop", "held")
PAD_TIP = 0.013          # the pad tips are this far past the tool point
LIFT = 0.15              # footrest lifted this far to clear the 28 mm pins (the hand sags under the load)
DROP_RAISE = 0.10        # drop variant: raised this much over the mat before the hand opens
OUT = 0.06               # after the lift the leg rest is carried this far straight out from the chair
SET_DOWN = (0.57, 0.71)  # where the hand carries the grip to (chair x, |y|) before lowering (the leg rest tips over inward when let go)
CARRY = math.radians(55.0)   # the footrest's turn outward (about the vertical) while it is carried and set down
HOME = (0.30, 0.34, 0.72)   # where each hand ends (world x, |y|, z)
CLEAR_Z = 0.55           # height the hand rises to after setting a leg rest down


def down_rot(yaw: float) -> np.ndarray:
    """Tool rotation for a straight-down gripper whose fingers close along the world y axis turned by ``yaw``."""
    f = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([f, np.cross(z, f), z])


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


def smooth(x):
    return x * x * (3 - 2 * x)


def footrest(T, arm, sy: int, variant: str):
    """Release, swing out, lift off and set down the footrest on this arm's side."""
    up = np.array([0.0, 0.0, 1.0])
    pin = np.array([T.PIN_X, sy * T.RAIL_Y])
    if variant == "no_lever":
        gx = 0.64                                        # below the paddle: the tube alone
        gy, gz = sy * T.RAIL_Y, T.hanger_z(gx)
    else:
        gx, gz = T.GRIP_X, T.GRIP_Z                      # the tube and the paddle together
        gy = sy * (T.RAIL_Y - T.TUBE_R + T.PADDLE_Y[1]) / 2
    rel = np.array([gx, gy]) - pin

    def hand(angle, lift=0.0):
        """World tool point and rotation with the grip carried ``angle`` (rad, outward) round the pin axis."""
        a = sy * angle
        c, s = math.cos(a), math.sin(a)
        x, y = pin + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
        return T.world((x, y, gz + lift)), down_rot(math.pi + a)

    p0, r0 = hand(0.0)
    yield from arm.move(p0 + 0.12 * up, r0, 2.5, grip=1.0)
    yield from arm.move(p0, r0, 1.5)
    yield from arm.set_grip(0.0, 1.0)
    yield from wait(0.3)
    n = round(3.0 * CONTROL_HZ)
    for k in range(1, n + 1):
        arm.aim(*hand(math.pi / 2 * smooth(k / n)))
        yield
    yield from wait(0.3)
    if variant == "swing_only":
        yield from arm.set_grip(1.0, 0.8)
        yield from arm.move(arm.pos + 0.12 * up, arm.rot, 1.0)
        return
    p1, r1 = hand(math.pi / 2, LIFT)
    yield from arm.move(p1, r1, 3.0)
    yield from wait(0.5)
    yield from arm.move(p1 + T.world((0.0, sy * OUT, 0.0)) - T.world((0.0, 0.0, 0.0)), r1, 1.0)
    over = T.world((SET_DOWN[0], sy * SET_DOWN[1], gz + LIFT))
    r1 = down_rot(math.pi + sy * CARRY)             # turned part of the way back: keeps the wrist inside its range
    yield from arm.move(over, r1, 2.5)
    if variant != "drop":
        # the footplate's underside starts 81 mm above the floor: bring it down to the floor
        drop = LIFT + T.FOOTPLATE[0][2] - T.FOOTPLATE[1][2] / 2
        yield from arm.move(over - drop * up, r1, 2.0)
        yield from wait(0.3)
    if variant == "held":
        yield from wait(6.0)
        return
    if variant == "drop":
        yield from arm.move(over + DROP_RAISE * up, r1, 1.0)
    yield from arm.set_grip(1.0, 0.8)
    yield from wait(0.4)
    yield from arm.move(arm.pos + 0.12 * up, r1, 1.0)


def go_home(arm, side: int):
    yield from arm.move(np.array([HOME[0], side * HOME[1], HOME[2]]), down_rot(math.pi), 2.0, grip=1.0)


def leg_rest_side(T, arm, side: int, variant: str):
    """One arm's leg rest, then home. ``side`` is +1 for the left arm (world +y: the chair's right side)."""
    sy = -side                                           # the chair faces the arms: its left is at world -y
    if not (variant == "one_footrest" and side < 0):
        yield from footrest(T, arm, sy, variant)
    if variant == "held":
        return
    yield from arm.move(np.array([arm.pos[0], arm.pos[1], CLEAR_Z]), down_rot(math.pi), 1.0)
    yield from go_home(arm, side)


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    yield from both(leg_rest_side(task, ctx.left, 1, variant), leg_rest_side(task, ctx.right, -1, variant))
    yield from wait(1.5)


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])      # bay.py has put the bases on the mounting shelf
        self.right = Arm("right", station.BASES["right"])

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
    frames: dict[str, list] = {}
    trace = []
    min_gap = float("inf")

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def log_row(steps, info):                      # logging only: the demo itself never reads a part
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)
        chair = st["chair"]
        deg = lambda name: round(math.degrees(chair.hinges[name]), 2) if name in chair.hinges else None  # noqa: E731
        row = {"step": steps, "score": info.get("score"),
               "chair_mm": [round(1000 * float(v), 1) for v in chair.pos],
               "chair_yaw_deg": round(math.degrees(task._yaw(chair)), 2), "chair_tilt_deg": round(task._tilt(chair), 2),
               "chair_v_mm_s": [round(1000 * float(v), 1) for v in chair.lin_vel],
               "L_mm": [round(1000 * float(v)) for v in ctx.left.pos], "R_mm": [round(1000 * float(v)) for v in ctx.right.pos]}
        try:
            from partkit import gripper_points
            gp = gripper_points(emb._env)
            for side, key in (("left", "Lact_mm"), ("right", "Ract_mm")):
                pts = gp[side]
                row[key] = [round(1000 * float(v)) for v in pts[4]]
                row[key[0] + "gap_mm"] = round(1000 * (float(np.linalg.norm(pts[0] - pts[1])) / 2 - 0.006), 1)
        except Exception as exc:  # noqa: BLE001
            row["gp_err"] = repr(exc)
        for tag in ("l", "r"):
            row[tag] = {k: deg(f"{k}_{tag}") for k in ("wheel", "fork", "caster")}
            fr = st.get(f"footrest_{tag}")
            if fr is None:
                continue
            row[tag].update({"swing_deg": round(math.degrees(task.swing_angle(chair, fr, tag)), 1),
                             "held": task.held(fr, tag, task._STATE["points"], task._STATE["with_hand"].get(tag)),
                             "slip_mm": {side: round(1000 * task._STATE["slip"].get((tag, side), 0.0), 1)
                                         for side in ("left", "right")},
                             "fr_mm": [round(1000 * float(v), 1) for v in fr.pos],
                             "fr_yaw_deg": round(math.degrees(task._yaw(fr)), 1), "fr_tilt_deg": round(task._tilt(fr), 1),
                             "lever_deg": round(math.degrees(fr.hinges["lever"]), 1),
                             "pads_mm": {side: [round(1000 * min(task.footrest_surface_dist(fr, tag, p), 9.999), 1)
                                                for p in task._STATE["points"].get(side, [])[:4]]
                                         for side in ("left", "right")},
                             "on_pins": task.on_pins(chair, fr, tag), "lowest_mm": round(1000 * task.lowest(fr, tag), 1),
                             "on_mat": task.on_mat(fr, tag), "set_down": task.footrest_set_down(st, tag),
                             "credit": task.footrest_credit(st, tag),
                             "gap_mm": round(1000 * min(task.gripper_gap(fr, tag, task._STATE["points"]), 9.999), 1)})
        row["moved"], row["dropped"] = bool(task._STATE["moved"]), dict(task._STATE["dropped"])
        return row

    snap(obs)
    try:
        trace.append(log_row(0, {}))
        print(json.dumps(trace[-1]), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"trace failed at step 0: {exc!r}", flush=True)
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
        min_gap = min(min_gap, float(np.linalg.norm(ctx.left.pos - ctx.right.pos)))
        if steps % 2 == 0 or steps == 1:
            try:
                row = log_row(steps, info)
                trace.append(row)
                if steps % 50 == 0:
                    print(json.dumps(row), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"trace failed at step {steps}: {exc!r}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "variant": os.environ.get("DEMO_VARIANT", ""),
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "min_tool_point_gap_mm": round(1000 * min_gap, 1), "final": trace[-1] if trace else None, "trace": trace}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=1))
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
