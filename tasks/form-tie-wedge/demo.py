"""Scripted full-score demo for the form-tie wedge task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Each arm works the tie in front of it, both at once. An arm reaches forward with its fingers pointing at the form,
20 degrees below level, takes its wedge by the two flanges near the thick end, lifts it out of its stand, carries it
to the tie with the keyhole's wide part level with the head, shifts sideways by the amount its own finger joints show the wedge to be off-centre in
its grip, and moves it onto the waler so the head comes through the keyhole. It lowers the wedge without pushing (it
only ever carries the wedge's weight), so the slot runs onto the rod and the wedge comes to rest on the tie; then it
lets go and the wedge hangs there, slack.
The arm then closes its fingers, lays them level across the wedge's thick end and presses it down, a little harder
each step, until its own joints show that the wedge has stopped although the push kept growing (the tie is tight) or
the push has reached about 6 N; then it lifts off.

Open loop on the scene: every position comes from the task file's fixed layout. The arm is compliant, so it corrects
its own sag and tilt from its own joint readings, and it feels the end of the drive the same way; nothing is read from
the simulated objects.

Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder.
Negative variants (each must score below 1):
    DEMO_VARIANT=hang_only   wedges slipped over the heads and let go without being driven (scores 0.25)
    DEMO_VARIANT=one_wedge   only the left arm fastens its tie (scores 0.5)
    DEMO_VARIANT=held        both driven tight but the fingers stay pressed on the wedges (scores 0.75)
DEMO_DEBUG=1 logs the wedges and ties every 20 steps (logging only).
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
PITCH = 0.35           # fingers point at the form, this far below level (rad)
GRIP_Z = 0.140         # where the fingers take the wedge, from its thin end (m)
GRIP_X = 0.012         # ... and how far the tool point sits before the wedge's bearing plane
PRESS = 0.003          # the wedge is held this far "into" the waler while it is lowered and driven (under 1 N)
LOWER = 0.062          # after the head is through the keyhole the wanted height goes down this far, only carrying
HANG_DRAW = 0.0012     # about how far a wedge hanging by its own weight draws the tie (for where its top is)
TOP_X = 0.020          # the closed fingers press 20 mm out from the waler face, across the web's top edge
DRIVE_SPEED = 0.010    # how fast the wanted height goes down while driving (m/s)
PUSH_LIMIT = 0.045     # most the arm adds to its command to push the wedge down (m; about 6 N)
STALL_PUSH = 0.008     # a stall: no travel while the push grew by this much command (m; about 1 N)
MIN_PUSH = 0.030       # ... and only counts once the push is this large (m; about 4 N: what seating takes)
PUSH_BELOW = 0.010     # the tool point may go this far below a tight wedge's top before the drive gives up


def _rot_axis(axis, angle) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def forward_grips(pitch: float) -> list:
    """Tool rotations with the fingers pointing forward (+x), ``pitch`` below level, closing across the wall (y):
    the downward grip (turned 0 or 180 degrees) tipped up about a tool axis. Each arm takes the one its joints reach."""
    from yam_arena.demo import grip_rotation

    want = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    out = []
    for yaw in (0.0, math.pi):
        down = np.asarray(grip_rotation(yaw), float)
        along = down[:, 2] if down[2, 2] < 0 else -down[:, 2]          # the fingers' direction in the downward grip
        flip = 1.0 if down[2, 2] < 0 else -1.0
        for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
            for sense in (1.0, -1.0):
                cand = down @ _rot_axis(axis, sense * (math.pi / 2 - pitch))
                if float(np.linalg.norm(flip * cand[:, 2] - want)) < 1e-6 and abs(along[2] + 1.0) < 1e-6:
                    out.append(cand)
    if not out:   # tool axes not as expected: x across the wall, z along the fingers
        out.append(np.array([[0.0, math.sin(pitch), math.cos(pitch)], [1.0, 0.0, 0.0],
                             [0.0, math.cos(pitch), -math.sin(pitch)]]))
    return out


def reachable_grip(arm, pos, pitch: float) -> np.ndarray:
    """The forward grip this arm's joints reach best at ``pos`` (IK error from its ready pose)."""
    from yam_arena import ik

    best, best_err = None, float("inf")
    for rot in forward_grips(pitch):
        _, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), arm.q, base_pos=arm.base, iters=200)
        if pe + 0.1 * re < best_err:
            best, best_err = rot, pe + 0.1 * re
    return best


def _rotvec(r) -> np.ndarray:
    """Rotation vector (axis times angle) of a rotation matrix."""
    c = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(c)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    n = float(np.linalg.norm(v))
    return np.zeros(3) if n < 1e-9 else v / n * angle


def _exp(v) -> np.ndarray:
    a = float(np.linalg.norm(v))
    return np.eye(3) if a < 1e-9 else _rot_axis(v, a)


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF) toward a wanted tool pose. The arm is compliant (it sags
    about a centimetre per newton), so on request it corrects itself from its own joint readings: it adds up the
    difference between the wanted and the measured tool pose and shifts its command by that."""

    def __init__(self, name: str, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.want_pos, self.want_rot = ik.tcp_pose(self.q, self.base)      # the wanted tool pose
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.off = np.zeros(3)         # position correction added to the command (m)
        self.roff = np.zeros(3)        # rotation correction (rotation vector, rad)

    def _solve(self):
        from yam_arena import ik

        pos, rot = self.want_pos + self.off, _exp(self.roff) @ self.want_rot
        q, pe, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q = q
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def correct(self, ctx, axes=(0, 1, 2), limit=0.05, turn=True):
        """One step of self-correction from the arm's own joints, on the given position axes (and the tool's turn)."""
        pos, rot = ctx.actual_pose(self.name)
        err = self.want_pos - pos
        for a in axes:
            self.off[a] = float(np.clip(self.off[a] + 0.2 * err[a], -limit, limit))
        if turn:
            self.roff = np.clip(self.roff + 0.2 * _rotvec(self.want_rot @ rot.T), -0.5, 0.5)

    def move(self, pos, rot, duration: float, grip: float | None = None, ctx=None, axes=(0, 1, 2), limit=0.05,
             turn=True):
        """Generator: take the wanted tool pose to ``pos``/``rot``; ``grip`` 1 open .. 0 closed (None keeps it).
        With ``ctx`` the arm corrects itself on ``axes`` as it goes; without, the last correction is kept."""
        from yam_arena import ik

        p0, r0, g0 = self.want_pos.copy(), self.want_rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self.want_pos, self.want_rot = p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s)
            if ctx is not None:
                self.correct(ctx, axes, limit, turn)
            self._solve()
            yield

    def set_grip(self, grip: float, duration: float = 0.8):
        yield from self.move(self.want_pos, self.want_rot, duration, grip)

    def lower_carrying(self, ctx, drop: float, duration: float):
        """Generator: lower the wanted height by ``drop`` while only ever carrying: the command never goes below
        where the joints say the tool is, so the arm cannot push down on what it holds."""
        z0 = float(self.want_pos[2])
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            pos, _ = ctx.actual_pose(self.name)
            want = z0 - drop * k / n
            self.want_pos[2] = max(want, float(pos[2]) - float(self.off[2]))
            self._solve()
            yield

    def push_down(self, ctx, lowest: float, speed: float, limit: float):
        """Generator: push down: the wanted height falls at ``speed`` (never below ``lowest``) while the arm corrects
        its pose from its own joints, adding up to ``limit`` of extra command. It stops as soon as its own joints show
        a stall: the tool has not gone down 0.2 mm in the last 0.75 s although the push grew by STALL_PUSH in that
        time and is above MIN_PUSH. It also stops 0.5 s after the push has reached the limit, or once the wanted height is reached."""
        full, zs, pushes = 0, [], []
        for _ in range(int(20 * CONTROL_HZ)):
            if self.off[2] > -limit + 1e-6:        # at the limit the wanted height waits: the push grows no further
                self.want_pos[2] = max(lowest, float(self.want_pos[2]) - speed / CONTROL_HZ)
            self.correct(ctx, axes=(0, 1, 2), limit=limit)
            self._solve()
            yield
            pos, _ = ctx.actual_pose(self.name)
            zs.append(float(pos[2]))
            pushes.append(float(self.off[2]))
            full = full + 1 if self.off[2] <= -limit + 1e-6 else 0
            n = 3 * CONTROL_HZ // 4
            stalled = (len(zs) > n and zs[-n - 1] - zs[-1] < 0.0002 and pushes[-n - 1] - pushes[-1] > STALL_PUSH
                       and self.off[2] < -MIN_PUSH)
            reached = self.want_pos[2] <= lowest + 1e-9 and abs(self.want_pos[2] - pos[2]) < 0.001
            if stalled or reached or full >= CONTROL_HZ // 2:
                return

    def relax(self, ctx, seconds: float = 1.0):
        """Generator: stop pushing or carrying: bring the commanded height back to where the joints say the tool is.
        The sideways corrections are kept, so the tool does not shift across the part it still holds."""
        for _ in range(max(1, round(seconds * CONTROL_HZ))):
            pos, _ = ctx.actual_pose(self.name)
            self.want_pos[2] = float(pos[2])
            self.off[2] *= 0.7
            self._solve()
            yield


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def together(*gens):
    """Run several generators in step until all are finished."""
    live = list(gens)
    while live:
        for g in list(live):
            try:
                next(g)
            except StopIteration:
                live.remove(g)
        if live:
            yield


def fasten(arm, ctx, T, y: float, variant: str):
    """One arm fastens the tie at ``y`` with the wedge standing in the stand in front of it."""
    f = np.array([math.cos(PITCH), 0.0, -math.sin(PITCH)])                 # along the fingers

    side = [0.0]       # how far to one side of the tool point the fingers turned out to hold the wedge (m)

    def tcp(plane_x: float, thin_end_z: float) -> np.ndarray:
        """Tool point for a wedge whose bearing plane is at ``plane_x`` and thin end ``thin_end_z`` above the table."""
        return np.array([plane_x - GRIP_X, y + side[0], T.TOP + thin_end_z + GRIP_Z])

    in_stand = tcp(T.HOLD_X, 0.0005)
    head_tip = T.FACE_X - T.HEAD_GAP0 - T.HEAD_T                           # the outer face of the tie's head
    level = T.TIE_Z - T.ENTRY_Z                                            # thin end height: keyhole on the head
    # Thin end height once the slack is gone: the head then sits where the wedge is HEAD_GAP0 + SLACK deep.
    z_tight = T.TIE_Z - T.wedge_z_at(T.HEAD_GAP0 + T.SLACK)
    rot = reachable_grip(arm, tcp(T.FACE_X + PRESS, level), PITCH)

    # 1. Reach in from the arm's side along the fingers and take the wedge by its flanges.
    yield from arm.move(in_stand - 0.040 * f, rot, 3.0, grip=1.0)
    yield from arm.move(in_stand, rot, 1.5, ctx=ctx)
    yield from arm.move(in_stand, rot, 0.5, ctx=ctx)
    yield from arm.set_grip(0.0, 1.0)
    yield from wait(0.3)
    # 2. Lift it out of the stand (the arm takes up the wedge's weight as it goes) and carry it to 10 mm in front
    #    of the head, keyhole level with the head.
    yield from arm.move(tcp(T.HOLD_X, T.FOOT_H + 0.025), rot, 3.0, ctx=ctx)
    yield from arm.move(tcp(head_tip - 0.010, level), rot, 3.0, ctx=ctx)
    yield from arm.move(tcp(head_tip - 0.010, level), rot, 1.5, ctx=ctx)
    # The two fingers close independently, so the wedge can sit a few millimetres to one side of the tool point.
    # The arm reads that from its own two finger joints and shifts sideways to put the wedge's slot on the tie's line.
    for _ in range(2):
        side[0] = ctx.finger_offset(arm.name)
        yield from arm.move(tcp(head_tip - 0.010, level), rot, 0.7, ctx=ctx)
    # 3. Onto the waler: the head comes through the keyhole's wide part and the nose meets the waler face. Height
    #    and side position stay corrected; the push into the waler is a fixed 3 mm of command, not corrected.
    yield from arm.move(tcp(T.FACE_X + PRESS, level), rot, 1.5, ctx=ctx, axes=(1, 2))
    side[0] = ctx.finger_offset(arm.name)              # centre the slot on the tie once more, on the waler
    yield from arm.move(tcp(T.FACE_X + PRESS, level), rot, 0.7, ctx=ctx, axes=(1, 2))
    # 4. Lower it, only carrying, until the tie takes its weight (the slot runs onto the rod and the ramp comes to
    #    rest on the head; the arm cannot push, so it simply stops there), and let go: the wedge hangs, slack.
    yield from arm.lower_carrying(ctx, LOWER, 4.0)
    yield from arm.relax(ctx, 1.0)
    yield from arm.set_grip(1.0, 1.0)
    yield from wait(1.0)
    if variant == "hang_only":
        yield from arm.move(arm.want_pos - 0.08 * f, rot, 1.5)
        return
    # 5. Draw back, close the fingers and bring them, level, over the wedge's thick end.
    z_hang = T.TIE_Z - T.wedge_z_at(T.HEAD_GAP0 + HANG_DRAW)                # thin end height of a hanging wedge
    over = np.array([T.FACE_X - TOP_X, y, T.TOP + z_hang + T.W_LEN + 0.045])
    flat = reachable_grip(arm, over, 0.0)
    yield from arm.move(arm.want_pos - 0.05 * f, rot, 1.0)
    yield from arm.move(over, flat, 2.5, grip=0.0, ctx=ctx)
    # 6. Drive it: press down on the thick end with the closed fingers, harder and harder, until the arm's own
    #    joints show the wedge has stopped (the tie is tight) or the push has reached about 6 N.
    lowest = T.TOP + z_tight + T.W_LEN - PUSH_BELOW
    yield from arm.push_down(ctx, lowest, DRIVE_SPEED, PUSH_LIMIT)
    if variant == "held":
        return
    # 7. Stop pushing and lift the fingers straight off, then draw back.
    yield from arm.relax(ctx, 1.0)
    yield from arm.move(arm.want_pos + np.array([0.0, 0.0, 0.03]), flat, 1.0)
    yield from arm.move(arm.want_pos + np.array([-0.03, 0.0, 0.05]), flat, 1.5)


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    variant = os.environ.get("DEMO_VARIANT", "")
    jobs = [fasten(ctx.left, ctx, task, task.TIE_Y, variant)]
    if variant != "one_wedge":
        jobs.append(fasten(ctx.right, ctx, task, -task.TIE_Y, variant))
    yield from together(*jobs)


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def actual_pose(self, name: str):
        """The arm's measured tool position and rotation (world), from its own joint positions."""
        from yam_arena import ik

        arm_ids = self.emb._ids[f"{name}_robot"][0]
        q = self.emb._base.scene[f"{name}_robot"].data.joint_pos.torch[0][arm_ids].cpu().numpy().astype(np.float64)
        pos, rot = ik.tcp_pose(q, self.left.base if name == "left" else self.right.base)
        return np.asarray(pos, float), np.asarray(rot, float)

    def finger_offset(self, name: str) -> float:
        """How far the held part sits to the side of the tool point (m, +y), from the arm's own two finger joints:
        half the difference of their readings (0 when the fingers are closed evenly)."""
        robot = self.emb._base.scene[f"{name}_robot"]
        jp, jn = robot.data.joint_pos.torch[0].cpu().numpy(), list(robot.joint_names)
        return 0.5 * (float(jp[jn.index("joint7")]) - float(jp[jn.index("joint8")]))

    def fingers(self, name: str) -> list:
        """The arm's two finger joint readings (mm), for the debug log."""
        robot = self.emb._base.scene[f"{name}_robot"]
        jp, jn = robot.data.joint_pos.torch[0].cpu().numpy(), list(robot.joint_names)
        return [round(1000 * float(jp[jn.index(j)]), 1) for j in ("joint7", "joint8") if j in jn]

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

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def trace(step, info):
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)
        form = st["form"]
        msg = f"step {step} score {info.get('score')}"
        for side, arm in (("l", ctx.left), ("r", ctx.right)):
            w = st[f"wedge_{side}"]
            head = w.to_local(task.head_world(form, side))
            msg += (f" | {side}: wedge {np.round(1000 * (w.pos - [0, 0, task.TOP]), 1)} up {w.rotation()[2, 2]:.3f}"
                    f" head_in_wedge_mm {np.round(1000 * head, 1)} drawn_mm {1000 * task.drawn(form, side):.2f}"
                    f" on_tie {task.on_tie(form, w, side)} tight {task.tight(form, w, side)}"
                    f" wedge_rpy_deg {np.round(np.degrees(_rotvec(w.rotation())), 1)}"
                    f" tcp_err_mm {np.round(1000 * (arm.want_pos - ctx.actual_pose(arm.name)[0]), 1)}"
                    f" tool_err_deg {np.round(np.degrees(_rotvec(arm.want_rot @ ctx.actual_pose(arm.name)[1].T)), 1)}"
                    f" off_mm {np.round(1000 * arm.off, 1)} fingers_mm {ctx.fingers(arm.name)}")
        msg += (f" | form_moved_mm {1000 * np.linalg.norm(form.pos - task._STATE['form0']):.2f}"
                f" released {task._STATE['released']}")
        print(msg, flush=True)

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
        if debug and steps % 20 == 0:
            trace(steps, info)
        elif steps % 100 == 0:
            print(f"step {steps} score {info.get('score')}", flush=True)
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
