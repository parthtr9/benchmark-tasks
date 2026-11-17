"""Scripted full-score demo for the wine-cubby task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Every target comes from the fixed layout in the task file and from timing. The only feedback used is each arm's own
joint readings: while an arm carries or pushes the 1.2 kg bottle, its command is trimmed until its measured tool point
(forward kinematics of its joint positions) is on the planned one, which takes out the sag under the load. No object
pose is read.

The right arm takes the bottle at its balance point (pads across the body), lifts it, turns it 60 degrees so its
base points at the rack, lines it up with the empty middle cubby and feeds it in base first until the fingers are
6 mm from the rack (7 cm in: the gripper is wider than the cubby). The bottle's weight is then still outside the cubby,
so the right hand keeps holding while the left hand takes the neck; the right hand lets go and moves aside, the left
hand slides the bottle in until its own hand is at the rack (20 cm in), lets go and leaves, and the right hand comes
back and pushes the cork end with closed fingers until the base is on the back panel.

Negative variants (DEMO_VARIANT=...):
  body_only   the right arm feeds the bottle in 7 cm and lets go: it tips out of the cubby, neck onto the table (0)
  no_push     the left hand lets go after sliding it 20 cm in and nothing pushes it home: 10 cm of neck still out (0.5)
  no_release  as the demo, but the right fingers stay on the cork end (0.75)
  lift_hold   a payload check, not a negative: lift the bottle 8.5 cm, hold it 3 s, lay it back on the mat (0)
Writes one MP4 per camera, setup.png and result.json to the output folder; the log carries a trace every 10 steps.
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
VARIANTS = ("", "body_only", "no_push", "no_release", "lift_hold")
GRIP_BELOW = 0.022        # body grip: the tool point this far below the bottle's axis (the bottle's top 11 mm under the palm)
NECK_BELOW = 0.008        # neck grip: the tool point this far below the axis
PUSH_BELOW = 0.030        # closed-finger push: the tool point this far below the axis, so the finger bodies cover the cork end
#                           from 9 mm under its centre up (set higher, they ride up over it: run 7)
FINGER_HALF = 0.034       # the finger bodies are 68 mm wide: this far from the tool point along the bottle
STANDOFF = 0.006          # the right hand's finger bodies stay this far from the rack's front face
LEFT_STANDOFF = 0.034     # the left hand's: its wrist meets the rack when the fingers are 22 mm from it (run 5)
NECK_GRIP_X = -0.122      # where on the bottle's axis (part frame) the left hand grips: on the foil capsule
NECK_OPEN = 0.48          # grip value: pads about 45 mm apart
ABOVE = 0.004             # the bottle is fed in this far above its resting height
KI, CORR_MAX, YAW_MAX = 0.2, 0.045, 0.35  # joint-feedback trim: gain per step and limits (m, rad)
PUSH_CORR_MAX = 0.10      # trim limit while a hand slides or pushes the bottle along the cubby floor (about 10 N)
XYZ, X_ONLY = np.ones(3), np.array([1.0, 0.0, 0.0])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each starting where the last command left it. With ``servo``
    the command is trimmed by the arm's own joint readings so the measured tool point follows the plan under load."""

    def __init__(self, name: str, base_pos, read=None):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.read = read                   # () -> measured tool point from the arm's joint positions, or None
        self.corr = np.zeros(3)
        self.corr_max = CORR_MAX
        self.yaw_corr = 0.0
        self.max_corr = 0.0

    def _solve(self, pos, rot, servo=None):
        from yam_arena import ik

        if servo is not None and self.read is not None:
            meas = self.read()
            if meas is not None:
                m_pos, m_rot = meas
                self.corr = np.clip(self.corr + KI * servo * (self.pos - m_pos), -self.corr_max, self.corr_max)
                self.max_corr = max(self.max_corr, float(np.abs(self.corr).max()))
                a, b = np.asarray(m_rot)[:, 0], self.rot[:, 0]     # measured and planned closing axes
                err = math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
                self.yaw_corr = float(np.clip(self.yaw_corr + KI * err, -YAW_MAX, YAW_MAX))
        q, pe, _ = ik.solve(np.asarray(pos, float) + self.corr, ik.matrix_to_quat_xyzw(_rz(self.yaw_corr) @ rot),
                            self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None, servo=None):
        """Generator: go to ``pos``/``rot`` (world tool point); ``grip`` 1 open .. 0 closed (None keeps it)."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s), servo)
            yield

    def hold(self, duration: float, grip: float | None = None, servo=None):
        yield from self.move(self.pos, self.rot, duration, grip, servo)

    def relax(self, duration: float = 0.6):
        """Generator: take the trim off again (after the load is gone)."""
        c0, y0 = self.corr.copy(), self.yaw_corr
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self.corr, self.yaw_corr = c0 * (1 - k / n), y0 * (1 - k / n)
            self._solve(self.pos, self.rot)
            yield


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


def chain(*gens):
    for g in gens:
        yield from g


def closing_yaws(axis: str):
    """The two grip yaws whose fingers close along world ``axis`` ("x" or "y"), gripper pointing straight down."""
    from yam_arena.demo import grip_rotation

    want = np.array([1.0, 0.0, 0.0]) if axis == "x" else np.array([0.0, 1.0, 0.0])
    out = [y for y in (0.0, math.pi / 2, math.pi, -math.pi / 2) if abs(float(grip_rotation(y)[:, 0] @ want)) > 0.99]
    if len(out) != 2:
        raise RuntimeError(f"no grip yaw closes along {axis}")
    return out


def ik_error(arm, poses):
    """Worst IK error (position + 0.05 x rotation) over a list of (pos, rot), solved one after the other."""
    from yam_arena import ik

    q, worst = arm.q, 0.0
    for pos, rot in poses:
        q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), q, base_pos=arm.base, iters=200)
        worst = max(worst, pe + 0.5 * re)
    return worst


def plan(T):
    """Every tool-point target, from the task file's fixed layout (world frame, m)."""
    col, row = T.EMPTY[0]                              # the empty middle cubby
    y_c, h = T.cubby_axis(col, row)
    axis_z = T.TOP + h                                 # the axis of a bottle lying in that cubby
    com_to_base = T.HALF - T.COM_X                     # the body grip is at the balance point
    p = {}
    p["pick"] = np.array([T.START_XY[0] + T.COM_X * math.cos(T.START_YAW), T.START_XY[1] + T.COM_X * math.sin(T.START_YAW),
                          T.TOP + T.START_Z - GRIP_BELOW])
    p["carry_dz"] = 0.085
    p["pre"] = np.array([T.FRONT - 0.030 - com_to_base, y_c, axis_z + ABOVE - GRIP_BELOW])   # base 3 cm from the rack
    p["fed"] = np.array([T.FRONT - STANDOFF - FINGER_HALF, y_c, axis_z + ABOVE - GRIP_BELOW])
    fed_origin_x = p["fed"][0] - T.COM_X               # the bottle's middle once fed in by the right hand
    p["neck"] = np.array([fed_origin_x + NECK_GRIP_X, y_c, axis_z - NECK_BELOW])
    p["slid"] = np.array([T.FRONT - LEFT_STANDOFF - FINGER_HALF, y_c, axis_z - NECK_BELOW])
    slid_cork_x = p["slid"][0] - NECK_GRIP_X - T.HALF
    p["behind"] = np.array([slid_cork_x - FINGER_HALF - 0.025, y_c, axis_z - PUSH_BELOW])
    seated_cork_x = T.FRONT + T.DEPTH - T.LENGTH
    p["home"] = np.array([seated_cork_x - FINGER_HALF + 0.004, y_c, axis_z - PUSH_BELOW])    # 4 mm past: a firm push
    p["depth_fed"] = p["fed"][0] + com_to_base - T.FRONT
    p["depth_slid"] = p["slid"][0] - NECK_GRIP_X + T.HALF - T.FRONT
    return p


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    L, R = ctx.left, ctx.right
    P = plan(T)
    up = np.array([0.0, 0.0, 1.0])
    lay = _rz(T.START_YAW)                             # how the bottle lies on the mat, from the rack's depth axis

    # Right hand: pads across the bottle as it lies on the mat; turned with it until they close along y at the rack.
    def choose(arm, poses_for):
        """Of the two grips that close along y, the one whose tool y axis points away from the rack (the wrist is not
        the same on both sides, and this way round the hand can come within 6 mm of the rack); the other one only if
        this one is out of reach."""
        cands = sorted(((ik_error(arm, poses_for(r)), r) for r in map(grip_rotation, closing_yaws("y"))),
                       key=lambda c: (c[1][0, 1] > 0.0, c[0]))
        pick = cands[0] if cands[0][0] < 0.02 else min(cands, key=lambda c: c[0])
        print(f"GRIP {arm.name}: tool y along x {pick[1][0, 1]:+.2f}, IK error {pick[0]:.4f} "
              f"(other: {[round(c[0], 4) for c in cands]})", flush=True)
        return pick[1]

    rot_in = choose(R, lambda r1: [(P["pick"] + 0.08 * up, lay @ r1), (P["pick"], lay @ r1),
                                   (P["pre"] + P["carry_dz"] * up, r1), (P["pre"], r1), (P["fed"], r1)])
    rot_pick = lay @ rot_in
    # Left hand: pads close along y on the neck.
    rot_l = choose(L, lambda r0: [(P["neck"] + 0.07 * up, r0), (P["neck"], r0), (P["slid"], r0), (P["home"], r0)])

    # 1. Right: over the bottle's balance point, down, close, lift.
    yield from R.move(P["pick"] + 0.08 * up, rot_pick, 2.5, grip=1.0, servo=XYZ)
    yield from R.move(P["pick"], rot_pick, 1.5, servo=XYZ)
    yield from R.hold(0.8, servo=XYZ)
    before = R.read() if R.read is not None else None
    yield from R.hold(1.0, grip=0.0)
    yield from wait(0.3)
    # The finger that touches first pushes the hand aside until the other one arrives (the 1.2 kg bottle stays put on
    # the rubber mat), so the hand's shift while closing, read from its joints, is where the bottle sits between the
    # fingers. Every later target of this hand allows for it.
    hand = np.zeros(3)
    if before is not None:
        shift = R.read()[0] - before[0]
        hand = rot_pick.T @ np.array([shift[0], shift[1], 0.0])
        print(f"HAND_SHIFT_MM {np.round(1000 * shift, 1).tolist()} in the gripper frame {np.round(1000 * hand, 1).tolist()}",
              flush=True)
    off_pick, off_in = rot_pick @ hand, rot_in @ hand
    off_pick[2] = off_in[2] = 0.0
    R.pos = R.pos + off_pick                       # the plan now starts from where the hand really is
    yield from R.move(P["pick"] + off_pick + P["carry_dz"] * up, rot_pick, 2.0, servo=XYZ)
    if variant == "lift_hold":
        yield from R.hold(3.0, servo=XYZ)
        yield from R.move(P["pick"] + off_pick + 0.002 * up, rot_pick, 2.0, servo=XYZ)
        yield from R.hold(0.6, grip=1.0)
        yield from R.move(R.pos + 0.08 * up, rot_pick, 1.2)
        return
    # 2. Turn it until its base points at the rack, on the way to the cubby's centre line, then down to its height.
    yield from R.move(P["pre"] + off_in + P["carry_dz"] * up, rot_in, 4.0, servo=XYZ)
    yield from R.hold(0.5, servo=XYZ)
    yield from R.move(P["pre"] + off_in, rot_in, 2.0, servo=XYZ)
    yield from R.hold(1.0, servo=XYZ)
    # 3. Feed it in base first until the fingers are 6 mm from the rack; then set its base down on the cubby floor.
    yield from R.move(P["fed"] + off_in, rot_in, 3.0, servo=XYZ)
    yield from R.hold(0.8, servo=XYZ)
    yield from R.move(P["fed"] + off_in - ABOVE * up, rot_in, 0.6)
    if variant == "body_only":
        yield from R.hold(0.6, grip=1.0)
        yield from R.move(R.pos + 0.10 * up, rot_in, 1.2)
        yield from wait(2.0)
        return
    # 4. Left: take the neck while the right hand still holds the bottle.
    yield from L.move(P["neck"] + 0.07 * up, rot_l, 2.0, grip=NECK_OPEN, servo=XYZ)
    yield from L.move(P["neck"], rot_l, 1.2, servo=XYZ)
    yield from L.hold(0.5, servo=XYZ)
    before = L.read() if L.read is not None else None
    yield from L.hold(0.8, grip=0.0)
    yield from wait(0.3)
    side = np.zeros(3)                             # as for the right hand: where the neck sits between the fingers
    if before is not None:
        side[1] = L.read()[0][1] - before[0][1]
        print(f"NECK_SHIFT_MM {1000 * side[1]:.1f}", flush=True)
    L.pos = L.pos + side
    # 5. Right lets go and leaves to its own side; only then left slides the bottle in until its own fingers are
    #    6 mm from the rack.
    yield from R.hold(0.6, grip=1.0)
    yield from R.move(R.pos + 0.09 * up, rot_in, 1.5)
    yield from R.relax()
    yield from R.move(np.array([P["fed"][0] - 0.07, -0.22, R.pos[2]]), rot_in, 2.2)
    L.corr_max = PUSH_CORR_MAX
    yield from L.move(P["slid"] + side, rot_l, 3.5, servo=X_ONLY)
    yield from L.hold(0.4, servo=X_ONLY)
    # 6. Left stops pushing, lets go of the neck and goes back to its own side.
    yield from L.relax(0.8)
    L.corr_max = CORR_MAX
    yield from L.hold(0.6, grip=NECK_OPEN)
    yield from L.move(L.pos + 0.07 * up, rot_l, 1.0)
    yield from L.move(np.array([P["neck"][0], 0.22, L.pos[2]]), rot_l, 1.6)
    if variant == "no_push":
        yield from wait(2.0)
        return
    # 7. Right comes back with closed fingers and pushes the cork end until the base is on the back panel.
    yield from R.move(P["behind"] + 0.09 * up, rot_in, 2.0, grip=0.0, servo=XYZ)
    yield from R.move(P["behind"], rot_in, 1.2, servo=XYZ)
    yield from R.hold(0.4, servo=XYZ)
    R.corr_max = PUSH_CORR_MAX
    yield from R.move(P["home"], rot_in, 3.0, servo=X_ONLY)
    yield from R.hold(0.8, servo=X_ONLY)
    yield from R.hold(0.3)
    if variant == "no_release":
        yield from wait(2.0)
        return
    yield from R.relax(0.4)
    yield from R.move(R.pos - np.array([0.06, 0.0, 0.0]), rot_in, 1.0)
    yield from R.move(R.pos + 0.10 * up, rot_in, 1.2)
    yield from wait(1.5)


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"], read=lambda: self.joint_tcp("left"))
        self.right = Arm("right", station.BASES["right"], read=lambda: self.joint_tcp("right"))

    def joint_tcp(self, name: str):
        """The arm's tool point and gripper rotation from its own joint positions (forward kinematics)."""
        from partkit import _np
        from yam_arena import ik, station
        from yam_arena.constants import ARM_JOINTS

        art = self.emb._env.unwrapped.scene[f"{name}_robot"]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        pos, rot = ik.tcp_pose(_np(art.data.joint_pos)[0][ids], station.BASES[name])
        return np.asarray(pos, float), np.asarray(rot, float)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


class DryCtx:
    """No simulator: for checking the plan's IK offline on a machine that has yam_arena."""

    def __init__(self):
        from yam_arena import station

        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])


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
    quiet = os.environ.get("DEMO_DEBUG", "1") == "0"
    trace: list = []
    print("PLAN " + json.dumps({k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else round(float(v), 4))
                                for k, v in plan(task).items()}), flush=True)

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def row(steps, info):
        """Logging only: where the bottle and both tool points really are."""
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)
        b = st["bottle"]
        base, cork = task.ends(b)
        S = task._STATE
        lag = {}
        for name, arm in (("L", ctx.left), ("R", ctx.right)):
            try:
                m_pos, m_rot = ctx.joint_tcp(arm.name)
                a, c = m_rot[:, 0], arm.rot[:, 0]
                lag[name] = np.round(1000 * (m_pos - arm.pos), 1).tolist() + [round(math.degrees(
                    math.atan2(a[0] * c[1] - a[1] * c[0], a[0] * c[0] + a[1] * c[1])), 1)]
            except Exception as exc:  # noqa: BLE001
                lag[name] = repr(exc)
        return {"step": steps, "score": info.get("score"), "base_mm": np.round(1000 * (base - [0, 0, task.TOP]), 1).tolist(),
                "cork_mm": np.round(1000 * (cork - [0, 0, task.TOP]), 1).tolist(), "tilt": round(S["tilt"], 2),
                "fit": S["fit"], "released": S["released"], "still": S["still"], "why": list(S["why"]),
                "tcp_err_mm": lag, "corr_mm": {"L": np.round(1000 * ctx.left.corr, 1).tolist(),
                                               "R": np.round(1000 * ctx.right.corr, 1).tolist()},
                "grip": [round(float(ctx.left.grip), 2), round(float(ctx.right.grip), 2)],
                "stock_shift_mm": {n: round(1000 * float(np.linalg.norm(st[n].pos - S["stock0"][n])), 1)
                                   for n in task.STOCK_NAMES} if S["stock0"] else None}

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
        if not quiet and steps % 10 == 0:
            try:
                r = row(steps, info)
                trace.append(r)
                print("TRACE " + json.dumps(r, default=str), flush=True)
            except Exception as exc:  # noqa: BLE001 - logging only
                print(f"trace unavailable: {exc!r}", flush=True)
        elif steps % 100 == 0:
            print(f"step {steps} score {info.get('score')}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    S = task._STATE
    result = {"task_score": score, "success": success, "steps": steps, "variant": os.environ.get("DEMO_VARIANT", ""),
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "max_trim_mm": {"left": round(1000 * ctx.left.max_corr, 1), "right": round(1000 * ctx.right.max_corr, 1)},
              "state": {k: S[k] for k in ("fit", "tilt", "released", "still", "failed", "dropped", "disturbed", "why",
                                          "finger_check")}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
