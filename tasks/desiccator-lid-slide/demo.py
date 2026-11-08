"""Scripted full-score demo for the desiccator task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Open loop: every pose comes from the fixed layout in the task file and from step counts. The left arm grips the +y
carrying lug and holds the desiccator (its command creeps 4 mm against the lid's pull). The right arm grips the knob and
slides the lid 8.5 cm toward -y at 2 cm/s, past the 4.5 cm where the seal lets go, waits, lifts it 6 cm and carries it to
the grey pad. Meanwhile the left arm lets go of the lug, takes the vial out of the open jar, stands it on the tray and
lets go. DEMO_DEBUG=1 prints (and writes debug_trace.txt) the fingers against their grip features, the lid offset, the
desiccator's displacement and each arm's lag.

Negative variants (DEMO_VARIANT=...):
  one_arm        the left arm never holds the lug: the lid drags the desiccator along (0)
  lift_straight  the right arm lifts the knob without sliding: the seal holds (0)
  lid_on_table   the lid is set down on the table beside the pad (0.5)
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



VARIANTS = ("", "one_arm", "lift_straight", "lid_on_table")
SLIDE = 0.085              # m: the seal lets go at 4.5 cm; soft arms lag, so command well past it
SLIDE_SPEED = 0.02         # m/s
LEFT_CREEP = 0.004         # m: the left command moves +y by this over the slide, against the lid's pull
LIFT = 0.06                # m the lid is lifted after the slide
G_KNOB, G_LUG, G_VIAL = 0.60, 0.55, 0.65      # finger opening (1 = 94 mm) when approaching each grip feature
KNOB_TCP_Z = 0.138         # TCP height above the table: pads on the knob's top 20 mm
LUG_TCP_Z = 0.055
VIAL_TCP_Z = 0.066


def waypoints(T):
    """Fixed-layout TCP targets (world), shared by the script and the offline reach check."""
    top = T.TOP
    bx, by = T.BASE_XY
    w = {}
    w["knob"] = np.array([bx, by, top + KNOB_TCP_Z])
    w["lug"] = np.array([bx, by + T.LUG_GRIP_Y, top + LUG_TCP_Z])
    w["vial"] = np.array([T.VIAL_XY[0], T.VIAL_XY[1], top + VIAL_TCP_Z])
    w["slid"] = w["knob"] + np.array([0.0, -SLIDE, 0.0])
    w["lifted"] = w["slid"] + np.array([0.0, 0.0, LIFT])
    lid_to_tcp = KNOB_TCP_Z - T.RIM_Z                       # TCP above the lid's flange underside (0.058)
    w["over_pad"] = np.array([T.PAD_XY[0], T.PAD_XY[1], top + lid_to_tcp + T.PAD_TOP + 0.09 + 0.005])
    w["pad_set"] = np.array([T.PAD_XY[0], T.PAD_XY[1], top + lid_to_tcp + T.PAD_TOP + 0.005])
    w["over_table"] = np.array([0.42, T.PAD_XY[1], top + lid_to_tcp + T.PAD_TOP + 0.09 + 0.005])
    w["table_set"] = np.array([0.42, T.PAD_XY[1], top + lid_to_tcp + 0.005])
    w["over_vial"] = w["vial"] + np.array([0.0, 0.0, 0.055])
    w["vial_up"] = np.array([T.VIAL_XY[0], T.VIAL_XY[1], top + 0.16])
    w["over_tray"] = np.array([T.TRAY_XY[0], T.TRAY_XY[1], top + 0.16])
    w["tray_set"] = np.array([T.TRAY_XY[0], T.TRAY_XY[1], top + VIAL_TCP_Z - T.PLATE_Z - 0.0005 + T.TRAY_TOP + 0.005])
    return w


def script(task, ctx):
    """The demo as a generator: one yield per control step. Uses only the task's fixed layout and step counts."""
    from yam_arena.demo import grip_rotation

    T = task
    L, R = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    w = waypoints(T)
    up = np.array([0, 0, 0.07])
    up_r = np.array([0, 0, 0.03])            # the knob is high: the right arm barely reaches above it
    yaw_r = reachable_yaw(R, w["knob"], 0.0)                 # fingers close along y: the pull is borne by one pad
    yaw_l = reachable_yaw(L, w["lug"], math.pi / 2)          # fingers close along x across the 30 mm lug
    yaw_v = reachable_yaw(L, w["vial"], math.pi / 2)         # and across the vial, inside the 110 mm mouth
    rot_r, rot_l, rot_v = grip_rotation(yaw_r), grip_rotation(yaw_l), grip_rotation(yaw_v)

    # 1. Both arms above their grip features, down, close.
    if variant == "one_arm":
        yield from R.move(w["knob"] + up_r, rot_r, 2.5, grip=G_KNOB)
        yield from R.move(w["knob"], rot_r, 1.5)
        yield from R.set_grip(0.0, 1.0)
    else:
        yield from both(R.move(w["knob"] + up_r, rot_r, 2.5, grip=G_KNOB), L.move(w["lug"] + up, rot_l, 2.5, grip=G_LUG))
        yield from both(R.move(w["knob"], rot_r, 1.5), L.move(w["lug"], rot_l, 1.5))
        ctx.mark("close")
        yield from both(R.set_grip(0.0, 1.0), L.set_grip(0.0, 1.0))
    yield from wait(0.5)

    # 2. Slide the lid toward -y (or lift straight), the left arm holding the lug and creeping against the pull.
    if variant == "lift_straight":
        yield from R.move(w["knob"] + np.array([0, 0, 0.035]), rot_r, 2.0)
        yield from wait(1.0)
        return
    n = round(SLIDE / SLIDE_SPEED * CONTROL_HZ)
    for k in range(1, n + 1):
        s = k / n
        R.aim(w["knob"] + np.array([0.0, -SLIDE * s, 0.0]), rot_r)
        if variant != "one_arm":
            L.aim(w["lug"] + np.array([0.0, LEFT_CREEP * s, 0.0]), rot_l)
        yield
    yield from wait(1.0)                                     # the arm catches up once the seal lets go

    # 3. Lift the lid, carry it to the pad (or the table beside it), lower, let go.
    yield from R.move(w["lifted"], rot_r, 1.5)
    over, setd = ("over_table", "table_set") if variant == "lid_on_table" else ("over_pad", "pad_set")
    left_job = None if variant == "one_arm" else _vial_job(ctx, w, rot_l, rot_v, up)
    right_job = _carry(R, w, rot_r, over, setd, up)
    yield from right_job                                     # the lid is on the pad and the right arm clear first
    if left_job is not None:
        yield from left_job
    yield from wait(0.5)


def _carry(R, w, rot_r, over, setd, up):
    yield from R.move(w[over], rot_r, 3.0)
    yield from R.move(w[setd], rot_r, 1.5)
    yield from wait(0.3)
    yield from R.set_grip(1.0, 0.8)
    yield from R.move(R.pos + up, R.rot, 1.0)


def _vial_job(ctx, w, rot_l, rot_v, up):
    L = ctx.left
    yield from L.set_grip(G_LUG, 0.8)                        # let go of the lug (the lid is already 6 cm up)
    yield from L.move(L.pos + np.array([0, 0, 0.06]), rot_l, 1.0)
    yield from L.move(w["over_vial"] + np.array([0, 0, 0.0]), rot_v, 2.0, grip=G_VIAL)
    yield from wait(0.5)
    yield from L.move(w["vial"], rot_v, 1.5)
    ctx.mark("close_vial")
    yield from L.set_grip(0.0, 1.0)
    yield from wait(0.3)
    yield from L.move(w["vial_up"], rot_v, 1.5)
    yield from L.move(w["over_tray"], rot_v, 2.0)
    yield from L.move(w["tray_set"], rot_v, 1.5)
    yield from wait(0.3)
    yield from L.set_grip(1.0, 0.8)
    yield from L.move(L.pos + up, L.rot, 1.0)


def _yaw_of(p):
    r = p.rotation()
    return math.atan2(r[1, 0], r[0, 0])


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.marks = []
        self.log_file = None

    def mark(self, tag):
        """Ask the debug trace to describe the fingers against their grip feature when the grip is closed."""
        self.marks.append((tag, 22))

    def actual_tcp(self, name):
        """The arm's TCP from its measured joints (logging only)."""
        from yam_arena import ik
        from yam_arena.constants import ARM_JOINTS

        robot = self.emb._env.unwrapped.scene[f"{name}_robot"]
        q = robot.data.joint_pos[0].detach().cpu().numpy().astype(float)
        names = list(robot.joint_names)
        qa = np.array([q[names.index(j)] for j in ARM_JOINTS])
        arm = self.left if name == "left" else self.right
        return ik.tcp_pose(qa, arm.base)[0], q[[names.index("joint7"), names.index("joint8")]]

    def trace(self, task, steps, info):
        st = self.states()
        base, lid, vial = st["base"], st["lid"], st["vial"]
        dbg = task.debug_fields()
        lag = {}
        for nm, arm in (("L", self.left), ("R", self.right)):
            try:
                p, g = self.actual_tcp("left" if nm == "L" else "right")
                lag[nm] = (np.round(1000 * (arm.pos - p), 1).tolist(), np.round(np.abs(g), 4).tolist())
            except Exception as exc:  # noqa: BLE001
                lag[nm] = repr(exc)
        b0 = dbg["base0"]
        disp = None if b0 is None else np.round(1000 * (base.pos - b0), 1).tolist()
        msg = (f"step {steps} score {info.get('score')} sealed {dbg['sealed']} force_api {dbg['force_api']}"
               f" lid_off_mm {1000 * task.lid_offset(lid, base):.1f} lid_lift_mm {1000 * task.lid_lift(lid, base):.1f}"
               f" lid_xyz_mm {np.round(1000 * lid.pos, 1).tolist()} base_disp_mm {disp}"
               f" base_lift_mm {1000 * dbg['peak_base_lift']:.1f} vial_xyz_mm {np.round(1000 * vial.pos, 1).tolist()}"
               f" vial_tilt {task.tilt(vial):.1f} lid_vz {lid.lin_vel[2]:.2f} failed {dbg['failed']}"
               f" released {dbg['released']} gap_mm {1000 * dbg['gap_min']:.0f}"
               f" lag_mm(cmd-actual,grip) {lag}")
        while self.marks:
            tag, _ = self.marks.pop()
            msg += f" | MARK {tag}"
        return msg

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
        debug = bool(os.environ.get("DEMO_DEBUG"))
        if steps % 100 == 0 or (debug and steps % 5 == 0):
            msg = f"step {steps} score {info.get('score')}"
            if debug:
                msg = ctx.trace(task, steps, info)
                with open(out / "debug_trace.txt", "a") as fh:
                    fh.write(msg + "\n")
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
