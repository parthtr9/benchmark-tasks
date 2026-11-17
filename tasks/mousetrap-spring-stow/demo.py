"""Scripted full-score demo for the set-mouse-trap task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Open loop: every target comes from the fixed layout in the task file and from timing. No object pose, joint state or
task state is read to steer (the log reads them).

The right arm does the whole job with the screwdriver; no gripper comes near the trap.
1. It picks the screwdriver up by its handle and brings the shaft, level, over the trap's pedal end.
2. It lowers the shaft onto the bait pedal. The catch lets go and the bar snaps over onto the shaft: the trap now
   bites the screwdriver.
3. It lifts the screwdriver with the trap hanging on it, carries it over the bin and lowers the trap inside.
4. It draws the screwdriver back over the bin's right wall: the wall holds the trap back, the shaft slides out from
   under the bar and the trap drops to the bin's floor.
5. It lays the screwdriver back on the bench and lets go.

Negative variants (DEMO_VARIANT=...):
  by_hand     the closed right gripper presses the pedal itself: its fingers are in the bar's path (0)
  carry_set   the left arm picks the set trap up by the sides of its board and puts it in the bin, still set (0)
  no_stow     the trap is sprung with the screwdriver and left on the bench (0.5)
  no_scrape   the sprung trap is held inside the bin, still clamped on the held screwdriver (0.75)
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
VARIANTS = ("", "by_hand", "carry_set", "no_stow", "no_scrape")
PAD_TIP = 0.013           # the pad tips are this far below the tool point
TIP_ON_PEDAL = 0.030      # the shaft's tip is brought this far from the trap's axle, over the bait pedal
OVER_Z = 0.033            # the shaft's axis this high above the table on the way in (pedal top: 17 mm)
PRESS_Z = 0.015           # ... and lowered to here: the pedal trips when the axis passes about 17 mm
SLOW = 2.0                # the lift, carry and lowering run this many times slower than the other moves
TIP_DX = 0.007            # the held screwdriver sits about 2 degrees askew: this puts its tip on the trap's centre line
CARRY_Z = 0.128           # the shaft's axis while the trap is carried (the hanging trap clears the bin's 76 mm rim)
BIN_TIP_IN = 0.0405       # in the bin, the shaft's tip this far inside the right wall's inner face
SCRAPE_UP = 0.009         # the shaft's axis this far above the rim while it is drawn back (it sags about 3 mm)
SCRAPE_BACK = 0.075       # drawn back this far: the wall strips the trap off the shaft
HAND_Z = 0.024            # by_hand: tool-point height when the closed pad tips have pressed the pedal down
BASE_GRIP_Z = 0.014       # carry_set: tool-point height gripping the board's sides (pad tips 1 mm above the table)
BASE_GRIP_BACK = 0.025    # carry_set: the grip's middle this far behind the axle, on the set bar's half
BASE_OPEN = 0.62          # grip value: pads about 58 mm apart around the 44.5 mm board


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

        q, pe, _ = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None):
        """Generator: go to ``pos``/``rot`` (world tool point); ``grip`` 1 open .. 0 closed (None keeps it)."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def hold(self, duration: float, grip: float | None = None):
        yield from self.move(self.pos, self.rot, duration, grip)


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


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
        worst = max(worst, pe + 0.05 * re)
    return worst


def plan(T):
    """Every tool-point target, from the task file's fixed layout (world frame, m)."""
    top = T.TOP
    grip_to_tip = T.TIP_X                                      # handle's middle to the shaft's tip
    p = {}
    p["driver"] = np.array([T.DRIVER_XY[0], T.DRIVER_XY[1], top + T.HANDLE_AF / 2])
    tip_y = T.TRAP_XY[1] - TIP_ON_PEDAL                        # the trap's pedal end points at -y
    p["over"] = np.array([T.TRAP_XY[0], tip_y - grip_to_tip, top + OVER_Z])
    p["press"] = np.array([T.TRAP_XY[0], tip_y - grip_to_tip, top + PRESS_Z])
    p["lift"] = np.array([T.TRAP_XY[0], tip_y - grip_to_tip, top + CARRY_Z])
    wall_in = T.BIN_XY[1] - T.BIN_L / 2 + T.BIN_T              # the bin's right wall, inner face
    bin_tip_y = wall_in + BIN_TIP_IN
    p["above_bin"] = np.array([T.BIN_XY[0], bin_tip_y - grip_to_tip, top + CARRY_Z])
    p["in_bin"] = np.array([T.BIN_XY[0], bin_tip_y - grip_to_tip, top + T.BIN_H + SCRAPE_UP])
    p["scraped"] = p["in_bin"] - [0.0, SCRAPE_BACK, 0.0]
    p["hand"] = np.array([T.TRAP_XY[0], T.TRAP_XY[1] - 0.026, top + HAND_Z])
    p["base"] = np.array([T.TRAP_XY[0], T.TRAP_XY[1] + BASE_GRIP_BACK, top + BASE_GRIP_Z])
    p["base_bin"] = np.array([T.BIN_XY[0], T.BIN_XY[1], top + T.BIN_T + BASE_GRIP_Z + 0.004])
    return p


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    press_z = float(os.environ.get("DEMO_PRESS_Z", PRESS_Z))
    slow = SLOW
    R, L = ctx.right, ctx.left
    P = plan(T)
    P["press"][2] = T.TOP + press_z
    for k in ("over", "press", "lift", "above_bin", "in_bin", "scraped"):
        P[k][0] += TIP_DX
    carry_z = CARRY_Z
    up = np.array([0.0, 0.0, 1.0])

    def best_rot(arm, poses):
        best = None
        for y in closing_yaws("x"):                       # pads close along x: across the handle, across the board
            r0 = grip_rotation(y)
            err = ik_error(arm, [(q, r0) for q in poses])
            if best is None or err < best[0]:
                best = (err, r0)
        return best

    if variant == "by_hand":
        err, rot = best_rot(R, [P["hand"] + 0.08 * up, P["hand"]])
        print("IK " + json.dumps({"hand_plan_err_mm": round(1000 * err, 2)}), flush=True)
        yield from R.move(P["hand"] + 0.08 * up, rot, 2.5, grip=0.0)
        yield from R.move(P["hand"], rot, 2.0)
        yield from wait(1.5)
        yield from R.move(P["hand"] + 0.08 * up, rot, 1.5)
        yield from wait(1.5)
        return
    if variant == "carry_set":
        err, rot = best_rot(L, [P["base"] + 0.08 * up, P["base"], P["base"] + 0.11 * up, P["base_bin"] + 0.11 * up,
                                P["base_bin"]])
        print("IK " + json.dumps({"base_plan_err_mm": round(1000 * err, 2)}), flush=True)
        yield from L.move(P["base"] + 0.08 * up, rot, 2.5, grip=BASE_OPEN)
        yield from L.move(P["base"], rot, 1.5)
        yield from L.hold(0.8, grip=0.0)
        yield from wait(0.2)
        yield from L.move(P["base"] + 0.11 * up, rot, 2.0)
        yield from L.move(P["base_bin"] + 0.11 * up, rot, 3.0)
        yield from L.move(P["base_bin"], rot, 2.5)
        yield from L.hold(0.6, grip=BASE_OPEN)
        yield from L.move(P["base_bin"] + 0.12 * up, rot, 1.5)
        yield from wait(1.5)
        return

    err, rot = best_rot(R, [P["driver"] + 0.08 * up, P["driver"], P["over"], P["press"], P["lift"], P["above_bin"],
                            P["in_bin"], P["scraped"]])
    print("IK " + json.dumps({"plan_err_mm": round(1000 * err, 2)}), flush=True)
    # 1. Pick the screwdriver up by its handle.
    yield from R.move(P["driver"] + 0.08 * up, rot, 2.5, grip=1.0)
    yield from R.move(P["driver"], rot, 1.5)
    yield from R.hold(0.8, grip=0.0)
    yield from wait(0.2)
    yield from R.move(P["driver"] + 0.05 * up, rot, 1.0)
    # 2. The shaft over the pedal end, then down onto the pedal: the bar snaps over onto the shaft.
    yield from R.move(P["over"] + 0.02 * up, rot, 2.5)
    yield from R.move(P["over"], rot, 0.8)
    yield from R.move(P["press"], rot, 1.2)
    yield from wait(1.0)
    if variant == "no_stow":
        yield from wait(3.0)
        return
    # 3. Lift the screwdriver with the trap biting it, carry it over the bin, lower the trap inside.
    yield from R.move(P["lift"], rot, 2.5 * slow)
    yield from R.move(P["above_bin"], rot, 3.0 * slow)
    yield from R.move(P["in_bin"], rot, 2.0 * slow)
    yield from wait(0.3)
    if variant == "no_scrape":
        yield from wait(3.0)
        return
    # 4. Draw the screwdriver back over the wall: the trap is stripped off and drops to the bin's floor.
    yield from R.move(P["scraped"], rot, 2.5)
    yield from wait(0.5)
    yield from R.move(P["scraped"] + (carry_z - T.BIN_H - SCRAPE_UP) * up, rot, 1.0)
    # 5. The screwdriver back on the bench.
    yield from R.move(P["driver"] + 0.06 * up, rot, 3.0)
    yield from R.move(P["driver"] + 0.002 * up, rot, 1.5)
    yield from R.hold(0.6, grip=1.0)
    yield from R.move(P["driver"] + 0.09 * up, rot, 1.2)
    yield from wait(1.5)


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def joint_tcp(self, name: str):
        """Logging only: the arm's tool point from its own joint positions (forward kinematics)."""
        from partkit import _np
        from yam_arena import ik, station
        from yam_arena.constants import ARM_JOINTS

        art = self.emb._env.unwrapped.scene[f"{name}_robot"]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        return np.asarray(ik.tcp_pose(_np(art.data.joint_pos)[0][ids], station.BASES[name])[0], float)

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
    quiet = os.environ.get("DEMO_DEBUG", "1") == "0"
    print("PLAN " + json.dumps({k: np.round(v, 4).tolist() for k, v in plan(task).items()}), flush=True)

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def row(steps, info):
        """Logging only: where the trap, its bar and the screwdriver really are."""
        from yam_arena.scene import object_states

        st = object_states(emb._env, task.NAMES)
        trap, driver = st["mouse_trap"], st["screwdriver"]
        S = task._STATE
        try:
            lag = np.round(1000 * (ctx.joint_tcp("right") - ctx.right.pos), 1).tolist()
        except Exception as exc:  # noqa: BLE001
            lag = repr(exc)
        rot = trap.rotation()
        home = [task.TRAP_XY[0], task.TRAP_XY[1], task.TOP]
        return {"step": steps, "score": info.get("score"),
                "trap_mm": np.round(1000 * (np.asarray(trap.pos) - home), 1).tolist(),
                "trap_pitch_roll_deg": [round(math.degrees(math.asin(max(-1.0, min(1.0, -rot[2, 0])))), 1),
                                        round(math.degrees(math.atan2(rot[2, 1], rot[2, 2])), 1)],
                "bow_deg": round(math.degrees(S["bow"]), 1), "pedal_deg": round(math.degrees(S["pedal"]), 1),
                "lock_deg": round(math.degrees(float(trap.hinges.get("lock", 0.0))), 1),
                "tip_mm": np.round(1000 * (driver.to_world((task.TIP_X, 0.0, 0.0)) - [0, 0, task.TOP]), 1).tolist(),
                "handle_mm": np.round(1000 * (np.asarray(driver.pos) - [0, 0, task.TOP]), 1).tolist(),
                "fired": S["fired"], "sprung": S["sprung"], "in_bin": S["in_bin"], "resting": S["resting"],
                "released": S["released"], "still": S["still"], "why": list(S["why"]),
                "tcp_err_mm": lag, "grip": round(float(ctx.right.grip), 2)}

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
                print("TRACE " + json.dumps(row(steps, info), default=str), flush=True)
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
              "state": {k: S[k] for k in ("fired", "fired_call", "bow", "sprung", "in_bin", "resting", "released",
                                          "still", "failed", "set_in_bin", "struck", "fell", "why", "finger_check")}}
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
