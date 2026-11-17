"""Scripted full-score demo for centrifuge-bucket-hang, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]
    python demo.py task.py --ik          # no simulator: IK error of the whole plan, step by step

Open loop: every target comes from the task file's fixed layout and the clock. The only feedback is each arm's own
joint readings (a trim that keeps the measured tool point on the plan under a bucket's weight). No object pose, rotor
angle or task state steers it.

The loose buckets have to go over the 200 mm chamber rim and down between two pegs. The arms can only do that with the
hand pitched 55 degrees and coming in over the rotor's hub, which reaches the place on the far side of the rotor (the
right arm the left place, the left arm the right place). At the start the empty places are at the front and back, so:

1. The left hand puts its closed fingertips into the empty front place and walks them a quarter turn round the rotor
   axis, pointing at the axis all the way: the free rotor turns with them and the two hung buckets ride round to the
   front and back. It then spreads its fingers between the place's two pegs, which centres the place on the hand,
   closes them and lifts out.
   Meanwhile the right hand takes its bucket off the mat by the neck and lifts it clear of the rim.
2. The right hand carries its bucket over the hub to the left place and lowers it between the pegs until both pockets
   are down on them, lets go and backs out.
3. The left hand does the same with its bucket into the right place.

Negative variants (DEMO_VARIANT=...):
  one_only   only the first bucket is hung (0.25)
  no_turn    the rotor is not turned: one bucket is hung in the back place, the front place cannot be reached (0.25)
  one_peg    the first bucket is lowered 17 degrees out of line and 8 mm off, so one pocket takes its peg and the other
             side rides on its peg by the band (0)
  unaligned  the first bucket starts a quarter turn round on its mat and is lowered that way: it sits on both pegs by its
             band (0)
  drop       the first bucket is let go over the chamber beside the rotor (0)
Writes one MP4 per camera, setup.png and result.json; DEMO_DEBUG=0 silences the trace printed every 10 steps.
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
VARIANTS = ("", "one_only", "no_turn", "one_peg", "unaligned", "drop")

# -- the plan's numbers ---------------------------------------------------------------------------------------------------
PITCH = math.radians(55.0)       # the hand's pitch below level while it holds a bucket
TCP_UP = 0.025                   # tool point above the bucket's pocket seats: the pads cover the whole 28 mm neck
#                                  (5 mm below the rim they held it by a corner and it rocked 3.5 degrees: sim run 2)
CARRY = 0.315                    # tool point above the bench when a bucket crosses the rim (bucket bottom 16 mm clear)
OVER_HUB = 0.300
ABOVE_PLACE = 0.290
PRESS = 0.004                    # the bucket is lowered 4 mm past seated (the pegs stop it)
PICK_HEADING = math.radians(30.0)   # the right hand's heading at its mat; the left hand's is the mirror image
OPEN_AFTER = 0.70                # fingers opened to 66 mm to let a hung bucket go (clear of the buckets either side)
TURN_PITCH = math.radians(65.0)  # the left hand's pitch while it turns the rotor
TURN_SKEW = math.radians(15.0)   # the hand points at the rotor axis, turned this much from it at the start (reach)
TURN_R = 0.074                   # the tool point's distance from the rotor axis while turning
TURN_UP = 0.004                  # tool point above the peg axis while turning
TURN_END = math.radians(77.0)    # the fingertips go from 180 deg (front) to here: the place they push ends at 90 deg
#                                  (the pushed peg leads the tool point by 12.8 degrees: 24 mm peg tip, 6 mm finger)
WEDGE = 0.47                     # finger opening (0..1) that spreads the fingertips against both pegs of the place
KI, CORR_MAX = 0.2, 0.02         # joint-feedback trim while carrying a bucket: gain per step and limit (m)
GRIP_HOLD = 0.426                # finger opening (0..1) commanded on the 52 mm neck: pads set to 40 mm, so each presses
#                                  with about 30 N and the neck stays centred between them (fully closed, both fingers
#                                  saturate and the bucket can sit up to 21 mm to one side: measured in a sim run)
NEAR = 0.010                     # the trim is frozen for the last 10 mm onto the pegs


def grip(heading: float, pitch: float, flip: float = 1.0) -> np.ndarray:
    """Tool rotation (columns: closing axis, y, approach) for a hand heading ``heading`` in the bench plane and pointing
    ``pitch`` below level; ``flip`` -1 is the same parallel grip turned half a turn about its approach axis."""
    h = np.array([math.cos(heading), math.sin(heading), 0.0])
    z = math.cos(pitch) * h + np.array([0.0, 0.0, -math.sin(pitch)])
    x = flip * np.array([-h[1], h[0], 0.0])
    return np.c_[x, np.cross(z, x), z]


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each from where the last command left it. With ``servo`` the
    command is trimmed by the arm's own joint readings, so the measured tool point follows the plan under the
    bucket's weight."""

    def __init__(self, name: str, base_pos, read=None):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.pos = np.asarray(self.pos, float)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.read = read
        self.corr = np.zeros(3)
        self.max_corr = 0.0

    def _solve(self, pos, rot, servo: bool = False):
        from yam_arena import ik

        if servo and self.read is not None:
            meas = self.read()
            if meas is not None:
                self.corr = np.clip(self.corr + KI * (self.pos - meas[0]), -CORR_MAX, CORR_MAX)
                self.max_corr = max(self.max_corr, float(np.abs(self.corr).max()))
        q, pe, _ = ik.solve(np.asarray(pos, float) + self.corr, ik.matrix_to_quat_xyzw(rot), self.q,
                            base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None, servo: bool = False):
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s), servo)
            yield

    def hold(self, duration: float, grip: float | None = None, servo: bool = False):
        yield from self.move(self.pos, self.rot, duration, grip, servo)

    def relax(self, duration: float = 0.5):
        """Let the trim go, smoothly."""
        c0 = self.corr.copy()
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self.corr = c0 * (1 - k / n)
            self._solve(self.pos, self.rot)
            yield

    def arc(self, centre, radius: float, a0: float, a1: float, z: float, rot_at, duration: float):
        """The tool point along a circle about the vertical through ``centre``, from angle a0 to a1; ``rot_at(a)`` is
        the hand's rotation at angle a."""
        from yam_arena import ik

        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            a = a0 + ik.smoothstep(k / n) * (a1 - a0)
            self._solve(np.array([centre[0] + radius * math.cos(a), centre[1] + radius * math.sin(a), z]), rot_at(a))
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


def best_flip(arm, poses) -> float:
    """+1 or -1: which half-turn of the grip has the smaller worst IK error over ``poses`` [(pos, heading, pitch)],
    solved one after another from the arm's ready pose."""
    from yam_arena import ik
    from yam_arena.constants import READY

    out = []
    for flip in (1.0, -1.0):
        q, worst = np.array(READY, float), 0.0
        for pos, heading, pitch in poses:
            q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(grip(heading, pitch, flip)), q,
                                 base_pos=arm.base, iters=300)
            worst = max(worst, pe + 0.05 * re)
        out.append((worst, flip))
    return min(out)[1]


def plan(T, side: str, variant: str = "") -> dict:
    """Tool-point targets (world, m) and hand headings for the arm on ``side`` hanging its bucket in the far place,
    from the task file's fixed layout. ``side`` 'right': bucket on the right mat, hung in the left place."""
    sy = -1.0 if side == "right" else 1.0                     # the mat's side; the place is on the other side
    up = np.array([0.0, 0.0, 1.0])
    mat = np.array([T.MAT_X, sy * T.MAT_Y, T.BUCKET_REST_Z + TCP_UP])
    place = np.array([T.XC, -sy * T.PEG_R * T.MM, T.PEG_Z + TCP_UP])
    h_pick, h_place = -sy * PICK_HEADING, -sy * math.pi / 2
    p = {"side": side, "sy": sy, "h_pick": h_pick, "h_mid": (h_pick + h_place) / 2, "h_place": h_place,
         "pick": mat,
         "lift": np.array([mat[0], mat[1], T.TOP + CARRY]),
         "rim": np.array([T.XC, sy * 0.10, T.TOP + CARRY]),
         "hub": np.array([T.XC, 0.0, T.TOP + OVER_HUB]),
         "above": np.array([place[0], place[1], T.TOP + ABOVE_PLACE]),
         "near": place + NEAR * up, "seat": place - PRESS * up}
    if variant == "one_peg":                                  # 17 degrees out of line and 8 mm further from the axis
        p["h_place"] = h_place - sy * math.radians(17.0)
        for k in ("above", "near", "seat"):
            p[k] = p[k] + np.array([0.0, -sy * 0.008, 0.0])
    if variant == "no_turn":                                  # the back place, reached over the hub from the front
        back = np.array([T.XC + T.PEG_R * T.MM, 0.0, T.PEG_Z + TCP_UP])
        p["h_place"], p["h_mid"] = 0.0, h_pick / 2
        p["rim"] = np.array([T.XC + 0.04, sy * 0.13, T.TOP + CARRY])
        p["hub"] = np.array([T.XC + 0.06, sy * 0.04, T.TOP + OVER_HUB])
        p["above"] = np.array([back[0], back[1], T.TOP + ABOVE_PLACE])
        p["near"], p["seat"] = back + NEAR * up, back - PRESS * up
    return p


def hang_poses(p) -> list:
    return [(p["lift"], p["h_pick"], PITCH), (p["pick"], p["h_pick"], PITCH), (p["lift"], p["h_pick"], PITCH),
            (p["rim"], p["h_mid"], PITCH), (p["hub"], p["h_place"], PITCH), (p["above"], p["h_place"], PITCH),
            (p["seat"], p["h_place"], PITCH)]


def take(arm, p, flip):
    """Take the bucket off its mat by the neck and lift it clear of the rim."""
    r = grip(p["h_pick"], PITCH, flip)
    ease = p["pick"] + np.array([0.0, 0.0, 0.010])              # the first 10 mm slowly, so the bucket does not rock
    yield from chain(arm.move(p["lift"], r, 2.5, 1.0), arm.move(p["pick"], r, 1.8), arm.hold(0.7, GRIP_HOLD),
                     arm.hold(0.3), arm.move(ease, r, 1.0, servo=True), arm.hold(0.3, servo=True),
                     arm.move(p["lift"], r, 1.8, servo=True))


def hang(arm, p, flip, variant=""):
    """Carry the held bucket over the rim and the hub, lower it onto the far place's pegs, let go and back out."""
    r_mid, r_place = grip(p["h_mid"], PITCH, flip), grip(p["h_place"], PITCH, flip)
    yield from arm.move(p["rim"], r_mid, 2.0, servo=True)
    if variant == "drop":
        yield from chain(arm.hold(0.5, 1.0), wait(1.0), arm.relax(0.3),
                         arm.move(p["lift"], grip(p["h_pick"], PITCH, flip), 2.0))
        return
    yield from chain(arm.move(p["hub"], r_place, 1.5, servo=True), arm.move(p["above"], r_place, 1.2, servo=True),
                     arm.move(p["near"], r_place, 2.2, servo=True), arm.hold(0.3, servo=True),
                     arm.move(p["seat"], r_place, 0.8), arm.hold(0.3), arm.hold(0.5, OPEN_AFTER), arm.relax(0.3),
                     arm.move(p["above"], r_place, 1.5), arm.move(p["hub"], r_place, 1.0),
                     arm.move(p["rim"], r_mid, 1.5), arm.move(p["lift"], grip(p["h_pick"], PITCH, flip), 1.5))


def turn_heading(a: float) -> float:
    """The turning hand's heading when its fingertips are at angle ``a`` round the rotor axis: at the axis, turned
    TURN_SKEW from it at the front (a = pi) and squarely at it from the left place (a = pi/2) on."""
    skew = -TURN_SKEW * (a - math.pi / 2) / (math.pi / 2) if a >= math.pi / 2 else 0.0
    return a + math.pi + skew


def turn_plan(T) -> dict:
    """The left hand's targets for turning the rotor a quarter turn (clockwise seen from above)."""
    z = T.PEG_Z + TURN_UP
    start = np.array([T.XC - TURN_R, 0.0, z])
    end = np.array([T.XC, TURN_R, z])
    return {"centre": (T.XC, 0.0), "z": z, "start": start, "end": end, "start_over": start + [0.0, 0.0, 0.12],
            "end_over": end + [0.0, 0.0, 0.13]}


def turn_poses(t) -> list:
    out = [(t["start_over"], turn_heading(math.pi), TURN_PITCH)]
    for a in np.linspace(math.pi, TURN_END, 12):
        out.append((np.array([t["centre"][0] + TURN_R * math.cos(a), t["centre"][1] + TURN_R * math.sin(a), t["z"]]),
                    turn_heading(a), TURN_PITCH))
    return out + [(t["end"], turn_heading(math.pi / 2), TURN_PITCH), (t["end_over"], turn_heading(math.pi / 2), TURN_PITCH)]


def turn_rotor(arm, t, flip):
    """Closed fingertips into the front place, walked round the rotor axis with the hand kept pointing at the axis
    (so the fingers stay across the place and push its leading peg squarely), then spread between the two pegs."""
    def rot_at(a):
        return grip(turn_heading(a), TURN_PITCH, flip)

    yield from chain(arm.move(t["start_over"], rot_at(math.pi), 2.5, 0.0), arm.move(t["start"], rot_at(math.pi), 1.2),
                     arm.arc(t["centre"], TURN_R, math.pi, TURN_END, t["z"], rot_at, 6.0),      # push the place round
                     arm.arc(t["centre"], TURN_R, TURN_END, math.pi / 2, t["z"], rot_at, 0.8),  # back to its middle
                     arm.hold(0.6, WEDGE), arm.hold(0.3), arm.hold(0.5, 0.0),                   # centre it on the hand
                     arm.move(t["end_over"], rot_at(math.pi / 2), 1.2))


def script(task, ctx):
    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    L, R = ctx.left, ctx.right
    pr, pl, t = plan(T, "right", variant), plan(T, "left"), turn_plan(T)
    fr, fl, ft = best_flip(R, hang_poses(pr)), best_flip(L, hang_poses(pl)), best_flip(L, turn_poses(t))
    if variant == "no_turn":
        yield from chain(take(R, pr, fr), hang(R, pr, fr))
        yield from wait(2.0)
        return
    # 1. the left hand turns the rotor a quarter turn while the right hand takes its bucket
    yield from both(turn_rotor(L, t, ft), take(R, pr, fr))
    # 2. the right hand hangs its bucket in the left place; the left hand takes its own bucket
    if variant in ("one_only", "one_peg", "unaligned", "drop"):
        yield from both(hang(R, pr, fr, variant), L.move(pl["lift"], grip(pl["h_pick"], PITCH, fl), 2.5, 1.0))
        yield from wait(2.0)
        return
    yield from both(hang(R, pr, fr), take(L, pl, fl))
    # 3. the left hand hangs its bucket in the right place
    yield from hang(L, pl, fl)
    yield from wait(1.0)


class Ctx:
    def __init__(self, emb=None):
        from yam_arena import station

        self.emb = emb
        read = (lambda name: (lambda: self.joint_tcp(name))) if emb is not None else (lambda name: None)
        self.left = Arm("left", station.BASES["left"], read=read("left"))
        self.right = Arm("right", station.BASES["right"], read=read("right"))

    def joint_tcp(self, name: str):
        """The arm's tool point and rotation from its own joint positions (forward kinematics)."""
        from partkit import _np
        from yam_arena import ik, station
        from yam_arena.constants import ARM_JOINTS

        art = self.emb._env.unwrapped.scene[f"{name}_robot"]
        ids = art.find_joints(ARM_JOINTS, preserve_order=True)[0]
        pos, rot = ik.tcp_pose(_np(art.data.joint_pos)[0][ids], station.BASES[name])
        return np.asarray(pos, float), np.asarray(rot, float)

    def fingers(self) -> dict:
        """Logging only: each arm's two finger joint positions (mm from closed)."""
        from partkit import _np

        out = {}
        for name in ("left", "right"):
            art = self.emb._env.unwrapped.scene[f"{name}_robot"]
            jn = list(art.joint_names)
            q = _np(art.data.joint_pos)[0]
            out[name[0].upper()] = [round(-1000 * float(q[jn.index(j)]), 1) for j in ("joint7", "joint8") if j in jn]
        return out

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
    if os.environ.get("DEMO_VARIANT") == "unaligned":
        os.environ["CENTRIFUGE_BUCKET_QUARTER_TURN"] = "1"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    gen = script(task, ctx)
    frames: dict[str, list] = {}
    debug = os.environ.get("DEMO_DEBUG", "1") != "0"
    min_gap = float("inf")
    import clearance

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    last_gap = {"mm": None}

    def row(steps, info):
        """Logging only: reads simulator state for the trace."""
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)
        status = task.status(st)
        S = task._STATE
        return {"step": steps, "score": info.get("score"), "rotor_deg": round(math.degrees(task.rotor_angle(st)), 2),
                "gap_mm": last_gap["mm"], "released": S["released"], "clear": S["clear"], "report": S["report"],
                "buckets": {n: {"place": s["place"], "pegs": s["pegs"], "tilt_deg": round(math.degrees(s["tilt"]), 1),
                                "pos": np.round(st[n].pos - [task.XC, 0, task.PEG_Z], 4).tolist()}
                            for n, s in status.items()},
                "trim_mm": {"L": np.round(1000 * ctx.left.corr, 1).tolist(),
                            "R": np.round(1000 * ctx.right.corr, 1).tolist()},
                "tcp": {"L": np.round(ctx.left.pos - [task.XC, 0, task.PEG_Z], 3).tolist(),
                        "R": np.round(ctx.right.pos - [task.XC, 0, task.PEG_Z], 3).tolist(),
                        "L_meas": np.round(ctx.joint_tcp("left")[0] - [task.XC, 0, task.PEG_Z], 3).tolist(),
                        "R_meas": np.round(ctx.joint_tcp("right")[0] - [task.XC, 0, task.PEG_Z], 3).tolist(),
                        "grip": [round(float(ctx.left.grip), 2), round(float(ctx.right.grip), 2)],
                        "fingers_mm": ctx.fingers()}}

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
        if steps % 5 == 0:
            try:
                gap, _ = clearance.min_clearance(ctx.left.q, ctx.left.grip, ctx.left.base, ctx.right.q,
                                                 ctx.right.grip, ctx.right.base)
                min_gap = min(min_gap, gap)
                last_gap["mm"] = round(1000 * gap, 1)
            except Exception as exc:  # noqa: BLE001 - logging only
                if steps == 5:
                    print(f"clearance unavailable: {exc!r}", flush=True)
        if debug and steps % 10 == 0:
            try:
                print("TRACE " + json.dumps(row(steps, info), default=str), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"trace unavailable: {exc!r}", flush=True)
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
              "min_arm_link_gap_mm": round(1000 * min_gap, 1) if min_gap < float("inf") else None,
              "report": S["report"], "dropped": S["dropped"], "moved": S["moved"], "finger_check": S["finger_check"]}
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


def dry_run(task_path: str, variant: str = ""):
    """No simulator: play the script and return every step's joint commands [(q_left, grip_left, q_right, grip_right)]
    and each arm's worst IK error (mm)."""
    import importlib

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)
    os.environ["DEMO_VARIANT"] = variant
    ctx = Ctx()
    steps = []
    for _ in script(task, ctx):
        steps.append((ctx.left.q.copy(), float(ctx.left.grip), ctx.right.q.copy(), float(ctx.right.grip),
                      ctx.left.pos.copy(), ctx.right.pos.copy()))
    return task, steps, {"left": ctx.left.worst_mm, "right": ctx.right.worst_mm}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ik", action="store_true", help="only play the plan through IK and print the worst errors")
    a = ap.parse_args()
    if a.ik:
        for v in VARIANTS:
            _, s, worst = dry_run(a.task, v)
            print(f"IK variant '{v}': {len(s)} steps, worst error left {worst['left']:.1f} mm, right {worst['right']:.1f} mm",
                  flush=True)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
