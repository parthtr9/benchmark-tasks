"""Scripted full-score demo for unused-knockout-close, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]
    python demo.py --offline            # kinematic run, no simulator: IK error, arm clearance and pad-tip margins only

Open loop from the fixed layout and the clock; no object pose or joint state is read. The left arm takes the 1/2 in
dome seal out of the tray by its head, carries it over the open knockout, sets its fingers into the hole (their taper
centres it) and lets go; then it closes its fingers and presses the dome down until the seal's fingers snap past the
wall, and moves away. The right arm stays clear.

Negative variants (DEMO_VARIANT=...):
  wrong_seal  the 3/4 in seal is set on the open knockout and pressed: it cannot go in and skids aside (0)
  no_press    the 1/2 in seal is set in the hole, let go and left unpressed (0.5)
  held        the seal is pressed home but the fingers stay on it (0.75)
DEMO_DEBUG=1 prints a trace every 10 steps. Writes one MP4 per camera, setup.png and result.json to the output folder.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
VARIANTS = ("", "wrong_seal", "no_press", "held")

CARRY_H = 0.19                    # carry height of the tool point above the table (the down grip's reach ceiling)
PRESS_PAST = 0.035                 # command the closed pad tips this far below the snapped dome's top


def grip_rot(heading: float, tilt: float) -> np.ndarray:
    """Approach along a horizontal heading tipped ``tilt`` down; fingers horizontal, across the heading."""
    a = np.array([math.cos(tilt) * math.cos(heading), math.cos(tilt) * math.sin(heading), -math.sin(tilt)])
    f = np.array([-math.sin(heading), math.cos(heading), 0.0])
    return np.column_stack([f, np.cross(a, f), a])


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
        self.worst_deg = 0.0

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(re))
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None):
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def follow(self, poses, duration: float):
        """Track a list of (pos, rot) evenly over ``duration`` (interpolating between them)."""
        from yam_arena import ik

        n = max(1, round(duration * CONTROL_HZ))
        m = len(poses) - 1
        for k in range(1, n + 1):
            u = ik.smoothstep(k / n) * m
            i = min(int(u), m - 1)
            s = u - i
            (p0, r0), (p1, r1) = poses[i], poses[i + 1]
            self._solve(p0 + s * (p1 - p0), ik.slerp_matrix(r0, r1, s))
            yield

    def set_grip(self, grip: float, duration: float = 0.6):
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


class Flow:
    def __init__(self):
        self.done = False


def down(yaw: float = 0.0) -> np.ndarray:
    from yam_arena.demo import grip_rotation

    return grip_rotation(yaw)


def left_arm(L, T, flow, variant):
    name = "seal_3q" if variant == "wrong_seal" else "seal_half"
    sx, sy = T.SEALS[name]["pos"]
    s_z = {"seal_half": T.SEAL_HALF.pos[2], "seal_3q": T.SEAL_3Q.pos[2]}[name]
    grip = np.array([sx, sy, s_z + 0.0085])          # pads on the head's rim, their tips 4.5 mm below it
    carry = T.TOP + CARRY_H
    ko = T.OPEN_KO
    yield from L.move(np.array([sx, sy, carry]), down(0.0), 3.0, grip=0.5)     # fingers 47 mm apart for the 26 mm head
    yield from L.move(grip, down(0.0), 1.5)
    yield from wait(0.8)                               # let the arm settle on the plan before closing
    yield from L.set_grip(0.0, 0.8)
    yield from L.move(np.array([sx, sy, carry]), down(0.0), 1.5)
    yield from L.move(np.array([ko[0], ko[1], carry]), down(0.0), 3.0)
    yield from wait(0.8)
    # set its fingers into the hole: the barbs rest on the hole's edge, the taper centres it
    yield from L.move(np.array([ko[0], ko[1], ko[2] + 0.0040 + 0.0085]), down(0.0), 2.0)   # barbs level with the edge: the taper is in the hole
    yield from wait(0.6)
    yield from L.set_grip(0.5, 0.6)                    # half open: the fingers stay clear of the connector beside it
    yield from L.move(L.pos + np.array([0, 0, 0.012]), down(0.0), 0.8)
    if variant == "no_press":
        yield from L.move(np.array([ko[0], ko[1], carry]), down(0.0), 1.2)
        # move clear: 0.5 counts only once released. Back along x first: the pads (closing along y, 47 mm apart) pass either
        # side of the unsnapped seal, whose dome is only about 2 mm below the pad tips at carry height
        yield from L.move(np.array([ko[0] - 0.08, ko[1], carry]), down(0.0), 1.2)
        yield from L.move(np.array([0.20, 0.22, carry]), down(0.0), 1.2)
        flow.done = True
        return
    yield from L.set_grip(0.0, 0.6)
    from yam_arena.constants import PAD_TIP_PAST_TCP

    dome_top = ko[2] + T.HEAD_T + T.DOME_H                  # the dome's top once snapped flush
    yield from L.move(np.array([ko[0], ko[1], dome_top + PAD_TIP_PAST_TCP + 0.004]), down(0.0), 1.0)
    yield from L.move(np.array([ko[0], ko[1], dome_top + PAD_TIP_PAST_TCP - PRESS_PAST]), down(0.0), 4.5)
    yield from wait(0.5)
    if variant == "held":
        yield from wait(10.0)
        return
    yield from L.move(np.array([ko[0], ko[1], carry]), down(0.0), 1.2)
    yield from L.move(np.array([0.20, 0.22, carry]), down(0.0), 1.2)
    flow.done = True


def right_arm(R, T, flow, variant):
    yield from R.move(np.array([0.22, -0.22, T.TOP + 0.15]), down(0.0), 2.0, grip=1.0)


def script(task, ctx):
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}")
    yield from both(left_arm(ctx.left, task, ctx.flow, variant), right_arm(ctx.right, task, ctx.flow, variant))


# -- clearance: every link hull of one arm against the other arm ------------------------------------------------------

class Clearance:
    def __init__(self):
        from yam_arena import analysis as A

        A.link_hulls = functools.lru_cache(None)(A.link_hulls)
        self.A = A
        self.min, self.at, self.pair, self.step = math.inf, None, None, 0

    def _hulls(self, cmd, base):
        from yam_arena.constants import ARM_JOINTS, FINGER_OPEN

        q = {n: float(v) for n, v in zip(ARM_JOINTS, cmd[:6])}
        f = float(cmd[6]) * FINGER_OPEN
        q["joint7"] = f
        q["joint8"] = f
        return self.A.world_hulls(self.A.fk_link_poses_world(q, base))

    def update(self, left, right) -> float:
        from yam_arena import station

        A = self.A
        self.step += 1
        hl = self._hulls(left.cmd, station.BASES["left"])
        hr = self._hulls(right.cmd, station.BASES["right"])
        best, pair = math.inf, None
        box_r = [(h.vertices.min(0), h.vertices.max(0)) for _, h in hr]
        for na, a in hl:
            alo, ahi = a.vertices.min(0), a.vertices.max(0)
            for (nb, b), (blo, bhi) in zip(hr, box_r):
                if np.maximum(blo - ahi, alo - bhi).max() > 0.06:
                    continue
                s = min(A._signed(a.vertices, b).min(), A._signed(b.vertices, a).min())
                if s < best:
                    best, pair = float(s), (f"L.{na}", f"R.{nb}")
        if best < self.min:
            self.min, self.at, self.pair = best, self.step, pair
        return best

    def summary(self) -> dict:
        return {"min_arm_link_gap_mm": round(1000 * self.min, 1) if self.min < math.inf else None,
                "min_clearance_step": self.at, "min_clearance_pair": list(self.pair) if self.pair else None}


class TipCheck:
    """Offline: the lowest pad-tip point of each gripper above the table, the tray floor and the box's top wall away
    from the open knockout. Margins in mm; negative means the tips would hit."""

    def __init__(self, task):
        self.T = task
        self.worst = {}

    def _note(self, key, margin, step):
        if key not in self.worst or margin < self.worst[key][0]:
            self.worst[key] = (margin, step)

    def update(self, clear, ctx, step):
        from yam_arena import station

        T = self.T
        for arm, name in ((ctx.left, "left"), (ctx.right, "right")):
            for hn, h in clear._hulls(arm.cmd, station.BASES[name]):
                if "tip" not in hn:
                    continue
                for x, y, z in h.vertices:
                    on_box = abs(x - T.BOX_X) <= T.BOX_D / 2 and abs(y) <= T.BOX_W / 2
                    near_open = math.hypot(x - T.OPEN_KO[0], y - T.OPEN_KO[1]) <= 0.016
                    if on_box and not near_open:
                        self._note(f"{name}_box_top", 1000 * (z - T.TOPW_Z), step)
                    else:
                        self._note(f"{name}_table", 1000 * (z - T.TOP - 0.004), step)

    def summary(self) -> dict:
        return {"tip_margin_mm": {k: [round(m, 1), s] for k, (m, s) in sorted(self.worst.items())}}


class Ctx:
    def __init__(self, emb=None):
        from yam_arena import station

        self.emb = emb
        self.flow = Flow()
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def offline(task_path: str) -> dict:
    """Kinematic run of the demo for every variant: IK error and minimum arm-to-arm clearance, no simulator."""
    sys.path.insert(0, str(Path(task_path).resolve().parent))
    import importlib

    task = importlib.import_module("task")
    out = {}
    for variant in VARIANTS:
        os.environ["DEMO_VARIANT"] = variant
        ctx = Ctx()
        clear = Clearance()
        tips = TipCheck(task)
        n = 0
        for _ in script(task, ctx):
            if n % 4 == 0:
                clear.update(ctx.left, ctx.right)
                tips.update(clear, ctx, n)
            n += 1
        out[variant or "demo"] = {"steps": n,
                                  "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
                                  "worst_ik_deg": {"left": round(ctx.left.worst_deg, 2),
                                                   "right": round(ctx.right.worst_deg, 2)},
                                  **clear.summary(), **tips.summary()}
        print(variant or "demo", json.dumps(out[variant or "demo"]), flush=True)
    return out


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def _tcp_actual(ctx) -> dict:
    """Debug only: each gripper's actual TCP (mm, world) from the simulator, for the trace."""
    from yam_arena.constants import TCP_OFFSET
    from partkit import PartState, _np

    base = ctx.emb._env.unwrapped
    origin = _np(base.scene.env_origins[0])
    out = {}
    for name, arm in (("L", ctx.left), ("R", ctx.right)):
        robot = base.scene[f"{'left' if name == 'L' else 'right'}_robot"]
        pose = _np(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")]
        rot = PartState(pos=pose[:3], quat_xyzw=pose[3:7], lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.zeros(3),
                        high=np.zeros(3)).rotation()
        tcp = pose[:3] + rot @ np.array(TCP_OFFSET) - origin
        out[name] = {"act": [round(1000 * float(v), 1) for v in tcp], "cmd": [round(1000 * float(v), 1) for v in arm.pos]}
    return out


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 4.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*"):
        stale.unlink()
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    gen = script(task, ctx)
    clearance = Clearance()
    frames: dict[str, list] = {}
    debug = bool(os.environ.get("DEMO_DEBUG"))
    trace: list = []

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
        try:
            clearance.update(ctx.left, ctx.right)
        except Exception as exc:  # noqa: BLE001 - logging only
            if steps == 1:
                print(f"clearance unavailable: {exc!r}", flush=True)
        if steps % 20 == 0 or (debug and steps % 10 == 0):  # logging only: reads simulator state
            S = task._STATE
            g = S.get("geo") or {}
            row = {"step": steps, "score": info.get("score"), "released": S["released"],
                   **{k: {kk: (round(float(vv), 2) if kk != "pos" else [round(1000 * float(x), 1) for x in vv])
                          for kk, vv in v.items()} for k, v in g.items()}}
            try:
                row["tcp"] = _tcp_actual(ctx)
            except Exception as exc:  # noqa: BLE001 - logging only
                row["tcp"] = repr(exc)
            trace.append(row)
            print(json.dumps(row), flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    S = task._STATE
    result = {"task_score": score, "success": success, "steps": steps, "variant": os.environ.get("DEMO_VARIANT", ""),
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              **clearance.summary(),
              "final": {k: {kk: (round(float(vv), 2) if kk != "pos" else [round(1000 * float(x), 1) for x in vv])
                            for kk, vv in v.items()} for k, v in (S.get("geo") or {}).items()},
              "why": list(S["why"])}
    if debug:
        result["trace"] = trace
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    if a.offline:
        offline(a.task or str(Path(__file__).with_name("task.py")))
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
