"""Scripted full-score demo for deal-tray-passport-retrieve, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]
    python demo.py task.py --ik          (no simulator: IK of every target along the path)

Open loop: every target comes from the task file's fixed layout (the tray's drawing, where the passport lies) and the
clock. No object pose and no joint state of anything steers it; the trace and result.json read the simulator for logging only.

1. The left hand closes its fingers, pitches forward so that its body stays clear of the glass, and sets its fingertips
   on the strip of the passport that shows on the customer's side, pressing a few millimetres into it.
2. Keeping that press, it draws the passport toward the customer along the dish's arc, up the ramp and over the rim
   (the path is the arc of the maker's drawing), then level across the worktop on to the green mat.
3. It lifts off and goes back to its ready pose.

Negative variants (DEMO_VARIANT=...):
  pinch       an open hand comes straight down on the passport where it lies and closes: the glass and the dish leave no
              grasp, and what it moves slides back (0)
  let_go      the passport is drawn up the ramp until its edge is at the rim, then let go: it slides back under the glass (0)
  drop_pinch  as let_go, then an open hand, pitched like the drawing hand so that the glass does not stop it, sets its
              two fingertips on the passport where it came to rest on the ramp, closes, lifts and goes to the mat:
              the passport is wider than the hand opens and lies on the steel, so the fingertips close over its cover
              and nothing is held (0)
  push_away   the fingertips push it toward the glass instead: it climbs the far ramp a little and slides back (0)
  overdraw    the draw is not stopped at the mat: the passport goes over the worktop's front edge (0 for good)
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
VARIANTS = ("", "pinch", "let_go", "drop_pinch", "push_away", "overdraw")

PITCH_DEG = 35.0          # the hand leans forward this far from straight down (its body stays on the customer's side)
PAD_TIP = 0.013           # the fingertips' end face is 13 mm past the tool point
HEEL = 0.0075             # that face is 15 mm wide: its lower edge (toward the robot) is the one that touches
TOUCH_U = 0.026           # where the fingertips press: this far toward the customer from the passport's middle
PRESS = 0.016             # commanded this far into the passport's top face. The fingertips' rounded ends touch about
                          # 3.5 mm later than this flat-ended estimate, so the position-controlled arm is held up about
                          # 12 mm (Isaac Sim: the draw also works at 12 and at 22 mm; at 9 mm the fingertips lose the
                          # passport on the ramp at 13 degrees of tilt)
PRESS_RAMP = 0.006        # and this much further by the rim: the arm lags behind the draw, and on the slope that eases the press
DRAW_SPEED = 0.025        # m/s along the path
SLIP = 0.020              # the arm lags about 12 mm behind the draw and the fingertips creep a little: aim this much further
REST_DX, REST_DY, REST_DZ = 0.045, -0.009, 0.0055      # a let-go passport's resting place from its start (drop_pinch):
                                                       # toward the customer, along the window, and its middle's rise
FLIP = np.diag([-1.0, -1.0, 1.0])       # the same grip turned half a turn about its approach axis


def cols(closing, approach):
    x, z = np.asarray(closing, float), np.asarray(approach, float)
    return np.column_stack([x, np.cross(z, x), z])


def pitched(deg: float = PITCH_DEG):
    """Fingers side by side along y, the approach axis leaning forward from straight down."""
    b = math.radians(deg)
    return cols((0.0, 1.0, 0.0), (math.sin(b), 0.0, -math.cos(b)))


DOWN_X = cols((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))        # straight down, fingers closing along x (the pinch variant)


def best_grip(arm, positions, rot):
    """The grip (rot or rot turned half a turn about its approach) whose worst IK error over ``positions``, solved one
    after another from the arm's ready pose, is smaller, with the wrist roll kept off its +-120 degree limit."""
    from yam_arena import ik

    out = []
    for R in (rot, rot @ FLIP):
        q, worst = arm.q.copy(), 0.0
        for pos in positions:
            q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(R), q, base_pos=arm.base, iters=200)
            worst = max(worst, pe + 0.05 * re + 0.01 * max(0.0, abs(float(q[5])) - 1.9))
        out.append((worst, R))
    return min(out, key=lambda t: t[0])[1]


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each from where the last command left it."""

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
        self.last_mm = 0.0
        self.read = read
        self.corr = np.zeros(3)      # no trim in this demo (kept for the offline rehearsal's interface)

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, _ = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.last_mm = pe * 1000
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None):
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def follow(self, points, rot, speed: float = DRAW_SPEED):
        """Run the tool point along a polyline at a constant speed (eased in and out over 0.5 s)."""
        from yam_arena import ik

        pts = [self.pos.copy()] + [np.asarray(p, float) for p in points]
        seg = [float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:])]
        total = sum(seg)
        cum = np.r_[0.0, np.cumsum(seg)]
        n = max(2, round(total / speed * CONTROL_HZ))
        ramp = min(0.5 * CONTROL_HZ / n, 0.5)
        for k in range(1, n + 1):
            u = k / n
            if u < ramp:                               # ease in
                s = 0.5 * u * u / ramp
            elif u > 1 - ramp:                         # ease out
                w = u - (1 - ramp)
                s = 1 - 1.5 * ramp + w - 0.5 * w * w / ramp
            else:
                s = u - 0.5 * ramp
            s = s / (1 - ramp) * total
            i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
            f = 0.0 if seg[i] <= 0 else min(1.0, (s - cum[i]) / seg[i])
            self._solve(pts[i] + f * (pts[i + 1] - pts[i]), rot)
            yield
        del ik

    def to_joints(self, q_target, duration: float, grip: float | None = None):
        """Move in joint space to ``q_target`` (no IK)."""
        from yam_arena import ik

        q0, g0 = self.q.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.q = q0 + s * (np.asarray(q_target, float) - q0)
            self.grip = g0 + s * (g1 - g0)
            pos, rot = ik.tcp_pose(self.q, self.base)
            self.pos, self.rot = np.asarray(pos, float), np.asarray(rot, float)
            self.cmd = np.r_[self.q, self.grip]
            yield


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def shift(arm, d, rot, duration: float, grip: float | None = None):
    """Generator: move by ``d`` from wherever the arm's command is when this step starts (not when it was built)."""
    yield from arm.move(arm.pos + np.asarray(d, float), rot, duration, grip)


# -- the path: where the fingertips' touching edge goes, from the tray's drawing ---------------------------------------------
def touch_path(T, end_x: float, n: int = 40):
    """The fingertip edge's targets (world x, z) while it presses on the passport and draws it from the dish's lowest
    line to where the passport's middle is at ``end_x``: (points up the arc, points over the rim, points on the worktop).

    Up the arc the passport is a chord of the dish turning about the arc's centre; over the rim it bridges from the rim
    to the arc until the fingertips pass the rim and press it flat; then it lies flat."""
    R, half, th_r, a = T.DISH_R, T.BOOK_D / 2, math.asin(T.DISH_HALF / T.DISH_R), T.BOOK_HALF_ANGLE
    ox, oz = T.GLASS_X, T.FLANGE_TOP - T.DISH_SAG + R            # the arc's centre
    arc, rim, flat = [], [], []
    for phi in np.linspace(0.0, th_r - a, n):
        t, nn = np.array([-math.cos(phi), math.sin(phi)]), np.array([math.sin(phi), math.cos(phi)])
        mid = np.array([ox, oz]) + R * math.cos(a) * np.array([-math.sin(phi), -math.cos(phi)])
        arc.append(mid + TOUCH_U * t + (T.BOOK_T - PRESS - PRESS_RAMP * phi / (th_r - a)) * nn)
    e0 = half - TOUCH_U                                           # the touching edge's distance from the passport's near edge
    p_rim = np.array([T.X_RIM, T.FLANGE_TOP])
    for d in np.linspace(0.0, e0, max(4, n // 4))[1:]:
        th2 = th_r - 2 * math.asin((T.BOOK_D - d) / (2 * R))
        q = np.array([ox, oz]) + R * np.array([-math.sin(th2), -math.cos(th2)])
        t = (p_rim - q) / np.linalg.norm(p_rim - q)
        nn = np.array([t[1], -t[0]])
        rim.append(p_rim + (d - e0) * t + (T.BOOK_T - PRESS - PRESS_RAMP) * nn)
    z_flat = T.COUNTER_TOP + T.BOOK_T - PRESS
    flat.append(np.array([T.X_RIM - 0.010, T.FLANGE_TOP + T.BOOK_T - PRESS - PRESS_RAMP / 2]))
    flat.append(np.array([T.X_FLANGE - 0.010, z_flat]))
    flat.append(np.array([end_x - TOUCH_U, z_flat]))
    return arc, rim, flat


def tool_point(touch_xz, y: float, pitch_deg: float = PITCH_DEG):
    """The tool point (world) that puts the fingertips' lower edge at ``touch_xz``."""
    b = math.radians(pitch_deg)
    a = np.array([math.sin(b), 0.0, -math.cos(b)])
    w = np.array([math.cos(b), 0.0, math.sin(b)])                # across the fingertips' end face, toward its upper edge
    return np.array([touch_xz[0], y, touch_xz[1]]) + HEEL * w - PAD_TIP * a


def plan(T, variant: str = "") -> dict:
    mat_x = (T.MAT_X[0] + T.MAT_X[1]) / 2
    end_x = T.COUNTER_X[0] - 0.06 if variant == "overdraw" else mat_x - SLIP
    arc, rim, flat = touch_path(T, end_x)
    y = T.TRAY_Y
    p = {"arc": [tool_point(q, y) for q in arc], "rim": [tool_point(q, y) for q in rim],
         "flat": [tool_point(q, y) for q in flat]}
    p["start"] = p["arc"][0]
    p["above"] = p["start"] + np.array([-0.005, 0.0, 0.09])
    # the pinch variant: an open hand straight down, its far finger as near the glass as its body allows
    book_top = T.BOOK_REST_Z + T.BOOK_T / 2
    p["pinch_above"] = np.array([T.GLASS_X - 0.062, y, T.FLANGE_TOP + 0.10])
    p["pinch"] = np.array([T.GLASS_X - 0.062, y, book_top - PAD_TIP - 0.003])
    # the drop_pinch variant: where a passport let go at the rim comes to rest (14 degrees up the ramp, 45 mm from the
    # lowest line and 9 mm along the window, as it did in every Isaac Sim run of let_go), and the mat
    rest = np.array([T.GLASS_X - REST_DX, T.FLANGE_TOP - T.DISH_SAG + T.BOOK_REST_GAP + REST_DZ + T.BOOK_T])   # its top face
    p["rest_touch"] = tool_point(rest - np.array([0.0, 0.006]), y + REST_DY)
    p["rest_above"] = p["rest_touch"] + np.array([-0.005, 0.0, 0.09])
    p["mat_above"] = tool_point((mat_x, T.COUNTER_TOP + 0.10), y)
    p["mat_down"] = tool_point((mat_x, T.COUNTER_TOP + 0.015), y)
    return p


def script(task, ctx):
    from yam_arena.constants import READY as READY_Q

    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    P = plan(T, variant)
    L = ctx.left
    if variant == "pinch":
        down = best_grip(L, [P["pinch_above"], P["pinch"]], DOWN_X)
        yield from L.move(P["pinch_above"], down, 3.0, 1.0)
        yield from L.move(P["pinch"], down, 2.0)
        yield from shift(L, (0, 0, 0), down, 1.2, 0.0)
        yield from wait(0.5)
        yield from L.move(P["pinch_above"], down, 1.5)
        yield from L.to_joints(READY_Q, 2.0, 1.0)
        yield from wait(3.0)
        return
    rot = best_grip(L, [P["above"], P["start"], P["arc"][-1], P["flat"][-1]], pitched())
    yield from L.move(P["above"], rot, 3.0, 0.0)
    yield from L.move(P["start"], rot, 1.5)
    yield from wait(0.4)
    if variant == "push_away":
        yield from shift(L, (0.03, 0.0, 0.0), rot, 1.5)
        yield from wait(0.5)
    elif variant in ("let_go", "drop_pinch"):
        yield from L.follow(P["arc"], rot)
        yield from wait(0.4)
        if variant == "drop_pinch":
            yield from shift(L, (0.0, 0.0, 0.07), rot, 1.0)
            yield from wait(1.5)                                  # the passport slides back and comes to rest
            yield from L.move(P["rest_above"], rot, 2.0, 1.0)     # open, still pitched: the glass does not stop it
            yield from L.move(P["rest_touch"], rot, 2.0)          # both fingertips on the cover, 47 mm either side
            yield from wait(0.3)
            yield from shift(L, (0, 0, 0), rot, 1.5, 0.0)         # close
            yield from wait(0.5)
            yield from L.move(P["rest_above"], rot, 1.5)
            yield from L.move(P["mat_above"], rot, 2.5)
            yield from L.move(P["mat_down"], rot, 1.5)
            yield from shift(L, (0, 0, 0), rot, 0.8, 1.0)
            yield from L.move(P["mat_above"], rot, 1.0)
            yield from L.to_joints(READY_Q, 2.0)
            yield from wait(2.0)
            return
    else:
        yield from L.follow(P["arc"] + P["rim"] + P["flat"], rot)
        yield from wait(0.4)
    yield from shift(L, (0.0, 0.0, 0.07), rot, 1.0)
    yield from L.to_joints(READY_Q, 2.0)
    yield from wait(2.0)


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
    debug = os.environ.get("DEMO_DEBUG", "1") != "0"
    variant = os.environ.get("DEMO_VARIANT", "")
    P = plan(task, variant)
    print("PLAN " + json.dumps({k: np.round(v, 4).tolist() for k, v in P.items()}), flush=True)

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def row(steps, info):
        """Logging only: reads simulator state for the trace."""
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)["passport"]
        S = task._STATE
        hist = S.get("hist") or []
        moved = max((float(np.linalg.norm(c - hist[-1][0], axis=1).max()) for c, _free in hist), default=0.0)
        return {"step": steps, "score": info.get("score"), "level": task.level(st), "released": S["released"],
                "still_mm": round(1000 * moved, 2), "window": len(hist), "violation": S["violation"],
                "passport": np.round(np.asarray(st.pos) - [0, 0, task.TOP], 4).tolist(),
                "tilt_deg": round(task.tilt_deg(st), 1),
                "yaw_deg": round(math.degrees(math.atan2(*(st.rotation() @ np.array([1.0, 0.0, 0.0]))[[1, 0]])), 1),
                "tcp": {"L_cmd": np.round(ctx.left.pos - [0, 0, task.TOP], 4).tolist(),
                        "L_meas": np.round(ctx.joint_tcp("left")[0] - [0, 0, task.TOP], 4).tolist(),
                        "grip": round(float(ctx.left.grip), 2)},
                "ik_mm": round(ctx.left.last_mm, 1)}

    snap(obs)
    steps, ended, success, info, trace = 0, None, False, {}, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 10 == 0 or res.terminated:
            try:
                trace = row(steps, info)
                if debug:
                    print("TRACE " + json.dumps(trace, default=str), flush=True)
            except Exception as exc:  # noqa: BLE001 - logging only
                print(f"trace unavailable: {exc!r}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    S = task._STATE
    result = {"task_score": score, "success": success, "steps": steps, "variant": variant,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "violation": S["violation"], "last_trace": trace}
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


def ik_check(task_path: str) -> None:
    """No simulator: yam_arena IK error along every variant's path, step by step."""
    import importlib

    from yam_arena import station

    sys.path.insert(0, str(Path(task_path).resolve().parent))
    task = importlib.import_module(Path(task_path).stem)

    class _Ctx:
        pass

    for variant in VARIANTS:
        os.environ["DEMO_VARIANT"] = variant
        ctx = _Ctx()
        ctx.left, ctx.right = Arm("left", station.BASES["left"]), Arm("right", station.BASES["right"])
        steps, worst_at, q6 = 0, (0.0, 0), []
        for _ in script(task, ctx):
            steps += 1
            q6.append(math.degrees(float(ctx.left.q[5])))
            if ctx.left.last_mm > worst_at[0]:
                worst_at = (ctx.left.last_mm, steps)
        print(f"IK {variant or 'demo':10s} steps {steps:4d}  worst {ctx.left.worst_mm:.2f} mm at step {worst_at[1]}  "
              f"wrist roll {min(q6):.0f}..{max(q6):.0f} deg", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out", nargs="?", default="demo_out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--ik", action="store_true", help="only print the IK error along every planned path")
    a = ap.parse_args()
    if a.ik:
        ik_check(a.task)
    else:
        run(a.task, a.out, a.max_steps, record=not a.no_record)
