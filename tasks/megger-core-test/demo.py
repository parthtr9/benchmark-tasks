"""Scripted full-score demo for the megger-core-test task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

Open loop: the layout of the task file, timing and the arms' own commands only (the faulty core is fixed in the task
file). The right arm closes on the tester's crank knob and turns the crank 1.5 turns per stud at about 0.6 rev/s. The
left arm takes the prod from its holder, lowers its tip onto stud 1, 2 and 3 in turn (the crank runs only while the tip
is down) and puts the prod back; then, fingers closed, it pushes the middle flag over to red.

Negative variants (DEMO_VARIANT=...):
  one_arm      the left arm alone: probes the three studs with the crank still, puts the prod back, flags the middle (0)
  crank_only   the right arm cranks 4.5 turns, the prod stays in its holder (0)
  guess        the left arm flags the middle core without testing (0)
  wrong_flag   everything right, but core 1 is flagged (0)
  no_return    everything right, but the prod is kept in the hand and used to push the middle flag (0.75)
  slow_crank   the crank turns at 0.25 rev/s (nothing is tested) (0)
  hover        the prod's tip stops 8 mm above each stud (nothing is tested) (0)
DEMO_DEBUG=1 prints (and saves in result.json) a trace every 10 steps for every variant: crank speed, the prod's tip
against each stud, contact timers, tested flags, needle, flag angles, each arm's lag behind its command and the minimum
arm-to-arm distance (URDF collision meshes, clearance.py; the prod and the knob counted as parts held by each hand).
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




VARIANTS = ("", "one_arm", "crank_only", "guess", "wrong_flag", "no_return", "slow_crank", "hover")
RATE = 0.6               # rev/s of the crank in the demo (the tester needs 0.4 averaged over 0.5 s)
SLOW_RATE = 0.25
TURNS = 1.5              # crank turns per stud
RAMP = 0.5               # s to reach / leave the crank speed
GRASP_OFF = 0.100        # TCP above the prod's tip bottom when held: the pad tips (13 mm lower) clear the holder's mouth
KNOB_TCP_Z = 0.152       # TCP height above the table at the knob (pad tips 13 mm lower, 4 mm above the crank bar)
TRANSIT = 0.105          # the prod's tip stays this high above the table between stations (the arms reach TCP 0.2 m up)
FLAG_START_X, FLAG_END_X = 0.325, 0.382
RELEASE_TIP_Z = 0.070    # the prod's tip is let go at this height above the table (grip bottom 4 mm above the flare)


def knob_xy(T, theta):
    return np.array([T.TESTER_XY[0] - T.CRANK_R * math.sin(theta), T.TESTER_XY[1] + T.CRANK_R * math.cos(theta)])


def crank_profile(turns, rate, hz=CONTROL_HZ):
    """Angle offsets (rad, clockwise = negative) for a trapezoid speed profile: ramp up, cruise, ramp down."""
    cruise = max(0.0, turns / rate - RAMP)
    v = []
    n_up, n_c = round(RAMP * hz), round(cruise * hz)
    for k in range(1, n_up + 1):
        v.append(rate * k / n_up)
    v += [rate] * n_c
    for k in range(n_up - 1, -1, -1):
        v.append(rate * k / n_up)
    d = np.cumsum(v) / hz
    d *= turns / d[-1]
    return -2 * math.pi * d


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    T = task
    L, R = ctx.left, ctx.right
    variant = os.environ.get("DEMO_VARIANT", "")
    if variant not in VARIANTS:
        raise ValueError(f"unknown DEMO_VARIANT {variant!r}; one of {VARIANTS}")
    top = T.TOP
    rate = SLOW_RATE if variant == "slow_crank" else RATE
    ctx.theta = 0.0
    ctx.holds = False
    ctx.pm = np.zeros(3)

    probe0 = np.array([T.PROBE_XY[0], T.PROBE_XY[1], top + T.PROBE_Z0])
    off = np.array([0.0, 0.0, GRASP_OFF])
    rot_l = grip_rotation(reachable_yaw(L, probe0 + off, 0.0))
    kx = knob_xy(T, 0.0)
    knob_tcp = np.array([kx[0], kx[1], top + KNOB_TCP_Z])
    rot_r = grip_rotation(reachable_yaw(R, knob_tcp, 0.0))
    up = np.array([0.0, 0.0, 0.04])

    def tcp(origin):
        return np.asarray(origin, float) + off

    def crank(turns):
        base = ctx.theta
        for d in crank_profile(turns, rate):
            ctx.theta = base + float(d)
            p = knob_xy(T, ctx.theta)
            R.aim(np.array([p[0], p[1], top + KNOB_TCP_Z]), rot_r)
            yield

    def stud_origin(i, lift):
        x, y = T.STUD_XY[i]
        return np.array([x, y, top + T.STUD_TOP + lift])

    def flag_push(i, grip):
        """Fingers closed, down beside flag i, push it away from the arms, back off, lift."""
        y = T.FLAG_XY[i][1]
        z = T.FLAG_PUSH_Z + 0.008
        yield from L.move(np.array([FLAG_START_X, y, top + 0.18]), rot_l, 2.5, grip=0.0)
        yield from L.move(np.array([FLAG_START_X, y, z]), rot_l, 1.5)
        ctx.mark("flag_start")
        yield from L.move(np.array([FLAG_END_X, y, z]), rot_l, 2.0)
        ctx.mark("flag_pushed")
        yield from wait(0.5)
        yield from L.move(np.array([FLAG_START_X + 0.015, y, z]), rot_l, 0.8)
        yield from L.move(np.array([FLAG_START_X + 0.015, y, top + 0.16]), rot_l, 1.2)
        yield from wait(2.0)

    # -- variants that do not use the prod ------------------------------------------------------------------------
    if variant == "guess":
        yield from flag_push(T.FAULTY, 0.0)
        return
    if variant == "crank_only":
        yield from R.move(knob_tcp + up, rot_r, 3.0, grip=1.0)
        yield from R.move(knob_tcp, rot_r, 1.5)
        yield from R.set_grip(0.0, 1.0)
        yield from wait(0.3)
        for _ in range(3):
            yield from crank(TURNS)
            yield from wait(1.0)
        yield from R.set_grip(1.0, 0.8)
        yield from R.move(R.pos + np.array([0.0, 0.0, 0.03]), rot_r, 1.0)
        yield from R.move(np.array([0.20, -0.32, top + 0.17]), rot_r, 2.0)
        yield from wait(2.0)
        return

    one_arm = variant == "one_arm"
    # 1. Left over the prod, down, close; right over the knob, down, close.
    if one_arm:
        yield from L.move(tcp(probe0) + up, rot_l, 3.0, grip=1.0)
    else:
        yield from both(L.move(tcp(probe0) + up, rot_l, 3.0, grip=1.0), R.move(knob_tcp + up, rot_r, 3.0, grip=1.0))
    ctx.mark("above")
    if one_arm:
        yield from L.move(tcp(probe0), rot_l, 1.5)
        yield from L.set_grip(0.0, 1.0)
    else:
        yield from both(L.move(tcp(probe0), rot_l, 1.5), R.move(knob_tcp, rot_r, 1.5))
        yield from both(L.set_grip(0.0, 1.0), R.set_grip(0.0, 1.0))
    ctx.holds = True
    ctx.mark("gripped")
    yield from wait(0.3)
    # 2. Lift the prod clear of the holder and carry it over stud 1.
    transit_z = top + TRANSIT
    yield from L.move(tcp(np.array([probe0[0], probe0[1], transit_z])), rot_l, 2.0)
    for i in range(3):
        hover = stud_origin(i, 0.02)
        yield from L.move(tcp(hover), rot_l, 3.0 if i == 0 else 1.4)
        depth = 0.008 if variant == "hover" else -0.001
        yield from L.move(tcp(stud_origin(i, depth)), rot_l, 1.0)
        ctx.mark(f"down{i + 1}")
        yield from wait(0.3)
        if one_arm:
            yield from wait(TURNS / rate + RAMP)
        else:
            yield from crank(TURNS)
        ctx.mark(f"cranked{i + 1}")
        yield from wait(0.3)
        yield from L.move(tcp(stud_origin(i, 0.02)), rot_l, 0.6)
    # 3. Right lets go and goes up; the prod goes back, or is kept.
    yield from L.move(tcp(np.array([stud_origin(2, 0)[0], stud_origin(2, 0)[1], transit_z])), rot_l, 1.0)

    def right_away():
        if not one_arm:
            yield from R.set_grip(1.0, 0.8)
            yield from R.move(R.pos + np.array([0.0, 0.0, 0.03]), rot_r, 1.0)
            yield from R.move(np.array([0.20, -0.32, top + 0.17]), rot_r, 2.0)

    if variant == "no_return":
        # keep the prod in the hand and push the middle flag with its tip
        y = T.FLAG_XY[T.FAULTY][1]
        zt = top + 0.082                     # tip bottom just inside the tab; the hand stays within reach
        yield from both(L.move(tcp(np.array([0.318, y, transit_z])), rot_l, 2.0), right_away())
        yield from L.move(tcp(np.array([0.318, y, zt])), rot_l, 1.2)
        ctx.mark("flag_start")
        yield from L.move(tcp(np.array([0.366, y, zt])), rot_l, 2.0)
        ctx.mark("flag_pushed")
        yield from wait(1.0)
        yield from L.move(tcp(np.array([0.335, y, zt])), rot_l, 0.8)
        yield from L.move(tcp(np.array([0.335, y, zt + 0.01])), rot_l, 1.0)
        yield from wait(2.0)
        return
    home_hover = np.array([probe0[0], probe0[1], transit_z])
    yield from both(L.move(tcp(home_hover), rot_l, 2.5), right_away())
    # lower until the grip's bottom edge is just above the holder's flared mouth, then let go: the prod drops the last
    # 6 cm through the flare and seats itself (held rigidly it wedged on the sleeve in run 1)
    yield from L.move(tcp(np.array([probe0[0], probe0[1], top + RELEASE_TIP_Z])), rot_l, 2.5)
    ctx.mark("seated")
    yield from wait(0.3)
    yield from L.set_grip(1.0, 0.8)
    ctx.holds = False
    yield from wait(0.8)
    yield from L.move(L.pos + np.array([0.0, 0.0, 0.02]), rot_l, 0.8)
    if variant in ("slow_crank", "hover"):
        yield from wait(2.0)
        return
    yield from flag_push(0 if variant == "wrong_flag" else T.FAULTY, 0.0)


def _measured_tcp(ctx, name):
    """The arm's own joint readings turned into a TCP position (the demo's logging only)."""
    from yam_arena import ik
    from yam_arena.constants import ARM_JOINTS

    robot = ctx.emb._env.unwrapped.scene[f"{name}_robot"]
    names = list(robot.joint_names)
    jp = robot.data.joint_pos[0].detach().cpu().numpy()
    q = np.array([jp[names.index(j)] for j in ARM_JOINTS], float)
    pos, _ = ik.tcp_pose(q, ctx.left.base if name == "left" else ctx.right.base)
    return np.asarray(pos, float)


def _clearance(ctx, task):
    """Smallest distance between any collision mesh of one arm and the other arm or the part the other holds, at the
    commanded joints (the URDF meshes, see clearance.py): the left hand holds the prod (or the prod stands in its
    holder), the right hand holds the knob."""
    import clearance

    left, right = ctx.left, ctx.right
    T = task
    if getattr(ctx, "holds", False):
        o = left.pos - np.array([0.0, 0.0, GRASP_OFF])
    else:
        o = np.array([T.PROBE_XY[0], T.PROBE_XY[1], T.TOP + T.PROBE_Z0])
    probe = np.vstack([clearance.box_points(o + [0, 0, T.TIP_L / 2], (2 * T.TIP_R, 2 * T.TIP_R, T.TIP_L)),
                       clearance.box_points(o + [0, 0, T.TIP_L + T.GRIP_L / 2], (2 * T.GRIP_R, 2 * T.GRIP_R, T.GRIP_L))])
    k = knob_xy(T, getattr(ctx, "theta", 0.0))
    knob = clearance.box_points((k[0], k[1], T.TOP + T.KNOB_CENTRE_Z), (2 * T.KNOB_R, 2 * T.KNOB_R, T.KNOB_H))
    return clearance.min_clearance(left.q, left.grip, left.base, right.q, right.grip, right.base, held_a=probe,
                                   held_b=knob)


def _trace(ctx, task, steps, event=None):
    """One logging row: crank speed, the prod's tip against each stud, timers, tested, needle, flags, lag, clearance."""
    st = ctx.states()
    s = task._STATE
    row = {"step": steps, "event": event, "omega_rev_s": round(s["omega"], 3), "volt": s["volt"],
           "crank_deg": round(math.degrees(task.crank_angle(st["tester"])), 1),
           "tip_vs_stud_mm": [[round(1000 * d, 1), round(1000 * dz, 1)] for d, dz in s["tip"]],
           "contact": s["contact"], "timers": list(s["timers"]), "tested": list(s["tested"]),
           "needle_cmd": s["needle"], "needle_deg": round(math.degrees(st["tester"].hinges.get("needle", 0.0)), 1),
           "flag_deg": [round(math.degrees(task.flag_angle(st["block"], i)), 1) for i in range(3)],
           "probe_mm": [round(1000 * float(v), 1) for v in (st["probe"].pos - [task.PROBE_XY[0], task.PROBE_XY[1], task.TOP])],
           "probe_tilt": round(task._tilt(st["probe"]), 1), "guess": s["guess"], "wrong": s["wrong"],
           "dropped": s["dropped"], "touched": s["touched"], "released": s["released"]}
    try:
        for n, arm in (("left", ctx.left), ("right", ctx.right)):
            row[f"lag_{n}_mm"] = [round(1000 * float(v), 1) for v in (arm.pos - _measured_tcp(ctx, n))]
    except Exception as exc:  # noqa: BLE001
        row["lag_error"] = repr(exc)
    try:
        d, names = _clearance(ctx, task)
        row["arm_clear_mm"], row["arm_clear_pair"] = round(1000 * d, 1), list(names)
        if d < ctx.min_clear[0]:
            ctx.min_clear = (d, names, steps)
    except Exception as exc:  # noqa: BLE001
        row["clear_error"] = repr(exc)
    return row


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.events, self.min_clear, self.max_omega = [], (9.0, (), 0), 0.0

    def mark(self, event):
        self.events.append(event)

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
    trace, debug = [], bool(os.environ.get("DEMO_DEBUG"))
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        ctx.max_omega = max(ctx.max_omega, task._STATE['omega'])
        event = ctx.events.pop(0) if ctx.events else None
        ctx.events.clear()
        if debug and (event or steps % 10 == 0):
            row = _trace(ctx, task, steps, event)
            trace.append(row)
            print("TRACE " + json.dumps(row), flush=True)
        elif steps % 20 == 0:
            try:
                d, names = _clearance(ctx, task)
                if d < ctx.min_clear[0]:
                    ctx.min_clear = (d, names, steps)
            except Exception as exc:  # noqa: BLE001
                print(f"clearance check failed: {exc!r}", flush=True)
        if not debug and steps % 100 == 0:
            print(f"step {steps} score {info.get('score')}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "variant": os.environ.get("DEMO_VARIANT", ""),
              "final": _trace(ctx, task, steps, "final"), "tested": list(task._STATE["tested"]),
              "guess": task._STATE["guess"], "wrong": task._STATE["wrong"], "dropped": task._STATE["dropped"],
              "touched": task._STATE["touched"], "max_omega_rev_s": round(ctx.max_omega, 3),
              "min_arm_clearance_mm": round(1000 * ctx.min_clear[0], 1), "min_arm_clearance_pair": list(ctx.min_clear[1]),
              "min_arm_clearance_step": ctx.min_clear[2], "trace": trace}
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
