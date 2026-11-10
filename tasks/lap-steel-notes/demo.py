"""Scripted full-score demo for the lap steel task, run in the yam_arena embodiment. Open loop: every target comes from the
fixed layout in the task file and from timing.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The left arm picks the steel bar off its rest by the middle (fingers across y), carries it over the neck and lays it
across the strings over the red dot (fret 3). The right arm, fingers closed, waits near the bridge and sweeps across
string 1 to pluck it. After half a second of ringing the left arm slides the bar to the green dot (fret 5) while the right
arm lifts, comes back and lowers; the same again for the blue dot (fret 7). Then the right arm lifts away, and the
left arm lays the bar back on its rest, opens and lifts clear.

Negative variants (each must score below 1):
    DEMO_VARIANT=one_arm      only the right arm plucks, three times, the bar on its rest: the open E (0)
    DEMO_VARIANT=bar_only     the left arm places and slides the bar, nothing is plucked (0)
    DEMO_VARIANT=bar_lifted   the bar is held 10 mm above the strings at each pluck: the open E (0)
    DEMO_VARIANT=two_notes    the second slide and pluck are left out (0.5)
    DEMO_VARIANT=no_return    the bar is left on the strings after the third note (0.75)

DEMO_DEBUG=1 prints a trace every 10 steps: the bar's contact against each target dot, plucks, notes, each arm's lag and
the least arm clearance on the commanded path (also in result.json).
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
PAD_PAST_TCP = 0.013     # the pad tips reach 13 mm past the tool point
GRIP_ABOVE_CENTRE = 0.0275   # the pads pinch the bar's handle: tips 27.5 mm above the bar's axis (15 mm into the handle)


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


def seq(*gens):
    for g in gens:
        yield from g


# -- the fixed points, from the layout in the task file -------------------------------------------------------------
STRIKE_X = (0.212, 0.238)     # the tip sweeps from here to here: string 1 is at 0.23, string 2 at 0.25
PLUCK_Y = -0.14
PLUCK_TIP_Z = 0.060           # the pad tip rides at the string plane
LIFT = 0.02
BAR_DOWN = 0.002              # the bar is commanded this far below the contact: the strings carry it
FRETS = (3, 5, 7)


def plan(task):
    T = task
    top = T.TOP

    def tcp_for_bar(x, y, centre_z):
        """Tool point for a bar centred at (x, y, centre_z above the table): the pads' tips sit in its handle, above its axis."""
        return np.array([x, y, top + centre_z + GRIP_ABOVE_CENTRE + PAD_PAST_TCP])

    def tcp_tip(x, y, tip_z):
        return np.array([x, y, top + tip_z + PAD_PAST_TCP])

    return dict(tcp_for_bar=tcp_for_bar, tcp_tip=tcp_tip)


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    from yam_arena.demo import grip_rotation

    L, R = ctx.left, ctx.right
    T = task
    P = plan(task)
    variant = os.environ.get("DEMO_VARIANT", "")
    brot = grip_rotation(math.pi / 2)         # pinch across y
    rot = grip_rotation(0.0)
    frets = FRETS[:2] if variant == "two_notes" else FRETS
    bar_x = T.REST_XY[0]
    z_rest = T.BAR_CENTRE_Z + 0.003           # set down 3 mm above the posts
    z_on = T.BAR_CENTRE_Z - BAR_DOWN + (0.012 if variant == "bar_lifted" else 0.0)
    z_hover = T.BAR_CENTRE_Z + 0.05
    up = np.array([0.0, 0.0, 0.05])

    def bar_at(y, z):
        return P["tcp_for_bar"](bar_x, y, z)

    r_start = P["tcp_tip"](STRIKE_X[0], PLUCK_Y, PLUCK_TIP_Z)
    r_end = P["tcp_tip"](STRIKE_X[1], PLUCK_Y, PLUCK_TIP_Z)
    r_hi = np.array([0, 0, LIFT])

    def pluck():
        """One stroke across string 1; ends 12 mm short of string 2."""
        yield from R.move(r_end, rot, 0.5)

    def reset_right():
        yield from R.move(r_end + r_hi, rot, 0.4)
        yield from R.move(r_start + r_hi, rot, 0.6)
        yield from R.move(r_start, rot, 0.4)

    def right_ready():
        yield from R.move(r_start + [0, 0, 0.10], rot, 2.5, grip=0.0)
        yield from R.move(r_start, rot, 1.5)

    def right_away():
        yield from R.move(R.pos + [0, 0, 0.12], rot, 1.0)

    # ---------- one_arm: the right arm plucks three times, the bar stays on its rest
    if variant == "one_arm":
        yield from right_ready()
        for _ in range(3):
            yield from wait(0.8)
            yield from pluck()
            yield from wait(0.9)
            yield from reset_right()
        yield from right_away()
        yield from wait(1.0)
        return

    # ---------- left: pick the bar off its rest
    rest_hi = bar_at(T.REST_XY[1], z_hover)
    yield from both(L.move(rest_hi, brot, 2.5, grip=1.0), (right_ready() if variant != "bar_only" else wait(0.1)))
    ctx.mark("spawn_settled")
    yield from L.move(bar_at(T.REST_XY[1], T.BAR_CENTRE_Z), brot, 1.5)
    ctx.mark("before_close")
    yield from L.set_grip(0.0, 1.0)
    yield from wait(0.8)
    ctx.mark("closed")
    for i, _ in enumerate(L.move(rest_hi, brot, 2.0)):
        yield
        if i == 9:
            ctx.mark("lift_plus_10")
    # carry to the first dot, then lay the bar on the strings
    y0 = T.FRET[frets[0]]
    yield from L.move(bar_at(y0, z_hover), brot, 2.5)
    yield from L.move(bar_at(y0, z_on + 0.02), brot, 1.0)
    yield from L.move(bar_at(y0, z_on), brot, 1.0)
    yield from wait(0.6)

    for i, f in enumerate(frets):
        if i:
            y = T.FRET[f]
            slide = L.move(bar_at(y, z_on), brot, 1.2)
            if variant == "bar_only":
                yield from slide
            else:
                yield from both(slide, reset_right())
            yield from wait(0.7)
        if variant != "bar_only":
            yield from wait(0.3)
            yield from pluck()
        yield from wait(0.9)                       # the note rings half a second; the bar stays

    # ---------- finish
    if variant != "bar_only":
        yield from R.move(r_end + r_hi, rot, 0.4)
        yield from right_away()
    if variant == "no_return":
        yield from wait(2.5)
        return
    yf = T.FRET[frets[-1]]
    yield from L.move(bar_at(yf, z_hover), brot, 1.5)
    yield from L.move(bar_at(T.REST_XY[1], z_hover), brot, 2.5)
    yield from L.move(bar_at(T.REST_XY[1], z_rest), brot, 1.5)
    yield from wait(0.4)
    yield from L.set_grip(1.0, 1.0)
    yield from wait(0.3)
    yield from L.move(L.pos + [0, 0, 0.10], brot, 1.5)
    yield from wait(2.0)


# -- offline: clearance on the commanded path -----------------------------------------------------------------------
class _DryCtx:
    def mark(self, name):
        pass

    def __init__(self):
        from yam_arena import station

        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])


def commanded_path(task):
    ctx = _DryCtx()
    traj = []
    for _ in script(task, ctx):
        traj.append((ctx.left.cmd.copy(), ctx.right.cmd.copy(), ctx.left.pos.copy(), ctx.left.grip))
    return traj, ctx


def clearance(task, stride: int = 2) -> dict:
    """Least distance (m) between any link hull of one arm and the other arm, and between the held bar and the right arm,
    along the commanded path, with the step where it occurs. Uses the real URDF collision hulls."""
    from yam_arena import analysis as A
    from yam_arena import station
    from yam_arena.constants import ARM_JOINTS, FINGER_OPEN

    traj, _ = commanded_path(task)

    def hulls(cmd, base):
        q = {n: float(v) for n, v in zip(ARM_JOINTS, cmd[:6])}
        f = float(cmd[6]) * FINGER_OPEN
        q["joint7"] = f
        q["joint8"] = f
        return A.world_hulls(A.fk_link_poses_world(q, base))

    def aabb(hs):
        return np.array([h.vertices.min(0) for _, h in hs]), np.array([h.vertices.max(0) for _, h in hs])

    best = dict(arms=(np.inf, 0, None), bar=(np.inf, 0, None))
    ts = np.linspace(-1, 1, 7)
    for step in range(0, len(traj), stride):
        lc, rc, lpos, lgrip = traj[step]
        hl, hr = hulls(lc, station.BASES["left"]), hulls(rc, station.BASES["right"])
        llo, lhi = aabb(hl)
        rlo, rhi = aabb(hr)
        gap = np.maximum(rlo[None] - lhi[:, None], llo[:, None] - rhi[None]).max(-1)
        for i, j in zip(*np.where(gap < 0.08)):
            s = min(A._signed(hl[i][1].vertices, hr[j][1]).min(), A._signed(hr[j][1].vertices, hl[i][1]).min())
            if s < best["arms"][0]:
                best["arms"] = (float(s), step, (hl[i][0], hr[j][0]))
        if lgrip < 0.5:      # bar in the left gripper: axis along x through the tool point, 
            c = lpos - np.array([0.0, 0.0, PAD_PAST_TCP + GRIP_ABOVE_CENTRE])
            pts = np.array([c + np.array([0.06 * t, 0, 0]) for t in ts])
            hpts = np.array([c + np.array([0.015 * t, 0, 0.0275]) for t in (-1, 0, 1)])
            for name, h in hr:
                s = min(float(A._signed(pts, h).min()) - 0.0125, float(A._signed(hpts, h).min()) - 0.02)
                if s < best["bar"][0]:
                    best["bar"] = (s, step, (name,))
    # pairs whose boxes are more than 8 cm apart are not examined: no pair found means more than 80 mm everywhere
    return {k: (dict(min_mm=round(1000 * v[0], 1), step=v[1], pair=v[2]) if np.isfinite(v[0])
                else dict(min_mm=">80 (boxes never within 80 mm)", step=None, pair=None)) for k, v in best.items()}


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.task = None
        self.marks: dict = {}
        self.steps = 0

    def mark(self, name: str) -> None:
        """Log, for result.json, the bar and its handle (world, mm) against the left tool point, now and at this step."""
        from partkit import _part_state
        from yam_arena import ik

        try:
            T = self.task
            bar = _part_state(self.emb._env, T.BAR)
            handle = bar.to_world([0.0, 0.0, T.HANDLE_Z])
            rob = self.emb._env.unwrapped.scene["left_robot"]
            q = np.asarray(getattr(rob.data.joint_pos, "torch", rob.data.joint_pos)[0].detach().cpu(), float)[:6]
            tcp, _ = ik.tcp_pose(q, self.left.base)
            ct = T.bar_contact(bar)
            rec = dict(step=self.steps, bar_mm=np.round(1000 * np.asarray(bar.pos), 1).tolist(),
                       bar_quat_xyzw=np.round(np.asarray(bar.quat_xyzw), 4).tolist(),
                       handle_mm=np.round(1000 * handle, 1).tolist(),
                       left_cmd_tcp_mm=np.round(1000 * self.left.pos, 1).tolist(),
                       left_actual_tcp_mm=np.round(1000 * tcp, 1).tolist(),
                       handle_minus_actual_tcp_xy_mm=np.round(1000 * (handle[:2] - tcp[:2]), 1).tolist(),
                       bar_contact_y_mm=round(1000 * ct["y"], 1), bar_angle_deg=round(ct["angle"], 1))
        except Exception as e:  # logging only
            rec = dict(step=self.steps, error=repr(e))
        self.marks[name] = rec
        print("MARK", name, json.dumps(rec), flush=True)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))

    def lag_mm(self) -> dict:
        """Distance (mm) from each arm's commanded tool point to where its joints put it."""
        from yam_arena import ik

        out = {}
        for side, arm in (("left", self.left), ("right", self.right)):
            rob = self.emb._env.unwrapped.scene[f"{side}_robot"]
            q = np.asarray(getattr(rob.data.joint_pos, "torch", rob.data.joint_pos)[0].detach().cpu(), float)[:6]
            tcp, _ = ik.tcp_pose(q, arm.base)
            out[side] = round(1000 * float(np.linalg.norm(tcp - arm.pos)), 1)
        return out


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
    ctx.task = task
    debug = bool(os.environ.get("DEMO_DEBUG"))
    clr = None
    try:
        clr = clearance(task)
        print("CLEARANCE", json.dumps(clr), flush=True)
    except Exception as e:  # logging only
        print("clearance check failed:", repr(e), flush=True)
    gen = script(task, ctx)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    max_lag = {"left": 0.0, "right": 0.0}
    seen_events = 0
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        ctx.steps = steps
        info = res.info
        snap(res.observation)
        S = task._STATE
        lag = ctx.lag_mm()
        for k, v in lag.items():
            max_lag[k] = max(max_lag[k], v)
        if debug:
            while seen_events < len(S["events"]):
                print(f"EVENT step {S['events'][seen_events][0]}: {S['events'][seen_events][1]}", flush=True)
                seen_events += 1
        if steps % 100 == 0 or (debug and steps % 10 == 0):
            msg = f"step {steps} score {info.get('score')}"
            if debug:
                ct = S["last"] or {}
                vs = {f: round(1000 * (ct.get("y", 0) - task.FRET[f]), 1) for f in task.CARD_ORDER}
                msg += (f" bar_y_minus_dot_mm {vs} gap_mm {round(1000 * ct.get('gap', 0), 1)} on {ct.get('on')}"
                        f" plucks {[(a, c) for a, b, c in S['plucks']]} played {S['played']} heard {S['notes_heard']}"
                        f" lag_mm {lag} min_clearance_mm {clr} failed {S['failed']}")
            print(msg, flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    S = task._STATE
    result = {"task_score": score, "success": success, "steps": steps, "failed": S.get("failed"),
              "notes_played": S.get("played"), "notes_heard": S.get("notes_heard"),
              "plucks": [(a, b, c) for a, b, c in S.get("plucks", [])], "events": S.get("events"),
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)},
              "max_lag_mm": max_lag, "arm_clearance": clr, "marks": ctx.marks}
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
