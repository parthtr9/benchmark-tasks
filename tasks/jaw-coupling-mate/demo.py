"""Scripted full-score demo for jaw-coupling-mate, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]
    python demo.py task.py --ik          # no simulator: IK error of the whole plan, step by step

Every target comes from the task file's fixed layout and the clock, and one decision from the arms' own joints: the
phase search stops turning when the right arm's measured tool point (forward kinematics of its own joint readings)
jumps forward, which is the hub dropping into the gaps. No object pose, joint state of the coupling or task state
steers it (the trace printed every 10 steps reads them for logging only).

1. The right hand takes the spider out of its holder by the tips of its two level legs (from outside the coupling's
   diameter, the only place fingers fit once the spider is between jaws), lowers it into the space between the fixed
   hub's jaws and the driven shaft's end, and moves it along the shafts into the jaws until it stops on the hub's face.
   Meanwhile the left hand comes down on the driven shaft's free end and closes on it.
2. The right hand puts its closed fingertips behind the loose hub and pushes it along the shaft until its jaw tips
   stand on the fixed hub's jaw tips, and keeps a light push on it.
3. The left hand swings round the shaft's axis like a crank, turning the shaft (and the hub on its key): quickly to
   15 degrees, where the jaws have dropped onto the spider's legs, then at 2.5 degrees per second until the right
   arm's own joint readings show the hub jumping forward into the free gaps (searched to 60 degrees, which covers
   any slip of the shaft in the grip); then the turn stops.
4. The right hand pushes the hub home (it stops with its jaw tips on the fixed hub's face, at the scribed line) and
   both hands let go and back off.

Negative variants (DEMO_VARIANT=...):
  no_spider  the spider is left out and the hubs are closed on each other (0)
  no_turn    the shaft is not turned: the hub stops with its jaw tips on the fixed hub's jaw tips (0.25)
  on_legs    the shaft is turned only part of the way: the hub stops with its jaws on the spider's legs (0.25)
  short      the hub is not pushed home after its jaws drop into the gaps (0.5)
  unseated   the spider is let go in the space between the hubs, not moved into the fixed hub's jaws (0)
Writes one MP4 per camera, setup.png and result.json; DEMO_DEBUG=0 silences the trace.
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
VARIANTS = ("", "no_spider", "no_turn", "on_legs", "short", "unseated")

# -- the plan's numbers ---------------------------------------------------------------------------------------------------
PITCH = math.radians(55.0)       # the right hand points 55 degrees below level (35 degrees from straight down): reach
OPEN_SPIDER = 0.85               # finger opening (0..1 of 94 mm) round the spider before closing
GRIP_SPIDER = 0.60               # commanded on the 63.6 mm across the leg tips: pads set to 56 mm
CARRY_UP = 0.075                 # the spider's centre above the shafts' axis while it is carried over the coupling
SEAT_PAST = 0.002                # the spider is moved 2 mm past its seat (the hub's face stops it)
SPIDER_TRIM = np.array([0.0, 0.0006, -0.0006])   # added to the spider's targets at the coupling: the arm's own
#                                  tracking error there in a sim run (0.6 mm to the right and high)
EASE_SPIDER = 0.72               # opening that lets the spider go by 2 mm a side, so it squares itself in the jaws
GRIP_SHAFT = 0.05                # commanded on the 15.9 mm shaft: pads set to 5 mm, for a firm hold
L_PITCH = math.radians(45.0)     # the left hand points 45 degrees below level onto the shaft's free end
STUB_TCP = 0.2180                # the left hand's tool point on the shaft, back from the fixed hub's jaw tips
STUB_OVER = 0.080                # it comes down onto the shaft from this far above
FAST_RATE, SLOW_RATE = math.radians(10.0), math.radians(2.5)   # the rest slowly: at 10 degrees per second the jaws
#                                  crossed the gaps (a window of about 2 degrees) before the pushing arm could follow
SEARCH_FAST = math.radians(-15.0)  # r2: the quick part of the turn stops this far, well short of the gaps (30 degrees)
SEARCH_END = math.radians(-60.0)   # r2: the slow search goes on to here if no drop is felt: covers 30 degrees of slip
DROP_MM = 4.0                      # r2: the pushing hand's measured tool point this much further along the shafts
#                                    (more than the 2.5 mm drop onto the spider's legs, which comes during the push)
DROP_WINDOW = 10                   # within this many steps (0.5 s), once the push is stalled: the hub has dropped
PUSH_UP = 0.026                  # the pushing hand's tool point above the axis: its closed fingertips, edge on to the
#                                  hub, touch the rear face 19 mm above the axis and clear the shaft by 4 mm
PUSH_TIP = 0.0095                # the fingertips' leading edge is this far ahead of the tool point along the shafts
PUSH_BEHIND = PUSH_TIP + 0.004   # the tool point starts with that edge 4 mm behind the loose hub's rear face
PUSH_LEAD = 0.006                # commanded past contact for a steady push of a few newtons
PUSH_HOME = 0.008                # and past the line at the end (the fixed hub's face stops the loose hub)


def grip(heading: float, pitch: float, flip: float = 1.0, roll: float = 0.0, edge_on: bool = False) -> np.ndarray:
    """Tool rotation (columns: closing axis, y, approach) for a hand heading ``heading`` in the bench plane and pointing
    ``pitch`` below level, rolled ``roll`` about the world x axis; ``flip`` -1 is the same parallel grip turned half a
    turn about its approach axis. ``edge_on``: the fingers close in the hand's vertical plane instead of across it (a
    quarter turn about the approach axis), so closed fingertips meet a face ahead of them edge on."""
    h = np.array([math.cos(heading), math.sin(heading), 0.0])
    z = math.cos(pitch) * h + np.array([0.0, 0.0, -math.sin(pitch)])
    x = flip * np.array([-h[1], h[0], 0.0])
    if edge_on:
        x = np.cross(z, x)
    r = np.c_[x, np.cross(z, x), z]
    c, s = math.cos(roll), math.sin(roll)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]) @ r


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each from where the last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.pos = np.asarray(self.pos, float)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, _ = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None, linear: bool = False):
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = k / n if linear else ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def hold(self, duration: float, grip: float | None = None):
        yield from self.move(self.pos, self.rot, duration, grip)

    def roll(self, rot_at, a0: float, a1: float, duration: float):
        """The hand turned about the shafts' axis through its tool point, at a steady rate."""
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            self._solve(self.pos, rot_at(a0 + (a1 - a0) * k / n))
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
    """+1 or -1: which half-turn of the grip has the smaller worst IK error over ``poses`` [(pos, rotation(flip))],
    solved one after another from the arm's ready pose."""
    from yam_arena import ik
    from yam_arena.constants import READY

    out = []
    for flip in (1.0, -1.0):
        q, worst = np.array(READY, float), 0.0
        for pos, rot_of in poses:
            q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(rot_of(flip)), q, base_pos=arm.base,
                                 iters=300)
            worst = max(worst, pe + 0.05 * re)
        out.append((worst, flip))
    return min(out)[1]


def plan(T) -> dict:
    """Tool-point targets (world, m) from the task file's fixed layout."""
    mm = T.MM
    axis_z = T.TOP + T.AXIS_Z
    x0 = T.X_TIPS

    def on_axis(w_mm, up=0.0):
        return np.array([x0 + w_mm * mm, 0.0, axis_z + up])

    pick = np.array(T.SPIDER_START)
    hub_back = -(T.LTB + T.PARK)                                   # the parked hub's rear face (axial mm)
    push0 = on_axis(hub_back, PUSH_UP) - [PUSH_BEHIND, 0.0, 0.0]
    touch = on_axis(hub_back, PUSH_UP) - [PUSH_TIP, 0.0, 0.0]       # fingertips on the parked hub's rear face

    def pushed(travel_mm, lead):
        return touch + [travel_mm * mm + lead, 0.0, 0.0]

    on_tips = T.PARK - T.JAW                                       # hub travel when its jaw tips reach the fixed hub's
    on_legs = on_tips + (T.JAW - T.SPIDER_W)                       # ... and when they reach a seated spider's legs
    return {
        "pick": pick, "pick_over": pick + [0.0, 0.0, 0.09],
        "gap_over": on_axis(-T.SPACE / 2, CARRY_UP) + SPIDER_TRIM,
        "gap": on_axis(-T.SPACE / 2) + SPIDER_TRIM,
        "seat": on_axis(T.JAW - T.SPIDER_W / 2) + [SEAT_PAST, 0.0, 0.0] + SPIDER_TRIM,
        "seat_back": on_axis(T.JAW - T.SPIDER_W / 2) - [0.005, 0.0, 0.0] + SPIDER_TRIM,
        "seat_over": on_axis(T.JAW - T.SPIDER_W / 2, 0.10),
        "push_over": push0 + [0.0, 0.0, 0.08], "push0": push0,
        "push_tips": pushed(on_tips, PUSH_LEAD), "push_legs": pushed(on_legs, PUSH_LEAD + 0.003),
        "push_in": pushed(on_legs, PUSH_LEAD + 0.007), "push_home": pushed(T.PARK, PUSH_HOME),
        "push_up": pushed(T.PARK, PUSH_HOME) + [-0.01, 0.0, 0.09],
        "stub": on_axis(-STUB_TCP * 1000), "stub_over": on_axis(-STUB_TCP * 1000, STUB_OVER),
    }


def right_rot(flip):
    return grip(0.0, PITCH, flip)


def push_rot(flip):
    return grip(0.0, PITCH, flip, edge_on=True)


def left_rot(flip, roll=0.0):
    return grip(0.0, L_PITCH, flip, roll)


def right_poses(p) -> list:
    keys = ("pick_over", "pick", "pick_over", "gap_over", "gap", "seat", "seat_over")
    return [(p[k], right_rot) for k in keys]


def push_poses(p) -> list:
    return [(p[k], push_rot) for k in ("push_over", "push0", "push_home", "push_up")]


def left_poses(p, sweep) -> list:
    out = [(p["stub_over"], left_rot), (p["stub"], left_rot)]
    out += [(p["stub"], (lambda f, a=a: left_rot(f, a))) for a in np.linspace(0.0, sweep, 9)]
    return out + [(p["stub_over"], (lambda f: left_rot(f, sweep)))]


def seat_spider(arm, p, flip, variant=""):
    """Spider from its holder, by the tips of its two level legs, into the fixed hub's jaws."""
    r = right_rot(flip)
    yield from chain(arm.move(p["pick_over"], r, 2.5, OPEN_SPIDER), arm.move(p["pick"], r, 1.5),
                     arm.hold(0.7, GRIP_SPIDER), arm.hold(0.3), arm.move(p["pick_over"], r, 1.2),
                     arm.move(p["gap_over"], r, 2.5), arm.move(p["gap"], r, 1.8), arm.hold(0.4))
    if variant != "unseated":
        # in, let go so the spider squares itself between the jaws, take hold again and press it home
        yield from chain(arm.move(p["seat"], r, 2.5, linear=True), arm.hold(0.3), arm.hold(0.4, EASE_SPIDER),
                         arm.move(p["seat_back"], r, 0.4), arm.hold(0.4, GRIP_SPIDER),
                         arm.move(p["seat"], r, 1.0, linear=True), arm.hold(0.3))
    yield from chain(arm.hold(0.5, OPEN_SPIDER), arm.hold(0.2), arm.move(p["seat_over"], r, 1.5))


def take_shaft(arm, p, flip):
    """The left hand comes down on the shaft's free end and closes on it."""
    r = left_rot(flip)
    yield from chain(arm.move(p["stub_over"], r, 3.0, 0.45), arm.move(p["stub"], r, 1.5), arm.hold(0.6, GRIP_SHAFT),
                     arm.hold(0.3))


def search_phase(ctx, L, R, p, fl, rr):
    """r2: turn the shaft slowly from SEARCH_FAST toward SEARCH_END while the right hand keeps pushing the hub, and
    stop turning as soon as the right arm's own joint readings show the hub has dropped into the spider's free gaps.
    Steers only on the arms' own joints and the clock (forward kinematics of the right arm's measured joints)."""
    push = R.move(p["push_in"], rr, 3.0, linear=True)
    pushing, hist, a, step = True, [], SEARCH_FAST, 0
    rate = SLOW_RATE / CONTROL_HZ
    while a > SEARCH_END:
        a = max(SEARCH_END, a - rate)
        L._solve(L.pos, left_rot(fl, a))
        if pushing:
            pushing = next(push, StopIteration) is not StopIteration
        yield
        step += 1
        try:
            x = float(ctx.joint_tcp("right")[0][0])
        except Exception:  # noqa: BLE001 - no simulator (the --ik plan check): search the whole range
            continue
        hist.append(x)
        if not pushing and len(hist) > DROP_WINDOW and 1000 * (x - hist[-1 - DROP_WINDOW]) > DROP_MM:
            print(f"phase found: shaft turned {math.degrees(a):.1f} deg, hand advanced"
                  f" {1000 * (x - hist[-1 - DROP_WINDOW]):.1f} mm in {DROP_WINDOW} steps", flush=True)
            break
    else:
        print(f"phase search reached {math.degrees(SEARCH_END):.0f} deg without a drop", flush=True)
    for _ in push:                                         # finish the push if the drop came early
        yield
    yield from R.hold(0.5)


def script(task, ctx):
    T = task
    variant = os.environ.get("DEMO_VARIANT", "")
    L, R = ctx.left, ctx.right
    p = plan(T)
    sweep = math.radians(-18.0) if variant == "on_legs" else SEARCH_END
    fr, fl = best_flip(R, right_poses(p)), best_flip(L, left_poses(p, sweep))
    rr = push_rot(best_flip(R, push_poses(p)))
    # 1. spider into the fixed hub's jaws; the left hand takes the driven shaft's free end
    if variant == "no_spider":
        yield from take_shaft(L, p, fl)
    else:
        yield from both(seat_spider(R, p, fr, variant), take_shaft(L, p, fl))
    # 2. closed fingertips behind the loose hub; push it up to the fixed hub's jaw tips
    yield from chain(R.move(p["push_over"], rr, 2.0, 0.0), R.move(p["push0"], rr, 1.2),
                     R.move(p["push_tips"], rr, 4.5, linear=True))
    # 3. turn the shaft through the phase while the push goes on: onto the legs, then into the gaps
    if variant == "no_turn":
        yield from R.move(p["push_legs"], rr, 2.0, linear=True)
    elif variant == "on_legs":
        # turned only part of the way, open loop (the hub stops with its jaws on the spider's legs)
        t_fast = abs(sweep) / FAST_RATE
        yield from both(L.roll(lambda a: left_rot(fl, a), 0.0, sweep, t_fast),
                        R.move(p["push_legs"], rr, t_fast, linear=True))
    else:
        # r2: a quick turn to well short of the gaps, then a slow search that stops when the pushing arm's own joint
        # readings show the hub has dropped forward into the gaps (its tool point jumps ahead along the shafts),
        # searching on to SEARCH_END if it has not (any slip of the shaft in the grip is covered)
        t_fast = abs(SEARCH_FAST) / FAST_RATE
        yield from both(L.roll(lambda a: left_rot(fl, a), 0.0, SEARCH_FAST, t_fast),
                        R.move(p["push_legs"], rr, t_fast, linear=True))
        yield from search_phase(ctx, L, R, p, fl, rr)
    # 4. push the hub home, let go, back off
    if variant not in ("no_turn", "on_legs", "short"):
        yield from chain(R.move(p["push_home"], rr, 2.5, linear=True), R.hold(0.4))
    yield from both(chain(L.hold(0.5, 0.45), L.move(p["stub_over"], L.rot, 1.5)),
                    chain(wait(0.3), R.move(R.pos + [-0.012, 0.0, 0.0], rr, 0.5),
                          R.move(p["push_up"], rr, 1.5)))
    yield from wait(1.0)


class Ctx:
    def __init__(self, emb=None):
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


# -- recording (video only: nothing here changes the physics, the scorer or what a policy sees) ------------------------
VIDEO_W, VIDEO_H = 1280, 720     # recorded videos are rendered at 1280 x 720 (the policy's cameras stay as the run sets them)
CLOSE_CAM_TARGET = (0.385, 0.0, 0.090)    # world x, y and height above the bench: the coupling, between both hubs' jaws
CLOSE_CAM_FROM = (0.385, 0.225, 0.160)    # the close camera: on the left of the coupling (no arm passes there), a little
#                                          above the shafts' axis, 0.23 m away, looking straight across the coupling


def _close_camera(top: float):
    """Pose (world position, xyzw quaternion, "world" convention: the camera looks along its +x) of the close camera."""
    tx, ty, tz = CLOSE_CAM_TARGET
    fx, fy, fz = CLOSE_CAM_FROM
    d = np.array([tx - fx, ty - fy, tz - fz])
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(-d[2], math.hypot(d[0], d[1]))          # positive: looking down
    qz = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    qy = (0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2))
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qy
    q = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
    return (fx, fy, top + fz), q


def _record_in_detail(top: float) -> None:
    """Render the recorded cameras at VIDEO_W x VIDEO_H and add yam_arena's spare camera ("overview_cam") as a close
    view of the coupling, by subclassing the embodiment when yam_arena.embodiment is first imported (that import has
    to wait until Isaac Sim's app is up, so it is hooked, not done here). The arms, the parts, the scorer and the
    policy's observation are unchanged."""
    import importlib.abc
    import importlib.util

    pos, rot = _close_camera(top)

    def patch(E):
        if getattr(E, "_close_cam_patched", False):
            return
        E.OVERVIEW_CAM_POS, E.OVERVIEW_CAM_ROT_XYZW = tuple(pos), tuple(rot)
        base = E.BimanualYamEmbodiment

        class _Detailed(base):
            def __init__(self, *a, **k):
                k["overview_cam"], k["cam_height"], k["cam_width"] = True, VIDEO_H, VIDEO_W
                super().__init__(*a, **k)

        E.BimanualYamEmbodiment = _Detailed
        E._close_cam_patched = True

    if "yam_arena.embodiment" in sys.modules:
        patch(sys.modules["yam_arena.embodiment"])
        return

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "yam_arena.embodiment":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                patch(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())


def _label(img, text):
    import cv2

    out = np.ascontiguousarray(img).copy()
    cv2.putText(out, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 3.0) -> dict:
    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    if os.environ.get("DEMO_VARIANT") == "no_spider":
        os.environ["JAW_COUPLING_NO_SPIDER"] = "1"
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    close = record and os.environ.get("DEMO_CLOSE_CAM", "1") != "0"
    if close:
        from yam_arena.task_file import TABLE_TOP

        _record_in_detail(TABLE_TOP["station"])
        cam_width, cam_height = VIDEO_W, VIDEO_H
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
            if close:
                try:
                    data = emb._base.scene["overview_cam"].data.output["rgb"]
                    data = data.torch if hasattr(data, "torch") else data
                    frames.setdefault("close_cam", []).append(np.ascontiguousarray(data[0, ..., :3].cpu().numpy()))
                except Exception as exc:  # noqa: BLE001 - the close view is a recording aid only
                    if "close_cam" not in frames:
                        print(f"close camera unavailable: {exc!r}", flush=True)
                        frames["close_cam"] = []

    last_gap = {"mm": None}
    origin = np.array(task.ORIGIN)

    def row(steps, info):
        """Logging only: reads simulator state for the trace."""
        from yam_arena.scene import object_states

        st = object_states(emb._env, emb.task.names)
        S = task._STATE
        rep = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in task.status(st).items() if k != "spider"}
        out_row = {"step": steps, "score": info.get("score"), "status": rep, "gap_mm": last_gap["mm"],
                   "spider_free": S["spider_free"], "all_free": S["all_free"], "still": S["report"].get("still"),
                   "tcp": {"L": np.round(1000 * (ctx.left.pos - origin), 1).tolist(),
                           "R": np.round(1000 * (ctx.right.pos - origin), 1).tolist(),
                           "L_meas": np.round(1000 * (ctx.joint_tcp("left")[0] - origin), 1).tolist(),
                           "R_meas": np.round(1000 * (ctx.joint_tcp("right")[0] - origin), 1).tolist(),
                           "grip": [round(float(ctx.left.grip), 2), round(float(ctx.right.grip), 2)],
                           "fingers_mm": ctx.fingers()}}
        if "spider" in st:
            sp = st["spider"]
            R = sp.rotation()
            out_row["spider"] = {"xyz_mm": np.round(1000 * (sp.pos - origin), 2).tolist(),
                                 "tilt_deg": round(math.degrees(math.acos(min(1.0, abs(float(R[0, 0]))))), 2),
                                 "phase_deg": round(math.degrees(math.atan2(float(R[2, 1]), float(R[1, 1]))), 2)}
        return out_row

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
              "min_arm_link_gap_mm": round(1000 * min_gap, 1) if min_gap < float("inf") else None,
              "report": S["report"], "finger_check": S["finger_check"]}
    if record:
        import imageio

        for cam, fs in frames.items():
            if fs:
                _write_mp4(fs, out / f"demo_{cam}.mp4")
        if frames.get("close_cam") and frames.get("top_cam"):
            # the reviewer's video: the whole bench (top camera) and the coupling close up, side by side, 1280 x 360
            import cv2

            n = min(len(frames["top_cam"]), len(frames["close_cam"]))
            both = []
            for k in range(n):
                a = cv2.resize(frames["top_cam"][k], (640, 360))
                b = cv2.resize(frames["close_cam"][k], (640, 360))
                both.append(np.hstack([_label(a, "whole bench"), _label(b, f"coupling, close  step {k}")]))
            _write_mp4(both, out / "demo_detail.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)
    return result


def dry_run(task_path: str, variant: str = ""):
    """No simulator: play the script and return every step's commands and each arm's worst IK error (mm)."""
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
