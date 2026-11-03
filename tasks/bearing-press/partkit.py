"""partkit: rigid parts built from several boxes, optionally with spring-loaded sliders, for yam_arena task files.

A task file uses it like this (keep this file next to the task file):

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from partkit import Part, Slider, on_step, on_reset

    LID = Part("lid", pos=(0.2, 0.0, 0.76), yaw=0.0, mass=0.04,
               boxes=[((0, 0, 0), (0.114, 0.064, 0.0075), (0.1, 0.3, 0.9))])
    OBJECTS = [LID, ...]

yam_arena's task-file objects are single primitives. A ``Part`` is one rigid body with several box and cylinder colliders
(a box with a pocket, a lid with tabs, a cap with lugs). Give it ``sliders`` and it becomes an articulation whose root is the part and whose
sliders are shapes on prismatic joints along one of the part's axes, each pushed to its rest stop by a spring drive.

Importing this module hooks ``yam_arena.scene`` (and the embodiment's step and reset) without modifying yam_arena: ``success()/score()`` receive a :class:`PartState` (the body origin's pose, not the bounding-box centre that
yam_arena reports for primitives, plus slider positions). ``on_step`` and ``on_reset`` register callbacks that run
after every control step and after every reset, with the Isaac Lab environment.

Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena import task_file
from yam_arena.task_file import Box, ObjectState

__all__ = ["Part", "PartState", "Slider", "on_reset", "on_step", "env", "write_pose", "install"]

Color = tuple[float, float, float]
BoxSpec = tuple[tuple[float, float, float], tuple[float, float, float], Color]  # centre, size, colour (part frame, m)
CylSpec = tuple  # (centre, radius, height, colour[, axis "X" | "Y" | "Z" = "Z"]) in the part frame


@dataclass(frozen=True)
class Slider:
    """Shapes on a prismatic joint along the part's ``axis`` ("X", "Y" or "Z"). ``rest`` is where its frame sits in
    the part frame; it can move by ``travel`` in the axis's negative direction, and a spring of ``rate`` N/m pushes it
    back to ``rest``."""

    name: str
    rest: tuple[float, float, float]
    boxes: tuple[BoxSpec, ...] = ()
    cylinders: tuple = ()
    travel: float = 0.004
    rate: float = 2500.0
    damping: float = 20.0
    mass: float = 0.002
    axis: str = "Z"


@dataclass(frozen=True)
class Part(Box):
    """A rigid part made of ``boxes``. It subclasses Box only so yam_arena's task-file checks accept it; ``size`` is
    its bounding box and ``pos`` is the body origin."""

    boxes: tuple[BoxSpec, ...] = ()
    cylinders: tuple[CylSpec, ...] = ()
    sliders: tuple[Slider, ...] = ()
    friction: float = 0.8
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def _check(self) -> None:
        if not self.boxes:
            raise ValueError(f"{self.name}: a Part needs at least one box")
        if self.mass is None or self.mass <= 0:
            raise ValueError(f"{self.name}: a Part must be loose (give it a mass)")


def part(name, pos, boxes, mass, yaw=0.0, sliders=(), color=(0.5, 0.5, 0.5), friction=0.8, cylinders=()) -> Part:
    """Build a Part; ``size`` is filled in from the shapes' extent."""
    def cyl_half(spec):
        r, h, axis = spec[1], spec[2], spec[4] if len(spec) > 4 else "Z"
        return np.array([h / 2 if axis == "X" else r, h / 2 if axis == "Y" else r, h / 2 if axis == "Z" else r])

    half = [np.array(s) / 2 for _, s, _ in boxes] + [cyl_half(c) for c in cylinders]
    centres = [np.array(c) for c, _, _ in boxes] + [np.array(c[0]) for c in cylinders]
    lo = np.min([c - h for c, h in zip(centres, half)], axis=0)
    hi = np.max([c + h for c, h in zip(centres, half)], axis=0)
    return Part(name=name, pos=tuple(pos), size=tuple((hi - lo).tolist()), mass=mass, color=color, yaw=yaw,
                boxes=tuple(boxes), cylinders=tuple(cylinders), sliders=tuple(sliders), friction=friction)


@dataclass(frozen=True)
class PartState(ObjectState):
    """The body origin's pose (``pos``, ``quat_xyzw``), velocities, bounds, and each slider's position (m, 0 at its
    rest stop, negative when pressed)."""

    sliders: dict = field(default_factory=dict)

    def rotation(self) -> np.ndarray:
        x, y, z, w = self.quat_xyzw
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def to_local(self, world_point) -> np.ndarray:
        return self.rotation().T @ (np.asarray(world_point, float) - self.pos)

    def to_world(self, local_point) -> np.ndarray:
        return self.pos + self.rotation() @ np.asarray(local_point, float)


# -- spawning -----------------------------------------------------------------------------------------------------

_SPECS: dict[str, Part] = {}


def _spawn_part(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the part under ``prim_path`` (pxr only): a plain Xform holding the body (and sliders) as siblings."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec: Part = _SPECS[cfg.part_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    phys = f"{prim_path}/Looks/physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    pm.func(phys, pm)
    looks: dict[Color, str] = {}

    def look(color: Color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.5)
            vis.func(path, vis)
            looks[color] = path
        return looks[color]

    def body(path: str, pos, mass: float):
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)
        rb.CreateAngularDampingAttr(1.0)  # light damping: small stiff parts otherwise chatter in tight fits
        return prim

    def collide(prim, color):
        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(0.0005)  # below the 1 mm fits, so tight parts do not touch on every face at once
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(prim.GetPath()), phys, stage=stage)
        bind_visual_material(str(prim.GetPath()), look(color), stage=stage)

    def boxes(path: str, specs, offset=(0.0, 0.0, 0.0)):
        for i, (centre, size, color) in enumerate(specs):
            cube = UsdGeom.Cube.Define(stage, f"{path}/box_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*(np.array(centre) - np.array(offset))))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            collide(cube.GetPrim(), color)

    def cylinders(path: str, specs, offset=(0.0, 0.0, 0.0)):
        for i, spec in enumerate(specs):
            centre, radius, height, color = spec[:4]
            cyl = UsdGeom.Cylinder.Define(stage, f"{path}/cyl_{i}")
            cyl.CreateAxisAttr(spec[4] if len(spec) > 4 else "Z")
            cyl.CreateRadiusAttr(radius)
            cyl.CreateHeightAttr(height)
            cyl.AddTranslateOp().Set(Gf.Vec3d(*(np.array(centre) - np.array(offset))))
            collide(cyl.GetPrim(), color)

    main = f"{prim_path}/body"
    body(main, (0.0, 0.0, 0.0), spec.mass)
    boxes(main, spec.boxes)
    cylinders(main, spec.cylinders)
    for s in spec.sliders:
        path = f"{prim_path}/{s.name}"
        body(path, s.rest, s.mass)
        boxes(path, s.boxes, offset=s.rest)
        cylinders(path, s.cylinders, offset=s.rest)
        j = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/{s.name}_slide")
        j.CreateAxisAttr(s.axis)
        j.CreateBody0Rel().SetTargets([Sdf.Path(main)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        j.CreateLocalPos0Attr(Gf.Vec3f(*s.rest))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLowerLimitAttr(-s.travel)
        j.CreateUpperLimitAttr(0.0)
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(s.rate)
        drive.CreateDampingAttr(s.damping)
        drive.CreateTargetPositionAttr(0.0)
    if spec.sliders:
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(main))
        art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(main))
        art.CreateEnabledSelfCollisionsAttr(False)
        art.CreateSolverPositionIterationCountAttr(32)
        art.CreateSolverVelocityIterationCountAttr(4)
    return root.GetPrim()


def _arena_object(spec: Part):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class PartSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_part)
        part_name: str = ""

    class PartObject(Object):
        def _generate_articulation_cfg(self):
            # Implicit actuators carry the springs; their default target is each slider's rest stop.
            actuators = {s.name: ImplicitActuatorCfg(joint_names_expr=[f"{s.name}_slide"], stiffness=s.rate,
                                                     damping=s.damping) for s in spec.sliders}
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators)
            return self._add_initial_pose_to_cfg(cfg)

    kind = ObjectType.ARTICULATION if spec.sliders else ObjectType.RIGID
    return PartObject(
        name=spec.name,
        object_type=kind,
        spawner_cfg=PartSpawnerCfg(part_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2),
                                                                       math.cos(spec.yaw / 2))),
    )


def _t(x):
    return getattr(x, "torch", x)


def _np(x) -> np.ndarray:
    return _t(x).detach().cpu().numpy().astype(np.float64)


def _part_state(environment, spec: Part) -> PartState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    probe = PartState(pos=pose[:3] - origin, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6],
                      low=np.zeros(3), high=np.zeros(3))
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.array(spec.size) / 2
    pts = np.array([probe.to_world(c) for c in corners])
    sliders = {}
    if spec.sliders:
        names = list(asset.joint_names)
        q = _np(asset.data.joint_pos)[0]
        sliders = {s.name: float(q[names.index(f"{s.name}_slide")]) for s in spec.sliders}
    return PartState(pos=probe.pos, quat_xyzw=probe.quat_xyzw, lin_vel=probe.lin_vel, ang_vel=probe.ang_vel,
                     low=pts.min(0), high=pts.max(0), sliders=sliders)


def write_pose(environment, name: str, pos, quat_xyzw, lin_vel=(0, 0, 0), ang_vel=(0, 0, 0)) -> None:
    """Teleport a part's body (env 0) to a world pose (relative to the env origin) with the given velocity."""
    import torch

    base = environment.unwrapped
    asset = base.scene[name]
    origin = _t(base.scene.env_origins[0])
    pose = torch.tensor([*pos, *quat_xyzw], dtype=origin.dtype, device=origin.device).unsqueeze(0)
    pose[:, :3] += origin
    asset.write_root_pose_to_sim_index(root_pose=pose)
    v = torch.tensor([*lin_vel, *ang_vel], dtype=origin.dtype, device=origin.device).unsqueeze(0)
    asset.write_root_velocity_to_sim_index(root_velocity=v)


# -- hooking yam_arena -------------------------------------------------------------------------------------------

_STEP: list = []
_RESET: list = []
_ENV: dict = {"env": None}


def on_step(fn):
    """Register ``fn(env)`` to run after every control step, before the task file is scored."""
    _STEP.append(fn)
    return fn


def on_reset(fn):
    """Register ``fn(env)`` to run after every reset, once objects have settled."""
    _RESET.append(fn)
    return fn


def env():
    return _ENV["env"]


def _patch_scene(scene) -> None:
    if getattr(scene, "_partkit_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Part):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return orig_task_object(spec)

    def object_states(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(environment, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = _part_state(environment, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._partkit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_partkit_patched", False):
        return
    base_task_step, base_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        for fn in _STEP:
            fn(self._env)
        return base_task_step(self, *a, **k)

    def reset(self, *a, **k):
        obs = base_reset(self, *a, **k)
        _ENV["env"] = self._env
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._partkit_patched = True


class _Finder(importlib.abc.MetaPathFinder):
    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader, patch = spec.loader, self.targets[fullname]
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch(module)

        loader.exec_module = exec_module
        return spec


def install() -> None:
    """Hook yam_arena (now for modules already imported, otherwise when they are). Idempotent."""
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
assert task_file
