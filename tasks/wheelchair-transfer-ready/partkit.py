"""partkit: jointed rigid parts for yam_arena task files (this copy: link trees, free-turning joints, closed linkages).

A task file uses it like this (keep this file next to the task file):

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from partkit import Part, Slider, Hinge, Loop, on_step, on_reset, part

yam_arena's task-file objects are single primitives. A ``Part`` is one loose rigid body with several box and cylinder
colliders. Give it ``sliders`` (prismatic joints) or ``hinges`` (revolute joints) and it becomes an articulation whose
root is the part. A hinge or slider may name another one as its ``parent`` (a caster wheel on its swivel fork, a link
on a lever), a hinge with ``lower=None`` turns without limit (a wheel), and a ``Loop`` closes a linkage with a pin
joint between two links that the joint tree does not connect (a four-bar). Shapes of every link are given in the PART
frame with all joints at 0. A box may carry a fourth element, its rotation in the part frame as a (w, x, y, z)
quaternion.

Importing this module hooks ``yam_arena.scene`` (and the embodiment's step and reset) without modifying yam_arena:
``success()/score()`` receive a :class:`PartState` (the body origin's pose plus joint positions). ``on_step`` and
``on_reset`` register callbacks that run after every control step and after every reset.

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

__all__ = ["Hinge", "Loop", "Material", "Part", "PartState", "Slider", "on_reset", "on_step", "env", "part",
           "write_pose", "install", "finger_points", "gripper_points"]

Color = tuple
BoxSpec = tuple   # (centre, size, colour[, rotation wxyz]) in the part frame, metres
CylSpec = tuple   # (centre, radius, height, colour[, axis "X" | "Y" | "Z" = "Z"]) in the part frame


@dataclass(frozen=True)
class Material:
    """Contact material of one link: friction, and optionally a compliant contact (a tyre): ``stiffness`` N/m and
    ``damping`` N s/m of the contact spring."""

    friction: float = 0.8
    stiffness: float = 0.0
    damping: float = 0.0


@dataclass(frozen=True)
class Slider:
    """Shapes on a prismatic joint along the part's ``axis``. ``rest`` is where its frame sits in the part frame; it
    can move by ``travel`` in the axis's negative direction (``sign=1``: positive), and a spring of ``rate`` N/m
    pushes it back to ``rest``."""

    name: str
    rest: tuple
    boxes: tuple = ()
    cylinders: tuple = ()
    travel: float = 0.004
    rate: float = 2500.0
    damping: float = 20.0
    mass: float = 0.002
    axis: str = "Z"
    sign: int = -1
    parent: str = ""
    material: Material | None = None


@dataclass(frozen=True)
class Hinge:
    """Shapes on a revolute joint about the part's ``axis`` through ``pivot`` (part frame), given at angle 0. It turns
    between ``lower`` and ``upper`` (rad, counter-clockwise positive about the axis); ``lower=None`` means no limits.
    A spring of ``stiffness`` N m/rad pulls it to ``target`` (which may lie beyond a limit, as a preload), ``damping``
    is in N m s/rad and ``friction`` is the joint's dry friction torque (N m). ``parent`` names the hinge or slider
    that carries this one (default: the part's body)."""

    name: str
    pivot: tuple
    boxes: tuple = ()
    cylinders: tuple = ()
    lower: float | None = 0.0
    upper: float | None = math.pi / 2
    stiffness: float = 0.0
    damping: float = 0.01
    target: float = 0.0
    friction: float = 0.0
    mass: float = 0.01
    axis: str = "Z"
    parent: str = ""
    material: Material | None = None


@dataclass(frozen=True)
class Loop:
    """A pin joint about ``axis`` through ``pivot`` (part frame) between links ``a`` and ``b`` (hinge or slider
    names; "" is the part's body) that closes a linkage. It is not part of the joint tree and has no state."""

    name: str
    a: str
    b: str
    pivot: tuple
    axis: str = "Y"


@dataclass(frozen=True)
class Part(Box):
    """A rigid part made of ``boxes`` and ``cylinders``. It subclasses Box only so yam_arena's task-file checks
    accept it; ``size`` is its bounding box and ``pos`` is the body origin."""

    boxes: tuple = ()
    cylinders: tuple = ()
    sliders: tuple = ()
    hinges: tuple = ()
    loops: tuple = ()
    friction: float = 0.8
    self_collide: bool = False        # links collide with each other (never a link with the one that carries it)
    com: tuple | None = None          # the body's centre of mass and principal moments (kg m2, part axes); None:
    inertia: tuple | None = None      # computed from the colliders
    angular_damping: float = 1.0
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def _check(self) -> None:
        if not self.boxes:
            raise ValueError(f"{self.name}: a Part needs at least one box")
        if self.mass is None or self.mass <= 0:
            raise ValueError(f"{self.name}: a Part must be loose (give it a mass)")
        names = [x.name for x in (*self.sliders, *self.hinges)]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: slider and hinge names must be unique")
        for x in (*self.sliders, *self.hinges):
            if x.parent and x.parent not in names:
                raise ValueError(f"{self.name}: {x.name} has unknown parent {x.parent!r}")
        for loop in self.loops:
            for end in (loop.a, loop.b):
                if end and end not in names:
                    raise ValueError(f"{self.name}: loop {loop.name} names unknown link {end!r}")


def _quat_matrix(wxyz) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _box(spec):
    """(centre, size, colour, rotation wxyz) of a box spec, with no rotation when it has none."""
    centre, size, color = spec[:3]
    return centre, size, color, (tuple(spec[3]) if len(spec) > 3 else (1.0, 0.0, 0.0, 0.0))


def part(name, pos, boxes, mass, yaw=0.0, sliders=(), color=(0.5, 0.5, 0.5), friction=0.8, cylinders=(),
         hinges=(), loops=(), **options) -> Part:
    """Build a Part; ``size`` is filled in from the body's shapes' extent."""
    def cyl_half(spec):
        r, h, axis = spec[1], spec[2], spec[4] if len(spec) > 4 else "Z"
        return np.array([h / 2 if axis == "X" else r, h / 2 if axis == "Y" else r, h / 2 if axis == "Z" else r])

    half = [np.abs(_quat_matrix(q)) @ (np.array(s) / 2) for _, s, _, q in map(_box, boxes)]
    half += [cyl_half(c) for c in cylinders]
    centres = [np.array(b[0]) for b in boxes] + [np.array(c[0]) for c in cylinders]
    lo = np.min([c - h for c, h in zip(centres, half)], axis=0)
    hi = np.max([c + h for c, h in zip(centres, half)], axis=0)
    spec = Part(name=name, pos=tuple(float(v) for v in pos), size=tuple((hi - lo).tolist()), mass=mass, color=color,
                yaw=yaw, boxes=tuple(boxes), cylinders=tuple(cylinders), sliders=tuple(sliders), hinges=tuple(hinges),
                loops=tuple(loops), friction=friction, **options)
    spec._check()
    return spec


@dataclass(frozen=True)
class PartState(ObjectState):
    """The body origin's pose (``pos``, ``quat_xyzw``), velocities, bounds, each slider's position (m, 0 at its
    rest stop) and each hinge's angle (rad; a free hinge's angle is not wrapped)."""

    sliders: dict = field(default_factory=dict)
    hinges: dict = field(default_factory=dict)

    def rotation(self) -> np.ndarray:
        x, y, z, w = self.quat_xyzw
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def to_local(self, world_point) -> np.ndarray:
        return self.rotation().T @ (np.asarray(world_point, float) - self.pos)

    def to_world(self, local_point) -> np.ndarray:
        return self.pos + self.rotation() @ np.asarray(local_point, float)


def state_at(spec: Part, pos=None, yaw=None, quat_xyzw=None, hinges=None, sliders=None, lin_vel=(0, 0, 0),
             ang_vel=(0, 0, 0)) -> PartState:
    """A PartState of ``spec`` at a pose (default: where the task file puts it), for offline scoring tests."""
    pos = np.asarray(spec.pos if pos is None else pos, float)
    if quat_xyzw is None:
        a = spec.yaw if yaw is None else yaw
        quat_xyzw = (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))
    hs = {h.name: 0.0 for h in spec.hinges}
    hs.update(hinges or {})
    ss = {s.name: 0.0 for s in spec.sliders}
    ss.update(sliders or {})
    return PartState(pos=pos, quat_xyzw=np.asarray(quat_xyzw, float), lin_vel=np.asarray(lin_vel, float),
                     ang_vel=np.asarray(ang_vel, float), low=pos - np.array(spec.size) / 2,
                     high=pos + np.array(spec.size) / 2, sliders=ss, hinges=hs)


# -- spawning -----------------------------------------------------------------------------------------------------

_SPECS: dict[str, Part] = {}


def _origins(spec: Part) -> dict:
    """Each link's origin in the part frame: the body's is (0, 0, 0), a hinge's its pivot, a slider's its rest."""
    out = {"": (0.0, 0.0, 0.0)}
    out.update({s.name: tuple(s.rest) for s in spec.sliders})
    out.update({h.name: tuple(h.pivot) for h in spec.hinges})
    return out


def _spawn_part(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the part under ``prim_path`` (pxr only): a plain Xform holding the body and its links as siblings."""
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

    def material(path: str, m: Material) -> str:
        pm = sim_utils.RigidBodyMaterialCfg(static_friction=m.friction, dynamic_friction=m.friction)
        pm.func(path, pm)
        if m.stiffness > 0:
            api = PhysxSchema.PhysxMaterialAPI.Apply(stage.GetPrimAtPath(path))
            api.CreateCompliantContactStiffnessAttr(float(m.stiffness))
            api.CreateCompliantContactDampingAttr(float(m.damping))
        return path

    phys = material(f"{prim_path}/Looks/physics", Material(friction=spec.friction))
    looks: dict = {}

    def look(color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.5)
            vis.func(path, vis)
            looks[color] = path
        return looks[color]

    def body(path: str, pos, mass: float, com=None, inertia=None):
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(mass)
        if com is not None:
            api.CreateCenterOfMassAttr(Gf.Vec3f(*(float(c) for c in com)))
        if inertia is not None:
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(*(float(c) for c in inertia)))
            api.CreatePrincipalAxesAttr(Gf.Quatf(1, 0, 0, 0))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)
        rb.CreateAngularDampingAttr(float(spec.angular_damping))
        return prim

    def collide(prim, color, phys_path):
        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(0.0005)  # below the 1 mm fits, so tight parts do not touch on every face at once
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(prim.GetPath()), phys_path, stage=stage)
        if color is None:      # a collider only: a mesh shape (meshkit) shows the part
            UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
        else:
            bind_visual_material(str(prim.GetPath()), look(color), stage=stage)

    def shapes(path: str, link, offset, phys_path):
        for i, (centre, size, color, rot) in enumerate(map(_box, link.boxes)):
            cube = UsdGeom.Cube.Define(stage, f"{path}/box_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*(np.array(centre) - np.array(offset))))
            if rot != (1.0, 0.0, 0.0, 0.0):
                cube.AddOrientOp().Set(Gf.Quatf(*rot))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            collide(cube.GetPrim(), color, phys_path)
        for i, c in enumerate(link.cylinders):
            centre, radius, height, color = c[:4]
            cyl = UsdGeom.Cylinder.Define(stage, f"{path}/cyl_{i}")
            cyl.CreateAxisAttr(c[4] if len(c) > 4 else "Z")
            cyl.CreateRadiusAttr(radius)
            cyl.CreateHeightAttr(height)
            cyl.AddTranslateOp().Set(Gf.Vec3d(*(np.array(centre) - np.array(offset))))
            collide(cyl.GetPrim(), color, phys_path)

    origin = _origins(spec)
    main = f"{prim_path}/body"
    path_of = lambda name: f"{prim_path}/{name}" if name else main  # noqa: E731

    def frames(j, a: str, b: str, point):
        j.CreateBody0Rel().SetTargets([Sdf.Path(path_of(a))])
        j.CreateBody1Rel().SetTargets([Sdf.Path(path_of(b))])
        j.CreateLocalPos0Attr(Gf.Vec3f(*(np.array(point) - np.array(origin[a])).tolist()))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(*(np.array(point) - np.array(origin[b])).tolist()))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

    body(main, (0.0, 0.0, 0.0), spec.mass, spec.com, spec.inertia)
    shapes(main, spec, (0.0, 0.0, 0.0), phys)
    for link in (*spec.sliders, *spec.hinges):
        path = path_of(link.name)
        body(path, origin[link.name], link.mass)
        mat = phys if link.material is None else material(f"{prim_path}/Looks/physics_{link.name}", link.material)
        shapes(path, link, origin[link.name], mat)
    for s in spec.sliders:
        j = UsdPhysics.PrismaticJoint.Define(stage, f"{path_of(s.name)}/{s.name}_slide")
        j.CreateAxisAttr(s.axis)
        frames(j, s.parent, s.name, s.rest)
        j.CreateLowerLimitAttr(-s.travel if s.sign < 0 else 0.0)
        j.CreateUpperLimitAttr(0.0 if s.sign < 0 else s.travel)
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(s.rate)
        drive.CreateDampingAttr(s.damping)
        drive.CreateTargetPositionAttr(0.0)
    for h in spec.hinges:
        j = UsdPhysics.RevoluteJoint.Define(stage, f"{path_of(h.name)}/{h.name}_hinge")
        j.CreateAxisAttr(h.axis)
        frames(j, h.parent, h.name, h.pivot)
        if h.lower is not None:
            j.CreateLowerLimitAttr(math.degrees(h.lower))   # USD revolute limits are in degrees
            j.CreateUpperLimitAttr(math.degrees(h.upper))
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(h.stiffness * math.pi / 180)  # per degree in USD; the actuator sets SI values
        drive.CreateDampingAttr(h.damping * math.pi / 180)
        drive.CreateTargetPositionAttr(math.degrees(h.target))
    for loop in spec.loops:
        j = UsdPhysics.RevoluteJoint.Define(stage, f"{path_of(loop.b)}/{loop.name}_loop")
        j.CreateAxisAttr(loop.axis)
        frames(j, loop.a, loop.b, loop.pivot)
        j.CreateExcludeFromArticulationAttr(True)      # a plain pin joint that closes the linkage
        j.CreateCollisionEnabledAttr(False)
    if spec.sliders or spec.hinges:
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(main))
        art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(main))
        art.CreateEnabledSelfCollisionsAttr(bool(spec.self_collide))
        art.CreateSolverPositionIterationCountAttr(32)
        art.CreateSolverVelocityIterationCountAttr(4)
    return root.GetPrim()


def _actuator(joint: str, stiffness: float, damping: float, friction: float = 0.0):
    from isaaclab.actuators import ImplicitActuatorCfg

    if friction > 0:
        try:
            return ImplicitActuatorCfg(joint_names_expr=[joint], stiffness=stiffness, damping=damping,
                                       friction=friction, dynamic_friction=friction)
        except TypeError:
            return ImplicitActuatorCfg(joint_names_expr=[joint], stiffness=stiffness, damping=damping,
                                       friction=friction)
    return ImplicitActuatorCfg(joint_names_expr=[joint], stiffness=stiffness, damping=damping)


def _arena_object(spec: Part):
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
            # Implicit actuators carry the springs, damping and joint friction; a spring's default target is the
            # joint's 0, moved to the hinge's ``target`` after every reset (see _reset_targets).
            actuators = {s.name: _actuator(f"{s.name}_slide", s.rate, s.damping) for s in spec.sliders}
            actuators.update({h.name: _actuator(f"{h.name}_hinge", h.stiffness, h.damping, h.friction)
                              for h in spec.hinges})
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators)
            return self._add_initial_pose_to_cfg(cfg)

    kind = ObjectType.ARTICULATION if spec.sliders or spec.hinges else ObjectType.RIGID
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
    sliders, hinges = {}, {}
    if spec.sliders or spec.hinges:
        names = list(asset.joint_names)
        q = _np(asset.data.joint_pos)[0]
        sliders = {s.name: float(q[names.index(f"{s.name}_slide")]) for s in spec.sliders}
        hinges = {h.name: float(q[names.index(f"{h.name}_hinge")]) for h in spec.hinges}
    return PartState(pos=probe.pos, quat_xyzw=probe.quat_xyzw, lin_vel=probe.lin_vel, ang_vel=probe.ang_vel,
                     low=pts.min(0), high=pts.max(0), sliders=sliders, hinges=hinges)


def set_hinge_target(environment, part_name: str, hinge_name: str, angle: float) -> None:
    """Move a hinge's spring target (rad) in env 0."""
    import torch

    asset = environment.unwrapped.scene[part_name]
    j = list(asset.joint_names).index(f"{hinge_name}_hinge")
    value = torch.full((1, 1), float(angle), device=_t(asset.data.joint_pos).device)
    command = getattr(getattr(asset, "actuators", None), "target_command", None)
    if command is not None and hasattr(command, "set_position_index"):
        command.set_position_index(value=value, joint_ids=[j])
    else:  # older Isaac Lab
        asset.set_joint_position_target(value, joint_ids=[j])


def _reset_targets(environment) -> None:
    """Springs with a preload: put each sprung hinge's target where the task file says (the default is 0)."""
    for spec in _SPECS.values():
        for h in spec.hinges:
            if h.stiffness > 0 and h.target != 0.0:
                set_hinge_target(environment, spec.name, h.name, h.target)


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


def gripper_points(environment) -> dict:
    """Points on each gripper (world, relative to the env origin), per arm ("left", "right"): the two pad faces at
    the tool point and at the pad tips (6 mm outside each pad), then the tool point and points 4 and 8 cm up the
    gripper body. score() does not see the robot; an ``on_step`` hook reads these."""
    from yam_arena.constants import PAD_TIP_PAST_TCP, TCP_OFFSET

    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    out = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        pose = _np(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")]
        jp, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
        half_gap = max(abs(jp[jn.index(j)]) for j in ("joint7", "joint8") if j in jn)
        r = PartState(pos=pose[:3], quat_xyzw=pose[3:7], lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.zeros(3),
                      high=np.zeros(3)).rotation()
        pts = []
        for along in (0.0, PAD_TIP_PAST_TCP):
            centre = pose[:3] + r @ (np.array(TCP_OFFSET) + [0.0, 0.0, along]) - origin
            pts += [centre + r[:, 0] * s * (half_gap + 0.006) for s in (1.0, -1.0)]
        tcp = pose[:3] + r @ np.array(TCP_OFFSET) - origin
        pts += [tcp, tcp - 0.04 * r[:, 2], tcp - 0.08 * r[:, 2]]
        out[side] = pts
    return out


def finger_points(environment) -> list:
    """Both arms' pad points (see :func:`gripper_points`) as one list."""
    pts = gripper_points(environment)
    return [p for side in ("left", "right") for p in pts[side][:4]]


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
        _reset_targets(self._env)
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
