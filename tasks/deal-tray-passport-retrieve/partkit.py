"""partkit: rigid parts built from several boxes, optionally with spring-loaded sliders, for yam_arena task files.

A task file uses it like this (keep this file next to the task file):

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from partkit import Part, Slider, Hinge, on_step, on_reset

    LID = Part("lid", pos=(0.2, 0.0, 0.76), yaw=0.0, mass=0.04,
               boxes=[((0, 0, 0), (0.114, 0.064, 0.0075), (0.1, 0.3, 0.9))])
    OBJECTS = [LID, ...]

yam_arena's task-file objects are single primitives. A ``Part`` is one rigid body with several box and cylinder colliders
(a box with a pocket, a lid with tabs, a cap with lugs). Give it ``sliders`` and it becomes an articulation whose root is the part and whose
sliders are shapes on prismatic joints along one of the part's axes, each pushed to its rest stop by a spring drive.
``hinges`` are shapes on revolute joints (a handle, a lever) about an axis through a pivot, held by a position
drive whose target the task can move at run time (:func:`set_hinge_target`). A box may carry a fourth element, its
rotation in the part frame as a (w, x, y, z) quaternion, for slanted bars and ramps. Slider and hinge shapes are given
in the PART frame (at the slider's rest position or the hinge's angle 0), not relative to the slider or pivot.

Importing this module hooks ``yam_arena.scene`` (and the embodiment's step and reset) without modifying yam_arena: ``success()/score()`` receive a :class:`PartState` (the body origin's pose, not the bounding-box centre that
yam_arena reports for primitives, plus slider positions). ``on_step`` and ``on_reset`` register callbacks that run
after every control step and after every reset, with the Isaac Lab environment.

Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena import task_file
from yam_arena.task_file import Box, ObjectState

__all__ = ["Hinge", "Part", "PartState", "Slider", "on_reset", "on_step", "env", "set_hinge_target", "set_slider_target", "write_pose",
           "finger_points", "install", "raise_arm_bases", "set_joint_stiffness"]

Color = tuple[float, float, float]
# centre, size, colour (part frame, m), and optionally the box's rotation in the part frame as a (w, x, y, z) quaternion
BoxSpec = tuple
CylSpec = tuple  # (centre, radius, height, colour[, axis "X" | "Y" | "Z" = "Z"]) in the part frame


@dataclass(frozen=True)
class Slider:
    """Shapes on a prismatic joint along the part's ``axis`` ("X", "Y" or "Z"). ``rest`` is where its frame sits in
    the part frame; it can move by ``travel`` in the axis's negative direction (``sign=1``: positive), and a spring of
    ``rate`` N/m pushes it back to ``rest``."""

    name: str
    rest: tuple[float, float, float]
    boxes: tuple[BoxSpec, ...] = ()
    cylinders: tuple = ()
    travel: float = 0.004
    rate: float = 2500.0
    damping: float = 20.0
    mass: float = 0.002
    axis: str = "Z"
    sign: int = -1
    lower: float | None = None   # explicit joint limits (m, relative to ``rest``); override ``travel``/``sign``
    upper: float | None = None
    effort: float = 0.0          # >0: the drive's largest force (N); with rate 0 and a large damping it is Coulomb friction
    parent: str = ""             # name of a hinge this slider is mounted on (it then turns with the hinge); "" = the part's body
    roll: float = 0.0            # the slide axis turned this much about the part's x axis (rad), e.g. along a slanted cam arm


@dataclass(frozen=True)
class Hinge:
    """Shapes on a revolute joint about the part's ``axis`` ("X", "Y" or "Z") through ``pivot`` (part frame). The
    shapes are given in the part frame at angle 0; the joint turns between ``lower`` and ``upper`` (rad, positive
    counter-clockwise about the axis). A position drive of ``stiffness`` N m/rad and ``damping`` N m s/rad pulls it to
    its target (``target`` at reset; the task can move it with :func:`set_hinge_target`)."""

    name: str
    pivot: tuple[float, float, float]
    boxes: tuple[BoxSpec, ...] = ()
    cylinders: tuple = ()
    lower: float = 0.0
    upper: float = math.pi / 2
    stiffness: float = 0.0
    damping: float = 0.01
    target: float = 0.0
    mass: float = 0.01
    axis: str = "Z"
    parent: str = ""        # name of another hinge this one is mounted on (a chain); "" = the part's body
    yaw: float = 0.0        # the joint axis turned this much about the part's z (rad), e.g. along a diagonal bar
    effort: float = 0.0     # >0: the drive's largest torque (N m); with stiffness 0 and damping it acts as Coulomb friction
    spheres: tuple = ()     # (centre, radius, colour) in the part frame at angle 0, like the boxes


@dataclass(frozen=True)
class Part(Box):
    """A rigid part made of ``boxes``. It subclasses Box only so yam_arena's task-file checks accept it; ``size`` is
    its bounding box and ``pos`` is the body origin."""

    boxes: tuple[BoxSpec, ...] = ()
    cylinders: tuple[CylSpec, ...] = ()
    spheres: tuple = ()          # (centre, radius, colour)
    fixed: bool = False          # bolted to the world (a fixed-base articulation or a static body)
    sliders: tuple[Slider, ...] = ()
    hinges: tuple[Hinge, ...] = ()
    friction: float = 0.8
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def _check(self) -> None:
        if not (self.boxes or self.spheres or self.cylinders):
            raise ValueError(f"{self.name}: a Part needs at least one box")
        if self.mass is None or self.mass <= 0:
            raise ValueError(f"{self.name}: a Part must be loose (give it a mass)")


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
         hinges=(), spheres=(), fixed=False, meta=None) -> Part:
    """Build a Part; ``size`` is filled in from the shapes' extent."""
    def cyl_half(spec):
        r, h, axis = spec[1], spec[2], spec[4] if len(spec) > 4 else "Z"
        return np.array([h / 2 if axis == "X" else r, h / 2 if axis == "Y" else r, h / 2 if axis == "Z" else r])

    half = [np.abs(_quat_matrix(q)) @ (np.array(s) / 2) for _, s, _, q in map(_box, boxes)]
    half += [cyl_half(c) for c in cylinders]
    half += [np.full(3, sp[1]) for sp in spheres]
    centres = [np.array(b[0]) for b in boxes] + [np.array(c[0]) for c in cylinders] + [np.array(sp[0]) for sp in spheres]
    lo = np.min([c - h for c, h in zip(centres, half)], axis=0)
    hi = np.max([c + h for c, h in zip(centres, half)], axis=0)
    return Part(name=name, pos=tuple(pos), size=tuple((hi - lo).tolist()), mass=mass, color=color, yaw=yaw,
                boxes=tuple(boxes), cylinders=tuple(cylinders), sliders=tuple(sliders), hinges=tuple(hinges),
                friction=friction, spheres=tuple(spheres), fixed=fixed, meta=dict(meta or {}))


@dataclass(frozen=True)
class PartState(ObjectState):
    """The body origin's pose (``pos``, ``quat_xyzw``), velocities, bounds, each slider's position (m, 0 at its
    rest stop, negative when pressed) and each hinge's angle (rad)."""

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

    def look(color) -> str:
        # a colour may carry a fourth number, its opacity (clear polycarbonate panes)
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            if len(color) > 3 and color[3] < 1.0:
                try:        # clear sheet: thin-walled glass, so what stands behind it shows through
                    if os.environ.get("PARTKIT_PANES", "glass") != "glass":
                        raise ValueError("plain panes asked for")
                    vis = sim_utils.GlassMdlCfg(glass_color=tuple(color[:3]), frosting_roughness=0.0, thin_walled=True,
                                                glass_ior=1.2)
                except Exception:  # noqa: BLE001 - older Isaac Lab: a translucent preview surface
                    vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb((0.30, 0.34, 0.36)), roughness=0.3,
                                                      opacity=float(color[3]))
            else:
                vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(tuple(color[:3])), roughness=0.5)
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

    grip = f"{prim_path}/Looks/physics_grip"       # a second material for the boxes listed in meta["grip_boxes"]
    if spec.meta.get("grip_boxes") or any(k.startswith("grip_boxes_") for k in spec.meta):
        gm = sim_utils.RigidBodyMaterialCfg(static_friction=float(spec.meta.get("grip_friction", 1.0)),
                                            dynamic_friction=float(spec.meta.get("grip_friction", 1.0)))
        gm.func(grip, gm)

    def collide(prim, color):
        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(float(spec.meta.get("contact_offset", 0.0005)))  # below the tightest fit, so tight parts do not touch on every face at once
        cp.CreateRestOffsetAttr(0.0)
        patch = float(spec.meta.get("torsional_patch", 0.0))
        if patch > 0 and str(prim.GetPath()).startswith(main + "/"):   # the main body's own shapes: a finite pad
            cp.CreateTorsionalPatchRadiusAttr(patch)                    # contact resists twisting about its normal
            cp.CreateMinTorsionalPatchRadiusAttr(patch)
        bind_physics_material(str(prim.GetPath()), phys, stage=stage)
        if color is None:      # a collider only: the part is shown by its meta["visual_boxes"]
            UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
        else:
            bind_visual_material(str(prim.GetPath()), look(color), stage=stage)

    def drawn(path: str, specs):
        """Boxes that are drawn but do not collide (a part's coloured layers over one plain collider)."""
        for i, (centre, size, color, rot) in enumerate(map(_box, specs)):
            cube = UsdGeom.Cube.Define(stage, f"{path}/draw_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
            if rot != (1.0, 0.0, 0.0, 0.0):
                cube.AddOrientOp().Set(Gf.Quatf(*rot))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            bind_visual_material(str(cube.GetPrim().GetPath()), look(color), stage=stage)

    def boxes(path: str, specs, offset=(0.0, 0.0, 0.0), grip_ids=()):
        for i, (centre, size, color, rot) in enumerate(map(_box, specs)):
            cube = UsdGeom.Cube.Define(stage, f"{path}/box_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*(np.array(centre) - np.array(offset))))
            if rot != (1.0, 0.0, 0.0, 0.0):
                cube.AddOrientOp().Set(Gf.Quatf(*rot))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            collide(cube.GetPrim(), color)
            if (path == main and i in spec.meta.get("grip_boxes", ())) or i in grip_ids:
                bind_physics_material(str(cube.GetPrim().GetPath()), grip, stage=stage)

    def cylinders(path: str, specs, offset=(0.0, 0.0, 0.0)):
        for i, spec in enumerate(specs):
            centre, radius, height, color = spec[:4]
            cyl = UsdGeom.Cylinder.Define(stage, f"{path}/cyl_{i}")
            cyl.CreateAxisAttr(spec[4] if len(spec) > 4 else "Z")
            cyl.CreateRadiusAttr(radius)
            cyl.CreateHeightAttr(height)
            cyl.AddTranslateOp().Set(Gf.Vec3d(*(np.array(centre) - np.array(offset))))
            if len(spec) > 5:                       # optional (w, x, y, z) rotation in the part frame
                cyl.AddOrientOp().Set(Gf.Quatf(*spec[5]))
            collide(cyl.GetPrim(), color)

    main = f"{prim_path}/body"
    body(main, (0.0, 0.0, 0.0), spec.mass)
    boxes(main, spec.boxes)
    drawn(main, spec.meta.get("visual_boxes", ()))
    cylinders(main, spec.cylinders)
    for i, (centre, radius, color) in enumerate(spec.spheres):
        sph = UsdGeom.Sphere.Define(stage, f"{main}/sph_{i}")
        sph.CreateRadiusAttr(radius)
        sph.AddTranslateOp().Set(Gf.Vec3d(*centre))
        collide(sph.GetPrim(), color)
    if spec.fixed:
        fj = UsdPhysics.FixedJoint.Define(stage, f"{main}/world_fix")
        fj.CreateBody1Rel().SetTargets([Sdf.Path(main)])
        # anchor at the part's placed pose (world frame), not at the world origin
        fj.CreateLocalPos0Attr(Gf.Vec3f(*(translation or (0.0, 0.0, 0.0))))
        fj.CreateLocalRot0Attr(Gf.Quatf(w, x, y, z))
        fj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        fj.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    for s in spec.sliders:
        path = f"{prim_path}/{s.name}"
        body(path, s.rest, s.mass)
        boxes(path, s.boxes, offset=s.rest)
        cylinders(path, s.cylinders, offset=s.rest)
        j = UsdPhysics.PrismaticJoint.Define(stage, f"{path}/{s.name}_slide")
        j.CreateAxisAttr(s.axis)
        if s.parent:
            ph = next(x for x in spec.hinges if x.name == s.parent)
            j.CreateBody0Rel().SetTargets([Sdf.Path(f"{prim_path}/{s.parent}")])
            j.CreateLocalPos0Attr(Gf.Vec3f(*(np.array(s.rest) - np.array(ph.pivot))))
        else:
            j.CreateBody0Rel().SetTargets([Sdf.Path(main)])
            j.CreateLocalPos0Attr(Gf.Vec3f(*s.rest))
        j.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        qr = (math.cos(s.roll / 2), math.sin(s.roll / 2), 0.0, 0.0)    # both joint frames carry the same turn
        j.CreateLocalRot0Attr(Gf.Quatf(*qr))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(*qr))
        lo = s.lower if s.lower is not None else (-s.travel if s.sign < 0 else 0.0)
        hi = s.upper if s.upper is not None else (0.0 if s.sign < 0 else s.travel)
        j.CreateLowerLimitAttr(lo)
        j.CreateUpperLimitAttr(hi)
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(s.rate)
        drive.CreateDampingAttr(s.damping)
        drive.CreateTargetPositionAttr(0.0)
        if s.effort > 0:
            drive.CreateMaxForceAttr(s.effort)
    for h in spec.hinges:
        path = f"{prim_path}/{h.name}"
        hprim = body(path, h.pivot, h.mass)
        if not (h.boxes or h.cylinders):    # a body without shapes still needs an inertia
            UsdPhysics.MassAPI.Apply(hprim).CreateDiagonalInertiaAttr(Gf.Vec3f(*(1e-4,) * 3))
        boxes(path, h.boxes, offset=h.pivot, grip_ids=spec.meta.get(f"grip_boxes_{h.name}", ()))   # e.g. a door's faces
        cylinders(path, h.cylinders, offset=h.pivot)
        for i, (centre, radius, color) in enumerate(h.spheres):
            sph = UsdGeom.Sphere.Define(stage, f"{path}/sph_{i}")
            sph.CreateRadiusAttr(radius)
            sph.AddTranslateOp().Set(Gf.Vec3d(*(np.array(centre) - np.array(h.pivot))))
            collide(sph.GetPrim(), color)
        j = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/{h.name}_hinge")
        j.CreateAxisAttr(h.axis)
        if h.parent:
            pj = next((x for x in spec.hinges if x.name == h.parent), None)
            ref = pj.pivot if pj is not None else next(x for x in spec.sliders if x.name == h.parent).rest
            j.CreateBody0Rel().SetTargets([Sdf.Path(f"{prim_path}/{h.parent}")])
            j.CreateLocalPos0Attr(Gf.Vec3f(*(np.array(h.pivot) - np.array(ref))))
        else:
            j.CreateBody0Rel().SetTargets([Sdf.Path(main)])
            j.CreateLocalPos0Attr(Gf.Vec3f(*h.pivot))
        j.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        qy = (math.cos(h.yaw / 2), 0.0, 0.0, math.sin(h.yaw / 2))   # both joint frames carry the same turn
        j.CreateLocalRot0Attr(Gf.Quatf(*qy))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(*qy))
        if math.isfinite(h.lower) and math.isfinite(h.upper):   # an infinite range leaves the joint free to spin
            j.CreateLowerLimitAttr(math.degrees(h.lower))   # USD revolute limits are in degrees
            j.CreateUpperLimitAttr(math.degrees(h.upper))
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(h.stiffness * math.pi / 180)  # per degree in USD; the actuator sets SI values
        drive.CreateDampingAttr(h.damping * math.pi / 180)
        drive.CreateTargetPositionAttr(math.degrees(h.target))
        if h.effort > 0:
            drive.CreateMaxForceAttr(h.effort)
    if spec.sliders or spec.hinges or spec.fixed:   # a fixed part is a fixed-base articulation, joints or not
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(main))
        art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(main))
        art.CreateEnabledSelfCollisionsAttr(bool(spec.meta.get("self_collide", False)))   # joined links never collide
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
            limit_kw = next((n for n in ("effort_limit_sim", "joint_effort_limit", "effort_limit")
                             if n in getattr(ImplicitActuatorCfg, "__dataclass_fields__", {})), None)
            actuators = {s.name: ImplicitActuatorCfg(joint_names_expr=[f"{s.name}_slide"], stiffness=s.rate,
                                                     damping=s.damping,
                                                     **({limit_kw: s.effort} if (s.effort > 0 and limit_kw) else {}))
                         for s in spec.sliders}
            for h in spec.hinges:
                kw = {limit_kw: h.effort} if (h.effort > 0 and limit_kw) else {}
                actuators[h.name] = ImplicitActuatorCfg(joint_names_expr=[f"{h.name}_hinge"], stiffness=h.stiffness,
                                                        damping=h.damping, **kw)
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators)
            cfg = self._add_initial_pose_to_cfg(cfg)
            if spec.meta.get("init_joint_pos"):          # joints that start away from 0 (e.g. a lid that starts open)
                cfg.init_state.joint_pos = dict(spec.meta["init_joint_pos"])
            if not (spec.sliders or spec.hinges):   # a fixed part without joints: no joint defaults to match
                cfg.init_state.joint_pos = {}
                cfg.init_state.joint_vel = {}
            return cfg

    kind = ObjectType.ARTICULATION if spec.sliders or spec.hinges or spec.fixed else ObjectType.RIGID
    return PartObject(
        name=spec.name,
        object_type=kind,
        spawner_cfg=PartSpawnerCfg(part_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(spec.pos),
                          rotation_xyzw=tuple(spec.meta.get("quat_xyzw", (0.0, 0.0, math.sin(spec.yaw / 2),
                                                                          math.cos(spec.yaw / 2))))),
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
    """Move a hinge's drive target (rad) in env 0; the drive pulls the hinge there from the next physics step."""
    import torch

    asset = environment.unwrapped.scene[part_name]
    j = list(asset.joint_names).index(f"{hinge_name}_hinge")
    value = torch.full((1, 1), float(angle), device=_t(asset.data.joint_pos).device)
    command = getattr(getattr(asset, "actuators", None), "target_command", None)
    if command is not None and hasattr(command, "set_position_index"):
        command.set_position_index(value=value, joint_ids=[j])
    else:  # older Isaac Lab
        asset.set_joint_position_target(value, joint_ids=[j])


def set_slider_target(environment, part_name: str, slider_name: str, position: float) -> None:
    """Move a slider's drive target (m along its axis, 0 at rest) in env 0."""
    import torch

    asset = environment.unwrapped.scene[part_name]
    j = list(asset.joint_names).index(f"{slider_name}_slide")
    value = torch.full((1, 1), float(position), device=_t(asset.data.joint_pos).device)
    command = getattr(getattr(asset, "actuators", None), "target_command", None)
    if command is not None and hasattr(command, "set_position_index"):
        command.set_position_index(value=value, joint_ids=[j])
    else:  # older Isaac Lab
        asset.set_joint_position_target(value, joint_ids=[j])


def set_joint_stiffness(environment, part_name: str, joint_name: str, stiffness: float) -> None:
    """Change one joint drive's stiffness (N/m or N m/rad) in env 0 at run time. ``joint_name`` is the full joint
    name ("<slider>_slide" or "<hinge>_hinge")."""
    import torch

    asset = environment.unwrapped.scene[part_name]
    j = list(asset.joint_names).index(joint_name)
    value = torch.full((1, 1), float(stiffness), device=_t(asset.data.joint_pos).device)
    write = getattr(asset, "write_joint_stiffness_to_sim_index", None)
    if write is not None:
        write(stiffness=value, joint_ids=[j], env_ids=[0])
    else:  # older Isaac Lab
        asset.write_joint_stiffness_to_sim(value, joint_ids=[j], env_ids=[0])


_RAISE = {"h": 0.0}


def raise_arm_bases(height: float) -> None:
    """Stand both arm bases ``height`` metres above their station mounts (on pedestals the task file lists in
    OBJECTS). Call it in the task file. The arms, their gains, the action interface and the cameras stay standard."""
    _RAISE["h"] = float(height)


def _raise_bases(scene) -> None:
    """Wrap ``scene.build_task_environment`` so the two arm articulations spawn ``_RAISE['h']`` higher."""
    if getattr(scene, "_partkit_raise", False):
        return
    orig_build = scene.build_task_environment

    def build_task_environment(embodiment, task, control_hz=None):
        h = _RAISE["h"]
        if h:
            from yam_arena import station

            def lift(scene_cfg):
                for key, side in (("left_robot", "left"), ("right_robot", "right")):
                    arm = getattr(scene_cfg, key, None)
                    if arm is not None:
                        x, y, z = (float(v) for v in station.BASES[side])
                        arm.init_state.pos = (x, y, z + h)

            lift(embodiment.scene_config)
        env = orig_build(embodiment, task, control_hz)
        if h and getattr(env, "env_cfg_callback", None) is not None:
            inner = env.env_cfg_callback

            def configure(env_cfg):
                env_cfg = inner(env_cfg)
                lift(env_cfg.scene)
                return env_cfg

            env.env_cfg_callback = configure
        return env

    scene.build_task_environment = build_task_environment
    scene._partkit_raise = True


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


def finger_points(environment) -> list:
    """Points on both arms' finger pads (world, relative to the env origin): at the TCP and at the pad tips, 6 mm
    outside each pad face. For "released" checks in an ``on_step`` hook; score() does not see the robot."""
    from yam_arena.constants import PAD_TIP_PAST_TCP, TCP_OFFSET

    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    pts = []
    for arm in ("left_robot", "right_robot"):
        robot = base.scene[arm]
        pose = _np(robot.data.body_link_pose_w)[0][list(robot.body_names).index("gripper")]
        jp, jn = _np(robot.data.joint_pos)[0], list(robot.joint_names)
        half_gap = max(abs(jp[jn.index(j)]) for j in ("joint7", "joint8") if j in jn)
        r = PartState(pos=pose[:3], quat_xyzw=pose[3:7], lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.zeros(3),
                      high=np.zeros(3)).rotation()
        for along in (0.0, PAD_TIP_PAST_TCP):
            centre = pose[:3] + r @ (np.array(TCP_OFFSET) + [0.0, 0.0, along]) - origin
            pts += [centre + r[:, 0] * side * (half_gap + 0.006) for side in (1.0, -1.0)]
    return pts


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
    _raise_bases(scene)


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
