# Fit a split wall plate round a plumbed-in copper stub-out and push it flat to the wall

A section of plywood wall stands on the bench with a 5/8 in copper supply stub-out coming through a rough hole, its
quarter-turn angle stop valve already fitted, as a plumber finds it at trim-out.
In front of it, a one-piece split floor and wall plate (Sioux Chief 926-2 SnapOne, chrome-plated polypropylene) stands
in the groove of a block.
A solid escutcheon can no longer go on once the valve is fitted; this one is split on one side and springs shut.
The robot must get the plate round the pipe between the valve and the wall, shut, and flat against the wall over the
hole.

Fitting a flange over a stub-out after the stop valve is on is routine trim-out work for plumbers (O*NET 47-2152.00)
and general maintenance and repair workers (49-9071.00).
The task tests a bimanual spread: the ring only opens far enough to pass the pipe when both of its halves are held and
turned apart about the side opposite the split, and the hands must keep it open against its spring while they carry it
over the valve and lower it onto the pipe.
Then it tests a push along a pipe: the plate must slide 52 mm along the pipe without cocking on it.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes, with spring-loaded sliders and hinges, for yam_arena task files |
| `demo.py` | The scripted full-score demo (open loop from the fixed layout) |
| `clearance.py` | Arm-to-arm clearance from the robot's collision meshes, for the demo's log |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs
with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the chrome plate round the copper pipe and push it flat against the wall so it covers the hole." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/split-escutcheon-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                                   # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=no_push python demo.py task.py out --no-record       # shut round the pipe, left where it was put
DEMO_VARIANT=one_arm python demo.py task.py out --no-record       # one hand takes the plate by one half
DEMO_VARIANT=no_spread python demo.py task.py out --no-record     # both hands, but the plate is not spread
```

The demo is open loop: it uses only the fixed layout in `task.py`, timing and the arms' own joint readings (a trim
that keeps each measured tool point on the plan under load).
The object poses it reads go to the log and `result.json` only.

1. Both hands come in from the sides, level, and pinch the plate standing in its block across its 11.4 mm depth, each
   on the middle of one half.
2. They lift it out of the groove and spread it open at the split: each half turns 0.28 rad about the top of the ring,
   the hands turning with it.
3. They carry it over the valve to its place 52 mm off the wall and lower it so the open split passes over the pipe.
4. They turn the halves back, so it springs shut round the pipe, let go and draw out sideways.
5. The right hand, pointing down, tipped 30 degrees toward the wall, its fingers open either side of the pipe, slides
   the plate along the pipe until it is flat against the wall, and moves away.

The hands pinch the plate 52 mm off the wall, not at it, because the gripper's body spans 10 cm along the fingers'
closing axis; the stub-out is 3-1/2 in long for that reason.

Measured in Isaac Sim, in the batch that also ran Dream-1:

- The demo reaches 1.0 (success) in 520 steps (26 s), with IK within 1.8 mm of every target.
  As the hands let go and draw out sideways, the ring springs briefly open and shuts again on the pipe by itself; the
  push then starts from the shut plate.
- `no_push` scores 0.5: shut round the pipe, 45 mm off the wall.
- `one_arm` scores 0: held by one half, the shut split lands on the pipe and the plate falls to the bench.
- `no_spread` scores 0: not spread, the plate is lowered onto the pipe and ends beside it, 23 mm off its axis.
- Dream-1 scored 0.0 in its one run (it ran to the 1,200-step limit).

## Scoring

Evaluated every control step; the score at the end of the episode counts.

- 1.0 (success): the plate is round the pipe, shut, flat against the wall, both grippers are at least 3 cm from it, and
  it has stayed within 1 mm for the last second.
- 0.75: shut and flat against the wall, but a gripper is still within 3 cm or the plate is still moving.
- 0.5: round the pipe and shut (the split closed to within 1 mm at the bore), but not flat against the wall.
- 0.25: round the pipe between the valve and the wall (its bore's centre within 2 mm of the pipe's axis), not shut.
- Otherwise 0.
- 0 for the rest of the trial if the plate was ever dropped off the table.

Flat against the wall means the middle of the plate's back edge within 1.5 mm of the wall and the plate square to the
pipe within 3 degrees; a shut plate on the pipe then covers the 30 mm rough hole.
In a real trial: the plate is shut if its split is closed to a fingernail's width, and flat if no gap shows at the wall
in a photo taken along the pipe.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S, plus about 1 minute for Isaac Sim to start (6 minutes on
a fresh instance).
The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

What is not modelled, most important first:

- **The ring's flex.** The real plate is one polypropylene ring that bends open at its split; here it is two rigid
  halves on a hinge at the top, opposite the split, with a spring of 0.2 N m/rad and a stop at 0.6 rad.
  That is about 1.6 N at the split to spread it 30 degrees, an estimate: Sioux Chief publishes no force.
  Measure the bought plate with a spring scale and report it.
- **The bore's grip.** The real bore is 0.625 in, the pipe's own size, with small flexible tabs that grip the pipe.
  Here the bore is a rigid 16-sided sleeve with 0.35 mm play, so the plate slides along the pipe on its own weight's
  friction only.
- **Fixed wall, pipe and valve.** The stub-out does not bend; nothing scores a bent pipe.
- **Shapes.** Plate and pipe are 16-sided prisms with their corners on the real radii; the plate's face is flat with a
  1.5 mm skirt rather than domed. Wall thicknesses, sleeve length and the 10 g mass are estimates.
- Contact offset on the plate 0.2 mm, rest offset 0. No force cap is enforced.

## Real-world build

- Plate: Sioux Chief 926-2 SnapOne floor and ceiling plate for 1/2 in CTS, chrome-plated polypropylene (spec sheet:
  ID 0.625 in, OD 2.53 in, height 0.45 in).
- Wall: an 18 mm plywood panel, 300 mm wide and 250 mm tall, on an 18 mm plywood foot 90 mm deep behind it, screwed to
  the bench, its front face 36 cm in front of the midpoint between the arm bases.
  A 30 mm square rough hole, its centre 120 mm above the bench.
- Pipe: 1/2 in nominal (5/8 in OD) type L copper, held behind the wall in a stub-out bracket, standing 85 mm out of
  the wall's face, square to it.
- Valve: a 1/2 in nominal compression by 3/8 in OD compression quarter-turn angle stop (for example BrassCraft
  G2CR19X), outlet up, handle toward the robot.
- Block: a 2x4 offcut on end, 89 x 38 x 113 mm, with a 13 mm groove 5 mm deep across its top along the robot's
  left-right, at 17 cm in front of the midpoint between the arm bases.
  The plate stands in the groove, split down, face toward the robot.
- Measure before a real trial, with a spring scale: the force at the split that spreads it 25 mm (the model uses about
  1.6 N), and the push that slides it along the pipe.
- Reset: spread the plate at its split, take it off the pipe and stand it in the block's groove.
