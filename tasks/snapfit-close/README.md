# Close a snap-fit case lid

A light grey case stands on a narrow foot, with an open pocket in its top and two spring-loaded orange latch plates
near its front end. Its blue lid lies on two rails beside it. The robot must lift the lid, set it into the pocket with
its orange tabs over the latches, and press the lid's front edge down until both latches click.

Closing a snap-fit cover finishes many electronics repairs and assemblies. The task tests placing a flat part into a
locating pocket and then applying sustained downward force at an off-centre point while the assembly is stabilized.
The latches sit 4.5 cm in front of the case centre, well past the edge of its 4 x 5 cm foot, so a press on the front
edge tips an unsteadied case: one arm must hold the case while the other presses.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, latch logic, `success()` and `score()` |
| `partkit.py` | Rigid parts built from several boxes, with spring-loaded sliders, for yam_arena task files |
| `demo.py` | The scripted full-score demo (open loop from the fixed layout and one look at the case's pose; never the latch state) |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs
with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the blue lid into the pocket of the grey case with its orange tabs over the orange latches, then press the front edge of the lid down until both latches click." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/snapfit-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                  # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_NO_BRACE=1 python demo.py task.py nobrace    # the same press with the left arm idle: the case tips, score 0
DEMO_VARIANT=no_press python demo.py task.py out  # lid placed, never pressed: score 0.5
```

1. The right arm grips the lid across its width, lifts it, turns it to the case and lowers it onto the raised latch
   plates, then lets go and parks.
2. The left arm grips the case's long sides 3.5 cm behind its centre, its wrist leaning away from the right arm.
3. The right arm, fingers closed, presses the middle of the lid's front edge slowly (6 mm/s), commanded 20 mm past
   where a seated lid is flush with the case rim (the arm's position control gives under the latch springs, so the
   pad ends about 1 mm below flush), holds, then both arms let go and lift clear.

The demo uses only what a policy could see: the fixed layout in the first camera frame (the task's constants,
with `object_jitter=0`), one look at the case's pose once the left arm's grip has settled (the case is a fixed part in
plain view of the cameras, and gripping it nudges it slightly) and its own arm commands. Everything after that look is
open loop; it never reads the task's internal state, slider positions or the latch state.

The demo reaches 1.0 in about 670 steps (34 s).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 0 if the case base was at any point moved more than 3 cm from its start or tilted more than 20 degrees.
- 1.0 (success, which must hold for 1 s): both latches clicked, the lid lies in the pocket (centre within 3 mm of the
  pocket centre, turned less than 5 degrees, tilted less than 10 degrees), let go (every finger pad of both arms at
  least 3 cm from the lid) and still (within 1 mm), both over the whole last second. With both latches clicked but the
  lid still pressed or moving at the end, 0.75.
- 0.75: one latch clicked after the lid was placed.
- 0.5: after being lifted, the lid was set down at rest in the pocket.
- 0.25: the lid was lifted 3 cm above its start.
- Otherwise 0.

A latch clicks when the lid's underside above it comes within 0.7 mm of the pocket floor. In a real trial, check with
a ruler or a side photo that the lid's front corners sit flush with the case's top edge (a real snap hook holds them
there).

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S, plus about 1 minute for Isaac Sim to start (6 minutes
on a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes. The latch plates are spring-loaded sliders, but the snap hook is not
modelled: a latch clicks when the lid is pressed within 0.7 mm of the floor above it, and once both have clicked the
lid is held in its seated pose relative to the case, rewritten every control step. No contact-force cap is enforced
in simulation.

## Real-world build

All parts are 3D-printed PLA boxes with the dimensions in `task.json`; the latch plates need two compression springs
of about 2.5 N/mm with at least 4 mm of travel, and a real snap hook over each plate (not specified here).
