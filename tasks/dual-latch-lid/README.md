# Release a two-latch lid and lift it out with both hands

A heavy grey case lies on the table with its blue lid set 13 mm down inside the rim. A slide latch at each end of the
lid has an orange tab standing up from it; its tongue reaches into a slot in the case's end wall. The robot must slide
both tabs toward the lid's middle at the same time, which pulls both tongues out of the walls, lift the lid out with
both arms, and set it down flat on the table beside the case.

Equipment covers, battery bays and storage cases often use a latch at each end that has to be held open while the lid
comes out. Two arms are needed by construction: the tabs are 20 cm apart, too far for one gripper, and a lid released
at one end stays trapped by the other latch. It sits too deep in the case to pivot or slide out (in the scripted check
below, one arm releasing one latch and lifting scores 0.25). The task tests coordinated two-handed manipulation of one
object: both arms must act on it at the same time and then move it together.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes and cylinders, with spring-loaded sliders, for yam_arena task files |
| `demo.py` | The scripted full-score demo |

`partkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Grip the two orange tabs on the blue lid, slide both toward the middle of the lid at the same time to release the latches, lift the lid out of the grey case, and set it down flat on the table to the right of the case." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/lid-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200   # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_ONE_ARM=1 python demo.py task.py onearm       # only the right arm acts: one latch releases, the lid stays (0.25)
```

1. The left arm grips the far tab and the right arm the near one, each holding the tab's upper part so its
   fingertips stay clear of the lid.
2. Both slide their tabs toward the lid's middle together, pulling both tongues out of the end walls.
3. Both lift 6 cm, still holding the tabs in, carry the lid 10 cm to the right, lower it onto the table and let go.

The demo reaches 1.0 (success) in about 310 steps (15 s).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 0 if the case was at any point moved more than 3 cm.
- 1.0 (success, which must hold for 1 s): after being lifted out, the lid lies flat on the table (underside within 5 mm
  of it, tilted under 10 degrees), its centre at least 8 cm to the side of the case's centreline, and still (moving less
  than 1 mm over the last second).
- 0.75: the lid was lifted with its underside 1 cm above the case's rim.
- 0.5: both latches were released at the same time (each tab slid at least 4 mm in, which takes its tongue out of the
  wall).
- 0.25: one latch was released.
- Otherwise 0.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S, plus about 1 minute for Isaac Sim to start (6 minutes on
a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes. Each latch is a prismatic joint with a spring drive and no preload, and
its tongue holds the lid in by contact alone. The case is heavy so it stays put. Light angular damping is added to
the parts. No contact-force cap is enforced in simulation.

## Real-world build

The case, lid, tabs and tongues are 3D-printed PLA with the dimensions in `task.py`; each latch needs a small
compression spring of about 0.3 N/mm with at least 6 mm of travel, and the case can be weighted (or screwed to a
board) to about 1.5 kg.
