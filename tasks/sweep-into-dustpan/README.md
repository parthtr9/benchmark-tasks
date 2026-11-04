# Sweep cubes into a hand-held dustpan with a brush

A light blue dustpan lies on the table with its open front facing left and its handle sticking out to the right. A
hand brush lies to its left, and five small orange wooden cubes are scattered on the table between the brush and the
pan. The robot must hold the pan still by its handle with one arm and, with the other arm, use the brush to sweep all
five cubes up the pan's thin front lip onto its floor, without touching the cubes with either gripper.

Sweeping debris into a dustpan is one of the most common two-handed tool tasks in homes, workshops and commercial
cleaning. It tests two things that pick-and-place does not: using a held tool to move several loose objects at once
(non-prehensile transport through a tool), and the second arm holding a receiving object still against the push
(the pan weighs 80 g; unheld, the brush shoves it away). Grippers may not touch the cubes, so carrying or flicking
them in by hand does not count.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()`, `score()` and `distance()` |
| `partkit.py` | Rigid parts built from boxes (optionally rotated), for yam_arena task files |
| `demo.py` | The scripted full-score demo and its negative variants |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer. `task.py` uses its step hook to watch where
the grippers' finger pads are relative to the cubes.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hold the blue dustpan still by its handle with one arm, and with the other arm use the hand brush to sweep all five orange cubes into the dustpan. Do not touch the cubes with the grippers." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/dustpan-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200        # full score: demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=no_hold python demo.py task.py nohold      # nobody holds the pan: the brush pushes it away (0)
DEMO_VARIANT=gripper python demo.py task.py gripper     # rakes with the bare open fingers (cubes touched, 0)
DEMO_VARIANT=short   python demo.py task.py short       # stops at the lip: cubes not on the pan (0)
```

1. The right arm grips the pan's handle, the left arm the brush's handle bar.
2. The left arm lifts the brush 2 mm off the table and sweeps it toward the pan, rising along the lip's slope, until
   the brush's front is 6.8 cm past the lip; the cubes ride up the lip ahead of it.
3. The brush goes straight up and back; the right arm keeps holding the pan.

The demo reaches 1.0 (success) in 299 steps (15 s). The three variants score 0.0: unheld, the pan is pushed 79 mm
away; the bare-finger rake touches the cubes it moves; stopped at the lip, the cubes stay on the table.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 0 if the pan's lip was at any point moved more than 3 cm across the table from where it started.
- A cube counts if it lies on the pan: its centre at least 2 cm past the front edge of the lip, between the side walls,
  in front of the back wall, and resting on the pan's floor (not on the table), and no gripper finger pad came within
  22 mm of its centre during the trial (in a real trial: judged from the video, a cube a finger touches is out).
- 0.2 per counting cube. With all five counting, 1.0 (success, which must hold for 1 s) needs the pan resting flat on
  the table (within 5 mm of it, tilted under 10 degrees) and everything still (moving less than 1 mm over 1 s);
  otherwise (pan lifted or tipped) 0.8.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S, plus about 1 minute for Isaac Sim to start (6 minutes on
a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The pan, brush and cubes are rigid bodies built from boxes; the brush's bristles are a rigid block, so the brush
pushes like a squeegee rather than flexing. The lip is a 1.5 mm plate sloped at 7 degrees whose front edge sits
0.3 mm above the table. Light angular damping is added to the parts. Finger contact with the cubes is judged by
distance from the finger pads (22 mm from a cube's centre), not by contact forces.

## Real-world build

Dustpan: 3D-printed PLA or bent sheet, 16 x 14.3 cm, 3 mm side walls 3 cm tall, a 4 cm back wall, a floor 5.7 mm
above the table and a 1.5 mm, 3.2 cm long lip sloping down to the table at 7 degrees, an 8 x 2 x 1.6 cm handle out of
the back wall at 2.4-4 cm height; about 80 g. Brush: a 14 x 3.5 x 2.2 cm bristle block on a 0.8 cm wooden back with a
10 x 2 x 2.2 cm handle bar on top, about 120 g (stiff short bristles). Cubes: five 15 mm wooden cubes, about 5 g each.
