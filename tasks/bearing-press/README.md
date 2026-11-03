# Press a bearing onto a robot link's shaft

A light grey printed robot link lies loose on the table with a steel shaft sticking out of its side: a standoff collar
ending in a shoulder, then plain shaft with a spring ball on its side. A square flanged bearing stands on edge in a
holder beside it. The robot must line the bearing's bore up with the shaft, slide it on, and press it along until it
clicks past the ball and sits against the shoulder. The ball then springs out behind the bearing, retaining it.

Fitting bearings onto shafts is a basic step in building robot joints, gearboxes and actuators. The task tests lining
a bore up with a shaft held out sideways from a light part, keeping the part square while pressing it along, and
pushing it past a retaining detent. The demo holds the link with the left arm; the same press with the left arm idle
also succeeded, so a careful single arm can do it, and a rough push that drags the link more than 3 cm scores 0.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes and cylinders, with spring-loaded sliders, for yam_arena task files |
| `demo.py` | The scripted full-score demo (closed loop on the measured bearing pose) |

`partkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Slide the bearing onto the end of the steel shaft that sticks out of the grey robot link and press it along the shaft until it clicks past the ball and sits against the shoulder." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/bearing-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200   # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_NO_BRACE=1 python demo.py task.py nobrace     # the same press with the left arm idle (also reaches 1.0)
```

1. The left arm grips the link's far end across its width.
2. The right arm grips the bearing's sides at its top and lifts it out of the holder.
3. It lines the bore up with the shaft 15 mm beyond the shaft's end and, closed loop on the measured bearing pose,
   presses it along at up to 10 mm/s, correcting its offset and yaw, past the ball, until it seats on the shoulder.
4. Both arms let go and lift away.

The demo reaches 1.0 (success) in 521 steps (26 s).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 0 if the link was at any point moved more than 3 cm or turned more than 15 degrees.
- 1.0 (success, which must hold for 1 s): the bearing is on the shaft (bore within 1 mm of the shaft's axis, square to
  it within 10 degrees), its inner face within 1 mm of the shoulder with the ball sprung back out behind it, and still
  (moving less than 1 mm over the last second).
- 0.75: on the shaft, it was pushed past the shaft's middle.
- 0.5: it was put on the shaft with at least 3 mm of shaft in its bore.
- 0.25: it was lifted 3 cm.
- Otherwise 0.

The 10-degree squareness allowance matches the bore's 0.5 mm of radial play, within which a released bearing can
rest up to about 8 degrees off square.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S, plus about 1 minute for Isaac Sim to start (6 minutes on
a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes and cylinders. The bearing is a solid square block with a square-section
bore (no rolling elements), and its 0.5 mm radial play on the shaft is looser than a real slip fit, so it doesn't bind
when pushed from its top edge. The ball is a prismatic joint with a spring drive and no preload. No contact-force cap is
enforced in simulation.

## Real-world build

The link, collar and holder are 3D-printed PLA with the dimensions in `task.py`; the shaft is an 8 mm steel rod; the
bearing is a square flanged bearing (or a printed 22 mm square block with a 9 mm bore); the detent is an off-the-shelf
spring ball plunger set into the side of the shaft, 1 mm proud.
