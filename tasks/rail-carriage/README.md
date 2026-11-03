# Slide a carriage onto a linear rail until it clicks

A grey printed linear rail lies loose on the table, pointing away from the arms: a bar with a T-shaped top, an orange
end stop part way along and a plain handle beyond it. A blue carriage block with a matching T-shaped channel stands
beside it. The robot must line the carriage up with the rail's open end, slide it on, and push it along until it clicks
past a spring plunger on the side of the rail and stops against the end stop. The plunger then springs back out behind
the carriage, trapping it.

Linear guides carry the moving axes of robot gantries, linear actuators and 3D printers, and fitting a carriage onto
its rail is a routine step in building them. The task tests lining a part up with a long, close-fitting profile (1 mm
of side play), keeping it square while sliding it 10 cm, and pushing through a detent. The rail weighs 150 g and is
loose. The demo holds its handle with the left arm; the same slide with the left arm idle also succeeded, so a careful
single arm can do it, and a rough push that drags the rail more than 3 cm scores 0.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes and cylinders, with spring-loaded sliders, for yam_arena task files |
| `demo.py` | The scripted full-score demo (closed loop on the measured carriage pose) |

`partkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Slide the blue carriage onto the open end of the grey rail and push it along the rail until it clicks past the plunger and stops against the orange end stop." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/rail-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200   # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_NO_BRACE=1 python demo.py task.py nobrace     # the same slide with the left arm idle (also reaches 1.0)
```

1. The left arm grips the rail's handle beyond the end stop.
2. The right arm grips the carriage's side walls (its pad tips stay clear of the table) and lifts it.
3. It lines the carriage up with the rail 1.5 cm short of the open end and, closed loop on the measured carriage pose,
   slides it on and along at up to 15 mm/s, correcting its height, sideways offset and yaw, until its rear edge has
   cleared the plunger; the end stop limits how far it goes.
4. Both arms let go and lift away.

For every grasp the demo picks whichever of the two equivalent parallel-grip orientations (180 degrees apart) the
wrist reaches exactly; at its joint limit the wrist otherwise leaves the fingers several degrees off and they twist
the light parts as they close.

The demo reaches 1.0 (success) in 858 steps (43 s).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 0 if the rail was at any point moved more than 3 cm or turned more than 15 degrees.
- 1.0 (success, which must hold for 1 s): the carriage is on the rail (within 2 mm of its centreline and 1.5 mm of its
  seated height, square within 5 degrees), between the plunger and the end stop with the plunger sprung back out, and
  still (moving less than 1 mm over the last second).
- 0.75: on the rail, its centre was pushed at least 6 cm from the rail's open end.
- 0.5: it was put on the rail with at least 1 cm of it over the rail.
- 0.25: it was lifted 3 cm.
- Otherwise 0.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S, plus about 1 minute for Isaac Sim to start (6 minutes on
a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes and cylinders. The plunger is a prismatic joint with a spring drive and no
preload; real ball plungers are stiffer and preloaded. Friction is uniform (0.5 on the rail, 0.3 on the carriage) and
light angular damping is added to the parts. No contact-force cap is enforced in simulation.

## Real-world build

The rail, end stop and carriage are 3D-printed PLA with the dimensions in `task.py`; the detent is an off-the-shelf
spring plunger with a rounded nose of about 8 mm, set into the side of the rail so it stands 2 mm proud.
