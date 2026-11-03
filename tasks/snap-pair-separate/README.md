# Pull apart a snap-together case with both hands

A 30 cm printed case lies on the table, made of two halves snapped together: a tongue on the near (blue) half sits in
a slot in the far (grey) half, held there by a spring plunger resting in a notch in the tongue. The robot must hold one
half in each gripper and pull them straight apart until the tongue comes out, leaving both halves on the table at least
3 cm apart.

Opening snap-together housings (battery packs, remotes, chargers, tool cases) is the first step of many repairs. Two
arms are needed by construction: the halves are light and loose, and the plunger holds them together far harder than
the table's friction holds either one, so pulling one half only drags the other along (in the scripted check below,
one arm pulling scores 0). Separating them takes opposing pulls along one line, with both halves kept level so the
tongue doesn't bind in its slot.

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
  --instruction "Hold the grey half and the blue half of the case and pull them straight apart until they come undone, leaving both halves on the table at least 3 cm apart." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/case-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200   # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_ONE_ARM=1 python demo.py task.py onearm       # only the right arm pulls: the other half slides along (0)
```

1. The left arm grips the far (grey) half and the right arm the near (blue) half, 10 cm from the seam, across their
   6 cm width.
2. Keeping both halves on the table, they pull them 4.5 cm apart each, past the plunger and off the tongue (lifted,
   the halves tilt a few degrees in the grips and the tongue binds).
3. Both let go and lift away.

The demo reaches 1.0 (success) in about 220 steps (11 s).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 1.0 (success, which must hold for 1 s): the tongue is out of the slot and both halves lie flat on the table (within
  5 mm of it, tilted under 10 degrees) with at least 3 cm between the tongue's tip and the grey half's seam face, still
  (moving less than 1 mm over the last second).
- 0.75: the halves were pulled apart until the tongue was out of the slot plus 5 mm.
- 0.5: they were pulled 6 mm apart, past the plunger.
- 0.25: they were pulled 2 mm apart.
- Otherwise 0.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S, plus about 1 minute for Isaac Sim to start (6 minutes on
a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes and cylinders. The plunger is a prismatic joint with a spring drive and no
preload, and the halves are held together by contact alone. Friction is uniform and light angular damping is added to
the parts. No contact-force cap is enforced in simulation.

## Real-world build

Both halves are 3D-printed PLA with the dimensions in `task.py`; the detent is an off-the-shelf spring plunger with a
rounded nose of about 6 mm (about 1 N/mm) set into the slot's wall so it stands 2 mm into the slot.
