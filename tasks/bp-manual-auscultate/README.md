# Take a manual blood pressure on a training arm

A fixed training arm lies along the table with a red mark on its top, a blue cuff and a dial on its upper part. Beside it
is a pump unit (a rubber bulb with a squeeze plate, and a bleed valve with a large black T-bar knob), and a stethoscope
chestpiece stands in a dish. The robot pumps the cuff above 160 mmHg, lets it down past 100 mmHg at 2 to 4 mmHg per second
while the chestpiece stays pressed on the red mark, empties the cuff and lets go of everything.

Medical assistants and nurses (O*NET 31-9092.00, 29-1141.00) do this with two hands. The valve bleeds at a rate set by the angle of its knob (it stays where it is set), the sounds are only heard while the chestpiece is pressed on the artery, and the
chestpiece is a plain pressed part (not spring-return), and the pump unit and the chestpiece are on opposite sides, each out of the other arm's reach, so one arm cannot do both.
It is a different skill from the board's hand-pump pressure task (one hand to a dial value) and from bag-mask ventilation.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, cuff pressure model (`on_step`), `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts built from boxes and cylinders with spring sliders and hinges |
| `demo.py` | The scripted full-score demo (open loop) and its negative variants |

## Run a trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the training arm's blood pressure with the stethoscope and cuff: pump the cuff above 160 mmHg, let it down past 100 mmHg at 2 to 4 mmHg per second while listening over the red mark the whole way, let the cuff empty, then let go of everything." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/bp-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                    # full score
DEMO_VARIANT=one_arm python demo.py task.py v --no-record           # right arm alone: 0
DEMO_VARIANT=rest_only python demo.py task.py v --no-record         # chestpiece set down and let go: 0
DEMO_VARIANT=dump python demo.py task.py v --no-record              # valve opened wide: 0
DEMO_VARIANT=slow python demo.py task.py v --no-record              # about 1.7 mmHg/s: 0.25
DEMO_VARIANT=no_empty python demo.py task.py v --no-record          # cuff not emptied: 0.5
DEMO_VARIANT=overpump python demo.py task.py v --no-record          # seven squeezes: 0
DEMO_DEBUG=1 ...                                                    # trace every 10 steps (also saved in result.json)
```

1. Left arm takes the chestpiece by its stem, sets it on the mark and presses it in about 5 mm (2 N), and holds.
2. Right arm, fingers closed, presses the squeeze plate four times (168 mmHg), takes the T-bar knob and rolls its wrist 0.24 rad
   (the knob sits near 0.28 in sim: about 3.0 mmHg/s), holds that for a fixed 24 s, rolls on to 1.1 rad to empty the cuff, lets go (the valve stays open) and lifts.
3. Left arm opens its fingers and lifts 6 cm. Open loop: only the fixed layout and timing.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 789 steps (39.5 s), with a measured reading of 3.0 mmHg/s and a closest approach between the two arms of 128 mm. The negative variants score: `one_arm` 0.0 (the chestpiece stays in its dish), `rest_only` 0.0 (set on the mark and let go), `dump` 0.0 (valve wide open, 39 mmHg/s), `slow` 0.25 (1.7 mmHg/s), `no_empty` 0.5 (valve shut at 83 mmHg) and `overpump` 0.0 (seven squeezes).

## Scoring

Evaluated every control step (20 Hz); the score at the end counts. The reading is the fall from 160 to 100 mmHg after
the cuff was above 160. Listening: chestpiece centre within 15 mm of the mark and its diaphragm pressed in 2.5 mm or more (about
1 N; its own weight presses 1.2 mm). Listening is scored from placement and press, not sound.

- 0: any flag (cuff above 260 mmHg, not listening for over 1 s of the fall, pumped again during the fall, chestpiece dropped or tipped,
  pump unit moved 3 cm or turned 20 degrees; flags are permanent), the cuff never above 160, or the fall faster than 6 or slower than 1 mmHg/s.
- 0.25: listened fall at 1 to 2 or 4 to 6 mmHg/s.
- 0.5: listened fall at 2 to 4 mmHg/s but the cuff above 20 mmHg, a gripper within 3 cm of the chestpiece or pump unit, or the chestpiece not upright on the arm or in the dish.
- 1.0: all of that, unchanged for 1 s (20 steps).

## Runtime and seeds

About 1 minute per trial at 20 Hz plus Isaac Sim start-up. The layout is fixed (`-E object_jitter=0`); no seeds.

## Fidelity

Rigid bodies from boxes and cylinders. Squeeze plate (25 mm travel, 90 N/m), chestpiece diaphragm (8 mm, 400 N/m) and the
dial needle are joints; the T-bar knob is a revolute joint with damping and no return spring. The cuff, leak (0.2 mmHg/s),
squeeze (+42 mmHg) and bleed (10 mmHg/s per radian, dump past 0.9 rad) are a number updated in the control loop; no tubing. The real bleed valve is a small thumbscrew; here it is a larger knob so the parallel gripper can turn it by wrist roll.

## Real-world build

A manual BP trainer arm with an aneroid cuff and bulb, a dish for the chestpiece, and a friction bleed valve.
