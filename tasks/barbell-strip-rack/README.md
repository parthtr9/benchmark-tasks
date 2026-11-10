# Strip both plates off a barbell in its rack

A short training barbell lies in two J-hook cradles, a rubber plate hanging on each sleeve against its collar. Two
rubber plate holders stand on the table beyond the bar's ends. The robot has to take both plates off and stand each
one in the holder on its own side, leaving the bar in the hooks.

Gym staff are taught to unload both sides of a bar together (or with a partner on the far side): a bar in rack hooks
balances only while both ends carry the same load, so the plate left on one end levers that end down over its hook and
the bare end flies up out of its cradle. The instruction states only the goal. One arm cannot do it; each arm strips
the plate on its own side and the two must keep in step.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts built from boxes and cylinders for yam_arena task files |
| `demo.py` | The scripted full-score demo (open loop) and its negative variants |

## Run a trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Strip the barbell: take both plates off it and stand each one in the plate holder on its own side, leaving the bar resting in the rack's hooks. Then let go of everything." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/barbell-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200
DEMO_VARIANT=one_arm python demo.py task.py one_arm --no-record      # only the left plate is stripped
DEMO_VARIANT=late_right python demo.py task.py late_right --no-record # the right arm starts after the left plate is off
DEMO_VARIANT=on_table python demo.py task.py on_table --no-record    # plates set on the table, not in the holders
DEMO_VARIANT=no_release python demo.py task.py no_release --no-record # hands stay closed on the plates
DEMO_VARIANT=skew python demo.py task.py skew --no-record            # right arm 0.6 s behind: still 1.0
DEMO_DEBUG=1 python demo.py task.py demo_out --no-record             # trace every 10 steps
```

Open loop from the fixed layout: both arms come down over their plate's top rim with the fingers 45 mm apart, close,
lift 1 cm so the bore floats on the sleeve, slide the plates outward 11 cm at 2.5 cm/s until they clear the sleeve ends,
lift 4 cm, carry them to the holders, stand them in the slots, open and lift away. The debug trace prints each arm's lag
(commanded against actual tool point), each plate's position along the bar, the lift of both bar ends and hook points,
and the minimum clearance between one arm (and the plate it holds) and the other arm; the same minimum and the step
where it occurs are written to `result.json`. Checked offline with the URDF collision hulls along the commanded path,
the closest approach is over 120 mm.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 383 steps (19 s). The negative variants score: `one_arm` 0.0, `late_right` 0.0 (the plate left on the bar is unheld when the other comes off), `on_table` 0.5, `no_release` 0.75. `skew` (right arm 0.6 s behind throughout) is a robustness check, not a negative: it scores 1.0 (394 steps). In the demo's debug trace the bar never moved (end and hook lift 0.0 mm, slide 0.1 mm), the arms lagged their commands by up to 14 mm (5 mm typical), and the commanded-path clearance between the arms (with the plate each holds) was at least 129.8 mm, at step 80.

## Scoring

Evaluated every control step; the score at the end counts (`SCORE_OVER = "end"`).

- 0 if a bar end ever rose more than 1.5 cm above its seat, the bar ever rose 1.2 cm or moved 1.2 cm forward or back at
  a hook (out of its cradle), the bar ever slid 3 cm along y, or a plate's lowest point ever fell 3 cm while no gripper
  held it (never undone).
- 0.25 one plate is off the bar (centre 21.25 cm or more from the bar's middle) and standing in its holder while the
  other is still on the bar.
- 0.5 both plates are off the bar and the bar is seated in its hooks (within 1 cm), but a plate is not standing
  upright (within 12 degrees) on its holder's slot floor.
- 0.75 both plates stand in their holders and the bar is seated, but a gripper is within 3 cm of a plate or the bar, or
  things are not still.
- 1.0 (success) all of that, grippers 3 cm clear, bar and plates still within 1 mm for 20 steps (1 s).

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S-class GPU. The layout is fixed (`-E object_jitter=0`).

## Fidelity

Rigid bodies (friction 0.5): the plates are rings of boxes around a real bore. The bar is short and light so that one
plate a side overbalances it, as two or three heavy plates do on a full-size bar. No contact-force cap is enforced.

## Real-world build

A 25 mm aluminium tube, 40 cm long, with 52 mm collars and 30 mm sleeves; two 14 cm rubber technique plates with 50 mm
bores (300 g each); two printed J-hook posts and two slotted rubber plate blocks.
