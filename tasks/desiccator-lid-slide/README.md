# Open a sealed desiccator and take out the sample

This is not a vacuum-loading task: nothing is pumped or sealed by the robot. The skill is mechanical and bimanual: the
lid is held on by a sliding seal that can only be slid sideways, and sliding it alone drags the whole desiccator along the
bench, so one arm must hold the base by a lug while the other slides the lid off, then the vial is taken out.

A glass desiccator (0.6 kg) with two carrying lugs is closed by a greased, vacuum-sealed glass lid (0.4 kg, 35 mm knob).
An amber sample vial stands on the porcelain plate inside. A grey pad lies to the right and a small white tray to the
left. The robot must take the vial out and stand it on the tray, put the lid on the pad, and not move the desiccator.

Chemical technicians (O*NET 19-4031.00) open a desiccator by holding its base and sliding the lid sideways: the seal
stops the lid lifting straight off, and a lid pushed alone drags the whole desiccator along the bench. In the task the
seal pulls the lid down with about 40 N (a script applies 40 N down on the lid and 40 N up on the base while the lid is
within 4.5 cm of the base axis and seated). The lid then slides at about 6.6 N while the desiccator slides on the bench
at about 2.5 N, so one arm alone fails: the desiccator moves. One arm must hold a lug (about 4 N of reaction) while the
other slides the lid 4.5 cm across, lifts it away and puts it on the pad; the vial is then taken out.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, seal (`on_step`), `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts built from boxes and cylinders, with a body-force helper, for yam_arena task files |
| `demo.py` | The scripted full-score demo (open loop), its negative variants and the `DEMO_DEBUG=1` trace |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the sample vial out of the desiccator and stand it on the white tray. Put the desiccator's lid on the grey pad, and do not move the desiccator itself." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/desiccator-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                            # full score: videos, setup.png, result.json
DEMO_VARIANT=one_arm python demo.py task.py one_arm --no-record             # the lid slid alone: the desiccator is dragged
DEMO_VARIANT=lift_straight python demo.py task.py lift_straight --no-record # the knob lifted straight up: the seal holds
DEMO_VARIANT=lid_on_table python demo.py task.py lid_on_table --no-record   # the lid set down beside the pad
DEMO_DEBUG=1 python demo.py task.py dbg --no-record                         # fingers, lid offset, base displacement, arm lag
```

The demo is open loop (fixed layout and step counts only): the left arm grips the +y lug and holds; the right arm grips
the knob and slides the lid 8.5 cm toward the pad side (y -0.21) at 2 cm/s (past the 4.5 cm where the seal lets go), waits, lifts it
6 cm and carries it to the pad, lowers and lets go. The left arm then lets go of the lug, takes the vial out, stands it on
the tray and lets go.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 667 steps (33.4 s). The negative variants score: `one_arm` 0.0 (the right arm slides the lid alone and drags the desiccator along the bench, past the 15 mm limit), `lift_straight` 0.0 (the seal holds the lid down; nothing comes off), `lid_on_table` 0.5 (vial on the tray, lid set down beside the pad). During the full demo the desiccator never moved more than 7 mm or rose more than 5.4 mm.

## Scoring

Evaluated every control step; the score at the end counts (`SCORE_OVER = "end"`).

- 0 at any time (never undone): the desiccator slid more than 15 mm, rose more than 8 mm or turned 10 degrees; the vial
  tipped more than 30 degrees; the lid dropped (fell faster than 0.7 m/s, about 2.5 cm) or tilted over 45 degrees.
- 1.0 (success, held 1 s): the vial upright (within 10 degrees) on the tray, the lid flat on the pad, both grippers open
  and every finger point 3 cm or more from the vial, lid and desiccator; vial, lid and desiccator within 1 mm over 20 steps.
- 0.75: the vial is on the tray and the lid on the pad, but a gripper is still within 3 cm or not open, or something
  is still moving.
- 0.5: the vial is on the tray and let go, but the lid is not on the pad.
- 0.25: the lid is off the desiccator (6 cm sideways or 1 cm up) but the vial is not on the tray.

## Runtime and seeds

1,200 control steps at 20 Hz (1 minute); `object_jitter=0`, no seeds.

## Fidelity

Rigid boxes and cylinders. A script models the seal as a force pair on the lid and base. Surface friction is set so the
lid slides at about 0.15 and the desiccator on the bench at about 0.25. No contact force cap is enforced.

## Real-world build

- A 150 mm glass vacuum desiccator with a ground-glass flange and a stopcock in its lid (lid 0.4 kg, base 0.6 kg with
  its porcelain plate), a 30 g amber sample vial, a 17 cm grey pad and an 8 cm white tray at the setup positions.
- Bench surface: smooth laminate or a PTFE sheet under the desiccator, chosen so the sealed desiccator slides at
  2 to 3 N (friction about 0.25); mark its outline on the surface.
- Seal: a thin, even film of high-vacuum silicone grease on the flange; seat the lid with a quarter twist.
- Vacuum: a hand vacuum pump with a gauge on the lid's stopcock. Draw down to 2.6 kPa (20 mmHg) below room pressure
  and close the stopcock. Over the 14 cm bore this holds the lid down with about 40 N, the force the simulation
  applies, so the lid cannot be lifted straight off by the arm but slides sideways.
- Force checks before every trial, with a 10 N spring scale: pull the lid sideways by its knob while holding the
  base, it should start to slide at 6 to 7 N; push the sealed desiccator along the bench, it should slide at 2 to
  3 N. Re-grease or re-pump until both are in range. Sliding the lid 4.5 cm off-axis breaks the seal.
- Reset: vial back on the plate, desiccator on its outline, re-grease if the film is broken, seat the lid, pump down,
  close the stopcock, repeat the two force checks.
