# Light a Bunsen burner from the bench gas tap

A weighted Bunsen burner stands at the middle of the table. To its right is a brass bench gas tap with a short lever
ending in a square grip block; to its left a cup-type spark striker stands in a slotted holder. The robot must light
the burner with the striker, using the tap, then put the striker back in its holder and let go.

Lab procedure is to hold the striker at the burner's mouth first, then open the gas and strike at once: gas that runs
unlit for more than a second or two builds a cloud that flares when it lights. The task tests that timing between two
hands. One hand has to hold the striker at the mouth, ready to squeeze; the other has to turn the tap. Opening the tap
first and fetching the striker afterwards takes several seconds, and a single hand cannot hold the tap's lever and the
striker at the mouth at once, so the gas is unlit for too long (flare) or never lit.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the gas/spark/flame mechanism, `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts built from boxes, with spring-loaded sliders and hinges, for yam_arena task files (plus a visual-only part for the flame) |
| `clearance.py` | Clearance checks between the arms and the scene, used by `demo.py` |
| `demo.py` | The scripted full-score demo (open loop on the fixed layout and timing) and its negative variants |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Light the Bunsen burner with the spark striker, using the gas tap beside it, then put the striker back in its holder and let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/bunsen-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                                   # full score: demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=one_arm python demo.py task.py one_arm --no-record                    # right arm opens the tap, nothing at the mouth
DEMO_VARIANT=gas_first python demo.py task.py gas_first --no-record                # tap opened first, striker fetched after
DEMO_VARIANT=cold_strike python demo.py task.py cold_strike --no-record            # striker squeezed at the mouth, tap shut
DEMO_VARIANT=no_return python demo.py task.py no_return --no-record                # lit, but the striker is left on the bench
```

1. The left arm grips the striker's grip zone in its holder (spring arms held 24 mm apart: no spark), lifts it and
   brings its cup 15 mm above the burner's mouth.
2. The right arm takes the tap's grip block and turns it a quarter turn counter-clockwise over 1 s. From 0.6 s into
   the turn the left hand squeezes the striker three times, 0.4 s apart; the flow is past 0.3 by then, so the first
   squeeze lights the burner (gas unlit for under 0.5 s of the 1.5 s allowed).
3. The right arm lets go of the tap (it stays open in its detent) and lifts away; the left arm lifts the striker off
   the mouth, sets it back in its holder, lets go and lifts away.

The arms never touch each other or the part the other holds: the demo logs the minimum distance between any link of
one arm and the other arm or its held part (URDF collision meshes), and it stays well above 30 mm.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 359 steps (17.9 s). The negative variants score: `one_arm` 0.0 (never lit; flare once gas has run 1.5 s), `gas_first` 0.0 (flare), `cold_strike` 0.0 (tap shut, never lit) and `no_return` 0.5 (burning, striker left on the bench). The logged minimum distance between the arms (and the part each holds) is 83.3 mm.

## Scoring

Evaluated every control step; the score at the end counts (`SCORE_OVER = "end"`).

- Gas flow is 0 until the tap lever has turned 15 degrees and rises to 1 at 90 degrees. The tap clicks into detents at
  0 and 90 degrees.
- A squeeze of the striker's grip zone from 22 mm or more down to 10 mm or less is a spark. It lights the burner if
  the cup is within 2.5 cm of the mouth's centre and the flow is 0.3 or more. The flame burns while the flow is 0.2 or
  more and goes out below that.
- Time with gas flowing and no flame adds up; over 1.5 s exceeds the trial's unlit-gas safety limit (permanent). It
  is an operating limit, not a claim that a flare would occur.
- 0 if the unlit-gas limit was ever exceeded, the striker was dropped (fell 3 cm within 0.2 s, or left the table), a
  gripper touched the burner's barrel, or the burner was never lit. Either arm may work the tap or the striker.
  Otherwise:
- 0.25 the burner was lit but the flame is out at the end.
- 0.5 burning, but the striker is not back in its holder (grip zone within 1 cm of its holder position, tilted under 10
  degrees) or the tap is open less than 60 degrees.
- 0.75 burning, tap open 60 degrees or more, striker in its holder, but a gripper is within 3 cm of the striker or the
  tap's grip block, or they have moved.
- 1.0 (success, which must hold for 1 s) all of that, and the striker has moved less than 1 mm for 20 control steps.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S-class GPU, plus start-up. The layout is fixed
(`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

Rigid bodies built from boxes and cylinders. The tap is a revolute joint with spring detents at both ends; the
striker's grip zone is two spring-loaded sliding pads (2.2 N to close). Gas, spark, flame and flare are computed each
step from the lever angle, the pad gap and the cup's distance to the mouth; the flame is a visual marker. No contact
force cap is enforced.

## Real-world build

- Burner, tap and striker: a Bunsen burner on its weighted base, a brass bench gas tap with a quarter-turn lever and a
  grip block, bolted to the table, and a cup-type flint spark striker in a slotted holder, at the setup positions.
- Fuel: a 190 g butane/propane cartridge with a 30 mbar low-pressure regulator, standing outside the arms' reach
  (x 60, y -40). 8 mm rubber laboratory gas tubing with worm-drive clips: 1 m from the regulator along the table's far
  edge to the tap's inlet, 50 cm from the tap's outlet to the burner's hose barb. Leak-test every joint with soapy
  water before the first trial.
- Safe trial procedure: run only under a fume hood or a local extraction arm, or in a room with open cross-ventilation,
  on a clear, non-flammable table top with nothing above the burner. A supervisor stands at the cartridge valve
  throughout and shuts it at once if the gas runs unlit past the 1.5 s limit, if the robot knocks anything, or on any
  doubt; the robot's emergency stop is within reach. A 60 x 60 cm polycarbonate shield stands between the burner and
  people; a CO2 extinguisher and a fire blanket are at hand. The grippers' pads are silicone and stay below the
  barrel's mouth.
- The 1.5 s unlit-gas limit is the trial's operating safety threshold: at this flow it keeps the released gas far
  below a flammable cloud. Exceeding it ends the trial with score 0; it does not mean a flare occurred.
- Reset: close the tap, striker back in its holder, flame out, 2 minutes to cool, clips and cartridge valve checked.
