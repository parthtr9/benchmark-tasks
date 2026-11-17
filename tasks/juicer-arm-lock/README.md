# Put a centrifugal juicer together and close its safety locking arm

A centrifugal juicer (Breville Juice Fountain Compact BJE200XL) stands on the counter after washing, half assembled.
The motor base carries its pulp container, the stainless filter basket is clicked down on the motor's drive coupling, and the wire safety locking arm is lowered toward the robot.
The cover, with its feed chute, stands on a draining board to the robot's left.

The robot has to finish the assembly the way the instruction booklet describes it:

1. "Place the juicer cover over the Nutri Disc and integrated pulp container. The ridges of the rim of the juicer cover should align with the small plastic wings on the integrated pulp container. Lower into position."
2. "Raise the safety locking arm up and lock into the two grooves on either side of the juicer cover. The safety locking arm should now be in a vertical position and locked into place on top of the juicer cover."

The juicer will not run unless the arm is locked upright over the cover, so everyone who uses one does this before every use, and again after washing it.
The instruction given to the robot states only the end state.

What makes it a manipulation task is that the parts check each other by contact:

- The cover is carried by its feed chute over the basket (reaching into the basket scores 0: the booklet warns that its cutting disc is sharp) and only goes down when the slots in its rim are over the container's two wings; turned, it rides 10 mm high on them.
- The locking arm's wire lies across a lug on each side of the cover only when the cover is down. If the cover is high, the lugs' guide ribs are in the wire's path and stop the arm 10 to 14 degrees short of upright.
- A filter basket that was not clicked down holds the cover up by its rim in the same way (a negative variant starts that way).
- The last few degrees of the arm's swing push the wire over a sprung ridge into each groove; let go short of that, the arm falls back.

Nearest tasks and how this differs: a food processor whose bowl and lid each twist to lock (#2021), a mini chopper whose lid is pressed on (#1760), a swing-top bottle whose over-centre bail clamps one stopper, and a lid pressed into two sprung latches.
Here nothing twists and nothing is pressed shut: one wire arm on the base swings over a stack of nested parts and only reaches its two grooves when every part under it is fully down.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes, with spring-loaded sliders and hinges, for yam_arena task files |
| `demo.py` | The scripted full-score demo and its negative variants (open loop) |
| `clearance.py` | Arm-to-arm clearance from the robot's collision meshes, used by the demo's log only |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the juicer together so that its safety locking arm is closed over the cover." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/juicer-arm-lock-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                                   # full score: demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=cover_only python demo.py task.py out --no-record    # cover seated, arm left down
DEMO_VARIANT=arm_only python demo.py task.py out --no-record      # no cover, arm raised and let go
DEMO_VARIANT=basket_high python demo.py task.py out --no-record   # basket starts not clicked down; cover placed, arm raised
DEMO_VARIANT=turned python demo.py task.py out --no-record        # cover set down a quarter turn out, arm raised
DEMO_VARIANT=short python demo.py task.py out --no-record         # arm raised short of the ridges and let go
DEMO_VARIANT=held python demo.py task.py out --no-record          # everything done, the gripper never lets go
```

The demo is open loop: every target comes from the fixed layout in `task.py` and the clock.
Its only feedback is each arm's own joint readings, used to keep the measured tool point on the plan under load.
It reads no object pose and no joint of the juicer.

1. The left hand takes the cover by its feed chute, carries it over the juicer, lowers it onto the container and lets go.
2. The right hand, turned a quarter turn so its pads are above and below, takes the locking arm's handle bar, swings it up along its arc over the cover, pushes the wire over the two ridges, lets go and moves away.

Measured in Isaac Sim with this package:

- The demo reaches 1.0 (success) in 514 steps (25.7 s): cover seated at step 260 (1.3 mm off centre, no gap), wire in both grooves at step 410, 0.5 mm above the groove floors. IK stays within 0.1 mm of every target and the two arms are never closer than 80 mm.
- `cover_only` 0.5: the cover is seated and the arm is still down.
- `arm_only` 0.0: with no cover the arm swings to its 95 degree stop and falls back when it is let go.
- `basket_high` 0.0: the unclicked basket holds the cover 7 to 9 mm high, the guide ribs stop the arm at 76.6 degrees, and the hand pushing on against them lifts the back of the base 14 mm (tipped past 5 degrees, which scores 0 for good). The basket is still 8.9 mm high at the end.
- `turned` 0.0: the cover rides on the wings (up to 11 mm high), the arm stops at 77.3 degrees and the base is tipped the same way.
- `short` 0.5: raised to 84.7 degrees and let go, the arm falls back; the cover stays seated.
- `held` 0.75: everything is in place but the gripper is still on the handle.
- Dream-1, one run of 1,200 steps with the instruction above: 0.0. Both arms drew back toward their bases and hovered there for the whole minute; neither touched the cover or the locking arm.

## Scoring

Evaluated every control step; the score at the end of the episode counts.
Each level includes the ones before it, and all are end-state measurements that can be taken with a ruler and a photo.

- 0.25: the cover rests on the pulp container with its chute over the basket: its axis within 10 mm of the juicer's, upright within 10 degrees, its rim ledge no more than 12 mm above the container's wall.
- 0.5: the cover is seated: the gap under its rim ledge is below 2 mm all round, it is within 3 mm of centred, and the basket is no more than 1 mm above its clicked-down position.
- 0.75: the locking arm is in both grooves: on each side the wire lies in the groove (its centre between 4.2 mm behind and 2.5 mm in front of the groove's middle, the ridge behind it sprung back to within 1 mm) less than 3 mm above the groove's floor.
- 1.0: the same, with both grippers at least 3 cm clear of the juicer and nothing moving more than 1 mm for 1 second. With the wire under 3 mm above both groove floors the cover cannot be lifted 3 mm.
- 0 for good if the base was moved more than 2 cm or tipped more than 5 degrees, if the basket or the cover left the counter, or if a gripper reached into the filter basket (any pad point inside the rim and below it).

## Runtime and seeds

One trial is 1,200 steps (1 minute at 20 Hz).
The layout is fixed (`object_jitter=0`) and the task uses no random numbers, so there are no seeds.

## Fidelity

- All parts are rigid bodies built from boxes at the real size. Base, pulp container and drive coupling are one loose 3.4 kg body; the basket (0.20 kg) and the cover (0.25 kg) are loose; the locking arm (0.19 kg) turns on a free hinge.
- The filter basket starts clicked down. Its click is four sprung catches in the coupling, a quarter turn apart (1,200 N/m each, 2.5 mm of engagement, 45 degree faces): about 20 N to push the basket past them on a bench model of the same parts (the real click is estimated at 10 to 20 N). It was left out of the robot's work because a push on one side of the rim cocks the basket on the coupling and it jams, and two position-controlled arms could not press it evenly enough (three trials).
- The arm's detent is a sprung ridge behind each groove on the cover (1,200 N/m, 1.5 mm proud, a 20 degree ramp on the way in and a 55 degree face on the way out). In the real juicer the give is in the wire arm and the moulded cover.
- The basket's drive lugs are not modelled.
- The food pusher and the juice jug are not in the scene.

## Real-world build

Buy a Breville Juice Fountain Compact BJE200XL (replacement parts: cover and filter basket are sold separately).
Published: 231 x 249 x 399 mm overall, 3.8 to 4.4 kg, 76 mm feed chute.
Every other figure below is scaled from the booklet's product drawing: measure the bought part and use its real value.

| Feature | Value used | Source |
|---|---|---|
| Motor base | 170 mm across, top 163 mm above the counter | estimate: measure the bought part |
| Pulp container | 206 mm across, wall top 246 mm above the counter, two wings 20 mm wide on its sides | estimate: measure the bought part |
| Filter basket | rim 144 mm across, cutting disc 90 mm across, 0.20 kg | estimate: measure the bought part |
| Basket click | about 20 N | estimate: push the basket down with a spring scale and read the peak |
| Cover | 216 mm across, 140 mm tall, chute 81 mm outside, 0.25 kg | chute bore published; the rest estimate: measure the bought part |
| Locking arm | 6 mm wire, legs 156 mm, 225 mm apart, pivots 125 mm above the counter, handle 100 mm long, 0.19 kg | estimate: measure the bought part |
| Arm detent | about 3 N at the handle to close | estimate: pull the handle through the last 5 degrees with a spring scale |
| Lowered arm | legs 5 degrees above level, handle bar 39 mm above the counter | estimate: measure the bought part |

Layout (cm from the midpoint between the arm bases, x forward, y to the robot's left):

- Juicer centred at x 40, y 0, its spout away from the robot and its lowered locking arm toward the robot; mark the base's outline on the counter.
- Basket clicked down on the coupling.
- A 25 x 25 x 3.5 cm wooden draining board fixed to the counter, centred at x 35, y 23.5, with the cover standing on its rim in the middle of it, turned as it sits on the juicer.
- Unplug the juicer for every trial; the motor is never run.

Reset: pull the locking arm back over the grooves and lower it, lift the cover off and stand it on the board's mark, check the basket is still clicked down, and put the base back on its outline.
