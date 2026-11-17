# Hang the cleaned swing buckets back on a centrifuge rotor

A laboratory bench centrifuge (Eppendorf Centrifuge 5702, 320 x 395 x 241 mm, 20 kg) stands between the arms with its lid open.
In its chamber is the four-place swing-bucket rotor A-4-38, which turns freely on the drive when the centrifuge is stopped.
Two round 85 ml buckets already hang opposite each other, in the left and right places; the front and back places are empty.
Two more buckets, just cleaned, stand on drying mats to the left and right of the centrifuge.
The robot must hang them so that every place holds a bucket that hangs on both pegs of its place and can swing.

Why it matters.
A swing-bucket rotor may only be run with all its places fitted with identical buckets, each hanging on both pegs and free to swing out.
The operating manual says so three times: "Check that all buckets are fully attached, lubricated lightly and can swing out freely."; "For mechanical stability, all the places must be fitted with identical buckets."; "Before inserting the buckets in the grooves, make sure the grooves are clean. Dirty grooves or pegs will prevent the buckets from swinging out evenly."
The same manual has rotor and buckets taken apart and cleaned every month, and at once after any spill (section 4.2), so the buckets come off and go back on as routine work.
A bucket left riding on a peg by its band, or caught on one peg, is thrown or jams when the rotor spins up.
Clinical laboratory technologists and technicians (O*NET 29-2011.00 "Set up, clean, and maintain laboratory equipment."; 29-2012.00 "Set up, maintain, calibrate, clean, and test sterility of medical laboratory equipment."; 343,000 workers), medical assistants (31-9092.00 "Set up medical laboratory equipment."; 833,900) and phlebotomists (31-9097.00 "Process blood or other fluid samples for further analysis by other medical professionals."; 143,900) spin samples every day in centrifuges of this kind and look after their rotors: about 1.32 million US workers (employment as shown on onetonline.org, read 2026-11-17).

What the task tests.

- Two pockets onto two pegs at once.
  Each place is a bay between two rotor lobes with a steel peg on either side, both on one axis.
  A bucket has a wider band under its neck with two pockets, open downward with flared mouths.
  The bucket has to go down through the bay with its body between the two peg tips (1 mm to spare each side) and its pockets in line with the pegs, until both pocket tops rest on the pegs.
  Turned out of line it stops on the pegs by its band, 12 mm high; out of line and off centre it catches one peg and hangs crooked.
- On a rotor that turns when touched.
  Nothing holds the rotor still, so a bucket that touches a lobe on the way down turns the rotor, and with it the two buckets already hanging.
- Over a rim, inside a chamber.
  The pegs are 150 mm above the bench, 50 mm below the chamber's rim, so the bucket is carried over the rim and lowered inside a 296 mm bowl.
- Bringing a place round.
  A bucket can only be lowered into a place the hand can get above; here the empty places start at the front and back, so the rotor has to be turned by hand, as a technician does, and the hung buckets ride round with it.

Nearest tasks, and the physical difference:
"Load two tubes into a centrifuge so it can spin" puts tubes into fixed pockets of a rotor and tests the balance rule; "Take a tube out of a centrifuge that is still spinning" is stop, wait and open; "Refit a fridge door bin" lowers a bin onto two hidden bosses on a fixed door; hanging a bracket by one keyhole slot on a fixed stud needs one feature on a part that cannot move.
Here two pockets seat at once on two pegs of a rotor that turns when touched, and the part has to hang as a free pendulum afterwards.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes, with hinges and sliders, for yam_arena task files |
| `demo.py` | The scripted full-score demo and its negative variants |
| `clearance.py` | Distance between the two arms from their commanded joints, logged by the demo |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Get the centrifuge rotor ready for a run: every place fitted with a bucket that hangs on both pegs of its place and swings freely." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/centrifuge-bucket-hang-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.
There are no seeds: `object_jitter=0`, one fixed layout.

## Scripted demo

```bash
python demo.py task.py demo_out                               # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=one_only python demo.py task.py out --no-record   # only one bucket is hung
DEMO_VARIANT=no_turn python demo.py task.py out --no-record    # the rotor is not turned: only the back place can be filled
DEMO_VARIANT=one_peg python demo.py task.py out --no-record    # a bucket lowered out of line and off centre
DEMO_VARIANT=unaligned python demo.py task.py out --no-record  # a bucket lowered a quarter turn out
DEMO_VARIANT=drop python demo.py task.py out --no-record       # a bucket let go over the chamber
python demo.py task.py --ik                                    # no simulator: IK error of every step of every variant
```

The arms cannot hold a bucket above the rim with the hand pointing straight down (the wrist runs out of travel at that height), and coming in from outside the rotor the hand would hit the rim.
With the hand pitched 55 degrees and coming in over the rotor's hub, each arm reaches the place on the far side of the rotor: the right arm the left place, the left arm the right place.
So the demo first brings the empty places round to the sides:

1. The left hand puts its closed fingertips into the empty front place and walks them a quarter turn round the rotor's axis.
   The free rotor turns with them and the two hung buckets ride round to the front and back.
   It then spreads its fingers between the place's two pegs, which centres the place on the hand, closes them and lifts out.
   Meanwhile the right hand takes its bucket off the mat by the neck and lifts it clear of the rim.
2. The right hand carries its bucket over the rim and the hub to the left place, lowers it between the pegs until both pockets are down on them, lets go and backs out.
3. The left hand does the same with its bucket into the right place.

The demo is open loop: every target comes from the task file's fixed layout and the clock.
It reads no object pose, no rotor angle and no task state (its debug trace does).

Results in simulation (Isaac Sim 6.1, `object_jitter=0`): see the table at the end of this file.

## Scoring

Evaluated every control step; the score at the end of the episode counts.
Everything is measurable in a real trial with a rule, a small level and the trial video.

| Score | State at the end |
|---|---|
| 1.0 | All four places hold a bucket hanging on both pegs, each within 3 degrees of plumb, both grippers out of the chamber, and buckets and rotor moving less than 1 mm for 1 second |
| 0.75 | The same, but something still moves |
| 0.5 | Both loose buckets hang on both pegs of a place, grippers 3 cm clear of them (but a gripper is still in the chamber, or a bucket is not plumb) |
| 0.25 | One of the two loose buckets hangs on both pegs of a place, grippers 3 cm clear of it |
| 0 | Anything else |

- A bucket **hangs on both pegs** when each of the place's two pegs is within 1.5 mm of its pocket's seat, up and down and across the pocket.
  On the real parts: the bucket's rim stands 30 mm above the lobe tops on both sides (a bucket riding on the pegs by its band stands 42 mm above them), and it swings when pushed.
- The score is 0 while any bucket is left on the rotor without both pegs in its pockets and the grippers clear of it: on one peg, or riding on the pegs by its band.
  This applies to the two buckets that hang at the start as well, so knocking one off its pegs loses everything until it is put right.
- The score is 0 for good if a bucket was dropped into the chamber (its pocket seats 15 mm or more below the pegs, inside the bowl) or if the centrifuge was moved more than 10 mm.
- Either loose bucket may go in either empty place, either way round, and either arm may do anything.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps, plus about 1 minute for Isaac Sim to start.
The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

Everything is full size and rigid, built from boxes.
The centrifuge housing, its chamber and its open lid are one fixed body.
The rotor is a separate body on a free revolute joint with 0.01 N m of friction and no other drive, so it turns whenever something pushes it.
Its eight pegs are 16-sided steel prisms; the buckets are hollow, with a 4 mm chamfer at the bottom, a band with two flared pockets and a neck, and they hang from the pegs by contact alone.
Nothing is detected or scripted: a bucket is hung when gravity holds its pockets on the pegs.
The bucket's cap thread, the adapters and tubes, the rotor's lettering and the lid latch are not modelled, and the lid does not move.
A hung bucket swings on its pegs through a full quarter turn without touching the rotor or the bowl.

Published dimensions: centrifuge 320 x 395 x 241 mm and 20 kg; bucket about 190 g; 13.5 cm centrifuging radius with the 85 ml adapter; tube 38 x 106 mm.
Scaled from Eppendorf's product photos and that radius, so measure the bought part: peg axes 75 mm from the rotor axis; pegs 8 mm across, standing 7 mm proud of the lobe ends; 62 mm between lobe ends (48 mm between peg tips); bucket 104 mm tall, 46 mm across its body, 56 mm across its band, 52 mm across its neck; pocket seats 9.5 mm wide, flaring to 21.5 mm at the band's lower edge; chamber 296 mm across with its rim 200 mm and the pegs 150 mm above the bench; rotor 0.9 kg; rotor friction 0.01 N m; lid open at 85 degrees.

## Real-world build

- An Eppendorf Centrifuge 5702 (order no. 5702 000.010 or its regional equivalent) with rotor A-4-38 and four round buckets (5702 720.003; spare buckets 5702 722.006, same weight class), unplugged, lid open, rotor nut tightened with the supplied key.
  Any four-place swing-bucket bench centrifuge whose buckets hang on rotor pegs works the same way.
- The centrifuge stands on its own feet on the bench, centred between the arms, its front face 1 cm behind the line of the arm bases; mark the feet on the bench to check the 10 mm limit.
- Rotor friction is the drive's own: with the centrifuge unplugged the rotor free-wheels.
  Check it with a spring scale hooked on a lobe's rim (95 mm from the axis): it should start to turn at about 0.1 N.
- Two silicone drying mats, 140 x 140 x 3 mm, lie on the bench with their centres 24 cm in front of the arm bases and 24.5 cm to either side of the centre line.
- Before each trial: turn the rotor so that two buckets hang in the left and right places and the front and back places are empty, and let it come to rest.
  Stand one cleaned bucket upright in the middle of each mat with its two pockets on a line turned 60 degrees from the front-back direction (the right-hand bucket's line runs from front left to back right, the left-hand bucket's from front right to back left).
- Scoring: a bucket hangs on both pegs when its rim is 30 mm above the lobe tops on both sides and it swings freely when pushed; plumb is checked with a small level across its rim once it has stopped.

## Results

To be filled in from the simulation runs.
