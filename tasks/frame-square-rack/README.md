# Square up a racked face frame by its diagonal

A 0.30 kg pine face frame (16 cm deep, 30 cm wide) lies on the bench, glued up but racked: its far rail is 8 mm to the
left of its near rail, so one corner-to-corner diagonal is 14 mm longer than the other. The glued joints are stiff
(about 2 N to shear them) and the frame is light (0.9 N to slide it), so the robot must squeeze the long diagonal with
one hand on each end of it until the two diagonals match, keep the frame inside the tape rectangle stuck around it, then
let go. Cabinetmakers and bench carpenters (O*NET 51-7011.00, 47-2031.00) do exactly this before the glue sets.

## How this differs from the nearest board tasks

- *Square up a racked gantry beam* (TK-RWRT5A): one rigid beam between guide rails is turned square by pulling one end
  (one arm can do it), then both ends are pulled together. Here nothing is guided, the part itself changes shape
  (a four-bar linkage sheared by equal and opposite forces on two different members), and one hand alone only slides
  the whole frame.
- *Push a wide GPU sled home without racking it* (TK-NKQG6Y): a rigid sled is pushed evenly so that it does not turn.
  Racking is something to avoid there; here it is the thing to undo, measured by the diagonals.
- Squeeze, pull-apart and press tasks on the board are compression or tension along a line; this is shear of a frame.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: frame, tape, flags, `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts from boxes with sliders and hinges (the far rail slider has Coulomb friction) |
| `demo.py` | The scripted full-score demo (open loop) and its negative variants |
| `clearance.py` | Arm-to-arm and arm-to-scene clearance checks used by `demo.py` |

## Run a trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The glued-up face frame on the bench is racked out of square. Make it square, so that its two corner-to-corner diagonals are equal, keep it on its outline, then let go of it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/frame-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200
DEMO_VARIANT=one_arm python demo.py task.py one_arm --no-record
DEMO_VARIANT=wrong_diagonal python demo.py task.py wrong_diagonal --no-record
DEMO_VARIANT=short python demo.py task.py short --no-record
DEMO_VARIANT=no_release python demo.py task.py no_release --no-record
DEMO_DEBUG=1 python demo.py task.py trace --no-record   # trace for any variant: finger points, diagonals, displacement, lag
```

The demo is open loop (the task file's layout, timing, the arms' own commands only):

1. The left arm closes (fingers along x) on the far-left corner block, the right arm on the near-right corner block.
2. In lockstep over 4 s the left hand moves toward -y and the right toward +y, 4 mm each plus a 12 mm allowance for the soft arms'
   lag against the 2 N joint friction, ending against the far rail's 0.5 mm stop past square; hold 1.5 s.
3. Both open, lift 6 cm and wait.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 271 steps (13.6 s), with the diagonals 0.3 mm apart. The negative variants score: `one_arm` 0.0 (the whole frame slides, the rack stays 8 mm), `wrong_diagonal` 0.0 (the rack grows past 9 mm), `short` 0.25 (60% of the squeeze, diagonals 7 mm apart), `no_release` 0.5 (square but still held). The smallest distance between the two arms' collision meshes (and the part the other hand holds) is 61 mm.

## Scoring

diff is the difference between the two diagonals at the end. 0 at once and for good if a corner is lifted more than
1 cm, an outer corner leaves the tape rectangle (20 mm margin), or the far rail is pushed to its 15 mm stop. Otherwise 0
for diff over 8 mm, 0.25 for 5 to 8 mm, 0.5 for 2 to 5 mm, and for 2 mm or less 0.5 until both grippers are 3 cm clear and
the frame has moved less than 1 mm over the last 20 steps (1 s), then 1.0.

## Fidelity

The glued frame is a four-bar linkage modelled as one slider: the far rail slides along y on the near rail, from 15 mm left (the joints' turning limit, 6.6 degrees) to 0.5 mm past square (where the tenons meet their mortise cheeks), with 2 N of Coulomb friction; the stiles are hinges turned to match. Tenons and mortises are not drawn. The stiles are 1 cm high so the open fingers clear them. Rigid parts, friction 0.3 on the table.

## Real-world build

- Parts: two pine rails 30 x 3 x 3 cm and two pine stiles 3 cm wide, 1 cm high, 10 cm between shoulders, with a
  darker 3 cm block glued on each rail end. Total 0.30 kg.
- Joints: each stile end has a stub tenon 10 mm wide (along the rail), 6 mm thick and 12 mm long. Each rail has a
  mortise 11.6 mm wide, 6.4 mm high and 13 mm deep, cut **off-centre**: with the frame square, the tenon lies against
  one cheek of the mortise (a 0.05 mm sliding fit, set with a feeler gauge) and all 1.6 mm of clearance is on the
  other side. The cheek side is chosen at each joint so that all four stiles turn the same way.
- Why these sizes: racking the far rail 8 mm turns each stile asin(8/130) = 3.5 degrees, which moves the tenon's tip
  12 x sin(3.5) = 0.7 mm sideways; the 15 mm limit is 6.6 degrees and 1.4 mm, inside the 1.6 mm clearance. The stile's
  shoulders are cut back 2 mm on the opening side (a shoulder corner 15 mm from the tenon's centre moves 1.7 mm toward
  the rail at 6.6 degrees), so nothing binds between the start and the limit.
- The near-square stop: turning back, each tenon comes to rest against its mortise cheek when the frame is square;
  the 0.05 mm fit lets the far rail go about 0.5 mm past square before all four cheeks bear. This is the stop the demo
  pushes against. Verify it with a ruler: the far rail goes 15 mm to the left and no more than 0.5 mm to the right.
- Glue and shear resistance: a thin film of PVA wood glue on the tenons, used within its 10 minute open time, or the
  same film of white lithium grease for repeated trials. Check before each trial with a spring scale hooked on the far
  rail's end while the near rail is held down: the far rail should start to slide at 1.5 to 2.5 N.
- Bench: a laminate top (friction about 0.3 against planed pine), with the yellow tape rectangle 2 cm outside the
  squared frame's outline.
- Reset: as in the task's reset notes; measure both diagonals with a tape measure to judge the trial.

## Revision notes

- First review asked for the joint construction, glue and fit, how the 2 N shear resistance is obtained, and how the
  near-square stop is realised. Added.
- Second review found that a 12 mm tenon with 0.2 mm side clearance cannot turn the 3.5 degrees the 8 mm start needs.
  The joint is redesigned: a 10 mm tenon in an 11.6 mm off-centre mortise, with the turning range, the start
  position, the stop near square and the shear check worked out and verifiable as above. Setup, fidelity note and
  reset notes are updated to match.
