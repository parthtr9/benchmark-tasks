# Put a hard pet carrier together and fit its door

A hard-sided pet carrier (MidWest Spree 19 in plastic-door carrier, model 1419SPR-PD) stands on the table part-way through its assembly, its door end toward the robot.
The plastic top rests on the plastic bottom at the back position, its hooked tabs standing in the holes of the bottom's rim, 11 mm short of home.
The two knobs on the bottom's front corners are unlocked, the four door hinges are fitted and open, and the door stands upright in a wooden stand to the robot's left.

The robot has to finish the assembly the way the maker's instructions describe it:

- "Slide the top slightly forward until it clicks into place on the plastic bottom."
- "Turn the knobs 90 degrees clockwise to lock the top and the bottom together."
- "With all four door hinges in the open position, insert the prongs of the door into the bottom and top of the carrier opening."
- "Turn all four door hinges to the closed position to secure the door in place."

A carrier like this is sold and stored as two nested shells and a loose door, and its halves are separated again to be washed, so whoever takes a cat or a small dog to the vet puts one together.
The instruction given to the robot states only the end state.

What makes it a manipulation task is that the parts hold each other by contact, in order:

- The top slides forward on four hooked tabs. At the back position their feet are under the holes and the top can be lifted off; slid home, the feet are under the bottom's rim and it cannot lift. On the real carrier the tabs flex and the top clicks home; in the simulator a bar under each of the top's rim flanges rides over a sprung ridge on the way instead (the click, about 3.7 N).
- A knob's quarter turn swings a cam bar up behind the top's front wall, so the top cannot slide back. Short of home the wall is over the cam bar, and the knob stops after about 36 degrees.
- Each door hinge is a three-sided barrel with a lever. Open, the barrel's mouth faces forward and a prong of the door can enter; closed, the mouth faces a fixed cheek and the prong is kept. A hinge closed before the door arrives keeps the prong out.
- Nothing holds the door in the opening until hinges are closed: let go with all four open, a 0.15 N nudge tips it out.
- The six fastenings on the front (two knobs, four hinges) sit 5 cm apart at each corner, and the two right-hand hinges have to be closed while the other hand holds the door 13 cm away.

Nearest tasks and how this differs: a folding produce crate whose hinged walls swing up to corner catches, a picture-frame back held flush while its clips are turned, a power strip hung on keyhole studs and slid to lock (#2335), and a rabbit lifted into a carrier (#1596).
Here two separate shells are joined by a slide onto hooked tabs, locked by knobs that only turn once the slide is complete, and a loose door is captured by its four prongs in four twist hinges.

## Revision 2: what was asked and what changed

Asked by the review (scoring):

> The real carrier uses flexing tabs rather than the simulated click ridges, so the required ridge-position check cannot be performed on the specified real object.
>
> Add real-world scoring checks for the top's clicked-home state: the specified carrier uses flexing tabs, not the simulated spring-loaded ridges. Define a practical visual or gentle functional check for the tabs and for the captured door prongs, and distinguish these checks from the simulator's millimetre-level state measurements.

Each request and where it is answered:

| The review asked | Answered in | How |
|---|---|---|
| The ridge-position check cannot be performed on the real carrier, whose top clicks home on flexing tabs | "Scoring" below and task.json `scoring` (0.25, "Real trial"); `fidelity`; task.json `setup` | The click ridges are measured only in the simulator and named everywhere as its stand-in for the flexing tabs; no real-trial check looks for them. |
| A practical visual or gentle functional check for the top's clicked-home state | "Scoring", 0.25, both "Real trial" lines | By eye: the top's front face flush with the bottom's (no step over 1 mm on a ruler laid across both), its sides within 2.5 mm, no rim gap a 1.5 mm feeler gauge enters at any corner. Gentle test: a 2 N pull back on the top's front ledge with a spring scale does not move it, and lifting the top a little by its carry handle lifts the whole carrier, because the tabs are under the bottom's rim. |
| A practical check for the captured door prongs | "Scoring", 0.75, both "Real trial" lines | By eye: each hinge lever at closed (within 10 degrees of pointing at the middle of the door) with the door's corner prong inside it, and the door within 5 mm of the sill. Gentle test: a 2 N outward pull at each corner of the door, spring scale hooked through the grille, brings no corner out. |
| Keep these checks apart from the simulator's millimetre-level measurements | "Scoring" and task.json `scoring`, every level | Each level has a "Simulator:" statement (the scorer's millimetres and degrees) and separate "Real trial:" statements (what a person sees or feels, and with which tool); the table "A real-trial check for each scoring level" below repeats the real checks only. |

The real checks use only what the maker's instructions and photos show of this carrier (the tabs and holes round the rim, the knobs' quarter turn to a stop, the four twist hinges and the door's four prongs) and a ruler, a 1.5 mm feeler gauge and a small spring scale.
The 2 N pulls are below the model's estimated 4.6 N to slide the top back over its click, so a clicked-home top passes them.

Also changed in this revision, in the scorer and its tests (prepared before the review; the scene, the parts and the instruction are as first submitted):

- The permanent 0 for moving the empty carrier more than 2 cm or tilting it more than 5 degrees is gone: sliding or turning the loose carrier does not stop an undamaged assembly, and a real carrier is not fixed to anything.
  A permanent 0 is kept only for the carrier tipping onto a side or an end (more than 45 degrees) or any of the three parts leaving the table, both visible in the video.
- No level counts while a gripper is within 3 cm of the carrier or the door, so a top, a knob or a door still held scores nothing; before, only 1.0 needed the grippers clear.
- The 1 s stillness window for 1.0 now covers each part's whole pose (its position and its turn, as the travel of its far corner), not only positions and joint angles.
- Variant scores change for two variants that end with a fingertip still on a part: `held` 0.75 to 0 and `hinges_first` 0.5 to 0. The demo stays 1.0; `knobs_first`, `no_knobs`, `door_open` and `two_hinges` keep 0, 0.25, 0.5 and 0.5.
  The revision's batches confirmed these and made the new Dream-1 run (results under "Scripted demo" below).
- The demo, in its path only: after letting go of the door the left hand now rises above the door and the shell before it crosses to the left-hand levers (in the first version its wrist passed close to the carrier's front left corner and the door), and each hand reaches its knob from 6 cm in front of it; a log of each hand's distance to every part was added.

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
  --instruction "Get the pet carrier ready for use: top locked onto the bottom and the door fitted and shut." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pet-carrier-assemble-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                                    # full score: demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=knobs_first python demo.py task.py out --no-record    # knobs turned before the top is slid home
DEMO_VARIANT=no_knobs python demo.py task.py out --no-record       # top slid home, knobs left unlocked, door and hinges done
DEMO_VARIANT=door_open python demo.py task.py out --no-record      # door put into the open hinges and let go
DEMO_VARIANT=two_hinges python demo.py task.py out --no-record     # only the right-hand pair of hinges closed
DEMO_VARIANT=hinges_first python demo.py task.py out --no-record   # all four hinges closed before the door is offered
DEMO_VARIANT=held python demo.py task.py out --no-record           # everything done, fingertips left on the last lever
python demo.py task.py --ik                                        # no simulator: IK error and arm-to-arm gap along the plan
```

The demo is open loop: every target comes from the fixed layout in `task.py` and the clock.
Its only feedback is each arm's own joint readings, used to keep the measured tool point on the plan under load.
It reads no object pose and no joint of the carrier.

1. The right hand's closed fingertips steady the bottom shell at its sill ledge; the left hand pinches the top's head ledge, pads above and below it, and draws the top forward until it has clicked home.
2. Each hand pinches the grip ridge of the knob on its side and turns it a quarter turn clockwise.
3. The left hand takes the door out of its stand by a bar of its upper sunburst, carries it across the front of the carrier, pushes its four prongs into the four open hinges, stands it on the sill and keeps holding it.
4. The right hand's closed fingertips push the two right-hand hinge levers round to closed; the left hand lets go of the door, backs off, rises above the door and the shell, comes down beside the two left-hand levers and pushes them round; both hands move away.

Measured in Isaac Sim with this package.
The demo, its video and the variants come from the revision's confirmation batch (92277d79, the `-submit` set); the Dream-1 run and its video from the revision's Dream batch (8a6ed59b).
Only `demo.py` changed between the two (two waypoints that keep the hands clear of parts they are not handling, and logging), so the Dream-1 run saw the final scene, scorer and instruction.

- The demo reaches 1.0 (success) in 1,371 steps (69 s of the 120 s limit): top home at step 140, both knobs locked at step 340, door fitted with all four hinges closed at step 1,290 (hinges at 88 to 90 degrees, prongs 1.6 to 2.9 mm from their hinge axes). IK stays within 0.1 mm of every target; the two arms' smallest gap is 40.5 mm.
- Clearance, measured every 5 steps (every step while the hands travel between tasks) from the arms' joint readings against the parts' simulated poses: away from the part being handled, no part of either hand comes within 15 mm of the carrier, the top or the door. While a hinge lever is pushed, the fingertips pass 11.6 to 12.3 mm from the door beside it (closest 11.6 mm, right hand, step 785). While reaching for and turning its knob, the left hand comes 7.3 mm from the top's front wall just above the knob (step 272) and 8.3 mm from the bottom shell (step 290); the right hand stays 14.8 mm or more from both. All figures are from the `HANDS` lines of the confirmation run's `demo.log` (batch 92277d79).
- `knobs_first` 0.0: the knobs' cam bars lift the front of the top, so it cannot go home; the pull drags the carrier, which by itself costs nothing.
- `no_knobs` 0.25: the top is home, the knobs are left unlocked, the door and hinges are done.
- `door_open` 0.5: the door is put into the open hinges and none is turned.
- `two_hinges` 0.5: only the right-hand pair of hinges is closed.
- `hinges_first` 0.0: with the hinges closed first the door's prongs cannot enter them, and the door is left leaning on the left hand.
- `held` 0.0: everything is in place but the fingertips are still on the last lever, so nothing counts.
- Dream-1, one run of 2,400 steps with the instruction above: 0.0. The left arm reached toward the door in its stand and once over the top's far left corner without taking hold of anything; the right arm stayed to the right of the carrier. The top, the knobs, the door and the hinges were not moved into place.

## Scoring

Evaluated every control step; the score at the end of the episode counts.
Each level includes the ones before it.
No level counts while a gripper is within 3 cm of the carrier or the door: a part still held or touched scores nothing.

Each level is given twice, kept apart:

- **Simulator:** what the scorer measures in the model, in millimetres and degrees.
- **Real trial:** what a person checks on the real carrier, by eye and by a gentle test, with a ruler, a 1.5 mm feeler gauge and a small spring scale.

The real carrier's top clicks home on flexing tabs; the simulator's sprung click ridges stand in for them, so the ridges are measured only in the simulator and are never looked for on the real carrier.

**0.25: the top is clicked home on the bottom.**

- Simulator: no more than 1 mm behind its forward stop, less than 2.5 mm off to either side, the gap under its rim below 1.5 mm at all four corners, and both click ridges back within 0.6 mm of their free height.
- Real trial, by eye: the top's front face is flush with the bottom's (a ruler laid across both shows no step over 1 mm), its sides are within 2.5 mm of the bottom's (ruler), and the rims meet all round (a 1.5 mm feeler gauge enters at none of the four corners).
- Real trial, gentle test (with the knobs locked or not): a 2 N pull back on the top's front ledge (spring scale) does not move it, and lifting the top a little by its carry handle lifts the whole carrier, because the tabs are under the bottom's rim (at the back position the top lifts off).

**0.5: both knobs are locked.**

- Simulator: each knob turned at least 80 of its 90 degrees, so its cam bar stands behind the top's front wall.
- Real trial: each knob's grip ridge, upright when unlocked, lies within 10 degrees of level, and a gentle further turn clockwise does not move it (it is at its stop).

**0.75: the door is fitted and shut.**

- Simulator: each of the four prongs within 4.5 mm of its hinge's axis, the door within 5 mm of its seat on the sill, and all four hinges turned at least 80 of their 90 degrees to closed.
  A prong inside a closed hinge cannot be more than 3.9 mm off its axis, so the 4.5 mm limit is passed only by captured prongs.
- Real trial, by eye: each hinge lever is turned to closed (within 10 degrees of pointing at the middle of the door) with the door's corner prong inside it, and the door's bottom edge is within 5 mm of the sill.
- Real trial, gentle test: a 2 N outward pull at each corner of the door (spring scale hooked through the grille) brings no corner out.

**1.0: the 0.75 state, held still.**

- Simulator: no point of the carrier, the top or the door moves more than 1 mm in the last second (the window covers each part's position and its turn).
- Real trial: no part visibly moves in the last second of the video, and pencil marks drawn across the top-to-bottom seam and at each corner of the door line up in the first and last frames of that second.

Sliding or turning the carrier on the table costs nothing; the carrier is loose, as in a home.
0 for good if the bottom shell was ever tipped more than 45 degrees (onto a side or an end), or the bottom shell, the top or the door ever left the table (video).

With both knobs locked the top can move back 0.7 mm at most, inside the 1 mm limit.
The 2 N pulls are well under the model's estimated forces (about 4.6 N to slide the top back over its click with the knobs unlocked), and a prong in a closed hinge is kept by the hinge's barrel, not by friction, so the pull only shows whether it is in.

## Runtime and seeds

One trial is 2,400 steps (2 minutes at 20 Hz).
The layout is fixed (`object_jitter=0`) and the task uses no random numbers, so there are no seeds.

## Fidelity

- All parts are rigid bodies built from boxes at the real size. Straight 4 mm walls stand in for the tapered, ribbed shells. The top shell has its moulded-in carry handle (a bar across a hand hole in the roof) and its roof and side vent slots; the bottom shell's vents are not modelled.
- The bottom shell (0.42 kg) with its two knobs, two bottom hinges and two click ridges is one loose articulated body; the top shell (0.38 kg) with its two top hinges is another; the door (0.12 kg) is loose.
- The click is a sprung ridge in each side rim of the bottom (800 N/m, 1.5 mm high, 25 degree ramp in, 35 degree face out). In the real carrier the give is in the flexing tabs. The ridge gives a click of the estimated force at the same 11 mm of slide, so the real trial checks the end state the simulator measures (flush fronts, no rim gap, a 2 N pull that does not move the top) rather than the ridges themselves.
- Knob and hinge detents are plain friction (0.03 N m each); the real parts click at the ends of their travel.
- Round parts (prongs, knob discs) are octagons inscribed in the real circle.
- Each sunburst of the door is 12 bars 6 mm wide; the real door has finer slats.
- The strap hooks are left out.

## Real-world build

Buy a MidWest Homes for Pets Spree 19 in plastic-door pet carrier (1419SPR-PD red, 1419SPB-PD blue or 1419SPG-PD green).
Published: 18.9 x 12.4 x 12.6 in (480 x 315 x 320 mm) outside, 17.91 x 11.5 x 12 in inside, doorway 7.16 x 7.63 in (182 x 194 mm), 2.14 lb (0.97 kg).
Every other figure below is scaled from the photos in the assembly instructions: measure the bought part and use its real value.

| Feature | Value used | Source |
|---|---|---|
| Seam between the shells | 152 mm above the table | estimate: measure the bought part |
| Shell masses | bottom 0.42 kg, top 0.38 kg | estimate: weigh the bought parts (0.97 kg in all is published) |
| Top's slide | 11 mm from the back position to home | estimate: measure the bought part |
| Tabs | four, two a side, 90 and 370 mm behind the front face; 4 x 6 mm stems with 10 mm feet | estimate: measure the bought part |
| Click | about 3.7 N on the top to slide it home, about 4.6 N to slide it back with the knobs unlocked | estimate: pull the top's front ledge with a spring scale and read the peak |
| Knobs | 38 mm across, grip ridge 36 x 9 x 10 mm, centres 140 mm up and 135 mm either side of the middle, quarter turn | estimate: measure the bought part |
| Knob torque | about 0.03 N m | estimate: 1.6 N on a spring scale hooked on the end of the grip ridge |
| Ledges | 22 mm deep, 4 mm thick; sill top 67 mm up, head ledge 273 to 277 mm up | estimate: measure the bought part |
| Door hinges | barrel 18 mm square with a 12 mm mouth, 20 to 22 mm long; lever 34 x 9 x 8 mm; axes 86 mm either side of the middle, 11 mm in front of the front face | estimate: measure the bought part |
| Hinge torque | about 0.03 N m | estimate: 0.9 N on a spring scale at the lever's tip |
| Door | 194 x 200 x 10 mm, 0.12 kg, four 6 mm prongs 20 mm long | estimate: measure and weigh the bought part |

Layout (cm from the midpoint between the arm bases, x forward, y to the robot's left):

- Carrier on its feet, door end toward the robot, front face at x 28, centred on y 0; mark the outline of its feet on the table.
- Top resting on the bottom with its tabs in the holes, slid back to the back position; both knobs unlocked; all four hinges open.
- A wooden block 6 x 12 x 10 cm with a slot 1.1 cm wide and 3 cm deep along its length, fixed to the table and centred at x 40, y 31, with the door standing in the slot, facing the robot.
- No animal is in or near the carrier during a trial.

Reset: turn the four hinges open and take the door out; stand it in the stand, the same way up and facing the robot.
Turn both knobs back to unlocked, slide the top back to the back position (leave it on the bottom), and put the bottom shell back on its outline.

## Prepared for revision

This section collects what a real trial needs beyond the sections above: every part and the one made fixture with size, material, mass and source, how each stated force is obtained and checked, a check for each scoring level, and why the operation matters.
It changes nothing in the scene or the instruction; the scorer is as revised (see "Revision 2").

### Parts and fixtures

| Part | Dimensions | Material | Mass | Source |
|---|---|---|---|---|
| Carrier: MidWest Homes for Pets Spree 19 in plastic-door carrier, 1419SPR-PD (red), 1419SPB-PD (blue) or 1419SPG-PD (green) | outside 480 x 315 x 320 mm (18.9 x 12.4 x 12.6 in); inside 455 x 292 x 305 mm; doorway 182 x 194 mm | moulded plastic, taken as polypropylene (friction 0.35) | 0.97 kg (2.14 lb) | retail listings. They differ on the outside width (12.21 to 12.5 in) and height (12.6 to 12.72 in), and only one gives the weight: measure and weigh the bought carrier |
| Its parts, as the maker's sheet lists them | "Knobs (2) Door Hinges (4) Strap Hooks (2) Door (1) Plastic Bottom (1) Plastic Top (1)" | moulded plastic | bottom 0.42 kg, top 0.38 kg, door 0.12 kg in the model | maker's assembly instructions INST_Spree_Plastic Door (1812); the mass split is an estimate: weigh the bought parts |
| Door stand, the one made fixture | block 60 x 120 x 100 mm with a slot 11 mm wide and 30 mm deep along its 120 mm length, centred | softwood or plywood | fixed to the table | made from the cut list below |
| Outline of the carrier's four feet | drawn round the feet with the carrier's front face 28 cm in front of the midpoint between the arm bases, centred | marker or tape | - | drawn at setup |

Door stand, cut list and assembly:

- One block 60 x 120 x 70 mm and two rails 24.5 x 120 x 30 mm, glued on top of the block flush with its long edges, which leaves the slot 11 mm wide and 30 mm deep between them.
  A single 60 x 120 x 100 mm block with an 11 mm groove sawn 30 mm deep along its length is the same thing.
- Fix it to the table with two screws from below or a clamp at its far end, centred 40 cm in front of the midpoint between the arm bases and 31 cm to the robot's left, the slot running left to right.
- The door (10 mm thick) stands in the slot with 0.5 mm of play a side, its bottom edge 70 mm above the table, facing the robot.

```
end view (mm)          11
                     |<->|
              +------+   +------+   ^
              | rail |   | rail |   | 30
              +------+---+------+   v
              |                 |   ^
              |      block      |   | 70
              |                 |   v
              +-----------------+
              |<----- 60 ------>|     120 long
```

Start state, by the maker's steps: the knobs are attached (step 2), the top is placed on the bottom with its tabs in the holes and not yet slid (first half of step 3), and the four hinges are pressed in and left open (steps 5 and 6).
The robot does the slide to the click (step 3), the knobs (step 4), the door (step 7) and the hinges (step 8); the strap hooks (step 9) are left off.
The maker's note for this model applies when the door is stood in the stand: "the door has triangular markings in the corner of one side only. As you place the door, this mark must go on the left side for the door to fit and operate properly."

### Measurements to take on the bought carrier

The real-world build table above lists every estimated size with the feature it belongs to.
The forces and the travel are the ones that decide the manipulation, so they are measured first; a length more than 2 mm off the model's value, or a force more than a quarter off, is put into `task.py` and the demo is run again before any real trial.

| Measure | Tool | Model's value | Constant in `task.py` |
|---|---|---|---|
| Travel of the top from the back position to home | ruler | 11 mm | `SLIDE` |
| Peak force to slide the top home, and to slide it back with the knobs unlocked | spring scale hooked on the top's head ledge | 3.7 N forward, 4.6 N back | `RIDGE_RATE`, `RIDGE_UP`, `RAMP_IN`, `RAMP_OUT` |
| Knob torque through its quarter turn, and whether it clicks at the ends | spring scale on the end of the grip ridge, 18 mm from the centre | 1.6 N (0.03 N m), no click | `KNOB_FRICTION` |
| Hinge torque through its quarter turn, and whether it clicks at the ends | spring scale at the lever's tip, 34 mm from the axis | 0.9 N (0.03 N m), no click | `HINGE_FRICTION` |
| How far a knob turns with the top short of home | protractor | stops at about 36 degrees | `CAM_X`, `CAM_HALF` |
| Masses of the bottom, the top and the door | kitchen scale | 0.42, 0.38, 0.12 kg | `mass=` of the three parts |
| Prong diameter, length and spacing; barrel mouth | calliper | 6 mm, 20 mm, 172 mm apart; 12 mm | `PIN_R`, `PIN_Z`, `PIN_Y`, `BARREL_IN` |

### How each stated force is obtained and checked

| Stated | In the simulation | Check in a real trial |
|---|---|---|
| Sliding the top home takes about 3.7 N at the click, about 4.6 N to slide it back | contact only: a bar under each of the top's rim flanges rides over a ridge sprung at 800 N/m and standing 1.5 mm high, up a 25 degree ramp going in and a 35 degree face coming out, with friction 0.35: 1.2 N a ridge going in, plus 1.3 N of drag from the 0.38 kg top; 4.6 N coming out | Hook a spring scale on the top's head ledge and pull it forward from the back position: read the peak before the click. Then pull it back with the knobs unlocked. |
| The bottom shell has to be steadied during the slide | the 0.8 kg of both shells on the table at friction 0.35 slides at about 2.7 N, less than the click | Pull the whole carrier along the table by its bottom shell with the spring scale and read the force at which it slides; then pull the top forward without holding the bottom and note whether the carrier slides before the click, as it does in the model. |
| Each knob and each hinge turns against about 0.03 N m | a friction drive on the joint: a damper of 0.05 N m s/rad capped at 0.03 N m, so the full 0.03 N m acts at 0.6 rad/s (a quarter turn in 2.6 s) or faster and less when the part is turned more slowly; there is no click at the ends of the travel | Spring scale on the end of the knob's grip ridge (model 1.6 N) and at a hinge lever's tip (model 0.9 N), pulled square to it through the quarter turn: read the peak and note any click. |
| A knob only turns once the top is home | contact only: short of home the top's front wall stands over the knob's cam bar, which stops the knob at about 36 degrees | With the top at the back position, turn a knob: it stops well short of a quarter turn. |
| Nothing holds the door until hinges are closed | contact and gravity only: a 0.15 N nudge tips the door out of four open hinges | With the four hinges open, stand the door in the opening and let go: a light touch tips it out. With one hinge closed, that corner stays. |

### A real-trial check for each scoring level

| Level | What to check | Tool |
|---|---|---|
| 0 for good: the bottom shell tipped over | In the video, the carrier on one of its sides or ends at any time (tipped more than 45 degrees). | video |
| 0 for good: a part left the table | The bottom shell, the top or the door on the floor at any time. | video |
| No level while held | At the end, both grippers are 3 cm or more from the carrier and the door. | the last video frame, ruler |
| 0.25: top clicked home | By eye: the top's front face flush with the bottom's (no step over 1 mm on a ruler laid across both) and its sides to 2.5 mm; a 1.5 mm feeler gauge enters under the rim at none of the four corners. Gentle test: a 2 N pull back on the top's front ledge does not move it, and lifting the top a little by its carry handle lifts the whole carrier. The flexing tabs themselves are not measured. | ruler, feeler gauge, spring scale |
| 0.5: both knobs locked | Each knob's grip ridge, upright when unlocked, lies within 10 degrees of level, and a gentle further turn clockwise does not move it. | protractor or by eye |
| 0.75: door fitted and shut | By eye: each of the four hinge levers within 10 degrees of pointing at the middle of the door, with the door's corner prong inside it; the door's bottom edge within 5 mm of the sill. Gentle test: a 2 N outward pull at each corner of the door (spring scale hooked through the grille) brings no corner out. | protractor, ruler, spring scale |
| 1.0 | The 0.75 state with no part visibly moving in the last second; pencil marks across the top-to-bottom seam and at each corner of the door line up in the first and last frames of that second. | video, pencil marks |

### Why this operation matters

- The category is Societal relatability: 94 million US households own at least one pet (American Pet Products Association, 2025 State of the Industry report), and a hard carrier is what takes a cat or a small dog to the vet.
- The operation scored is the one the maker's sheet asks of every buyer, because the carrier is sold as six kinds of loose parts: "Slide the top slightly forward until it clicks into place on the plastic bottom.", "Turn the knobs 90 degrees clockwise to lock the top and the bottom together.", "With all four door hinges in the open position, insert the prongs of the door into the bottom and top of the carrier opening." and "Turn all four door hinges to the closed position to secure the door in place."
- The same sheet describes taking the door off again as ordinary use: "To remove the door completely: turn all four hinges to the open position and remove the door."
- What makes it more than placing parts is the order the mechanism forces: a knob cannot turn before the top is home, a hinge closed early keeps its prong out, and the door falls out unless it is held while the first hinges are closed.
- What is not claimed: no figure was found for how often an owner takes a carrier apart and puts it together again.

Sources read on 2026-11-17: the maker's assembly instructions (midwesthomes4pets.com, INST_Spree_Plastic Door (1812), 4 pages), retail listings for the sizes and the weight, and the American Pet Products Association's 2025 State of the Industry release.
