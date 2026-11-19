# Set an exam table's stirrups

The foot end of an exam table's stirrup shelf is clamped on the bench, its foot end toward the robot.
It is built as the bench fixture in `fabrication/` (printed parts, steel bar and plywood, about $175, with drawings and a bill of materials), built to the motions of a Midmark Ritter 204 table's stirrups (draw, unfold, lift and drop into a notch); the fixture is the hardware of record, and its dimensions are the simulation's by construction.
Both stirrups are stowed: each stirrup lies folded flat along its stirrup bar, and the bar is pushed all the way into the shelf, so only the stirrup's hinge end shows at the front.
The robot is told only:

> Set this exam table's stirrups for an exam: both out, unfolded and locked in their outermost positions.

Setting the stirrups is part of preparing the room for a pelvic examination, and the case for the task rests on how often that is done.
About 44.2 million pelvic examinations were performed in the US in 2012 (the US Preventive Services Task Force's evidence review, JAMA 2017, from CDC survey data; summarised in American Family Physician, 2017, https://aafp.org/afp/2017/0815/od1.html), and about 34.7 million in 2018.
They are typically done in the lithotomy position, with the patient's feet in the table's stirrups, in primary care and obstetrics and gynecology clinics; the stirrups are set before each one.
The maker's user guide gives the job in three steps: "A) Pull the stirrup out, then unfold. B) Lift the stirrup slightly, then move it left or right as desired. C) Release stirrup to lock in position."
Its caution: "Be sure the stirrups are locked in place before using. The stirrups will not support the patient's entire weight."

The instruction names only the end state.
The parts hold the order, as on the real table:

- a folded stirrup cannot turn up while any of it is inside the shelf, because the top edge of the shelf's window stops it;
- the bar cannot swing sideways while its detent post is in the guide bracket's groove (all along the draw) or in a notch, and the post drops out only when the stirrup end is lifted;
- a stirrup swung out and let go between lock positions comes to rest unlocked on the bracket's teeth, tilted 4 degrees, and looks set (the wrong state the caution warns about).

Each arm sets the stirrup on its own side: the two stirrups are 49 cm apart and swing out to 89 cm apart, so neither arm can reach both.

Nearest board tasks: extending a suitcase's handle until both rails lock (TK-28LP2A, accepted) and unfolding a child's step stool until each leg's lock clicks (TK-2RU7AY, not accepted).
Each of those is one motion into one lock.
Here each stirrup chains five motions that gate each other: drawn out, unfolded, lifted out of its detent, shifted sideways and dropped into a notch, and the robot has to leave it locked rather than merely positioned.

## Files

| File | What |
|---|---|
| `task.py` | The scene (the shelf with its windows and guide brackets, the two stirrup assemblies) and the scorer |
| `partkit.py` | Rigid parts built from boxes and cylinders, with chained hinges and sliders (each stirrup: boss swing, bar lift, bar draw, stirrup roll pin) |
| `meshkit.py` | Mesh shapes: the stirrup's teardrop loop (a bent round rod) |
| `demo.py` | The scripted full-score demo and its negative variants |
| `task.json` | The submission text |
| `fabrication/` | The bench fixture: printable parts, drawings, bill of materials, forces and how each is set and checked |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set this exam table's stirrups for an exam: both out, unfolded and locked in their outermost positions." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/exam-stirrup-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out                                  # full score
DEMO_VARIANT=one_side python demo.py task.py one_side             # only the left stirrup set (0.5)
DEMO_VARIANT=no_unfold python demo.py task.py no_unfold           # both drawn, swung and locked, left folded (0)
DEMO_VARIANT=unfold_first python demo.py task.py unfold_first     # tries to turn each stowed stirrup up first: the window stops it (0)
DEMO_VARIANT=no_lift python demo.py task.py no_lift               # drawn and unfolded, pushed sideways without lifting: the post stops it
DEMO_VARIANT=between python demo.py task.py between               # swung only 18 deg and let down between notches: unlocked (0.5)
DEMO_VARIANT=held python demo.py task.py held                     # as the demo, the right hand left on its stirrup (0.5)
python demo.py task.py --ik                                       # IK error of every planned target, no simulator
python demo.py task.py --dry                                      # the script on the IK alone: IK error and arm-to-arm gap
```

Both arms work at once, each on its own side.
Each hand pinches the stirrup's hinge end with the hand pitched 45 degrees and draws the bar out 40 cm to its collar, then lets go.
It pinches the folded stirrup's arm from straight above and carries it round its roll pin to 96 degrees, the hand turning with it.
Still holding it, it lifts (the bar tilts about the boss and the post drops out of the bracket), swings it 24 degrees outward round the boss with the hand turning to match, and lets it down so the post rises into the outermost notch.
It lets go and backs off through its open fingertips.
Every target comes from the fixed layout and the clock, and the lift is trimmed by the arm's own joint readings; the demo reads no object pose or joint state.

An offline rehearsal in MuJoCo (`repairkit/tasks/exam_stirrup_set.py`: the task's own shapes, joints and scorer, each hand sprung to its commanded pose at 230 N/m) gives: demo 1.0 in 565 steps; one_side 0.5, no_unfold 0, unfold_first 0, no_lift 0.5, between 0.5, held 0.5.
Isaac results for this version (Isaac Sim 6.1, yam_arena, 20 Hz; batch 9f944782):

| Run | What it does | Score |
|---|---|---|
| demo | both stirrups drawn, unfolded, lifted, swung 24 deg out and dropped into the outermost notch | 1.0 in 584 steps |
| `one_side` | only the left stirrup set | 0.5 |
| `no_unfold` | both drawn, swung and locked, left folded | 0 |
| `unfold_first` | the stowed stirrup will not turn up (the window stops it); then drawn, swung and locked folded | 0 |
| `no_lift` | the sideways push without a lift is stopped by the post (the swing stays under 1 deg); the hands then let go and withdraw, and on Isaac both stirrups tip back over as their hands let go (a demo artefact: in MuJoCo both stay upright, 0.5) | 0 |
| `between` | swung 18 deg and let down between notches: each rests unlocked on the teeth, tilted 3.5 deg | 0.5 |
| `held` | as the demo, the right hand left on its stirrup | 0.5 |

Dream-1, one run at the 1,200-step limit on this version (batch 838b6cd6; the same batch re-ran the demo and every variant with the scores above): 0.0.
Both hands reached the stirrups' hinge ends at the face and stayed there without drawing either bar, and the run ended at the step limit.

## Scoring

Checked with a rule, a protractor on the pivot boss and the trial video once everything is still.
Each stirrup is scored on its own:

| Points per stirrup | State |
|---|---|
| 2 | Set: out (below) and locked in the outermost notch: the bar within 2 deg of 24 deg outward and within 1.5 deg of level, the post up in that notch |
| 1 | Out: drawn to within 8 mm of its collar and unfolded (within 5 deg of its open stop), but not locked in the outermost notch (straight, in another notch, or resting unlocked between notches) |
| 0 | Otherwise, including drawn but still folded |

"Drawn to within 8 mm of its collar": the roll pin starts 20 mm in front of the shelf's face and the bar draws 400 mm to its collar, so the pin is then 420 mm out; within 8 mm means at least 412 mm.
A stirrup earns its points only once both grippers have been at least 3 cm from it (its bar and its stirrup) and it has been still, both for the last 1 s (in simulation no point of it moves more than 1 mm; in a real trial no visible motion in the last second of the video); a stirrup still held or still moving earns 0 points, and the other stirrup still counts.

Score: the two stirrups' points / 4 (0, 0.25, 0.5, 0.75, 1.0).
0 for good if a stirrup is ever forced 3 deg or 3 mm past one of its stops (the draw collar, the open stop, the boss's swing stops, the bar's lift stops).

## Runtime and seeds

1,200 steps (1 minute at 20 Hz); the demo finishes in 584 steps.
`object_jitter=0`: one fixed layout, no random seeds.

## Fidelity

Rigid bodies at the bench fixture's dimensions, which are the model's by construction (`fabrication/make_fixture.py` reads them from `task.py`); the shelf is fixed.
Each stirrup is a chain of free joints with real stops and light damping: the pivot boss's swing (stops at 15 deg inward and 27 deg outward), the bar's lift in the boss (-3 to +8 deg, about a pin 10 mm below the bar's centre line), the bar's draw (0 to 400 mm, the collar) and the stirrup's roll pin (folded to 100 deg open).
The boss holds the bar from below: a T-rail hangs 4 mm under the bar and the trunnion's lips run in the gap, so a folded stirrup lying on the bar slides over the boss.
The order is held by contact alone: the shelf's window over the folded stirrup, and the lateral lock's groove, notches and teeth against the detent post.
The lock spring is the fixture's compression spring (2.13 N/mm, 11.2 N preload, 75 mm behind the boss axis), modelled as a rotational spring of 12 N m/rad at the bar's lift preloaded 4 deg, the same stiffness; lifting the unfolded stirrup out of a notch takes about 7 to 9 N at its arm.
Contact friction 0.3.
On a Midmark table the same mechanism's sizes are not published; a lab using one measures them (below).

## Real-world build

### The bench fixture (the specified hardware, about $175)

For a lab without an exam table, `fabrication/` specifies the stirrup mechanism as a bench fixture that keeps the draw, the unfold and the lock: 17 printable parts, five dimensioned drawings, a cut list and a priced bill of materials ($175.36, $97.36 of it at prices listed on 2026-11-18), and the procedure in `fabrication/README.md`.
Its critical dimensions are exact by construction and are the ones the simulation uses: `fabrication/make_fixture.py` reads every size, angle, stop, spring constant and mass from `task.py` and writes the parts and drawings from them.

- Shelf: a 670 x 580 x 220 mm plywood box (3/4 in base, 1/4 in sides, top and face) with two 90 mm windows in the face, 121.7 to 163.2 mm above the bench, centred 245 mm either side; clamped to the bench, its rear 320 mm on a 2 x 4 stand.
- Bars: 565 mm of 3/4 x 1/4 in steel flat, 140 mm above the bench; a 1/2 x 1/8 in aluminium T-rail on spacers under each; the wide spacer is the draw collar; the post is an M6 set screw standing 30 mm above the bar.
- Pivot boss: printed pedestal, yoke (swing stops 15 deg in, 27 deg out) and trunnion (lift on two dowel pins), all below the bar.
- Lateral lock: a printed notch plate (notches at -12, 0, 12 and 24 deg, 8 mm wide, 5 mm deep) and basswood groove rails under a plywood ceiling strip; a 2.1 N/mm compression spring on a printed roller presses the bar's tail up (preload set with an M8 screw).
- Stirrups: a steel arm, a bent 1/4 in rod core in a printed loop shell, a printed clevis on an M5 roll pin with an M3 open-stop screw at 100 deg; 0.30 kg each.

### The Midmark table (optional real object)

The same mechanism on the maker's table: the upperwrap and footrest shelf (Midmark service parts list SF-1864, page E-18, item 9) with both stirrup assemblies (stirrup 020-0337-00-216 with roll pin, stirrup bar 051-1003-01), guide brackets (050-5027-00) with their leaf springs (016-0400-00) and pivot bosses (053-0387-00).
Prices seen November 2026: the stirrup assembly set 002-0719-09 about $650 new (PartsSource); a used Ritter 204 or 604 table with stirrups about $800 to $1,500; a lab that already has a Ritter 203, 204, 222, 223 or 604 table can run the trial on it in place.
Midmark publishes the parts but no dimensions, so a lab using a table measures it with the table at the end of this section and runs the demo again with its values.

| Fixture part | Midmark part (no.) |
|---|---|
| Plywood shelf with two windows | Upperwrap and footrest shelf (E-18 item 9) |
| Printed stirrup on its roll pin | Stirrup 020-0337-00-216, roll pin 042-0001-39 |
| 3/4 x 1/4 in steel bar with its T-rail | Stirrup bar 051-1003-01 |
| Printed pedestal, yoke and trunnion | Pivot boss 053-0387-00 |
| Notch plate, groove rails and ceiling strip | Guide bracket 050-5027-00 |
| Compression spring on its roller | Guide bracket spring 016-0400-00 (a leaf spring) |
| Wide spacer under the bar | The draw stop (screw 040-0010-47 in the boss) |
| Detent post (M6 set screw) | The bar tail's engagement with the guide bracket, which the parts list shows only as the bracket and its spring |

### Placement

The shelf is clamped to the bench with its front face 650 mm in front of the arm bases' line, centred between the arms.
Its rear 320 mm overhangs the bench's far edge and rests on the stand, whose rail is level with the bench top.
The arms are at their ready poses with the grippers open.

### How each property is set and checked

- Draw: with the stirrup folded, pull the bar out by its hinge end until the collar stops it; the roll pin's axis is then 420 mm from the shelf's face (it starts 20 mm out), checked with a rule.
- Lock spring: with the bar drawn and straight and the stirrup taken off, hook a spring scale under the detent post (75 mm behind the boss) and pull straight down.
  The post starts to leave its notch at about 26 N and is clear of it (5 mm down) at about 37 N.
  Of the 26 N, about 15 N is the bar's own weight (front-heavy about the boss: 1.13 N m with its T-rail) and about 11 N is the spring, set with the M8 screw under it.
- Lift to unlock: with the stirrup unfolded on its stop and the bar locked straight, hook the spring scale on the stirrup's arm 115 mm above the roll pin and pull straight up.
  The bar starts to tilt at about 7 N, and the arm rises about 33 mm (about 9 N by then, the bar tilted 3.8 deg) before the post is out of the notch and the bar can swing.
- Lock positions: with a protractor on the boss, the bar drops into a notch at 12 deg inward, straight, 12 and 24 deg outward (within 1 deg); let down between notches it rests tilted about 4 deg, stirrup end up, on the teeth.
- Unfold: the stirrup cannot be turned up more than about 20 deg until it is drawn at least 250 mm out; fully out it opens to its stop 10 deg past upright and stays there.
- Masses: each bar with its T-rail 0.60 kg, each stirrup 0.30 kg (kitchen scale).

### Real-trial checks

Per stirrup, from the trial video and with a ruler and a protractor once everything is still:
"out" means drawn to within 8 mm of the collar, which is the roll pin at least 412 mm from the face (fully drawn it is 420 mm, from 20 mm at the start), and the stirrup leaning on its open stop;
"set" means out and locked in the outermost notch: the bar at 24 deg outward (within 2 deg) and level (within 1.5 deg), with the post up in that notch (seen from the side between the base and the ceiling strip; a locked bar does not swing when nudged sideways without lifting, which is checked after the trial).
Each stirrup is judged on its own: it counts only if no gripper has been within 3 cm of it and it has not moved, both for the last second of the video; a stirrup still held or moving earns nothing, and the other is still scored.

| Score | What to check |
|---|---|
| 1.0 | Both stirrups set |
| 0.75 | One set, the other out |
| 0.5 | One set and the other not out (still in, or still folded), or both out |
| 0.25 | One out, the other not out |
| 0 | Neither stirrup out (a stirrup still held or moving counts as not out); 0 for good once a stirrup has been forced past a stop (bent bar, stirrup past its open stop, bar pulled past its collar) |

A stop forced 3 deg or 3 mm past its limit is damage and is checked after the trial, not from the video, against marks made at set-up:
- the draw collar: a witness line scribed on the bar 3 mm beyond where it meets the boss; if the line has passed into the boss, or the collar has shifted on the bar, the bar was pulled past its collar;
- the stirrup's open stop: a 3 mm feeler gauge between the stirrup's arm and its stop face; if the stirrup now leans past its stop by more than the gauge, the stop or the roll pin has been bent;
- the swing stops and the lift stops: with a protractor on the boss, the bar must stop at 15 deg inward and 27 deg outward, and the lift must stop 8 deg up, as measured at set-up (within 3 deg); a bar that now goes further has bent or sheared a stop.

### Measurements on a Midmark table (only if one is used instead of the fixture)

Midmark publishes the part numbers (SF-1864, page E-18) but no dimensions or forces; a lab using a table measures these and replaces the model's values where they differ by more than a quarter (the demo is then run again). The fixture needs none of this: its values are the model's.

| Measure | Part (Midmark no.) | Tool | Model's value | Constant in `task.py` |
|---|---|---|---|---|
| Draw: roll pin from the shelf's face, stowed and fully out | stirrup bar 051-1003-01, pivot boss 053-0387-00 | steel rule | roll pin 20 mm out stowed, 400 mm travel, 420 mm out drawn | `STICK`, `PIVOT_BACK`, `DRAW` |
| Bar section and length | 051-1003-01 | calliper | 19 x 6.35 mm, 565 mm | `BW`, `BT`, `X_FRONT0`, `X_TAIL0` |
| Lock positions: the angles at which the bar drops into a notch | guide bracket 050-5027-00 | protractor on the boss | 12 deg inward, straight, 12 and 24 deg outward | `NOTCHES`, `OUTER` |
| Notch width and depth (or the lift needed to clear a notch) | 050-5027-00 | calliper, or the stirrup end's rise before the bar swings | 8 x 5 mm (33 mm rise at the arm) | `NOTCH_W`, `NOTCH_D` |
| The guide-bracket spring's force | guide bracket spring 016-0400-00 | spring scale under the bar's tail, the bar's own weight subtracted (see the force checks) | about 11 N on the post | `LIFT_K`, `LIFT_PRE` |
| Lift to unlock, at the stirrup's arm 115 mm above the roll pin | the assembly | spring scale | 7 N to start, 9 N at release | follows from the spring and the masses |
| Stirrup: reach from the roll pin, loop width, open-stop angle, mass | stirrup 020-0337-00-216 | calliper, protractor, scale | 250 mm, 70 mm, 100 deg, 0.30 kg | `LOOP_S0`, `LOOP_L`, `LOOP_W`, `UNFOLD_STOP`, `STIRRUP_MASS` |
| Bar mass | 051-1003-01 | scale | 0.60 kg (the fixture's bar with its T-rail) | `BAR_MASS` |
| Swing stops | pivot boss 053-0387-00 | protractor | 15 deg inward, 27 deg outward | `SWING_IN`, `SWING_OUT` |

### Reset

For each stirrup: lift its end slightly, swing it straight until the post drops into the groove's end, fold the stirrup flat along its bar, and push the bar into the shelf until only the stirrup's hinge end (38 mm) shows.
Return the arms to their ready poses.

## Revision 2: what was asked and what changed

The first version was not accepted (revisable). The reviewer wrote:

> Importance (not met): Preparing examination rooms is regular work for medical assistants, but their total workforce does not establish the number regularly setting pelvic-exam stirrups; the reported examination volume supports a case under Other.
>
> Scoring (not met): The real-trial check requires the roll pin 400 mm from the face, while the simulated 400 mm draw starts with the pin about 20 mm outside it.
>
> Clarity and reproducibility (not met): The quoted 400 mm travel and 400 mm final distance from the face are inconsistent, and the real mechanism's critical dimensions remain estimates.
>
> Feasibility (not met): The least expensive stated purchase is an $800 used table, with no lower-cost route for a lab without existing equipment.
>
> What to change: The reported total number of medical assistants does not show how many regularly set stirrups. Please change the domain to Other and explain the impact of this clinical preparation task using the examination volume. Provide an affordable, repeatable way for a lab without an existing exam table to obtain or build the real stirrup mechanism, retaining its draw, unfold and locking behavior. Measure the real hardware and make the setup and score consistent: the simulated pin starts about 20 mm in front of the face and travels 400 mm, whereas the real-trial check requires a final distance of 400 mm. Define the out and locked checks using the actual collar and outermost notch, and reconcile the scoring table's statement that a held stirrup scores zero with the per-stirrup partial-credit rule.

Each request, and where it is answered:

| Request | What changed | Where |
|---|---|---|
| Change the domain to Other and explain the impact using the examination volume | Category Other. The case is the examination volume: about 44.2 million US pelvic examinations in 2012 (USPSTF evidence review, JAMA 2017, from CDC data), 34.7 million in 2018, the stirrups set before each. No workforce total is presented as the case. | task.json `category` and `context`; the opening of this README |
| An affordable, repeatable way for a lab without an exam table to obtain or build the real stirrup mechanism, retaining its draw, unfold and locking behaviour | A bench fixture, about $175: 17 printable parts, five dimensioned drawings, a cut list, a priced bill of materials, the build procedure, and each force with how it is set and checked. It keeps the 400 mm draw to a collar, the unfold through the window to the open stop, the spring-loaded lateral lock and the outermost notch. | `fabrication/` (and `fabrication/README.md`); "Real-world build" above |
| Measure the real hardware | The fixture is the hardware of record. Its critical dimensions are exact by construction, and they are the simulation's: `fabrication/make_fixture.py` reads every size, angle, stop, spring constant and mass from `task.py` and generates the printed parts and drawings from them, so there is nothing left to estimate. The bar's mass in the model is now the fixture's (0.60 kg with its T-rail). The Midmark table is kept as the optional real object; its sizes are not published, so a lab using one measures them with the table given. | "Real-world build", "Fidelity", "Measurements on a Midmark table" above; task.json `setup` and `fidelity` |
| Make the setup and score consistent (pin 20 mm in front, 400 mm travel, a 400 mm final distance in the check) | One set of numbers everywhere: the roll pin starts 20 mm in front of the face; the bar draws 400 mm to its collar; the pin is then 420 mm out. "Out" is within 8 mm of the collar, which is the pin at least 412 mm out. The scorer already measured the draw from the collar; only the texts were wrong. | task.json `setup` and `scoring`; "Scoring", "How each property is set and checked", "Real-trial checks" and the measurement table above |
| Define the out and locked checks using the actual collar and outermost notch | "Out": drawn to within 8 mm of the collar (the fixture's draw stop under the bar, which meets the boss) and unfolded within 5 deg of the open stop. "Set": out and locked in the outermost notch, with the post up in that notch and the bar at 24 deg (within 2) and level (within 1.5). | task.json `scoring`; "Scoring" and "Real-trial checks" above |
| Reconcile "a held stirrup scores zero" with the per-stirrup partial credit | Each stirrup is judged on its own: a held or moving stirrup earns 0 points and the other still counts. This is what the scorer always did; the real-trial table's 0 row now says it, and a test checks it (one stirrup held, the other set: 0.5). | task.json `scoring`; "Scoring" and "Real-trial checks" above |

One more change, found while laying out the fixture.
In the first version each stowed stirrup lay folded through its pivot boss's body, a cylinder round the bar.
The simulation allowed this only because parts of one jointed assembly do not collide.
The boss now holds the bar from below: a T-rail under the bar, the trunnion's lips in the gap, and the lift pin 10 mm below the bar's centre line.
The folded stirrup slides over it, in the model and on the fixture.
The motions and the scoring were not affected.
The demo, the variants and the Dream-1 run are new runs on this version.
