# Set an exam table's stirrups

The foot end of a manual examination table (Midmark Ritter 204 class) is clamped on the bench, its foot end toward the robot.
It is the table's real "upperwrap and footrest shelf" sub-assembly, sold as a service part (Midmark parts list SF-1864, page E-18, item 9), with its upholstered top.
Both stirrups are stowed: each stirrup lies folded flat along its stirrup bar, and the bar is pushed all the way into the shelf, so only the stirrup's hinge end shows at the front.
The robot is told only:

> Set this exam table's stirrups for an exam: both out, unfolded and locked in their outermost positions.

Setting the stirrups is part of preparing an exam room for a pelvic examination.
Medical assistants (O*NET 31-9092, 833,900 employed) have "Prepare treatment rooms for patient examinations" among their listed tasks, and registered nurses (29-1141, 3,465,400 employed) "Prepare patients for and assist with examinations or treatments".
No count of how often each worker sets stirrups was found; what is sourced is the number of examinations that use them: about 44.2 million pelvic examinations were performed in the US in 2012 (American Family Physician, 2017, https://aafp.org/afp/2017/0815/od1.html), in primary care and obstetrics and gynecology.
The maker's user guide gives the job in three steps: "A) Pull the stirrup out, then unfold. B) Lift the stirrup slightly, then move it left or right as desired. C) Release stirrup to lock in position."
Its caution: "Be sure the stirrups are locked in place before using. The stirrups will not support the patient's entire weight."

The instruction names only the end state.
The parts hold the order, as on the real table:

- a folded stirrup cannot turn up while any of it is inside the shelf, because the top edge of the shelf's window stops it;
- the bar cannot swing sideways while its detent post is in the guide bracket's groove (all along the draw) or in a notch, and the post drops out only when the stirrup end is lifted;
- a stirrup swung out and let go between lock positions comes to rest unlocked on the bracket's teeth, tilted 4 degrees, and looks set (the wrong state the caution warns about).

Each arm sets the stirrup on its own side: the two stirrups are 49 cm apart and swing out to 89 cm apart, so neither arm can reach both.

Nearest board tasks: swinging a mitre saw's turntable to its 45 degree detent with the squeeze release (TK-K64735, another team, pending), extending a suitcase's handle until both rails lock (TK-28LP2A, accepted) and unfolding a child's step stool until each leg's lock clicks (TK-2RU7AY, pending).
Each of those is one motion into one lock.
Here each stirrup chains five motions that gate each other: drawn out, unfolded, lifted out of its detent, shifted sideways and dropped into a notch, and the robot has to leave it locked rather than merely positioned.
Our own nearest task slides a carriage onto a rail; the draw here is only the first of the five.

## Files

| File | What |
|---|---|
| `task.py` | The scene (the shelf with its windows and guide brackets, the two stirrup assemblies) and the scorer |
| `partkit.py` | Rigid parts built from boxes and cylinders, with chained hinges and sliders (each stirrup: boss swing, bar lift, bar draw, stirrup roll pin) |
| `meshkit.py` | Mesh shapes: the stirrup's teardrop loop (a bent round rod) |
| `demo.py` | The scripted full-score demo and its negative variants |
| `task.json` | The submission text |

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

An offline rehearsal in MuJoCo (`repairkit/tasks/exam_stirrup_set.py`: the task's own shapes, joints and scorer, each hand sprung to its commanded pose at 230 N/m) gives: demo 1.0 in 620 steps; one_side 0.5, no_unfold 0, unfold_first 0, no_lift 0.5, between 0.5, held 0.5.
Isaac results (Isaac Sim 6.1, yam_arena, 20 Hz):

| Run | What it does | Score |
|---|---|---|
| demo | both stirrups drawn, unfolded, lifted, swung 24 deg out and dropped into the outermost notch | 1.0 in 584 steps |
| `one_side` | only the left stirrup set | 0.5 |
| `no_unfold` | both drawn, swung and locked, left folded | 0 |
| `unfold_first` | the stowed stirrup will not turn up (the window stops it); then drawn, swung and locked folded | 0 |
| `no_lift` | the sideways push without a lift is stopped by the post (the swing stays under 1 deg); the hand then lets go and withdraws: on Isaac the right stirrup stays upright and unlocked (1 point), the left one tips back over as its hand lets go (a demo artefact: in MuJoCo both stay upright, 0.5) | 0.25 |
| `between` | swung 18 deg and let down between notches: each rests unlocked on the teeth, tilted 3.5 deg | 0.5 |
| `held` | as the demo, the right hand left on its stirrup | 0.5 |

Dream-1, one run at the 1,200-step limit (the same batch re-ran the demo and every variant with the scores above): 0.0.
It reached toward the shelf twice and hovered near the bars' front ends but never drew a stirrup, and the run ended at the step limit.

## Scoring

Checked with a ruler, a protractor on the pivot boss and the trial video once everything is still.
Each stirrup counts only once both grippers have been at least 3 cm from it (its bar and its stirrup) and no point of it has moved more than 1 mm, both for the last 1 s (a stirrup held still and then let go counts only after a second of being let go).

| Points per stirrup | State |
|---|---|
| 2 | Fully out (within 8 mm of its draw collar: a post can rise into a notch anywhere in the notch's 16 mm band), unfolded (within 5 deg of its open stop) and locked in its outermost position: swung within 2 deg of 24 deg outward and let down, the bar within 1.5 deg of level with the post up in the notch |
| 1 | Out and unfolded, but not locked in the outermost position (straight, in another position, or resting unlocked between positions) |
| 0 | Otherwise, including drawn but still folded |

Score: the two stirrups' points / 4 (0, 0.25, 0.5, 0.75, 1.0).
0 for good if a stirrup is ever forced 3 deg or 3 mm past one of its stops (the draw collar, the open stop, the boss's swing stops, the bar's lift stops).

## Runtime and seeds

1,200 steps (1 minute at 20 Hz); the demo finishes in about 610 steps.
`object_jitter=0`: one fixed layout, no random seeds.

## Fidelity

Rigid bodies at the parts' listed or estimated sizes, the shelf fixed.
Each stirrup is a chain of free joints with real stops and light damping: the pivot boss's swing (stops at 15 deg inward and 27 deg outward), the bar's lift in the boss (-3 to +8 deg), the bar's draw (0 to 400 mm, the collar) and the stirrup's roll pin (folded to 100 deg open).
The order is held by contact alone: the shelf's window over the folded stirrup, and the guide bracket's groove, notches and teeth against the detent post.
The guide bracket's leaf spring is a rotational spring of 12 N m/rad at the bar's lift, preloaded 4 deg: 2,100 N/m and about 11 N at the post (lifting the unfolded stirrup out of a notch takes 7 to 9 N at its arm).
Contact friction 0.3 (steel on steel, lightly greased).
Estimated, not measured: the bar's and stirrup's sizes and masses, the draw, the boss's position, the post, the notch angles and depth, the spring.
Every internal size is an estimate: the maker's parts list draws the parts but gives no dimensions, so each value is listed in the parts table below with what to measure on the bought part.
The detent post on the bar's tail is this model's own choice, not a listed part: it stands for the real engagement between the bar's tail and the guide bracket (050-5027-00), which the parts list shows only as the bracket and its leaf spring, so the groove, notches and teeth the post rides in are an estimate of that engagement.

## Real-world build

A real trial uses the maker's parts: the upperwrap and footrest shelf (E-18 item 9) with both stirrup assemblies (stirrup 020-0337-00-216 with roll pin, stirrup bar 051-1003-01), guide brackets (050-5027-00) with their leaf springs (016-0400-00) and pivot bosses (053-0387-00), from a Ritter 204 or 604 table or as service parts.
The dimensions below are this model's estimates; measure the bought parts and use those.

Cost and the cheapest route (prices seen November 2026; check before ordering):

- New: the stirrup assembly set 002-0719-09 (both stirrups, bars, guide brackets, springs and bosses; listed to fit the 203, 204, 222, 223, 604, 622 and 623 tables) is sold new by medical parts dealers, for example PartsSource at about $650 a set.
  The upperwrap and footrest shelf is a separate service part, ordered by the table's serial number from a Midmark dealer.
- Cheapest: a used or refurbished Ritter 204 or 604 table with its stirrups, listed at about $800 to $1,500 on used-equipment sites (DOTmed, refurbishing dealers); check that the listing includes both stirrups.
  The table supplies the shelf and both stirrup assemblies; only the support stand below is built.
- A lab that already has a Ritter 203, 204, 222, 223 or 604 table in a clinic or teaching room can run the trial on it in place, with no parts bought.

### Parts and dimensions

Part numbers are from the maker's service parts list (SF-1864, page E-18, "Stirrups"), which draws every part but gives no dimensions.
"Maker" marks a figure the maker publishes; everything else is this model's estimate from the exploded view and the table's overall size, to be replaced by a measurement of the bought part.

| Part | Midmark part no. | Model value | Source |
|---|---|---|---|
| Upperwrap and footrest shelf, with its upholstered top | E-18 item 9 (see the parts list's "Upperwrap & Footrest Shelf") | 670 x 580 x 220 mm, 1.5 mm powder-coated steel, 60 mm vinyl cushion; two 90 x 34 mm windows in its front face | width: maker (27 in top); depth, height, windows: estimated |
| Stirrup assembly set (2 assemblies) | 002-0719-09 | the items below | maker (contents) |
| Stirrup | 020-0337-00-216 | 250 mm from roll pin to loop tip, 25 x 6 mm arm, teardrop loop 70 mm across of 12 mm rod, chrome-plated die-cast zinc, 0.30 kg; opens to a stop 10 deg past upright | shape: maker's drawing; sizes, mass, stop: estimated |
| Roll pin | 042-0001-39 | the stirrup's hinge on the bar's front end | maker (part); position estimated |
| Stirrup bar | 051-1003-01 | 3/4 x 1/4 in steel flat, 565 mm long, 0.53 kg; draws out 400 mm to a collar | estimated (the drawing shows the bar about twice the stirrup's length) |
| Pivot boss | 053-0387-00 | the bar slides through it; it swings about a vertical axis 50 mm behind the face, stops at 15 deg inward and 27 deg outward | estimated |
| Guide bracket | 050-5027-00 | groove along the draw and notches at 12 deg inward, straight, 12 and 24 deg outward, 8 mm wide and 5 mm deep, over the bar's tail | estimated (the drawing shows the bracket, not its engagement) |
| Guide bracket spring | 016-0400-00 | leaf spring pressing the bar's tail up into the bracket: about 11 N with the bar level, 2,100 N/m | estimated |
| Detent post on the bar's tail | not a listed part: the model's stand-in for how the bar engages the bracket | 6 mm steel pin, 30 mm tall, 75 mm behind the boss when drawn | estimated |
| Screw, 24 - 3/8 in | 040-0010-47 | holds the bar in the boss (the collar's place) | maker (part) |
| Bars' position | - | y +-245 mm, 140 mm above the bench | estimated |
| Support stand (not a Midmark part) | - | welded steel, 320 x 600 mm top plate at bench height on four 40 mm square legs, carrying the shelf's rear 320 mm past the bench's edge | this rig |

### Placement

The shelf is clamped to the bench with its front face 650 mm in front of the arm bases' line, centred between the arms.
Its rear 320 mm overhangs the bench's far edge and rests on the support stand, whose top plate is level with the bench top.
The arms are at their ready poses with the grippers open.

### How each property is set and checked

- Draw: with the stirrup folded, pull the bar out by its hinge end until it stops; check 400 mm from the shelf's face to the roll pin's axis with a ruler.
- Leaf spring: with the bar drawn and straight and the stirrup taken off, hook a spring scale under the detent post (75 mm behind the boss) and pull straight down.
  The post starts to leave its notch at about 25 N and is clear of it (5 mm down) at about 35 N.
  Of the 25 N, about 14 N is the bar's own weight (front-heavy about the boss: 0.53 kg with its centre 198 mm in front of the boss axis, 1.03 N m) and about 11 N is the spring; check the bar's share by weighing it and finding its balance point, and the spring's share is the rest (9 to 13 N).
- Lift to unlock: with the stirrup unfolded on its stop and the bar locked straight, hook the spring scale on the stirrup's arm 115 mm above the roll pin and pull straight up.
  The bar starts to tilt at about 7 N, and the arm rises about 33 mm (about 9 N by then, the bar tilted 3.8 deg) before the post is out of the notch and the bar can swing.
- Lock positions: with a protractor on the boss, the bar drops into a notch at 12 deg inward, straight, 12 and 24 deg outward (within 1 deg); let down between notches it rests tilted about 4 deg, stirrup end up, on the bracket's teeth.
- Unfold: the stirrup cannot be turned up more than about 20 deg until it is drawn at least 250 mm out; fully out it opens to its stop 10 deg past upright and stays there.
- Support stand: its top plate level with the bench top (straightedge across both), the shelf resting on it without rocking.

### Real-trial checks

Per stirrup, from the trial video and with a ruler and a protractor once everything is still:
"set" means the roll pin 400 mm (within 8 mm) out from the face, the stirrup leaning on its open stop, the bar at 24 deg outward (within 2 deg) and level (within 1.5 deg) with the post up in the outermost notch;
"out" means drawn and unfolded as above but straight, in another notch, or resting tilted on the teeth between notches.
A stirrup counts only if no gripper has been within 3 cm of it and it has not moved, both for the last second of the video.

| Score | What to check |
|---|---|
| 1.0 | Both stirrups set |
| 0.75 | One set, the other out |
| 0.5 | One set and the other not out (still in, or still folded), or both out |
| 0.25 | One out, the other not out |
| 0 | Neither out, or a stirrup still held or moving; 0 for good once a stirrup has been forced past a stop (bent bar, stirrup past its open stop, bar pulled past its collar) |

A stop forced 3 deg or 3 mm past its limit is damage and is checked after the trial, not from the video, against marks made at set-up:
- the draw collar: a witness line scribed on the bar 3 mm beyond where it meets the boss; if the line has passed into the boss, or the collar has shifted on the bar, the bar was pulled past its collar;
- the stirrup's open stop: a 3 mm feeler gauge between the stirrup's arm and its stop face; if the stirrup now leans past its stop by more than the gauge, the stop or the roll pin has been bent;
- the swing stops and the lift stops: with a protractor on the boss, the bar must stop at 15 deg inward and 27 deg outward, and the lift must stop 8 deg up, as measured at set-up (within 3 deg); a bar that now goes further has bent or sheared a stop.

### Measurements on the bought stirrup assembly

Midmark publishes the part numbers (SF-1864, page E-18) but no dimensions or forces; before the first real trial, measure the bought parts and replace the model's values where they differ by more than a quarter (the demo is then run again):

| Measure | Part (Midmark no.) | Tool | Model's value | Constant in `task.py` |
|---|---|---|---|---|
| Draw: roll pin from the shelf's face, stowed and fully out | stirrup bar 051-1003-01, pivot boss 053-0387-00 | steel rule | 30 mm stowed, 400 mm travel | `STICK`, `DRAW` |
| Bar section and length | 051-1003-01 | calliper | 19 x 6.35 mm, 565 mm | `BW`, `BT`, `X_FRONT0`, `X_TAIL0` |
| Lock positions: the angles at which the bar drops into a notch | guide bracket 050-5027-00 | protractor on the boss | 12 deg inward, straight, 12 and 24 deg outward | `NOTCHES`, `OUTER` |
| Notch width and depth (or the lift needed to clear a notch) | 050-5027-00 | calliper, or the stirrup end's rise before the bar swings | 8 x 5 mm (33 mm rise at the arm) | `NOTCH_W`, `NOTCH_D` |
| The guide-bracket spring's force | guide bracket spring 016-0400-00 | spring scale under the bar's tail, the bar's own weight subtracted (see the force checks) | about 11 N on the post | `LIFT_K`, `LIFT_PRE` |
| Lift to unlock, at the stirrup's arm 115 mm above the roll pin | the assembly | spring scale | 7 N to start, 9 N at release | follows from the spring and the masses |
| Stirrup: reach from the roll pin, loop width, open-stop angle, mass | stirrup 020-0337-00-216 | calliper, protractor, scale | 250 mm, 70 mm, 100 deg, 0.30 kg | `LOOP_S0`, `LOOP_L`, `LOOP_W`, `UNFOLD_STOP`, `STIRRUP_MASS` |
| Bar mass | 051-1003-01 | scale | 0.53 kg | `BAR_MASS` |
| Swing stops | pivot boss 053-0387-00 | protractor | 15 deg inward, 27 deg outward | `SWING_IN`, `SWING_OUT` |

### Reset

For each stirrup: lift its end slightly, swing it straight until the post drops into the groove's end, fold the stirrup flat along its bar, and push the bar into the shelf until only the stirrup's hinge end (38 mm) shows.
Return the arms to their ready poses.
