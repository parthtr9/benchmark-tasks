# Close a jaw coupling: spider into the fixed hub, loose hub turned to mesh and slid to its line

A small drive is bolted to the bench.
A worm gear reducer's output shaft carries one hub of a Lovejoy L-100 jaw coupling; in line with it ends a 5/8 in driven shaft that turns freely in two pillow blocks.
The coupling's second hub sits on the driven shaft's key, slid back from its working place, which a line scribed round the shaft marks.
The coupling's Hytrel spider stands in a holder beside the drive.
The robot must put the spider between the fixed hub's jaws, bring the loose hub's jaws into the spider's free gaps and slide the hub to its line.
The job stops before the loose hub's set screw is tightened.

Why it matters.
A jaw coupling is the commonest way to join a drive to a driven shaft, and its spider is the part that wears: it is renewed on a schedule and after every overload.
The maker's installation guide (Lovejoy, L-Line jaw couplings, 2012) gives the steps: hubs onto the shafts over their keys, "insert the spider into one of the hubs", bring the hubs to the "G" spacing, then tighten the second hub's set screw.
Industrial machinery mechanics (O*NET 49-9041.00: "Repair or replace broken or malfunctioning components of machinery or equipment."; "Reassemble equipment after completion of inspections, testing, or repairs."; 446,900 US workers), machinery maintenance workers (49-9043.00: "Install, replace, or change machine parts and attachments, according to production specifications."; 59,700) and millwrights (49-9044.00; 40,700), 547,300 in all, do this on conveyors, pumps, fans and mixers (head counts and task statements under "Who does this job, and how many").

What the task tests.

- A six-legged part between three jaws.
  The spider's legs are 10.9 mm thick and the gaps between a hub's jaws take two legs and the other hub's jaw.
  The spider only goes in along the shaft, legs between jaws, and once it is in there is a jaw either side of every leg and a hub face behind it, so it can be held only by the tips of its legs, from outside the coupling's diameter.
- A phase found by contact.
  The loose hub starts 30 degrees out of phase.
  Pushed straight in, its jaw tips stop on the fixed hub's jaw tips; turned part of the way they drop 2.5 mm onto the spider's legs and stop again; only within about a degree of the right phase do they drop into the gaps.
- A shaft that turns when touched.
  The driven shaft is free in its bearings and the hub is keyed to it, so pushing the hub does not turn it and turning the shaft does not push it: one hand has to turn the shaft while the other keeps the hub pressed forward.
- A slide to a line.
  The jaws are as long as the hubs' catalogue spacing (18.0 mm), so a hub pushed fully home stops with its jaw tips on the other hub's face, with its rear face on the scribed line.
  A hub left short is a coupling that runs on the tips of its jaws.

Nearest tasks, and the physical difference:
"Seat a keyed coupling and prove drive transfer" lowers a printed coupling over two unequal drive dogs; "align a coupled shaft" works on a coupling that is already closed; meshing a flexspline, keying a pulley onto a shaft and meshing a gear train each bring one toothed or keyed part onto a fixed one.
Here a loose six-legged spider goes between two three-jaw hubs, one of them on a shaft that turns when touched, the phase is found by contact, and the hub is slid to the catalogue spacing.

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
  --instruction "Couple the gear reducer to the driven shaft: the spider between the jaws of both hubs and the loose hub at the scribed line on its shaft." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/jaw-coupling-mate-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.
There are no seeds: `object_jitter=0`, one fixed layout.

## Scripted demo

```bash
python demo.py task.py demo_out                               # full score: writes demo_detail.mp4 (the reviewer's video), demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=no_spider python demo.py task.py out --no-record  # the hubs are closed with the spider left out
DEMO_VARIANT=no_turn python demo.py task.py out --no-record    # the shaft is not turned
DEMO_VARIANT=on_legs python demo.py task.py out --no-record    # the shaft is turned only part of the way
DEMO_VARIANT=short python demo.py task.py out --no-record      # the hub is not pushed home
DEMO_VARIANT=unseated python demo.py task.py out --no-record   # the spider is let go between the hubs
python demo.py task.py --ik                                    # no simulator: IK error of every step of every variant
```

1. The right hand takes the spider out of its holder by the tips of its two level legs, lowers it into the space between the fixed hub's jaws and the driven shaft's end, and moves it along the shafts into the jaws.
   It lets go for a moment so the spider squares itself between the jaws, takes hold again and presses it home against the hub's face.
   Meanwhile the left hand comes down on the driven shaft's free end and closes on it.
2. The right hand puts its closed fingertips behind the loose hub and pushes it along the shaft until its jaw tips stand on the fixed hub's jaw tips, and keeps a light push on it.
3. The left hand swings round the shaft's axis like a crank, turning the shaft and the hub on its key: quickly to 15 degrees, by which point the jaws have dropped onto the spider's legs, then at 2.5 degrees per second.
   The jaws slide round on the legs and drop into the free gaps; the right arm's own joint readings show its hand jump forward as the hub drops in (4 mm in half a second), and the turn stops there.
   If no drop is felt the search goes on to 60 degrees, which covers any slip of the shaft in the grip.
   The gaps are open to the jaws for about 2 degrees of turn, so a fixed fast swing can pass over them.
4. The right hand pushes the hub home, and both hands let go and back off.

Every target comes from the task file's fixed layout and the clock, and one decision comes from the arms' own joints: the phase search stops when the right arm's measured tool point jumps forward.
The demo reads no object pose, no joint of the coupling and no task state (its debug trace does).

### What to watch in the demo video

`demo_detail.mp4` shows the whole bench (the top camera) and the coupling close up, side by side; each frame is labelled with its step (20 steps a second).
The close view is a recording camera on the coupling's left, a little above the shafts' axis, where no arm passes; it is for the video only, and the policy's cameras, the physics and the scorer are the same as in every run.

| Steps (time) | What happens |
|---|---|
| 0-100 (0-5 s) | Both arms move in; the left hand closes on the driven shaft's free end |
| 100-180 (5-9 s) | The right hand carries the spider by its two level leg tips and lowers it between the driven shaft's end and the fixed hub's jaws |
| 180-320 (9-16 s) | The spider goes along the shafts into the fixed hub's jaws, is let go so it squares itself, taken again and pressed home; its six legs then show between the three jaws |
| 360 (18 s) | The right hand lets go of the spider (0.25) |
| 420-530 (21-27 s) | The right hand's closed fingertips push the loose hub along its shaft until its jaw tips stand on the fixed hub's jaw tips, then on the spider's legs (2.5 mm further) |
| 540-655 (27-33 s) | The left hand turns the shaft slowly; at about 29 degrees the loose hub's jaws face the free gaps and drop in (in the close view the hub jumps forward) |
| 655-720 (33-36 s) | The right hand pushes the hub home: its rear face reaches the scribed line, the jaws interleave with the spider's legs between them |
| 740-800 (37-40 s) | Both hands let go and back off; 0.75 once they are clear, 1.0 a second later |

Results in simulation (Isaac Sim 6.1, `object_jitter=0`): see the table at the end of this file.

## Scoring

Evaluated every control step; the score at the end of the episode counts.
Everything is measurable in a real trial with a rule, a feeler gauge and the trial video.

| Score | End state |
|---|---|
| 0.25 | The spider sits between the fixed hub's jaws: its centre within 1.5 mm of the shafts' axis, within 5 degrees of square, its near face no more than 1 mm proud of the jaw tips, grippers at least 3 cm clear of it |
| 0.5 | That, and the loose hub's jaw tips are 3 mm or more past the spider's near face, in its free gaps, with the grippers at least 3 cm clear of the loose hub and the driven shaft |
| 0.75 | That, and the loose hub's rear face is within 1 mm of the scribed line |
| 1.0 | That, with both grippers at least 3 cm clear of the spider, the loose hub and the driven shaft, and nothing moving for 1 second (in simulation no tracked point moves more than 1 mm; in a real trial no visible motion in the last second of the video) |
| 0 | Every other state, including the two hubs closed on each other with the spider left out |

Which arm does what is not scored.
Both machines are bolted down and cannot be moved, so there is no rule about moving them.

## Runtime

A trial is at most 1,200 steps (1 minute at 20 Hz).
The demo's step count and wall time are in the results table.

## Fidelity

- Full size, rigid, built from boxes; nothing is detected or scripted, and no force is applied by code.
- Hubs: 16-sided with their corners on the real 32.26 mm radius. Each jaw is a wedge whose flanks are parallel to the legs beside it. Jaw tips are flat and square, as in the maker's photos of its hubs.
- Spider: a rigid body with 0.5 mm of play per side between each leg and the jaws beside it. The real Hytrel spider (55 Shore D) is a close fit and gives a little.
- The spider's leg edges carry a 0.6 mm 45-degree facet. The maker's photo of its moulded spiders shows leg edges rounded to about 1 mm; the facet lies inside that radius. It is the only lead-in in the model, and the radius is an estimate: measure the bought spider.
- The fixed hub stands with one jaw straight up, so the two level legs rest on the jaws below them and the rigid spider settles by its play (0.5 mm). With a jaw straight down it would settle 1.0 mm, which a close-fitting spider does not do.
- The loose hub slides on a joint carried by the driven shaft's free joint, which is the key: the hub turns with the shaft and slides along it. The key starts engaged. Sliding takes 1.4 N (0.70 kg at friction 0.2); the bearings' drag is 0.02 N m.
- The driven shaft's end stands 19.5 mm from the fixed hub's jaw tips, because a spider can only enter a hub's jaws along the shaft and needs its own width (15.49 mm) and 2 mm a side to be brought in. The loose hub therefore overhangs its shaft's end when coupled and sits on 15.55 mm of shaft, the shaft's own diameter, which is the engagement the maker's guide asks for. This is why the coupling is an L-100: a smaller L-075 hub would be left on 5.6 mm of shaft.
- Not modelled: the set screws (see "The set screws" below), the key as a separate part, the spider's raised spacing dots, the reducer's inside.
- Estimates to measure on the bought parts: the spider ring's outer radius (18.5 mm), the radius where the jaws start (19.5 mm), the leg-edge radius, the bearings' drag, the reducer's shaft height on its riser.

### The set screws

The maker's guide ends the fitting with "tighten second hub on shaft": each L-100 hub has one 5/16-18 set screw over its keyway, tightened to 150-165 in-lb (17-19 N m, the guide's table 1 for L099 and L100 hubs with a set screw 5/16 in long or more).
The task models the job up to the step before that one: spider in, loose hub turned to mesh and slid to the catalogue spacing (G, 18.0 mm), which is where the guide's next step tightens the screw.
It leaves the screw out for two reasons, both measured, not assumed:

- Threads do not mesh in this simulator at this size.
  A 5/16-18 screw has a 1.41 mm pitch and, in a standard class 2A/2B fit, 0.02 to 0.15 mm of axial play (from the fit's pitch-diameter allowance and tolerances), below the 0.5 mm contact offset every part here uses.
  When we modelled a coarser thread (2.21 mm pitch, 0.12 mm play) the nut rode the thread crests and advanced 0.01 to 0.09 mm a turn instead of 2.21, so a screw turned in would not advance as a real one does.
- The torque is beyond the arm.
  17 N m needs about 190 N at the end of a 5/32 in hex key's 90 mm long arm (or 170 N on a 100 mm T-handle).
  Measured in this arena, the arm pushes about 100 N at most along a forward-pointing gripper and about 20 N sideways from a down-pointing one, and the pinch holding the key would have to resist the same torque.

So the 1.0 state is the coupling closed at its catalogue spacing with the screw still backed out, as the guide has it just before its last step; the axial position the screw would lock is the one the scorer checks (the rear face within 1 mm of the scribed line).
No step that secures the hub without a thread exists on this coupling: L100 finished-bore hubs are sold only with a set screw over the keyway (no clamp-hub version of the L-100), the key already stops the hub turning and is engaged from the start, and the retaining ring in the maker's guide belongs to the LS-type coupling, a different hub and spider.

Real-trial check for the screw: before the trial the loose hub's set screw is backed out until its tip is below the bore (the hub slides under about 1.4 N, checked with the spring scale); the person checks after the trial that it is still backed out, so a score cannot come from a screw tightened by the robot or the setup; tightening it to 17-19 N m with a torque wrench and a 5/32 in hex bit is done by the person after the trial, before the drive is run, and is not scored.

## Real-world build

### Cost

The coupling side is ordinary catalogue hardware: two Lovejoy L100 x 5/8 in hubs at about $22 to $65 each, the L099/100 Hytrel spider at about $89 to $100 (Zoro, HVH Industrial; the nitrile spider for the same hubs is about $27 to $36), two UCP202-10 pillow blocks, a 5/8 in keyed shaft and a base plate (prices seen November 2026; they vary by supplier).
The fixed side's worm gear reducer is the costly part: a Boston Gear 713 lists at about $850.
The task never turns it; it only holds the fixed hub on a shaft that does not turn by hand.
So a lab without a reducer builds the fixed side as a housing of the reducer's size (73 x 108 x 118 mm, printed or cut from plywood, on its 31 mm riser) carrying a 120 mm length of the same 5/8 in keyed shaft, held in it by two set-screw shaft collars or a pillow block with its set screws tightened and pinned, so it cannot turn or slide.
The fixed hub then goes on that stub exactly as on the reducer's output (face flush with the shaft's end, one jaw straight up, set screw at 17-19 N m).
With that, the whole rig costs about $200 to $300.


| Part | What to buy or make |
|---|---|
| Hubs | 2 x Lovejoy L100 x 5/8 in, 3/16 x 3/32 in keyway, sintered iron (68514411510) |
| Spider | Lovejoy L099/100 open-centre Hytrel spider (68514411486) |
| Fixed side | A worm gear reducer with a 5/8 in output shaft, 30:1 or higher so it does not back-drive (Boston Gear 713 size: housing 73 x 108 x 118 mm, 5.4 kg), on a steel riser that brings its shaft 90 mm above the bench |
| Driven side | 5/8 in keyed steel shaft, 218.5 mm long, in two UCP202-10 pillow blocks 50 mm apart on a 60 mm steel base frame, the bearings' own set screws tightened so the shaft turns in them and cannot slide |
| Holder | A 60 x 80 x 6 mm board with two 10 x 46 x 24 mm blocks 16.5 mm apart |

- Bolt both machines to the bench with the shafts in line within 0.4 mm (the guide's limit for this coupling) and the driven shaft's end 19.5 mm from the fixed hub's jaw tips.
- Fixed hub: on the reducer's shaft over its key, face flush with the shaft's end, one jaw straight up, set screw tightened to 17-19 N m (150-165 in-lb, the guide's table 1 for L-type hubs).
- Loose hub: on the driven shaft over its key, set screw backed out. It should slide under about 1.4 N: check with a spring scale hooked on its rim and pulled along the shaft; clean and oil the shaft lightly if it takes more.
- Scribed line: push the loose hub home against the fixed hub with a spider in place, scribe round the shaft at the hub's rear face, and pull the hub back 41.5 mm.
- The driven shaft should turn under about 2.5 N at its surface (0.02 N m): check with a cord wound once round the shaft and a spring scale.
- Mark the shaft and the nearer pillow block so the start phase (30 degrees from meshing) can be set again.
- Nothing here is powered: the reducer has no motor fitted.

## Results

Isaac Sim 6.1 with the yam_arena embodiment, `object_jitter=0`, from each run's `result.json`.

| Run | What it does | Score | Steps | Hub from its line | Spider seated |
|---|---|---|---|---|---|
| demo | Full sequence | 1.0 | 804 | 0.00 mm | yes |
| `no_spider` | The spider is left in its holder; the hub is turned and pushed home. The hubs close on each other, jaw between jaw, with no spider | 0.0 | 847 | 0.00 mm | no |
| `no_turn` | The shaft is not turned. The hub stops with its jaw tips on the fixed hub's jaw tips, 18 mm short of the line | 0.25 | 688 | 18.03 mm | yes |
| `on_legs` | The shaft is turned 18 of the 30 degrees. The jaws drop 2.5 mm onto the spider's legs and stop there, 15.5 mm short | 0.25 | 684 | 15.49 mm | yes |
| `short` | The hub is not pushed home after its jaws drop into the gaps | 0.5 | 801 | 7.65 mm | yes |
| `unseated` | The spider is let go between the hubs without being moved into the fixed hub's jaws. It falls out of the coupling and the hubs close without it | 0.0 | 1001 | 0.00 mm | no |

This version's runs (batch 9fba166f): the phase search stopped at 29.1 degrees in the demo and in `short`, and searched on to 60 degrees where there was nothing to drop into (`no_spider`, `unseated`; hence their longer runs).
The demo's largest IK error is 0.08 mm and the two arms never come closer than 46.4 mm.
The demo takes 804 of the 1,200 steps (40 s of simulated time; 2 to 3 minutes of wall time on one GPU, most of it start-up).

Dream-1, one run at the 1,200-step limit on this version (batch 34f7bd1f; the same batch re-ran the demo and every variant with the scores above): 0.0.
It brought its right hand over the spider in its holder and held it there for the rest of the minute; the left arm barely moved, the spider stayed in the holder and the loose hub was not touched.

## Prepared for revision

This section gathers, in one place, the sources, the real-trial measurements and the head-count argument that a review is most likely to ask for.
Nothing in it changes the task file, the scorer, the demo or the instruction.
Where a value is an estimate or has not been checked against the bought part, it says so.

### Dimensions and sources

| Item | Value used | Source |
|---|---|---|
| Hub, Lovejoy L100 x 5/8 in, 3/16 x 3/32 in keyway (68514411510) | 64.52 mm across, 35.05 mm through the bore, 53.09 mm long with jaws (jaws 18.04 mm), sintered iron | Lovejoy (Timken) CAD listing for the L-100 hub (cad.timken.com, item 68514447123, 15 mm bore version of the same body); distributor listings for 68514411510 give the 5/8 in bore and 3/16 x 3/32 in keyway, e.g. https://www.motion.com/products/sku/00515864 and https://kscdirect.com/itemDetail/5294990/l100-hub-5-8-3-16x3-32kw/ |
| Hub mass | 0.70 kg | 1.56 lb listed for the 15 mm bore; 0.9 mm larger bore, so slightly less; weigh the bought hub |
| Hub spacing G | 18.03 mm | L-100 hub CAD listing (0.71 in); the installation guide's table 4 rounds it to 0.750 in |
| Spider, Lovejoy L/AL099/100 open-centre Hytrel (68514411486) | 64.52 mm across the legs, legs 10.92 mm thick, 26.16 mm hole, 15.49 mm wide, 55 Shore D, 27 g | Lovejoy (Timken) CAD listing 68514411486; distributor listings give 0.61 in wide, 2.54 in OD, 1.03 in ID, 55D, e.g. https://www.zoro.com/lovejoy-jaw-coupling-insert-hytrel-spider-68514411486/i/G316035427/ |
| Spider ring outer radius | 18.5 mm | Estimate from the maker's photo; measure the bought spider with calipers |
| Radius where the jaws start | 19.5 mm | Estimate; measure the bought hub with calipers |
| Spider leg-edge radius | modelled as a 0.6 mm 45-degree facet inside a radius of about 1 mm | Estimate from the maker's photo in the L-Line installation guide; measure with a radius gauge |
| Leg-to-jaw play | 0.5 mm per side (model) | Model choice for the simulator; the real spider is a close fit (see "Fits and forces" below) |
| Pillow blocks, UCP202-10, 2 off | 5/8 in bore, shaft centre 30.2 mm (1 3/16 in) above the foot, foot 127 x 38 mm, bolt holes 95 mm (3 3/4 in) apart, cast iron | Distributor listings, e.g. https://www.applied.com/c-ami-bearings-inc--ucp202-10/p/101699951 |
| Gear reducer, Boston Gear 713 size, 30:1, 5/8 in output shaft | housing 108 x 73 x 118 mm; modelled as a fixed block | Distributor listings for the 713 size give a 4-1/4 x 2.88 x 4.66 in housing, 5/8 in output shaft and 1.33 in centre distance. The F713-30-B5 units (56C motor flange and quill input, e.g. https://www.applied.com/c-brands/c-altra-industrial-motion/c-boston-gear/f713-30-b5-j/700-Series-C-Face-Quilled-Reducer/p/100992030) are listed at about 13 lb (5.9 kg); the model shows the plain version with a solid input shaft, which one distributor lists as 713-30-G at 12 lb (5.4 kg), not confirmed on the listing page itself. The reducer is bolted down, so its mass has no effect on the trial. Confirm the part number with the supplier and weigh the bought unit |
| Driven shaft | 5/8 in (15.875 mm) keyed steel shaft, 218.5 mm long, 3/16 in keyway, 0.32 kg | Cut from 5/8 in keyed shaft stock (1045 or similar steel); 3/16 in square key at least 60 mm long at the coupling end, so it stays under the hub over its whole 41.5 mm travel (57 mm is needed) |
| Base frame under the pillow blocks | steel block 60 mm high, 88 mm along the shaft, 140 mm across | Make: a 60 x 88 x 140 mm steel block or welded box, top faced flat, drilled and tapped for the two pillow blocks' bolts (95 mm centres, blocks 50 mm apart centre to centre) and for two bench bolts |
| Riser under the reducer | steel plate 110 x 130 mm, 31 mm in the model | Make: thickness = 90.2 mm minus the bought reducer's base-to-output-shaft-centre height (measure it), drilled for the reducer's mounting holes; shim to align the shafts within 0.4 mm |
| Spider holder | 60 x 80 x 6 mm board with two 10 x 46 x 24 mm blocks glued 16.5 mm apart | Make: hardwood or plywood; the slot is the spider's width plus 1 mm |
| Scribed line | 15.55 mm back from the driven shaft's end, 41.5 mm ahead of the parked hub's rear face | Scribe it with the hub pushed home against the fixed hub with a spider in place (see "Real-world build") |

### Fits and forces: what is modelled, and how each is measured in a real trial

Every stated force in the setup is a property of a bought part, measured with a spring scale or force gauge before the first trial and checked again at each reset.
If a measured value differs from the model's, the model's value is changed to the measurement and the demo and the Dream-1 run are repeated.

| Quantity | Model | How it is measured | What changes if the reading differs |
|---|---|---|---|
| Loose hub sliding on its shaft and key, set screw backed out | 1.4 N, the hub's weight at friction 0.2 (Coulomb friction: the slider's drive force is capped at 1.4 N) | Spring scale (0 to 10 N) hooked on the hub's rim, pulled along the shaft at a steady slow speed; three pulls each way; record the peak | Set the slide friction to the measured peak and rerun. In simulation the YAM arm pushes tens of newtons when commanded past contact, so a reading of a few newtons does not change the task. A reading over 10 N means a burr or a tight key: dress the key and keyway and oil lightly until it slides under 5 N |
| Driven shaft turning in its two bearings | 0.02 N m (estimate) | Cord wound once round the shaft, spring scale on its end, pulled tangentially: 0.02 N m reads about 2.5 N at the 7.94 mm shaft radius; three pulls | Set the turning friction to the reading and rerun |
| Fixed hub on the reducer's output | does not turn (fixed) | Try to turn the fixed hub by hand, then with a torque wrench to 2 N m: it must not move | If it moves, clamp the reducer's input shaft with a shaft collar bolted to a bracket, so the output cannot turn |
| Spider into a hub's jaws | rigid spider, 0.5 mm play per side, no insertion resistance beyond friction (0.4 hub, 0.5 spider) | Not yet measured. Clamp a hub in a vise, jaws up, set the spider on its jaws legs between jaws, push it home with a push-pull force gauge on a flat plate at a few mm per second, three trials, record the peak; measure leg thickness and the gap between jaws with calipers at three places | If the peak is a few newtons or less, the model's free fit stands. If it is higher, add matching resistance to the spider's last millimetres of travel (a sprung or friction stop in the task file) and rerun. In simulation the YAM gripper holds a 2 kg load (about 20 N) without slipping, so a peak well under that is within the robot's reach; a higher peak needs the demo rerun with the measured resistance |
| Loose hub's jaws into the spider's free gaps | 0.25 mm play on the tightest flank, plus the leg facets | Same gauge, pushing the loose hub by its rear face once its jaws face the gaps | As above |

The real Hytrel spider is a close fit and gives a little; the model's play is what the simulator needs to keep rigid flank-on-flank contacts stable.
The insertion force is the one value above that has not yet been measured on the bought parts.

### Who does this job, and how many

Verified head counts, employment in 2025:
BLS Occupational Outlook Handbook, "Industrial Machinery Mechanics, Machinery Maintenance Workers, and Millwrights" (https://www.bls.gov/ooh/installation-maintenance-and-repair/industrial-machinery-mechanics-and-maintenance-workers-and-millwrights.htm).
Task statements from O*NET OnLine (https://www.onetonline.org/link/summary/49-9041.00, .../49-9043.00, .../49-9044.00, .../49-9071.00).

| Occupation (O*NET) | US workers, 2025 | Task statements on O*NET that cover fitting a coupling's parts (quoted) |
|---|---|---|
| Industrial machinery mechanics (49-9041.00) | 446,900 | "Repair or replace broken or malfunctioning components of machinery or equipment."; "Reassemble equipment after completion of inspections, testing, or repairs."; detailed work activity "Replace worn, damaged, or defective mechanical parts." |
| Maintenance workers, machinery (49-9043.00) | 59,700 | "Install, replace, or change machine parts and attachments, according to production specifications."; "Reassemble machines after the completion of repair or maintenance work." |
| Millwrights (49-9044.00) | 40,700 | "Replace defective parts of machine, or adjust clearances and alignment of moving parts."; "Assemble and install equipment, using hand tools and power tools." |
| Total of the three | 547,300 | |

General maintenance and repair workers (49-9071.00, 1,621,800) have the task statements "Assemble, install, or repair wiring, electrical or electronic components, pipe systems, plumbing, machinery, or equipment." and "Clean or lubricate shafts, bearings, gears, or other parts of machinery."
They are not needed for the 500,000 total.

The argument for this specific operation:

- The three machinery occupations above, 547,300 workers, keep production machinery running.
  Many motor-, gearbox- and pump-driven machines have a flexible coupling between the drive and the driven shaft, and the jaw coupling is a common, inexpensive type for small drives.
- A jaw coupling's elastomer spider is its wearing part.
  The maker's L-Line installation guide gives its fitting as part of installing the coupling ("insert the spider into one of the hubs", then bring the hubs to the "G" spacing and tighten the set screw).
  The spider is renewed when its legs wear, so the operation recurs over the coupling's life.
- Replacing a worn spider is exactly the O*NET task "Repair or replace broken or malfunctioning components of machinery or equipment" followed by "Reassemble equipment after completion of inspections, testing, or repairs" for 49-9041, and "Install, replace, or change machine parts" and "Reassemble machines" for 49-9043.

Not verified: no O*NET task statement names couplings or spiders, and no source found gives the share of these workers who fit jaw couplings, or how often.
The claim that these workers do so regularly rests on the jaw coupling being a common drive coupling on small machinery and on the spider being its wearing part, not on a survey.

### Real-trial check for each score

All checks use a steel rule, calipers with a depth rod, a feeler gauge set and the trial video.
The legs of an L-100 spider reach the hubs' outside diameter (64.52 mm, the same as the hubs), so its legs stay visible between the jaws from outside, before and after the loose hub closes; nothing scored is hidden inside the coupling.

| Score | Simulator (what the scorer measures) | Real trial (what the person checks) |
|---|---|---|
| 0.25 | The spider's centre within 1.5 mm of the shafts' axis, within 5 degrees of square, its near face no more than 1 mm proud of the jaw tips; both grippers at least 3 cm from it | The loose hub is still back, so the fixed hub's jaws and the spider are open to view. Depth rod of the calipers from a jaw tip to the spider's near face at each of the six legs: no leg face more than 1 mm proud of the jaw tips (a fully seated spider sits 2.5 mm below them). Straightedge laid along the hub's outside diameter over each leg tip, feeler gauge under it: no leg tip more than 1.5 mm proud of the hub's diameter (this bounds the spider's centre to 1.5 mm off the axis). Square: the depth readings at opposite legs differ by less than 5.6 mm (5 degrees across 64.5 mm); between the jaws the spider cannot tilt anywhere near this far. Video: no gripper within 3 cm of the spider at the end |
| 0.5 | That, and the loose hub's jaw tips 3 mm or more past the spider's near face, in its free gaps; the grippers at least 3 cm from the loose hub and the driven shaft | Rule from the scribed line to the loose hub's rear face: distance d. With the spider's near face s mm below the fixed hub's jaw tips (the 0.25 reading), the loose hub's jaw tips are (18.0 - d - s) mm past the spider's near face, so 0.5 needs d no more than 15.0 - s mm; for a fully seated spider (s = 2.5 mm) that is d no more than 12.5 mm. The hub can only get this far with its jaws in the spider's free gaps: out of phase, its jaws stop on the fixed hub's jaw tips (d = 18.0 mm) or on the spider's legs (d = 15.5 mm). The interleaved jaws, each with a spider leg either side, are seen directly from the side |
| 0.75 | That, and the loose hub's rear face within 1 mm of the scribed line | Rule or feeler gauge: the loose hub's rear face within 1 mm of the scribed line. Cross-check: the gap between the two hub bodies is the catalogue G, 18.0 mm (calipers across the coupling at its outside diameter) |
| 1.0 | That, and for the whole last second both grippers at least 3 cm from the spider, the loose hub and the driven shaft and no tracked point (hub travel, a rim point, the spider's pose) moving more than 1 mm | As 0.75, and in the video both grippers are at least 3 cm from the spider, the loose hub and the driven shaft (a 3 cm block next to the coupling as a scale reference), and over the last second (20 frames) the scribed line, an index mark on the hub's rim and the spider's legs show no visible motion (the simulator's 1 mm limit is its own measure; a person sees only that nothing moves) |
| 0 | Every other state | Anything else: in particular, the hubs closed with no spider between them (the jaws of both hubs interleave with nothing between, visible from the side), or the spider still in the holder or on the bench |

The scorer in simulation uses the same quantities: the hub's travel on its shaft (the rule reading from the line), the spider's pose relative to the fixed hub, and the fingertips' distance from the parts' surfaces.

### Other likely questions

- The driven shaft ends 19.5 mm short of the fixed hub's jaw tips, so the loose hub overhangs it when coupled.
  This is because both shafts are fixed on the bench and the spider can only enter a hub's jaws along the shaft.
  The maker's guide allows overhung hubs and asks for hub engagement equal to the shaft's diameter, which the L-100 hub gives (15.55 mm of a 15.875 mm shaft, 98%).
- Which arm does what is not scored, and one hand gripping the loose hub itself to turn and slide it is allowed.
- The loose hub's set screw is not tightened: see "The set screws" under Fidelity for why (threads below the simulator's contact resolution, 17-19 N m beyond the arm) and how a real trial checks it.

## Revision 2: what was asked and what changed

The first version was not accepted (revisable). The reviewer wrote:

> The demo should show more detail

What changed for that:

- The demo video is now `demo_detail.mp4`: the whole bench and the coupling close up, side by side, each frame labelled with its step.
  The first version's demo video was the top camera at 320 x 176 pixels, in which the coupling is a few pixels across and is under both arms for most of the demo.
  Now the cameras are recorded at 1280 x 720.
  A close camera stands on the coupling's left, where no arm passes; there the spider seating in the jaws, the hub's jaws resting on its legs while the shaft turns, the drop into the gaps and the push home can all be followed.
  The close camera is yam_arena's spare recording camera, used for the video only; the policy's cameras, the physics and the scorer are unchanged, so the Dream-1 run (made through inspect-robots, not the demo script) sees what it saw before.
- "What to watch in the demo video" (under Scripted demo) gives the step and time of each moment.

Made at the same time, because a person reviews this version:

- The demo finds the meshing phase from the arm's own joint readings instead of turning the shaft through a fixed angle: it stops turning when the pushing hand jumps forward as the hub drops into the gaps.
  The fixed angle missed once in one session; the search was confirmed on two sessions before this revision.
- The loose hub's 0.5 and 0.75 count only once the grippers have let go of the hub and the shaft, and 1.0 needs them clear for the whole last second, not only on the last step.
- The set screws: a section under Fidelity says why the 5/16-18 screw (17-19 N m) is left out and how a real trial checks it.
- Cost: a section under Real-world build gives the parts' prices and a cheaper fixed side (a locked stub shaft in a housing of the reducer's size instead of an $850 worm reducer), so the rig costs about $200 to $300.
- Stillness and the real-trial checks: each score level now gives what the simulator measures and what a person checks; the simulator's 1 mm stillness is a real trial's "no visible motion in the last second of the video".
- Clearance: an offline replay of the demo against every arm collision mesh finds contacts only at the fingertips, on the parts being handled (the spider, the driven shaft, the loose hub) and on the fixed hub's outside where the spider's leg tips sit flush with it during insertion; no palm, wrist or arm link comes within 15 mm of anything.
