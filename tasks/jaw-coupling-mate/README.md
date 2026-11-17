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
Industrial machinery mechanics (O*NET 49-9041.00: "Repair or replace broken or malfunctioning components of machinery or equipment."; "Reassemble equipment after completion of inspections, testing, or repairs."; 446,900 US workers) and general maintenance and repair workers (49-9071.00: "Assemble, install, or repair wiring, electrical or electronic components, pipe systems, plumbing, machinery, or equipment."; 1,621,800) do this on conveyors, pumps, fans and mixers.

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
python demo.py task.py demo_out                               # full score: writes demo_*_cam.mp4, setup.png, result.json
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
3. The left hand swings round the shaft's axis like a crank, turning the shaft and the hub on its key: quickly for the first 25 degrees, then at 2.5 degrees per second.
   The jaws drop onto the spider's legs, slide round on them, and drop into the free gaps.
   The gaps are open to the jaws for about 2 degrees of turn, and a position-controlled arm needs a fraction of a second to follow the hub in, so a fast swing passes over them (it did in one run at 10 degrees per second).
4. The right hand pushes the hub home, and both hands let go and back off.

The demo is open loop: every target comes from the task file's fixed layout and the clock.
It reads no object pose, no joint of the coupling and no task state (its debug trace does).

Results in simulation (Isaac Sim 6.1, `object_jitter=0`): see the table at the end of this file.

## Scoring

Evaluated every control step; the score at the end of the episode counts.
Everything is measurable in a real trial with a rule, a feeler gauge and the trial video.

| Score | End state |
|---|---|
| 0.25 | The spider sits between the fixed hub's jaws: its centre within 1.5 mm of the shafts' axis, within 5 degrees of square, its near face no more than 1 mm proud of the jaw tips, grippers at least 3 cm clear of it |
| 0.5 | That, and the loose hub's jaw tips are 3 mm or more past the spider's near face, in its free gaps |
| 0.75 | That, and the loose hub's rear face is within 1 mm of the scribed line |
| 1.0 | That, with both grippers at least 3 cm clear of the spider, the loose hub and the driven shaft, and nothing moving more than 1 mm for 1 second |
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
- Not modelled: the set screws, the key as a separate part, the spider's raised spacing dots, the reducer's inside.
- Estimates to measure on the bought parts: the spider ring's outer radius (18.5 mm), the radius where the jaws start (19.5 mm), the leg-edge radius, the bearings' drag, the reducer's shaft height on its riser.

## Real-world build

| Part | What to buy or make |
|---|---|
| Hubs | 2 x Lovejoy L100 x 5/8 in, 3/16 x 3/32 in keyway, sintered iron (68514411510) |
| Spider | Lovejoy L099/100 open-centre Hytrel spider (68514411486) |
| Fixed side | A worm gear reducer with a 5/8 in output shaft, 30:1 or higher so it does not back-drive (Boston Gear 713 size: housing 73 x 108 x 118 mm, 5.4 kg), on a steel riser that brings its shaft 90 mm above the bench |
| Driven side | 5/8 in keyed steel shaft, 218.5 mm long, in two UCP202-10 pillow blocks 50 mm apart on a 60 mm steel base frame, the bearings' own set screws tightened so the shaft turns in them and cannot slide |
| Holder | A 60 x 80 x 6 mm board with two 10 x 46 x 24 mm blocks 16.5 mm apart |

- Bolt both machines to the bench with the shafts in line within 0.4 mm (the guide's limit for this coupling) and the driven shaft's end 19.5 mm from the fixed hub's jaw tips.
- Fixed hub: on the reducer's shaft over its key, face flush with the shaft's end, one jaw straight up, set screw tightened to the torque in the guide's table 2.
- Loose hub: on the driven shaft over its key, set screw backed out. It should slide under about 1.4 N: check with a spring scale hooked on its rim and pulled along the shaft; clean and oil the shaft lightly if it takes more.
- Scribed line: push the loose hub home against the fixed hub with a spider in place, scribe round the shaft at the hub's rear face, and pull the hub back 41.5 mm.
- The driven shaft should turn under about 2.5 N at its surface (0.02 N m): check with a cord wound once round the shaft and a spring scale.
- Mark the shaft and the nearer pillow block so the start phase (30 degrees from meshing) can be set again.
- Nothing here is powered: the reducer has no motor fitted.

## Results

Isaac Sim 6.1 with the yam_arena embodiment, `object_jitter=0`, from each run's `result.json`.

| Run | What it does | Score | Steps | Hub from its line | Spider seated |
|---|---|---|---|---|---|
| demo | Full sequence | 1.0 | 802 | 0.00 mm | yes |
| `no_spider` | The spider is left in its holder; the hub is turned and pushed home. The hubs close on each other, jaw between jaw, with no spider | 0.0 | 616 | 0.00 mm | no |
| `no_turn` | The shaft is not turned. The hub stops with its jaw tips on the fixed hub's jaw tips, 18 mm short of the line | 0.25 | 688 | 18.03 mm | yes |
| `on_legs` | The shaft is turned 18 of the 30 degrees. The jaws drop 2.5 mm onto the spider's legs and stop there, 15.5 mm short | 0.25 | 684 | 15.49 mm | yes |
| `short` | The hub is not pushed home after its jaws drop into the gaps | 0.5 | 818 | 7.16 mm | yes |
| `unseated` | The spider is let go between the hubs without being moved into the fixed hub's jaws. It falls out of the coupling and the hubs close without it | 0.0 | 770 | 0.00 mm | no |

The demo's largest IK error is 0.08 mm and the two arms never come closer than 46.4 mm.
The demo takes 802 of the 1,200 steps (40 s of simulated time; 2 to 3 minutes of wall time on one GPU, most of it start-up).

Dream-1, one run at the 1,200-step limit: 0.0.
It took its right hand to the spider in its holder and stayed over it for the rest of the minute; the left arm barely moved, the spider stayed in the holder and the loose hub was not touched.
