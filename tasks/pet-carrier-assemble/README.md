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

- The top slides forward on four hooked tabs. At the back position their feet are under the holes and the top can be lifted off; slid home, the feet are under the bottom's rim and it cannot lift. A bar under each of the top's rim flanges rides over a sprung ridge on the way (the click, about 3.7 N).
- A knob's quarter turn swings a cam bar up behind the top's front wall, so the top cannot slide back. Short of home the wall is over the cam bar, and the knob stops after about 36 degrees.
- Each door hinge is a three-sided barrel with a lever. Open, the barrel's mouth faces forward and a prong of the door can enter; closed, the mouth faces a fixed cheek and the prong is kept. A hinge closed before the door arrives keeps the prong out.
- Nothing holds the door in the opening until hinges are closed: let go with all four open, a 0.15 N nudge tips it out.
- The six fastenings on the front (two knobs, four hinges) sit 5 cm apart at each corner, and the two right-hand hinges have to be closed while the other hand holds the door 13 cm away.

Nearest tasks and how this differs: a folding produce crate whose hinged walls swing up to corner catches, a picture-frame back held flush while its clips are turned, a power strip hung on keyhole studs and slid to lock (#2335), and a rabbit lifted into a carrier (#1596).
Here two separate shells are joined by a slide onto hooked tabs, locked by knobs that only turn once the slide is complete, and a loose door is captured by its four prongs in four twist hinges.

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
4. The right hand's closed fingertips push the two right-hand hinge levers round to closed; the left hand lets go and pushes the two left-hand levers round; both hands move away.

Measured in Isaac Sim with this package:

- The demo reaches 1.0 (success) in 1,298 steps (65 s of the 120 s limit): top home at step 140, both knobs locked at step 340, door fitted with all four hinges closed at step 1,240 (hinges at 89 to 90 degrees, prongs 2.0 to 2.6 mm from their hinge axes). IK stays within 0.1 mm of every target; the two arms' smallest gap is 39.5 mm, while the right hand closes the right-hand hinges beside the left hand holding the door.
- `knobs_first` 0.0: the knobs' cam bars lift the front of the top, it cannot go home, and the pull drags the carrier more than 2 cm.
- `no_knobs` 0.25: the top is home, the knobs are left unlocked, the door and hinges are done.
- `door_open` 0.5: the door is put into the open hinges and none is turned.
- `two_hinges` 0.5: only the right-hand pair of hinges is closed.
- `hinges_first` 0.5: with the hinges closed first the door's prongs cannot enter them.
- `held` 0.75: everything is in place but the fingertips are still on the last lever.
- Dream-1, one run of 2,400 steps with the instruction above: 0.0. The left arm went to the door in its stand and hovered at it for the two minutes without taking it out; the right arm stayed near its start. The top, the knobs and the hinges were not touched.

## Scoring

Evaluated every control step; the score at the end of the episode counts.
Each level includes the ones before it, and all are end-state measurements that can be taken with a ruler and a photo.

- 0.25: the top is home on the bottom: no more than 1 mm behind its forward stop, less than 2.5 mm off to either side, the gap under its rim below 1.5 mm at all four corners, and both click ridges back within 0.6 mm of their free height.
- 0.5: both knobs are locked: turned at least 80 of their 90 degrees, so both cam bars stand behind the top's front wall.
- 0.75: the door is fitted and shut: each of its four prongs is within 4.5 mm of its hinge's axis, the door is within 5 mm of its seat on the sill, and all four hinges are turned at least 80 of their 90 degrees to closed.
- 1.0: the same, with both grippers at least 3 cm clear of the carrier and the door and nothing moving more than 1 mm for 1 second.
- 0 for good if at any point the bottom shell was moved more than 2 cm or tipped more than 5 degrees, or the top or the door left the table.

A prong inside a closed barrel can be at most 3.9 mm from the hinge's axis, so the 4.5 mm limit is passed only by captured prongs.
With both knobs locked the top can move back 0.7 mm at most, inside the 1 mm limit.

## Runtime and seeds

One trial is 2,400 steps (2 minutes at 20 Hz).
The layout is fixed (`object_jitter=0`) and the task uses no random numbers, so there are no seeds.

## Fidelity

- All parts are rigid bodies built from boxes at the real size. Straight 4 mm walls stand in for the tapered, ribbed shells. The top shell has its moulded-in carry handle (a bar across a hand hole in the roof) and its roof and side vent slots; the bottom shell's vents are not modelled.
- The bottom shell (0.42 kg) with its two knobs, two bottom hinges and two click ridges is one loose articulated body; the top shell (0.38 kg) with its two top hinges is another; the door (0.12 kg) is loose.
- The click is a sprung ridge in each side rim of the bottom (800 N/m, 1.5 mm high, 25 degree ramp in, 35 degree face out). In the real carrier the give is in the tabs.
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
| Click | about 3.7 N on the top to slide it home, about 5 N to slide it back with the knobs unlocked | estimate: pull the top's front ledge with a spring scale and read the peak |
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
