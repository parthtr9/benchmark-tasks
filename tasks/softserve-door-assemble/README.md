# Reassemble a soft-serve freezer door

A cleaned soft-serve freezer door (Taylor C706 single-spout door, X87683) lies on its back on a rubber mat, its top
toward the robot. Its draw valve (X56072), O-rings lubricated, has been started in its bore by hand and sits 8 mm short
of home with its slot toward the handle lugs. The draw handle (X56246) lies on the door's front face and the handle pin
(055819) on the mat. The robot must leave the door reassembled: the valve home in its bore, the handle's fork over the
valve's bar with its knuckle between the door's two lugs, the pin through lug, knuckle and lug, and the handle raised.

Fast food and counter workers (O*NET 35-3023, 3.86 million) tear the door down at every brush-cleaning, at the interval
local health code sets, and the operator's manual fits the valve and handle at the bench before the door is hung (steps
14-16). The three parts constrain each other in sequence: the fork only straddles the bar with the valve home, the
knuckle's hole only lines up with the lugs' with the fork on the bar, and the pin only goes through when all three holes
line up, while the handle is held.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the O-rings' drag and centring, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes, for yam_arena task files |
| `demo.py` | The scripted full-score demo (open loop on the layout, trimmed only by the arms' own joint readings) |
| `clearance.py` | Arm-to-arm clearance from the YAM collision meshes, for the demo's trace |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs
with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Reassemble the freezer door: draw valve in its bore, draw handle engaged on it and pinned, handle left raised." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/softserve-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                                       # full score
DEMO_VARIANT=pin_no_handle python demo.py task.py out --no-record     # pin through the empty lugs: 0.25
DEMO_VARIANT=reversed      python demo.py task.py out --no-record     # valve turned half a turn: 0
DEMO_VARIANT=no_pin        python demo.py task.py out --no-record     # handle placed, no pin: below 1.0
```

1. The left hand presses down on the door's face (the door would slide under the valve's drag). The right hand, pointing
   at the door with its fingers closed up and down so they pass under the lugs, pushes the valve's head home against
   the O-rings' drag.
2. The left hand takes the handle off the door's face, carries it over its place and lowers it: the knuckle drops
   between the lugs and the fork straddles the valve's bar. It holds the lever a few millimetres above fitted.
3. The right hand takes the pin by its bent end and presses it lightly on the knuckle while the left hand moves the
   lever 1 mm either way, so the knuckle's hole passes through line with the lugs', then pushes the pin through.
4. The right hand lets go and leaves, then the left; the handle settles raised on its pin.

The demo reaches 1.0 (success) in 1,001 steps (50 s) of the 1,800-step limit, with the arms at least 12.7 mm apart, and
did so in four Isaac Sim runs in a row. Each negative variant scores below 1.0 (Isaac Sim batch: pin_no_handle 0.25,
reversed 0, no_pin 0.25: let go unpinned, the knuckle rises 18 mm out of the lugs). Dream-1, given only the instruction, scored 0
in one run of 1,800 steps.

## What changed since the first Dream run

- The scorer now gives each level only once no gripper finger is within 3 cm of the parts it credits (the valve for 0.25, and the handle for 0.5, and the pin for 0.75), so a part held in place earns nothing (before, those levels also counted while held). The instruction is unchanged.
- Rerun on the changed scorer: the Isaac Sim batch gave the same scores (demo 1.0 in 1,001 steps; pin_no_handle 0.25, reversed 0, no_pin 0.25), and the one new Dream-1 run scored 0 in 1,800 steps, as the first did.

## Scoring

End state, each level including the ones before it.

- 0.25: the valve bottomed in its bore (within 2 mm of the ledge, axis within 2.5 mm) with its slot within 20 degrees of
  facing the lugs.
- 0.5: and the handle's knuckle between the lugs (within 3 mm of their hole axis) with the valve's bar inside the fork.
- 0.75: and the pin through both lugs and the knuckle (within 1.5 mm of the lug axis).
- 1.0: and the handle within 12 degrees of raised, both grippers at least 3 cm clear, nothing moving more than 1 mm for
  1 s.
- Each level counts only once no gripper finger is within 3 cm of the parts it credits (the valve for 0.25, and the handle
  for 0.5, and the pin for 0.75): a part held in place earns nothing.
- 0 for good if the door was at any point moved more than 10 mm.

The 3 mm for the knuckle is the play of the holes: 8 mm holes round the 6.35 mm pin let a pinned knuckle sit up to
2.3 mm off. The 2.5 mm for the valve's axis covers the modelled O-rings' play (below).

A valve left a few millimetres short of home is drawn home by closing the handle, as on the real machine, so it is not
scored as a fault.

## Fidelity

Rigid parts at catalogue size. The valve's three lubricated O-rings are rigid 16-sided bands with 1.3-1.5 mm play in the
bore, plus a radial spring of 2 N/mm per ring for their squeeze, which centres the valve, and 12 N of Coulomb friction along
the bore, holding it still up to that force and resisting turning the same way (always on while the valve is in
its bore, keyed only to its own motion there; the 12 N breakaway is an estimate: measure the bought part with a spring
scale). Stuck, the valve gives up to 3 mm under a load and comes back when the load goes; sliding, it stores no stretch,
so it stays where a push leaves it. Applied each 1/240 s physics step, the friction holds a load built up over 50 ms
to about 11 N, and one applied within a single step to about 7 N (offline tests). Both act only across or along the bore and never hold the handle (estimates: check the drag with a spring
scale). The door, handle and valve sizes and masses are from parts listings and the O-ring spec; the pin, the lug
spacing, the valve's slot and bar and the handle's fork (a plain 6.5 mm notch round the 5 mm bar) are scaled from the
manual's drawings and are estimates. The handle's flow-adjustment screw is not modelled.

## Real-world build

### Parts (all Taylor service parts for the C706 / C707 single-spout freezer)

| Part | Taylor no. | Source | As modelled |
|---|---|---|---|
| Freezer door, single spout | X87683 | Taylor C706 operator's manual parts list; Taylor dealers (e.g. Parts Town) | 0.87 kg stainless casting; bore 21.8 mm (0.86 in) with a 10 mm, 14 degree lead-in; two handle lugs 5 mm thick, 28 mm apart inside (estimate) |
| Draw valve | X56072 | [Parts Town TAF-X56072](https://www.partstown.com/taylor-freezers/tafx56072) | 160 mm (6.3 in) long, 19 mm (0.75 in) body, 0.09 kg; a 10 mm slot across its head and a 5 mm bar in it |
| Draw valve O-rings (3) | 014402 | Taylor manual parts list | 7/8 in x 0.103 in, lubricated |
| Draw handle | X56246 | [AllPoints X56246 listing](https://www.restaurantsupply.com/products/allpoints-x56246-oem-handle-draw-valve-stainless-steel-manual-operation) | 0.31 kg stainless, 174 mm over all; knuckle 24 mm wide with an 8 mm hole; fork plate 8 mm thick with a 6.5 mm notch |
| Handle pin | 055819 | Taylor manual parts list (no online listing found) | 1/4 in (6.35 mm) stainless, 75 mm straight with a 25 mm bent end, 0.04 kg |
| Rubber mat | - | any bar mat | 300 x 560 x 4 mm |

Measured from listings: valve length and body diameter, O-ring size, part masses where listed.
Scaled from the manual's exploded drawings (estimates, to be measured on the bought parts before trials): lug spacing and thickness, hole sizes, knuckle width, fork notch, the valve's slot and bar, pin length.

### Forces and how each is set and checked

| Force | Value | How it is obtained | Check before each session |
|---|---|---|---|
| Valve sliding in its bore (three lubricated O-rings) | 12 N (estimate) | Taylor's food-safe lubricant on the O-rings, a thin even film as the manual directs | Push the valve in with a push-pull spring scale on its head: 10 to 14 N while it slides; add or wipe lubricant to bring it into range |
| Door sliding on the mat | about 8 N (0.87 kg, rubber friction about 0.9) | rubber mat, dry door back | Pull the empty door along the mat with the spring scale: it must slide at 6 to 10 N, below the valve's 12 N, so an unheld door moves when the valve is pushed |
| Handle on its pin | free | 8 mm holes round the 6.35 mm pin | The pin drops through lugs and knuckle by its own weight when the holes line up |

### Real-trial checks for each scoring level

- Door moved (0 for good): tape the door's outline on the mat before the trial; afterwards measure the largest offset from the outline with a rule (limit 10 mm).
- 0.25, valve home: draw a 2 mm wide band on the valve body 0 to 2 mm behind its head; the valve is home when the band is inside the bore and the head's face is within 2 mm of the bore's top face (feeler gauge or rule across the face). Slot toward the lugs: a top photo with the lug line drawn through it; the slot's long edge within 20 degrees of square to that line.
- 0.5, handle engaged: in the top photo the fork straddles the valve's bar and the knuckle sits between the lugs; the 3 mm hole alignment is checked by passing a 2 mm drill rod (8 mm holes less twice 3 mm) through lug, knuckle and lug by hand after the trial, without moving the handle.
- 0.75, pinned: the pin is visible through both lugs, its bent end on one side and at least 5 mm of its tip on the other.
- 1.0: the handle's angle from raised measured with a phone inclinometer on the handle (within 12 degrees); both grippers at least 3 cm away and nothing moving for 1 s, from the trial video.

### Set-up

A Taylor C706 or C707 door assembly with its draw valve, O-rings, handle and pin, cleaned, on the rubber mat on the bench.
Start the valve in its bore by hand until its head is 8 mm short of home (a second mark on the valve 8 mm behind the head), slot toward the lugs; lay the handle on the door's front face and the pin on the mat at their outlines.

## Who does this, and how often

Fast food and counter workers (O*NET 35-3023) list as a core task "Prepare and serve cold drinks, or frozen milk drinks or desserts, using drink-dispensing, milkshake, or frozen custard machines", and cleaning food preparation equipment among their work activities.
A soft-serve freezer without heat treatment is taken apart, washed, sanitised and reassembled every day (for example the Vermont frozen-dessert factsheet: "completely disassembled, washed and sanitized on a daily basis"); heat-treatment models are torn down every 14 days (Taylor).
The door's draw valve, handle and pin are fitted at the bench before the door is hung (Taylor C706 manual, steps 14-16), so this exact operation is part of the closing or opening routine wherever such a freezer is in service: quick-service restaurants, ice-cream stands, convenience stores and cafeterias.
Not verified here: a head count of the workers who do this particular step; the claim rests on the occupation's core task and the daily cleaning requirement.

## Runtime and seeds

One fixed layout (`object_jitter=0`), no randomness. Run limit 1,800 steps (1.5 minutes at 20 Hz): three parts are
fitted one after another, the pin while the handle is held.
