# Remove a wheelchair's swing-away leg rests: squeeze the release, swing out, lift off the pins, set down on the floor

## Revision

The review of the previous version asked for two changes; everything else in the task is unchanged (instruction,
chair, leg rests, latch, layout, every other scoring rule, run limit).

1. Feasibility: "The shelf projects forward from a 30 mm floor-standing back panel without specified bracing or
   anchoring, making the loaded real-world arm mount unstable. [...] Provide a stable, braced or securely anchored
   support for the arm-mounting shelf, with dimensions and attachment details sufficient to keep both arms and their
   loads from tipping the mount during trials."
   **Changed:** the shelf now sits on a closed, floor-anchored plywood stand: the back panel plus four full-depth
   upright panels (two end panels and two webs directly under the arm bases), glued and screwed into a box, and tied
   to a concrete floor by six steel angles with one concrete screw anchor each (four at the back, two at the front
   inside the end panels). Every dimension and fastener is under "Station" and "Real-world build", with a tipping
   check in both directions (safety factor 4.9 forward and 2.0 backward, under deliberately severe loads). The
   simulation models the same stand as fixed bodies; nothing of it is ahead of the shelf's front edge, so the floor,
   the chair, the mats and the camera's view are as before.
2. Note: "The simulated drop check treats grips away from the hanger tube as unsupported; consider detecting secure
   grasps elsewhere on the leg rest."
   **Changed:** a leg rest now counts as held when a closed gripper's two finger pads are both within 25 mm of any
   solid part of it (hanger tube, footplate, release lever and paddle, lever bracket, hinge plates or sleeves),
   measured against the leg rest's own geometry in its own frame, with the release lever at its current angle.
   Before, only the tool point's distance to the hanger tube counted. Because the grasp zone now covers the whole
   part, a hand also has to carry it: the leg rest must move with that gripper (the place on the leg rest where the tool
   point is moves no more than 45 mm along the leg rest over two control steps). A leg rest that slips or falls out of a hand still near it
   is therefore unsupported at any speed, and one lowered quickly in a closed hand is never dropped. A leg rest carried by its footplate and lowered to the mat is no longer called dropped; one released
   above the mat still is (both are offline tests).

A Drive Medical Silver Sport 2 wheelchair (18 in seat, desk arms, swing-away footrests, item SSP218DDA-SF) stands
open on the floor in front of the two arms, facing them, parked for a bed-to-chair transfer: both wheel locks are
already on. Its two swing-away leg rests (footrests) hang on their hinge pins, latched pointing forward. The robot
must take both leg rests off the chair and set them down on the blue floor mats beside it.

Each leg rest hangs by two sleeves on two 28 mm hinge pins on the chair's front post and is held forward by a spring
latch: a tooth on the release lever sits in a notch of an index plate under the lower pin. The only way off is the
one the chair's manual gives: press the red paddle toward the hanger tube (that pulls the tooth out of the notch),
swing the leg rest out about its pins, lift it straight up the 28 mm of the pins, and then carry the 0.9 kg part,
held at arm's length by its tube, down to the floor and let go of it there rather than dropping it.

## Why this task

Clearing the footrests is a step of every assisted transfer from bed to wheelchair, and the nurse aide competency
exam tests it: Credentia's NNAAP skills list, Skill 22 "Transfers from bed to wheelchair using transfer belt", step 4
"Before assisting to stand, footrests are folded up or removed" (https://credentia.com/storage/va/va-nnaap-skills-list.pdf).
The open nurse-aide textbook used in Wisconsin technical colleges (M. S. Reuter, *Nursing Assistant*, CC BY 4.0)
makes removal the step: section 8.10, transfer from bed to chair with a sit-to-stand, "Position the wheelchair
appropriately, remove the foot pedals, and lock the brakes"; section 8.9, with a gait belt, "Remove the foot pedals
from the wheelchair if needed". The chair's manual adds "When getting in or out of wheelchair, do not stand on or apply
weight to foot rests". O*NET 31-1131.00 Nursing Assistants: "Lift or assist others to lift patients to move them on or
off beds, examination tables, surgical tables, or stretchers"; BLS OEWS (May 2023) counts 1,351,760 nursing
assistants (https://www.bls.gov/oes/2023/may/oes311131.htm). Category: Impact on jobs, because removing the leg rests
is a routine part of a core duty (transfers) of an occupation of more than a million US workers.

The chair's owner's manual gives the procedure the robot must infer from the end state: "Release lock by pushing on
the front rigging lever. Swing-away foot rest /leg rest out and sideways. Lift to remove." (Drive Medical Silver Sport
2 user manual, https://apria.com/hubfs/User%20Manual/Drive%20Silver%20Sport%202%20Wheelchair%20User%20Manual.pdf).

The physical crux: the leg rest comes off along one path only. Pulling or lifting a latched leg rest does nothing:
the tooth in the notch stops it turning, and a cap over the index plate stops the tooth, and so the leg rest, from
lifting. The grip has to take the tube and the paddle together so that the squeeze releases the latch, the hand has to
turn the part about an axis it does not see (the pins), then lift along that axis, and the arm then carries 0.9 kg low
and to the side, where it sags. The failures that look like success are a leg rest swung out but still on its pins,
one let go above the mat, or one left in the hand.

The ability is a latch release and a lift-off along hinge pins, followed by a controlled set-down. The wheel locks are
on at the start and are not part of the task; the chair is not folded or unfolded.

## Files

| File | What |
|---|---|
| `task.py` | The task file: the chair (body with the wheel locks drawn on, two locked rear wheels, two caster forks and casters), the two leg rests (each with its spring-loaded release lever), the mounting shelf's anchored stand, two floor mats, and the scorer |
| `bay.py` | Puts the two arm bases on a mounting shelf 0.65 m above the floor instead of the station table, and gives the stand's dimensions (see "Station") |
| `partkit.py` | Jointed rigid parts for yam_arena task files (hinges, contact materials) |
| `meshkit.py` | Meshes built in code (tubes, wheel slices, the latch's index plate) attached to partkit parts |
| `demo.py` | The scripted full-score demo and the negative variants |

## Station

The task replaces the station's table by a 360 x 1,300 x 40 mm mounting shelf on an anchored stand, so that the floor
in front of the arms is free and the arm bases stand 0.65 m above it. On the station table they are 0.76 m up, which
puts the leg rests' set-down poses (the footplate on the floor beside the chair) out of reach: an IK probe at 0.7625 m
missed poses below about 0.40 m above the floor by 50-200 mm. `bay.py` changes four constants of `yam_arena.station`
when the task file is imported; arms, grippers, control gains, action interface and cameras are the standard ones, the
top camera keeps its calibrated pose, and the run uses `--embodiment yam_arena --scorer task_score` as usual.

### The arm mount

Millimetres: x forward from the line through the arm bases, y to the robot's left, heights above the floor. All
plywood is birch plywood (about 680 kg/m3).

| Part | Size | Place | Attachment |
|---|---|---|---|
| Shelf | 360 x 1,300 x 40 (two 20 mm sheets glued and screwed) | x -310 to 50, top 647.5 high | glued (PVA) and screwed down into every panel under it, 5 x 70 mm wood screws every 150 mm |
| Back panel | 1,300 wide, 607.5 high, 30 thick | x -310 to -280, on the floor | screwed from behind into the edges of the four uprights, 5 x 70 mm wood screws every 150 mm, glued |
| End panels (2) | 330 deep, 607.5 high, 30 thick | x -280 to 50, y 620 to 650 and -620 to -650, on the floor | as above |
| Webs (2) | 330 deep, 607.5 high, 30 thick | x -280 to 50, y 290 to 320 and -290 to -320 (under the arm bases), on the floor | as above |
| Rear floor angles (4) | steel angle 50 x 50 x 5, 100 long | on the back panel's rear face, centred at y -550, -200, 200, 550; anchors at x -335 | two M8 bolts each through the back panel (30 mm washers, nyloc nuts); one anchor each |
| Front floor angles (2) | steel angle 50 x 50 x 5, 100 long | inside each end panel at x -70 to 30; anchors at x -20, y 595 and -595 | two M8 bolts each through the end panel; one anchor each |
| Floor anchors (6) | 1/4 in x 2-3/4 in hex-head concrete screw anchors, 1-3/4 in (44 mm) embedment, 3/16 in pilot hole | through the angles' floor legs | concrete slab of at least 2,000 psi (14 MPa), at least 100 mm from slab edges and joints |
| Arms | the two standard arms | bases at x 0, y 305 and -305, 650 high, each over its web | through-bolted to the shelf with M6 bolts, washers and nyloc nuts at the arm base's mounting holes |

The shelf, back panel and four uprights form a closed box: the uprights brace the back panel and the shelf against
racking in both directions, and each arm's base moment goes straight down the web under it. Nothing of the stand is
ahead of the shelf's front edge (x 50), so the floor in front, where the chair and the mats are, stays free; the
uprights and the front angles are under the shelf, where the arms cannot reach. The simulation models the stand as
fixed bodies at these sizes (the shelf is the station's table resized); in the top camera's view it is hidden under
the shelf.

### Tipping check

Self-weight of the stand: shelf 12.7 kg (centre at x -130), back panel 16.1 kg (x -295), four uprights 16.4 kg
(x -115); 45.2 kg in all, its centre at x -183.

Forward: both arms reach forward holding a leg rest, and the stand would tip about its front edge (x 50). Load per
arm, deliberately severe: the whole arm, taken as 6 kg with its gripper and wrist camera, plus the 0.9 kg leg rest,
plus a 50 N extra pull (a leg rest jammed on its pins), all at x 800, farther than the arm reaches (lever 0.75 m).

- Overturning moment: 2 x (6.9 kg x 9.81 m/s2 + 50 N) x 0.75 m = **177 N m**.
- Resisting moment: self-weight 45.2 kg x 9.81 x 0.233 m = 104 N m; the four rear anchors at x -335 (lever 0.385 m),
  each counted at a design tension of 0.5 kN (well below the maker's published allowable tension for this anchor
  and embedment in 2,000 psi concrete; check the data sheet of the anchor bought): 4 x 500 N x 0.385 m = 770 N m;
  together **874 N m**.
- Safety factor **4.9** (2.8 counting only two of the four rear anchors). Without anchors it would be 0.6 under this
  load, which is why the stand is anchored. With all four anchors the factor stays above 2 for arms up to 20 kg each.

Backward: both arms press down on the floor in front, which pushes the arms up, and the stand would tip about the
rear angles' back edge (x -360). Load: 100 N down per arm (the most the arm gives straight down, near its base) at
x 800 (lever 1.16 m).

- Overturning moment: 2 x 100 N x 1.16 m = **232 N m**.
- Resisting moment: self-weight 45.2 kg x 9.81 x 0.177 m = 78 N m; the arms' own weight taken at their bases,
  2 x 6 kg x 9.81 x 0.36 m = 42 N m; the two front anchors at x -20 (lever 0.34 m), 2 x 500 N x 0.34 m = 340 N m;
  together **460 N m**.
- Safety factor **2.0**.

Sliding: the arms' sideways and pulling forces (well under 100 N each) go into the six anchors in shear.

## Run a trial

From this folder, in the Isaac Lab-Arena environment:

```bash
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Take both leg rests off the wheelchair and set them down on the floor mats beside it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir logs
```

## Scripted demo

```bash
python demo.py task.py demo_out                          # full score: demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=one_footrest python demo.py task.py out     # only one leg rest taken off and set down
DEMO_VARIANT=swing_only   python demo.py task.py out     # both released and swung out on their pins, not lifted off
DEMO_VARIANT=no_lever     python demo.py task.py out     # tube taken below the paddle: the latch stays in
DEMO_VARIANT=drop         python demo.py task.py out     # both raised 10 cm over the mats and let go there
DEMO_VARIANT=held         python demo.py task.py out     # both lowered to the mats but never let go
```

Each arm works the leg rest in front of it, both at once. The open gripper comes down over the hanger tube and the
red release paddle together, 13 cm ahead of the hinge pins, and closes on both; the squeeze presses the paddle toward
the tube, which pulls the latch tooth out of its notch. The hand swings the leg rest a quarter turn outward about the
pins, lifts it 15 cm (the pins are 28 mm long; the hand sags under the load), moves it 6 cm straight out, carries it
over the mat, lowers it until the footplate is on the floor, opens and rises, and the arm goes home.

The demo uses only what a policy could observe: the fixed layout in the task file (`object_jitter=0`), timing and the
arms' own joint commands. It never reads a part's pose or a joint of the chair or the leg rests. `result.json` logs
the parts' states for checking only.

Results (batch 6ec2e5225a414e1ebf9e1a30bec878c4; every value is what the scorer printed):

| Run | Score | What happens |
|---|---|---|
| demo | 1.0 | both leg rests released, swung out, lifted off and set down on the mats; full score at step 428 (21.4 s); the chair shifted 6.7 mm while the leg rests were swung out (limit 100 mm) |
| one_footrest | 0.45 | one leg rest set down, the other left latched |
| swing_only | 0.2 | both released and swung out on their pins (0.1 each), not lifted off |
| no_lever | 0.0 | tube taken below the paddle: the latch tooth stays in its notch and the leg rest does not turn |
| drop | 0.0 | both raised 10 cm over the mats (about 25 cm above the floor) and let go: they fall out of the hands (the place at the tool point moves 59-76 mm in the first 0.1 s) and are dropped |
| held | 0.4 | both off their pins and on the floor, still in the grippers (0.2 each); held all the way down, never dropped |

After the hand opens, each leg rest tips over onto its side on the mat (it does not stand on its footplate); it is
still on the mat and is scored as set down. The demo's worst IK error is 26 mm, at the outermost set-down pose; the
plan is open loop, so this is part of the commanded motion and the run still reaches 1.0.

## Scoring

Evaluated every control step; the score at the end of the episode counts.

- 0 if the chair has at any time left its parked place: rear axle midpoint moved more than 100 mm along the floor,
  turned more than 10 degrees, or tipped more than 10 degrees. With its wheels locked the chair moves that far only if
  it is dragged or tipped over, and it is then no longer where the transfer was set up. Smaller shifts are not
  penalised.
- Each leg rest, by its state at the end:
  - 0.45 if it is set down: off its pins (lower sleeve more than 40 mm from the pin axis, or above the pin tops),
    resting on a mat (lowest point within 30 mm of the floor; the hanger tube's two ends and the footplate centre
    over one mat), no gripper within 30 mm of it, and never dropped;
  - 0.2 if it is off its pins but not set down (still in a gripper, or lying off the mats);
  - 0.1 if it is still on its pins but turned 45 degrees or more outward (latch released, not lifted off);
  - 0 if it is still latched forward, and 0 for good once it was dropped.
- Dropped means it lost its support: off its pins, its lowest point more than 30 mm above the floor, and no gripper
  holding it, for two control steps (0.1 s) in a row. A gripper holds it when its fingers are closed to a gap under
  60 mm with both finger pads within 25 mm of any part of the leg rest (hanger tube, footplate, release lever and
  paddle, lever bracket, hinge plates or sleeves) and the leg rest moves with that gripper: the place on the leg rest
  where the gripper's tool point is moved no more than 45 mm along the leg rest over the last two control steps (it
  may pivot in the grip, but not slide or fall out of it).
  So a leg rest that slips or falls out of a hand still near it counts as unsupported at any speed, and one lowered
  quickly in a closed hand is never dropped. In the simulation the pads' distance is measured to the leg rest's own
  solid parts in its own frame, with the release lever at its current angle.
- With both set down the score is 0.9, and 1.0 once nothing has moved for 1 s. The simulation check: the chair and
  both leg rests stayed within 1 mm for 20 control steps.

Possible scores: 0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.65, 0.9, 1.0.

In a real trial: a tape measure for the chair's place (floor marks at the rear axle ends) and the leg rests' height
above the floor; the video for the swing angle, for whether a hand is closed on a leg rest, for a drop (the leg rest
falls away from the hand from more than 30 mm above the mat) and for stillness (no visible motion for 20
frames at 20 fps).

## Runtime and seeds

About 1 minute per trial at the full 1,200 steps, plus Isaac Sim start-up. The layout is fixed (`object_jitter=0`);
there are no seeds to choose. The simulation is deterministic: the same demo gives the same trajectory.

## Fidelity

Rigid parts at the chair's catalogue sizes. Locked rear wheels are locked joints (lock holding torque not modelled); casters roll and swivel on 120-sided convex treads. Leg-rest latch (tooth, notch, paddle spring), pin and sleeve sizes and the mass split are estimates, not measured from the product. The latch releases by contact only. Sling seat and back are rigid panels; frame bends, handrims and lock linkage are visual only. Pins greased (friction 0.15). Anchored arm stand is fixed.

Nothing in the code sets the latch, the pins or a leg rest's state: the lever is a sprung hinge on the leg rest, the
tooth, notch, sleeves and pins are colliders, and release, swing and lift-off happen through contact with the
gripper. Each wheel collides as convex pie slices with a rim point every 1.5 degrees (3 on a caster), because a
single cylinder prim collides as a coarse prism and makes the chair hop. The locked rear wheels are joints locked at
their start angle: on the real chair the lock shoe pressed into the tyre holds them by friction, which this task does
not test.

What is sourced, and what is estimated (no drawing of the leg rest or its latch is published; none of the latch
figures is measured from a drawing or from the part):

| Item | Value | Source |
|---|---|---|
| Model, configuration | Silver Sport 2, 18 in seat, desk arms, swing-away footrests (SSP218DDA-SF) | Drive Medical product sheet |
| Seat width / depth / height | 18 x 16 in; seat to floor 17.5-19.5 in (19.5 modelled) | product sheet; retailer specification |
| Back height, overall height, open width | 16 in; 36 in; 26 in | retailer specification |
| Chair weight | 41 lb (18.6 kg), modelled with the leg rests | retailer specification |
| Rear wheels, casters | 24 x 1 in solid urethane; 8 in casters | product sheet |
| Wheel locks | push-to-lock, "Locks are engaged by pushing handle completely forward" | owner's manual |
| Leg rests | swing-away, plastic footplates with heel loops, part STDS3J24SF (about 4 lb a pair as shipped); "Release lock by pushing on the front rigging lever. Swing-away foot rest /leg rest out and sideways. Lift to remove."; footplates no lower than 2-1/2 in from the floor | part listings; owner's manual |
| Leg rest mass | 0.9 kg each | estimated from the shipping weight |
| Leg-rest latch | 8 mm pins 28 mm long, 89 mm apart; sleeve, 44 mm index plate, 9.4 mm notch, 8 mm tooth and paddle geometry; lever spring 2 N m/rad (2.5 N at the paddle at rest, 8 N fully pressed) | estimated |
| Wheel-lock linkage (drawn only) | pivots, link lengths, 45 degree travel | estimated |
| Masses of the chair's parts | frame 11.3 kg, rear wheel 1.9 kg, caster 0.35 kg, fork 0.25 kg | estimated, summing to the listed 41 lb |
| Tube size, caster trail, footplate size | 22 mm tube, 45 mm trail, 150 x 200 mm plate | estimated |
| Contact properties | tyre friction 0.9, pins and sleeves 0.15 (greased), leg rest tube 0.8 | estimated |

The forces above (paddle spring, friction) are simulation values. A real trial checks them on the bought chair: the
paddle force with a spring scale at the paddle's centre, and the lift-off force (the leg rest's weight plus pin
friction) with the spring scale hooked on the hanger tube.

## Real-world build

- A Drive Medical Silver Sport 2 wheelchair, 18 in seat, desk arms, swing-away footrests (SSP218DDA-SF): about $220
  (retail, November 2026). Spare leg rests (STDS3J24SF, pair): about $50. Nothing else from the chair's maker is
  needed.
- A smooth vinyl or sealed-concrete floor on a concrete slab, level to 1 mm over the chair's footprint (check with a
  spirit level).
- The arm mount described under "Station": birch plywood (one 1,300 x 360 x 40 mm shelf, one 1,300 x 607.5 x 30 mm
  back panel, four 330 x 607.5 x 30 mm uprights), PVA glue, about 60 wood screws 5 x 70 mm, six 100 mm pieces of
  50 x 50 x 5 mm steel angle, twelve M8 bolts with 30 mm washers and nyloc nuts, six 1/4 in x 2-3/4 in concrete
  screw anchors, and M6 bolts for the arm bases. Assemble the box, square it, set it in place, drill the anchor holes
  through the angles and anchor it; then bolt the arms on with their bases 650 mm above the floor at y +305 and
  -305 mm. Before the first trial, push forward and pull up on each arm's base with 200 N (a spring or bathroom
  scale): the stand must not move.
- Chalk or tape marks for the chair's rear axle ends at x 850 mm, y +295 and -295 mm; casters trailing, chair square
  to the arms.
- Two blue 700 x 660 mm floor mats (2 mm rubber) at x 50 to 750 mm, |y| 340 to 1,000 mm.
- A spring scale (0-50 N) to record the paddle and lift-off forces once per chair, and a camera at the station's top
  camera pose.

## Reset

Take the leg rests off the mats and hang each one, pointing forward, on its two pins until the latch clicks into its
notch. Check that both wheel-lock handles are pushed fully forward and that the chair does not roll when pushed by
hand at the seat front. Put the chair back on its floor marks.
