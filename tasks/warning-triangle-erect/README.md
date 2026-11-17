# Set up a folded highway warning triangle

A folded warning triangle of the kind every US truck carries (FMVSS 125; Deflecto 73-0711-00 Emergency Warning Triangle Kit class: 438 mm across and 420 mm tall when set up, 32 mm thick folded, 1.64 kg with its weighted base) stands on its base inside a tape outline between the arms.
Its three bars lie face to face in one pack along the base: the bottom bar and the two side arms, each hinged at one end of the bottom bar and folded down along it.
The robot must leave it set up: both arms raised and joined at the apex, the triangle turned a quarter turn across its base, standing on the marked spot.

Why it matters.
49 CFR 393.95(f) requires a commercial motor vehicle to carry "three bidirectional emergency reflective triangles that conform to the requirements of Federal Motor Vehicle Safety Standard No. 125", and 49 CFR 392.22(b) requires the driver of a vehicle stopped on the road or shoulder to put the warning devices out "as soon as possible, but in any event within 10 minutes".
FMVSS 125 itself requires that the device be "designed to be erected, and replaced in its container, without the use of tools".
Each of the three triangles has to be set up before it is carried to its place.
Heavy and tractor-trailer truck drivers (O*NET 53-3032.00, 2,221,200 workers; task "Check vehicles to ensure that mechanical, safety, and emergency equipment is in good working order.") and light truck drivers (53-3033.00, 1,068,500 workers) carry and use these (employment as shown on onetonline.org, read 2026-11-17).

What the task tests.

- Two loose hinged arms that must be up at the same time.
  Each arm swings up in the triangle's own plane on a loose rivet and falls back under its own weight (0.19 N m at the rivet when folded) unless it is held.
  The triangle only holds itself once one arm's pin is in the other arm's slot, so one arm has to be held up while the other is brought to it.
- A snap made out of reach.
  The apex is 38 cm above the table, beyond the grippers; the arms can only be worked near their rivets, and the pin has to find the slot from the geometry of the two hinges.
  The pin enters as the two bars spring about 4 mm apart (about 0.4 N at the pin) and is then held by the slot's walls.
- A pack that cannot be pinched apart.
  Folded, the three bars lie face to face, 3.9 mm apart, so a pinch on an arm clamps the whole pack; an arm has to be raised from below its lower edge, which is 41 mm above the table.
- A swivel that turns under a light push.
  The triangle turns on its base about a vertical swivel (0.08 N m), so a hand that pushes an arm away instead of up swings the whole triangle.
  The standing footprint, base across the triangle with the two corner feet, only exists after the quarter turn.

Nearest tasks, and the physical difference:
"Set out a stopped truck's warning triangles" places solid triangle tokens at distances on a scale road and nothing is set up; "Raise a two-piece arch" leans two loose blocks on each other; "Put up a folding produce crate's four walls" swings walls up to pawl catches on their neighbours, each wall within reach along its whole height; "Unfold and place a mini tripod" spreads three legs.
Here two loose arms must be held up together for a snap that is out of reach, on a part that turns on its base when pushed.

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
  --instruction "Set up the warning triangle so that it stands on its base on the marked spot." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/warning-triangle-erect-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.
There are no seeds: `object_jitter=0`, one fixed layout.

## Scripted demo

```bash
python demo.py task.py demo_out                         # full score
DEMO_VARIANT=one_arm python demo.py task.py out --no-record
```

The demo is open loop: every target comes from the task file's fixed layout and the clock.
The only feedback is each arm's own joint readings: a lifting hand trims its height, reach and pitch by what its own joints say it gives under an arm's weight (about 7 mm and 4 degrees), and the right hand finds the right arm's edge by going in until its joints show it held back.
It reads no object pose, hinge angle or task state.

1. The right hand pinches the near end of the folded pack from above and walks it a quarter turn round the swivel to the stop.
   The pack now lies across the base, facing the robot, with nothing under its bars.
2. The right hand, fingers closed and pitched 45 degrees, goes under the lower edge of the left arm near its tip end and lifts it to 24 degrees (0.5 N there).
3. The left hand, pitched 15 degrees, goes under the same arm 13 cm from its rivet.
   The right hand lets the arm down onto it and then rests its fingertips on the folded right arm's face, which keeps the triangle against its swivel stop.
   The left hand raises the arm to just past 60 degrees and holds it.
4. The right hand goes under the right arm 10 cm from its rivet and raises it, levelling once its fingertips are above the bottom bar.
   The pin on the left arm rides up the bevelled rim of the right arm's slot, the bars spring apart and the pin drops in.
5. The hands lower away one after the other and the arms settle on the pin.

Negative variants (`DEMO_VARIANT=`), each closing one unintended strategy:

| Variant | What it does | Score | Why |
|---|---|---|---|
| `turn_only` | turns the pack across the base and stops | 0.25 | no arm is raised |
| `one_arm` | raises the left arm and lets go | 0.25 | the arm falls back |
| `no_hold` | lets the left arm go before the right arm arrives | 0.25 | nothing for the pin to meet; both arms end folded |
| `short` | stops the right arm 45 mm short | 0.25 | the pin never reaches the slot; let go, both arms fall back |
| `no_turn` | goes straight to raising the arms | 0.0 | the base lies under the bars, so nothing can go under an arm |
| `drag` | keeps the pinch after the turn and pulls sideways | 0.25 | the arm cannot shift the 1.64 kg triangle 5 cm on its grip pads, and nothing is raised |

The rule that a base moved more than 5 cm or tipped over scores 0 for good is checked in the offline tests (`tests/test_warning_triangle_erect.py`), since the arm could not move the triangle that far in the `drag` run.

## Scoring

End state, measured with a rule, a protractor or level and the trial video.

- 0.25: one arm is raised more than 45 degrees from folded, or the triangle is turned 80 to 100 degrees on its base.
- 0.5: both arms are joined at the apex (the pin at least 2 mm into the slot and within 3 mm of its seat at the slot's end) with both grippers at least 3 cm clear of the triangle.
- 0.75: joined and clear, turned 80 to 100 degrees across the base, and the triangle's plane within 5 degrees of vertical.
- 1.0: that state with nothing moving more than 1 mm for 1 second.
- 0 for good if at any point the base was moved more than 5 cm from its mark or was tilted more than 30 degrees (the triangle went over).

The order of the steps and which arm does what are not scored.
A triangle that is joined but still along its base scores 0.5: it stands in still air (it goes over when leaned 20 degrees), but it is not set up.

## Results

Isaac Sim 6.1 with the standard `yam_arena` embodiment, 20 Hz, one fixed layout.

- Scripted demo: 1.0, success after 879 steps (three runs, the same each time); smallest distance between the two arms 36 mm.
- Negative variants: `turn_only` 0.25, `one_arm` 0.25, `no_hold` 0.25, `short` 0.25, `no_turn` 0.0, `drag` 0.25.
- Dream-1, one run of 1,200 steps: 0.0.
  Both arms went to the folded pack and handled it where it lay along the base for the whole minute; it was not turned across the base and no arm was raised.

## Runtime

One trial is 1,200 steps (1 minute at 20 Hz).
The demo reaches full score in about 880 steps (44 seconds).

## Fidelity

Full size and mass, rigid boxes on joints.
The arms turn on free rivets (0.004 N m).
Each bar also bends out of the triangle's plane about its rivet end on a spring (20 N m/rad, 170 N/m at the tip), so the pin snaps into the slot by contact alone, about 0.4 N at the pin; nothing is detected or scripted.
The swivel is a free joint with stops at 0 and a quarter turn and 0.08 N m of friction.
The round 8 mm pin is modelled as the square inscribed in its circle; the slot is sized to the round pin.
Bar faces have friction 0.8 (0.65 against the gripper's pads), the pin and slot rim 0.3.
Published by the maker: 438 x 420 mm set up, 32 mm thick folded, 1.64 kg.
Estimated from the maker's drawing, photos and instruction sheet (measure the bought part): bar width 56 mm and thickness 8 mm, the order of the bars in the pack, base 356 x 32 x 30 mm and 1.3 kg, feet, rivet and swivel friction, pin and slot position and size, the slot's bevelled rim, and the bars' bending stiffness.
The reflectors are colour only.
This model of triangle has no wire stay.

## Real-world build

- One triangle from a Deflecto 73-0711-00 Emergency Warning Triangle Kit (or any FMVSS 125 triangle of the same pattern: three bars, two rivets, pin and slot at the apex, weighted swivel base).
- A table with a yellow tape outline 37.6 x 5.2 cm for the base, centred 30 cm in front of the midpoint between the arm bases, long side pointing away from the robot.
- Fold the triangle, turn the pack along the base and set the base in the outline with the pin arm's rivet at the far end.
- Check before the first trial and once per session:
  - an arm raised to 60 degrees and let go falls back flat (if a rivet holds an arm up, work it loose);
  - a spring scale hooked at the end of the bottom bar turns the pack on the base with about 0.4 N, and the swivel stops at a quarter turn;
  - with one arm held at 60 degrees, a spring scale pushing the other arm's tip toward it reads under 2 N as the pin snaps in.
- Measure the bought triangle and replace the estimated numbers in `task.py` (top of the file) if they differ by more than 2 mm.
- Nothing here is hazardous; the triangle weighs 1.64 kg and the only moving parts are its two arms.

## Reset

Spring the two arms apart at the apex to lift the pin out of the slot, fold both arms down along the bottom bar (the pin arm last, on the outside), turn the pack back along the base to its stop and set the base in the outline.
