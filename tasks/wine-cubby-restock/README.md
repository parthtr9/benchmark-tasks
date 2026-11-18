# Lay a full wine bottle into a wine rack cubby and push it home

A low pine wine rack with three square cubbies in a row is fixed to the table with its open front toward the robot.
The left and right cubbies hold a full bottle each; the middle one is empty. A third full 750 ml bottle (1.2 kg) lies
on a rubber bar mat in front of the rack, at an angle to it. The robot must put it away in the empty cubby, base first
so the neck faces out, pushed in until its base is on the rack's back panel.

Category: Societal relatability. Putting a bottle of wine away in a wine rack at home is a familiar household chore.
Most wine racks hold bottles lying on their side, and the usual reason given is that the wine then stays against the
cork, so the cork does not dry out and let air in. Whoever unpacks the shopping or a delivery slides each bottle into a
free cubby base first, neck out, and pushes it back. What makes it different from standing a bottle on a shelf:

- The bottle goes in lying down, along its own length, into a 10 cm opening with 12.5 mm of play on each side.
- The hand cannot go in with it. A gripper around the 7.5 cm body is wider than the cubby, so a body grip stops with
  about 7 cm of the bottle inside.
- The bottle balances 11 cm from its base. Let go with 7 cm inside, its weight is still outside the cubby and it
  tips out, neck down. Support has to pass to the rack, or to a second grip further out, before the body grip opens.
- The last part of the way it has to be pushed by its cork end.

## Revision

The review asked: "Please change the domain to societal relatability and explain why this familiar wine-storage
activity is recognizable. Also revise the drop penalty so that a controlled downward movement of a held bottle is not
scored as a drop; specify a video-checkable way to identify actual loss of support or an unintended impact."

What changed:

- Category is now Societal relatability; the context describes putting a bottle away in a home wine rack and why
  bottles are kept lying down, and no longer cites occupations.
- The drop rule now detects loss of support, not speed. An end coming down more than 2 cm in 0.2 s is a drop only
  if it did not move with a gripper that touched the bottle (within 5 mm, checked against points spaced about 5.5 mm
  over each gripper's finger faces and palm) at the start of those 0.2 s: measured in that gripper's own frame, the
  end shifted more than 2 cm. A held bottle can be lowered or tilted at any speed. A bottle let go with its weight
  outside the cubby tips out while the opening hand stays where it is, so it still scores 0. New demo variant `quick_lower` brings the held bottle down 5 cm in 0.2 s and still scores 1.0. Nothing else changed:
  instruction, setup and the rest of the scoring are as before.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: rack, bottles, mat, `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts built from boxes and cylinders for yam_arena task files (here: the bottles) |
| `demo.py` | The scripted full-score demo and its negative variants |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs
with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the wine bottle that lies on the mat away in the empty space of the wine rack, neck out and pushed all the way in." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/wine-cubby-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200         # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=body_only python demo.py task.py body_only   # feed it in 7 cm with the body grip, let go: it tips out (0)
DEMO_VARIANT=no_push python demo.py task.py no_push       # slide it 20 cm in by the neck and leave: not pushed home (0.5)
DEMO_VARIANT=no_release python demo.py task.py no_release # fully in, but the fingers stay on the cork end (0.75)
DEMO_VARIANT=quick_lower python demo.py task.py quick_lower # as the demo, plus the held bottle brought down 5 cm in 0.2 s (1.0)
DEMO_VARIANT=lift_hold python demo.py task.py lift_hold   # payload check only: lift, hold 3 s, lay it back (0)
```

1. The right arm takes the bottle at its balance point with the pads across the body, lifts it, turns it 60 degrees
   so the base points at the rack, and lines it up with the empty cubby.
2. It feeds the bottle in base first until its fingers are 6 mm from the rack (7 cm in) and sets the base down on the
   cubby floor, still holding.
3. The left arm takes the neck. The right arm lets go and moves aside.
4. The left arm slides the bottle in until its own hand is at the rack (20 cm in), stops pushing, lets go and goes
   back to its side: the weight is now inside and the bottle lies level by itself.
5. The right arm comes back, closes its fingers and pushes the cork end until the base is on the back panel (4 cm of
   neck left showing), then backs off 6 cm and lifts away.

Every target comes from the task file's fixed layout and from timing. The only feedback is each arm's own joint
readings (forward kinematics of its joint positions): while an arm carries or pushes the bottle, its command is
trimmed until its measured tool point is on the planned one, which takes out about 3 cm of sag under the 1.2 kg load;
and the sideways shift of the hand while its fingers close on the bottle tells where the bottle sits between the
fingers. The demo reads no object pose (its log does).

In simulation: demo 1.0 (success) in 945 steps (47 s); body_only 0 (the bottle tips out and the drop rule fires),
no_push 0.5, no_release 0.75, quick_lower 1.0 (success) in 1,009 steps (the held bottle came down 5.1 cm in 0.2 s,
with the hand: not a drop).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 0 for the whole trial if a stocked bottle was shifted more than 1 cm, or if the bottle lost support: an end of it
  came down more than 2 cm within 0.2 s without moving with a gripper that was touching it (any part of the gripper
  within 5 mm of the bottle) at the start of those 0.2 s, that is, it shifted more than 2 cm against every such
  gripper (it was dropped, slipped in the grip, or was let go with its weight outside the cubby and tipped out). In
  the 20 fps video: over 4 frames an end of the bottle goes down more than 2 cm and no hand that was on the bottle
  goes down with it. Lowering or tilting the bottle in a hand, however fast, is never a drop. Neither flag is ever
  cleared.
- 1.0 (success, which must hold for 1 s): the bottle lies base first in the empty cubby, level within
  3 degrees, its cork end at most 45 mm out of the rack's front face (40 mm when the base touches the back panel),
  both grippers at least 3 cm from it, and nothing has moved more than 1 mm for 1 s.
- 0.75: the same, but a gripper is within 3 cm of the bottle or it has not been still for 1 s.
- 0.5: at least 15 cm of the bottle is inside the empty cubby, level within 3 degrees, grippers at least 3 cm away.
- 0.25: the bottle's base is at least 2 cm inside the empty cubby.
- Otherwise 0. A bottle put in neck first does not count. Which arm does what is not scored.

All of it can be read off a real trial with a ruler at the rack's front face, a small level on the bottle and the
trial video.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S, plus about 1 minute for Isaac Sim to start (6 minutes on
a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The bottle, cubby and board sizes are those of a real 750 ml Bordeaux bottle and one row of a pine cubby rack. Each
bottle is one rigid body of coaxial cylinders (body, label band, three-step shoulder, neck, foil capsule); its 1.2 kg
is spread evenly over them, which puts the balance point 11 cm from the base, as in a full bottle. The wine does not
slosh. Glass, pine and rubber share one friction value (0.6). The rack is fixed, as when screwed to a wall or counter; the
two stocked bottles are loose. No contact-force cap is enforced in simulation. The bottle is carried by one arm at
its balance point (about 12 N, no tipping moment in the grip); sliding it along the cubby floor takes about 7 N.

## Real-world build

- Rack: 12 mm planed pine (or plywood) boards. A bottom and a top board 34.8 x 26 cm and four uprights 10 x 26 cm,
  giving three cubbies 10 x 10 cm and 26 cm deep in a row; a 6 mm hardboard back panel 34.8 x 12.4 cm nailed on.
  Screw it to the table through the bottom board with its front face 33 cm from the line between the arm bases,
  centred between the arms. One row of any bought cubby rack with 10 cm cubbies about 26 cm deep and a closed back
  does the same job; note its cubby depth and use "cork end at most 5 mm further out than a seated bottle" as the
  fully-in limit.
- Bottles: three full, corked 750 ml Bordeaux-shape wine bottles (30 cm long, 7.5 cm across, 1.2 to 1.3 kg; weigh
  them). Check the balance point by resting one across a pencil: about 11 cm from the base.
- Mat: a black rubber bar mat cut to 12 x 36 cm, 5 mm thick, taped down, with the bottle's outline marked on it.
- Start: two bottles in the left and right cubbies, pushed back to the panel, necks out. The third on the mat, base
  toward the robot's right and the rack, 60 degrees off the rack's depth direction (positions in `task.json`).
- Safety: full glass bottles; put a rubber sheet on the table in front of the rack and keep people out of reach of
  the arms during a trial. No hazardous materials.
- Measuring: a ruler across the rack's front face for how far the cork end stands out, a small spirit level or an
  angle gauge on the bottle, and the 20 fps trial video for the drop rule (an end coming down more than 2 cm within 4
  frames while no hand that was on the bottle comes down with it) and for the stocked bottles (compare first and last frames; a ruler at each cork end confirms 1 cm).
