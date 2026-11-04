# Load a press die, then cycle it with the two-hand control

A grey benchtop horizontal press stands on the table: a base plate with an open-topped die running toward the robot,
closed at the front by a dark anvil block, and an orange ram that rests at the back of the die and slides forward along
it. Two green palm buttons sit on the front corners of the base plate, 30 cm apart. A red aluminium blank lies on the
table in front of the press. The robot must set the blank in the die, take both grippers out of the press, push both
buttons at the same time, one with each gripper, hold them until the ram has closed the blank against the anvil, and
let go.

Power presses are cycled with a two-hand control so that the operator's hands are on the buttons, not in the die, while
the ram moves. The control is the safety device; using it correctly is the skill. Here the press behaves like a real
two-hand relay: the ram moves only while both buttons are held, only if the second was pushed within 0.5 s of the first,
and a button held down early (or weighted down) locks the control out until both are released. The task tests three
things a press operator does every cycle and that are not on the board: loading a die, keeping out of the point of
operation while the machine moves, and a synchronised two-arm action on a machine control rather than on an object.
Cycling an empty die, or putting a gripper in the die while the ram is forward, scores 0.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the two-hand relay (`control()`), `success()`, `distance()` and `score()` |
| `partkit.py` | Rigid parts built from boxes, with spring-loaded sliders, for yam_arena task files |
| `demo.py` | The scripted full-score demo and its negative variants |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs
with the standard `yam_arena` embodiment and `task_score` scorer. The relay runs after every control step and sets the
ram's drive target.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the red blank into the die in front of the orange ram, take both grippers out of the press, then push the two green buttons at the same time, one with each gripper, and hold them until the ram has closed on the blank. Then let go of the buttons." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/press-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                   # full score: demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=one_button python demo.py task.py v1 --no-record      # only one button: the ram stays back (0.25)
DEMO_VARIANT=late python demo.py task.py v2 --no-record            # second button 1 s late: the relay refuses (0.25)
DEMO_VARIANT=no_blank python demo.py task.py v3 --no-record        # cycles an empty die (0)
DEMO_VARIANT=weighted python demo.py task.py v4 --no-record        # blank set on one button as a tie-down (0)
```

1. The left arm pinches the blank's top edges across the die's width, carries it over the die, re-aims from where the
   blank really is in its grip just above the rails, sets it down and lifts out of the press.
2. Both arms close their grippers over their buttons and push them in the same 1.5 s move, so they go down within a
   step of each other.
3. They hold for 3.5 s while the ram closes the blank against the anvil, then lift off; the ram returns.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 544 steps (27 s). The
negative variants score: `one_button` 0.25 and `late` 0.25 (the blank is seated but the ram never moves), `no_blank`
0.0 (the ram cycles on an empty die) and `weighted` 0.0 (the control refuses, the die stays empty).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 0, for the rest of the trial, if the press base moved more than 1 cm or turned more than 5 degrees; if the ram moved
  forward more than 1.5 cm with the blank not seated in the die; or if any checked point of a gripper (its pad tips,
  tool point, and 4 and 8 cm up its body) was in the die zone while the ram was more than 1.5 cm forward. The die zone
  runs from the anvil's back to the ram's rest face, 6.6 cm wide, up to 8 cm above the base plate.
- 1.0 (success, which must hold for 1 s): the ram closed the seated blank against the anvil and is back (within 5 mm of
  its rest), both buttons are released, the blank is still seated, both grippers are out of the die zone, and the blank
  and ram moved less than 1 mm over the last second.
- 0.75: the ram closed on the seated blank (pushed it against the anvil, within 3 mm).
- 0.5: the ram started forward (1.5 cm) with the blank seated, which needs both buttons pushed together.
- 0.25: the blank was seated in the die: flat on the base plate within 3 mm, tilted under 5 degrees, its centre within
  4 mm of the die's centreline and anywhere from against the anvil to 6 mm behind its marked place.
- Otherwise 0.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps, plus about 1 minute for Isaac Sim to start (6 minutes the first
time). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes. The ram and buttons are prismatic joints with spring drives; the
two-hand relay is evaluated in the control loop at 20 Hz and moves the ram's position target forward at 6 cm/s while
enabled, and back at 10 cm/s otherwise. The ram's drive is soft (1,500 N/m), so it presses on the blank with a few
newtons. The gripper's position for the die-zone check comes from its joint angles.

## Real-world build

A frame of aluminium extrusion or plywood with the dimensions in `task.py`, a linear actuator or a small pneumatic
cylinder mounted horizontally with a foam-faced 4 cm ram and a force limit under 10 N, and two palm buttons wired to a standard two-hand
safety relay (type IIIA, 0.5 s synchronism). The die zone is checked from video or with a light curtain across the
top and front of the die.
