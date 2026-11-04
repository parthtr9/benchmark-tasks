# Fit a two-pin lamp tube into its sockets and turn it to lock

An 18-inch two-pin lamp tube lies in two low cradles on the table. Beyond it, two fixed sockets face each other 44.6 cm
apart. Each has a narrow slot running from its top down to a pocket. The robot must lift the tube with both arms,
lower it between the sockets so the two pins on each end go down the slots into the pockets, and turn the whole tube a
quarter turn. The pins then lie across the slots, and a spring plunger in each end cap clicks into a hole in the
socket, holding the tube in place.

Replacing two-pin fluorescent and LED tubes is routine building maintenance. Two arms are needed by construction:

- **Both ends at once:** both ends must go into their sockets together, and the whole tube must turn as one.
- **The middle is out of reach:** over the sockets, neither arm can reach the tube's middle.
- **One grip is not enough:** a single parallel grip on a light round tube lets it twist and swing about the grip. In
  the scripted check below, one arm gripping as near the middle as it reaches lifts the tube, but the tube twists in
  the grip and drops, scoring 0.25.

The task tests carrying one long object with two arms, lining up an insertion at both ends together, and turning the
object about its own axis with both hands.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes and cylinders, with spring-loaded sliders, for yam_arena task files |
| `demo.py` | The scripted full-score demo (closed loop on the measured tube pose) |

`partkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lift the white lamp tube with both hands, lower it between the two grey sockets so the pins on each end go down the slots, and turn the tube a quarter turn so it locks in." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/lamp-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200   # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_ONE_ARM=1 python demo.py task.py onearm       # the right arm alone, gripping 10 cm off the middle (0.25)
```

1. Both arms grip the tube 15 cm either side of its middle, fingers closing across it, and lift it out of the cradles.
2. They carry it over the sockets, its pins upright and its axis 3 cm above the pockets.
3. Closed loop on the measured tube pose, they line each end up over its slot and lower it at up to 5 mm/s until both
   ends' pins are in the pockets and the end caps rest on the ledges.
4. Both turn the tube a quarter turn together about its axis, until the plungers click into their holes.

The demo reaches 1.0 (success) in 460 steps (23 s).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 1.0 (success, which must hold for 1 s): both ends' pins are in the pockets (each end's axis within 3 mm of its
  pocket's centre, the tube centred between the sockets within 3 mm), the tube is turned at least 75 degrees from
  pins-upright, and it has moved less than 1 mm over the last second.
- 0.75: with both ends in the pockets, it was turned at least 45 degrees.
- 0.5: both ends' pins were put down into the pockets.
- 0.25: the tube was lifted 3 cm.
- Otherwise 0.

The pins can't come back up the slots once the tube is turned more than about 10 degrees. The 75-degree threshold
stands in for the full quarter turn a real holder needs before its contacts close.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S, plus about 1 minute for Isaac Sim to start (6 minutes on
a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes and cylinders. The plungers are prismatic joints with spring drives and no
preload. The sockets' turning rotors are modelled as fixed square pockets the pins turn in, and the pins carry no
electrical contact. The tube is a light rigid body (no glass breakage). Friction is uniform and light angular damping
is added to the tube. No contact-force cap is enforced in simulation.

## Real-world build

The tube is a standard 18-inch two-pin (G13) tube or a printed dummy of the same size. The sockets are two G13 lamp
holders, or printed blocks with the dimensions in `task.py`, screwed to a board 10 cm above the table, 44.6 cm apart
(face to face). Each needs a small spring ball plunger set into the tube's end caps, or a hole for one in the socket.
The cradles are two printed V-blocks.
