# Lock a bayonet cap into its socket

A light grey box stands on a narrow foot, with a round socket near its front end: a 26 mm opening in a 4 mm roof,
with two slots for the cap's lugs, over a chamber with a spring plate. A blue cap (barrel, square grip head, two orange
lugs) stands beside it. The robot must line the lugs up with the slots, push the cap down against the spring until the
lugs clear the roof, and turn it a quarter turn counterclockwise until it stops. The spring then holds the lugs against
the roof's underside, so the cap is locked.

Bayonet mounts close battery compartments, lamp sockets, filter housings and quarter-turn connectors. The task tests
something threading does not: aligning a part about its axis to a keyed opening, then combining a sustained push with
a wrist turn to a stop. The box is loose on a narrow foot, so a rough push or turn knocks it out of place (score 0).
The demo steadies it with the left arm; the same motion with the left arm idle knocked the box out of place and scored
0. A gentler single-arm strategy might still succeed: once the held cap is in the socket's 1 mm fit, it partly steadies
the box itself.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes and cylinders, with spring-loaded sliders, for yam_arena task files |
| `demo.py` | The scripted full-score demo (open loop from the fixed layout and one look at the box's pose; never task state) |

`partkit.py` hooks `yam_arena.scene` and the embodiment's reset without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pick up the blue cap by its square head, line its orange lugs up with the slots in the round socket on the grey box, push it down, and turn it a quarter turn counterclockwise until it stops." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/bayonet-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                  # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_NO_BRACE=1 python demo.py task.py nobrace    # the same push and turn with the left arm idle
DEMO_VARIANT=no_twist python demo.py task.py out  # push and let go without turning
DEMO_VARIANT=no_push python demo.py task.py out   # turn without pushing past the roof
```

1. The right arm grips the cap's square head, lifts it, and holds it ahead of the box with its lugs turned to the slots.
2. The left arm grips the box's long sides 3.5 cm behind its centre, its wrist leaning away from the right arm.
3. The right arm moves over the socket, waits 1 s for the brace to settle, takes one look at the box's pose, and from
   then on works open loop: it lowers the cap into the slots, pushes it 4 mm past the roof's underside (at most
   5 mm/s; 2 mm more than needed, for slip in the grip), turns it 95 degrees counterclockwise against the stop, eases
   off so the spring seats the lugs under the roof, lets go, and both arms lift clear.

The demo uses only what a policy could see: the fixed layout in the first camera frame (the task's constants,
with `object_jitter=0`), one look at the box's pose once the left arm's grip has settled (the box is a fixed part in
plain view of the cameras, and gripping it nudges it a few mm) and its own arm commands. Everything after that look is
open loop; it never reads the task's internal state, slider positions or the cap's depth in the socket.

The demo reaches 1.0 (success) in 636 steps (32 s).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 0 if the box was at any point moved more than 3 cm, tilted more than 20 degrees or turned more than 20 degrees.
- 1.0 (success, which must hold for 1 s): the cap is locked, with its axis within 3 mm of the socket centre, upright
  within 10 degrees, lugs turned more than 80 degrees from the slots and held up against the roof's underside (within
  1.5 mm of that height), let go (every finger pad of both arms at least 3 cm from the cap) and still (within 1 mm
  and 2 degrees), both over the whole last second. A locked cap still held or moving at the end scores 0.75.
- 0.75: with the lugs pushed below the roof, the cap was turned more than 45 degrees.
- 0.5: after being lifted, the cap was pushed in with its lugs below the roof's underside.
- 0.25: the cap was lifted 3 cm.
- Otherwise 0.

The lug angle is measured modulo 180 degrees because the cap has two-fold symmetry; the stops enforce the direction.
Wrong actions don't score 1.0. Measured with the demo variants: pushing the cap in and letting go without turning it
scores 0.5 (it rises back out on the spring); turning it without pushing past the roof scores 0 (the slot walls hold
the lugs, and the forced turn drags the box out of place).

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S, plus about 1 minute for Isaac Sim to start (6 minutes on
a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes and a cylinder. The lugs, slots, roof and stops are real colliders, so
locking comes from contact alone; the spring under the cap is a prismatic joint with a spring drive and no preload.
Contact offsets are 0.5 mm (below the 1 mm fits) and light angular damping (1.0) is added to the parts so they don't
chatter. A 30 g part's instantaneous velocity is noisy in simulation, so "still" is judged from its pose over 1 s. No contact-force
cap is enforced in simulation.

## Real-world build

All parts are 3D-printed PLA with the dimensions in `task.py`; the plate needs a compression spring of about 1.5 N/mm
with at least 6 mm of travel.
