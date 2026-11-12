# Seat a hot-swap drive carrier with its cam handle

A heavy grey drive bay lies on the table with its mouth toward the arms. A blue drive carrier sits in its channel,
pulled 4 cm out. An orange cam handle on a vertical pin at the carrier's front stands open, pointing at the arms.
The robot must slide the carrier fully in with the handle open, then swing the handle closed across the carrier's
front until it latches.

Replacing hot-swap drives is the most frequent hands-on repair in a data centre, and the storage behind AI
training and inference clusters depends on it; automating it removes a human bottleneck on scaling compute. The handle is how the drive is seated. Near the end of the
carrier's travel, the hook on the handle's short end swings behind a catch post at the bay's mouth. Closing the
handle then levers the carrier the last few millimetres in against the connector's spring and latches it. A carrier
pushed in and let go springs back out. If the handle is closed before the carrier is in, the hook hits the front of
the post and the carrier stops 1 cm short. The task tests operating a hinged lever on the part being inserted, in the
right order: insert with it open, then close it along its arc once the hook can catch. In other insertion tasks the
robot moves the part directly.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the handle's latch (`on_step`), `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts built from boxes, with spring-loaded sliders and hinged handles, for yam_arena task files |
| `demo.py` | The scripted full-score demo (closed loop) and its negative variants |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Slide the blue drive carrier fully into the grey bay with its orange handle open, then swing the handle closed across the carrier's front until it latches, so the drive stays seated." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/drive-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                         # full score: demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=no_close python demo.py task.py no_close --no-record       # shoves the carrier by its face, handle left open
DEMO_VARIANT=close_first python demo.py task.py close_first --no-record # handle shut first: the hook stops at the post
DEMO_VARIANT=short python demo.py task.py short --no-record             # pushed 2 cm only: the hook misses the post
```

1. The right arm grips the pull tab and pushes the carrier in along the open handle (the push runs through the pin,
   so the handle does not turn) until the connector plunger bottoms out.
2. It opens its fingers into a fork around the tab (the carrier springs back a few millimetres). Then it swings the
   tab round the pin, tracking the measured pin position and handle angle each step. The hook levers the carrier
   home and the handle latches.
3. It lets go and lifts away.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 293 steps (15 s). On
the way the hook levers the carrier from f = 2 mm to 9 mm and the handle latches at 85 degrees. The negative
variants score: `no_close` 0.25 (even a direct 2 cm shove on the carrier's front face only gets it to f = 4 mm
against the plunger, and it stays unlatched), `close_first` 0.0 (the hook stops at the post with the carrier 27 mm
out) and `short` 0.0 (the handle shuts in front of the post with the carrier 32 mm out).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). f is the carrier's
front face along the bay, 0 at the mouth and positive inside; the handle's angle is measured from open (pointing at
the arms) toward shut (across the carrier's front, 90 degrees).

- 0 if the bay was at any point moved 2 cm or turned 10 degrees, or the carrier was lifted 12 mm off the bay floor or
  tilted 15 degrees.
- 1.0 (success, which must hold for 1 s): f at least 7 mm (seated; latched it sits at 9 mm) and the handle at least
  85 degrees, f steady to 1 mm over the last second.
- 0.75: the handle reached 45 degrees with f at least -8 mm (the hook is behind the catch post).
- 0.5: f reached 7 mm at some point (pushed home by hand).
- 0.25: f reached -8 mm (far enough in for the hook) while the handle was still under 30 degrees.
- Otherwise 0.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S-class GPU, plus about 1 minute for Isaac Sim to start (6
minutes on a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes. The handle is a revolute joint and the connector plunger a prismatic
joint with a spring drive (800 N/m). The hook levers the carrier by contact with the post alone. Below 85 degrees the
handle's drive holds it where it was left, like a stiff hinge. From 85 degrees on it latches and is driven into its
90-degree stop for the rest of the episode. The bay is heavy so it stays put. No contact-force cap is enforced in
simulation.

## Real-world build

Bay, carrier and handle are 3D-printed PLA with the dimensions in `task.py` (the bay can be weighted or screwed to a
board to about 2.5 kg). The handle turns on a 3 mm steel pin with a ball-detent latch at the shut position. The
plunger is a printed block on a compression spring of about 0.8 N/mm with 9 mm of travel.
