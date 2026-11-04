# Lever up a crate and slide dunnage under it

A closed blue crate stands on the table, too wide to grip. A red pry lever is pinned on a stand in front of it, its
flat toe in a pocket under the crate's front edge and its handle rising back toward the arms. A wooden block lies to
the left. The robot must press the lever down to lift the crate's front edge, hold it, slide the block under the
raised edge with the other arm, then let the lever go so the crate comes to rest on the block.

Freight and stock handlers lever heavy crates up to slip dunnage under them so forks or straps can get beneath. The
task tests two arms doing different jobs at once, each depending on the other: one arm keeps a tool loaded (the
lever, which springs back and drops the crate as soon as it is let go) while the other places a part through the gap
the tool opens. The lever must be pressed along its arc, not just pushed: the knob swings 3 cm toward the robot as it
goes down. Then the order matters: block in first, lever released second.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts built from boxes, with spring-loaded sliders and hinged levers, for yam_arena task files |
| `demo.py` | The scripted full-score demo (closed loop) and its negative variants |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Press the red lever's handle down to lift the front of the blue crate and hold it there, slide the wooden block under the crate's raised front edge with the other arm, then let go of the lever so the crate rests on the block." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/crate-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                                  # full score: demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=no_hold python demo.py task.py no_hold --no-record                  # lever never pressed: block pushed at the closed edge
DEMO_VARIANT=early_release python demo.py task.py early_release --no-record      # lever let up before the block goes in
DEMO_VARIANT=short python demo.py task.py short --no-record                      # block only 1 cm under before the crate comes down
```

1. The right arm grips the knob and presses the lever down to its stop, tracking the measured lever angle each step
   (the gripper pitches with the knob). The toe lifts the crate's front edge.
2. Holding it there, the left arm picks the block up by its back end, brings it low to the crate's front face to the
   left of the lever and slides it 4 cm under the raised edge, sets it down and lets go.
3. The right arm lets the lever back up slowly; the crate comes down onto the block. It lets go and lifts away.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 374 steps (19 s). The
negative variants score: `no_hold` 0.0 (the block meets the closed front edge and the lever is never pressed),
`early_release` 0.25 (the edge was lifted, but it is down again when the block arrives) and `short` 0.25 (the block
goes in only 1 cm, short of the 2.5 cm needed).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). Heights are of the
middle of the crate's front bottom edge above the table.

- 0 if at any point the crate's front edge moved 3 cm across the table, the crate turned 10 degrees or tilted 10
  degrees sideways, its front edge rose more than 5 cm (tipped, not levered), or the lever stand moved 3 cm or turned
  10 degrees.
- 1.0 (success, which must hold for 1 s): the crate rests on the block (its underside within 3 mm of the block's top
  over the block's front end, front edge 1-3.2 cm up), the block lies flat on the table with its front end 2.5-12 cm
  under the crate and within the crate's width, and the lever is let go (within 6 degrees of its rest stop).
- 0.75: the crate rests on the block as above, but the lever is still held down.
- 0.5: at some point the block was 2.5 cm or more under the front edge while that edge was 1.2 cm or more up.
- 0.25: the front edge was lifted 1.2 cm or more at some point.
- Otherwise 0.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S-class GPU, plus about 1 minute for Isaac Sim to start (6
minutes on a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes. The lever is a revolute joint on its stand with a weak return spring
(1.5 N m/rad, enough to hold it on its rest stop) and a lower stop 19 degrees down. It lifts the crate only by contact
between its toe and the roof of the crate's pocket; the crate pivots on its back edge. Crate, stand and block are
loose and can be shoved or tipped. No contact-force cap is enforced in simulation.

## Real-world build

The crate is a printed or plywood box (22 x 26 x 10 cm) weighted to 2 kg, with the pocket cut into its front bottom
edge. The lever is a printed or aluminium bar on a 4 mm steel pin through a weighted printed stand (2 kg), with a small
torsion spring and a printed stop. The block is a 12 x 3.6 x 1.4 cm piece of pine.
