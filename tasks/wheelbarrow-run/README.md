# Wheel a loaded wheelbarrow to a line

A green mini wheelbarrow stands on its wheel and two rear legs straight ahead of the arms, four red clay bricks side by
side in its tray and its two round handle grips pointing back at the arms. An orange tape line crosses the table 12 cm
ahead of its wheel. The robot must lift the barrow by both handles, wheel it forward until the wheel is on the line,
and set it down on its legs without spilling the bricks.

Moving bricks, soil and concrete by wheelbarrow is daily work for construction laborers and landscapers (O*NET
47-2061.00, 37-3011.00). The task tests a coordinated two-arm lift of a load that pivots on a single wheel: the arms
must raise the handles together so the barrow balances on its wheel, keep it level while it rolls, and lower it
together. The mechanism makes it two-handed. Lifted by one round grip, the loaded barrow rolls about that handle onto
its other side; pushed without lifting, its legs drag along the table; lifted by the middle of the tray, its legs drag
too. All of those score 0.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes and cylinders, with hinges and spring-loaded sliders, for yam_arena task files |
| `demo.py` | The scripted full-score demo (open loop from the fixed layout) |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs
with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lift the wheelbarrow by both handles, wheel it forward until its wheel is on the orange line, and set it down on its legs without spilling the bricks." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/wheelbarrow-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                               # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=one_handle python demo.py task.py out --no-record   # lift and push by the right handle only
DEMO_VARIANT=no_lift python demo.py task.py out --no-record      # grip both handles and push without lifting
DEMO_VARIANT=tray_grip python demo.py task.py out --no-record    # lift and push by the middle of the tray
```

The demo is open loop from the fixed layout in `task.py`; nothing measured from the barrow or the bricks steers it.

1. Each arm grips its handle grip from above.
2. Together they raise the grips 4.5 cm: the barrow pitches about its axle and balances on its wheel, legs about
   2.5 cm up.
3. They push it 12 cm forward; the wheel rolls onto the line.
4. They lower it onto its legs, let go and lift clear.

The arms' grip is turned whichever way (0 or 180 degrees about the vertical) their IK reaches best.

The demo reaches 1.0 (success) in 346 steps (17 s), with IK within 0.1 mm of every target, and the two arms' links
never come closer than 216 mm (`min_arm_link_gap_mm` in `result.json`): the grips are 28 cm apart so the arms work
either side of the tray without touching. The three variants each score 0 in Isaac Sim: by one handle the barrow rolls
about it and the other leg drags; without a lift both legs drag; by the tray the fingers touch it off the grips.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 0 if at any time the barrow tipped more than 20 degrees sideways, a brick left the tray, a leg slid more than 2 cm
  along the table (summed while it touches the table), or a gripper touched the barrow anywhere but its two handle
  grips (any finger pad within 12 mm of the frame, tray or legs). These never clear.
- 1.0 (success, which must hold for 1 s): standing on both legs with the axle within 2 cm of the line and 3 cm of the
  centreline, square within 10 degrees, every brick in the tray, both arms' finger pads at least 3 cm from the barrow,
  and the barrow still (within 1 mm over the last second).
- 0.75: it was balanced on its wheel (both legs at least 15 mm up) with the axle within 2 cm of the line.
- 0.5: it was wheeled 6 cm forward balanced on its wheel.
- 0.25: it was balanced on its wheel.
- Otherwise 0.

The 0.25, 0.5 and 0.75 milestones are kept once reached, as the scoring text says ("was"); a violation at any time
still scores 0.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S, plus about 1 minute for Isaac Sim to start (6 minutes on
a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes and cylinders; the line is fixed tape 0.6 mm thick. The wheel spins on a
free axle (limited to 6 radians either way, about 18 cm of rolling, more than the 12 cm run) and rolls by contact
friction, so the barrow steers and resists sideways sliding like a real wheel. The bricks are loose. The handle grips
are round, so a single pinch can't stop the loaded barrow rolling about its handle. Contact offsets are 0.5 mm and
light angular damping is added to the barrow. No contact-force cap is enforced in simulation.

## Real-world build

A model wheelbarrow, 36 cm long: a steel or aluminium frame with two legs, a printed PLA tray (14 x 14 x 3 cm), a
6 cm rubber-tyred wheel on a free axle and round 2 cm handle grips 28 cm apart, 350 g in all (dimensions in `task.py`). Four
6 x 3 x 3 cm clay bricks (100 g each; miniature building bricks are sold in this size). The line is orange tape.
