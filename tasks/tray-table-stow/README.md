# Stow and latch a seatback tray table

A grey seatback stands on the table, reclined so its top leans toward the arms; it is full height (about 20 cm) at the left, where the latch tab sits, and cut down to 14 cm at the right so the hand holding the tray clears it. A light wooden tray table is hinged
at its foot and lies flat toward the arms, with a small tongue at its free edge to hold. High on the seatback an
orange latch tab on a boss rests flat, pointing left. The robot must fold the tray up against the seatback and hold it
there, flip the latch tab down over the tray's top edge with the other arm, and then let go of the tray.

Flight attendants check that every tray is stowed and latched before takeoff and landing. The task tests two arms
doing different jobs under an enforced order: the tray is pulled open by a weak spring and the lean of the seat, so it
falls open the moment it is let go before the latch is over its edge, and a latch flipped first blocks the tray from
folding at all. The tray must be swung along its arc, not pushed.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()`, `score()`, `distance()`, latch mechanism |
| `partkit.py` | Rigid parts built from boxes, with spring-loaded sliders and hinges, for yam_arena task files |
| `demo.py` | The scripted full-score demo (closed loop) and its negative variants |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fold the tray table up against the seat back and hold it there while you flip the latch tab down over its top edge, then let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tray-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                               # full score: demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=one_arm python demo.py task.py one_arm --no-record                # tray folded and released, tab never flipped
DEMO_VARIANT=tab_first python demo.py task.py tab_first --no-record            # tab flipped first, blocks the tray
DEMO_VARIANT=hold_no_latch python demo.py task.py hold_no_latch --no-record    # tray held folded, tab never flipped
```

1. The right arm slides its open fingers along the tray's plane onto the tongue, closes, and swings the tray up along
   its arc to about 80 degrees, tracking the measured tray angle each step.
2. Holding it there, the left arm, fingers closed, presses the tab's tip down round its hinge line until the detent
   snaps it over the tray's top edge.
3. The right arm opens; both arms back out 15 cm and lift away.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 435 steps (22 s). The
negative variants score: `one_arm` 0.25 (the tray is folded and released, the tab never flipped, so it falls open),
`tab_first` 0.0 (the flipped tab stops the tray short of 75 degrees) and `hold_no_latch` 0.5 (the tray is held at 80
degrees but the tab is never flipped).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 0 if the seatback moved 3 cm or turned 10 degrees.
- 1.0 (success, which must hold for 20 scorer calls): tray at 75 degrees or more, tab down 70 degrees or more, and no
  gripper finger within 3 cm of the tray.
- 0.75: the same, but a gripper is still within 3 cm of the tray.
- 0.5: tray at 75 degrees or more but the tab is not down (the arm is still holding it).
- 0.25: the tray reached 75 degrees at some point but ends lower.
- Otherwise 0 (never reached 75 degrees, which includes flipping the tab first).

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S-class GPU, plus about 1 minute for Isaac Sim to start (6
minutes on a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

Rigid bodies built from boxes. The tray is a revolute joint with a weak spring and gravity opening it, stopping at
81.5 degrees. The tab is a revolute joint whose spring target switches at 45 degrees (snap-over detent). A tab flipped
first hangs in front of the tray's edge and blocks it. Once the tray is at 75 degrees or more with the tab down 70
degrees or more, the latch is engaged for good: the tray's hinge drive is stiffened to 1 N m/rad aimed at 80 degrees,
so the tray stays stowed however the arms let go. No contact-force cap is enforced in simulation.

## Real-world build

A plywood or printed seatback (3 cm thick, 30 cm wide, 20 cm tall at the left and 14 cm at the right, weighted to 6 kg) leaning 10 degrees, with a 11 x 24 x 1.2 cm pine tray
on a piano hinge at its foot and a light return spring, and a printed tab on a pin with a small over-centre spring.
