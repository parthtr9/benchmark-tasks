# Snap a chalk line between two pencil ticks

A sheet of plywood lies on the table with a pencil tick on its left and right edges. A chalk line's hook is caught on
the left edge at the left tick, and the line's reel (a blue case with a nozzle at one end) stands on a pad near the
right arm. The robot must snap a straight chalk line from the left tick to the right tick, then put the reel back on its
pad and let go.

Carpenters (O*NET 47-2031.00) snap a line to mark a long straight cut: one hand pins the reel's nozzle on the far mark so
the line is taut, the other plucks it. A line plucked slack leaves only a smear, and a line pinned off the mark leaves a
wrong line that cannot be erased. The task tests two hands with different jobs at the same time (tension on a mark,
pluck in the middle), and one arm alone cannot do both: the hand that holds the reel is the only thing keeping the line
taut, and once it lets go the line is slack.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the taut/pluck logic (`on_step`), `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts built from boxes for yam_arena task files (here with non-colliding markers for the line and marks) |
| `demo.py` | The scripted full-score demo (open loop from the fixed layout and timing) and its negative variants |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The chalk line's hook is caught on the left edge of the plywood at the pencil tick. Snap a straight chalk line from that tick to the pencil tick on the right edge, then put the reel back on its pad and let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/chalk-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                                  # full score
DEMO_VARIANT=one_arm python demo.py task.py one_arm --no-record                   # pin, let go, pluck slack
DEMO_VARIANT=off_tick python demo.py task.py off_tick --no-record                 # nozzle pinned 10 mm off the tick
DEMO_VARIANT=no_return python demo.py task.py no_return --no-record               # reel set down 10 cm from its pad
```

The demo is open loop: after 1.5 s of settling it uses only the layout in `task.py`, timing and its own joint readings.

1. The right arm grips the reel on its pad, carries it to put the nozzle on the right tick and presses it on the plywood.
2. The left arm lowers open over the line halfway along (pad tips 8 mm over the plywood), closes, lifts 2 cm and opens.
3. The right arm lifts the reel, returns it to its pad, lets go and lifts away.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 589 steps (29 s). The negative variants score: `one_arm` 0.0, `off_tick` 0.25, `no_return` 0.5.

## Scoring

Evaluated every control step; the score at the end counts (`SCORE_OVER = "end"`).

- A snap: a gripper closes within 1 cm of the line and at least 6 cm from both ends, lifts it 1.5 cm or more, then opens.
  It is taut if a gripper is closed on the reel with the nozzle tip within 3 mm of the plywood top, otherwise it is a smear.
  A taut snap draws a mark from the hook to where the nozzle was. Snaps were never undone.
- 0: no taut snap was made, or a taut snap ended more than 15 mm from the right tick.
- 0.25: a taut snap ended 3-15 mm from the tick (even if a good one was also made).
- 0.5: a good taut snap (within 3 mm), but the reel is not on its pad (middle within 2 cm) or a gripper is within 3 cm of it.
- 1.0 (success, held for 20 scorer calls = 1 s): the same with the reel on its pad and let go.

## Runtime and seeds

About 1 minute per trial for 1,200 steps on an L40S-class GPU, plus Isaac Sim start-up. The layout is fixed
(`-E object_jitter=0`).

## Fidelity

Rigid bodies built from boxes. The 2 mm line and the chalk marks are non-colliding markers moved by the task file; the
taut/pluck logic reads the reel pose and the grippers' pad positions every step. No contact force cap.

## Real-world build

A chalk line reel (200 g) on a 12 mm plywood sheet with two pencil ticks 40 cm apart and a hook clipped to the edge.
