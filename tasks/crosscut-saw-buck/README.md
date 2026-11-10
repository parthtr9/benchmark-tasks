# Buck a log with a two-person crosscut saw

A pine log lies on a small sawbuck with a white chalk mark across it and a starter kerf in the mark. A two-person
crosscut saw stands in the kerf: a thin steel blade with a vertical beech handle at each end, one in front of each arm.
The robot has to saw through the log with one hand on each handle, then let go of the saw. The blade can only be
pulled: a handle pushing it kinks it.

Forest and conservation workers and trail-crew sawyers (O*NET 45-4011.00) buck logs this way. A crosscut blade cuts on
the pull and buckles on the push, so the two sawyers pull it in turn from opposite ends while the other lets their end
follow. One arm alone can make one stroke; bringing the blade back needs a pull from the other end, and a push kinks
it. The task tests two arms taking turns on one tool, the follower yielding exactly enough that its handle never pushes.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the saw's pull-only rule, `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts built from boxes, with sliders and hinges, for yam_arena task files |
| `demo.py` | The scripted full-score demo (open loop) and its negative variants |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Saw through the log at the chalk mark with the two-person crosscut saw, one hand on each handle, then let go of the saw. The thin blade can only be pulled: it kinks if either handle pushes it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/saw-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                                    # full score
DEMO_VARIANT=one_arm python demo.py task.py one_arm --no-record                     # left pulls once, then pushes back
DEMO_VARIANT=one_arm_pull python demo.py task.py one_arm_pull --no-record           # left pulls once and stops
DEMO_VARIANT=anchor python demo.py task.py anchor --no-record                       # right holds still, left pulls
DEMO_VARIANT=push_stroke python demo.py task.py push_stroke --no-record             # the follower's handle pushes
DEMO_VARIANT=short python demo.py task.py short --no-record                         # four strokes
DEMO_VARIANT=no_release python demo.py task.py no_release --no-record               # hands stay on the handles
DEMO_DEBUG=1 DEMO_VARIANT=... python demo.py ...                                    # trace in the log and result.json
```

Open loop from the fixed layout: both arms close on their handles (fingers across x); eight strokes of 2 s with 0.5 s
pauses, the left arm pulling first; the follower's command trails the leader's by 0.1 s and sits in the middle of its
8 cm of play; each leader takes up its slack in the pause. Then both open and lift 5 cm. The arms stay 0.35 m or more
apart (hull distance, checked on the URDF collision meshes along the whole commanded path; the figure is logged in
`result.json`).

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 586 steps (29 s). The negative variants score: `one_arm` 0.0 (the left arm's push kinks the blade), `one_arm_pull` 0.0 and `anchor` 0.0 (no stroke completes without the other hand), `push_stroke` 0.0 (kink), `short` 0.25 (four strokes) and `no_release` 0.75 (cut through, hands still on the handles). In the demo the arms lag their commands by at most 13 mm, the follower's link stays at least 36 mm clear of its pushing end, and the smallest distance between the arms' collision hulls and the other arm's handle is 497 mm.

## Scoring

Evaluated every control step; the score at the end counts (`SCORE_OVER = "end"`).

- A stroke is the blade travelling 8 cm or more to one side with a gripper within 3 cm of each handle; strokes
  alternate sides by themselves (a second pull to the same side needs the blade brought back first). Eight strokes cut
  through.
- 0 if the blade ever kinks (it moves away from a handle at over 1 cm/s while that handle's link is within 3 mm of its
  pushing end), a gripper is ever within 1 cm of the blade, or fewer than 2 strokes were made. Permanent.
- 0.25 for 2-4 strokes, 0.5 for 5-7.
- 0.75 for 8 strokes but a gripper is still within 3 cm of the saw or the saw is not still.
- 1.0 (success, which must hold for 1 s): 8 strokes, no flag, every gripper 3 cm or more from the saw, and the blade and
  both handles moving less than 1 mm over the last 20 steps.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S-class GPU, plus about 1 minute for Isaac Sim to start. The
layout is fixed (`-E object_jitter=0`); there are no seeds.

## Fidelity

The parts are rigid. The saw's rule is scripted in `task.py`: the blade follows a handle pulling at the far end of
its 8 cm link and is pushed by one at the near end; a push over 1 cm/s kinks it, permanently. The blade rides on a
slide in the kerf, with no real cutting or drag; the descent is not simulated: depth is counted per stroke and the
kerf widens 1 mm per stroke. No contact-force cap is enforced.

## Real-world build

A 10 cm pine branch on a printed sawbuck; a thin spring-steel blade 50 cm long with two printed handles joined to the
blade ends through 8 cm slotted links; a cardboard shroud so the blade cannot be pushed through the log.
