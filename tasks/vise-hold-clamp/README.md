# Hold a block in a vise while the jaw closes

A grey bench vise stands on the table, turned so its 7 cm gap opens sideways toward the right arm. A light wooden block
stands on the table to the right of it. The robot must carry the block into the gap with one arm and hold it upright so
that its top is 25 to 45 mm above the tops of the jaws, push the vise's sliding jaw shut on it with the other arm, then
let go of both. The clamp, not the hands, has to keep the block up.

Machinists hold a part at height in the open jaws of a vise with one hand while they close the jaw with the other
(O*NET 51-4041.00). The gap has no floor at working height: the base is 15 mm off the table and the jaws stand 115 mm,
so a block let go before the jaw has closed falls to the base. The task tests two arms with different jobs at once:
one holds a part still at a set height (top 25-45 mm over the jaws, standing square) while the other drives a quick-action
clamp shut with a fixed-distance push on a tab. The release order is part of the task: the pushing arm backs off, then the
holding arm lets go.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the jaw's ratchet (`on_step`), `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts built from boxes, with spring-loaded sliders and hinges (adds `set_slider_target`), for yam_arena task files |
| `demo.py` | The scripted full-score demo (open loop from the fixed layout) and its negative variants |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hold the wooden block upright between the vise jaws so its top stands 25 to 45 mm above the tops of the jaws, push the grey sliding jaw shut against it with the other arm, then let go of both." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/vise-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                                  # full score: demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=release_early python demo.py task.py release_early --no-record      # right arm lets go before the jaw is shut
DEMO_VARIANT=no_close python demo.py task.py no_close --no-record                # jaw never pushed
DEMO_VARIANT=one_arm python demo.py task.py one_arm --no-record                  # only the left arm works
DEMO_VARIANT=high python demo.py task.py high --no-record                        # block held with its top 60 mm over the jaws
```

1. The right arm grips the block by its upper 25 mm, carries it low (clear of the jaw tops) to the open end of the gap,
   slides it in down the middle, lowers it to a top 35 mm over the jaws and moves it against the fixed jaw.
2. The left arm, fingers closed, comes to the push tab on the sliding jaw's outer side (a tall tab, 70 mm from the block
   and 41 mm over the jaw tops) and pushes it at 15 mm/s, a fixed 55 mm (open loop, from the layout), then holds 1 s: the gap closes to
   the block's width and takes 2 mm of preload. The left arm backs off.
3. The right arm opens and lifts away. The ratchet holds the jaw, and the clamp holds the block.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 414 steps (21 s).
The negative variants score: `release_early` 0.25 (the block drops to the base before the jaw closes), `no_close`
0.25 (the jaw is never pushed and the block drops when the right arm lets go), `one_arm` 0.0 (the right arm
never brings the block) and `high` 0.5 (clamped, but its top is 60 mm over the jaws, outside the band).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 0 if at any point the vise moved 2 cm or turned 5 degrees, or at the end the block is not between the jaws.
- 0.25: the block is between the jaws (even dropped onto the base) but not clamped.
- 0.5: clamped (jaw gap within 1 mm of the block's 5 cm width, block bottom 5 mm or more above the base) but its top is
  not 25 to 45 mm above the jaw tops.
- 0.75: clamped with its top in that band, but the block is tilted or turned more than 5 degrees from the vise.
- 1.0 (success, which must hold for 20 scorer calls = 1 s): clamped, in the band, square, and still. Whether the arms
  have let go is not measured directly; the arms are in the way in any trial where the block is not still and clamped.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S-class GPU, plus about 1 minute for Isaac Sim to start (6
minutes on a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The vise and block are rigid bodies built from boxes. The sliding jaw is a prismatic joint with a 3 N/mm spring drive
and 50 N s/m damping; a ratchet resets its drive target to its most closed position each control step and, once the
jaw has closed on the block, adds a 2 mm preload (about 6 N). Clamping force comes from friction only (block 0.9, jaws
1.0). The vise and block are loose and can be shoved. No contact-force cap is enforced in simulation.

## Real-world build

A small bench vise (a quick-action or ratchet-jaw type) with 10 cm jaws on a 20 x 22 cm base weighted to 6 kg, with a
tall tab bolted to the outside of the sliding jaw, and a 4 x 5 x 9 cm pine block.
