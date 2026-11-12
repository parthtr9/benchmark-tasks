# Snap a scored wall tile over a rod

A glazed ceramic wall tile (10 x 24 x 0.7 cm, 0.38 kg) is scored across its middle and laid score-down over a steel rod
(8 mm) on a plywood cutting board. Tile setters snap such tiles by pressing both halves down at once with the score over
a nail or rod. The tile is modelled as two 0.19 kg halves on a hinge along the score: it only bends ends-down, a spring
(6 N m/rad) holds it flat, and at 3 degrees of bend (about 3 N at each end, 0.10 m from the rod) it snaps for good, after
which the hinge swings freely. The tile starts tipped toward +y (the +y end on the board, the -y end raised about 1.5 cm).
The robot must snap it along the score without lifting it off the rod, then let go.

Two arms are needed by construction: pressing the raised end only rotates the whole rigid tile on the rod (the other end
rises, nothing bends), pressing the end already down does nothing, and a single gripper opened to straddle the score
chips it (a finger within 6 cm of the score at the snap scores 0.25). Bending needs one end held down while the other
is pressed (O*NET 47-2044, Tile and Stone Setters).

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: board, rod, hinged tile, the snap and flags hook, `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts built from boxes with revolute joints, for yam_arena task files |
| `demo.py` | The scripted full-score demo (open loop), its negative variants and the debug trace |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The tile is scored across its middle. Snap it in two along the score over the steel rod, without lifting it off the rod, then let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tile-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200
DEMO_VARIANT=one_arm python demo.py task.py one_arm --no-record      # right presses the raised end only: it rocks
DEMO_VARIANT=left_only python demo.py task.py left_only --no-record  # left presses the +y end only: nothing bends
DEMO_VARIANT=straddle python demo.py task.py straddle --no-record    # one gripper, open 6 cm, over the score
DEMO_VARIANT=no_release python demo.py task.py no_release --no-record
DEMO_VARIANT=short_press python demo.py task.py short_press --no-record
DEMO_DEBUG=1 python demo.py task.py trace --no-record                # trace: arm lag, hinge, tilt, fingers vs the zones
```

The demo is open loop (layout, timing, the arms' own commands only): both arms close their fingers and move over
(0.29, +-0.10); the left arm lowers at 2 cm/s onto the +y end and holds; the right arm lowers the same way on the -y end
past the snap point (the board stops the tile); hold 0.5 s, lift both 6 cm, wait. The press zones are 55 mm long.

Results: in the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 244 steps (12.2 s). The negative variants score: one_arm 0.0, left_only 0.0, straddle 0.0, no_release 0.5, short_press 0.0.

## Scoring

Evaluated every control step; the score at the end counts (`SCORE_OVER = "end"`).

A clean snap is exactly two pieces, the fracture within 2 mm of the score line across the whole width, and no chip
wider than 3 mm. In simulation the halves cannot chip, so its two causes are checked instead: a finger within 6 cm of
the score at the snap, or the score more than 3 mm off the rod's top line before it.

- 1.0 (success, which must hold for 1 s): the tile snapped (3 degrees of bend), cleanly, both halves lie on the board,
  both grippers are at least 3 cm from them, and the pieces moved less than 1 mm over the last 20 steps.
- 0.5: clean snap, but a gripper is within 3 cm of a piece, a piece is not on the board, or something still moves.
- 0.25: snapped with a finger within 6 cm of the score at that moment, or with the score ever more than 3 mm off the rod's
  top line before the snap (a ragged edge; kept for good).
- 0: never snapped, or a piece left the board (for good).

## Runtime and seeds

About 1 minute per trial plus Isaac Sim start-up. The layout is fixed (`-E object_jitter=0`).

## Fidelity

Rigid bodies. The tile is two boxes on a hinge along the score, ends-down only, spring 6 N m/rad (about 3 N at 0.10 m from the rod bends it 3 degrees). At 3 degrees a script makes the break permanent: the halves stay joined by the hinge, whose drive is then driven toward a new target (9.7 degrees, about 1.5 N per half) so they lean down onto the board. Nothing fractures or chips, so a ragged break is judged from its causes. The board is a 30 kg rigid body. No contact force cap.

## Real-world build

- Tile: one material and size throughout, a glazed ceramic wall tile 100 x 240 x 7 mm (0.38 kg), all tiles from the
  same box.
- Score: a straight groove across the middle of the glazed face, cut with a 1 mm diamond wheel on a tile saw with a
  depth stop, 6.0 mm deep, leaving a 1.0 +- 0.1 mm web (check with a depth gauge at both edges and the middle). A
  hand-scored 7 mm tile needs far more force than these arms have; the groove depth is what sets the break force.
- Break-force check: snap three tiles from each prepared batch by hanging weights 10 cm from the rod on both ends;
  they should break between 2.5 and 3.5 N per end. Adjust the web by 0.1 mm and repeat if not.
- Rod and board: an 8 mm steel rod, 14 cm long, taped along the middle of a 20 x 32 x 1.2 cm plywood board clamped
  to the table.
- Judging a trial: count the pieces (exactly two), lay a ruler along the fracture (within 2 mm of the groove across
  the whole width) and look for chips (none wider than 3 mm); then the positions as in the scoring.
- Reset: a fresh tile from the batch for every trial.
