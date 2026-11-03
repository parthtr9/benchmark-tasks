# Snap a module onto a DIN rail

A light grey panel stands at the back of the table with a steel top-hat DIN rail (TS35 x 7.5) across it, closed by a
dark end clamp at each end. A blue 22.5 mm DIN-rail module stands upright in a low holder in front of the right arm,
its back turned toward the panel: a fixed hook at the top of its back, an orange spring latch at the bottom. The robot
must hook the top of the module's back over the rail's upper lip, swing the module's bottom in against the rail and
push it until the latch clicks behind the rail's lower lip.

Breakers, relays, power supplies and terminal blocks are mounted this way in every control cabinet, a large share of
an electrician's and panel builder's work (O*NET 47-2111.00, 51-2022.00), and one that factories are trying to
automate. The task tests a motion that straight insertion does not: engaging a hook on one edge, rotating the part
about that hook, then pushing at a point away from it until a spring latch snaps over the opposite edge. Order
matters, and the geometry enforces it: pushed straight in, the hook stops against the front of the upper lip; hung on
the hook and let go, the module's weight swings its bottom in, but the latch rests on the lower lip about 3.5 mm off the
rail until it is pushed; the end clamps stop it being slid on from an end.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes (optionally rotated) and cylinders, with spring-loaded sliders, for yam_arena task files |
| `demo.py` | The scripted full-score demo (closed loop on the measured module pose) |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs
with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pick up the blue module, hook the top of its back over the upper edge of the metal rail on the panel, swing its bottom in against the rail and push it until its latch clicks onto the rail." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/din-rail-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                          # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=no_push python demo.py task.py out --no-record   # hook it and let go without pushing
DEMO_VARIANT=no_hook python demo.py task.py out --no-record   # carry it level to the rail and push, without hooking
```

1. The right arm grips the housing near its top and lifts it out of the holder.
2. It carries the module in front of the rail with its bottom swung 12 degrees away from the panel, lifts the hook
   12 mm above the upper lip, moves it over the lip and lowers it onto the lip.
3. It swings the module to 5 degrees and lets go: the module hangs on its hook, the latch resting on the lower lip.
4. With closed fingers, closed loop on the measured module pose, it pushes the housing's front just below the latch at
   10 mm/s until the latch clicks behind the lower lip, then backs off and withdraws.

The arm's grip is turned whichever way (0 or 180 degrees about the vertical) its IK reaches best at the hooking poses.

The demo reaches 1.0 (success) in 420 steps (21 s), with IK within 0.1 mm of every target.

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 1.0 (success, which must hold for 1 s): the module is mounted: hooked over the upper lip, its back within 1.5 mm of
  the upper lip and 1 mm of the lower lip, square to the rail within 3 degrees, at hanging height, its latch sprung
  back behind the lower lip, between the end clamps, and still (within 1 mm over the last second).
- 0.75: after hanging on its hook, its bottom was pushed to within 2 mm of the lower lip.
- 0.5: after being lifted, it hung on the rail by its hook (the hook's tooth behind the upper lip).
- 0.25: it was lifted 3 cm.
- Otherwise 0.

The 1.5 mm at the top is the hook's clearance: a mounted module's weight swings its bottom in, so the hook's tooth
bears on the back of the upper lip and the top of the back stands up to 1 mm off it. The bottom can only be within
1 mm of the lower lip with the latch tooth behind that lip; hung but not latched, it rests on the chamfer about 3.5 mm
off. Wrong orders don't score 1.0: hooked and let go without a push scores 0.5, and pushed straight in without hooking
scores 0.25 (measured with the demo variants in Isaac Sim, and over 20 trials each in a MuJoCo prototype of the same
parts).

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S, plus about 1 minute for Isaac Sim to start (6 minutes on
a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes; the rail, end clamps, panel and holder are fixed. The hook, the rail's
lips and the latch's tooth (a square back face and a 45 degree chamfer, built from a rotated box) are real colliders,
so the module is held on the rail by contact alone and the latch clicks only when the lip has pushed it down 1.5 mm.
The latch is a prismatic joint with a spring drive (0.8 N per mm, no preload) and its own low-friction material (0.2).
The rail's lips are 2 mm thick, where a real rail's are 1 mm, so a pushed latch can't tunnel through them. Contact
offsets are 0.5 mm and light angular damping (1.0) is added to the module. No contact-force cap is enforced in
simulation.

## Real-world build

The rail is a standard steel TS35 x 7.5 DIN rail cut to 22 cm, screwed to a 1 cm plywood or steel panel clamped to the
table, with a standard end clamp at each end. The module is a 22.5 mm DIN-rail housing (a relay, timer or empty
enclosure of 22.5 x 55 x 90 mm, about 80 g, or a 3D-printed copy of the one in `task.py`); any housing whose clip
fits TS35 works the same way. The holder is two blocks taped to the table either side of the module's base.
