# Release a push-fit fitting and pull the pipe out

A copper pipe is pushed into a brass push-fit coupling that is fixed to a wall plate on the table. A grab tooth inside
the coupling bites the pipe, so it cannot be pulled out until the orange release collet on the coupling's mouth is
pressed in. The robot must press the collet in and hold it there with one arm while the other arm pulls the pipe out,
then lay the pipe down flat on the table, clear of the fitting.

Plumbers and heating technicians do this whenever a section of push-fit pipework is re-routed or replaced. Two arms
are needed by the mechanism: the tooth bites again the moment the collet is let go, so pressing it and then pulling
with the same hand leaves the pipe locked (in the scripted checks below, pulling without pressing scores 0, and
pressing then letting go before pulling also scores 0: the pipe stays locked and the hard pull moves the fitting). The task tests holding a release with one hand while the other
does the work, on a part that gives no visible sign of being locked.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the collet-to-tooth release, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes and cylinders, with spring-loaded sliders, for yam_arena task files |
| `demo.py` | The scripted full-score demo |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs
with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Press the orange release collet on the brass fitting in and hold it there while you pull the copper pipe out of the fitting, then lay the pipe down flat on the table." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/pushfit-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200    # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_NO_PRESS=1 python demo.py task.py nopress      # pull without pressing the collet: the pipe stays (0)
DEMO_EARLY_RELEASE=1 python demo.py task.py early   # press, let go, then pull: the tooth bites again (0)
```

1. The left arm closes its fingers and pushes the collet's near side bar in with them, beside the pipe, until the
   collet is pressed 3 mm, and holds it there.
2. The right arm grips the pipe from above 17 cm from the collet, so the two wrists stay at least 13 cm apart, and
   pulls it out along its axis at 10 mm/s, keeping its measured hand position on the pipe's axis so the pipe does
   not cock in the bore.
3. Once the pipe's end is 1.5 cm out of the mouth the left arm lets go; the right arm lifts the pipe, lays it flat on
   the table in front of the fitting and lets go.

The demo reaches 1.0 (success) in about 550 steps (28 s).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).

- 0 if the fitting was at any point moved more than 5 mm.
- 1.0 (success, which must hold for 1 s): after being pulled out, the pipe lies flat on the table (axis level within
  10 degrees, resting on the table) with every point at least 4 cm from the coupling's mouth, and is still.
- 0.75: after the collet was pressed, the pipe's end came at least 5 mm out of the mouth (pulled 3 cm).
- 0.5: after the collet was pressed, the pipe was pulled 5 mm (past the tooth).
- 0.25: the collet was pressed 2.5 mm.
- Otherwise 0.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S, plus about 1 minute for Isaac Sim to start (6 minutes on
a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes and cylinders. The grab ring is modelled as one tooth on a stiff slider
that drops into a 3 mm groove in the pipe; a real grab ring's teeth bite plain pipe. While the collet (a spring slider)
is pressed 2.5 mm, the task lifts the tooth, standing in for the cone inside a real collet. The bore is square with
1.5 mm of play and no O-ring. The fitting assembly is held by cleats. No contact-force cap is enforced in simulation.

## Real-world build

A real 15 mm push-fit coupling and two lengths of 15 mm copper pipe: a 5 cm stub fixed into a wall plate and the
coupling clamped to a base board, and a 30 cm loose pipe pushed in to the coupling's stop. The base board is screwed
or clamped to the table; a wooden block supports the loose pipe's far end. Scoring is measured with a ruler.
