# Reset a tripped breaker under a spring-shut cover

A grey service panel stands on the table with four breakers in a row. Each has a white body with a red end (ON) and a
green end (OFF) and a black toggle that slides along x between them. Three breakers are ON. The fourth has tripped: its
toggle sits midway. A grey cover plate lies just above the toggles and is pulled shut by a spring: it can only be
reached while it is pushed back about 14 cm by a tab on its front edge, and it slides shut again the moment it is let
go. The robot has to hold the cover open with one arm and reset the tripped breaker with the other, without switching
any other breaker off.

Resetting a tripped breaker is routine electrical and facilities work (O*NET 49-9071.00). The breaker is trip-free:
pushed straight to ON it springs back to the middle and stays tripped, so the reset is a two-stage move: fully OFF
(past -8 mm), then ON (past +8 mm). The cover forces the second arm: with one arm the toggles cannot be reached.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: panel, toggles, cover, the breaker logic (`on_step`), `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts built from boxes, with spring-loaded sliders, for yam_arena task files |
| `demo.py` | The scripted full-score demo (open loop on the fixed layout, both arms) and its negative variants |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "On the de-energised training panel, reset the tripped breaker behind the grey cover and leave every breaker on. ON is the red end of each breaker." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/breaker-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                        # full score: demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=on_only python demo.py task.py on_only --no-record        # push breaker 3 toward ON only: it springs back
DEMO_VARIANT=off_only python demo.py task.py off_only --no-record      # push it OFF and stop
DEMO_VARIANT=cycle_neighbour python demo.py task.py cyc --no-record    # cycle breaker 4 off and on first
DEMO_VARIANT=wrong_off python demo.py task.py wrong --no-record        # switch breaker 4 off, then reset breaker 3
DEMO_VARIANT=one_arm python demo.py task.py one --no-record            # open the cover, let go, work over the shut cover
```

1. The right arm closes its fingers and pushes the cover's tab back until the toggles are uncovered, then keeps its
   fingers against the tab.
2. The left arm, fingers closed, pushes breaker 3's toggle from its +x face to -13 mm (OFF), backs off, lifts over it,
   comes down on the -x face and pushes it through the middle to +13 mm (ON), at a fixed rate with fixed
   dwell times.
3. The left arm lifts away and the right arm lets go: the cover slides shut.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 333 steps (17 s). The
negative variants score: `on_only` 0.0 (the toggle springs back each time), `off_only` 0.5, `cycle_neighbour` 0.75
(breaker 4 was off for a while), `wrong_off` 0.0 (breaker 4 left off) and `one_arm` 0.0 (the cover shuts again, and the
left arm only rides over it).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). A toggle's position is
measured along x from its middle, positive toward the red end.

- 0 if the panel moved 2 cm or turned 5 degrees, a breaker other than breaker 3 is off at the end, or breaker 3's toggle
  never went 5 mm toward OFF.
- 0.25: pushed toward OFF but still tripped at the end.
- 0.5: reset to OFF but ends OFF.
- 0.75: breaker 3 ends ON but another breaker was off at some point.
- 1.0 (success, which must hold for 1 s): breaker 3 ends ON, no other breaker was ever off, all toggles still for 1 s.

A breaker is off once its toggle passes -8 mm and on again past +8 mm.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps on an L40S-class GPU, plus about 1 minute for Isaac Sim to start (6
minutes on a fresh instance). The layout is fixed (`-E object_jitter=0`); there are no seeds to choose.

## Fidelity

The parts are rigid bodies built from boxes. Each toggle is a prismatic joint (+-14 mm) on a 60 N/m position spring;
the cover is a prismatic joint with a 20 N/m spring. The trip-free logic is simulated: a toggle passing -8 / +8 mm
switches its breaker and the spring target follows (on +12 mm, off -12 mm, tripped 0). No click or contact-force effects.
The panel is heavy (3 kg).

## Real-world build

A printed or sheet-metal panel with four rocker or slide switches (white bodies with red and green ends), a trip-free
mechanism on one of them (a ratchet that returns it to the middle unless it is first driven to OFF), and a hinged or
sliding weatherproof cover on a spring of about 20 N/m with a tab on its front edge.
