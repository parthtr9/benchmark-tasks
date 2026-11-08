# Feed a cross-slide table along a 45-degree line

The table of a manual mill stands in front of the arms: a dark grey base with a scribed 45-degree line from a green start
dot to a red end dot (70 mm in x and in y). Two bars cross over it. The blue saddle bar lies along y and slides along x;
the orange cross-slide bar lies along x and slides along y. The pointer is the crosshair where the two bars cross. Each
bar has a black feed handle (a 30 mm post): the saddle's at the left end, the cross slide's at the right. Pushing the left
handle feeds the pointer in x, pushing the right one feeds it in y. Both slides act like lead screws: they hold their place
under about 1 N and move against damping above it. The robot must bring the pointer from the start dot to the end dot by
working both handles at once, in one steady pass, then let go.

Machinists and tool makers (O*NET 51-4041.00) cut a chamfer or an angled slot by turning both feed handles together,
one in each hand. Feeding one axis and then the other cuts a step, and stopping burns the work. This is not the
two-hand press: there the hands press two buttons at one instant and then hold. Here each hand moves along a path that
has to match the other hand's through the whole pass, so the task tests coordination in time, not simultaneity at a
point. One arm alone cannot do it, since the second handle is out of its reach (0.52 m from the other base).

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, lead screws and pass log (`on_step`), `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts built from boxes and cylinders, with spring-loaded sliders and hinges, for yam_arena task files |
| `demo.py` | The scripted full-score demo (open loop) and its negative variants |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this
folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the pointer along the scribed 45-degree line from the start dot to the end dot by working both feed handles at once, in one steady pass, then let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/cross-slide-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200                          # full score
DEMO_VARIANT=one_arm python demo.py task.py one --no-record              # only the left arm feeds
DEMO_VARIANT=x_then_y python demo.py task.py xy --no-record              # one slide, then the other
DEMO_VARIANT=staircase python demo.py task.py stair --no-record          # in step to 40 mm, then 20 mm moves (scores 0)
DEMO_VARIANT=half_speed_pause python demo.py task.py pause --no-record   # a 3 s stop at 35 mm
DEMO_DEBUG=1 python demo.py task.py demo_out                             # fingertip vs handle at close, slides, arm lag
```

The demo is open loop: it uses only the fixed layout, step counts and the arms' own joint readings.

1. Both arms come down on their handles (the left closes across x, the right across y, so the fingers push the post the way
   it travels) and close firmly.
2. Both feed 74 mm together over 6 s (the slides stop at 70 mm; the extra 4 mm covers the arms' lag).
3. Both let go and lift 7 cm.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 291 steps (14.6 s). The negative variants score: `one_arm` 0.0 (the right handle is out of the left arm's reach, so the pointer never gets to the end dot), `x_then_y` 0.0 (peak e of 70 mm), `staircase` 0.0 (peak e of 20 mm, over the 15 mm limit; the 0.25 credit needs both slides 35 mm along with e never over 8 mm, which the lagging arms did not reach in the simulation) and `half_speed_pause` 0.5 (a 3 s stop).

## Scoring

Evaluated every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`). dx and dy are the two
slides' travel (0-70 mm), e = |dx - dy|.

- 1.0 (success, which must hold for 1 s): both slides 68 mm or more, peak e 8 mm or less, no stop of 2 s or longer once
  motion starts, within 15 s of the first 2 mm of motion, both grippers 3 cm or more from both handles.
- 0.5: on the end dot with peak e 15 mm or less, but with a stop of 2 s or longer, a pass over 15 s, peak e over 8 mm, or a
  gripper within 3 cm of a handle.
- 0.25: both slides reached 35 mm with e never over 8 mm, but the pass failed after that.
- Otherwise 0; 0 if the mill moved 1 cm. A stop or an excursion stays on the record: catching up does not erase it.

## Fidelity

Rigid bodies built from boxes. The two bars are prismatic joints on the base (7 cm travel). A lead screw is modelled with
the joint's position drive: a 1,000 N/m spring holds the slide, and the target follows the slide only once it is pushed
more than 1 mm off (1 N breakaway), with 100 N s/m damping while it moves (about 3 N more at 3 cm/s). The handles are
push posts; turning a crank is not modelled, and the bars do not carry each other, so the crosshair stands for the
pointer. No contact-force cap is enforced in simulation.

## Real-world build

A small cross-slide table (or a printed one) with two lead screws or friction-held slides, a plate scribed with a 45-degree
line, and 3 cm posts as feed handles. A ruler and a stopwatch score a trial.
