# Play three notes on a lap steel guitar with the tone bar

A lap steel guitar lies flat on a low stand on the table: six strings run along its neck, a chrome-steel tone bar sits on a
felt rest beyond the nut, coloured dots on the neck edge mark frets 3 (red), 5 (green) and 7 (blue), and a tab card in
front of the guitar shows a red, a green and a blue dot in that order. The robot plays the card's three notes on the first
string, one at a time, and puts the bar back on its rest.

This is a two-hand technique by mechanism, not "one hand steadies while the other acts": the bar hand moves to a different
position for every note, in time with the plucks, and sets the pitch; the plucking hand, 30 cm away, sounds it. A lap steel
has no frets, so nothing can be tapped out by one hand: the bar alone makes no sound, and plucking with no bar gives only the
open string (a wrong note). The bar hand must lay the bar on the string at the right spot before the pluck, keep it there
for half a second while the note rings, then slide it on for the next. Steel guitarists (O*NET 27-2042) and instrument techs
who play-test a setup (49-9063) work this way. Wrong notes are permanent.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: guitar, bar, card, the pluck, pitch and note rules (`tick()`), `success()`, `distance()`, `score()` |
| `partkit.py` | Rigid parts built from boxes and cylinders for yam_arena task files |
| `demo.py` | The open-loop scripted full-score demo, its negative variants, the arm-clearance check and the debug trace |

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Play the three notes on the tab card, in order, on the lap steel guitar's first string: for each one lay the steel tone bar across the strings over the neck dot of the card's colour and pluck the string, letting the note ring for half a second. Then put the bar back on its rest and take both hands away." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/lap-steel-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200               # full score
DEMO_VARIANT=one_arm python demo.py task.py v1 --no-record     # right arm plucks three times, bar on its rest: open E (0)
DEMO_VARIANT=bar_only python demo.py task.py v2 --no-record    # bar placed and slid, nothing plucked (0)
DEMO_VARIANT=bar_lifted python demo.py task.py v3 --no-record  # bar 10 mm above the strings at each pluck: open E (0)
DEMO_VARIANT=two_notes python demo.py task.py v4 --no-record   # third note left out (0.5)
DEMO_VARIANT=no_return python demo.py task.py v5 --no-record   # bar left on the strings (0.75)
DEMO_DEBUG=1 python demo.py task.py dbg --no-record            # trace every 10 steps
```

The demo is open loop: targets come from the fixed layout in `task.py` and from timing only.

1. Left arm pinches the bar's handle (a 3 cm block on top of its middle) across y, lifts it off the rest, carries it over the neck and lays it across the
   strings over the red dot, commanded 2 mm below the strings' top so the strings carry it.
2. Right arm, fingers closed, waits near the bridge and sweeps +x across string 1 (6 mm past it, 12 mm short of string 2).
3. After 0.9 s of ringing the left arm slides the bar to the green dot while the right arm lifts, returns and lowers; same for
   the blue dot.
4. Right arm lifts away; left arm lays the bar back on its rest, opens and lifts clear; wait for the 1 s stillness window.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 673 steps (33.7 s). The negative variants score: one_arm 0.0, bar_only 0.0, bar_lifted 0.0, two_notes 0.5, no_return 0.75. Minimum clearance: arm hulls never within 80 mm of each other; held bar at least 135 mm from the right arm (all runs).

## Scoring

See `scoring` in `task.json`. The score at the end of the episode counts. Notes kept once sounded; any wrong note,
dropped bar, pressed guitar or moved guitar is 0 for the rest of the trial.

## Clean motion

The demo checks the offline distance between every link collision hull of each arm and the other arm, and between the
held bar and the right arm, along the whole commanded path (real URDF hulls and IK). The least distances and the step
where they occur are printed in the `CLEARANCE` line of the debug log and stored in `result.json`. The hands stay at least
0.28 m apart.

## Runtime and seeds

About 35 s of motion per trial (700 steps) of a 1,200-step limit. Fixed layout (`-E object_jitter=0`), no seeds.

## Fidelity

Rigid boxes and a cylinder. Strings are physical rods only over the neck, where the bar rides on them; near the bridge
they are painted and the grippers pass over them (a finger crossing the string line at the right height and speed plucks).
Pluck, pitch and notes are scripted rules; there is no sound or tuner. String spacing is widened to 20 mm.

## Real-world build

A lap steel (or a slab of wood with six strings), a 25 mm steel tone bar with a handle block, a phone tuner app to read each note, and a printed
card with coloured dots matching dots on the neck.
