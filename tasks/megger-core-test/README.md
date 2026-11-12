# Find the leaking cable core with a hand-crank insulation tester

A hand-crank insulation tester (a "megger") makes its test voltage only while its generator handle turns; the reading
drops away the moment the cranking stops. So an electrician cranks with one hand while the other holds the red line
probe on a conductor, and has to move the probe from core to core while the crank keeps turning. Here a cable has three
cores on a terminal block, one of which has failed insulation (0.5 megohm; which one is set per trial and hidden). The robot has to crank with
one arm while the other arm rests the probe on each stud in turn, read the needle, push the flag beside the failed core
over to red, put the probe back in its holder and let go of everything. Electricians (O*NET 47-2111.00, 49-2092.00).

The reading needs both arms at once: the crank arm turns continuously while the probe arm travels across three studs,
and neither alone produces a reading (a probe left on a stud with the crank still, or cranking with the probe in its
holder, reads nothing). Guessing is a permanent flag: a flag pushed to red before all three cores were tested scores 0.

## How this differs from the nearest board tasks

- *Hold a hand coffee grinder steady with one hand and turn its crank three times with the other* (TK-HFPKM7): one arm
  steadies a part, the other turns a crank, nothing is measured. Here neither arm holds a part still: the crank arm and
  the probe arm both move, the crank only matters while the probe is on a stud, and the task is a diagnosis (read the
  needle, mark the failed core).
- *Map a two-wire harness with two flying probes* (TK-63V7SR) and *two-pole tester on a proving unit* (TK-MRUUD9):
  probes with a meter that reads from the contacts alone. No generator, so nothing needs a second hand turning.
- *Find the rigged spinner* (TK-FCQSKJ): statistics from repeated spins; no tool use or crank.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: tester, terminal block, probe, holder, crank speed, test timers, flags, `success()`, `score()`, `distance()` |
| `partkit.py` | Rigid parts from boxes with hinges (crank, knob, needle, flags) |
| `clearance.py` | Arm-to-arm distance from the URDF collision meshes (demo trace and offline check) |
| `demo.py` | The scripted full-score demo (open-loop moves; reads the dial from the top camera image) and its negative variants |

## Run a trial

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "One core of this cable has failed insulation. Test all three cores with the hand-crank insulation tester and its red probe before moving any flag (the tester only reads while its handle is being cranked), then push the flag beside the failed core over to red, put the probe back in its holder and let go of everything." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/megger-dream
```

The faulty core is a hidden trial configuration: put `MEGGER_FAULTY=0`, `1` or `2` in front of the command (default
1, the middle stud). Run all three and report the mean; the scorer checks the flag against the configured core.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200
DEMO_VARIANT=one_arm python demo.py task.py one_arm --no-record
DEMO_VARIANT=crank_only python demo.py task.py crank_only --no-record
DEMO_VARIANT=guess python demo.py task.py guess --no-record
DEMO_VARIANT=wrong_flag python demo.py task.py wrong_flag --no-record
DEMO_VARIANT=no_return python demo.py task.py no_return --no-record
DEMO_VARIANT=slow_crank python demo.py task.py slow_crank --no-record
DEMO_VARIANT=hover python demo.py task.py hover --no-record
DEMO_DEBUG=1 python demo.py task.py trace --no-record   # trace for any variant
```

The demo's moves are open loop (the task file's layout, timing and the arms' own commands only). To find the faulty
core it reads the dial from the top camera image the robot receives: a healthy core swings the red needle to the
infinity end of the scale while the crank turns, the faulty one does not. It reads no simulator state:

1. The left arm closes on the prod in its holder and the right arm on the crank knob.
2. The left arm lifts the prod out and lowers its tip onto stud 1 (1 mm of light pressure); the right arm cranks 1.5
   turns at 0.6 rev/s. The left arm lifts 2 cm, moves to stud 2, lowers; 1.5 turns. Stud 3 the same way.
3. The right arm lets go and lifts away. The left arm lowers the prod over its holder until the grip's bottom edge is just above the flared mouth and opens, so the prod drops 6 cm and seats itself; it lifts, then, fingers
   closed, pushes the tab of the flag beside the core whose reading was in the dial's red band away from the arms and lifts.

Every 10 steps `DEMO_DEBUG=1` prints (and saves in `result.json`) the crank speed, the prod's tip against each stud, the
contact timers, tested flags, needle, flag angles, each arm's lag and the minimum arm-to-arm distance.

In the official simulation stack (Isaac Sim 6.1, yam_arena) the demo reaches 1.0 (success) in 947 steps (47.4 s) with the faulty core at each of the three positions (`MEGGER_FAULTY=0`, `1`, `2`): it reads the dial to find it. The negative variants score: `one_arm` 0.0 (no core tested, the flag push is a guess), `crank_only` 0.0 (nothing tested), `guess` 0.0 (flag before testing), `wrong_flag` 0.0 (the next core's flag pushed), `no_return` 0.75 (all tested, the faulty core's flag red, prod kept in the hand), `slow_crank` 0.0 (0.25 rev/s tests nothing), `hover` 0.0 (tip 8 mm above the studs tests nothing). The smallest distance between the two arms' collision meshes (and the prod and knob each hand holds) is 88.5 mm (link4 against link4, step 237, while the left hand lowers onto stud 1); the tip stayed within 2.4 mm of each stud's axis and 0.1 mm of its top at contact, and the crank ran at 0.5-0.6 rev/s while tested (0.41 at the end of the first ramp-down).

## Scoring

A core is tested when the prod's tip rests on its stud (within 2 mm of the top and 6.5 mm of the axis, prod within 30
degrees of vertical) for 1 s while the crank runs at 0.4 rev/s or more (averaged over 0.5 s). 0 for good if a flag is
pushed to red before all three cores were tested, a flag is pushed on a healthy core, the prod is dropped or a gripper
touches a stud; also 0 if no core was tested. Otherwise 0.25 for one or two cores tested, 0.5 for all three tested but
the faulty core's flag not at red, 0.75 for only the faulty core's flag at red but the prod not back in its holder (1 cm, 10
degrees), a gripper within 3 cm of the prod, knob or flags, or not still, and 1.0 with all met and the prod, flags and
crank still for 1 s (20 steps).

## Fidelity and real-world build

The test voltage, the 1 s timer, the needle and the readings are computed in code from the crank speed and the prod's
position; the prod's lead and the earth lead are not simulated (the earth lead is already clipped to the cable armour).
The studs are plain posts with nothing to clip a lead to, so the prod-and-crank method is the one this block allows. In
a real trial: a hand-crank megger, a terminal block with three test studs, a toggle flag per core, a wooden holder.

## Real-world build and trial configurations

- Cable and block: 1 m of three-core 1.5 mm2 PVC flex, isolated from any supply (both ends on the bench), clamped
  behind the wooden terminal block; each core soldered to the underside of one brass stud, its far end sleeved and
  left open. A copper earth bar under the block is wired to the tester's earth terminal with the tester's own lead.
- The fault: a 0.5 megohm, 1 W resistor on spade connectors under the block, from one stud to the earth bar. Healthy
  cores read over 200 megohm. Any of the three cores can be the faulty one: move the resistor between trials, out of
  the robot's sight. In simulation set `MEGGER_FAULTY=0`, `1` or `2` (default 1) before the run; the scorer checks
  the flag against that core, so the dial has to be read.
- Tester: a 500 V hand-crank insulation tester (output limited below 2 mA), its line terminal on the red prod's lead.
- Safe isolation: the cable is never connected to a supply; the block and tester stand on an insulating mat; nobody
  touches the studs while the crank turns; after each trial touch the earth lead to each stud to discharge it. The
  grippers' pads are insulating.
- Trials: run the three configurations (faulty core 0, 1 and 2) and report the mean.
