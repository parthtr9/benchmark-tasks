# Wedge a wall form's two snap ties tight

A section of a hand-set concrete wall form stands on the table with its outside toward the arms: 3/4 in plywood
sheathing on a 2x3 sole plate, a double 2x4 waler nailed to it, and the ends of two 1/4 in snap ties coming out
through the gap between the two waler timbers, 16 in apart. Each tie end has a 1/2 in button head and is slack by
3 mm. Two steel snap-tie wedges stand in wooden stands in front of the arms. The robot must fasten both ties: slip
each wedge's keyhole over a tie head, bring the wedge down so its slot catches the tie behind the head, and drive it
until its ramp has drawn the tie out, the slack is gone and the wedge bears on the waler.

Setting forms is a core part of rough carpentry and concrete work. Carpenters (O*NET 47-2031.00), construction
laborers (47-2061.00) and cement masons and concrete finishers (47-2051.00), together more than 2 million US
workers, build, set and strip forms for footings and walls, and a hand-set wall form has a snap tie with a wedge on
each face every 16 to 24 inches in both directions: hundreds of wedges on a small wall, every one put on and driven by
hand. A wedge left hanging slack lets the form move when the concrete is poured.

What the task tests: a taper fastener on a headed tie. The wedge has to be presented upright with its keyhole level
with the head and moved onto the waler so the head comes through; it then has to come down so that the narrow slot
runs onto the tie's rod behind the head (the slot is 3.6 mm wider than the rod), and be driven along its own length,
against the tie, until the slack is taken up, and it must stay put when let go. It is not the same as hanging a part
on keyhole studs (nothing is tightened there, the part hangs by its weight) or pushing a pair of shims into a gap
(no headed tie to catch): here the score is the tie's draw.

Everything the wedges touch is true size: plywood, waler timbers, the 5/8 in tie gap, the ties and their
heads, the tie spacing and the wedges. Only the form's height (250 mm) and length (600 mm) are cut.

Forces, stated plainly. An empty form's ties carry no load until the concrete is poured: fastening a tie means
taking up its slack until the wedge is snug on the waler. This task is that hand-setting, with a stated resistance:
the slack resists with 2 N per mm, 6 N when it is gone. A wedge hung on the tie and let go draws it about 1.3 mm by
its own weight (2.5 N), which is not tight; seating it takes a push of about 4 to 6 N more along the wedge (the tie load
times the two sliding frictions plus the 1 in 5.3 ramp, less the wedge's weight), which a small arm delivers. On site
a carpenter usually follows with a light hammer tap to lock the wedge against vibration during the pour; that tap,
and the load the tie carries under wet concrete (up to its 10 kN rating), are not part of the task and are not
claimed. The wedge here holds by its own friction: its ramp is self-locking.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes and cylinders, with spring-loaded sliders, for yam_arena task files |
| `demo.py` | The scripted full-score demo and its negative variants |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs
with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough, from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fasten both snap ties of the wall form tight against its double waler with the two steel wedges." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/form-tie-wedge-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode. There are no
seeds: `object_jitter=0`, one fixed layout.

## Scripted demo

```bash
python demo.py task.py demo_out --max-steps 1200       # full score: writes demo_*_cam.mp4, setup.png, result.json
DEMO_VARIANT=hang_only python demo.py task.py hang     # wedges hung on the ties and let go, not driven
DEMO_VARIANT=one_wedge python demo.py task.py one      # only the left tie is fastened
DEMO_VARIANT=held python demo.py task.py held          # both driven tight but never let go
```

Each arm works the tie in front of it, both at once:

1. It reaches forward with its fingers pointing at the form, 20 degrees below level, and takes its wedge by the two
   flanges near the thick end.
2. It lifts the wedge out of its stand and carries it to 10 mm in front of the tie's head, with the keyhole's wide
   part level with the head. Its two fingers close independently, so the wedge can sit a few millimetres to one side
   of the tool point; the arm reads that from its own two finger joints and shifts sideways to put the slot on the
   tie's line.
3. It moves the wedge onto the waler: the head comes through the keyhole and the nose meets the waler.
4. It lowers the wedge, only ever carrying it (it never pushes down), until the slot has run onto the rod and the
   tie has taken the wedge's weight, and lets go: the wedge hangs there, slack.
5. It draws back, closes its fingers and lays them level across the wedge's thick end.
6. It presses the wedge down, a little harder each step, until its own joints show that the wedge has stopped
   although the push kept growing (the tie is tight), or the push has reached about 6 N.
7. It stops pushing and lifts off.

The demo is open loop on the scene: every position comes from the task file's fixed layout. The arms are compliant
(they give about a centimetre per newton), so each arm corrects its own sag and tilt from its own joint readings and
feels the end of the drive the same way. It reads nothing from the simulated objects (its debug log does).

Results in simulation (Isaac Sim 6.1, `object_jitter=0`):

| Run | Score | Steps | Why |
|---|---|---|---|
| Demo | 1.0 (success) | 684 (34 s) | Both ties drawn to their stops, fingers lifted off, everything still for 1 s |
| `hang_only` | 0.25 | 598 | Both wedges hang on their ties; their own weight draws each tie only part of the way (1.3 to 1.5 mm in test runs), short of the 2.5 mm that counts as tight |
| `one_wedge` | 0.5 | 808 | The left tie is tight; the right wedge is still in its stand |
| `held` | 0.75 | 742 | Both ties are tight, but the fingers are still pressed on the wedges |

Largest IK error along the commanded path: 0.1 mm. The two arms work 40 cm apart and never come near each other.

## Scoring

Measured at the end of the trial with a caliper and a ruler.

- 0 if the form section was moved more than 5 mm at any time.
- A wedge **hangs on a tie** when it is thick end up with its web toward the arms, the tie's rod in its slot with
  the head 38 to 142 mm from the wedge's thin end, and the head outside the web.
- A tie is **tight** when a wedge hangs on it, the tie is drawn out at least 2.5 of its 3 mm of slack (head underside
  at least 12.2 mm outside the waler face) and the wedge's bearing edges are within 2 mm of the waler face at the tie.

| Score | State at the end |
|---|---|
| 1.0 | Both ties are tight, both grippers are at least 30 mm from both wedges, and neither wedge nor tie has moved more than 1 mm for 1 s |
| 0.75 | Both ties are tight, but a gripper is within 30 mm of a wedge or something still moves |
| 0.5 | One tie is tight |
| 0.25 | A wedge hangs on a tie |
| 0 | Anything else |

Either wedge may go on either tie, and either arm may do either tie. Pulling a tie out by hand scores nothing: a tie
only counts with a wedge on it.

A trial takes at most 1,200 steps (1 minute at 20 Hz).

## Fidelity

Rigid bodies built from boxes and cylinders. Each tie end is a rod on a slider: 3 mm of travel against a 2 N/mm
spring to a hard stop stands in for the slack between the form and the tie's spreader cone; the opposite form face
and the concrete are not modelled. The keyhole is a square with 45 degree shoulders (a real one is round) and the
wedge's tapered flanges are stepped. The wedge is hand-set, not struck with a hammer, and is not nailed afterwards.
Tie tension is not capped, so an over-driven tie cannot break. Friction coefficients: form 0.5, wedge 0.3.

## Real-world build

Parts (a lumber yard and a concrete-accessory supplier; about 30 USD):

- Form: a 600 x 250 mm piece of 3/4 in (19 mm) form plywood; nailed to one face, a 600 mm length of 2x3
  (38 x 64 mm) lying flat at the foot as a sole plate, and two 600 mm lengths of 2x4 (38 x 89 mm) on edge with their
  lower faces 49 and 103 mm above the foot, leaving a 16 mm (5/8 in) gap centred 95 mm up. The waler stands 25 mm
  proud of the sole plate. Drill two 5/8 in holes through the plywood in the gap, 406 mm (16 in) apart. Weight about
  4.4 kg.
- Kickers: 38 x 38 mm timber screwed to the bench: a 600 mm strip against the back of the plywood's foot and a 25 mm
  block in front of each end of the sole plate.
- Ties: two standard 1/4 in snap ties with 1/2 in button heads (any wall size with a 4-3/4 in lumber-and-wedge
  end), cut off 60 mm behind the plywood. On each tie behind the plywood: a washer against the plywood, a compression
  spring of 2 N/mm, and a shaft collar. Set the collar so that the head's underside rests 9.7 mm outside the waler
  face with the spring just touching, and fit a second collar on the tie inside the waler gap, 3 mm in front of the
  plywood, so the tie stops after 3 mm of draw, where the head's underside is 12.7 mm outside the waler. Check with a
  spring scale hooked under the head: 6 N (within 1 N) at the stop.
- Wedges: two standard pressed-steel snap-tie wedges, about 156 x 60 mm and 0.25 kg (weigh yours; any wedge sold for
  1/4 in snap ties fits). Check that a wedge hung on a slack tie and let go leaves the head less than 12 mm outside
  the waler, and that pushed home it reaches 12.7 mm.
- Stands: for each wedge, two pine posts 9 x 19 x 65 mm, 72 mm apart, joined by a 10 x 10 mm rail 55 to 65 mm above
  the bench, and a 10 x 12 mm stop strip 90 mm long on the bench 30 mm behind the rail; screwed to the bench so a
  wedge standing between the posts, leaning on the rail, has its bearing edges 200 mm in front of the arm-base line
  (100 mm short of the waler face), in line with its tie.

Layout: as in `task.json` (waler face 300 mm in front of the line through the arm bases, ties at 203 mm either side of
the centre line and 95 mm above the bench).

Reset: knock both wedges up off the ties, stand each in its stand thick end up with its bearing edges toward the form,
and check each head's underside is 9.7 mm outside the waler face.

Scoring a real trial: after the arms stop, measure from the waler face to each head's underside with a caliper (tight
is 12.2 mm or more), look that each wedge's slot is on the tie with both flange edges on the waler (a 2 mm feeler
must not enter), and measure the grippers' distance to the wedges with a ruler. Nothing here is hazardous: the form is
empty and the tie load is 6 N; keep fingers clear of the wedge while the arm drives it.
