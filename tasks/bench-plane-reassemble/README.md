# Put a bench plane back together after honing: iron onto both adjusters, lever cap hooked and cam locked

A Stanley Bailey No. 4 smoothing plane (12-904 / 1-12-004: 245 mm long, 50 mm cutter, 1.9 kg) stands on its sole on the bench with its iron out for honing, its toe toward the robot, so the frog's 45 degree face looks up at the robot.
Beside it, on two battens, lie the honed iron with its cap iron screwed on and the lever cap with its cam lever raised.

The robot is told: "Put the plane back together: iron and cap iron on the frog with both adjusters engaged, and the lever cap locked."

To do that it has to lay the iron assembly bevel down on the frog over the standing lever-cap screw and slide it down the frog until the depth adjuster's yoke stands in the cap iron's window and the lateral lever's boss in the iron's slot.
Both engagements are hidden under the iron: they are found by sliding until the iron drops 1.2 mm and stops.
Set down and let go, the iron stays where it was put, lying on top of the yoke; it engages only when it is slid.
Then the lever cap goes on with the wide end of its keyhole over the screw's head, is slid down under the head, and its cam lever is pressed flat onto its stop.

Why it matters.
The bench plane is in daily use today, not a museum tool: Stanley, Lie-Nielsen and Veritas all sell it new.
Finish carpenters and cabinetmakers reach for it where a power planer or sander would tear the work or cannot go: taking the last shavings off a door that binds in its frame, easing a drawer or a sash, scribing a cabinet side or a worktop to an uneven wall, and flushing a joint or a lipping before finishing.
Its iron comes out for honing and goes back several times in a working day, and every time it has to be put back like this.
O*NET lists for carpenters (47-2031.00, 889,700 US jobs in 2025) "Shape or cut materials to specified measurements, using hand tools, machines, or power saws." and for cabinetmakers and bench carpenters (51-7011.00, 81,800) "Trim, sand, or scrape surfaces or joints to prepare articles for finishing."
A plane put back with the yoke out of its window cannot be set for depth, and one with a loose lever cap chatters or drops its iron.

Nearest existing tasks, and the difference:
PR 1472 (a hacksaw blade over two pins, then a wing nut);
PR 2383 (a table saw's riving knife back in its clamp);
PR 1897 (a safety razor blade on its posts);
our own reciprocating-saw blade task (a tang pushed into a spring clamp until a pin drops in).
Each puts one blade on locators that can be seen.
Here a plate goes onto two adjusters that it hides, found by sliding it until it drops, and a second part is hooked under a screw head and locked with a cam.

## Revision: what was asked and what changed

A review before submission asked for three changes.
This version makes them; the instruction and the scoring rules are the same as before.

1. Asked: the iron engaged by itself when let go on the frog, so the step "slide it until the yoke and the boss drop in" was not needed.
   Give the iron a hold on the frog so that it stays where it is set down and engages only when slid, and add a run that sets it down and lets go.
   Changed: the iron's back on the frog's face, and the cap iron on the yoke's tab, now have friction 2.0 (every other contact keeps 0.3).
   An iron set down and let go stays where it lies: in the new run `let_go` it lay 5.8 mm up the slope from its place, on top of the yoke's tab, from the moment the hand let go to the end of the minute, and the run scores 0.25.
   The demo now shows the step itself: it sets the iron down, lets go, and then pushes the iron down the frog by its top end with two fingertips until it drops over the yoke.
   The same run stopped there (`no_cap`) scores 0.5.
2. Asked: the clamp was modelled at 22 N against an estimated 100 to 300 N; raise it as far as the simulator allows and give a source for the real figure.
   Changed: the cam now clamps with a measured 103 N (the spring plate, preloaded to 100 N, yields 1.08 mm in the demo run).
   No measurement of a lever cap's force has been published, so the real range is taken from a published cam-lever formula and test table (Elesa+Ganter, see Fidelity): about 140 to 290 N for a firm thumb press.
   103 N is below that range and is as far as PhysX goes here: at 150 N the cam closed, but the fingers slid off the lever and shoved the plane 11 mm (inside the 20 mm limit, but not a clean close), at 200 N the lever stopped 5.6 degrees short, and with the lever cap's original light pieces no preload above 22 N was carried at all (Fidelity has the numbers).
   To close the stiffer cam the demo's push on the lever now uses the joint-reading trim, which builds about 18 N at the fingertips.
3. Asked: say who does this today.
   Changed: the task card's context and "Why it matters" above name the present-day work (taking the last shavings off a binding door, easing a drawer or sash, scribing a cabinet to a wall, flushing a joint) and the makers who sell the plane new.

Two smaller changes came with these.
The lever cap is now pressed flat on the cap iron while the hand slides it under the screw's head (before, it slid under only when the hand let go).
The 1 s stillness test now follows each part's whole pose (three points 100 mm apart on each part), not only its origin.

All media are new: the demo, the six negative runs and the Dream-1 run were made with this version.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: plane, iron assembly, lever cap, battens, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes, with spring sliders and hinges, for yam_arena task files |
| `demo.py` | The scripted full-score demo and its negative variants |
| `task.json` | The task card |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the plane back together: iron and cap iron on the frog with both adjusters engaged, and the lever cap locked." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/bench-plane-dream
```

## Scripted demo

```bash
python demo.py task.py demo_out                          # full score
DEMO_VARIANT=let_go     python demo.py task.py v0 --no-record
DEMO_VARIANT=askew      python demo.py task.py v1 --no-record
DEMO_VARIANT=no_cap     python demo.py task.py v2 --no-record
DEMO_VARIANT=cam_open   python demo.py task.py v3 --no-record
DEMO_VARIANT=edge_touch python demo.py task.py v4 --no-record
DEMO_VARIANT=shove      python demo.py task.py v5 --no-record
python demo.py task.py --dry                             # no simulator: the script on the IK alone
```

The demo is open loop: every target comes from the task file's fixed layout and the clock.
The only feedback is the arm's own joint readings: a trim that keeps the tool point and the gripper's orientation, both computed from its encoders, on the plan.
It is on while the arm carries a part (the wrist gives 3 to 4 degrees under the iron), while it pushes the iron down the frog and while it pushes the cam lever.
The arm is soft, about 0.25 N per mm at the fingertips, so under the cam's load the hand would otherwise trail the plan by several centimetres; the trim is what builds the push.
No object pose, joint state or scorer state steers it.

1. The right hand takes the iron assembly across its width with its fingertips, from straight above, lifts it and tips over 45 degrees, which brings the iron to the frog's angle.
   It brings the iron in from up the slope, 19 mm off the frog's face (over the screw's head, under the front knob's bulb), lowers it along the frog's normal over the lever-cap screw, 5 mm up the slope from its place, and lets go.
   The iron stays there, lying on top of the yoke's tab.
   The hand then puts two fingertips, 34 mm apart, behind the iron's top end (either side of the lateral lever's thumb piece) and pushes the iron down the frog until it drops over the yoke and the yoke stops it.
2. It takes the lever cap across its width, lowers it with the keyhole's wide end over the screw's head, presses it flat on the cap iron, slides it down until the screw stops it in the narrow end, and lets go.
3. With closed fingers pointing down it pushes the raised cam lever over and down onto its stop, then moves away.
   The cam takes about 18 N at the fingertips, most of it straight down, so the 1.9 kg plane stays where it stands (0.3 mm from its start at the end of the run) and no steadying hand is needed; none is scored.

| Run | What it does | Score |
|---|---|---|
| demo | all three steps | 1.0 (1,050 steps) |
| `let_go` | sets the iron down on the frog, 5 mm up the slope from its place, lets go and leaves it for the rest of the minute | 0.25 (the iron stayed 5.8 mm up the slope for the whole minute, lying on the yoke's tab, 1.2 mm off the frog there) |
| `askew` | lays the iron on turned 5 degrees about the lever-cap screw and lets go, then lever cap and cam as usual | 0.0 |
| `no_cap` | stops after the iron | 0.5 |
| `cam_open` | lever cap hooked, cam lever left raised | 0.75 |
| `edge_touch` | the right hand first closes on the iron's cutting edge, then does everything else | 0.0 |
| `shove` | after locking the cap, the left hand pushes the plane's knob 9 cm sideways | 0.75 (the plane turned 45 degrees and moved 19 mm) |

Why `askew` fails: turned 5 degrees, the 50.8 mm iron no longer fits between the cheeks, 53.3 mm apart; its corners land on them and it never lies on the frog (it ends propped across them, its lower end 16 mm off the frog's face).
The lever cap then has nothing to lie on, and nothing is engaged, hooked or locked.

Why `let_go` fails: the iron stays where the hand leaves it, with the cap iron lying on top of the yoke's tab, 1.2 mm off the frog there, so neither adjuster is engaged.
`no_cap` is the same run with the push down the frog, and scores 0.5: the slide is what engages the two adjusters.

Two wrong assemblies cannot be produced on this plane and are covered by scorer tests instead of demo runs.
An iron turned end for end cannot lie over the screw (its slot does not reach), and an iron laid bevel up would need the gripper to come from under the frog.

## Scoring

End state, read from the video, a ruler and a 0.6 mm feeler gauge.

| Score | Condition |
|---|---|
| 0.25 | The iron assembly lies on the frog over the lever-cap screw, bevel down and cutting edge toward the mouth (within 10 degrees), its back within 3 mm of the frog's face at the screw, both grippers at least 3 cm from it. |
| 0.5 | Also the yoke's tab stands inside the cap iron's window and the lateral lever's boss inside the iron's slot, the iron's back within 0.6 mm of the frog's face at both. The iron then cannot slide more than 1 mm along the frog. |
| 0.75 | Also the lever cap lies on the cap iron the right way round with the screw within 2 mm of the narrow end of its keyhole, its plate under the screw's head, both grippers at least 3 cm from it. |
| 1.0 | Also the cam lever is within 3 degrees of its closed stop, the plane has moved no more than 20 mm and tipped or turned no more than 10 degrees, both grippers are at least 3 cm from the plane, and plane, iron and lever cap have not moved more than 1 mm for 1 s. |

- 0 while the lever cap lies over an iron that is bevel up or has its cutting edge up the slope.
- 0 for good if the plane was tipped more than 45 degrees off its sole (knocked over); turning it flat on the bench is not tipping.
- 0 for good if a gripper came within 15 mm of the iron's cutting edge (blade contact).

`success()` and `score()` use the same test, and 1.0 is returned only when the 1 s window holds (0.75 until then).
In a real trial the 0.5 level is checked with the feeler gauge under the iron at the yoke and by turning the depth wheel a quarter turn each way: the iron must follow it.

## Runtime and seeds

About 1 minute per trial for the full 1,200 steps, plus about 1 minute for Isaac Sim to start.
The demo reaches full score in 1,050 steps (88% of the run limit).
The layout is fixed (`object_jitter=0`); there are no random seeds.

## Dream-1 run

One run of this version at the run limit (1,200 steps): score 0.0.
Dream-1 drew both arms in toward their bases, brought its right hand down at the lever cap on its batten and stayed there for the rest of the minute; it lifted neither part, and the plane was not touched.

## Fidelity

Everything is rigid shapes at full size, and nothing is scripted or detected.

- Plane body: one loose 1.44 kg body (sole, cheeks, frog, knob, tote, depth wheel) with the frog's face, the lever-cap screw (shank, head, chamfer) and the yoke's tab fixed on it.
  The depth wheel does not turn; the yoke stands where the wheel was left.
- Lateral lever: a free hinge on the frog (8 degrees each way, light friction); its boss finds the iron's slot by the slot's edges pushing it.
- Iron assembly: one loose 0.27 kg part, the iron with its slot, the cap iron with its hump, slot and yoke window, and the cap iron screw's head on the back, which runs in the frog's channel.
  The 25 degree bevel is a step.
- Lever cap: a loose 0.19 kg part with a keyhole, a cam lever on a hinge and a spring plate under the cam.
  Of the 0.19 kg, the cam lever has 40 g and the spring plate's two pieces 30 g each, and the part is solved with 128 solver passes a step instead of the usual 32: PhysX needs both to carry the clamp's force through these small pieces (see Clamp force).
  The plate is two pieces joined by a spring: the cam's eccentric pushes the upper one down, the lower one presses the cap iron.
  The cap pivots on the screw's head, so the push at its top end presses its toe onto the cap iron.
  The eccentric passes its high point 15 degrees before the stop, so the lever stays down.
  The cam's pin turns freely: the raised lever leans back on its stop, and on the frog its eccentric rests on the spring plate, whose own 2 N return spring holds the lever up until it is pushed.
- Clamp force: with the cam shut the spring plate presses the cap iron with 103 N.
  That is a measured figure, not a setting: the plate is a 2,500 N/m spring preloaded to 100 N, and in the demo run it yields 1.08 mm, which it can only do if the contact carries the preload.
  The lever takes about 18 N at the fingertips, 30 mm from its pivot, to close (from the arm's joint readings and its joint stiffness).
  No measurement of a bench plane's lever-cap force has been published: makers' instructions, patents, woodworking research and forum threads give only setting rules ("just enough pressure so that I have to press firmly to close the lever cap"; "about the same amount of pressure you apply to open a typical car door").
  The nearest published figures are for industrial clamping levers with an eccentric cam.
  Elesa+Ganter's data sheet "Clamping and manual forces in clamping levers with eccentrical cam" gives, from test series, 1,250 to 1,750 N of clamping force for 75 N on a 33 mm lever, and the formula FS = FH x lH / (lU x (µw + µ1) + lA x µ2) with µw = 4 h / (2 π lU).
  With this cam's sizes (lH 30 mm, lU 8.8 mm, h 3.2 mm of stroke, lA 2 mm, steel on steel 0.2 at both places; estimates, to be measured on the bought plane) the formula gives about 7 N of clamp for each newton on the lever, so a firm thumb press of 20 to 40 N gives about 140 to 290 N.
  The modelled 103 N is below that range, and it is as far as this simulation goes.
  What limits it is PhysX, in two ways.
  With the lever cap's pieces at 8 to 20 g and 32 solver passes, raising the preload did nothing: at 60, 100, 200 and 250 N the cam closed with the same 5 N at the fingertips as at 22 N, and the plate's foot sank 0.7 to 0.8 mm into the cap iron instead of the spring yielding.
  With the heavier pieces and 128 passes the force is carried, and the full demo was run at three settings: 100 N closes cleanly (the plane ends 0.3 mm from its start); at 150 N the cam closes and the plate carries 152 N, but the fingers slide off the lever as it snaps down and shove the plane 11 mm; at 200 N the lever stops 5.6 degrees short of its stop.
  The lever-cap screw is the user's adjustment for exactly this force, and the real trial sets it by the lever's closing force (see below).
- Friction: 0.3 for steel on steel and steel on cast iron, with one exception: the iron's back on the frog's face, and the cap iron on the yoke's tab, have 2.0.
  That is more than a real frog gives (handbook value for dry steel on cast iron: about 0.4), and it is deliberate.
  With 0.3 the simulated iron, let go anywhere above its place, ran down the 45 degree frog and dropped over the yoke by itself, so letting go was as good as sliding.
  With 2.0 it stays where it is put (in the `let_go` run it did not move from 5.8 mm up the slope for the rest of the minute), and the adjusters are engaged only by sliding it.
  The value is what PhysX needs, not a material's: at 1.3, still above the 1.0 a 45 degree slope calls for, the let-go iron crept 0.06 mm/s and ran off down the frog after 40 s.
  A real iron on a clean dry frog slides down when let go and may or may not drop over the yoke; a real trial is scored from the end state all the same.
- Rounds (screw, boss, cam) are prisms of 8 to 24 sides with their corners on the true circle.

## Real-world build

Buy a Stanley Bailey No. 4 (12-904 or 1-12-004) and use its own iron, cap iron and lever cap.
Two 38 x 38 mm planed softwood battens, 130 and 106 mm long, screwed or clamped to the bench, hold the two loose parts clear of the bench so the fingertips can reach under their edges.

Sizes taken from the maker and from published measurements:

| Part | Size | Source |
|---|---|---|
| Plane | 245 mm long, 50 mm cutter, 1.9 kg | Stanley data sheet 1-12-004 |
| Iron | 2 in (50.8 mm) wide, 7-1/2 in (190.5 mm) long, 3/32 in (2.4 mm) thick, ground at 25 degrees | Stanley 12-313 replacement iron listing |
| Iron's slot | 0.450 in (11.43 mm) wide | Lee Valley's figure for Stanley planes |
| Lateral lever's disc | 7/16 in (11.11 mm) | measured disc quoted in a restoration thread |
| Yoke | stands 9/64 in (3.6 mm) above the frog's face | iron maker's fitting note for Bailey planes |
| Cap iron's yoke window | front edge 3-11/16 in (93.7 mm) or more behind the cap iron's front | same fitting note |
| Lever cap | 2 in wide, 4-5/16 in (110 mm) long | measured No. 4 cap |

Estimates in the model: measure the bought part and correct `task.py` before a real trial.

| Estimate | Value in the model | Measure the bought part |
|---|---|---|
| Sole width and thickness, cheek height and thickness | 60.3 mm, 4 mm, 46 mm at the peak, 3.5 mm (53.3 mm between the cheeks) | measure the bought part |
| Mouth, knob and tote positions | mouth 67 mm from the toe, knob 37 mm in front of it, tote 120 to 176 mm behind it | measure the bought part |
| Frog's face | 48 mm wide, reaching 131 mm up the slope from the mouth, channel 20 mm wide and 4.5 mm deep up to 55 mm | measure the bought part |
| Lever-cap screw | 9/32 in (7.1 mm) shank, 10.5 mm head 2.2 mm thick with a 1.5 mm chamfer, 60 mm up the slope | measure the bought part |
| Yoke's tab | 8 mm (5/16 in) wide narrowing to 5 mm at the tip, 4.0 mm along the slope, 99.3 mm up the slope | measure the bought part |
| Lateral lever | 9 mm strip, pivot 127 mm up the slope, boss 17 mm below it standing 2.3 mm with a rounded top, 8 degrees of swing each way | measure the bought part |
| Iron's slot length | 30 to 128 mm from the cutting edge | measure the bought part |
| Cap iron | 114.3 mm long, 2.0 mm thick, hump 3.6 mm high, slot 12.7 x 38 mm, window 12.7 x 4.8 mm, screw head 16.5 mm | measure the bought part |
| Lever cap | 50 mm wide, keyhole 16 mm wide end and 8 mm narrow end 12 mm apart, cam pivot 94 mm from the toe, cam lever 38 mm, eccentric 3.2 mm of travel | measure the bought part |
| Masses of the loose parts | iron and cap iron 0.27 kg (from the steel's volume), lever cap 0.19 kg | weigh the bought parts |

Forces and how to set and check them:

- Lever cap: with the cap hooked and the cam lever raised, turn the lever-cap screw until the cap just slides under its head (about 0.8 mm of play).
  Then pull the cam lever shut with a spring scale hooked 30 mm from its pivot: adjust the screw a little at a time until it closes at 15 to 20 N.
  By the cam's leverage that is about 100 N on the cap iron, the force the model carries.
- Lateral lever: it must swing under a fingertip (about 0.2 N at the boss); a drop of oil on its rivet if it does not.
- Frog and iron: wipe the frog's face and the iron's back clean and dry, with no oil or wax, so that the iron holds on the frog as well as a real one can.
  A real iron will still slide down the frog when let go, and may or may not drop over the yoke by itself: that is the one place where the real parts behave differently from the model, and a real trial is scored from the end state all the same.

Start of each trial (also in `task.json`): plane on its outline, toe toward the robot; depth wheel turned until the yoke holds a seated iron's edge 2 mm inside the mouth, then left alone; lateral lever centred; iron assembly on its batten, bevel down, cutting edge toward the robot; lever cap on its batten, toe toward the robot, cam lever raised 110 degrees from shut.
The cutting edge is sharp: the person who resets the trial handles the iron by its sides, and a strip of tape over the edge is allowed in a real trial since nothing touches it in the task.

## Offline checks

- `tests/test_bench_plane_reassemble.py` in the team repository: 36 tests of the scorer's levels, failure states, the let-go iron, the friction materials, fits and the cam's geometry, without the simulator.
- A MuJoCo rehearsal built from this task file's own shapes plays the demo's script with a two-finger hand on a model of the arm's joint stiffness (it reproduces the sag measured on Isaac under the iron to 0.5 mm): 1.0 at 0.7, 1 and 1.5 times that stiffness; `let_go` 0.25, `no_cap` 0.5, `cam_open` 0.75, `edge_touch` 0.
