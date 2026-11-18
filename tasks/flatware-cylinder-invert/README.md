# Invert washed teaspoons into a clean flatware cylinder

Two Vollrath 52642 perforated nylon flatware cylinders stand on the counter mouth to mouth.
The lower one is upright and holds 14 washed teaspoons standing handles-down, bowls up, as they came out of the dish machine.
The upper one is empty and sits upside down on it, the two collars together, nothing fastening them.
The robot is told only the end state: every spoon handles-up in the clean cylinder standing on the counter, the other cylinder set down beside it, no spoon touched or spilled.

Dishwashers (O*NET 35-9021) and food preparation workers (35-2021) do this after every rack of flatware.
The washed flatware goes through the machine eating-ends up so it is cleaned, but it is served handles-up so nobody touches the eating ends.
A Navy dishwasher manual gives the step: "place an empty cylinder over clean, air-dried silverware; invert".
Vollrath's and U.S. Chemical's warewashing guides say the same (handles up, in a clean cylinder).
The trial starts after that first step: the empty cylinder is already on the full one.
What is tested is the turn itself: two loose containers stay together only through the grip on their collars, the loose spoons slide from one into the other as they turn, and the old cylinder then has to come off over the spoons' handles without pulling any out.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes, for yam_arena task files |
| `demo.py` | The scripted full-score demo (open loop on the layout, trimmed only by the arms' own joint readings) |
| `clearance.py` | Arm-to-arm clearance from the YAM collision meshes, for the demo's trace |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

In the Isaac Lab-Arena environment from the simulation walkthrough (`source /data/IsaacLab-Arena/yam.env`), from this folder:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Leave every spoon handles-up in the clean top cylinder standing on the counter, with the other cylinder set down beside it and no spoon touched or spilled." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/flatware-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                                    # full score
DEMO_VARIANT=one_arm    python demo.py task.py out --no-record     # one hand pinches the seam and turns: spills, 0
DEMO_VARIANT=no_pinch   python demo.py task.py out --no-record     # hands on the old cylinder's body only: 0
DEMO_VARIANT=early_lift python demo.py task.py out --no-record     # old cylinder lifted off before setting down: 0
FLATWARE_LAYOUT=pour    python demo.py task.py out --no-record     # rehearsal only, not the task: one hand pours (below)
python demo.py task.py --ik                                        # no simulator: IK of every target and the turn
```

1. Each hand points at the stack from its own side and pinches the two collars together across the seam, fingers closing up and down.
2. Both lift the pair 75 mm, turn it over about the line between the hands (each wrist rolls half a turn) and set it down: the spoons slide through the seam into the clean cylinder, handles up.
3. Both let go and back off, then squeeze the old cylinder's body between their closed fingers, lift it clear of the handles, carry it aside and set it down.

Why one arm is not enough: the 95 mm body fills the gripper's opening, so a hand can hold a cylinder only by pinching its collar on one side.
Holding the loose pair by one side of the seam, the pair parts in the turn (`one_arm`).
The other one-arm route, standing the empty cylinder up beside the full one and pouring, fails at its first lift: in a rehearsal that starts with the empty cylinder already upright (`FLATWARE_LAYOUT=pour`), the full cylinder, 0.63 kg hanging 56 mm from a one-sided collar pinch, fell out of the hand as it was lifted.

The demo reaches 1.0 (success) in 629 steps (31 s) of the 1,800-step limit, with the arms at least 75 mm apart, and did so in four Isaac Sim runs in a row.
Each negative variant scores 0 (Isaac Sim batch: one_arm 0, no_pinch 0, early_lift 0 for good: a finger came within 5 mm of a spoon), and so does the one-arm pour rehearsal.
Dream-1, given only the instruction, scored 0 in one run of 1,800 steps.

## What changed since the first Dream run

- The scorer now gives 0.25 and 0.5 only once both grippers are at least 3 cm from the new cylinder, and 0.75 only once they are clear of both cylinders, so a cylinder held in place earns nothing (before, those levels also counted while held). The instruction is unchanged.
- Rerun on the changed scorer: the Isaac Sim batch gave the same scores (demo 1.0 in 629 steps; one_arm 0, no_pinch 0, early_lift 0), and the one new Dream-1 run scored 0 in 1,800 steps, as the first did.

## Scoring

End state, each level including the ones before it.

- 0.25: every spoon inside the new cylinder (its middle within the body), that cylinder upright (axis within 18 degrees).
- 0.5: and every spoon handle up (within 45 degrees).
- 0.75: and the new cylinder standing on the table, with the other one set down on the table off it (axes more than 122 mm apart).
- 1.0: and both grippers at least 3 cm from both cylinders, nothing moving more than 1 mm for 1 s.
- 0.25 and 0.5 count only once no gripper finger is within 3 cm of the new cylinder, and 0.75 only once none is within 3 cm of either cylinder: a cylinder held in place earns nothing.
- 0 for good if a gripper finger point came within 5 mm of a spoon's surface at any time, or a spoon left both cylinders.

How the old cylinder is set down (upright or upside down) is not scored: the instruction does not say.

## Fidelity

Rigid parts.
Sources: Vollrath lists the 52642 cylinder at 3-3/4 in across, 5-5/8 in high and 0.34 lb, and 4.4 in wide, which gives the 112 mm collar by which it hangs in its holder.
Collar and wall thickness (4 mm, 2 mm) are not listed: estimates from photographs (measure the bought part).
The teaspoon is the Winco 0005-01 "Dots" heavyweight 18/0 pattern, listed 6-1/4 in long, 1-5/16 in wide and 0.9 lb a dozen (159 mm, 33 mm, 34 g); its bowl length and depth and its handle's section are estimates.
Each cylinder has 16-sided walls and a polygonal bottom with their corners on the real radius; the drain holes are drawn but carry no load.
Each spoon is a flat handle and a flat bowl.
Flat bowls cannot nest the way real ones do, so 14 stand in the cylinder in rows, 1 mm apart, where a real cylinder takes 15 to 20 pieces.

## Real-world build

### Parts

| Part | Source | Size and mass | Measured or estimated |
|---|---|---|---|
| 2 x Vollrath 52642 perforated nylon flatware cylinder | [Vollrath product page](https://www.vollrathfoodservice.com/products/smallwares/warewashing-handling/flatware-washing-systems/vertical-flatware-washing-system-cylinders-and/52642); [WebstaurantStore listing](https://www.webstaurantstore.com/vollrath-52642-white-perforated-high-temperature-nylon-flatware-cylinder/92252642WH.html) | 3-3/4 in (95 mm) across, 5-5/8 in (143 mm) high, 0.34 lb (0.154 kg); 4.4 in (112 mm) wide at its collar | Listed. Collar 4 mm and wall 2 mm thick: estimates from photographs, to be measured on the bought cylinders |
| 14 x Winco 0005-01 "Dots" heavyweight 18/0 teaspoon | [McDonald Paper listing](https://mcdonaldpaper.com/winco-0005-01-dots-heavy-weight-mirror-finish-stainless-steel-teaspoon-12-pack) | 6-1/4 in (159 mm) long, 1-5/16 in (33 mm) wide, 0.9 lb a dozen (34 g each) | Listed. Bowl length 48 mm and depth 8 mm, handle 10 x 2.5 mm: estimates, to be measured |

No custom parts and no fixtures: the cylinders and spoons stand loose on the bench.

### Layout

Tape the stack's outline (a 112 mm circle) at 350 mm in front of the midpoint between the arm bases, on the centre line.
Stand the 14 spoons handles-down, bowls up, in one cylinder in rows across x 9 mm apart: two side by side in the five middle rows, one in each of the four outer rows (a printed card with the 14 bowl outlines, laid in the cylinder's bottom, makes this repeatable).
Stand that cylinder on the outline and put the second, empty cylinder upside down on it, collar on collar.
Use dry cylinders and spoons.

### Forces

No force is set or adjusted: the task depends only on the parts' listed masses (0.154 kg per cylinder, 0.48 kg of spoons) and on nylon and steel friction.
Check once a session that the pair's collars are dry: a dry stack slides apart under a light sideways push, and nothing holds the two cylinders together but the hands.

### Real-trial checks for each scoring level

- 0 for good, a spoon touched: before the trial dust both grippers' fingertips with UV-fluorescent powder; afterwards inspect every spoon under a UV lamp, and any powder on a spoon counts as a touch. (The simulator's limit is stricter: a finger point within 5 mm of a spoon.)
- 0 for good, a spoon spilled: any spoon outside both cylinders at any time in the video.
- 0.25: every spoon inside the new cylinder and that cylinder upright (photo from above and from the side).
- 0.5: every spoon handle up, read from the photo (bowl below the handle, within 45 degrees of vertical).
- 0.75: the new cylinder standing on the table and the other set down on the table off it; measure the distance between the two cylinders' centres with a rule (more than 122 mm).
- 1.0: both grippers at least 3 cm from both cylinders and nothing moving for 1 s, from the video.

## Who does this, and how often

Dishwashers (O*NET 35-9021) list "Wash dishes, glassware, flatware, pots, or pans, using dishwashers or by hand" and "Place clean dishes, utensils, or cooking equipment in storage areas" among their tasks; food preparation workers (35-2021) take on the same warewashing in smaller kitchens.
Washed flatware runs through the dish machine eating-ends up in perforated cylinders and is turned handles-up into a clean cylinder for service after every load; the Navy dishwasher manual gives the step: "place an empty cylinder over clean, air-dried silverware; invert", and Vollrath's and U.S. Chemical's warewashing guides give the same handles-up rule.
Not verified here: a head count of the workers who do this particular step; the claim rests on the occupations' task statements and the procedure in the warewashing guides.

## Runtime and seeds

One fixed layout (`object_jitter=0`), no randomness.
Run limit 1,800 steps (1.5 minutes at 20 Hz).
