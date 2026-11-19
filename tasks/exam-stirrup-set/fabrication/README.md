# Bench fixture: the stirrup mechanism for a lab without an exam table (about $175)

This is the specified real hardware for the task.
It reproduces the foot end of the exam table's stirrup shelf with both stirrup assemblies, and keeps every mechanism the task tests:
- the 400 mm draw to a collar;
- the unfold to an open stop, through a window that blocks it until the stirrup is drawn clear;
- the lift out of the lateral lock against a spring;
- the swing between notches;
- the lock in the outermost notch.

It is built to the simulated model's own dimensions.
`make_fixture.py` reads every size, angle, stop, spring constant and mass from `../task.py`, so the fixture's critical dimensions are exact by construction and are the ones the simulation uses.
A Midmark Ritter 204 or 604 table is the optional real object (see the task README).

## Files

| File | What |
|---|---|
| `make_fixture.py` | Generates everything below from the task file: `uv run --no-project --with numpy --with manifold3d python fabrication/make_fixture.py` (from the task folder) |
| `stl/` | 17 printable parts (binary STL, millimetres; all fit a 256 mm bed) |
| `drawings/1_plan.svg` | Plan view: shelf box, windows, boss axes, ceiling strips, pillars, support rails |
| `drawings/2_section.svg` | Section along a bar at the boss, with every height |
| `drawings/3_notch_plate.svg` | The lateral lock's notch plate: radius, notch angles and widths, groove |
| `drawings/4_bar.svg` | Bar drilling: roll pin, spacer screws, draw collar, post |
| `drawings/5_stirrup.svg` | Stirrup: clevis, roll pin, open-stop screw, arm, loop |
| `BOM.csv` | Bill of materials with prices and the total |
| `cut_list.csv` | Plywood, lumber and stock cuts |
| `parts.csv` | Printed parts: sizes, volumes, filament |
| `fixture_data.json` | All of it as data (dimensions, pieces, bill of materials, forces, checks) |
| `checks.txt` | The geometry checks the script ran |

## Cost

$175.36 in total (`BOM.csv`): $97.36 at prices listed on 2026-11-18 (steel flat bar, three plywood panels), the rest estimates for small stock and hardware.
It assumes access to a 3D printer (about 0.9 kg of PETG) and basic tools: a drill, an M6 tap, a hacksaw, a vise.

## How it is built

- **Shelf (670 x 580 x 220 mm):** a 3/4 in plywood base board with 1/4 in plywood sides, top and front face.
  - The front face has two windows, 90 mm wide, from 121.7 to 163.2 mm above the bench, centred 245 mm either side.
  - The base board is clamped to the bench with its front face 650 mm in front of the arm bases' line.
  - Its rear 320 mm overhangs the bench's far edge onto a stand: a 2 x 4 in rail on edge on two 2 x 4 in legs.
- **Bars:** two 565 mm lengths of 3/4 x 1/4 in steel flat, 140 mm above the bench and 490 mm apart.
  - A 1/2 x 1/8 in aluminium T-rail hangs 4 mm under each bar on printed spacers (M3 countersunk screws through bar, spacer and rail).
  - The one wide spacer, 497 to 503 mm from the bar's front end, is the draw collar.
  - The detent post is an M6 set screw in a tapped hole 555 mm from the front end, standing 30 mm above the bar and flush below.
  - Each bar assembly weighs 0.60 kg.
- **Pivot boss:**
  - a printed pedestal on the base board;
  - a yoke that swings on an M8 axle bolt; a slot in it and a pin in the pedestal give the stops at 15 degrees inward and 27 degrees outward;
  - a trunnion that turns on two 5 mm dowel pins in the yoke's cheeks.
  The trunnion's lips run in the 4 mm gap between the bar and its T-rail, so it holds the bar both ways while all of the boss stays below the bar's underside.
  A folded stirrup lying on the bar therefore slides over it.
- **Lateral-lock spring:** a compression spring (2.1 N/mm) sits in the yoke's tower, 75 mm behind the boss axis, and pushes a printed roller up under the T-rail.
  An M8 screw under it sets the preload (11.2 N with the bar level).
  This is the model's lift spring exactly: 12 N m/rad at the lift is 2,130 N/m at 75 mm.
- **Lateral lock:** each side has a 1/4 in plywood ceiling strip on four printed pillars, its underside 173.5 mm above the bench.
  Under it are the printed notch plate and two basswood rails.
  The notch plate has teeth round the post's path at 75 mm radius, with notches at -12, 0, 12 and 24 degrees, 8 mm wide and 5 mm deep.
  The rails form the 8 mm groove along the draw.
- **Support rail:** a 24 mm plywood strip on two printed risers under each stowed bar's tail half, its top at the T-rail's underside.
- **Stirrups (0.30 kg each):**
  - the arm is a 128 mm offcut of the bar steel, with printed edge strips bringing it to 25 mm;
  - the loop is a 1/4 in steel rod bent on the printed former to the 70 x 100 mm teardrop, inside a printed 12 mm shell;
  - a printed clevis fits round the bar's front end on an M5 shoulder bolt (the roll pin);
  - an M3 open-stop screw across the clevis lands on the bar's top face at 100 degrees;
  - a 60 mm steel slug in the junction block, plus washers, brings each to its mass.

## Forces: how each is set and checked

| Force | Value | Model | How it is set | How it is checked |
|---|---|---|---|---|
| Lateral-lock spring | 11.2 N preload, 2.13 N/mm at the post | lift spring 12 N m/rad, 4 deg preload | M8 screw under the spring | bar drawn and straight, stirrup off: spring scale under the post, pulled straight down: starts out of its notch at about 26 N (15 N of it the bar's front-heavy weight), clear (5 mm down) at about 37 N |
| Lift to unlock | about 7 N to start, 9 N at release | follows from the spring and masses | by the spring | stirrup open, bar locked straight: spring scale on the arm 115 mm above the roll pin, pulled straight up |
| Draw | about 5 to 8 N (estimate) | contact friction 0.3 | PTFE tape on the groove's ceiling if high | spring scale on the clevis, pulled along the bar |
| Stirrup mass | 0.30 kg +- 0.02 | 0.30 kg | washers in the junction block | kitchen scale |
| Bar assembly mass | 0.60 kg | 0.60 kg | - | kitchen scale |
| Swing stops | 15 deg inward, 27 deg outward | the same | printed slot | protractor on the base board |
| Open stop | 100 deg | the same | position of the M3 stop screw | protractor against the bar |

## Where it differs from the model

- The shelf is plywood instead of sheet steel, and the model's 60 mm upholstered pad is left off.
  A 2 in foam pad on the top matches its look; nothing in the task touches the top.
- The arm is 6.35 mm thick (the model's is 6 mm).
- The lateral-lock spring is a compression spring under the bar.
  The model applies the same stiffness at the lift joint: 12 N m/rad there is 2.13 N/mm at 75 mm.

## Geometry checks (from `checks.txt`)

- The open-stop screw meets the bar's top face at 100.0 degrees, and clears it by 3.4 mm when folded.
- The boss stays below the bar's underside (136.8 mm): its cheeks top out at 135.5 mm and the trunnion at 136.3 mm.
  The folded stirrup lies on the bar's top face (143.2 mm), so it slides over the boss.
- The T-rail clears the window's lower edge by 5.4 mm with the bar tilted 3 degrees front-down.
- The spring is compressed 5.2 mm at preload and 15.7 mm at full lift, of 38.1 mm free length.
- The spring tower stays within 86.7 mm of the boss axis; the support rail starts 100 mm behind it.
