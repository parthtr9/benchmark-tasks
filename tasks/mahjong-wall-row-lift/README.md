# Build a mahjong wall: lift a loose row of 17 tiles onto the other row

On a mahjong mat lie one player's 34 tiles for the wall, face down in two loose rows of 17, side by side.
The wall section is finished when the near row lies on top of the far row, tile over tile, two high and straight, with every tile still face down.
The robot is told only that end state, and that the row goes up in a single lift.

Hand-stacked mahjong is played this way wherever there is no automatic table: before every hand, each of the four players builds 17 stacks of two.
A Japanese city's public guide to hand stacking gives the step: "両手で下山をはさみ前に押し出す。同様に上山を前に押しだし、下山の上にのせる。" (pinch the lower row between both hands and push it forward; in the same way push the upper row forward and put it on the lower row).
An English beginners' page describes the same lift as something that "takes some hand dexterity and practice".

What is tested is the lift itself.
Seventeen loose tiles are a beam only while they are pressed together end to end.
Pressed too lightly, the row sags apart as it leaves the mat.
Pressed hard by a hand that lets the press tilt its fingers, the end tiles tilt with them, the thrust moves to the tiles' top edges, and the row breaks the same way.
And the row is set down on a second loose row, which must stay where it is.

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
  --instruction "Finish this section of the mahjong wall: put the near row of tiles on top of the far row in a single lift, leaving a straight wall two tiles high with every tile face down." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/mahjong-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                                       # full score
DEMO_VARIANT=one_arm       python demo.py task.py out --no-record     # one hand lifts its end tile; the row stays: 0
DEMO_VARIANT=light         python demo.py task.py out --no-record     # a 6 mm press: the row sags apart on the way up: 0
DEMO_VARIANT=early_release python demo.py task.py out --no-record     # press let go half way across, 35 mm up: 0
DEMO_VARIANT=crooked       python demo.py task.py out --no-record     # set down 8 mm off the far row's line: below 1
DEMO_VARIANT=no_trim       python demo.py task.py out --no-record     # fingers left to tilt under the press: 0
python demo.py task.py --ik                                           # no simulator: IK of every target
```

1. Each hand comes straight down over one end tile of the near row, fingers closing across the row, and pinches it.
2. Both hands press toward each other: each is commanded 60 mm past where the row stops it, which through the arm's own springiness (about 0.2 N per mm sideways) is a press of about 12 N.
   While pressing, each hand keeps its fingers upright and on its line by its own joint readings.
3. Both lift the row 35 mm, carry it 60 mm to above the far row, and lower it until it bears lightly on the far row's tiles.
4. Both ease the press off slowly, each by as much as its own joints say the press had bent it, letting the trims go with it, then open and leave.

The demo is open loop on the fixed layout and the clock.
Its only feedback is each arm's own joint readings (its measured tool pose); no tile's pose steers it.

Why one arm is not enough: the row is 328 mm long and a gripper opens 94 mm.
One hand can lift an end tile, or a few neighbouring tiles; the rest of the row stays on the mat (`one_arm`).
One arm could build the same wall four tiles at a time in five trips; that is a different act, the instruction's "in a single lift" excludes it, and the scorer gives it 0 (covered by an offline test).

The demo reaches 1.0 (success) in 434 steps (22 s) of the 1,200-step limit, with the upper row within 0.9 mm across and 2.0 mm along of tile over tile, the far row not moved (under 0.1 mm), and the arms at least 112 mm apart.
The score stays at 0.75 from the step the grippers are clear until the wall has stood a full second by itself.
Two earlier Isaac Sim batches of the same demo, scored before the release was made part of that second, reached 1.0 in 415 and 418 steps.
Dream-1, given only the instruction, scored 0 in its one run of 1,200 steps (batch d94b8372): both hands came down beside the two rows and stayed there; no tile was lifted. The same batch's demo and variants repeated the scores above (demo 1.0; one_arm 0, light 0, early_release 0, crooked 0.5, no_trim 0).
The negative variants in Isaac Sim:

- `one_arm`: 0. One hand lifts its end tile onto the far row; the other 16 tiles stay on the mat.
- `light`: 0. With a 6 mm press (under 1 N once the gaps have closed) the hands carry only their two end tiles across; the row is never in the air.
- `early_release`: 0. Let go 35 mm up, half way across, the tiles fall in a heap between the rows; only 3 of the 17 end lying face down.
- `crooked`: 0.5. The row lies on the far row, 8 mm off its line: on top, but not tile over tile.
- `no_trim`: 0. With the hands left to tilt (13 degrees at this press), the row sags apart as it leaves the mat and the far row is knocked 25 mm.

Press commanded per hand against outcome (the press is about 0.2 N per mm): 40, 60 and 85 mm all reached 1.0 in an earlier Isaac Sim batch with 12 g tiles; in the offline rehearsal with these tiles 30 mm (about 6 N) fails and 35 to 85 mm (7 to 17 N) succeed.

## Scoring

Nothing counts without the one lift: at one moment all 17 tiles of the near row were at least 20 mm above the mat together, still one row.
As it stands, 20 mm leaves 4 mm above a tile lying on the far row (16 mm), so a row lifted only 16 to 20 mm scores 0 even if it is then set down right; the demo lifts 35 mm. A revision to 17.5 mm is prepared, not applied.
A tile lying on the far row is 16 mm above the mat, so tiles stacked a few at a time never meet it.
Once made, the lift stays made (it is a thing that happened, not an end state).

With the lift made, the highest level reached, each including the ones before it:

- 0.25: the lifted row is one row again, lying flat and face down, on the mat or on the far row: every tile within 5 mm of the straight line through its two end tiles, and no joint between neighbours open more than 3 mm.
- 0.5: and all 17 lie on the far row (underside 16 mm above the mat, within 4 mm).
- 0.75: and each lies within 5 mm, along and across the row, of the far tile beneath it; every far tile lies within 5 mm of where it started; all 34 are face down.
  As it stands, the far row is judged by where it lies: a finished wall whose far row was pushed more than 5 mm as a whole scores 0.5. The demo moves it under 1 mm; a revision that judges the far row by being straight and intact instead is prepared, not applied.
- 1.0: and for one whole second no tile's corner has moved more than 1 mm and both grippers have stayed at least 3 cm from every tile. A wall held still and then let go has to stand a full second by itself.

Every level counts only while both grippers are at least 3 cm from every tile, so a wall held together by the hands earns nothing.
0 for the whole trial if a tile leaves the table.
A tile turned over or pushed out of line on the mat is not a fault by itself: only the end state counts, and the end state needs all 34 face down.

## Fidelity

Rigid-body physics; nothing is scripted or attached.
The row is held only by contact and friction between the tiles and the finger pads, and falls apart whenever the physics says so.

- Tiles: each collides as one solid block at real size and mass (26 x 19 x 16 mm, 13.7 g) with square edges. What is drawn on it is drawn only: an ivory face layer, a yellow back layer, and the back's rounded top edge as a 1 mm step all round, so the seams between neighbouring tiles show as they do on real tiles. Real tiles' edges are rounded by about 1 mm; the simulated side faces are flat to the edge. The faces are not drawn, since every tile lies face down.
- Size and mass are as a product review lists them for Marushin Sangyo's urea-resin hand-play tiles ("高級麻雀牌 竹": 26 x 19 x 16 mm, "牌1つの重さは、約13.5~14g"; the same maker's "金龍": about 26 x 19 x 16 mm, 13.7 g a tile). The maker's own page was not read. Sellers' listings for Marushin sets (Marushin Netto Service, on kakaku.com) give "26×19×16mm" and a standard set weight of "約2000g" without case or accessories, about 14 g a tile; ALBAN's own page for its AMOS BEGIN hand-stacking tiles gives "約 26.0 × 19.5 × 16.0 mm" and "約 12 g" a tile (https://www.alban.co.jp/products/detail/708). So 26 mm hand-play sets run from 12 to 15 g a tile, and the demo was rehearsed at 12, 13.7 and 15 g; weigh the bought set.
- Estimates from photographs, for the drawing only: the back layer's thickness (6 mm) and the edge radius (1 mm).
- Friction: tile on tile 0.3 and tile on mat 0.4 are estimates (measure the bought set: tilt a tile on a tile, and on the mat, until it slides; the tangent of the angle is the friction).
- Mat: a rigid 2 mm sheet fixed to the table; a real mat gives slightly under a fingertip.
- Start gaps: a uniform 0.3 mm between neighbours; real rows pushed against a straightedge have gaps from 0 to about 0.5 mm.

## Real-world build

### Parts

| Part | What to buy | Size and mass | Measured or estimated |
|---|---|---|---|
| 34 mahjong tiles | A standard hand-play riichi set of 26 mm urea-resin tiles, for example Marushin Sangyo's "高級麻雀牌 竹" | 26 x 19 x 16 mm, 13.5 to 14 g a tile, as a product review lists it | Size and mass as listed; weigh ten tiles and measure one with a caliper |
| Mahjong mat | Any roll-up hand-play mat of about 68 to 70 cm square (for example Nintendo's Yakuman mat, listed 680 x 680 mm) | Lay it with 60 cm in front of the arms | Its thickness (2 mm in the simulator) is not critical |

No custom parts and no fixtures.

### Layout

Tape two lines on the mat, 25.0 and 31.0 cm in front of the midpoint between the arm bases, and a mark on the centre line.
Lay 17 tiles face down along each line against a straightedge, their 26 mm sides pointing away from the robot, each row centred on the centre line with its tiles just touching, then take the straightedge away.

### Forces

No force is set or adjusted: the task depends only on the tiles' mass and friction.
For reference, the row weighs 2.3 N; held by its two end tiles it needs an end-to-end press of roughly 7 N or more to stay a beam (the demo uses about 12 N).
To check a set: press a row of 17 between two flat blocks, one on a kitchen scale turned on its side, and lift; note the reading at which the row first holds.

### Real-trial checks for each scoring level

- The one lift: from the side-view video, a frame in which all 17 near-row tiles are above a 20 mm gauge block standing on the mat, in one row.
- 0.25: a straightedge along the lifted row: no tile more than 5 mm off it, no joint open more than 3 mm (feeler or ruler), every tile flat and face down.
- 0.5: all 17 lie on the far row (photo from the side).
- 0.75: with a ruler from above, each upper tile within 5 mm of the tile beneath it along and across the row; each far tile within 5 mm of its taped line and of its start mark; no face showing.
- 1.0: from the video, one whole second in which both grippers stay at least 3 cm from every tile and nothing moves.
- 0 for the whole trial: a tile on the floor or off the table's edge at any time.

## Who does this, and how often

Mahjong and its regional variants "are played throughout the Sinosphere in East and Southeast Asia and have also become popular in Western countries" (Wikipedia, "Mahjong"), and the same article calls it, as of 2010, "the most popular table game in Japan".
Wherever it is played without an automatic table, every player builds a wall section by hand before every hand, many times in an evening:

- Chinese play: "Each player then stacks a row of 18 tiles, two tiles high in front of them (for a total of 36 tiles)." (Wikipedia, "Mahjong", Stacking tiles)
- Japanese play: 17 stacks of two per player (Japanese Wikipedia, "壁牌": "34枚の牌は、17枚を上下2段に重ねて並べる"); with an automatic table "洗牌および砌牌の作業はすべて機械により自動的に行われる".
- American play: "each player stacking tiles in a line (wall), 2 high and 19 long, in front of their racks" (mahjong4friends.com, how to play American mah jongg).

The lift of a whole row between both hands is the method a Japanese city's public guide teaches (quoted above); an English beginners' page calls it something that takes practice.
Not verified here: a count of players, and how many of them lift the row whole rather than a few tiles at a time.

## Runtime and seeds

One fixed layout (`object_jitter=0`), no randomness.
Run limit 1,200 steps (1 minute at 20 Hz).
