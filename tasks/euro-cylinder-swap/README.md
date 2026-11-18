# Change a door's euro lock cylinder: steady the door, find the key angle that frees the cam, draw it out, fit the new one

A 250 x 300 mm section of a 44 mm door hangs on two butt hinges from a hinge post and stands open, as a door does while its lock is worked on.
In its free edge is a euro-profile mortice deadlock (Union L2149, 2.5 in case) with a brass double euro cylinder (ERA 5-pin 30/30, 60 mm) in it, key in.
The cylinder's fixing screw has already been taken out and lies in a magnetic parts tray on the bench, next to the new cylinder (the same ERA 30/30 in satin nickel, with its key in, as supplied), which stands on its end.

The robot is told only the goal: "Change the door's lock cylinder for the new one on the bench, leaving the new one fitted in the lock and the old one in the tray, the door where it was; the fixing screw stays in the tray."

What makes it hard is how a euro cylinder is held.
Its cam turns with the key between the two side plates of the lock case.
At the key-out position the cam rests 30 degrees off the cylinder's centre line and overhangs the plates' profile-shaped apertures by about 5 mm, so the cylinder cannot slide even with the screw out.
Only with the key turned about 30 degrees (within about 8 degrees either way) is the cam in line with the web, and then the cylinder slides.
The cam cannot be seen.
Published guides tell a person to find the angle by feel: "about 25 degrees off of the unlocked key removal position", "the cam is not visible so this has to be done by feel" (PracticalDIY), "try turning it and pulling the lock gently towards you. When it aligns you'll feel the lock start moving" (Lockshop Warehouse).
The key is captive whenever it is turned away from the key-out position, which is why pulling the turned key pulls the cylinder.
The door is on free hinges, so a pull or push on the cylinder swings it unless the other hand steadies the door.
The new cylinder goes in the same way: turned key, light push, until it is centred between the plates, and only there can the key be turned back, which locks it in.

Why it matters.
Changing the lock cylinder after moving in, after losing a key or after a break-in is one of the commonest do-it-yourself lock jobs in the UK and Europe, where the euro cylinder is the standard door lock; retailers and repair sites publish step-by-step guides for it (iFixit "Euro Cylinder Replacement", PracticalDIY, Lockshop Warehouse).
It is also routine work for general maintenance and repair workers (O*NET 49-9071.00) and locksmiths (49-9094.00).

Nearest existing tasks, and the difference:
PR 2290 reverses a mortise lock's latchbolt (a sprung bolt in a bench-held case, no key and no door);
PR 843 keys two plugs alike (pinning, with the plugs out);
PR 1263 locks a multipoint door (operating a lock);
the accepted lever-set task (PR 2018) fits a spindle and posts through a latch and drives screws;
a bayonet cap is pushed against a spring and turned to a stop.
Here there is no spring and no stop: the turn is a search for a hidden phase under a light pull, the same key is captive or free depending on its angle, and the part passes right through a door that swings.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: door, lock, both cylinders (each with its key in), tray and screw, `success()` and `score()` |
| `partkit.py` | Rigid parts built from boxes, with spring sliders and hinges, for yam_arena task files |
| `meshkit.py` | Convex mesh shapes for yam_arena task files (the cylinders' chamfered barrel and round web, the tray, the screw) |
| `demo.py` | The scripted full-score demo and its negative variants |
| `task.json` | Title, instruction, setup, scoring and fidelity for the board |

All geometry is built in code by the task's authors from the dimensions below; no third-party models or library assets are used.

## Run a trial

```bash
inspect-robots run --embodiment yam_arena --policy dream -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 \
  --instruction "Change the door's lock cylinder for the new one on the bench, leaving the new one fitted in the lock and the old one in the tray, the door where it was; the fixing screw stays in the tray." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir runs/dream
```

## Scripted demo

```bash
python demo.py task.py demo_out                 # full score
DEMO_VARIANT=no_turn python demo.py task.py out_no_turn --no-record
DEMO_VARIANT=wrong_angle python demo.py task.py out_wrong_angle --no-record
DEMO_VARIANT=cam_in_line python demo.py task.py out_cam_in_line --no-record
DEMO_VARIANT=no_brace python demo.py task.py out_no_brace --no-record
python demo.py task.py --dry                    # no simulator: the script on the arm kinematics alone
```

The demo is open loop.
Every target comes from the task file's fixed layout and the clock.
The only feedback is each arm's own joint readings: a trim that keeps the measured tool point on the plan under load, and the feel for the cam described below.
It reads no object pose, no joint of the lock or the door and no task state.

1. The left hand grips the door's free edge above the lock and holds it for the whole job.
2. The right hand takes the old key by its bow and feels for the cam the way a person does: it turns the key with no load on it, gives a light tug (its target is set 5 mm ahead of where its own joints say the hand is, brought on over 0.3 s), and if its joints show the hand has not moved 4 mm it eases off and tries another angle.
   The angles tried are 7 degrees each way, then 14, 21, 28, 35 and 42 degrees each way in turn: the old cam's angle is a trial setting the demo does not read, its window is 17 degrees wide, and the plug follows the key about 3 degrees behind, so one try falls inside the window for any cam angle up to 42 degrees either way.
   Turning under a steady pull is avoided: it wedges the cam's corner in the edge of the aperture.
   When the hand moves, the cam is in line, and it draws the cylinder out by the key.
3. It turns the hand down so that the cylinder hangs from its key, sets the cylinder's far end on the tray's floor, lays it down and lets go.
4. It comes down over the new cylinder's key and takes the barrel by its sides, the key's bow between its fingers, turns level and pushes the cylinder into the door's hole.
   If its joints show the hand has stopped at the door's face, the cylinder's end is beside the hole: the hand circles outward under a lighter push (2.4 mm a turn, 7 mm at most) until it moves in.
   The cam, still out of line, stops the cylinder on the door's face 25 mm in.
5. The hand opens, slides back to the key's bow, closes on it and lifts it level.
   A push with the key at 14 degrees meets the cam on the door's face; with the key at 29 degrees the push goes in.
   Held against the door's face by its cam, the cylinder was exactly 25 mm deep, so the hand pushes it on by the known 27 mm to centred.
   It turns the key back to 5 degrees in 4 degree steps (the cam then stands 25 degrees out of line); if the wrist's own reading falls 2.5 degrees behind its command, the cam is caught on a plate, and the hand moves the cylinder 0.8 to 2.4 mm along its axis (by its own joint readings, under a light load) and tries again.
6. The right hand lets go of the key and backs away, the left hand lets go of the door, and both hands move away.
   The keys stay in their cylinders, as a turned key does.

Negative variants, and why each fails:

| Variant | What the hands do | What happens | Score |
|---|---|---|---|
| `no_turn` | The old key is pulled without being turned | The cam holds the cylinder in the case: it moves 1.1 mm and stops | 0 |
| `wrong_angle` | The old key is turned 10 degrees only and pulled hard | The cam, 20 degrees off line, holds the cylinder in the case: it moves 1.1 mm and stops | 0 |
| `cam_in_line` | As the demo, but the new key is not turned back | The cylinder is centred but its cam is in line, so it is free to slide out again | 0.5 |
| `no_brace` | As the demo, but the left hand stays away from the door | The door swings with the pull (14 degrees) and the cylinder stays in the lock (0.7 mm out) | 0 |

## Revision 2: what changed

Applied before posting, after an internal pre-review; the first version was never posted, and the scorer change came with a new Dream-1 run:

- The 1 s stillness for full score now counts only steps with both grippers clear of all three parts (the old cylinder, the new cylinder and the door): before, parts held motionless for a second and then let go scored 1.0 on the first released step.
- The old cylinder's cam angle is a trial setting (`ECS_CAM_DEG`, default 30 as before), and the demo's search starts 7 degrees off key-out and works outward both ways, so it really searches: it frees the cylinder at whichever angle the trial uses, by the arm's own joint readings.
- The run limit stays 1,800 steps (90 s): the search makes the demo longer (1,192 steps at the default angle, against 917 before), so a 1,200-step limit would leave no margin.
- Nothing else changed: the scene at the default setting, the instruction, the levels, the variants and the forces section.

## Scoring

End state, each level including the ones before it, and each counting only once both grippers are at least 3 cm from the parts it names.
Everything is read from a ruler, a protractor on the hinge line and the video.

- 0.25: the old cylinder is wholly out of the door and lies in the tray.
- 0.5: the new cylinder is through the lock case with its fixing-screw hole within 1.5 mm of the forend's screw hole (measured along the cylinder: both ends stand 8 mm proud of the door's faces when it is centred), its axis within 2 mm and 3 degrees of the lock's.
- 0.75: its cam is at least 15 degrees off the centre line (it then overhangs the case aperture by about 1.5 mm, so the cylinder cannot be drawn out).
- 1.0: also the door is back within 5 degrees of its start angle and let go, the screw is in the tray, and none of the parts (each cylinder and its key, the door, the screw) has moved more than 1 mm for 1 s.
- 0 for good: a cylinder left the bench (5 cm or more below the bench top).
  A door swung by a pull is not a fault in itself: it only has to end where it started.

The 1.5 mm tolerance is the play the lock itself leaves: the 9.7 mm cam sits between plates 12 mm apart, so a cylinder whose cam has been turned back inside the case is within 1.15 mm of centred.
In a real trial the cam's angle is read from the key's angle, and checked by a light pull on the key: a cylinder whose cam is out of line does not move.
The scorer does not care which arm does what.

## Runtime and seeds

One trial is 1,800 control steps at 20 Hz (90 s); with revision 2 the demo reaches 1.0 in 1,192 steps at the default cam angle (Isaac Sim batches da4ce8d9 and b20845fc), and in 936 and 1,132 steps with the cam set to 18 and -25 degrees (batch dec7cd9d).
The first version's demo took 917 steps (batches 90b6b6eb, 7ae512f6 and 2101e933); its Dream-1 run (2101e933, 0.0) was on the first scorer and is replaced by the run on this one.
Dream-1, given only the instruction, one run of 1,800 steps on this scorer (batch b20845fc): 0.0.
The left arm went to the old cylinder's key at the door's free edge and pressed on the door, which swung; the right arm knocked the new cylinder over and pushed it along the bench toward the tray.
The old cylinder never left the lock.
`object_jitter=0`: one fixed layout, no seeds.

## Fidelity

- Rigid bodies only.
  The cylinder is a body, and a plug on a hinge carrying the cam and the key: with the right key in, plug and key turn as one, and a turned key cannot be withdrawn.
  The pin tumblers are therefore not modelled, and the keys stay in their cylinders for the whole job (the task never takes a key out).
  The second keyway on the far side is left out.
- The lock case is its two side plates with the profile aperture, 12 mm apart, in a 44 mm door with its own slightly larger hole.
  The bolt, levers and springs inside the case are not modelled: the bolt is withdrawn, and between the key-out and in-line positions the cam does not reach them.
- Threads are excluded: the fixing screw starts removed and stays in the tray.
  Measured on this simulator's contact model, a fine thread rides its crests and advances 0.01 to 0.09 mm a turn instead of its pitch, so driving the M5 screw cannot be simulated faithfully.
  The job therefore ends with the new cylinder captive by its cam: turned 15 degrees or more out of line, the cam overhangs the case aperture and the cylinder cannot be drawn out (the 0.75 and 1.0 levels), which is the state in which a fitter drives the screw home.
- Friction: 0.2 between cylinder, case and door hole; 0.5 on the door's faces.
  A finger pad's contact on the cylinder's barrel is given a 6 mm patch radius, so that a pinched barrel resists twisting between the pads as it does between rubber pads.
  The plug's turning friction is 0.006 N m of Coulomb friction on the plug's hinge (about 0.5 N at the key's bow, 12 mm from the axis), standing for the pin stacks and the cam's clutch spring; the first build's sprung shoe rubbing the bore snagged on the bore's box edges and was removed (see History).
  It is an estimate and may be lower than a real cylinder's: if a spring scale on the bought key's bow reads more (see the force checks below), the constant is raised to match and the demo run again.
  The door's hinges have 0.03 N m of Coulomb friction on their hinge (estimate), so a door that is let go comes to rest.
- Round parts are 16-sided with their corners on the real radius, and holes have their flats on the real bore, so the 0.5 mm fit is 0.5 to 0.8 mm round the barrel.
- Estimates, to be measured on the bought parts: plug 13 mm; cam cut-out 10.3 mm; 0.5 mm a side between cylinder and case apertures; 1 mm a side in the door's hole (a 19 mm bore and a 12 mm slot); 0.8 mm chamfer on the cylinder's ends; the cylinder's height in the case (36 mm above its lower edge); plates 1.5 mm; door section 1.3 kg; hinge friction; tray wall drawn 2.5 mm thick.

## Real-world build

Parts:

| Part | Source | Size and mass | From the standard or maker, or estimated |
|---|---|---|---|
| ERA 5-pin double euro cylinder 30/30, x2 (brass 4051-31, satin 4050-51), 3 keys each | [ERA sheet](https://checkout.mkm.com/content/products/pdf/49443/era-4545-euro-double-6-pin-cylinder-satin-b114161_TechData_1.pdf); [ITS listing](https://www.its.co.uk/pd/4050-51-ERA-3030-Euro-Profile-5-Pin-Double-Cylinder---Satin-_ERA405051.htm) | 33 x 17 x 60 mm, 0.21 kg, M5 fixing screw | Listed |
| Euro profile (barrel 17 mm, web 10 mm, axis to web bottom 24.3 mm, axis to screw hole 19.5 mm, cam rest 30 degrees) | DIN 18252 profile; [Giesse drawing](https://products.quanex-international.com/pub/media/scambio/GIESSE/SCHEDA_TECNICA/EN/FAC_MEZZO_CILIND_SING_PROF_EUROPEO.pdf), [Kaba sheet](https://www.specifiedby.com/kaba/kaba-expert/Kaba_Kaba-experT_Specifications_double-cylinder-europrofile-dz.pdf) | as stated | Standard |
| Cam (tip radius 15 mm, 6.6 mm wide, 9.7 mm thick, hub 16.7 mm) | [SimonsVoss cam drawing](https://help.simons-voss.com/system3060/cylinderax/en-GB/1216287627.html) | as stated | Maker's drawing |
| Union L2149 euro-profile mortice deadlock, 2.5 in | [Union sheet](https://www.lockshop-warehouse.co.uk/acatalog/2149.pdf) | case 65 x 108 x 15 mm, backset 44.5 mm, forend 25.5 x 152.5 mm, 0.58 kg | Listed; plates 1.5 mm estimated |
| 44 mm FD30 door blank, a 250 x 300 mm piece | [LPD Firecheck](https://lpddoors.co.uk/products/door-blank-firecheck-blanks-hardwood-lipped-fd30-44mm/) | 44 mm thick, 14.6 kg/m2, so the piece is about 1.1 to 1.3 kg | Listed thickness and areal mass |
| Two 75 mm steel butt hinges; 32 x 115 mm door lining (320 mm) as the hinge post; 18 mm plywood base 160 x 405 mm | any | as stated | Builder's choice |
| Magnetic parts tray | [Laser 2908](https://lasertools.toolconnection.co.uk/pis/2908_PIS.pdf) | 150 mm across, 20 mm deep, about 0.34 kg | Listed size; mass from a similar tray |
| Push-on rubber key caps (identifier caps), one per cylinder's key | any | about 26 x 26 x 6 mm over the bow | Estimate |

Estimated (to measure on the bought parts): the plug's 13 mm, the cam cut-out's 10.3 mm, 0.5 mm a side between the cylinder and the case apertures, 1 mm a side in the door's hole, the 0.8 mm chamfer on the cylinder's ends, the cylinder's height in the case, the plug's turning friction (0.006 N m) and the door hinges' friction (0.03 N m).

Build:

1. Cut the mortise for the lock case in one 300 mm edge of the door piece, with the forend's centre 116 mm above the piece's bottom edge (138 mm above the bench once hung), and fit the lock with its two forend screws.
2. Bore the cylinder hole through both faces with the lock's template: a 19 mm bore and a 12 mm slot, 44.5 mm from the edge.
3. Screw the hinge post upright to the base board and hang the door piece on it with the two hinges so that it clears the base by 4 mm and swings freely both ways from the position square to the post's wide face.
   Clamp the base board to the bench with the door's near face 300 mm in front of the arm bases and its free edge 74.5 mm left of the centre line.
4. Fit the brass cylinder without its fixing screw, key in, cam side toward the forend as normal; lay the screw in the tray.
   Withdraw the bolt.

Forces, and how each is checked once a session:

| Force | In the model | Check |
|---|---|---|
| The cam holding the cylinder | contact (cam against the case plates) | With the key at key-out, or turned 10 degrees from it, the cylinder must not pull out under 10 N on a spring scale hooked through the key's bow |
| The cylinder sliding with the cam in line | friction 0.2 in the case and the door's hole | With the key turned about 30 degrees clockwise it must slide out under less than 3 N |
| The plug turning | 0.006 N m of Coulomb friction on the plug's hinge (estimate) | With the cylinder held, a spring scale hooked on the key's bow 12 mm from the plug's axis, pulled square to the bow, starts the key turning: the model's value is 0.5 N; record the real reading, and a reading over 0.7 N (0.008 N m) goes into `task.py` |
| The door swinging | 0.03 N m of Coulomb friction on the door's hinge (estimate) | A spring scale at the free edge, 250 mm from the hinge line, starts the door swinging at about 0.12 N (0.03 N m); it comes to rest when let go |

Also measure the cam's rest angle and the key angle at which the cylinder slides; the scoring thresholds assume 30 degrees between them.

The old cylinder's release angle is a trial setting (`ECS_CAM_DEG`, degrees, negative for the other way; default 30, the scene as first submitted): where a fitted cylinder's cam rests depends on the make and on which way the key was last turned, so in a real trial the key may have to be turned either way.
It is not stated anywhere the policy can read: the instruction does not mention it and the cam is hidden inside the case.
The scripted demo searches for it by feel (turn, light tug, ease off, turn further, both ways in turn) and frees it at whichever angle the setting puts it (Isaac: see "Revision 2").
To set up a real trial, fit the old cylinder with its key turned to the chosen angle before turning it back to key-out, or use a cylinder whose cam rests on the other side.

Real-trial checks for each scoring level:

- 0.25: the old cylinder lies in the tray (photo from above), no gripper within 3 cm of it.
- 0.5: the new cylinder is centred: both ends stand 8 mm proud of the door's faces, measured with a rule on each side (within 1.5 mm).
- 0.75: its cam is out of line: the key's angle is read against a mark on the door's face, and a light pull on the key (under 10 N) does not move the cylinder.
- 1.0: the door is back within 5 degrees of its start angle (a protractor on the hinge line), the screw is in the tray, no gripper is within 3 cm of the parts, and nothing moves for 1 s in the video.
- 0 for good: a cylinder on the floor.

Reset:
put the old cylinder back in the lock (key turned to slide it in, then turned back to the key-out position and left in), the door square to the post, the new cylinder standing on its inner end on the bench at x 150 mm, y -270 mm with its key up and its web toward the robot, the tray at x 285 mm, y -410 mm with the screw in it.

Safety: nothing hazardous; the door piece can pinch fingers at the hinge side.

## History

This is a revival.
The first build (2026-11-17) modelled each key as a loose part held by one sprung pin tumbler, and the plug's friction by a sprung shoe rubbing the bore.
Its eight Isaac runs scored 0: the key slid over the pin, the pads slipped off a bare 2.2 mm key bow, and the plug stopped at 19 degrees whatever the key was turned to (the shoe snagging on the bore's box edges).
The revival (2026-11-18) keeps the key in its plug as one hinge, as a turned key is, gives the plug and the door their friction on their hinges, scores each level only after release with a whole-pose 1 s stillness check, and judges the door by where it ends.
It passed on its third Isaac batch.
