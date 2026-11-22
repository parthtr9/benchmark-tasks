# Clamp spring and mount verification

This is a dimensional and catalogue calculation, not a report of a built fixture or measured instrument. The final spring-scale and free-motion acceptance checks are required before a real trial.

## Manufacturer source

[Lee Spring LC 030D 12 S product specification](https://www.leespring.com/product/compression-spring-lc030d12s-stainless-steel), checked 2026-11-22: OD 7.62 mm, ID 6.10 mm, free length 38.10 mm (tolerance 1.22 mm), solid height 10.52 mm, stainless steel, rate 7.53 N/cm = 0.753 N/mm, rate tolerance 0.75 N/cm = 0.075 N/mm; quantity 1–19 price8.31 USD. Use the N/cm value rather than the rounded0.75 N/mm display.

The earlier BOM incorrectly used12.8 mm as the old spring's solid height; its actual solid height is3.10 mm. Even corrected, that short spring had essentially no nominal margin at15 degrees after preload. The new spring provides substantially more usable travel.

## Mount, all dimensions in mm above the bench

The unchanged simulation puts the rocker pivot at y10,z201.45 at the start. Its upper lever surface is1.95 above the pivot. A vertical guided follower contacts that surface at x466,y49:39 mm forward of the pivot. Its2 x2 x1 mm tip has corners filed to0.5 mm radius; it slides across the polished/waxed lever top, rather than forcing the spring to bend sideways. An8 x8 x1 mm backing closes the guide-rod hole below the seat and connects the tip to it. The10 x10 x2 seat is4 mm above the contact, with top z207.45. An80 x5 mm smooth steel rod is glued2 mm into its5 mm bore, flush with the seat's underside; it passes through the bridge's5.2 mm vertical guide bore. The spring surrounds this rod (ID6.10 mm). The rod must slide with less than0.1 N drag without the spring.

The raised bridge's underside is z243.10; its5 mm rod is free to emerge above it. Initial spring-seat separation35.65 mm gives compression38.10-35.65=2.45 mm. The bridge upright intersects the carriage plate at x490–498,y-19–-10,z219.3–220.3 and reaches the bridge underside at243.10; glue both mating faces. The guide and upright are outside the clamp lip/cassette path. All these heights move together with the carriage, so its59 mm stroke does not change spring preload. The guide rod must have60 mm clear space above the bridge at maximum head height; no cover may be added there.

## Travel and force

Let t be rocker opening in radians, r=39 mm, h=1.95 mm and q=0.1015 m be the vertical distance from pivot to lever foot. A vertical follower on the rotated top surface rises

    rise(t) = r*tan(t) + h*(1/cos(t)-1)  [mm]
    compression(t) = 2.45 + rise(t)
    length(t) = 38.10 - compression(t)
    torque(t) = 0.753*compression(t)*0.039  [N m]
    foot_arm(t) = 0.1015*cos(t) - 0.040*sin(t)  [m]
    foot_force(t) = torque(t)/foot_arm(t)

At0 degrees: spring length35.65 mm, compression2.45 mm, torque0.0720 N m (model0.072 N m). At15 degrees: rise10.52 mm, length25.13 mm, force4.34 N at the foot (model about4.4 N). At18 degrees, the demo's opening, length22.88 mm; at0.45 rad (25.78 degrees, model upper stop), length16.60 mm, more than6 mm above10.52 mm solid height. Length decreases monotonically over the model interval[-0.02,0.45] because d(rise)/dt = r*sec(t)^2 + h*sec(t)*tan(t)>0. Thus the upper stop is the worst compression. Use a protractor to check the full0.45 rad range before use; do not force the fixture beyond that operating limit. Coil bind is never used as a stop.

With manufacturer rate tolerance, preload set between2.1 and3.1 mm, follower contact varying by1 mm either side, and0.1 N guide drag, the15-degree foot force remains approximately3.5–5.4 N. Use measured free length of the individual spring to set preload; do not assume the nominal length. Remove material or insert a shim under the upper spring seat to trim preload only within2.1–3.1 mm. Measure final force3.5–5.5 N at15 degrees. At maximum opening measure spring length at least14.0 mm (3.48 mm clear of catalogue solid height), and verify the follower tip remains on the lever's8 mm-wide upper surface throughout. Failure of any check invalidates the fixture until adjusted.

The rod's5 mm diameter is below the manufacturer's5.79 mm recommended rod diameter; the spring's7.62 mm OD does not pass through the5.2 mm rod-guide hole: it seats on the bridge underside. Do not drill an8 mm through-hole for the spring. The spring is guided internally, not constrained by an undersized outer pocket.

## Cassette height correction

The task has CASS_Z_TOP=190 mm and crank radius29.5 mm. At the initial60-degree crank angle, drop=29.5*(1-cos60)=14.75 mm, so cassette centre starts175.25 mm and reaches190.00 mm at top. The earlier fixture note160.3/175 subtracted the crank drop twice. No simulated geometry or initial pose was changed.
