# block-truth

**How long does the round take?**

> Two halves. This file covers **driven** rounds: a van visiting delivery points
> from a depot. [`WALK.md`](WALK.md) covers **walked** blocks - a postie on foot
> covering streets - which is a different job with different arithmetic, a
> different routing problem, and its own database shaped like the pipeline that
> already measures it. The two share the geometry, the interval machinery and
> the habit of checking rather than assuming.

Three questions, kept apart on purpose:

| | |
|---|---|
| A round that **has run** | what it actually took, derived from its records |
| A round that **has not** | what it will take, earliest and latest |
| A round **still to be planned** | how it could be driven, and what each way costs |

The first is a ground truth - the label a forecast is scored against - and
nothing in it is predicted. The second is a forecast and every screen that shows
one says so. They are never blended, and `kind` on every response says which one
you are looking at.

```
$ block-truth 2026-05-30 20260530-D01-V02
20260530-D01-V02   2026-05-30   D01-V02 / Large Van / D01 / Leeds
   [GROUND TRUTH]

  completion time      14:50       <- the answer
  duration             7h 13m   (433.3 min over 41 stops, 10.57 min each)

  how it is made up
    block start        07:37
  + depot, load/brief      18.0 min
  + travel, all legs      237.3 min   (includes the run out to stop 1)
  + service at doors      141.3 min   (38 delivered, 3 not)
  + return to depot        36.7 min   (residual: nobody is served on the way home)
  = completion         14:50

  checks
    [ok ] arrival chain: ... (worst gap 0.98 min at stop 7)
    [ok ] depot time agrees two ways: from the first arrival 18.00, from the totals 17.99
    [ok ] parts sum to the recorded duration (identity): ... it must, the return leg is the leftover
    [ok ] the drive home is possible: 36.7 min for the 12.3 km back to the depot;
          this round drove 3.54 min/km all day, so 9-99 min
    ...
```

---

## 1. Run it

```bash
pip install -r requirements.txt
./run.sh                 # seeds a database if there is none, then serves on :8100
```

Then open <http://127.0.0.1:8100>.

Step by step:

```bash
python3 -m block_truth.seed --days 21          # builds data/blocks.db
uvicorn block_truth.api:app --port 8100
```

Nothing is downloaded and nothing external is called. The database is built
locally in a couple of seconds.

**Already have the delivery pipeline's database?** Import its recorded rounds
instead of seeding synthetic ones:

```bash
python3 -m block_truth.seed --from-delivery-db ../delivery-agent/data/delivery.db
```

## 2. What is in the database

One SQLite file, `data/blocks.db`, three tables.

| Table | What it holds |
|---|---|
| `depots` | where rounds start and end |
| `blocks` | one vehicle's round on one day; `source` is `recorded`, `manual` or `excel` |
| `stops` | one delivery per row, in visiting order, with coordinates |

A **recorded** block has an arrival, a service time and a leg on every stop. An
**entered** block has none of them - that is the whole difference between a
ground truth and an estimate, and it is why `source` exists.

The seed is synthetic, and the point of how it is built is that it is
*coherent*: stops are placed on the map first and the times follow from the
geometry. So there is a real speed in those records, and the calibration below
has something genuine to recover rather than a number someone wrote down.

## 3. The ground truth

### How a round is laid out

```
block start --[depot: load and brief]--> depart
    `-[leg 1]-> stop 1 -[service]-> -[leg 2]-> stop 2 -[service]-> ... ->
                                          stop N -[service]-> -[return]-> back
```

Each stop records the leg that **led to it** and the time spent at its door.
Everything between leaving the depot and finishing the last doorstep is the sum
of those two columns; two pieces sit outside them, the depot time before the
first leg and the return leg after the last door.

| Quantity | From |
|---|---|
| door to door | last arrival + its service - first arrival | the stops alone |
| **depot minutes** | first arrival - block start - the first leg | the stops alone |
| **return leg** | recorded duration - work to the last door | needs the duration |
| **completion** | block start + recorded duration | |

The return leg is the one thing no stop row carries, because nobody is served on
the way home. Where a duration is recorded it falls out as the residual; where
it is not, completion is reported as of the last doorstep and `completion_source`
says so, rather than inventing a drive home.

### The checks

A completion time is only as good as the records under it, so each one comes
with assertions:

| Check | Catches |
|---|---|
| arrival chain | a stop that does not follow the one before it by its service plus the next leg |
| depot time agrees two ways | the two independent derivations disagreeing |
| return leg is not negative | a round recorded as ending before its last doorstep |
| no negative service or travel | impossible times |
| stop sequence is unbroken | a missing stop |
| the drive home is possible | a recorded duration that is simply too long |
| parts sum to the duration | nothing - and it says so |

The last row is labelled `(identity)` in the output because that is what it is.
Once the return leg is *defined* as the leftover, the parts sum to the duration
by construction and the test can never fail. Leaving it unlabelled would have
been the worst kind of check: one that looks like verification and performs
none.

Labelling it exposed the hole underneath. A duration recorded an hour too long
inflates only the leftover, and no stop row contradicts it, because the drive
home is exactly the part nothing else measures. What the stop rows cannot say,
the map can: the round's own legs against its own coordinates give the pace it
actually drove at, and the leftover has to be in keeping with the distance home
at something like that pace. The tolerance is taken from the recorded history
rather than picked - the leg home runs at 0.4 to 0.9 of the day's minutes per
kilometre, so the band is 0.2 to 2.0 - and across 1,291 seeded blocks on four
different seeds it produces no false alarms while catching an hour of invented
duration.

Arrival times are recorded to the minute, so a chain of N stops accumulates up
to half a minute of rounding per link; the tolerances allow for that and no
more. `--audit` lists only the blocks that fail something; on clean data it says
how many it checked and found nothing wrong with.

An audit that never fires proves nothing, so `tests/test_audit.py` tests it the
only way that means anything: copy the database, damage one block in it on
purpose, and insist the audit catches that block, by the right check, and blames
no other. Seven faults, one at a time. That is the test that found the vacuous
check above.

## 4. The estimate

### Why the earliest and the latest are not the sum of per-stop extremes

The tempting way to build a range is to add up the best case at every stop for
the minimum and the worst at every stop for the maximum. It is also badly wrong.
Forty stops all landing on their best case is not a fast day, it is a fantasy,
and a planner told "between 3h10 and 9h40" has learned nothing.

Errors partly cancel. Independent noise adds in *variance*, not in width, so a
round of N stops has a spread that grows like sqrt(N) while its length grows
like N - the longer the round, the tighter its interval in relative terms.

But not everything cancels. Traffic, weather and who turned up miss the whole
day together, and that part does not shrink with N at all. Without it, a long
round's interval would collapse to nothing, which is wrong in the other
direction. So:

```
variance  =  SUM(per-stop variance)  +  (common_mode x duration)^2
                  shrinks with N          does not shrink at all
```

The common-mode fraction is **measured**, not chosen: predict every recorded
round from its geometry alone, then solve for the fraction that makes the
model's own variance match the spread of the errors. On the seeded data it comes
out around 6%.

### Does the interval mean what it says?

Calibrate on the first three quarters of the history, estimate the rest:

| Nominal | Covers | Mean width |
|---|---|---|
| 80% | 79.1% | 0.81 h |
| 90% | 89.5% | 1.04 h |
| 95% | 95.3% | 1.24 h |

Point-estimate error is 5.2% of duration. `tests/test_estimate.py` runs that
coverage check as a test, so it cannot quietly drift.

### What gets fitted

Nothing here is a constant someone picked. From the recorded rounds:

- **travel**, by regressing each leg's recorded minutes on the road distance
  implied by its two ends. The slope is minutes per kilometre - the reciprocal
  of speed - and the intercept is what parking costs regardless of distance.
- **service time** per (door type, parcel size), with failed stops kept separate
  because a failed stop is a short stop.
- **depot time** per vehicle type, and the **return leg** per kilometre.
- **failure rate** per door type, which feeds back into the expected service time.

The **estimator** tab shows all of it with the evidence behind each number. On
the seeded data the fit recovers what the seed put in: London comes back at
12.9 km/h against 13 seeded, Leeds at 19.8 against 21, depot load at
12.9 / 16.9 / 22.5 minutes against 13 / 17 / 23.

## 5. The month ahead

A month of planned rounds, one per vehicle per working day, each with a forecast
and at least three ways it could be driven.

```bash
python3 -m block_truth.schedule --days 30      # or ./run.sh, which does it for you
block-truth --schedule --date 2026-11-03 --depot D01 --max-minutes 540 --max-km 130
block-truth --itineraries 20261103-D01-V00-P
```

**What is measured and what is not.** The *shape* of the schedule is read out of
the recorded history: which vehicles are on the road and of what type, how many
stops each actually carries, when it leaves, and how a Monday differs from a
Saturday. The *demand* is not - there is no order book in this database, and no
amount of history will produce next Tuesday's parcels. So the stops are drawn the
way the seeded history draws them: a realistic shape to plan against, not a
promise about which doors will be on the round. Every row carries
`source='planned'`, planned rounds never feed the calibration, and the tab says
so at the top.

A planned round is stored as a *route* - nearest neighbour with the crossings
taken out - with each stop's position on the original manifest kept beside it.
That second column is what lets "we saved two hours" mean anything.

### At least three ways to drive it

The same stops, four orders, each costed by the same estimator under the same
day. Averaged over the rounds of one day:

| Option | Hours | Waiting | Km | Windows kept |
|---|---|---|---|---|
| Steadiest | 4.80 | - | 40 | 70% |
| Least driving | 4.84 | - | 41 | 73% |
| Promises first | 6.11 | 19 min | 57 | **90%** |
| As the work arrived | 10.36 | - | 128 | 74% |

That table is the entire point. Keeping another seventeen percent of delivery
promises costs about an hour and a quarter and sixteen kilometres - and nobody
but the planner can say whether that is worth it, so the app does not decide. It
prices the choice and leaves it.

- **Least driving** is nearest neighbour then 2-opt until no swap helps. Shortest
  wheels on road; takes no notice of the clock or the promises.
- **Promises first** takes the nearest stop still reachable inside its window, and
  waits at the door when it arrives early. The obvious version of this - always
  drive to whatever is due soonest - was tried first and was a disaster,
  measurably: four times the kilometres to save a handful of windows. Deadlines
  alone are not a route.
- **Steadiest** is greedy on *predicted minutes at the hour each leg would be
  driven*, then 2-opt against the clock rather than the odometer. It declines to
  drive a long leg through the peak, and goes around what is closed. Where the
  day is quiet it comes out identical to "least driving" and is merged into it
  rather than listed twice under a different name.
- **As the work arrived** is the raw manifest. Not a serious option - it is the
  only honest answer to "better than what?".

Two options that come out as the same drive are merged, because four names for
one route would be a lie about having choices.

### Which one, and who decides

The app recommends one and the operator decides. The recommendation is not a
ranking nobody can inspect - it is one line of arithmetic, printed:

> **Recommended: Promises first.** 5h 42m of driving plus 2.5 broken promises at
> 20 min = 392 - the lowest of the 3 options, by 13 min over Least driving.

Two steps, both stated. An option that breaks an operating limit is set aside,
unless they all do - a round has to be driven by somebody, so if every option is
over the limit the app names the least bad one and says that is what it has
done. Among what is left, the lowest total of hours plus the price of the
promises it breaks.

**That price is the one number here that is not measured, and it is on the
screen as a box you can type in.** There is no fact in the delivery records
about how a depot trades an hour of vehicle time against a customer who was
promised a morning: that is a business decision. The default is twenty minutes a
missed window, half that for one that could go either way. Set it to zero and
the recommendation becomes "shortest day"; set it high and it becomes "keep every
promise". An operator who disagrees with the default can see exactly what to
change, instead of arguing with a verdict.

**Choosing applies it.** Pressing *Drive this one* reorders the round and records
who chose it and when. From then on every forecast of that round - on the
schedule, in the day's total hours, in its own detail - follows the choice, and
a choice against the recommendation is kept and shown as such rather than
quietly corrected.

The choice is a *policy*, not only an order. Picking "promises first" buys an
hour of waiting at doors, and the schedule charges for it; an earlier version
applied the order and dropped the waiting, which silently handed back time the
operator had decided to spend. The order is applied by rewriting the round's
visiting order rather than stored next to it, so there is one answer to "what
order is this round in" instead of two that drift apart. The manifest position
is left untouched, which is what makes *back to the manifest order* work and the
alternatives offerable afterwards.

### Seeing all of them, not just the winner

Every option is drawn, not only the chosen one:

- **A thumbnail each**, same stops, same scale, in its own colour. A tidy loop
  next to a scribble says more in one glance than any table: on a 27-stop round
  the plan is 24 km, promises-first is 39 km, and the raw manifest is 93 km of
  criss-crossing.
- **Traced over each other** on the full map, each line toggleable from the key.
  The manifest order starts switched off, because over thirty stops it is a
  hairball that hides everything underneath it. (The first version of this drew
  all of them at once and was unreadable - the thumbnails exist because of it.)
- **Stop by stop, side by side.** One row per stop in the order the work
  arrived, one column pair per option: the position it visits that stop at and
  the time it is forecast to get there, red where a promise is missed and amber
  where it could go either way. So you can see that the stop promised
  08:00-12:00 is visited 13th at 09:39 by one option, 15th at 10:09 by another,
  and first at 08:23 by a third.

**Operating limits** - shortest and longest round, maximum kilometres, maximum
stops - are *reported, never enforced*. A round over a limit is shown in red with
the amount it is over by. Silently dropping the only option that gets the work
done is not a kindness to whoever has to do it.

## 6. Traffic, measured - and closures, declared

Two kinds of knowledge, and the app never confuses them.

### Traffic is measured

Nobody tells this app that the morning peak is bad. Every recorded leg has a
distance implied by its two ends and a time recorded against it, and it was
driven at a known hour on a known weekday. Divide what it took by what the city's
average pace says it should have taken, average those ratios by hour and by
weekday, and the shape of the day falls out of the records:

```
   07:00  1.138  ################                slower    696 legs
   08:00  1.201  ########################        slower  2,116 legs
   09:00  1.025  ###                             slower  2,129 legs
   12:00  0.873  ###############                 faster    930 legs
   16:00  1.122  ##############                  slower    379 legs
   17:00  1.100  ############                    slower    222 legs
```

The curve is **relative**: 1.00 is the city's own all-day average, because the
per-kilometre figure the estimator already uses has the average traffic baked
into it. Hours with thin evidence are pulled back toward 1.00 in proportion to
how little stands behind them, so an hour with eleven legs cannot swing a
forecast.

Three things had to be got right for this to mean anything, and each one was
wrong first:

1. **Measure the ratio on the driving only.** A leg is minutes-per-kilometre
   times kilometres *plus* a fixed cost for parking and walking, and traffic
   slows the first part while leaving the second alone. Taking the ratio of the
   whole leg mixes them and reads every peak as shallower than it is - it put the
   08:00 peak at 1.13 instead of 1.20.
2. **Do not count the same variation twice.** The per-leg spread was fitted
   across the whole day, so it already contained the peaks the curve now
   explains. Left alone, a city with sharp peaks is charged for them twice and
   its rounds get quoted far wider than their evidence supports - London's
   intervals were over-covering at 0.61 standard deviations while Newcastle's
   under-covered at 1.45. The leg regression is refitted against kilometres
   *times the factor for the hour they were driven*.
3. **Cross-fit the leftover.** Measuring what the clock cannot explain using a
   curve fitted on the very same days flatters the curve. Two folds by date - fit
   on one half, measure on the other, then swap.

**Does it pay for itself?** On days the fit never saw:

| | Mean absolute error | 90% interval width |
|---|---|---|
| Assuming a flat day | 6.2 - 7.0% of duration | 1.16 - 1.37 h |
| With the measured traffic | 5.2 - 6.0% | 0.95 - 1.13 h |

About a sixth off the error and a sixth off the width, across four seeds. The
part of a round that is "something happened to the whole day" falls from about
7.2% to 5.7% once the clock accounts for it - that is the same fact seen from the
other side. `tests/test_estimate.py` asserts the improvement, so it cannot
quietly go away.

The honest caveat: traffic-aware intervals cover about a point less than nominal
(roughly 88% at a nominal 90%, against 89% for the flat model), which is inside
the sampling noise of 165 held-out rounds. Both models are mildly optimistic.
Widening them by a fudge factor would hide that rather than fix it, so the
figures are reported as measured.

### Closures are declared

A closed road leaves no trace in delivery records, and no amount of history will
produce one. So disruptions are rows somebody enters - by hand on the **On the
road** tab, or through the API - and `source` on each row keeps them apart from
anything measured. There is no live feed here: nothing in this app reaches the
network.

| Kind | What it does to a leg |
|---|---|
| `closure` | driven around: the detour is added in kilometres and minutes |
| `works`, `congestion`, `event` | slows the part of the leg inside the area, by `severity` |

A closure is a detour, not a delay. For a straight leg crossing a blocked disc,
going round it adds about `(pi/2 - 1)` times the chord it replaces - which for a
leg through the centre gives exactly the half-circumference answer, so the
approximation is right where it matters most. And because a detour is a guess at
a route nobody has driven, it adds *uncertainty* as well as minutes: 40% of
whatever it adds. A closure makes a forecast both longer and vaguer, which is the
truth about it.

A stop that sits **inside** a closure is named, loudly, on the round:

> **Stops 13, 14, 15 are inside something declared closed.** No route fixes that -
> they need moving to another day or another vehicle. The forecast still delivers
> them, so treat it as optimistic.

Declaring a closure across the middle of a round, and watching the plan change:

```
before            3.91 h    26.6 km
closure, 1.2 km   4.02 h    28.7 km   detour 2.1 km, interval +/-14.3 -> +/-14.6 min
congestion 2.2x   4.17 h    26.6 km   16 min from the works, no extra distance
                                      (and nothing at all outside 07:00-10:00)
```

## 7. Entering a round

**By hand.** Pick a depot, then click the map to drop stops in visiting order.
The estimate updates as you go - expected completion, earliest, latest, and
where the time goes. Click a stop to remove it. There is a *paste coordinates*
fallback for when you already have a list.

**From a spreadsheet.** Download the template, or bring your own: column names
are matched loosely, so `Stop No`, `stop_no` and `sequence` all work, and so do
`Y`/`X` for coordinates. Only sequence, lat and lon are required.

Every row is validated on its own and every problem is reported with its row
number:

```
row 3  lat/lon            'oops', -1.54 are not coordinates
row 4  point_type         'Mansion' is not one of House, Flat, Office, ...
row 4  promised_from/to   the window closes (09:00) before it opens (13:00)
row 5  sequence           stop 3 appears twice
```

**Nothing is written until the whole sheet parses.** A half-imported round is
worse than a rejected one.

## 8. The map

Two pictures of the same round, and they are **not the same claim**. A switch
above the map chooses between them.

**As measured** is the drawing: these stops, in this order, legs scaled by the
road factor. Every number in this app comes from it, and it needs nothing from
the network.

**On a map** puts the same stops where they really are, on real tiles. The
dashed line between them is still the straight leg - it looks like a route and
is not one, and the panel says so. Pressing *trace it along the roads* asks a
routing service for a path that follows streets and draws it beside the dashed
one. That path changes no forecast; it is a third party's opinion about roads
this app has never seen.

What it does buy is a measurement worth having:

> **59.5 km by road** against 47.4 km as the crow flies: a road factor of
> **1.26** here, where this app assumes 1.3.

That is the `ROAD_FACTOR` constant being checked against reality, per block. If
a patch comes back at 1.45, the forecasts for it are short and now you know.

Three deliberate choices:

- **Leaflet ships with the app** (`web/vendor/leaflet`, BSD-2-Clause), not from a
  CDN. The rest of the page asks nothing of the network and works in a building
  with no route to the internet; pulling the map library from cdnjs would have
  quietly ended that.
- **The tile URL and the router are settings on the page.** The defaults are
  OpenStreetMap and the public OSRM demo; an internal tile server and an
  internal router work as well, and for real delivery points they are the right
  answer.
- **Nothing leaves the machine until you press the button.** Tracing the roads
  sends the round's coordinates to whatever address is set as the router, and
  the page says so immediately above the button.

If the tiles do not load, the stops and the path are still placed correctly and
the page says the tiles are missing. If the router cannot be reached, it says
that too, and the drawn view is untouched.

Depot, every stop in visiting order, then back. Arrows show direction, stops are
numbered, and the scale bar is there because distance on a drawn map means
nothing without one. Recorded rounds colour failed stops red; estimated rounds
colour by whether the stop makes its promised window, with amber for *could go
either way*.

**It is the order of visits, not a road route.** There is no road network and no
GPS trace in the data, so legs are straight lines between coordinates, scaled by
a road factor of 1.30 to approximate driven distance. That is a real measurement
of where the vehicle went and in what order; it is not a route a satnav would
give you, and nothing here pretends otherwise.

## 9. Using it

**The page** at `/` - blocks, the month ahead, on the road, new block, spreadsheet,
estimator.

**HTTP**, docs at `/api/docs`:

| | |
|---|---|
| `GET /api/blocks` | by date, source, depot |
| `GET /api/block/{id}` | ground truth or estimate, with the route |
| `POST /api/estimate` | estimate a round without saving it |
| `POST /api/block` | save a round |
| `POST /api/upload` | parse a workbook, save nothing |
| `GET /api/template.xlsx` | the template |
| `GET /api/calibration` | what the estimator learned |
| `GET /api/audit/{date}` | recorded blocks whose records disagree |
| `GET /api/datacheck` | rows that will break a forecast, found before they do |
| `GET /api/schedule/days` | the month ahead, a row per day |
| `GET /api/schedule` | planned rounds with predictions; filters and limits |
| `GET /api/itineraries/{id}` | the ways to drive one planned round, each costed |
| `POST /api/itineraries/{id}/choose` | drive this one from now on |
| `POST /api/itineraries/{id}/reset` | back to the order the work arrived in |
| `POST /api/schedule/generate` | rebuild the plan from the fleet on record |
| `GET /api/conditions` | the traffic curve and the declared disruptions |
| `POST /api/disruption` | declare a closure or a jam |
| `DELETE /api/disruption/{id}` | take one back off |

**Command line**:

```bash
block-truth 2026-05-30 20260530-D01-V00      # one block, with its working
block-truth 2026-05-30                        # every block that day
block-truth --audit 2026-05-30                # only blocks that fail a check
block-truth --calibration                     # what the estimator learned
block-truth --import round.xlsx --depot D01 --date 2026-06-02
block-truth 2026-05-30 --csv > labels.csv     # for scoring a forecast against

block-truth --plan-month 30                   # plan the next 30 days
block-truth --schedule --date 2026-11-03      # what is planned, with predictions
block-truth --itineraries 20261103-D01-V00-P  # the ways to drive one round
block-truth --itineraries BLOCK --choose windows --by alex    # pick one, for good
block-truth --itineraries BLOCK --reset-order                 # back to the manifest
block-truth --itineraries BLOCK --per-missed-window 60        # reprice a promise
block-truth --conditions                      # the traffic curve and what is closed
block-truth --datacheck                       # rows that will break a forecast
block-truth --schedule --date 2026-11-03 --max-minutes 540 --max-km 130
block-truth --itineraries BLOCK --no-traffic  # forecast a flat day, for comparison
```

Exit codes: `0` fine, `1` a block failed a check, `2` could not answer at all.

**As a library** - the calculation has no I/O and takes plain objects:

```python
from block_truth.truth import compute
from block_truth.model import Stop
compute("B1", "2026-05-14", 474.0,
        [Stop(1, 500.0, 4.0, 26.0, "Delivered")], duration_recorded=278.1)

from block_truth.estimate import estimate, StopPlan
from block_truth.calibrate import fit
estimate("B2", "2026-06-01", "07:30", (53.80, -1.55),
         [StopPlan(1, 53.805, -1.545)], fit("data/blocks.db"))
```

## 10. Tests

```bash
python3 -m unittest discover -s tests -v      # 146 tests
```

- `test_truth.py` - hand-built rounds where the answer is known by construction,
  plus minute-rounded arrivals, a round past midnight, a broken chain, a
  duration shorter than the work it records, a missing stop, no duration at all.
- `test_estimate.py` - that the fit recovers what the seed put in, that the 90%
  interval covers about 90% out of sample, and the properties that must hold
  whatever the data says: longer rounds have relatively tighter intervals, the
  interval never collapses, the parts always sum to the expected duration.
- `test_excel.py` - odd column names, a title row, Excel time cells, blank
  trailing rows, five kinds of bad row at once, and storing a block.
- `test_audit.py` - the checks against a database broken on purpose, one fault
  at a time, each one caught by the right check and blaming nothing else.
- `test_schedule.py` - that the traffic curve recovers the shape the seed hid in
  the records (the peaks where they were put, Saturday lightest, Friday worst,
  London peaking hardest), that a declared closure lengthens *and* widens the
  forecast while congestion costs time without distance, that the routing options
  are genuinely different drives and that keeping promises costs something, and
  that a planned round is never mistaken for one that ran. Also that choosing an
  option applies exactly that order and survives a reopen, that it carries its
  waiting policy into every later forecast, that the manifest survives so the
  choice can be undone, and that the recommendation obeys its own arithmetic and
  moves when the price of a promise moves.

The suite needs nothing but the standard library and openpyxl.

## 11. What this does not do

- **It does not know the roads.** It reorders stops - that is what the month-ahead
  tab is for - but every leg is a straight line scaled by a constant, so a route
  it calls shortest is shortest as the crow flies, not as a satnav would have it.
  A real road network would change the numbers and not the method.
- **Traffic is this fleet's own history, not the city's.** If the vans only ever
  work mornings, the app knows nothing about four o'clock and says so, with the
  leg count behind every hour and a curve pulled back toward 1.00 where the
  evidence is thin.
- **Nothing here is live.** There is no feed of incidents, roadworks or weather;
  disruptions are declared by hand and the app says `declared` wherever one
  changes an answer. Nothing in this app reaches the network at all.
- **The month ahead is a plan, not an order book.** The fleet, the volumes and
  the weekday pattern are measured; the stops on a future round are a realistic
  shape to plan against and nothing more.
- **It does not repair records.** A failing check is reported, not patched. The
  completion time is still returned, with the reason it should not be trusted.
- **It does not learn from entered blocks.** Only recorded rounds feed the
  calibration, because an estimate is not evidence.
- **The estimate is only as good as the history behind it.** `blocks_behind_it`
  on every estimate says how many rounds it rests on, and a database with none
  falls back to priors and says `blocks_used: 0`.
- **A row it cannot use is named, not silently dropped.** A stop with no
  coordinates used to reach the distance arithmetic and come back as "must be
  real number, not NoneType" from inside `math.radians` - a stack trace naming a
  trigonometric function and not the block, with the whole month-ahead page
  blank behind it. Such rows are now skipped, counted and listed on the page,
  because a forecast quietly made from eighty of a hundred stops is worse than
  an error. `--datacheck` finds them before they bite.
- Arrival times recorded to the minute put a floor of about half a minute per
  stop on how precisely anything inside a round can be placed. The completion
  time does not inherit that, since it comes from the start and the duration.
