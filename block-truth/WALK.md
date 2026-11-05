# Walked blocks

The other half of this repository. `README.md` covers **driven** rounds - a van
visiting delivery points from a depot. This covers **walked** blocks: a postie
on foot covering streets, which is a different job with different arithmetic and
a different right answer for routing.

It is built to sit alongside a pipeline that already exists - OSM segments,
geofences sized by delivery-point distance, pings matched to segments, segment
traversal times summed into a "does take" per device. Nothing that pipeline
computes is computed again here. What is added is the other three questions:

| | |
|---|---|
| **does take** | theirs: what the walk took, from the pings |
| **is it measured right** | the audit: coverage, overlap, passes, fence width |
| **should take** | the forecast, by region, walker, vehicle and mail type |
| **what should it cost** | the route, as an arc problem rather than a tour |
| **what does it need** | resourcing: people, overtime, and what December does |

```bash
walk --build                     # two years of mock data, about 35 seconds
walk --block DO01-B00            # one block, every question at once
walk --fences DO01-B00           # how wide the geofences should be
walk --model                     # what the model learned, and from how much
walk --routes                    # where the walking is being wasted
walk --peak 2026                 # what December does to the establishment
```

---

## 1. The mock warehouse

Eight delivery offices, twelve blocks each, two years of walks: **96 blocks,
7,000 segments, 28,000 delivery points, 59,520 walks, 2.3 million segment
traversals, 780,000 pings.** It builds in about 35 seconds and comes to roughly
400 MB.

The tables carry *their* names and *their* granularity, so the adapter to a real
warehouse is a mapping and not a translation:

| Table | What it is |
|---|---|
| `osm_segments` | straight pieces of road between curves and junctions; 10-25 m urban, 50-150 m rural, as on their slide |
| `segment_geofence` | the buffer, with the mean DP distance it was sized from |
| `dp_segment_join` | their existing delivery-point-to-segment join, with distances |
| `pings` / `ping_osm_match` | raw fixes and their existing match table, with the walking flag at 2.32 m/s |
| `segment_traversals` | entry and exit per segment, per device, **per pass** |
| `block_traversal_totals` | the "does take" sum their second slide produces |
| `block_days` | who walked it, in what weather, with how much of each kind of mail |

Three retention windows, because two years of ten-second pings would be a
hundred million rows to make a point a fortnight makes: **block totals for the
full two years**, **per-segment traversals for 180 days**, **raw pings for 5
days**. All three are flags (`--detail-days`, `--ping-days`).

### What is deliberately in the data

Times are never written down. Streets are laid out, doors are placed along them
at an offset from the road, a route is walked, and the times fall out of pace,
distance and what is at each door. So a model fitted to it recovers a pace
because there is one in there to recover. Also in there on purpose:

- **Inefficiency.** The mock rounds are walked greedily, nearest street next -
  what people do, and not optimal. Without it the optimiser would have nothing
  to find and we would be marking our own homework.
- **Loops in the street network.** An early version generated trees, and on a
  tree every route must walk every street exactly twice: all three routing
  options came out identical, to four significant figures, and the "optimiser"
  looked useless because the problem was trivial.
- **The Christmas peak**, as a *parcel* peak - letters barely move, parcels
  triple, and a parcel is eight and a half seconds at a door.
- **Walkers who start slow**, relief cover on somebody else's block, weather,
  darkness, and GPS that is wrong by six to twelve metres.
- **Dead time on no geofence**: the walk from the van, a break, a doorstep down
  a long drive. Without it "does take" would equal the elapsed time and the
  coverage figure would have nothing to measure.

---

## 2. What the audit found in the measurement

Three things, all of which change the answer by more than the thing being
measured, and all of which apply to the real pipeline and not just to the mock.

### Geofences overlap, and the sum double-counts

Urban segments are 10-25 m. A buffer sized from the delivery-point distance
comes out around 10 m. The capsules around consecutive segments therefore
overlap heavily, and **a ping sits inside two to six fences at once.**

Read "entry and exit per segment, sum the traversal times" literally - every
fence the ping is in - and a 4.5 hour walk is reported as **9.5 hours**. Across
blocks the inflation runs **1.9x to 4.6x**.

The fix is in their own data: let the ping-to-OSM match decide which single
segment owns each ping, and use the geofence only to decide whether the ping
counts at all. Then the same walk comes back at 4.43 hours against a true 4.53.

**The question for them: which of those two is in production?** It decides
whether every does-take figure downstream is right or doubled.

### Stitching a pass across a gap can also double-count

Even with one segment per ping, taking a pass as "first to last ping on that
segment within a time window" lets two passes overlap in time when the walker
weaves between adjacent streets. The totals then exceed the length of the walk -
118% of it, on one block.

A pass has to be a run of *consecutive* pings, and the time has to be attributed
by interval: each gap between two pings belongs to the segment that owned the
earlier one. The intervals then tile the walk exactly - nothing lost, nothing
counted twice. (Measuring each run from its own first to its own last ping
instead loses every gap *between* runs, which on an urban block is more than
half the walk: coverage collapsed to 42% when that was tried.)

### The fences are narrower than the GPS error

Sizing the buffer from the delivery-point distance alone ignores the other thing
that decides whether a ping lands inside it: how wrong the handset is. In a
terraced street the doors are five metres off the road, the fence is eight
metres wide, and the device is routinely ten metres out.

At the width their rule produces:

| | fence | does-take | coverage | pings in no fence |
|---|---|---|---|---|
| urban | 9.4 m | 4.13 h | 87% | 18% |
| suburban | 19.7 m | 1.76 h | 92% | 12% |
| rural | 52.8 m | 2.21 h | 97% | 4% |

Widen to about 1.4x and coverage goes to 98% everywhere; wider than that and
nothing more is found, because the fences start swallowing the next street.
`walk --fences BLOCK` plots that curve for any block. The suggested rule:

```
buffer = max(mean DP distance x 1.6, 2 x median device accuracy)
```

**Urban blocks are currently under-measured by about 13%.** A standard time
built on them is thirteen per cent short, and the walkers know it even if the
spreadsheet does not.

---

## 3. Should take

Fitted in the shape of the job rather than as a regression on whatever columns
exist: `length / pace + time at the doors`, with what the day is doing applied
as a multiplier on the walking.

**Out of sample, fitted on year one and tested on the whole of year two:**

| | |
|---|---|
| bias | **-1.0%** |
| mean absolute error | **4.4%** |
| median absolute error | 2.6% |
| 90% interval covers | 94% (conservative) |

Error is flat across urban, suburban and rural (4.1-4.9%).

### What it learned, against what the generator hid

| | fitted | seeded |
|---|---|---|
| pace, urban / suburban / rural | 1.09 / 1.10 / 1.27 m/s | 1.02 / 1.18 / 1.34 |
| faster with nothing to deliver | 1.18x | 1.22x |
| rain / heavy rain | 1.15 / 1.30 | 1.09 / 1.18 |
| relief cover | 1.18x | 1.11x |
| a walker in their first month | 1.25x | 1.26x |
| a parcel | 10.3 s | 8.5 s |
| a signed-for item | 48 s | 42 s |

### Three things it cannot do, said plainly

1. **The walk up the path cannot be separated from the time at the door.** The
   metres of path on a segment and the number of doors on it are the same column
   to within a few per cent. Fitting both gave a terraced front door 3.5 seconds
   and a flat fifty, and was wrong about both. They are now reported together,
   per area, which is identified and interpretable - and a block whose paths are
   unusual for its area will be underestimated.
2. **The split between pace and door time is only as good as the spread of block
   densities.** With urban and rural blocks in the sample it lands within a
   quarter; with only urban blocks it wanders to 0.54 m/s, which is not a pace.
   It is held inside the physically possible band - never faster at work than in
   transit, never below two thirds of it - and the doors are refitted around
   whatever that costs. **The sum is what the forecast uses, and the sum is
   accurate to 4%.**
3. **Items per door are constant within a day**, so "seconds per door" and
   "seconds per item" cannot be told apart from one day's data. They are
   separated across the year, which is the main thing two years of history buys
   that six months does not.

---

## 4. Routing: an arc problem, not a tour

A van visits points; a postie covers streets. The shortest tour of a block's
delivery points is nonsense on foot - it has no idea that one pass down a
terrace serves both sides. The right problem is the **Rural Postman**: cover
every edge that has mail on it, at least total length, and come home.

`graph.rural_postman` is Frederickson's heuristic - join the required streets
into one component, make every junction even by doubling the cheapest
connections, take an Euler circuit. The odd junctions are paired greedily rather
than by minimum-weight matching, so it is a good route and **not a proven best
one, nor always the best of the candidates offered**: on a block whose junctions
pair badly the plain nearest-street route can win, and both are costed so the
shorter one is what gets recommended.

Across the 96 blocks: it helps on 54 of them, and the total is **217 minutes a
day, or 1,122 walker-hours a year.** The worst single block is 12.6 minutes a
day - 65 hours a year on one walk.

---

## 5. Resourcing

The forecast, added up, with two things that are easy to get wrong:

**Uncertainty does not add.** Twenty blocks with half an hour of doubt each do
not make a day with ten hours of doubt. The hours add; the spreads combine in
quadrature plus whatever is common to the whole office that day. An office
planned on summed worst cases is an office with people standing around.

**A walker is a person, not a pool of hours.** Five blocks of six hours is five
people out for six hours, not thirty hours to divide by a shift. And a block at
eight and a half hours does not need a second walker - it needs an hour of
overtime. Only past the maximum shift does it have to be *split*, which is a
different and much more expensive answer.

```
week from     hours/day  overtime/day  to split  extra people
2026-11-02        474.6          0.8h       0.0           0.1
2026-11-30        499.4          4.2h       0.0           0.6
2026-12-14        518.6          7.8h       0.0           1.0
2026-12-28        479.1          1.1h       0.0           0.1
```

The seeded Christmas is deliberately conservative - about +12% on the walk. A
real one is larger; `generate.true_season` is the dial.

---

## 6. What this does not do

- **It does not know the roads.** Distances are straight lines between segment
  ends. Where a real OSM network is available, the same algorithms run on it
  unchanged and the numbers get better.
- **Nothing here is live.** No traffic feed, no incident feed, no weather
  service: the weather in the mock is generated, and in production would be a
  column somebody fills in.
- **The demand side is synthetic.** The fleet, the volumes, the weekday pattern
  and the seasonality are shaped from the recorded history; which doors get what
  next Tuesday is not knowable and is not claimed.
- **It does not write to their tables.** Everything this application owns is
  prefixed `walk_`. If the real tables arrive, the mock ones are dropped and the
  adapter points at the real ones; nothing else should need to notice.

---

## 7. Tests

```bash
python3 -m unittest tests.test_walk -v      # 45 tests, about 15 seconds
python3 -m unittest discover -s tests       # 192, both halves
```

The three that matter: that derived traversals **tile the walk** (never more
than it took, never losing the gaps - both failure modes happened and each
silently moved every number downstream by tens of per cent); that the model
**recovers what the generator hid**; and that a route **covers every street and
comes home**, since a route that misses one is not a faster route.
