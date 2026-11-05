"""
Two years of walked blocks, built so the model has something real to find.

The rule this follows is the same one that made the first database worth
calibrating against: **geometry and behaviour first, times second**. Streets are
laid out, delivery points are placed along them at an offset from the road, a
route is walked, and the times fall out of pace, distance and what is at each
door. Nothing writes down "a block takes 3 hours". That is why a model fitted to
this data recovers a pace - there is a pace in there to recover.

What is deliberately in the data
--------------------------------

* **Inefficiency.** The mock rounds are walked greedily, nearest street next,
  which is what people do and is not optimal. Without it the optimiser would
  have nothing to find and we would be marking our own homework.
* **The Christmas peak**, the January trough, and the summer dip - because
  resourcing that cannot see December is useless.
* **Walkers who start slow.** A new walker is about a quarter slower and takes
  months to come right; relief walkers covering an unfamiliar block are slower
  than the regular.
* **Weather**, which slows a walk far more than it slows a van.
* **Dead time off the geofences** - the walk from the van, a break, a crossing -
  so that "does take" does not equal elapsed time, and the coverage figure has
  something to measure.
* **GPS that is not perfect**: pings every few seconds with noise on them, so
  entry and exit times carry the error they carry in reality.

The constants below are the truth of this world. `walk/model.py` never imports
them: it has to find them in the records, and `tests/test_walk.py` checks that
it does.

    python3 -m walk.generate --years 2
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import random
import time
from pathlib import Path

from .graph import (Edge, Graph, dist_m, habitual_route, offset_point,
                    point_segment_distance_m, to_ll, to_m)
from .passes import passes_from
from .schema import (DEFAULT_WALK_DB, WALKING_SPEED_LIMIT_MPS, connect, counts)

# ------------------------------------------------------------- the world
OFFICES = [
    ("DO01", "Leeds Central DO",      "Yorkshire",     53.7997, -1.5492),
    ("DO02", "Manchester East DO",    "North West",    53.4808, -2.2426),
    ("DO03", "Birmingham South DO",   "West Midlands", 52.4862, -1.8904),
    ("DO04", "Bristol North DO",      "South West",    51.4545, -2.5879),
    ("DO05", "Southwark DO",          "London",        51.4934, -0.0098),
    ("DO06", "Newcastle West DO",     "North East",    54.9783, -1.6178),
    ("DO07", "Norwich Rural DO",      "East",          52.6309,  1.2974),
    ("DO08", "Cardiff Bay DO",        "Wales",         51.4816, -3.1791),
]

STREETS = ["High St", "Church Ln", "Mill Rd", "Station Rd", "Victoria Rd", "Park Ave",
           "Queens Rd", "Albert St", "Kings Way", "Bridge St", "Oak Dr", "Elm Gr",
           "Chapel St", "Market St", "Grange Rd", "Meadow Ln", "Riverside", "Castle St",
           "Woodland Way", "The Crescent", "Orchard Cl", "Sunnybank", "Fairfield Rd"]

# segment length by area, as on their first slide: 10-25 m urban, 50-150 m rural
SEG_LEN = {"urban": (11.0, 25.0), "suburban": (18.0, 40.0), "rural": (55.0, 150.0)}
# how far a delivery point sits off the road - and so how wide their geofence is
DP_OFFSET = {"urban": (2.5, 9.0), "suburban": (6.0, 18.0), "rural": (12.0, 70.0)}
# metres of street per delivery point
DP_SPACING = {"urban": (4.0, 9.0), "suburban": (9.0, 20.0), "rural": (28.0, 75.0)}
BLOCK_SIZE = {"urban": (1100, 2000), "suburban": (1400, 2600), "rural": (2200, 4200)}

DP_MIX = {
    "urban":    [("terrace", .46), ("flat", .30), ("house", .10), ("business", .12),
                 ("locker", .02)],
    "suburban": [("house", .62), ("terrace", .18), ("flat", .12), ("business", .07),
                 ("locker", .01)],
    "rural":    [("house", .48), ("farm", .34), ("business", .12), ("terrace", .05),
                 ("locker", .01)],
}
ACCESS_FOR = {
    "terrace": [("front door", .86), ("gate", .14)],
    "house":   [("front door", .62), ("gate", .33), ("long drive", .05)],
    "flat":    [("buzzer", .55), ("stairs", .40), ("reception", .05)],
    "business": [("reception", .55), ("front door", .45)],
    "farm":    [("long drive", .70), ("gate", .30)],
    "locker":  [("front door", 1.0)],
}

# ----------------------------------------------------- the truth to recover
# Walking pace between doors, metres per second. A postie on a round does not
# walk at a commuter's 1.4 m/s: they are carrying, and they stop constantly.
TRUE_PACE = {"urban": 1.02, "suburban": 1.18, "rural": 1.34}
# ...and faster on a street with nothing to deliver on it
TRUE_DEADHEAD_BONUS = 1.22

# How much of the walk up the path actually gets walked. Down a farm drive, all
# of it, twice. Along a terrace, almost none: the walker moves along the row and
# never returns to the middle of the road between two front doors. Charging the
# full there-and-back for every urban door added an hour to a three-hour walk,
# which is the kind of error that looks like a model and is really a wrong
# picture of the job.
TRUE_OFFSET_SHARE = {"urban": 0.40, "suburban": 0.65, "rural": 1.0}

# Seconds at the door, before the items in the hand
TRUE_SERVICE_S = {"terrace": 7.0, "house": 9.5, "flat": 23.0, "business": 17.0,
                  "farm": 14.0, "locker": 38.0}
TRUE_ACCESS_S = {"front door": 0.0, "gate": 4.5, "stairs": 11.0, "buzzer": 13.0,
                 "reception": 16.0, "long drive": 0.0}     # the drive is in the offset
# Seconds per item of each kind, on top
TRUE_ITEM_S = {"letters": 0.45, "packets": 3.2, "parcels": 8.5, "signed": 42.0,
               "special": 55.0}

TRUE_WEATHER = {"clear": 1.00, "overcast": 1.01, "rain": 1.09, "heavy rain": 1.18,
                "wind": 1.05, "snow": 1.31, "ice": 1.40}
TRUE_DARK = 1.07                    # before sunrise or after sunset
TRUE_NEW_WALKER = 1.26              # on day one
TRUE_EXPERIENCE_HALFLIFE_D = 70.0   # how fast that wears off
TRUE_RELIEF = 1.11                  # covering someone else's block
TRUE_LOAD_PER_KG = 0.0022           # fractional slowdown per kg carried

# Volume through the year. The December peak is the whole reason resourcing
# exists, and it is a *parcel* peak: letters barely move, parcels triple, and
# every extra parcel is eight and a half seconds at a door. A season multiplier
# applied evenly to all mail - which is what the first version of this did -
# puts six per cent on a Christmas walk instead of fifteen, and makes the peak
# look like something you can absorb.
def true_season(d: dt.date) -> float:
    doy = d.timetuple().tm_yday
    delta = abs(doy - 349)
    delta = min(delta, 365 - delta)                 # wrap around the new year
    peak = 1.0 + 1.15 * math.exp(-(delta / 16.0) ** 2)
    if doy > 358 or doy < 6:
        peak = 1.12                                 # the lull between the holidays
    jan = -0.18 * math.exp(-((doy - 18) / 14.0) ** 2)
    summer = -0.10 * math.exp(-((doy - 205) / 32.0) ** 2)
    return max(0.5, peak + jan + summer)


TRUE_WEEKDAY_VOLUME = {0: 1.14, 1: 1.02, 2: 0.98, 3: 1.00, 4: 1.06, 5: 0.80}


def _pick(rng, weighted):
    return rng.choices([a for a, _ in weighted], weights=[w for _, w in weighted])[0]


# ------------------------------------------------------------ the network
def build_network(rng, lat0: float, lon0: float, area: str,
                  block_id: str) -> tuple[Graph, list[dict], list[dict]]:
    """Lay out the streets of one block, then put the doors along them."""
    lo, hi = SEG_LEN[area]
    target = rng.uniform(*BLOCK_SIZE[area])

    g = Graph()
    nodes_xy: dict[str, tuple[float, float]] = {}

    def node_at(x: float, y: float) -> str:
        """Nodes within a few metres are the same junction."""
        for nid, (nx, ny) in nodes_xy.items():
            if (nx - x) ** 2 + (ny - y) ** 2 < 36:
                return nid
        nid = f"{block_id}-N{len(nodes_xy):04d}"
        nodes_xy[nid] = (x, y)
        lat, lon = to_ll(x, y, lat0, lon0)
        g.add_node(nid, lat, lon)
        return nid

    segments: list[dict] = []
    way_id = int(block_id[-4:], 36) * 100 if block_id[-4:].isalnum() else 1000

    def lay_street(x: float, y: float, bearing: float, length: float,
                   name: str, way: int) -> list[str]:
        """A street as a chain of short straight segments, bending gently."""
        placed, cur = [], node_at(x, y)
        done, seq = 0.0, 0
        while done < length:
            seg_len = min(rng.uniform(lo, hi), length - done)
            if seg_len < lo * 0.4:
                break
            bearing += rng.gauss(0, 0.09)          # a street is never quite straight
            x += seg_len * math.cos(bearing)
            y += seg_len * math.sin(bearing)
            nxt = node_at(x, y)
            if nxt == cur:
                continue
            key = f"{block_id}-S{len(segments):04d}"
            a_lat, a_lon = g.nodes[cur]
            b_lat, b_lon = g.nodes[nxt]
            real = dist_m(a_lat, a_lon, b_lat, b_lon, lat0, lon0)
            g.add_edge(Edge(key, cur, nxt, real, required=True))
            segments.append({
                "segment_id": key, "block_id": block_id, "osm_way_id": way,
                "way_seq": seq, "street_name": name, "from_node": cur, "to_node": nxt,
                "from_lat": a_lat, "from_lon": a_lon, "to_lat": b_lat, "to_lon": b_lon,
                "length_m": real, "area_type": area,
            })
            placed.append(key)
            cur, done, seq = nxt, done + seg_len, seq + 1
        return placed

    # a spine, then streets off it, some of which loop back
    names = rng.sample(STREETS, k=min(len(STREETS), 9))
    bearing = rng.uniform(0, 2 * math.pi)
    laid = lay_street(0.0, 0.0, bearing, min(target * 0.34, 620), names[0], way_id)
    total = sum(g.edges[k].length_m for k in laid)
    i = 1
    while total < target and i < len(names):
        anchor = rng.choice(laid)
        e = g.edges[anchor]
        base = e.a if rng.random() < 0.5 else e.b
        blat, blon = g.nodes[base]
        bx, by = to_m(blat, blon, lat0, lon0)
        side = bearing + (math.pi / 2 if rng.random() < 0.5 else -math.pi / 2)
        side += rng.gauss(0, 0.25)
        length = min(rng.uniform(140, 520), target - total)
        new = lay_street(bx, by, side, length, names[i], way_id + i)
        if new:
            laid += new
            total += sum(g.edges[k].length_m for k in new)
        i += 1

    # Close some loops. Without this the network is a tree, and on a tree every
    # route must walk every street exactly twice - there is no better order, and
    # an optimiser has nothing to find. Real streets are not trees: you can walk
    # round a block of houses. So where two ends finish up near each other
    # without being joined, join them, which is what the back lane or the
    # footpath between the closes does in the real world.
    ends = [n for n in g.nodes if g.degree(n) <= 2]
    joined = 0
    for i, a in enumerate(ends):
        if joined >= max(2, len(segments) // 12):
            break
        alat, alon = g.nodes[a]
        for b in ends[i + 1:]:
            if any(k for _o, k in g.neighbours(a) if _o == b):
                continue
            blat, blon = g.nodes[b]
            d = dist_m(alat, alon, blat, blon, lat0, lon0)
            if 25 < d < 110 and rng.random() < 0.55:
                key = f"{block_id}-S{len(segments):04d}"
                g.add_edge(Edge(key, a, b, d, required=True))
                segments.append({
                    "segment_id": key, "block_id": block_id, "osm_way_id": way_id + 90,
                    "way_seq": joined, "street_name": "link", "from_node": a,
                    "to_node": b, "from_lat": alat, "from_lon": alon,
                    "to_lat": blat, "to_lon": blon, "length_m": d, "area_type": area,
                })
                joined += 1
                break

    # the van parks at the first node
    start_node = segments[0]["from_node"] if segments else next(iter(g.nodes))

    # a few connecting streets with nothing on them: a snicket, a service road.
    # These are what make it a *rural* postman problem rather than a Chinese one.
    for s in segments:
        if rng.random() < 0.035:
            g.edges[s["segment_id"]].required = False
            s["required"] = 0
        else:
            s["required"] = 1

    # --------------------------------------------------------- the doors
    dps: list[dict] = []
    joins: list[dict] = []
    for s in segments:
        if not s["required"]:
            continue
        spacing = rng.uniform(*DP_SPACING[area])
        n = max(0, int(s["length_m"] / spacing))
        if area != "rural":
            n = int(n * rng.choice([1, 2, 2]))       # both sides of the street, usually
        for j in range(n):
            t = (j + 0.5) / max(n, 1)
            side = 1 if (j % 2 == 0 or area == "rural") else -1
            off = rng.uniform(*DP_OFFSET[area])
            lat, lon = offset_point(s["from_lat"], s["from_lon"],
                                    s["to_lat"], s["to_lon"], t, off * side)
            dp_type = _pick(rng, DP_MIX[area])
            households = (rng.randint(4, 24) if dp_type == "flat"
                          else rng.randint(1, 2) if dp_type == "terrace" else 1)
            dp_id = f"{s['segment_id']}-D{j:02d}"
            dps.append({
                "dp_id": dp_id, "block_id": block_id, "lat": lat, "lon": lon,
                "dp_type": dp_type, "access_type": _pick(rng, ACCESS_FOR[dp_type]),
                "households": households,
                "address": f"{rng.randint(1, 180)} {s['street_name']}",
            })
            joins.append({"dp_id": dp_id, "segment_id": s["segment_id"],
                          "distance_m": round(off, 2), "is_primary": 1,
                          "side": "left" if side > 0 else "right"})
    # A street with no doors on it has nothing to deliver, whatever it was
    # marked as when it was laid out. Saying so here keeps the stored data and
    # the routing graph talking about the same set of streets - without it the
    # optimiser is asked to cover a different block from the one that was walked,
    # and the comparison between them is meaningless.
    have = {j["segment_id"] for j in joins}
    for s in segments:
        s["required"] = 1 if s["segment_id"] in have else 0
        g.edges[s["segment_id"]].required = bool(s["required"])

    return g, segments, dps, joins, start_node


def geofence_for(segment: dict, joined: list[dict], area: str) -> dict:
    """Their step 3: buffer width from the average DP distance to the segment.

    Floors and ceilings are mine, not theirs, and they matter: a segment with no
    doors on it would otherwise get a zero-width fence that no ping can ever be
    inside, and one rural farm track would get a fence wide enough to swallow
    the next village.
    """
    if joined:
        mean_d = sum(j["distance_m"] for j in joined) / len(joined)
    else:
        mean_d = {"urban": 5.0, "suburban": 10.0, "rural": 25.0}[area]
    buffer_m = min(max(mean_d * 1.6, 8.0), 60.0)
    return {"segment_id": segment["segment_id"], "buffer_m": round(buffer_m, 2),
            "mean_dp_distance_m": round(mean_d, 2), "dp_count": len(joined),
            "method": "mean dp distance"}


# --------------------------------------------------------------- the walks
def _service_seconds(dp: dict, items: dict, rng) -> float:
    """How long this door takes today, before the walk up the path."""
    base = TRUE_SERVICE_S[dp["dp_type"]] + TRUE_ACCESS_S[dp["access_type"]]
    base *= 1.0 + 0.04 * (dp["households"] - 1) if dp["dp_type"] == "flat" else 1.0
    s = base
    for kind, n in items.items():
        s += TRUE_ITEM_S[kind] * n
    return s * rng.gauss(1.0, 0.10)


def _weather_for(rng, d: dt.date, region: str) -> tuple[str, float, float, float]:
    m = d.month
    wet = 0.46 if m in (10, 11, 12, 1, 2, 3) else 0.28
    if region in ("Wales", "North West"):
        wet += 0.10
    r = rng.random()
    if m in (12, 1, 2) and r < 0.045:
        w = rng.choice(["snow", "ice"])
    elif r < wet * 0.45:
        w = "rain"
    elif r < wet * 0.58:
        w = "heavy rain"
    elif r < wet * 0.78:
        w = "wind"
    elif r < 0.75:
        w = "overcast"
    else:
        w = "clear"
    temp = 11 + 9 * math.sin((d.timetuple().tm_yday - 110) / 365.25 * 2 * math.pi)
    return (w, round(temp + rng.gauss(0, 2.4), 1),
            round(max(0.0, rng.gauss(4, 4)) if "rain" in w else 0.0, 1),
            round(abs(rng.gauss(9 if w == "wind" else 3.5, 2)), 1))


def _items_per_dp(rng, d: dt.date) -> dict[str, float]:
    """How much mail each door gets today, by kind.

    Each kind moves differently through the year. Letters are nearly flat - the
    bills come whatever the month. Parcels are what Christmas is made of, and
    they are also what costs time at the door, so the two facts have to be kept
    apart or the peak disappears into an average.
    """
    season = true_season(d)
    weekday = TRUE_WEEKDAY_VOLUME.get(d.weekday(), 1.0)
    j = lambda s: max(0.0, rng.gauss(1.0, s))
    return {
        "letters": 1.55 * (0.85 + 0.15 * season) * weekday * j(0.13),
        "packets": 0.21 * season ** 1.15 * weekday * j(0.20),
        "parcels": 0.115 * season ** 1.60 * weekday * j(0.24),
        "signed": 0.019 * season ** 1.25 * weekday * j(0.30),
        "special": 0.0035 * season * weekday * j(0.40),
    }


def _experience_factor(walker: dict, d: dt.date) -> float:
    """A new walker is slower, and stops being slower over a few months."""
    started = dt.date.fromisoformat(walker["started_on"])
    days = max(0, (d - started).days)
    return 1.0 + (TRUE_NEW_WALKER - 1.0) * math.exp(-days / TRUE_EXPERIENCE_HALFLIFE_D)


def simulate_walk(block, statics, route, walker, is_relief, weather, dark,
                  items, rng, start_s: float):
    """Walk the block once and return every segment traversal it produced.

    Times come out of pace, distance and what is at each door - never out of a
    table of durations. The walker is slowed by the weather, by being new, by
    being on somebody else's block, and by the weight of what they are carrying.
    """
    area = block["area_type"]
    slow = (TRUE_WEATHER[weather] * _experience_factor(walker, items["date"])
            * (TRUE_RELIEF if is_relief else 1.0) * (TRUE_DARK if dark else 1.0))
    per_dp = items["per_dp"]
    kg = 7.0 + 2.4 * per_dp["parcels"] * 10 + 0.4 * per_dp["packets"] * 10
    slow *= 1.0 + TRUE_LOAD_PER_KG * kg
    pace = TRUE_PACE[area] / slow                     # metres per second

    rows, t = [], start_s
    off_fence = 0.0
    for step in route:
        s = statics[step.key]
        if step.serviced:
            walk_s = s["length_m"] / pace
            offset_s = s["offset_m"] * TRUE_OFFSET_SHARE[area] / pace
            svc = s["svc_base_s"]
            for kind, rate in per_dp.items():
                svc += s["dp_count"] * rate * TRUE_ITEM_S[kind]
            dur = (walk_s + offset_s + svc) * rng.gauss(1.0, 0.055)
        else:
            dur = s["length_m"] / (pace * TRUE_DEADHEAD_BONUS) * rng.gauss(1.0, 0.07)
        dur = max(dur, 1.0)
        rows.append((step.key, step.pass_no, t, t + dur, dur, 1 if step.serviced else 0))
        t += dur
        # the bits that are on nobody's geofence: a crossing, a look at the frame
        gap = max(0.0, rng.gauss(2.0, 3.0))
        t += gap
        off_fence += gap
    if rng.random() < 0.22:                           # a break, most days not taken
        br = rng.uniform(240, 900)
        t += br
        off_fence += br
    return rows, t - start_s, off_fence


def build_world(con, rng, blocks_per_office: int, start_date: dt.date):
    """Offices, blocks, streets, doors, geofences, people. The standing data."""
    con.executemany("INSERT OR REPLACE INTO delivery_offices VALUES (?,?,?,?,?,?)",
                    [(o[0], o[1], o[2], "United Kingdom", o[3], o[4]) for o in OFFICES])
    world = {}
    seg_rows, fence_rows, dp_rows, join_rows, block_rows = [], [], [], [], []
    walkers, devices, vehicles = [], [], []

    for do_id, _name, region, dlat, dlon in [(o[0], o[1], o[2], o[3], o[4])
                                             for o in OFFICES]:
        for b in range(blocks_per_office):
            # One of each, in turn. The mix matters more than it looks: the
            # model separates "seconds per metre" from "seconds per door" using
            # the fact that a rural block has seventy metres of street per door
            # and an urban one has five. An office with only urban blocks gives
            # it nothing to work with, and the fitted pace comes out half what
            # anybody walks at.
            area = ("urban", "suburban", "rural")[b % 3]
            block_id = f"{do_id}-B{b:02d}"
            brng = random.Random(f"{block_id}|network")
            # blocks sit a few km out from the office
            ang = brng.uniform(0, 2 * math.pi)
            km = brng.uniform(0.6, 3.0) if area == "urban" else \
                brng.uniform(2.0, 9.0) if area == "suburban" else brng.uniform(6.0, 22.0)
            lat0 = dlat + km / 111.0 * math.cos(ang)
            lon0 = dlon + km / (111.0 * math.cos(math.radians(dlat))) * math.sin(ang)

            g, segs, dps, joins, start_node = build_network(brng, lat0, lon0, area,
                                                            block_id)
            if not segs:
                continue
            by_seg_joins: dict[str, list[dict]] = {}
            for j in joins:
                by_seg_joins.setdefault(j["segment_id"], []).append(j)
            dp_by_id = {d["dp_id"]: d for d in dps}

            statics = {}
            for s in segs:
                js = by_seg_joins.get(s["segment_id"], [])
                fence_rows.append(tuple(geofence_for(s, js, area).values()))
                svc_base = 0.0
                for j in js:
                    dp = dp_by_id[j["dp_id"]]
                    base = TRUE_SERVICE_S[dp["dp_type"]] + TRUE_ACCESS_S[dp["access_type"]]
                    if dp["dp_type"] == "flat":
                        base *= 1.0 + 0.04 * (dp["households"] - 1)
                    svc_base += base
                statics[s["segment_id"]] = {
                    "length_m": s["length_m"], "dp_count": len(js),
                    "svc_base_s": svc_base,
                    "offset_m": sum(2 * j["distance_m"] for j in js),
                    "buffer_m": fence_rows[-1][1],
                }
            route = habitual_route(g, start_node, brng)
            drive = (brng.uniform(4, 12) if area == "urban"
                     else brng.uniform(8, 20) if area == "suburban"
                     else brng.uniform(15, 40))
            planned = f"{7 + (b % 3):02d}:{brng.choice(['00', '15', '30', '45'])}"
            # the frame takes longer the more doors are on it
            dp_n = len(dps)
            prep = 26 + dp_n * 0.085 + brng.uniform(-4, 8)
            block_rows.append((block_id, do_id, f"{do_id} block {b:02d}", area,
                               "park and loop" if area != "urban" else "walk",
                               planned, round(drive, 1), round(prep, 1)))
            seg_rows += [(s["segment_id"], s["block_id"], s["osm_way_id"], s["way_seq"],
                          s["street_name"], s["from_node"], s["to_node"], s["from_lat"],
                          s["from_lon"], s["to_lat"], s["to_lon"], s["length_m"],
                          s["area_type"]) for s in segs]
            dp_rows += [(d["dp_id"], d["block_id"], d["lat"], d["lon"], d["dp_type"],
                         d["access_type"], d["households"], d["address"]) for d in dps]
            join_rows += [(j["dp_id"], j["segment_id"], j["distance_m"], j["is_primary"],
                           j["side"]) for j in joins]
            world[block_id] = {"graph": g, "segments": segs, "statics": statics,
                               "route": route, "area_type": area, "do_id": do_id,
                               "region": region, "start_node": start_node,
                               "planned_start": planned, "dps": dps}

        # the people: a regular walker per block plus relief cover, with turnover
        # a walker for each block plus two for cover: tight, as these offices are
        n_walk = blocks_per_office + 2
        for w in range(n_walk):
            wid = f"{do_id}-W{w:02d}"
            # some started years ago, some joined during the two years we hold
            if rng.random() < 0.72:
                started = start_date - dt.timedelta(days=rng.randint(400, 4000))
            else:
                started = start_date + dt.timedelta(days=rng.randint(10, 600))
            walkers.append((wid, do_id, f"Walker {wid[-2:]} of {do_id}",
                            started.isoformat(), None))
            devices.append((f"{wid}-PDA", wid, rng.choice(["PDA-4", "PDA-5", "PDA-5s"]),
                            rng.choice([5, 10, 10, 15])))
        for v in range(max(2, blocks_per_office // 2)):
            vehicles.append((f"{do_id}-V{v:02d}", do_id,
                             rng.choices(["van", "trolley", "bike"],
                                         weights=[.76, .18, .06])[0]))

    con.executemany("INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?)", block_rows)
    con.executemany("INSERT OR REPLACE INTO osm_segments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    seg_rows)
    con.executemany("INSERT OR REPLACE INTO segment_geofence VALUES (?,?,?,?,?)",
                    fence_rows)
    con.executemany("INSERT OR REPLACE INTO delivery_points VALUES (?,?,?,?,?,?,?,?)",
                    dp_rows)
    con.executemany("INSERT OR REPLACE INTO dp_segment_join VALUES (?,?,?,?,?)", join_rows)
    con.executemany("INSERT OR REPLACE INTO walkers VALUES (?,?,?,?,?)", walkers)
    con.executemany("INSERT OR REPLACE INTO devices VALUES (?,?,?,?)", devices)
    con.executemany("INSERT OR REPLACE INTO vehicles VALUES (?,?,?)", vehicles)
    con.commit()
    return world


# ------------------------------------------------------------------ pings
def _segment_grid(segments: list[dict], cell_m: float = 40.0):
    """A coarse spatial index, so matching a ping does not scan every street."""
    if not segments:
        return {}, 0.0, 0.0, cell_m
    lat0 = sum(s["from_lat"] for s in segments) / len(segments)
    lon0 = sum(s["from_lon"] for s in segments) / len(segments)
    grid: dict[tuple[int, int], set[str]] = {}
    for s in segments:
        ax, ay = to_m(s["from_lat"], s["from_lon"], lat0, lon0)
        bx, by = to_m(s["to_lat"], s["to_lon"], lat0, lon0)
        n = max(2, int(s["length_m"] / (cell_m / 2)) + 1)
        for i in range(n + 1):
            t = i / n
            x, y = ax + t * (bx - ax), ay + t * (by - ay)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    grid.setdefault((int(x // cell_m) + dx, int(y // cell_m) + dy),
                                    set()).add(s["segment_id"])
    return grid, lat0, lon0, cell_m


def _near(grid, lat0, lon0, cell_m, lat, lon):
    x, y = to_m(lat, lon, lat0, lon0)
    return grid.get((int(x // cell_m), int(y // cell_m)), ())


def walk_track(block_statics, rows, rng):
    """Where the device was, second by second, as the walk happened.

    A walker on a serviced street is not moving smoothly along the road: they
    walk a few metres, turn up a path, stand at a door for half a minute, come
    back. The dwell is modelled *at the door*, offset from the road, because
    that is what decides whether the ping lands inside the geofence - and on a
    rural block with long drives, a lot of it does not.
    """
    track = []                                     # (ts, lat, lon)
    for key, pass_no, t0, t1, dur, serviced in rows:
        s = block_statics[key]
        a = (s["from_lat"], s["from_lon"])
        b = (s["to_lat"], s["to_lon"])
        if not serviced or not s["dps"]:
            n = max(2, int(dur / 2))
            for i in range(n):
                f = i / n
                track.append((t0 + f * dur,
                              a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
            continue
        # share the time: walking along the road, and standing at each door
        dwell_total = max(0.0, dur * 0.72)
        walk_total = dur - dwell_total
        per_dp = dwell_total / max(len(s["dps"]), 1)
        t = t0
        last_t_frac = 0.0
        for (frac, dlat, dlon, _base) in s["dps"]:
            leg = walk_total * max(frac - last_t_frac, 0.0)
            n = max(1, int(leg / 2))
            for i in range(n):
                f = last_t_frac + (frac - last_t_frac) * (i / n)
                track.append((t + leg * (i / n),
                              a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
            t += leg
            n = max(1, int(per_dp / 2))
            for i in range(n):
                track.append((t + per_dp * (i / n), dlat, dlon))   # at the door
            t += per_dp
            last_t_frac = frac
        leg = max(0.0, t1 - t)
        n = max(1, int(leg / 2))
        for i in range(n):
            f = last_t_frac + (1 - last_t_frac) * (i / n)
            track.append((t + leg * (i / n),
                          a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
    return track


def emit_pings(track, interval_s: int, accuracy_m: float, rng, start_ping_id: int,
               device_id: str, block_day_id: int):
    """Sample the track at the device's interval, with GPS error on it."""
    if not track:
        return [], start_ping_id
    out = []
    pid = start_ping_id
    t_next = track[0][0]
    prev = None
    for i, (ts, lat, lon) in enumerate(track):
        if ts + 1e-9 < t_next:
            continue
        t_next = ts + interval_s
        err = accuracy_m / 111320.0
        plat = lat + rng.gauss(0, err * 0.9)
        plon = lon + rng.gauss(0, err * 1.4)
        if prev is None:
            speed = 0.0
        else:
            dt_s = max(ts - prev[0], 0.5)
            speed = dist_m(prev[1], prev[2], plat, plon) / dt_s
        speed = min(speed, 3.2)
        out.append((pid, device_id, block_day_id, round(ts, 1), plat, plon,
                    round(speed, 3), accuracy_m))
        prev = (ts, plat, plon)
        pid += 1
    return out, pid


def derive_traversals(pings, segments, statics, grid_bits, gap_s: float = 45.0,
                      method: str = "nearest match"):
    """Their slide 2, done on the pings we just emitted - and a warning about it.

    Two readings of "use the segment geofences to find entry and exit times for
    each segment" are possible, and they do not give the same answer:

    * **nearest match** (what this does): each ping belongs to one segment - the
      one their existing ping-to-OSM table matched it to - and the geofence
      decides whether the ping counts at all. Every second of the walk is
      charged once.

    * **geofence any** (the naive reading): a ping counts for every segment whose
      buffer contains it. On a street of 17 m segments with 10 m buffers the
      capsules overlap heavily, so a ping is routinely inside three or four
      fences at once, and the per-segment times sum to **two or three times the
      length of the walk**. On this data the naive reading turns a 4.5 hour walk
      into 9.5 hours.

    The second reading is not a straw man: it is what "buffer each segment, find
    entry and exit per segment, sum the traversal times" says if read literally,
    and nothing on the slide rules it out. `walk/truth.py` measures the gap
    between the two on real pings and reports it, because which one their
    production job implements decides whether every number downstream is right
    or doubled.

    What the honest reading still loses, and should: a walker standing at a door
    down a 40 m farm drive is outside every fence, so that service time is not
    counted - and the entry and exit instants are only as sharp as the ping
    interval.
    """
    grid, lat0, lon0, cell = grid_bits
    matches = []
    timeline = []
    by_fence: dict[str, list[float]] = {}
    for (pid, _dev, _bd, ts, lat, lon, speed, _acc) in pings:
        best, best_d = None, 1e9
        inside = []
        for sid in _near(grid, lat0, lon0, cell, lat, lon):
            s = statics[sid]
            d, _t = point_segment_distance_m(lat, lon, s["from_lat"], s["from_lon"],
                                             s["to_lat"], s["to_lon"])
            if d < best_d:
                best, best_d = sid, d
            if d <= s["buffer_m"]:
                inside.append(sid)
        if best is None:
            continue
        matches.append((pid, best, round(best_d, 2),
                        1 if speed < WALKING_SPEED_LIMIT_MPS else 0))
        for sid in inside:
            by_fence.setdefault(sid, []).append(ts)
        timeline.append((ts, best if best in inside else None))

    ps = passes_from(timeline)
    out = [(p.segment_id, p.pass_no, p.entry_ts, p.exit_ts, p.seconds, p.ping_count)
           for p in ps]
    return out, matches


HOLIDAYS = {(12, 25), (12, 26), (1, 1)}


def _daylight(d: dt.date, hour: float) -> bool:
    """Roughly: is it light? A winter round that starts at seven is not."""
    doy = d.timetuple().tm_yday
    half = 4.0 + 2.5 * math.sin((doy - 80) / 365.25 * 2 * math.pi)   # hours from noon
    return (12 - half) <= hour <= (12 + half)


def generate(db_path: str | Path = None, years: float = 2.0,
             blocks_per_office: int = 12, detail_days: int = 180,
             ping_days: int = 5, seed_value: int = 20,
             end_date: str | None = None) -> dict:
    """Build the whole mock warehouse: standing data, then every walk."""
    t_start = time.time()
    rng = random.Random(seed_value)
    con = connect(db_path)
    con.commit()                       # PRAGMAs cannot run inside a transaction
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    for t in ("ping_osm_match", "pings", "segment_traversals", "block_traversal_totals",
              "block_days", "dp_segment_join", "delivery_points", "segment_geofence",
              "osm_segments", "blocks", "devices", "walkers", "vehicles",
              "delivery_offices", "walk_routes", "walk_forecasts"):
        con.execute(f"DELETE FROM {t}")
    con.commit()

    end = (dt.date.fromisoformat(end_date) if end_date
           else dt.date.today() - dt.timedelta(days=1))
    start = end - dt.timedelta(days=int(round(365.25 * years)))
    world = build_world(con, rng, blocks_per_office, start)

    # per-block aggregates, so a day outside the detail window costs one
    # calculation instead of a hundred
    for bid, w in world.items():
        st = w["statics"]
        served = {s.key for s in w["route"] if s.serviced}
        w["agg"] = {
            "serviced_len": sum(st[k]["length_m"] for k in served),
            "offset_m": sum(st[k]["offset_m"] for k in served),
            "svc_base": sum(st[k]["svc_base_s"] for k in served),
            "dp_count": sum(st[k]["dp_count"] for k in served),
            "dead_len": sum(st[s.key]["length_m"] for s in w["route"] if not s.serviced),
            "n_steps": len(w["route"]),
            "n_served": len(served),
        }
        # how much per-segment noise survives into the block total
        shares = []
        for s in w["route"]:
            d = st[s.key]["length_m"] + (st[s.key]["svc_base_s"] if s.serviced else 0)
            shares.append(d)
        tot = sum(shares) or 1.0
        w["agg"]["noise_share"] = math.sqrt(sum((x / tot) ** 2 for x in shares))
        # geometry each segment needs for pings and matching
        seg_by_id = {s["segment_id"]: s for s in w["segments"]}
        for k, s in st.items():
            g = seg_by_id[k]
            s.update({"from_lat": g["from_lat"], "from_lon": g["from_lon"],
                      "to_lat": g["to_lat"], "to_lon": g["to_lon"], "dps": []})
        for dp in w["dps"]:
            row = con.execute("SELECT segment_id, distance_m FROM dp_segment_join "
                              "WHERE dp_id=?", (dp["dp_id"],)).fetchone()
            if not row:
                continue
            s = st[row["segment_id"]]
            _d, t = point_segment_distance_m(dp["lat"], dp["lon"], s["from_lat"],
                                             s["from_lon"], s["to_lat"], s["to_lon"])
            s["dps"].append((t, dp["lat"], dp["lon"],
                             TRUE_SERVICE_S[dp["dp_type"]]))
        for s in st.values():
            s["dps"].sort()

    walkers_by_do: dict[str, list[dict]] = {}
    for r in con.execute("SELECT * FROM walkers"):
        walkers_by_do.setdefault(r["do_id"], []).append(dict(r))
    device_of = {r["walker_id"]: r["device_id"] for r in con.execute("SELECT * FROM devices")}
    interval_of = {r["device_id"]: r["ping_interval_s"]
                   for r in con.execute("SELECT * FROM devices")}
    vehicles_by_do: dict[str, list[str]] = {}
    for r in con.execute("SELECT * FROM vehicles"):
        vehicles_by_do.setdefault(r["do_id"], []).append(r["vehicle_id"])

    detail_from = end - dt.timedelta(days=detail_days)
    ping_from = end - dt.timedelta(days=ping_days)

    bd_rows, trav_rows, total_rows, ping_rows, match_rows = [], [], [], [], []
    ping_id = 1
    block_day_id = 0
    grids = {}
    n_days = 0
    day = start
    while day <= end:
        if day.weekday() == 6 or (day.month, day.day) in HOLIDAYS:
            day += dt.timedelta(days=1)
            continue
        n_days += 1
        detail = day >= detail_from
        do_weather = {}
        for bid, w in world.items():
            do_id = w["do_id"]
            if do_id not in do_weather:
                do_weather[do_id] = _weather_for(rng, day, w["region"])
            weather, temp, rain, wind = do_weather[do_id]

            pool = [x for x in walkers_by_do[do_id]
                    if dt.date.fromisoformat(x["started_on"]) <= day]
            if not pool:
                continue
            idx = (int(bid[-2:]) + (day.year * 2 + day.month // 6)) % len(pool)
            regular = pool[idx]
            is_relief = rng.random() < 0.16
            walker = rng.choice(pool) if is_relief else regular
            is_relief = walker["walker_id"] != regular["walker_id"]
            device = device_of.get(walker["walker_id"])

            per_dp = _items_per_dp(rng, day)
            agg = w["agg"]
            dps_served = agg["dp_count"]
            items = {k: int(round(v * dps_served)) for k, v in per_dp.items()}
            hour = int(w["planned_start"][:2]) + rng.gauss(0, 0.25)
            light = _daylight(day, hour)
            start_s = hour * 3600
            block_day_id += 1

            if detail:
                rows, elapsed, off_fence = simulate_walk(
                    {"area_type": w["area_type"]}, w["statics"], w["route"], walker,
                    is_relief, weather, not light,
                    {"per_dp": per_dp, "date": day}, rng, start_s)
                does_take = sum(r[4] for r in rows)
            else:
                slow = (TRUE_WEATHER[weather] * _experience_factor(walker, day)
                        * (TRUE_RELIEF if is_relief else 1.0)
                        * (1.0 if light else TRUE_DARK))
                kg = 7.0 + 24 * per_dp["parcels"] + 4 * per_dp["packets"]
                slow *= 1.0 + TRUE_LOAD_PER_KG * kg
                pace = TRUE_PACE[w["area_type"]] / slow
                svc = agg["svc_base"] + agg["dp_count"] * sum(
                    per_dp[k] * TRUE_ITEM_S[k] for k in per_dp)
                does_take = ((agg["serviced_len"]
                              + agg["offset_m"] * TRUE_OFFSET_SHARE[w["area_type"]])
                             / pace + svc
                             + agg["dead_len"] / (pace * TRUE_DEADHEAD_BONUS))
                does_take *= rng.gauss(1.0, 0.055 * agg["noise_share"] * math.sqrt(
                    max(agg["n_steps"], 1)))
                rows = []
                off_fence = agg["n_steps"] * 2.0
                elapsed = does_take + off_fence

            bd_rows.append((block_day_id, bid, day.isoformat(), walker["walker_id"],
                            device, rng.choice(vehicles_by_do[do_id]), weather, temp,
                            rain, wind, 1 if light else 0, items["letters"],
                            items["packets"], items["parcels"], items["signed"],
                            items["special"], dps_served,
                            f"{int(start_s // 3600):02d}:{int(start_s % 3600 // 60):02d}"))

            if detail and day >= ping_from and device:
                if bid not in grids:
                    grids[bid] = _segment_grid(w["segments"])
                track = walk_track(w["statics"], rows, rng)
                emitted, ping_id = emit_pings(track, interval_of.get(device, 10),
                                              rng.choice([6.0, 8.0, 12.0]), rng,
                                              ping_id, device, block_day_id)
                derived, matches = derive_traversals(emitted, w["segments"],
                                                     w["statics"], grids[bid])
                ping_rows += emitted
                match_rows += matches
                for (sid, pass_no, a, b2, dur, n) in derived:
                    trav_rows.append((block_day_id, sid, device, pass_no, a, b2, dur,
                                      n, 1, "ping"))
                total_rows.append((block_day_id, device, sum(d[4] for d in derived),
                                   len({d[0] for d in derived}), len(w["segments"]),
                                   len(derived),
                                   min((d[2] for d in derived), default=None),
                                   max((d[3] for d in derived), default=None), "ping"))
            else:
                if detail:
                    for (key, pass_no, t0, t1, dur, serviced) in rows:
                        trav_rows.append((block_day_id, key, device, pass_no, t0, t1,
                                          dur, 0, serviced, "simulated"))
                total_rows.append((block_day_id, device or "-", does_take,
                                   agg["n_served"], len(w["segments"]),
                                   agg["n_steps"], start_s, start_s + elapsed,
                                   "simulated"))

            if len(bd_rows) >= 20000:
                _flush(con, bd_rows, trav_rows, total_rows, ping_rows, match_rows)
        day += dt.timedelta(days=1)

    _flush(con, bd_rows, trav_rows, total_rows, ping_rows, match_rows)
    con.commit()
    out = counts(con)
    con.execute("VACUUM")
    out.update({"days": n_days, "seconds": round(time.time() - t_start, 1),
                "detail_from": detail_from.isoformat(),
                "pings_from": ping_from.isoformat()})
    con.close()
    return out


def _flush(con, bd, trav, totals, pings_, matches):
    if bd:
        con.executemany("INSERT OR REPLACE INTO block_days VALUES "
                        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", bd)
        bd.clear()
    if trav:
        con.executemany("INSERT OR REPLACE INTO segment_traversals VALUES "
                        "(?,?,?,?,?,?,?,?,?,?)", trav)
        trav.clear()
    if totals:
        con.executemany("INSERT OR REPLACE INTO block_traversal_totals VALUES "
                        "(?,?,?,?,?,?,?,?,?)", totals)
        totals.clear()
    if pings_:
        con.executemany("INSERT OR REPLACE INTO pings VALUES (?,?,?,?,?,?,?,?)", pings_)
        pings_.clear()
    if matches:
        con.executemany("INSERT OR REPLACE INTO ping_osm_match VALUES (?,?,?,?)", matches)
        matches.clear()
    con.commit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--blocks-per-office", type=int, default=12)
    ap.add_argument("--detail-days", type=int, default=180,
                    help="how far back the per-segment traversals go (default 180)")
    ap.add_argument("--ping-days", type=int, default=5,
                    help="how far back the raw pings go (default 5)")
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--end-date", default=None)
    a = ap.parse_args()
    r = generate(a.db, a.years, a.blocks_per_office, a.detail_days, a.ping_days,
                 a.seed, a.end_date)
    print(f"built in {r['seconds']}s: {r['blocks']} blocks, {r['osm_segments']:,} segments, "
          f"{r['delivery_points']:,} delivery points")
    print(f"  {r['block_days']:,} walks from {r['from']} to {r['to']} ({r['days']} days)")
    print(f"  {r['segment_traversals']:,} segment traversals (per-segment detail from "
          f"{r['detail_from']})")
    print(f"  {r['pings']:,} pings with {r['ping_osm_match']:,} OSM matches (from "
          f"{r['pings_from']})")
