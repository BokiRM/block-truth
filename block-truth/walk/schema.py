"""
The database, shaped like the one the work already lives in.

Their pipeline, from the two slides, is:

    OSM segments  ->  geofence per segment, width from DP distance
    pings         ->  matched to OSM, filtered to walking speed
    geofences     ->  entry and exit per segment, per device
    sum           ->  "does take" per device per block

So these tables carry *their* names and *their* granularity, not ours. The point
is that the adapter between their system and ours is a thin mapping rather than
a translation, and that nothing they already compute is computed again here:

    ping_osm_match      their existing ping-to-OSM matching table
    dp_segment_join     their existing DP-to-segment join
    segment_geofence    the buffer, sized from the DP distances in that join
    segment_traversals  entry and exit per segment, per device, per pass

What is ours sits in its own tables with a `walk_` prefix, and never writes to
theirs. If their real tables arrive, the mock ones are dropped and the adapter
points at the real ones; nothing else in the app should need to notice.

One deliberate difference from a toy: a segment can be walked more than once in
a round - down one side and back up the other, or as a dead leg to get to the
next street - so traversals carry a `pass_no`, and only the delivering pass has
service time in it. Their sum adds every pass, which is right for "time on the
block" and wrong for "time per segment", and that distinction has to survive
into our tables or we will quietly lose it.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_WALK_DB = Path(os.environ.get(
    "WALK_DB", Path(__file__).resolve().parent.parent / "data" / "walk.db"))

# ---------------------------------------------------------------- vocabulary
AREA_TYPES = ["urban", "suburban", "rural"]
DP_TYPES = ["house", "terrace", "flat", "business", "farm", "locker"]
ACCESS_TYPES = ["front door", "gate", "stairs", "buzzer", "reception", "long drive"]
DELIVERY_TYPES = ["letters", "packets", "parcels", "signed for", "special delivery"]
VEHICLE_TYPES = ["van", "trolley", "bike", "on foot"]
WEATHER = ["clear", "overcast", "rain", "heavy rain", "wind", "snow", "ice"]

# Their filter, from slide 2: a device is walking when its matched speed is
# below this. 2.32 m/s is 8.35 km/h - a brisk jog, so it is generous on purpose,
# which is why the "two consecutive walking pings" rule exists beside it.
WALKING_SPEED_LIMIT_MPS = 2.32

SCHEMA = """
-- ----------------------------------------------------------------- places
CREATE TABLE IF NOT EXISTS delivery_offices (
    do_id        TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    region       TEXT NOT NULL,
    country      TEXT NOT NULL,
    lat          REAL NOT NULL,
    lon          REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS blocks (
    block_id     TEXT PRIMARY KEY,
    do_id        TEXT NOT NULL REFERENCES delivery_offices(do_id),
    name         TEXT NOT NULL,
    area_type    TEXT NOT NULL,              -- urban | suburban | rural
    round_type   TEXT NOT NULL,              -- walk | park and loop
    planned_start TEXT NOT NULL,             -- HH:MM the walker is due on the block
    drive_minutes REAL NOT NULL,             -- office to the block and back, in the van
    -- Sorting the frame, bagging up, and the walk to the van. It is not part of
    -- "does take" - no geofence covers the sorting office - but it is most of
    -- the difference between the walk and the shift, and a resourcing plan that
    -- leaves it out is short by an hour and a half per walker per day.
    prep_minutes  REAL NOT NULL DEFAULT 60
);

-- A segment is a straight piece of road between curves and junctions: roughly
-- 10-25 m in a town, 50-150 m in the country, as on their first slide. Nodes
-- are shared between segments that meet, which is what makes the block a graph
-- and lets a route be planned over it.
CREATE TABLE IF NOT EXISTS osm_segments (
    segment_id   TEXT PRIMARY KEY,
    block_id     TEXT NOT NULL REFERENCES blocks(block_id),
    osm_way_id   INTEGER NOT NULL,
    way_seq      INTEGER NOT NULL,
    street_name  TEXT,
    from_node    TEXT NOT NULL,
    to_node      TEXT NOT NULL,
    from_lat     REAL NOT NULL,
    from_lon     REAL NOT NULL,
    to_lat       REAL NOT NULL,
    to_lon       REAL NOT NULL,
    length_m     REAL NOT NULL,
    area_type    TEXT NOT NULL
);

-- The buffer around each segment. Their slide: width depends on the average
-- distance from the segment to the delivery points joined to it, using the
-- existing DP-to-segment join. Stored rather than recomputed so that what the
-- entry and exit times were measured against is on record.
CREATE TABLE IF NOT EXISTS segment_geofence (
    segment_id          TEXT PRIMARY KEY REFERENCES osm_segments(segment_id),
    buffer_m            REAL NOT NULL,
    mean_dp_distance_m  REAL,
    dp_count            INTEGER NOT NULL DEFAULT 0,
    method              TEXT NOT NULL DEFAULT 'mean dp distance'
);

CREATE TABLE IF NOT EXISTS delivery_points (
    dp_id        TEXT PRIMARY KEY,
    block_id     TEXT NOT NULL REFERENCES blocks(block_id),
    lat          REAL NOT NULL,
    lon          REAL NOT NULL,
    dp_type      TEXT NOT NULL,
    access_type  TEXT NOT NULL,
    households   INTEGER NOT NULL DEFAULT 1,   -- a block of flats is one DP, many doors
    address      TEXT
);

-- Their existing join. One row per (DP, segment) with the distance; the primary
-- one is the segment the DP is delivered from.
CREATE TABLE IF NOT EXISTS dp_segment_join (
    dp_id        TEXT NOT NULL REFERENCES delivery_points(dp_id),
    segment_id   TEXT NOT NULL REFERENCES osm_segments(segment_id),
    distance_m   REAL NOT NULL,
    is_primary   INTEGER NOT NULL DEFAULT 1,
    side         TEXT,                         -- left | right of the segment
    PRIMARY KEY (dp_id, segment_id)
);

-- ------------------------------------------------------------- the people
CREATE TABLE IF NOT EXISTS walkers (
    walker_id    TEXT PRIMARY KEY,
    do_id        TEXT NOT NULL REFERENCES delivery_offices(do_id),
    name         TEXT NOT NULL,
    started_on   TEXT NOT NULL,
    left_on      TEXT
);

CREATE TABLE IF NOT EXISTS devices (
    device_id        TEXT PRIMARY KEY,
    walker_id        TEXT REFERENCES walkers(walker_id),
    model            TEXT,
    ping_interval_s  INTEGER NOT NULL DEFAULT 10
);

CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id   TEXT PRIMARY KEY,
    do_id        TEXT NOT NULL REFERENCES delivery_offices(do_id),
    vehicle_type TEXT NOT NULL
);

-- --------------------------------------------------------------- the work
-- One row per block per day it was walked: who walked it, on what, in what
-- weather, with how much of each kind of mail. This is the operational record
-- the forecast is made against.
CREATE TABLE IF NOT EXISTS block_days (
    -- A surrogate key on purpose. The natural one - block and date - is
    -- nineteen characters, and it is carried by two million traversal rows and
    -- three indexes over them; as an integer that is a fifth of the database.
    block_day_id  INTEGER PRIMARY KEY,
    block_id      TEXT NOT NULL REFERENCES blocks(block_id),
    date          TEXT NOT NULL,
    walker_id     TEXT REFERENCES walkers(walker_id),
    device_id     TEXT REFERENCES devices(device_id),
    vehicle_id    TEXT REFERENCES vehicles(vehicle_id),
    weather       TEXT,
    temp_c        REAL,
    rain_mm       REAL,
    wind_mps      REAL,
    daylight      INTEGER NOT NULL DEFAULT 1,
    items_letters INTEGER NOT NULL DEFAULT 0,
    items_packets INTEGER NOT NULL DEFAULT 0,
    items_parcels INTEGER NOT NULL DEFAULT 0,
    items_signed  INTEGER NOT NULL DEFAULT 0,
    items_special INTEGER NOT NULL DEFAULT 0,
    dps_served    INTEGER NOT NULL DEFAULT 0,
    actual_start  TEXT,
    UNIQUE (block_id, date)
);

-- Raw pings, for a recent window only: two years of them at ten seconds would
-- be a hundred million rows to make a point that a fortnight makes just as well.
-- Older history carries the traversals without the pings behind them, and says
-- so in `segment_traversals.source`.
CREATE TABLE IF NOT EXISTS pings (
    ping_id       INTEGER PRIMARY KEY,
    device_id     TEXT NOT NULL,
    block_day_id  INTEGER NOT NULL REFERENCES block_days(block_day_id),
    ts            REAL NOT NULL,              -- seconds since midnight, local
    lat           REAL NOT NULL,
    lon           REAL NOT NULL,
    speed_mps     REAL NOT NULL,
    accuracy_m    REAL NOT NULL DEFAULT 8.0
);

-- Their existing ping-to-OSM matching table.
CREATE TABLE IF NOT EXISTS ping_osm_match (
    ping_id      INTEGER PRIMARY KEY REFERENCES pings(ping_id),
    segment_id   TEXT NOT NULL REFERENCES osm_segments(segment_id),
    distance_m   REAL NOT NULL,
    is_walking   INTEGER NOT NULL             -- speed < 2.32 m/s
);

-- The output of their second slide: entry and exit per segment, per device.
-- `pass_no` because a segment can be walked more than once in a round, and
-- `serviced` because only one of those passes has the doors in it.
CREATE TABLE IF NOT EXISTS segment_traversals (
    block_day_id INTEGER NOT NULL REFERENCES block_days(block_day_id),
    segment_id   TEXT NOT NULL REFERENCES osm_segments(segment_id),
    device_id    TEXT NOT NULL,
    pass_no      INTEGER NOT NULL,
    entry_ts     REAL NOT NULL,
    exit_ts      REAL NOT NULL,
    traversal_s  REAL NOT NULL,
    ping_count   INTEGER NOT NULL DEFAULT 0,
    serviced     INTEGER NOT NULL DEFAULT 1,
    source       TEXT NOT NULL DEFAULT 'ping',  -- ping | simulated
    PRIMARY KEY (block_day_id, segment_id, device_id, pass_no)
);

-- Their "does take": per device, the sum of its segment traversal times.
CREATE TABLE IF NOT EXISTS block_traversal_totals (
    block_day_id      INTEGER NOT NULL REFERENCES block_days(block_day_id),
    device_id         TEXT NOT NULL,
    does_take_s       REAL NOT NULL,
    segments_covered  INTEGER NOT NULL,
    segments_in_block INTEGER NOT NULL,
    passes            INTEGER NOT NULL,
    first_entry_ts    REAL,
    last_exit_ts      REAL,
    source            TEXT NOT NULL DEFAULT 'ping',
    PRIMARY KEY (block_day_id, device_id)
);

CREATE INDEX IF NOT EXISTS ix_seg_block     ON osm_segments(block_id);
CREATE INDEX IF NOT EXISTS ix_dpj_segment   ON dp_segment_join(segment_id);
CREATE INDEX IF NOT EXISTS ix_bd_date       ON block_days(date);
CREATE INDEX IF NOT EXISTS ix_bd_block      ON block_days(block_id);
CREATE INDEX IF NOT EXISTS ix_bd_walker     ON block_days(walker_id);
CREATE INDEX IF NOT EXISTS ix_trav_bd       ON segment_traversals(block_day_id);
CREATE INDEX IF NOT EXISTS ix_trav_segment  ON segment_traversals(segment_id);
CREATE INDEX IF NOT EXISTS ix_ping_bd       ON pings(block_day_id);
CREATE INDEX IF NOT EXISTS ix_totals_bd     ON block_traversal_totals(block_day_id);

-- ------------------------------------------------------------------ ours
-- Everything below is this application's own and is never written by theirs.
CREATE TABLE IF NOT EXISTS walk_routes (
    block_id     TEXT PRIMARY KEY REFERENCES blocks(block_id),
    strategy     TEXT NOT NULL,
    order_json   TEXT NOT NULL,               -- the segment order, with passes
    deadhead_m   REAL NOT NULL DEFAULT 0,
    total_m      REAL NOT NULL DEFAULT 0,
    chosen_by    TEXT,
    chosen_at    TEXT
);

CREATE TABLE IF NOT EXISTS walk_forecasts (
    block_day_id INTEGER PRIMARY KEY REFERENCES block_days(block_day_id),
    should_take_s REAL NOT NULL,
    low_s        REAL NOT NULL,
    high_s       REAL NOT NULL,
    confidence   INTEGER NOT NULL DEFAULT 90,
    basis        TEXT,
    made_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def connect(path: str | Path = None, read_only: bool = False) -> sqlite3.Connection:
    p = Path(path or DEFAULT_WALK_DB)
    if read_only:
        if not p.exists():
            raise FileNotFoundError(f"no database at {p}")
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(p, check_same_thread=False)
        con.executescript(SCHEMA)
    con.row_factory = sqlite3.Row
    return con


def counts(con: sqlite3.Connection) -> dict[str, int]:
    out = {}
    for t in ("delivery_offices", "blocks", "osm_segments", "delivery_points",
              "walkers", "devices", "vehicles", "block_days", "pings",
              "ping_osm_match", "segment_traversals", "block_traversal_totals"):
        try:
            out[t] = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            out[t] = 0
    r = con.execute("SELECT min(date), max(date) FROM block_days").fetchone()
    out["from"], out["to"] = (r[0], r[1]) if r else (None, None)
    return out
