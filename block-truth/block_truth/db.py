"""
The application's own database.

Three tables and no more than they need to be. A block that was *recorded* has
arrival, service and travel times on every stop; a block that was *entered* -
by hand or from a workbook - has none of them, and that is the whole difference
between a ground truth and an estimate.

    depots   where rounds start and end
    blocks   one vehicle's round on one day
    stops    one delivery, in visiting order, with coordinates

`source` on a block says where it came from, and the UI never shows a measured
number and a predicted one in the same style.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(os.environ.get(
    "BLOCK_TRUTH_DB", Path(__file__).resolve().parent.parent / "data" / "blocks.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS depots (
    depot_id   TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    city       TEXT NOT NULL,
    region     TEXT NOT NULL,
    lat        REAL NOT NULL,
    lon        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS blocks (
    block_id        TEXT PRIMARY KEY,
    delivery_date   TEXT NOT NULL,
    vehicle_id      TEXT,
    vehicle_type    TEXT,
    depot_id        TEXT NOT NULL REFERENCES depots(depot_id),
    start_time      TEXT NOT NULL,              -- HH:MM, when the vehicle is due out
    duration_minutes REAL,                      -- recorded blocks only
    source          TEXT NOT NULL DEFAULT 'recorded',  -- recorded | manual | excel
    note            TEXT,
    -- Which routing suggestion this round is currently driving, and who said so.
    -- The choice is applied by rewriting the stops' visiting order, so there is
    -- one answer to "what order is this round in" rather than two that can drift
    -- apart. These columns only record where that order came from.
    chosen_key      TEXT,       -- as_planned | shortest | windows | steady | as_received
    chosen_by       TEXT,       -- 'recommendation' or whoever picked it
    chosen_at       TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stops (
    block_id          TEXT NOT NULL REFERENCES blocks(block_id) ON DELETE CASCADE,
    sequence          INTEGER NOT NULL,     -- visiting order
    -- Where this stop sat on the manifest before anybody routed it. Kept so the
    -- routing suggestions can be measured against the order the work arrived
    -- in: without it, "we saved two hours" has nothing to be two hours better
    -- than. NULL for recorded rounds, which were never a manifest.
    manifest_sequence INTEGER,
    address           TEXT,
    postcode          TEXT,
    lat               REAL NOT NULL,
    lon               REAL NOT NULL,
    point_type        TEXT NOT NULL DEFAULT 'House',
    parcel_size       TEXT NOT NULL DEFAULT 'Medium',
    promised_from     TEXT,                     -- HH:MM, optional
    promised_to       TEXT,
    arrival_time      TEXT,                     -- recorded blocks only
    service_minutes   REAL,
    travel_minutes    REAL,
    outcome           TEXT,
    PRIMARY KEY (block_id, sequence)
);

-- Declared, not measured. A closed road leaves no trace in the delivery
-- records, so somebody has to say it is closed: by hand, from a workbook, or
-- one day from a feed. `source` keeps that distinction, and the UI shows these
-- in the forecast colour, never the measured one.
CREATE TABLE IF NOT EXISTS disruptions (
    disruption_id TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,            -- closure | works | congestion | event
    description   TEXT NOT NULL,
    lat           REAL NOT NULL,
    lon           REAL NOT NULL,
    radius_km     REAL NOT NULL,
    from_date     TEXT NOT NULL,
    to_date       TEXT NOT NULL,
    from_time     TEXT,                     -- time of day, optional
    to_time       TEXT,
    severity      REAL NOT NULL DEFAULT 1.5,  -- ignored for a closure: that is a detour
    depot_id      TEXT,                     -- whose patch it is in, for filtering
    source        TEXT NOT NULL DEFAULT 'manual',   -- manual | excel | feed
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_blocks_date   ON blocks(delivery_date);
CREATE INDEX IF NOT EXISTS ix_blocks_source ON blocks(source);
CREATE INDEX IF NOT EXISTS ix_blocks_depot  ON blocks(depot_id);
CREATE INDEX IF NOT EXISTS ix_disr_dates    ON disruptions(from_date, to_date);
"""

# The vocabulary the estimator is calibrated over. Keeping it short and explicit
# beats free text, because every value here has to have been seen often enough
# for its service time to mean something.
POINT_TYPES = ["House", "Flat", "Office", "Business", "Parcel Locker"]
PARCEL_SIZES = ["Letter", "Small", "Medium", "Large", "Bulky"]
VEHICLE_TYPES = ["Small Van", "Large Van", "Box Truck"]


# Columns added after the first release. `CREATE TABLE IF NOT EXISTS` does
# nothing to a table that already exists, so a database made by an earlier
# version keeps its old shape and every query against a new column fails. These
# are added on open instead, which is cheap and keeps old files working.
LATER_COLUMNS = [
    ("stops", "manifest_sequence", "INTEGER"),
    ("blocks", "chosen_key", "TEXT"),
    ("blocks", "chosen_by", "TEXT"),
    ("blocks", "chosen_at", "TEXT"),
]


def _migrate(con: sqlite3.Connection) -> None:
    for table, column, decl in LATER_COLUMNS:
        try:
            have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        except sqlite3.OperationalError:
            continue
        if have and column not in have:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    con.commit()


def connect(path: str | Path = None, read_only: bool = False) -> sqlite3.Connection:
    p = Path(path or DEFAULT_DB)
    if read_only:
        if not p.exists():
            raise FileNotFoundError(f"no database at {p}")
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(p, check_same_thread=False)
        con.executescript(SCHEMA)
        _migrate(con)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init(path: str | Path = None) -> sqlite3.Connection:
    """Create the file and the schema if they are not there yet."""
    return connect(path)


def counts(con: sqlite3.Connection) -> dict[str, int]:
    out = {}
    for t in ("depots", "blocks", "stops"):
        out[t] = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
    out["recorded"] = con.execute(
        "SELECT count(*) FROM blocks WHERE source='recorded'").fetchone()[0]
    out["planned"] = con.execute(
        "SELECT count(*) FROM blocks WHERE source='planned'").fetchone()[0]
    # entered by a person, as opposed to recorded by a vehicle or planned by the
    # scheduler; three different things and the chips on the page say so
    out["entered"] = out["blocks"] - out["recorded"] - out["planned"]
    try:
        out["disruptions"] = con.execute("SELECT count(*) FROM disruptions").fetchone()[0]
    except sqlite3.OperationalError:          # an older file, before disruptions
        out["disruptions"] = 0
    return out
