"""
HTTP: block completion times, recorded and forecast.

    uvicorn block_truth.api:app --port 8100     # then open http://127.0.0.1:8100

Two kinds of answer come out of this service and they are never blended:

    recorded block  ->  ground truth   derived from what the vehicle did
    entered block   ->  estimate       expected, earliest and latest

`kind` on every block response says which one it is, and the page styles them
differently so a forecast can never be read as a measurement.
"""
from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .db import PARCEL_SIZES, POINT_TYPES, VEHICLE_TYPES, counts
from .excel import parse, template
from .itinerary import MINUTES_PER_MISSED_WINDOW, Limits
from .repository import Repository
from .schedule import generate
from .traffic import KINDS
from .truth import TruthError

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "web" / "app.html"

app = FastAPI(title="Block completion time", version="2.0", description=__doc__,
              docs_url="/api/docs", redoc_url=None)


@lru_cache(maxsize=1)
def repo() -> Repository:
    try:
        return Repository(os.environ.get("BLOCK_TRUTH_DB") or None)
    except Exception as e:                                   # pragma: no cover
        raise HTTPException(503, str(e))


def _err(e: Exception):
    raise HTTPException(400, str(e))


# ----------------------------------------------------------------------- page
@app.get("/", include_in_schema=False)
def index():
    if not PAGE.exists():
        raise HTTPException(503, "web/app.html is missing")
    return FileResponse(PAGE, media_type="text/html")


@app.get("/vendor/{path:path}", include_in_schema=False)
def vendor(path: str):
    """The map library, served from here rather than from a CDN.

    The page asks nothing of the network and works in a building with no route
    to the internet. Map *tiles* have to come from somewhere, and that somewhere
    is a setting - an internal tile server does as well as a public one - but
    the library itself ships with the app.
    """
    f = (ROOT / "web" / "vendor" / path).resolve()
    if not str(f).startswith(str((ROOT / "web" / "vendor").resolve())) or not f.is_file():
        raise HTTPException(404, "not found")
    kinds = {".js": "application/javascript", ".css": "text/css",
             ".png": "image/png", ".svg": "image/svg+xml", ".txt": "text/plain"}
    return FileResponse(f, media_type=kinds.get(f.suffix, "application/octet-stream"))


@app.get("/api/health")
def health():
    r = repo()
    return {"ok": True, "database": str(r.path), **counts(r.con),
            "vocabulary": {"point_types": POINT_TYPES, "parcel_sizes": PARCEL_SIZES,
                           "vehicle_types": VEHICLE_TYPES}}


# --------------------------------------------------------------------- browse
@app.get("/api/bootstrap")
def bootstrap():
    r = repo()
    return {"depots": r.depots(), "dates": r.dates(),
            "point_types": POINT_TYPES, "parcel_sizes": PARCEL_SIZES,
            "vehicle_types": VEHICLE_TYPES, "counts": counts(r.con)}


@app.get("/api/blocks")
def blocks(date: str | None = None, source: str | None = None,
           depot_id: str | None = None, limit: int = Query(500, ge=1, le=5000)):
    return {"rows": repo().blocks(date, source, depot_id, limit)}


@app.get("/api/block/{block_id}")
def block(block_id: str, confidence: int = Query(90, ge=50, le=99)):
    conf = min((80, 90, 95, 99), key=lambda c: abs(c - confidence))
    try:
        return repo().block_view(block_id, confidence=conf)
    except TruthError as e:
        raise HTTPException(404, str(e))


@app.get("/api/calibration")
def calibration(refresh: bool = False):
    """What the estimator learned from the recorded rounds, and from how many."""
    return repo().calibration(refresh=refresh).as_dict()


# ------------------------------------------------------------ the month ahead
@app.get("/api/schedule/days")
def schedule_days(days: int = Query(35, ge=1, le=90), from_date: str | None = None):
    """One row per planned day: the work on it and the hours it is predicted to take."""
    r = repo()
    rows = r.schedule_days(days=days, from_date=from_date)
    return {"rows": rows, "problems": r.last_problems[:40],
            "problem_count": len(r.last_problems)}


@app.get("/api/schedule")
def schedule(date: str | None = None, depot_id: str | None = None,
             vehicle_id: str | None = None, city: str | None = None,
             region: str | None = None, confidence: int = Query(90, ge=50, le=99),
             traffic: bool = True, max_minutes: float | None = None,
             min_minutes: float | None = None, max_km: float | None = None,
             max_stops: int | None = None, limit: int = Query(300, ge=1, le=2000)):
    conf = min((80, 90, 95, 99), key=lambda c: abs(c - confidence))
    lim = Limits(min_minutes, max_minutes, max_km, max_stops)
    r = repo()
    rows = r.schedule(date, depot_id, vehicle_id, city, region, confidence=conf,
                      use_traffic=traffic, limit=limit, limits=lim)
    return {"rows": rows, "limits": lim.as_dict(),
            "problems": r.last_problems[:40],
            "problem_count": len(r.last_problems)}


@app.get("/api/itineraries/{block_id}")
def itineraries(block_id: str, confidence: int = Query(90, ge=50, le=99),
                traffic: bool = True, max_minutes: float | None = None,
                min_minutes: float | None = None, max_km: float | None = None,
                max_stops: int | None = None,
                minutes_per_missed_window: float = Query(
                    MINUTES_PER_MISSED_WINDOW, ge=0, le=240)):
    """Several ways to drive one planned round, each costed under the same day."""
    conf = min((80, 90, 95, 99), key=lambda c: abs(c - confidence))
    try:
        return repo().itineraries(block_id, confidence=conf, use_traffic=traffic,
                                  limits=Limits(min_minutes, max_minutes, max_km,
                                                max_stops),
                                  minutes_per_missed_window=minutes_per_missed_window)
    except TruthError as e:
        raise HTTPException(404, str(e))


@app.post("/api/itineraries/{block_id}/choose")
def choose_itinerary(block_id: str, key: str, by: str = "operator",
                     confidence: int = Query(90, ge=50, le=99), traffic: bool = True,
                     max_minutes: float | None = None, min_minutes: float | None = None,
                     max_km: float | None = None, max_stops: int | None = None,
                     minutes_per_missed_window: float = Query(
                         MINUTES_PER_MISSED_WINDOW, ge=0, le=240)):
    """Drive this one: applies the order to the round and records who chose it."""
    conf = min((80, 90, 95, 99), key=lambda c: abs(c - confidence))
    try:
        return repo().choose_itinerary(
            block_id, key, by=by, confidence=conf, use_traffic=traffic,
            limits=Limits(min_minutes, max_minutes, max_km, max_stops))
    except TruthError as e:
        raise HTTPException(400, str(e))


@app.post("/api/itineraries/{block_id}/reset")
def reset_itinerary(block_id: str):
    """Put the round back in the order the work arrived in."""
    try:
        return repo().clear_choice(block_id)
    except TruthError as e:
        raise HTTPException(400, str(e))


@app.post("/api/schedule/generate")
def schedule_generate(days: int = Query(30, ge=1, le=90),
                      from_date: str | None = None):
    """Rebuild the plan for the month ahead from the fleet and volumes on record."""
    r = repo()
    try:
        out = generate(r.path, days=days, from_date=from_date)
    except Exception as e:
        raise HTTPException(400, str(e))
    r.calibration(refresh=True)
    return out


# ------------------------------------------------------- conditions on the road
@app.get("/api/conditions")
def conditions(date: str | None = None):
    """Traffic, measured from the records; disruptions, declared by somebody."""
    r = repo()
    t = r.traffic()
    return {"date": date, "traffic": t.as_dict(),
            "disruptions": [d.as_dict() for d in r.disruptions(date)],
            "kinds": KINDS,
            "note": ("The traffic curve is measured from recorded legs and is "
                     "relative to each city's own average pace. Disruptions are "
                     "declared: somebody entered them.")}


class DisruptionIn(BaseModel):
    kind: str = "closure"
    description: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0.04, le=50)
    from_date: str
    to_date: str
    from_time: str | None = None
    to_time: str | None = None
    severity: float = Field(1.5, ge=1.0, le=6.0)
    depot_id: str | None = None


@app.post("/api/disruption")
def add_disruption(d: DisruptionIn):
    r = repo()
    try:
        did = r.add_disruption(**d.model_dump())
    except TruthError as e:
        _err(e)
    return {"disruption_id": did, "saved": True}


@app.delete("/api/disruption/{disruption_id}")
def delete_disruption(disruption_id: str):
    try:
        repo().delete_disruption(disruption_id)
    except TruthError as e:
        raise HTTPException(404, str(e))
    return {"deleted": disruption_id}


@app.get("/api/datacheck")
def datacheck():
    """Rows that will break something later, found before they do.

    Written after a live database produced a 500 from inside a trigonometric
    function because one stop had no coordinates. The stack trace named
    `math.radians`; it did not name the block. This does.
    """
    con = repo().con
    out = {}
    out["stops_without_coordinates"] = [dict(r) for r in con.execute(
        "SELECT block_id, sequence, address FROM stops "
        "WHERE lat IS NULL OR lon IS NULL LIMIT 50")]
    out["stops_off_the_earth"] = [dict(r) for r in con.execute(
        "SELECT block_id, sequence, lat, lon FROM stops "
        "WHERE lat NOT BETWEEN -90 AND 90 OR lon NOT BETWEEN -180 AND 180 LIMIT 50")]
    out["blocks_without_stops"] = [r["block_id"] for r in con.execute(
        "SELECT block_id FROM blocks b WHERE NOT EXISTS "
        "(SELECT 1 FROM stops s WHERE s.block_id = b.block_id) LIMIT 50")]
    out["blocks_with_gaps_in_the_sequence"] = [dict(r) for r in con.execute(
        "SELECT block_id, count(*) stops, min(sequence) lo, max(sequence) hi "
        "FROM stops GROUP BY block_id HAVING hi - lo + 1 <> stops LIMIT 50")]
    out["depots_without_coordinates"] = [r["depot_id"] for r in con.execute(
        "SELECT depot_id FROM depots WHERE lat IS NULL OR lon IS NULL")]
    out["planned_without_manifest"] = con.execute(
        "SELECT count(*) n FROM stops s JOIN blocks b USING(block_id) "
        "WHERE b.source='planned' AND s.manifest_sequence IS NULL").fetchone()["n"]
    out["ok"] = not any(v for k, v in out.items() if k != "ok")
    return out


@app.get("/api/audit/{date}")
def audit(date: str):
    """Recorded blocks whose own records do not add up. Empty is the good answer."""
    r = repo()
    bad = []
    for b in r.blocks(date=date, source="recorded"):
        try:
            v = r.block_view(b["block_id"])
        except TruthError:
            continue
        if v.get("kind") == "truth" and not v.get("ok", True):
            bad.append({"block_id": b["block_id"], "vehicle_id": b["vehicle_id"],
                        "stops": b["stops"],
                        "failed": [c for c in v["checks"] if not c["ok"]]})
    return {"date": date, "failing": len(bad), "rows": bad}


# ---------------------------------------------------------------------- write
class StopIn(BaseModel):
    sequence: int = Field(ge=1)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    address: str | None = None
    postcode: str | None = None
    point_type: str = "House"
    parcel_size: str = "Medium"
    promised_from: str | None = None
    promised_to: str | None = None


class BlockIn(BaseModel):
    block_id: str | None = None
    delivery_date: str
    depot_id: str
    start_time: str = "07:30"
    vehicle_id: str | None = None
    vehicle_type: str = "Large Van"
    note: str | None = None
    source: str = "manual"
    stops: list[StopIn]


@app.post("/api/block", status_code=201)
def create(block: BlockIn):
    r = repo()
    bid = block.block_id or r.suggest_block_id(block.delivery_date, block.depot_id)
    try:
        r.create_block(bid, block.delivery_date, block.depot_id,
                       [s.model_dump() for s in block.stops],
                       start_time=block.start_time, vehicle_id=block.vehicle_id,
                       vehicle_type=block.vehicle_type, note=block.note,
                       source=block.source)
    except (TruthError, ValueError) as e:
        _err(e)
    return r.block_view(bid)


@app.post("/api/estimate")
def preview(block: BlockIn):
    """Estimate a round without saving it, so the form can show it as it is built."""
    r = repo()
    depot = next((d for d in r.depots() if d["depot_id"] == block.depot_id), None)
    if depot is None:
        raise HTTPException(400, f"no depot {block.depot_id}")
    if not block.stops:
        raise HTTPException(400, "a block needs at least one stop")
    from .estimate import StopPlan, estimate as run
    try:
        e = run(block_id=block.block_id or "preview", delivery_date=block.delivery_date,
                start=block.start_time, depot=(depot["lat"], depot["lon"]),
                stops=[StopPlan(s.sequence, s.lat, s.lon, s.point_type, s.parcel_size,
                                s.address, s.postcode, s.promised_from, s.promised_to)
                       for s in block.stops],
                cal=r.calibration(), city=depot["city"],
                vehicle_type=block.vehicle_type,
                attributes={"depot_id": depot["depot_id"], "city": depot["city"],
                            "vehicle_type": block.vehicle_type, "source": "preview"})
    except ValueError as e2:
        _err(e2)
    out = e.as_dict()
    fake_block = {"depot_lat": depot["lat"], "depot_lon": depot["lon"],
                  "depot_id": depot["depot_id"], "depot_name": depot["name"]}
    out["route"] = r.route(fake_block,
                           [{**s.model_dump(), "arrival_time": None,
                             "service_minutes": None, "travel_minutes": None,
                             "outcome": None} for s in block.stops], out)
    return out


@app.delete("/api/block/{block_id}")
def delete(block_id: str):
    try:
        repo().delete_block(block_id)
    except TruthError as e:
        _err(e)
    return {"deleted": block_id}


@app.get("/api/suggest-id")
def suggest_id(date: str, depot_id: str):
    return {"block_id": repo().suggest_block_id(date, depot_id)}


# ---------------------------------------------------------------------- excel
@app.get("/api/template.xlsx", include_in_schema=True)
def download_template():
    return Response(
        template(), media_type="application/vnd.openxmlformats-officedocument."
                               "spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="block-template.xlsx"'})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), sheet: str | None = Form(None)):
    """Parse a workbook and hand back what it found. Nothing is saved here.

    The caller sees the stops and every problem, fixes the sheet or the form,
    and then posts to /api/block. A half-imported round is worse than a
    rejected one.
    """
    name = (file.filename or "").lower()
    if not name.endswith((".xlsx", ".xlsm")):
        raise HTTPException(400, "expected an .xlsx workbook")
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(413, "that workbook is larger than 8 MB")
    try:
        p = parse(data, sheet)
    except Exception as e:
        raise HTTPException(400, f"could not read the workbook: {e}")
    return {
        "ok": p.ok, "sheet": p.sheet, "stops": p.stops, "meta": p.meta,
        "columns_found": sorted(p.header), "stop_count": len(p.stops),
        "errors": [{"row": e.row, "column": e.column, "problem": e.problem}
                   for e in p.errors],
    }
