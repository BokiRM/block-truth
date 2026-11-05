"""
Blocks in and out of a workbook.

A spreadsheet is how a depot actually hands over a round, so the import is
forgiving about shape and strict about meaning: column names are matched
loosely, every row is validated on its own, and a bad row is reported with its
row number rather than taking the upload down with it.

Nothing is written until the whole sheet parses. A half-imported round is worse
than a rejected one.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from openpyxl import Workbook, load_workbook

from .db import PARCEL_SIZES, POINT_TYPES
from .fmt import hhmm, to_minutes

# What a column may be called. Compared after lowercasing and stripping
# everything that is not a letter or a digit.
ALIASES = {
    "sequence": ["sequence", "seq", "stop", "stopno", "stopnumber", "order", "no", "nr",
                 "redosled", "redni"],
    "lat": ["lat", "latitude", "y", "gpslat", "sirina"],
    "lon": ["lon", "lng", "long", "longitude", "x", "gpslon", "duzina"],
    "address": ["address", "addressline", "street", "adresa"],
    "postcode": ["postcode", "postalcode", "zip", "zipcode", "ptt"],
    "point_type": ["pointtype", "type", "doortype", "premises", "tip", "tipmesta"],
    "parcel_size": ["parcelsize", "size", "parcel", "velicina"],
    "promised_from": ["promisedfrom", "windowfrom", "from", "slotfrom", "odd", "od"],
    "promised_to": ["promisedto", "windowto", "to", "slotto", "do"],
}
REQUIRED = ["sequence", "lat", "lon"]


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


@dataclass
class RowError:
    row: int
    column: str
    problem: str


@dataclass
class ParsedSheet:
    stops: list[dict] = field(default_factory=list)
    errors: list[RowError] = field(default_factory=list)
    header: dict[str, int] = field(default_factory=dict)
    meta: dict[str, str] = field(default_factory=dict)
    sheet: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.stops) and not self.errors


def _match_header(cells) -> dict[str, int]:
    """Find each known field in the header row, whatever it is called."""
    found = {}
    for idx, cell in enumerate(cells):
        key = _norm(cell)
        if not key:
            continue
        for field_name, names in ALIASES.items():
            if field_name in found:
                continue
            if key in names:
                found[field_name] = idx
                break
    return found


def _clock(value, row: int, column: str, errors: list[RowError]) -> str | None:
    if value in (None, ""):
        return None
    if hasattr(value, "hour"):                       # a real Excel time
        return f"{value.hour:02d}:{value.minute:02d}"
    if isinstance(value, (int, float)):              # Excel's fraction of a day
        if 0 <= value <= 1:
            return hhmm(round(value * 1440))
        errors.append(RowError(row, column, f"{value!r} is not a time of day"))
        return None
    try:
        return hhmm(to_minutes(str(value)))
    except ValueError:
        errors.append(RowError(row, column, f"{value!r} is not a time like 09:30"))
        return None


def _choice(value, allowed, default, row, column, errors):
    if value in (None, ""):
        return default
    want = _norm(value)
    for a in allowed:
        if _norm(a) == want:
            return a
    errors.append(RowError(row, column,
                           f"{value!r} is not one of {', '.join(allowed)}"))
    return default


def parse(data: bytes | io.BytesIO, sheet: str | None = None) -> ParsedSheet:
    """Read stops out of a workbook. Reports every problem it finds, not the first."""
    wb = load_workbook(io.BytesIO(data) if isinstance(data, bytes) else data,
                       data_only=True, read_only=True)
    ws = wb[sheet] if sheet else wb.worksheets[0]
    out = ParsedSheet(sheet=ws.title)

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        out.errors.append(RowError(0, "-", "the sheet is empty"))
        return out

    # a header anywhere in the first few rows, so a title line above it is fine
    header_at, header = None, {}
    for i, r in enumerate(rows[:8]):
        found = _match_header(r)
        if all(k in found for k in REQUIRED):
            header_at, header = i, found
            break
    if header_at is None:
        out.errors.append(RowError(
            1, "-", "could not find a header row with at least "
                    "sequence, lat and lon - download the template to see the shape"))
        return out
    out.header = header

    # a "key: value" line above the header carries the block's own details
    for r in rows[:header_at]:
        for cell in r:
            if isinstance(cell, str) and ":" in cell:
                k, v = cell.split(":", 1)
                if _norm(k) in ("blockid", "block", "date", "deliverydate", "vehicle",
                                "vehicleid", "vehicletype", "depot", "depotid",
                                "start", "starttime"):
                    out.meta[_norm(k)] = v.strip()

    seen = set()
    for n, r in enumerate(rows[header_at + 1:], start=header_at + 2):
        if all(c in (None, "") for c in r):
            continue
        get = lambda f: (r[header[f]] if f in header and header[f] < len(r) else None)

        # a row with a number in the sequence column and nothing else is an unused
        # line of the template, not a mistake
        if get("lat") in (None, "") and get("lon") in (None, ""):
            if any(get(f) not in (None, "") for f in
                   ("address", "postcode", "point_type", "parcel_size",
                    "promised_from", "promised_to")):
                out.errors.append(RowError(n, "lat/lon",
                                           "the row has details but no coordinates"))
            continue

        try:
            seq = int(float(get("sequence")))
        except (TypeError, ValueError):
            out.errors.append(RowError(n, "sequence", f"{get('sequence')!r} is not a number"))
            continue
        if seq in seen:
            out.errors.append(RowError(n, "sequence", f"stop {seq} appears twice"))
            continue
        seen.add(seq)

        try:
            lat, lon = float(get("lat")), float(get("lon"))
        except (TypeError, ValueError):
            out.errors.append(RowError(n, "lat/lon",
                                       f"{get('lat')!r}, {get('lon')!r} are not coordinates"))
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            out.errors.append(RowError(n, "lat/lon", f"{lat}, {lon} is not on Earth"))
            continue

        pf = _clock(get("promised_from"), n, "promised_from", out.errors)
        pt_ = _clock(get("promised_to"), n, "promised_to", out.errors)
        if pf and pt_ and to_minutes(pf) > to_minutes(pt_):
            out.errors.append(RowError(n, "promised_from/to",
                                       f"the window closes ({pt_}) before it opens ({pf})"))

        out.stops.append({
            "sequence": seq, "lat": lat, "lon": lon,
            "address": (str(get("address")).strip() if get("address") else None),
            "postcode": (str(get("postcode")).strip() if get("postcode") else None),
            "point_type": _choice(get("point_type"), POINT_TYPES, "House",
                                  n, "point_type", out.errors),
            "parcel_size": _choice(get("parcel_size"), PARCEL_SIZES, "Medium",
                                   n, "parcel_size", out.errors),
            "promised_from": pf, "promised_to": pt_,
        })

    wb.close()
    if not out.stops and not out.errors:
        out.errors.append(RowError(header_at + 2, "-", "the sheet has a header but no stops"))
    out.stops.sort(key=lambda s: s["sequence"])
    return out


def template(rows: int = 12) -> bytes:
    """A workbook shaped the way the importer likes, with a worked example in it."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Stops"

    ws.append(["block_id: BLOCK-001"])
    ws.append(["date: 2026-06-01"])
    ws.append(["vehicle_type: Large Van"])
    ws.append(["depot_id: D01"])
    ws.append(["start_time: 07:30"])
    ws.append([])
    head = ["sequence", "address", "postcode", "lat", "lon", "point_type",
            "parcel_size", "promised_from", "promised_to"]
    ws.append(head)

    example = [
        (1, "12 High St", "LS1 4AB", 53.8010, -1.5470, "House", "Medium", "08:00", "12:00"),
        (2, "40 Church Ln", "LS1 5CD", 53.8042, -1.5398, "Flat", "Small", "08:00", "12:00"),
        (3, "7 Mill Rd", "LS2 1EF", 53.8089, -1.5321, "Business", "Large", "12:00", "18:00"),
    ]
    for e in example:
        ws.append(list(e))
    for i in range(len(example) + 1, rows + 1):
        ws.append([i, None, None, None, None, None, None, None, None])

    for col, width in zip("ABCDEFGHI", (10, 26, 12, 12, 12, 16, 14, 16, 14)):
        ws.column_dimensions[col].width = width

    notes = wb.create_sheet("How to fill this in")
    for line in [
        ("What this sheet is for",),
        ("One row per stop, in the order the vehicle will visit them.",),
        ("",),
        ("Required", "sequence, lat, lon"),
        ("Optional", "address, postcode, point_type, parcel_size, "
                     "promised_from, promised_to"),
        ("",),
        ("point_type", ", ".join(POINT_TYPES)),
        ("parcel_size", ", ".join(PARCEL_SIZES)),
        ("times", "09:30, or a real Excel time cell"),
        ("",),
        ("Column names", "matched loosely: 'Stop No', 'stop_no' and 'sequence' all work"),
        ("Block details", "the 'key: value' lines above the header; they can also be "
                          "typed into the form after upload"),
        ("",),
        ("Nothing is saved", "until every row parses. Problems are listed with their "
                             "row number so they can be fixed in place."),
    ]:
        notes.append(list(line))
    notes.column_dimensions["A"].width = 18
    notes.column_dimensions["B"].width = 78

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
