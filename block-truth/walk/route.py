"""
Re-walking a block in a better order, and saying what that is worth.

The block as walked is on record: which segments, in which order, with which
passes serviced. The question this answers is whether a better order exists and
what it would save - in metres, in minutes, and in the only unit a delivery
office actually spends, which is a walker's time.

Why this is not the routing in the driven half of the app
---------------------------------------------------------

A van visits points. A postie covers streets. Asking for the shortest tour of
the delivery points on a walked block produces a route that crosses the road a
hundred times and never once walks a terrace down and back, because a tour of
points has no idea that one pass of a street serves everything on both sides of
it. The right problem is the Rural Postman: cover every *edge* that has mail on
it, at least total length, returning to the van.

`graph.rural_postman` is Frederickson's heuristic for it. What it buys over the
way blocks are actually walked is measured here rather than asserted, on the
block's own geometry, and converted into time through the fitted pace so that
"420 metres less" becomes "six minutes a day, twenty-six hours a year".
"""
from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field

from .graph import (Edge, Graph, covers_everything, deadhead_m, habitual_route,
                    route_length_m, rural_postman)
from .model import WalkModel, walking_seconds
from .schema import connect


@dataclass
class RouteOption:
    key: str
    name: str
    why: str
    gives_up: str
    serviced_m: float
    deadhead_m: float
    total_m: float
    passes: int
    minutes: float
    covers_all: bool
    order: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"key": self.key, "name": self.name, "why": self.why,
                "gives_up": self.gives_up,
                "serviced_m": round(self.serviced_m), "deadhead_m": round(self.deadhead_m),
                "total_m": round(self.total_m), "passes": self.passes,
                "minutes": round(self.minutes, 1), "covers_all": self.covers_all,
                "deadhead_share": round(self.deadhead_m / max(self.total_m, 1), 3)}


def load_graph(con, block_id: str) -> tuple[Graph, str]:
    """The block's streets, as a graph, with the van where the first one starts."""
    g = Graph()
    rows = con.execute(
        "SELECT s.*, (SELECT count(*) FROM dp_segment_join j "
        "             WHERE j.segment_id = s.segment_id) dps "
        "FROM osm_segments s WHERE s.block_id=? ORDER BY s.osm_way_id, s.way_seq",
        (block_id,)).fetchall()
    if not rows:
        raise LookupError(f"no segments for block {block_id}")
    for r in rows:
        g.add_node(r["from_node"], r["from_lat"], r["from_lon"])
        g.add_node(r["to_node"], r["to_lat"], r["to_lon"])
    for r in rows:
        g.add_edge(Edge(r["segment_id"], r["from_node"], r["to_node"],
                        r["length_m"], required=r["dps"] > 0,
                        data={"dps": r["dps"]}))
    return g, rows[0]["from_node"]


def as_walked(con, block_id: str) -> dict | None:
    """What the records say the route actually is, averaged over the days of it."""
    r = con.execute(
        "SELECT count(DISTINCT t.block_day_id) days, "
        "       sum(CASE WHEN t.serviced=1 THEN s.length_m ELSE 0 END) serv, "
        "       sum(CASE WHEN t.serviced=0 THEN s.length_m ELSE 0 END) dead, "
        "       count(*) passes FROM segment_traversals t "
        "JOIN osm_segments s USING(segment_id) WHERE s.block_id=? AND t.source='simulated'",
        (block_id,)).fetchone()
    if not r or not r["days"]:
        return None
    return {"serviced_m": r["serv"] / r["days"], "deadhead_m": r["dead"] / r["days"],
            "passes": r["passes"] / r["days"], "days": r["days"]}


def options(con, block_id: str, m: WalkModel, area_type: str | None = None,
            rng_seed: int = 7) -> list[RouteOption]:
    """Ways to walk this block, costed in minutes through the fitted pace."""
    import random

    g, start = load_graph(con, block_id)
    if area_type is None:
        row = con.execute("SELECT area_type FROM blocks WHERE block_id=?",
                          (block_id,)).fetchone()
        area_type = row["area_type"] if row else "urban"
    pace = m.pace_for(area_type)
    bonus = m.deadhead_bonus

    def cost(serviced: float, dead: float) -> float:
        return (serviced / pace + dead / (pace * bonus)) / 60.0

    out: list[RouteOption] = []

    walked = as_walked(con, block_id)
    if walked:
        out.append(RouteOption(
            "as_walked", "As it is walked now",
            "The route on record, averaged over the days it has been walked.",
            "Nothing is assumed about it - this is the thing to beat.",
            walked["serviced_m"], walked["deadhead_m"],
            walked["serviced_m"] + walked["deadhead_m"], round(walked["passes"]),
            cost(walked["serviced_m"], walked["deadhead_m"]), True))

    steps = rural_postman(g, start)
    if steps:
        serviced = sum(g.edges[s.key].length_m for s in steps if s.serviced)
        out.append(RouteOption(
            "rural_postman", "Cover every street, least walking",
            "Join the streets with mail on them into one piece, even out the "
            "junctions by doubling the cheapest connections, then take the circuit "
            "that results. This is the Rural Postman problem, which is the one a "
            "walked block actually poses.",
            "A good route, not a proven best one, and not always even the best "
            "of the ones offered here: the junctions are paired greedily rather "
            "than by minimum-weight matching, and on a block whose odd junctions "
            "happen to pair badly the nearest-street route can come out shorter. "
            "Which is why both are costed and the shorter one is what gets "
            "recommended.",
            serviced, deadhead_m(g, steps), route_length_m(g, steps), len(steps),
            cost(serviced, deadhead_m(g, steps)), covers_everything(g, steps),
            [s.key for s in steps]))

    greedy = habitual_route(g, start, random.Random(rng_seed))
    if greedy:
        serviced = sum(g.edges[s.key].length_m for s in greedy if s.serviced)
        out.append(RouteOption(
            "nearest_street", "Nearest street next",
            "From the van, always walk to the nearest street that still has mail "
            "on it. This is how rounds tend to get walked when nobody has planned "
            "them.",
            "It makes each next choice well and the whole badly: it strands "
            "streets behind it and has to come back for them.",
            serviced, deadhead_m(g, greedy), route_length_m(g, greedy), len(greedy),
            cost(serviced, deadhead_m(g, greedy)), covers_everything(g, greedy),
            [s.key for s in greedy]))

    out.sort(key=lambda o: o.minutes)
    return out


def saving(con, block_id: str, m: WalkModel, days_a_year: int = 310) -> dict:
    """What the better route is worth on this block, in minutes and in hours a year."""
    opts = options(con, block_id, m)
    if len(opts) < 2:
        return {"block_id": block_id, "note": "not enough routes to compare"}
    best = opts[0]
    now = next((o for o in opts if o.key == "as_walked"), opts[-1])
    gain_min = now.minutes - best.minutes
    return {
        "block_id": block_id,
        "best": best.key, "best_name": best.name,
        "now_minutes": round(now.minutes, 1),
        "best_minutes": round(best.minutes, 1),
        "minutes_a_day": round(gain_min, 1),
        "metres_a_day": round(now.total_m - best.total_m),
        "hours_a_year": round(gain_min * days_a_year / 60.0, 1),
        "deadhead_now": round(now.deadhead_m),
        "deadhead_best": round(best.deadhead_m),
        "covers_all": best.covers_all,
        "options": [o.as_dict() for o in opts],
    }


def fleet_saving(con, m: WalkModel, limit: int = 200,
                 days_a_year: int = 310) -> dict:
    """The same question across every block: where is the walking being wasted?"""
    blocks = con.execute(
        "SELECT block_id, do_id, area_type FROM blocks ORDER BY block_id LIMIT ?",
        (limit,)).fetchall()
    rows, failed = [], []
    for b in blocks:
        try:
            s = saving(con, b["block_id"], m, days_a_year)
        except LookupError:
            continue
        if "minutes_a_day" not in s:
            failed.append(b["block_id"])
            continue
        s.update({"do_id": b["do_id"], "area_type": b["area_type"]})
        s.pop("options", None)
        rows.append(s)
    rows.sort(key=lambda r: -r["minutes_a_day"])
    gains = [r["minutes_a_day"] for r in rows]
    return {
        "blocks": len(rows),
        "minutes_a_day_total": round(sum(gains), 1),
        "hours_a_year_total": round(sum(r["hours_a_year"] for r in rows), 1),
        "mean_minutes_a_day": round(st.fmean(gains), 2) if gains else 0,
        "blocks_where_it_helps": sum(1 for g in gains if g > 1),
        "blocks_already_good": sum(1 for g in gains if g <= 1),
        "worst_first": rows[:25],
        "could_not_route": failed,
    }
