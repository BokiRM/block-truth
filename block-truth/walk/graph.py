"""
The block as a graph, and two ways to walk it.

A walked block is not a list of places to visit. It is a set of *streets* that
all have to be covered, and the walker ends up back at the van. That is the
Rural Postman Problem - cover a required subset of the edges of a graph at the
least cost - and it is a different problem from the travelling salesman one that
a driven round poses. Routing a walk as a tour of its delivery points produces
nonsense: it has no idea that one pass down a terraced street serves both sides,
or that a cul-de-sac must be walked in and back out again.

Two route builders live here and they are deliberately different:

* `habitual_route` is how the rounds in the mock history are actually walked:
  greedy, from the van, nearest uncovered street next. It is what people do, and
  it is not optimal. The generated history has to contain real inefficiency or
  an optimiser has nothing to find and we would be marking our own homework.

* `rural_postman` is the optimiser the app offers. Frederickson's heuristic:
  join the required edges into one component, make every vertex even by
  duplicating shortest paths between odd ones, then take an Euler circuit. The
  odd vertices are paired greedily rather than by minimum-weight matching -
  Blossom is the exact answer and is not worth its complexity here - so the
  result is a good route and not a proven optimum, which is what the app says.

Distances are metres on a local flat projection. Over a block - a kilometre or
two - the error from pretending the Earth is flat is a few centimetres, far
below the GPS noise this all sits on.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

M_PER_DEG_LAT = 110574.0


def m_per_deg_lon(lat: float) -> float:
    return 111320.0 * math.cos(math.radians(lat))


def to_m(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    return ((lon - lon0) * m_per_deg_lon(lat0), (lat - lat0) * M_PER_DEG_LAT)


def to_ll(x: float, y: float, lat0: float, lon0: float) -> tuple[float, float]:
    return (lat0 + y / M_PER_DEG_LAT, lon0 + x / m_per_deg_lon(lat0))


def dist_m(lat1, lon1, lat2, lon2, lat0=None, lon0=None) -> float:
    lat0 = lat0 if lat0 is not None else (lat1 + lat2) / 2
    dx = (lon2 - lon1) * m_per_deg_lon(lat0)
    dy = (lat2 - lat1) * M_PER_DEG_LAT
    return math.hypot(dx, dy)


def point_segment_distance_m(plat, plon, alat, alon, blat, blon) -> tuple[float, float]:
    """Distance from a point to a segment, and how far along the segment it falls.

    The second number is the fraction 0..1 of the way from a to b, which is what
    turns a ping into a position on a street rather than just a distance from it.
    """
    lat0 = (alat + blat) / 2
    ax, ay = to_m(alat, alon, lat0, alon)
    bx, by = to_m(blat, blon, lat0, alon)
    px, py = to_m(plat, plon, lat0, alon)
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1e-9:
        return math.hypot(px - ax, py - ay), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy), t


def offset_point(alat, alon, blat, blon, t: float, offset_m: float):
    """A point `offset_m` to the side of a segment, `t` of the way along it.

    This is where a delivery point sits: off the road, at the end of a path.
    """
    lat0 = (alat + blat) / 2
    ax, ay = to_m(alat, alon, lat0, alon)
    bx, by = to_m(blat, blon, lat0, alon)
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L               # unit normal
    x, y = ax + t * dx + nx * offset_m, ay + t * dy + ny * offset_m
    return to_ll(x, y, lat0, alon)


# ---------------------------------------------------------------- the graph

@dataclass
class Edge:
    key: str
    a: str                      # node id
    b: str                      # node id
    length_m: float
    required: bool = True       # has delivery points on it
    data: dict = field(default_factory=dict)


@dataclass
class Graph:
    nodes: dict[str, tuple[float, float]] = field(default_factory=dict)   # id -> lat,lon
    edges: dict[str, Edge] = field(default_factory=dict)
    _adj: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # node -> (other, key)

    def add_node(self, node_id: str, lat: float, lon: float) -> str:
        self.nodes[node_id] = (lat, lon)
        self._adj.setdefault(node_id, [])
        return node_id

    def add_edge(self, e: Edge) -> None:
        self.edges[e.key] = e
        self._adj.setdefault(e.a, []).append((e.b, e.key))
        self._adj.setdefault(e.b, []).append((e.a, e.key))

    def neighbours(self, node: str):
        return self._adj.get(node, [])

    def degree(self, node: str) -> int:
        return len(self._adj.get(node, []))

    # ------------------------------------------------------------- distances
    def dijkstra(self, source: str) -> tuple[dict[str, float], dict[str, tuple[str, str]]]:
        dist = {source: 0.0}
        prev: dict[str, tuple[str, str]] = {}
        pq = [(0.0, source)]
        seen = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            for v, key in self.neighbours(u):
                nd = d + self.edges[key].length_m
                if nd < dist.get(v, math.inf) - 1e-9:
                    dist[v] = nd
                    prev[v] = (u, key)
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    def path(self, a: str, b: str) -> list[str]:
        """Edge keys along the shortest path from a to b."""
        if a == b:
            return []
        _, prev = self.dijkstra(a)
        if b not in prev:
            return []
        out, cur = [], b
        while cur != a:
            cur, key = prev[cur]
            out.append(key)
        out.reverse()
        return out

    def other_end(self, key: str, node: str) -> str:
        e = self.edges[key]
        return e.b if e.a == node else e.a


@dataclass
class Step:
    """One traversal of one segment, in the order it is walked."""
    key: str
    from_node: str
    to_node: str
    serviced: bool          # are the doors on this segment done on this pass?
    pass_no: int


def _walk_steps(g: Graph, keys: list[str], start: str,
                served: set[str], counter: dict[str, int]) -> tuple[list[Step], str]:
    """Turn a list of edge keys into steps, following the nodes along."""
    steps, cur = [], start
    for key in keys:
        e = g.edges[key]
        nxt = g.other_end(key, cur)
        counter[key] = counter.get(key, 0) + 1
        first_time = e.required and key not in served
        if first_time:
            served.add(key)
        steps.append(Step(key, cur, nxt, first_time, counter[key]))
        cur = nxt
    return steps, cur


def habitual_route(g: Graph, start: str, rng) -> list[Step]:
    """How the block actually gets walked: greedily, and not optimally.

    From where the van is parked, walk to the nearest street that still has mail
    on it, do it, and repeat. Where two are equally close the walker's own habit
    decides, which is the `rng` - two walkers on the same block take slightly
    different routes, and neither takes the best one.
    """
    required = [k for k, e in g.edges.items() if e.required]
    left = set(required)
    served: set[str] = set()
    counter: dict[str, int] = {}
    steps: list[Step] = []
    cur = start
    while left:
        dist, prev = g.dijkstra(cur)
        best, best_cost = None, math.inf
        for key in left:
            e = g.edges[key]
            for end in (e.a, e.b):
                c = dist.get(end, math.inf) + rng.uniform(0, 12)   # habit, not optimality
                if c < best_cost:
                    best_cost, best = c, (key, end)
        if best is None:
            break
        key, entry = best
        # walk to the start of that street...
        if entry != cur:
            hop = []
            node = entry
            while node != cur:
                node, k = prev[node]
                hop.append(k)
            hop.reverse()
            got, cur = _walk_steps(g, hop, cur, served, counter)
            steps += got
        # ...then down it
        got, cur = _walk_steps(g, [key], cur, served, counter)
        steps += got
        left.discard(key)
    # and back to the van
    home = g.path(cur, start)
    got, cur = _walk_steps(g, home, cur, served, counter)
    return steps + got


# ------------------------------------------------------------ the optimiser

def _components(g: Graph, keys: list[str]) -> list[set[str]]:
    """Connected components of the sub-graph made of these edges, as node sets."""
    adj: dict[str, set[str]] = {}
    for k in keys:
        e = g.edges[k]
        adj.setdefault(e.a, set()).add(e.b)
        adj.setdefault(e.b, set()).add(e.a)
    seen, out = set(), []
    for n in adj:
        if n in seen:
            continue
        stack, comp = [n], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack += [v for v in adj.get(u, ()) if v not in comp]
        seen |= comp
        out.append(comp)
    return out


def rural_postman(g: Graph, start: str) -> list[Step]:
    """Cover every street that has mail on it, at as little extra walking as possible.

    Frederickson's heuristic, in four moves:

    1. take the streets that must be covered;
    2. join their components together with shortest paths, cheapest first, so
       the thing can be walked in one piece;
    3. make every junction even by duplicating the shortest path between pairs
       of odd ones - a graph can only be walked in one closed circuit if every
       vertex has even degree, and an odd vertex is exactly a place the walker
       would have to arrive at once more than they leave;
    4. take an Euler circuit of the result (Hierholzer).

    Step 3 pairs the odd vertices greedily, nearest first. The exact answer is a
    minimum-weight perfect matching - Blossom - and on a block with a dozen odd
    junctions the greedy pairing is typically within a few per cent of it. The
    app says "a good route", never "the best route", for exactly this reason.
    """
    required = [k for k, e in g.edges.items() if e.required]
    if not required:
        return []

    # multigraph: how many times each edge is in the walk
    mult: dict[str, int] = {k: 1 for k in required}

    def add_path(keys: list[str]):
        for k in keys:
            mult[k] = mult.get(k, 0) + 1

    # 2. join the components
    comps = _components(g, required)
    if start not in {n for c in comps for n in c}:
        # the van is parked off the delivered streets: walk in from it
        nearest = min((n for c in comps for n in c),
                      key=lambda n: len(g.path(start, n)) or 1e9)
        add_path(g.path(start, nearest))
        comps = _components(g, list(mult))
    while len(comps) > 1:
        best = None
        base = comps[0]
        for other in comps[1:]:
            for a in base:
                dist, _ = g.dijkstra(a)
                for b in other:
                    d = dist.get(b, math.inf)
                    if best is None or d < best[0]:
                        best = (d, a, b)
        if best is None or best[0] is math.inf:
            break
        add_path(g.path(best[1], best[2]))
        comps = _components(g, list(mult))

    # 3. even out the odd vertices
    def odd_nodes() -> list[str]:
        deg: dict[str, int] = {}
        for k, m in mult.items():
            e = g.edges[k]
            deg[e.a] = deg.get(e.a, 0) + m
            deg[e.b] = deg.get(e.b, 0) + m
        return [n for n, d in deg.items() if d % 2 == 1]

    odds = odd_nodes()
    guard = 0
    while odds and guard < 200:
        guard += 1
        a = odds[0]
        dist, _ = g.dijkstra(a)
        b = min((n for n in odds[1:]), key=lambda n: dist.get(n, math.inf), default=None)
        if b is None:
            break
        add_path(g.path(a, b))
        odds = odd_nodes()

    # 4. Hierholzer, from the van
    adj: dict[str, list[str]] = {}
    for k, m in mult.items():
        e = g.edges[k]
        for _ in range(m):
            adj.setdefault(e.a, []).append(k)
            adj.setdefault(e.b, []).append(k)
    used: dict[str, int] = {}

    def take(node: str) -> str | None:
        for i, k in enumerate(adj.get(node, [])):
            if used.get(k, 0) < mult.get(k, 0):
                # is this copy still free?
                adj[node].pop(i)
                return k
        return None

    begin = start if start in adj else next(iter(adj))
    stack = [(begin, None)]
    circuit: list[tuple[str, str]] = []       # (node arrived at, edge used)
    while stack:
        node, how = stack[-1]
        k = take(node)
        if k is None:
            stack.pop()
            circuit.append((node, how))
            continue
        used[k] = used.get(k, 0) + 1
        nxt = g.other_end(k, node)
        # drop the matching copy from the other end's list
        if k in adj.get(nxt, []):
            adj[nxt].remove(k)
        stack.append((nxt, k))
    circuit.reverse()

    served: set[str] = set()
    counter: dict[str, int] = {}
    steps: list[Step] = []
    cur = circuit[0][0] if circuit else begin
    for node, key in circuit[1:]:
        if key is None:
            continue
        e = g.edges[key]
        counter[key] = counter.get(key, 0) + 1
        first_time = e.required and key not in served
        if first_time:
            served.add(key)
        steps.append(Step(key, cur, node, first_time, counter[key]))
        cur = node
    return steps


def route_length_m(g: Graph, steps: list[Step]) -> float:
    return sum(g.edges[s.key].length_m for s in steps)


def deadhead_m(g: Graph, steps: list[Step]) -> float:
    """Metres walked that deliver nothing: second passes and connecting streets."""
    return sum(g.edges[s.key].length_m for s in steps if not s.serviced)


def covers_everything(g: Graph, steps: list[Step]) -> bool:
    need = {k for k, e in g.edges.items() if e.required}
    return need <= {s.key for s in steps if s.serviced}
