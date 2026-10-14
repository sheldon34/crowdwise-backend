"""
Route recommendation for the crowd backend.

Model
  * The building is a graph: nodes are zones (node id == the MQTT zone name, e.g.
    topic wifi/congestion/lobby), edges are walkways/doors/stairs. Some nodes are exits.
  * Live people counts come from the `readings` table (latest reading per zone).
  * Normal mode:      steer people away from crowded zones (least-congested path).
  * Evacuation mode:  send each zone to an exit, balancing load across exits so
                      everyone does not pile into the same door. Blocked routes and
                      hazard zones are never used.
  * Safety defaults:  a zone with no fresh sensor data is treated as full, never empty.
                      Recommendations only change when the new route is clearly better.

Advice is per zone (not per person), which keeps the system privacy-preserving.
"""
import asyncio
import heapq
import math
import threading
from typing import Callable, Optional

from fastapi import APIRouter, Body, Query
from fastapi.responses import JSONResponse

INF = math.inf
MODES = ("normal", "evacuation")

DEFAULT_CAPACITY = 50.0     # people a zone holds comfortably
DEFAULT_EXIT_RATE = 60.0    # people per minute an exit can pass
WALK_M_PER_MIN = 80.0       # ~1.3 m/s; converts path cost <-> minutes
STALE_LOAD = 1.0            # load assumed for zones with no fresh data
CROWDED_LOAD = 0.8          # load ratio at/above which a zone counts as crowded
CALM_LOAD = 0.5             # load ratio at/below which a zone counts as calm
MAX_LOAD = 2.0              # cap so one hot zone cannot dominate costs
HYSTERESIS = 0.85           # new route must be >= 15% cheaper to replace the old one
CONGESTION_WEIGHT = {"normal": 2.0, "evacuation": 1.0}

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    name        TEXT,
    capacity    REAL    NOT NULL DEFAULT 50,
    is_exit     INTEGER NOT NULL DEFAULT 0,
    exit_rate   REAL    NOT NULL DEFAULT 60,
    hazard      INTEGER NOT NULL DEFAULT 0,
    hazard_note TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    a       TEXT NOT NULL,
    b       TEXT NOT NULL,
    length  REAL NOT NULL DEFAULT 10,
    blocked INTEGER NOT NULL DEFAULT 0,
    note    TEXT,
    UNIQUE (a, b)
);
"""

LATEST_SQL = """
SELECT r.* FROM readings r
JOIN (SELECT zone, MAX(ts) AS m FROM readings GROUP BY zone) x
  ON r.zone = x.zone AND r.ts = x.m
GROUP BY r.zone
"""

# Demo layout (seeded only when no graph exists). Replace it with PUT /api/graph.
DEMO_NODES = [  # id, name, capacity, is_exit, exit_rate
    ("lobby", "Lobby", 80, 0, 0),
    ("hall_a", "Hall A", 120, 0, 0),
    ("hall_b", "Hall B", 100, 0, 0),
    ("cafeteria", "Cafeteria", 60, 0, 0),
    ("corridor", "Main corridor", 40, 0, 0),
    ("exit_n", "North exit", 0, 1, 90),
    ("exit_s", "South exit", 0, 1, 60),
    ("exit_e", "East exit", 0, 1, 40),
]
DEMO_EDGES = [  # a, b, length (m)
    ("lobby", "corridor", 10), ("lobby", "hall_a", 15), ("lobby", "exit_n", 8),
    ("hall_a", "corridor", 12), ("hall_a", "exit_n", 14),
    ("corridor", "hall_b", 14), ("corridor", "exit_s", 12),
    ("hall_b", "cafeteria", 10), ("hall_b", "exit_s", 10),
    ("cafeteria", "exit_e", 9),
]


def err(status: int, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status)


def _num(v) -> Optional[float]:
    if v is None or v == "" or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


class Graph:
    """Immutable snapshot of the building + live state, used for one computation."""

    def __init__(self, nodes: dict, edges: list):
        self.nodes = nodes
        self.edges = edges
        self.adj = {nid: [] for nid in nodes}
        self.edge_map = {}
        for e in edges:
            a, b = e["a"], e["b"]
            if a in nodes and b in nodes:
                self.adj[a].append((b, e))
                self.adj[b].append((a, e))
                self.edge_map[(a, b)] = e
                self.edge_map[(b, a)] = e
        self.exits = [n for n, v in nodes.items() if v["is_exit"]]

    def name(self, z: str) -> str:
        return self.nodes[z]["name"] or z

    # --- cost functions: weight(edge, destination_node) -> cost or INF ---
    def weight(self, mode: str) -> Callable:
        k = CONGESTION_WEIGHT[mode]

        def w(e, v):
            if e["blocked"] or v["hazard"]:
                return INF
            return e["length"] * (1 + k * min(v["load"], MAX_LOAD) ** 2)

        return w

    @staticmethod
    def static_weight(e, v):
        if e["blocked"] or v["hazard"]:
            return INF
        return e["length"]

    def dijkstra(self, src: str, weight: Callable):
        dist, prev = {src: 0.0}, {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, INF):
                continue
            if u != src and self.nodes[u]["is_exit"]:
                continue  # exits are terminal: you leave the building there
            for v, e in self.adj[u]:
                w = weight(e, self.nodes[v])
                if w == INF:
                    continue
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = (u, e)
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    @staticmethod
    def build_path(prev: dict, src: str, dst: str) -> list:
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]][0])
        path.reverse()
        return path

    def path_cost(self, path: list, weight: Callable) -> float:
        if len(path) < 2:
            return INF
        total = 0.0
        for i, (u, v) in enumerate(zip(path, path[1:])):
            e = self.edge_map.get((u, v))
            if e is None or (i > 0 and self.nodes[u]["is_exit"]):
                return INF
            w = weight(e, self.nodes[v])
            if w == INF:
                return INF
            total += w
        return total

    def path_length(self, path: list) -> float:
        return sum(self.edge_map[(u, v)]["length"] for u, v in zip(path, path[1:]))


class Routing:
    def __init__(self, db, db_lock, stale_ms: int, current_mode: Callable, broadcast: Callable,
                 now_ms: Callable, seed_demo: bool = True):
        self.db, self.db_lock = db, db_lock
        self.stale_ms = stale_ms
        self.current_mode = current_mode
        self.broadcast = broadcast
        self.now_ms = now_ms
        self.mem_lock = threading.RLock()
        self.last = {}      # hysteresis memory: key -> last recommended path
        self.last_adv = {}  # last advice signature broadcast per zone
        with self.db_lock:
            self.db.executescript(SCHEMA)
            self.db.commit()
            empty = self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0
        if seed_demo and empty:
            self.replace_graph(
                [(i, n, c or DEFAULT_CAPACITY, x, r or DEFAULT_EXIT_RATE) for i, n, c, x, r in DEMO_NODES],
                [(*sorted((a, b)), ln) for a, b, ln in DEMO_EDGES],
            )
        self.api = APIRouter()
        self._register()

    # ---------- storage ----------
    def replace_graph(self, nodes: list, edges: list) -> None:
        with self.db_lock, self.db:
            self.db.execute("DELETE FROM edges")
            self.db.execute("DELETE FROM nodes")
            self.db.executemany(
                "INSERT INTO nodes (id, name, capacity, is_exit, exit_rate) VALUES (?, ?, ?, ?, ?)", nodes)
            self.db.executemany("INSERT OR REPLACE INTO edges (a, b, length) VALUES (?, ?, ?)", edges)
        with self.mem_lock:
            self.last.clear()
            self.last_adv.clear()

    def load_graph(self) -> Graph:
        with self.db_lock:
            nodes = {r["id"]: dict(r) for r in self.db.execute("SELECT * FROM nodes").fetchall()}
            edges = [dict(r) for r in self.db.execute("SELECT * FROM edges").fetchall()]
            latest = {r["zone"]: r for r in self.db.execute(LATEST_SQL).fetchall()}
        now = self.now_ms()
        for nid, n in nodes.items():
            n["is_exit"], n["hazard"] = bool(n["is_exit"]), bool(n["hazard"])
            cap = n["capacity"] if n["capacity"] and n["capacity"] > 0 else DEFAULT_CAPACITY
            r = latest.get(nid)
            if r is None and n["is_exit"]:      # exits normally have no sensor
                n.update(people=0.0, level="LOW", age_ms=None, stale=False, load=0.0)
            elif r is None:                      # never reported: unknown, assume full
                n.update(people=None, level="UNKNOWN", age_ms=None, stale=True, load=STALE_LOAD)
            else:
                age = now - r["ts"]
                stale = age > self.stale_ms
                n.update(people=r["people"], level=r["level"], age_ms=age, stale=stale,
                         load=STALE_LOAD if stale else r["people"] / cap)
        for e in edges:
            e["blocked"] = bool(e["blocked"])
        return Graph(nodes, edges)

    @staticmethod
    def graph_json(g: Graph) -> dict:
        return {
            "nodes": [
                {"id": n["id"], "name": n["name"], "capacity": n["capacity"], "is_exit": n["is_exit"],
                 "exit_rate": n["exit_rate"], "hazard": n["hazard"], "hazard_note": n["hazard_note"],
                 "people": n["people"], "level": n["level"], "load": round(n["load"], 2),
                 "stale": n["stale"], "age_ms": n["age_ms"]}
                for n in g.nodes.values()
            ],
            "edges": [{"id": e["id"], "a": e["a"], "b": e["b"], "length": e["length"],
                       "blocked": e["blocked"], "note": e["note"]} for e in g.edges],
        }

    # ---------- evacuation planning (load-balanced across exits) ----------
    def evac_plan(self, g: Graph, include=()):
        """Assign every evacuating zone to an exit. Busiest zones choose first; each
        exit's already-assigned people add a queueing cost, so load spreads out."""
        evac = {z for z, n in g.nodes.items() if not n["is_exit"] and self.current_mode(z)["mode"] == "evacuation"}
        evac |= {z for z in include if z in g.nodes and not g.nodes[z]["is_exit"]}
        w = g.weight("evacuation")
        assigned = {x: 0.0 for x in g.exits}

        def queue_cost(x):
            rate = g.nodes[x]["exit_rate"] or DEFAULT_EXIT_RATE
            return assigned[x] / max(rate, 1.0) * WALK_M_PER_MIN

        plan = {}
        with self.mem_lock:
            for z in sorted(evac, key=lambda z: -(g.nodes[z]["people"] or 0)):
                key = (z, "exit", "evacuation")
                dist, prev = g.dijkstra(z, w)
                best = None
                for x in g.exits:
                    if x in dist:
                        total = dist[x] + queue_cost(x)
                        if best is None or total < best[0]:
                            best = (total, x)
                if best is None:  # trapped: no open route to any exit
                    plan[z] = None
                    self.last.pop(key, None)
                    continue
                total, x = best
                path, kept = g.build_path(prev, z, x), False
                old = self.last.get(key)
                if old and old != path and old[0] == z and old[-1] in assigned:
                    c = g.path_cost(old, w)
                    if c < INF:
                        old_total = c + queue_cost(old[-1])
                        if total >= HYSTERESIS * old_total:   # not clearly better: keep old
                            path, x, total, kept = old, old[-1], old_total, True
                self.last[key] = path
                plan[z] = {"exit": x, "path": path, "cost": total, "stable": kept}
                assigned[x] += g.nodes[z]["people"] or 0.0
        return plan, assigned

    def _stabilize(self, key, path, total, g, w):
        with self.mem_lock:
            old = self.last.get(key)
            if old and old != path and old[0] == path[0]:
                c = g.path_cost(old, w)
                if c < INF and total >= HYSTERESIS * c:
                    return old, c, True
            self.last[key] = path
            return path, total, False

    # ---------- recommendations ----------
    def _go(self, g, out, path, total, mode, kept, extra_reasons=(), **extra):
        """Build a 'go' response, explaining which crowded zones the route avoids."""
        sd, sp = g.dijkstra(path[0], Graph.static_weight)
        avoids = []
        if path[-1] in sd:
            shortest = g.build_path(sp, path[0], path[-1])
            if shortest != path:
                avoids = [
                    {"zone": z, "level": g.nodes[z]["level"], "people": g.nodes[z]["people"],
                     "load": round(g.nodes[z]["load"], 2)}
                    for z in shortest[1:-1] if z not in path and g.nodes[z]["load"] >= CROWDED_LOAD
                ]
        reasons = list(extra_reasons)
        if avoids:
            reasons.append("avoids crowded " + ", ".join(f"{a['zone']} ({a['level']})" for a in avoids))
        if kept:
            reasons.append("keeping previous route (new one is not clearly better)")
        if mode == "evacuation":
            reasons.append("blocked routes and hazard zones excluded")
        return {**out, "action": "go", "to": path[-1], "path": path,
                "path_names": [g.name(z) for z in path],
                "distance_m": round(g.path_length(path)),
                "est_minutes": round(total / WALK_M_PER_MIN, 1),
                "avoids": avoids, "stable": kept, "reason": "; ".join(reasons), **extra}

    def recommend(self, g: Graph, src: str, to: Optional[str] = None, mode: Optional[str] = None, plan=None):
        if src not in g.nodes:
            raise KeyError(src)
        mode = mode or self.current_mode(src)["mode"]
        node = g.nodes[src]
        out = {"from": src, "mode": mode, "ts": self.now_ms(), "zone_level": node["level"],
               "zone_people": node["people"], "zone_stale": node["stale"]}
        if node["is_exit"]:
            return {**out, "action": "arrived", "message": "This is an exit."}

        w = g.weight(mode)
        if to is None and mode == "evacuation":
            to = "exit"

        # --- best exit ---
        if to == "exit":
            if mode == "evacuation":
                if plan is None or src not in plan[0]:
                    plan = self.evac_plan(g, include=(src,))
                routes, assigned = plan
                ent = routes.get(src)
                if ent is None:
                    return {**out, "action": "shelter", "trapped": True,
                            "message": "No open route to an exit from this zone. Stay put and signal for help; "
                                       "rescuers are prioritised for this zone."}
                x = ent["exit"]
                return self._go(g, out, ent["path"], ent["cost"], mode, ent["stable"],
                                [f"{g.name(x)} chosen; {round(assigned[x])} people assigned to it in total"],
                                exit=x, exit_assigned_people=round(assigned[x]),
                                message=f"Evacuate via {g.name(x)}.")
            dist, prev = g.dijkstra(src, w)
            cands = [x for x in g.exits if x in dist]
            if not cands:
                return {**out, "action": "stay", "message": "No open route to an exit."}
            x = min(cands, key=dist.get)
            path, total, kept = self._stabilize((src, "exit", mode), g.build_path(prev, src, x), dist[x], g, w)
            return self._go(g, out, path, total, mode, kept, message=f"Head to {g.name(path[-1])}.")

        # --- explicit destination ---
        if to is not None:
            if to not in g.nodes:
                raise KeyError(to)
            dist, prev = g.dijkstra(src, w)
            if to not in dist:
                return {**out, "action": "stay", "message": f"No open route to {g.name(to)}."}
            path, total, kept = self._stabilize((src, to, mode), g.build_path(prev, src, to), dist[to], g, w)
            return self._go(g, out, path, total, mode, kept, message=f"Go to {g.name(to)}.")

        # --- normal-mode steering: crowded zone -> nearest calm zone ---
        if node["stale"]:
            return {**out, "action": "stay", "message": "No live data for this zone, so no advice is given."}
        if node["load"] < CROWDED_LOAD and node["level"] != "HIGH":
            return {**out, "action": "stay", "message": f"{g.name(src)} is not crowded."}
        dist, prev = g.dijkstra(src, w)
        cands = [z for z, n in g.nodes.items()
                 if z != src and not n["is_exit"] and not n["hazard"] and not n["stale"]
                 and n["load"] <= CALM_LOAD and z in dist]
        if not cands:
            return {**out, "action": "stay",
                    "message": f"{g.name(src)} is crowded but no calmer zone is reachable."}
        dest = min(cands, key=dist.get)
        path, total, kept = self._stabilize((src, "steer", mode), g.build_path(prev, src, dest), dist[dest], g, w)
        d = g.nodes[path[-1]]
        return self._go(g, out, path, total, mode, kept,
                        [f"{g.name(src)} is crowded ({node['level']})"],
                        message=f"{g.name(src)} is crowded. Move to {g.name(path[-1])} "
                                f"({round(d['people'] or 0)} people).")

    def all_advice(self) -> dict:
        g = self.load_graph()
        plan = self.evac_plan(g)
        return {z: self.recommend(g, z, plan=plan) for z, n in g.nodes.items() if not n["is_exit"]}

    # ---------- rescue priorities ----------
    def rescue_priorities(self) -> dict:
        g = self.load_graph()
        zones = []
        for z, n in g.nodes.items():
            if n["is_exit"]:
                continue
            dist, _ = g.dijkstra(z, Graph.static_weight)
            trapped = not any(x in dist for x in g.exits)
            people = n["people"] or 0.0
            score, reasons = people, []
            if trapped:
                score += 1000
                reasons.append("no open route to any exit")
            if n["hazard"]:
                score += 500
                reasons.append("hazard reported" + (f": {n['hazard_note']}" if n["hazard_note"] else ""))
            if n["stale"]:
                score += 50
                reasons.append("no live sensor data, count unverified"
                               + (f" (last known {people:.0f})" if n["people"] is not None else ""))
            elif n["load"] > 1.0:
                reasons.append("over capacity")
            if not reasons and people <= 0:
                continue
            zones.append({"zone": z, "name": g.name(z), "score": round(score, 1), "people": n["people"],
                          "level": n["level"], "trapped": trapped, "hazard": n["hazard"],
                          "stale": n["stale"], "reasons": reasons})
        zones.sort(key=lambda r: -r["score"])
        for i, r in enumerate(zones, 1):
            r["rank"] = i
        return {
            "mode": self.current_mode(None)["mode"],
            "zones": zones,
            "blocked_edges": [{"id": e["id"], "a": e["a"], "b": e["b"], "note": e["note"]}
                              for e in g.edges if e["blocked"]],
            "hazard_zones": [z for z, n in g.nodes.items() if n["hazard"]],
        }

    # ---------- push notifications over SSE ----------
    def tick(self) -> None:
        for z, a in self.all_advice().items():
            sig = (a["mode"], a["action"], tuple(a.get("path") or ()))
            if self.last_adv.get(z, ("normal", "stay", ())) != sig:
                self.last_adv[z] = sig
                self.broadcast("route", a)

    async def loop(self, interval: float) -> None:
        while True:
            try:
                await asyncio.to_thread(self.tick)
            except Exception as e:  # never let the loop die
                print("route tick failed:", e)
            await asyncio.sleep(interval)

    # ---------- HTTP API ----------
    def _register(self) -> None:
        api = self.api

        @api.get("/api/graph")
        def get_graph():
            return self.graph_json(self.load_graph())

        @api.put("/api/graph")
        def put_graph(body: dict = Body(...)):
            raw_nodes, raw_edges = body.get("nodes"), body.get("edges") or []
            if not isinstance(raw_nodes, list) or not raw_nodes:
                return err(400, "nodes must be a non-empty list")
            nodes, ids = [], set()
            for n in raw_nodes:
                nid = str(n.get("id") or "").strip()[:64] if isinstance(n, dict) else ""
                if not nid:
                    return err(400, 'each node needs an "id"')
                if nid in ids:
                    return err(400, f"duplicate node id: {nid}")
                ids.add(nid)
                cap, rate = _num(n.get("capacity")), _num(n.get("exit_rate"))
                nodes.append((nid, str(n.get("name") or nid)[:100],
                              cap if cap and cap > 0 else DEFAULT_CAPACITY, 1 if n.get("is_exit") else 0,
                              rate if rate and rate > 0 else DEFAULT_EXIT_RATE))
            edges = []
            for e in raw_edges:
                a, b = (str(e.get("a", "")), str(e.get("b", ""))) if isinstance(e, dict) else ("", "")
                if a not in ids or b not in ids or a == b:
                    return err(400, f"edge must join two different known zones: {a!r} - {b!r}")
                ln = _num(e.get("length"))
                edges.append((*sorted((a, b)), ln if ln and ln > 0 else 10.0))
            self.replace_graph(nodes, edges)
            return self.graph_json(self.load_graph())

        @api.post("/api/edges/{edge_id}/block", status_code=201)
        def block_edge(edge_id: int, body: dict = Body(...)):
            blocked, note = bool(body.get("blocked", True)), body.get("note")
            with self.db_lock, self.db:
                cur = self.db.execute("UPDATE edges SET blocked = ?, note = ? WHERE id = ?",
                                      (int(blocked), note if blocked else None, edge_id))
            if cur.rowcount == 0:
                return err(404, "unknown edge")
            ev = {"type": "edge", "id": edge_id, "blocked": blocked, "note": note}
            self.broadcast("graph", ev)
            return ev

        @api.post("/api/nodes/{node_id}/hazard", status_code=201)
        def set_hazard(node_id: str, body: dict = Body(...)):
            hazard, note = bool(body.get("hazard", True)), body.get("note")
            with self.db_lock, self.db:
                cur = self.db.execute("UPDATE nodes SET hazard = ?, hazard_note = ? WHERE id = ?",
                                      (int(hazard), note if hazard else None, node_id))
            if cur.rowcount == 0:
                return err(404, "unknown zone")
            ev = {"type": "node", "id": node_id, "hazard": hazard, "note": note}
            self.broadcast("graph", ev)
            return ev

        @api.get("/api/route")
        def get_route(src: str = Query(..., alias="from"), to: Optional[str] = None, mode: Optional[str] = None):
            if mode is not None and mode not in MODES:
                return err(400, f"mode must be one of: {', '.join(MODES)}")
            try:
                return self.recommend(self.load_graph(), src, to=to, mode=mode)
            except KeyError as e:
                return err(404, f"unknown zone: {e.args[0]}")

        @api.get("/api/routes")
        def get_routes():
            return self.all_advice()

        @api.get("/api/rescue/priorities")
        def get_rescue():
            return self.rescue_priorities()
