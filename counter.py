"""Ball-on-board counting core: no UI, no networking. Feed it frames, get events.

Per frame:
  1. identify()     which unlabeled markers are the board (centre + 315° and 45° rim markers,
                    plus the north rim marker in between) and which is the ball, from the
                    board's shape alone, so marker IDs changing mid-trial don't matter
  2. to_map()       project the ball onto the board, in the board map's frame (cm)
  3. Counter.update gate crossings and edge hits

The geometry mirrors MATCH/analysis/board.py (board_axes, to_board, board_to_lab).
World positions are in mm; board coordinates in cm, north = away from the participant.
"""
import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from itertools import combinations

import numpy as np

R_BOARD = 39.0                  # cm, centre -> rim marker
GATES = (0, 90, 180, 270)       # compass degrees on the board map
ROT = 135                       # board axes (u toward the 315° marker) -> map frame, degrees
NORTH_TOL_MM = 10.0             # north marker match: rigid, so ~1 mm off; the ball can't get this close to it


@dataclass
class Settings:
    """Scoring rules and detection thresholds, kept in settings.json next to the app."""
    gate_points: float = 1.0
    edge_penalty: float = 0.5
    gate_r_min: float = 20.0        # cm; a crossing only counts inside the white band...
    gate_r_max: float = 30.0        # ...between these two radii
    same_gate_twice: bool = False   # False: rocking back and forth over one gate counts once
    edge_hit_r: float = 37.0        # cm from the centre that counts as hitting the edge
    edge_rearm_r: float = 35.0      # the ball must come back inside this before the next hit
    centre_rim_mm: float = 390.0    # board marker triangle: centre <-> rim marker
    rim_rim_mm: float = 556.0       # 315° rim marker <-> 45° rim marker
    north_nw_mm: float = 315.0      # north rim marker <-> 315° marker (centre <-> north: centre_rim_mm)
    north_ne_mm: float = 285.0      # north rim marker <-> 45° marker
    marker_tol_mm: float = 15.0     # how far the board distances may deviate from the above
    max_gap_s: float = 0.5          # ball lost for longer than this: don't bridge the gap

    @classmethod
    def load(cls, path):
        """Read settings (unknown keys ignored), then write the file back so it lists every key."""
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            saved = {}
        s = cls(**{f.name: saved[f.name] for f in fields(cls) if f.name in saved})
        try:
            path.write_text(json.dumps(asdict(s), indent=2))
        except OSError:
            pass
        return s


def identify(markers, s):
    """(k,3) unlabeled marker positions (mm) -> (centre, ne, nw, ball) positions, or None.

    Board: a triangle whose sides match centre-rim, centre-rim, rim-rim (the 315° and 45°
    markers). The rim markers are told apart by turning direction: seen from above, going
    from the 45° marker to the 315° marker around the centre is anticlockwise. "Above" is
    the room axis the board's normal points along most (the board is roughly level), so
    this works with Z-up CSV exports and Y-up streams alike.
    North marker: the 4th board marker, on the rim between the 315° and 45° markers, at
    north_nw_mm / north_ne_mm from them. The ball near the rim can mimic a rim marker and
    form a second plausible triangle; only the real 315°/45° pair has the north marker
    next to it, so it decides between them. Every triangle is scored on all its distances
    plus the north match (capped, so a hidden north marker or a 3-marker board still work).
    The north marker lies on the board too and is never taken for the ball.
    Ball: the leftover marker lying on the board, if any (else ball is None).
    """
    m = markers[~np.isnan(markers).any(axis=1)]
    if len(m) < 3:
        return None
    d = np.linalg.norm(m[:, None] - m[None], axis=2)
    north_err = lambda i, ne, nw: max(abs(d[i, nw] - s.north_nw_mm), abs(d[i, ne] - s.north_ne_mm))
    best = None
    for c in range(len(m)):
        near = np.flatnonzero(np.abs(d[c] - s.centre_rim_mm) < s.marker_tol_mm)
        for a, b in combinations(near, 2):
            if abs(d[a, b] - s.rim_rim_mm) >= s.marker_tol_mm:
                continue
            n = np.cross(m[a] - m[c], m[b] - m[c])
            ne, nw = (a, b) if n[np.argmax(np.abs(n))] > 0 else (b, a)
            north = min((north_err(i, ne, nw) for i in range(len(m)) if i not in (c, a, b)), default=np.inf)
            err = (abs(d[a, b] - s.rim_rim_mm) + abs(d[c, a] - s.centre_rim_mm) + abs(d[c, b] - s.centre_rim_mm)
                   + min(north, s.marker_tol_mm))
            if best is None or err < best[0]:
                best = (err, c, ne, nw)
    if best is None:
        return None
    _, c, ne, nw = best

    # not the ball: the north marker (and any duplicate of it). Its mirror image across the
    # 315°-45° line, 16 cm from the centre, has the same two distances, hence the centre check.
    others = {i for i in set(range(len(m))) - {c, ne, nw}
              if not (north_err(i, ne, nw) < NORTH_TOL_MM and d[i, c] > s.centre_rim_mm / 2)}

    # ball: nearest-to-the-board-face marker inside the rim (hands on the handles are outside)
    u, v, nrm = board_axes(m[c], m[ne], m[nw])
    ball, best_h = None, 80.0                                  # mm: at most this far off the face
    for i in others:
        rel = m[i] - m[c]
        h = abs(rel @ nrm)
        if np.hypot(rel @ u, rel @ v) < (R_BOARD + 2) * 10 and h < best_h:
            ball, best_h = m[i], h
    return m[c], m[ne], m[nw], ball


def board_axes(centre, ne, nw):
    """Orthonormal board frame: u toward the 315° marker, v in-plane, n out of the face."""
    u = (nw - centre) / np.linalg.norm(nw - centre)
    n = np.cross(nw - centre, centre - ne)
    n /= np.linalg.norm(n)
    return u, np.cross(n, u), n


def to_map(p, centre, ne, nw):
    """World point (mm) -> (x east, y north) cm on the board map. Height above the board is dropped."""
    u, v, _ = board_axes(centre, ne, nw)
    bu, bv = (p - centre) @ u / 10, (p - centre) @ v / 10
    th = math.radians(ROT)
    return bu * math.cos(th) - bv * math.sin(th), bu * math.sin(th) + bv * math.cos(th)


def ball_on_map(markers, s):
    """-> ((x, y) cm or None, board found?)"""
    found = identify(markers, s)
    if found is None:
        return None, False
    centre, ne, nw, ball = found
    return (None if ball is None else to_map(ball, centre, ne, nw)), True


@dataclass
class Event:
    t: float                # seconds, source clock
    frame: int
    kind: str               # "gate" or "edge"
    delta: int = 1          # +1, or -1 for a manual correction
    gate: int = -1          # compass degrees, gates only
    direction: str = ""     # "cw" / "ccw", gates only
    manual: bool = False


def _wrap(a):
    return (a + 180) % 360 - 180


class Counter:
    """Gate crossings and edge hits from successive ball positions on the map."""

    def __init__(self, s):
        self.s = s
        self.reset()

    def reset(self):
        self.gates = self.edges = 0
        self.events = []
        self._prev = None           # (t, radius, compass angle)
        self._last_gate = None
        self._armed = True

    @property
    def score(self):
        return self.gates * self.s.gate_points - self.edges * self.s.edge_penalty

    def update(self, t, frame, xy):
        """One frame. xy = ball on the map (cm), or None if not seen. Returns new events."""
        if xy is None:
            return []
        s, new = self.s, []
        r, ang = math.hypot(*xy), math.degrees(math.atan2(xy[0], xy[1])) % 360

        # gate: the ball's angle passes the gate's angle while inside the white band.
        # |d| < 45 on both sides rules out the jump from 359° to 0° on the far side.
        if self._prev is not None and t - self._prev[0] <= s.max_gap_s:
            _, r0, a0 = self._prev
            for g in GATES:
                d0, d1 = _wrap(a0 - g), _wrap(ang - g)
                crossed = (d0 < 0) != (d1 < 0) and abs(d0) < 45 and abs(d1) < 45
                if crossed and s.gate_r_min <= (r0 + r) / 2 <= s.gate_r_max:
                    if s.same_gate_twice or g != self._last_gate:
                        self.gates += 1
                        new.append(Event(t, frame, "gate", gate=g, direction="cw" if d1 > d0 else "ccw"))
                    self._last_gate = g

        # edge: count once per visit to the rim (hysteresis between rearm and hit radius)
        if self._armed and r > s.edge_hit_r:
            self.edges += 1
            self._armed = False
            new.append(Event(t, frame, "edge"))
        elif not self._armed and r < s.edge_rearm_r:
            self._armed = True

        self._prev = (t, r, ang)
        self.events += new
        return new

    def manual(self, t, frame, kind, delta):
        """A manual correction (+1 / -1 gate or edge), logged as such."""
        if kind == "gate":
            self.gates += delta
        else:
            self.edges += delta
        ev = Event(t, frame, kind, delta=delta, manual=True)
        self.events.append(ev)
        return ev


def read_motive_csv(path):
    """Motive CSV export -> (frame numbers, times s, (N, k, 3) unlabeled marker positions mm).

    Only markers without a ":" in their name (skeleton markers are "<skeleton>:<marker>").
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    h = next(i for i, r in enumerate(rows) if r[:2] == ["Frame", "Time (Seconds)"])
    kind = next(r for r in rows[:h] if r[1:2] == ["Type"])
    name = next(r for r in rows[:h] if r[1:2] == ["Name"])
    rotpos, axis = rows[h - 1], rows[h]
    cols = {}
    for j in range(2, len(axis)):
        if kind[j] == "Marker" and rotpos[j] == "Position" and axis[j] in "XYZ" and ":" not in name[j]:
            cols.setdefault(name[j], {})[axis[j]] = j
    idx = [c[a] for c in cols.values() for a in "XYZ"]
    num = lambda x: float(x) if x else np.nan
    data = [r for r in rows[h + 1:] if r]
    frames = np.array([int(r[0]) for r in data])
    times = np.array([float(r[1]) for r in data])
    pos = np.array([[num(r[j]) if j < len(r) else np.nan for j in idx] for r in data])
    return frames, times, pos.reshape(len(data), len(cols), 3)
