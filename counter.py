"""Ball-on-board counting core: no UI, no networking. Feed it frames, get events.

The board is a Motive rigid body whose four markers are labelled Center, North, NE and NW
(NE = 45° rim marker, NW = 315° rim marker); the ball is the one unlabeled marker lying on
the board. A "frame" here is (board, free): board (4,3) mm in ROLES order, free (k,3) mm
unlabeled markers.

Per frame:
  1. identify()     board axes from the labelled markers, ball = leftover marker on the board
  2. to_map()       project the ball onto the board, in the board map's frame (cm)
  3. Counter.update gate crossings and edge hits

The geometry mirrors MATCH/analysis/board.py (board_axes, to_board, board_to_lab).
World positions are in mm; board coordinates in cm, north = away from the participant.
"""
import csv
import json
import math
from dataclasses import asdict, dataclass, fields

import numpy as np

R_BOARD = 39.0                  # cm, centre -> rim marker
GATES = (0, 90, 180, 270)       # compass degrees on the board map
ROT = 135                       # board axes (u toward the 315° marker) -> map frame, degrees
ROLES = {"center": 0, "centre": 0, "north": 1, "ne": 2, "nw": 3}   # marker label -> row of `board`


def role(name):
    """Marker name ("Board:NE", "ne", ...) -> row of the board array, or None if not a board marker."""
    return ROLES.get((name or "").rsplit(":", 1)[-1].strip().lower())


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


def identify(board, free):
    """-> (centre, ne, nw, ball) positions (mm), or None if a board marker needed for the axes is missing.

    Ball: the free marker lying on the board (inside the rim, nearest to its face), if any
    (else None). Hands on the handles are outside the rim.
    ponytail: needs Center, NE and NW; a hidden one means no board this frame. Rebuild it from
    North and the rigid body pose (as board.py's fill_board does) if that loses frames.
    """
    centre, _, ne, nw = board
    if np.isnan([centre, ne, nw]).any():
        return None
    u, v, nrm = board_axes(centre, ne, nw)
    ball, best_h = None, 80.0                                  # mm: at most this far off the face
    for p in free:
        if np.isnan(p).any():
            continue
        rel = p - centre
        h = abs(rel @ nrm)
        if np.hypot(rel @ u, rel @ v) < (R_BOARD + 2) * 10 and h < best_h:
            ball, best_h = p, h
    return centre, ne, nw, ball


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


def ball_on_map(frame):
    """frame = (board, free) -> ((x, y) cm or None, board found?)"""
    found = identify(*frame)
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
    """Motive CSV export -> (frame numbers, times s, (N,4,3) board markers mm, (N,k,3) free markers mm).

    Board markers are the ones labelled Center / North / NE / NW (any prefix, e.g. "Board:NE"),
    in ROLES order. Free markers are the unlabeled ones: no ":" in the name (skeleton markers
    are "<skeleton>:<marker>").
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    h = next(i for i, r in enumerate(rows) if r[:2] == ["Frame", "Time (Seconds)"])
    kind = next(r for r in rows[:h] if r[1:2] == ["Type"])
    name = next(r for r in rows[:h] if r[1:2] == ["Name"])
    rotpos, axis = rows[h - 1], rows[h]
    cols = {}
    for j in range(2, len(axis)):
        if kind[j] == "Marker" and rotpos[j] == "Position" and axis[j] in "XYZ":
            cols.setdefault(name[j], {})[axis[j]] = j
    board = {role(n): c for n, c in cols.items() if role(n) is not None}
    missing = [k for k in ("Center", "North", "NE", "NW") if ROLES[k.lower()] not in board]
    if missing:
        raise ValueError(f"board markers not found in the export: {', '.join(missing)} "
                         "(rename the board's markers in Motive's Labels pane)")
    free = [c for n, c in cols.items() if role(n) is None and ":" not in n]
    num = lambda x: float(x) if x else np.nan
    pick = lambda r, cs: [num(r[c[a]]) if c[a] < len(r) else np.nan for c in cs for a in "XYZ"]
    data = [r for r in rows[h + 1:] if r]
    frames = np.array([int(r[0]) for r in data])
    times = np.array([float(r[1]) for r in data])
    b = np.array([pick(r, [board[i] for i in range(4)]) for r in data]).reshape(len(data), 4, 3)
    f = np.array([pick(r, free) for r in data]).reshape(len(data), len(free), 3)
    return frames, times, b, f
