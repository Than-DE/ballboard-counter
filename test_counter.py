"""Self-checks for counter.py. Run: python test_counter.py [path/to/motive_export.csv]"""
import math
import pathlib
import sys
import tempfile

import numpy as np

import counter as C

S = C.Settings()


def rot135(bu, bv):
    th = math.radians(C.ROT)
    return bu * math.cos(th) - bv * math.sin(th), bu * math.sin(th) + bv * math.cos(th)


def test_identify_any_orientation():
    """Tilted board + ball in Z-up and Y-up worlds -> correct ball position on the map."""
    rng = np.random.default_rng(1)
    for up in ("z", "y"):
        for _ in range(20):
            # board frame: u toward the 315° marker, v = n x u, n up; tilt it a little
            yaw, tilt = rng.uniform(0, 2 * np.pi), rng.uniform(-.3, .3)
            u = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            n = np.array([np.sin(tilt) * -np.sin(yaw), np.sin(tilt) * np.cos(yaw), np.cos(tilt)])
            u -= (u @ n) * n; u /= np.linalg.norm(u)
            v = np.cross(n, u)
            c = rng.uniform(-500, 500, 3) + [0, 0, 1000]
            nw, ne = c + 390 * u, c - 390 * v            # centre -> 315° is u, 45° -> centre is v
            th = np.radians(47.7)                        # north: 47.7° from the 315° toward the 45° marker
            north = c + 390 * (np.cos(th) * u - np.sin(th) * v)
            bu, bv = rng.uniform(-250, 250, 2)
            ball = c + bu * u + bv * v + 15 * n
            hand = c + 600 * u                           # a hand on a handle, outside the rim
            pts = np.array([c, north, ne, nw, ball, hand])
            if up == "y":                                # right-handed Y-up: (x, y, z) -> (x, z, -y)
                pts = pts[:, [0, 2, 1]] * [1, 1, -1]
            xy, found = C.ball_on_map((pts[:4], pts[4:][rng.permutation(2)]))
            assert found and np.allclose(xy, rot135(bu / 10, bv / 10)), (up, xy)


def test_missing():
    """A hidden centre marker -> no board; no marker on the board -> board but no ball."""
    pts = np.array([[0, 0, 0], [0, 390, 0], [275, 275, 0], [-275, 275, 0.]])
    far = np.array([[0, 0, 900.]])
    assert C.ball_on_map((pts, far)) == (None, True)
    pts[0] = np.nan
    assert C.ball_on_map((pts, far)) == (None, False)


def test_names():
    for n in ("Board:Center", "centre", "Board 2:NE", " nw ", "Board:North"):
        assert C.role(n) is not None, n
    for n in ("Board 2:Marker 001", "Unlabeled 1031", "Suhana:RUArm", None):
        assert C.role(n) is None, n


def test_csv(tmp):
    """A small export with the board labelled, an unlabeled ball and a skeleton marker."""
    names = ["Board:Center", "Board:North", "Board:NE", "Board:NW", "Unlabeled 7", "Sk:RFIN"]
    pts = [(0, 0, 0), (0, 390, 0), (275, 275, 0), (-275, 275, 0), (50, 60, 15), (0, 0, 900)]
    hdr = lambda *cells: ",".join(("", *cells))
    marker = lambda x: [x] * 3
    lines = ["Format Version,1.25",
             "",
             hdr("Type", *[t for _ in names for t in marker("Marker")]),
             hdr("Name", *[n for n in names for _ in "XYZ"]),
             hdr("ID", *["i"] * 18),
             hdr("Parent", *[""] * 18),
             hdr("", *["Position"] * 18),
             "Frame,Time (Seconds)," + ",".join("XYZ" * 6)]
    lines.append("0,0.0," + ",".join(str(c) for p in pts for c in p))
    lines.append("1,0.0166," + ",".join(str(c) for p in pts for c in p))
    p = tmp / "t.csv"
    p.write_text("\n".join(lines))
    frames, t, board, free = C.read_motive_csv(p)
    assert board.shape == (2, 4, 3) and free.shape == (2, 1, 3)
    assert (board[0, 2] == (275, 275, 0)).all() and (free[0, 0] == (50, 60, 15)).all()
    xy, found = C.ball_on_map((board[0], free[0]))
    assert found and xy is not None


def test_gates():
    c = C.Counter(S)
    for i, a in enumerate(np.linspace(10, 370, 200)):    # one full lap at r = 25 cm
        c.update(i / 60, i, (25 * math.sin(math.radians(a)), 25 * math.cos(math.radians(a))))
    assert c.gates == 4, c.gates
    assert {e.direction for e in c.events} == {"cw"}

    c = C.Counter(S)                                     # rocking over the 90° gate
    for i, a in enumerate([80, 100, 80, 100, 80]):
        c.update(i / 60, i, (25 * math.sin(math.radians(a)), 25 * math.cos(math.radians(a))))
    assert c.gates == 1, c.gates

    c = C.Counter(S)                                     # lap outside the white band: no gates
    for i, a in enumerate(np.linspace(10, 370, 200)):
        c.update(i / 60, i, (34 * math.sin(math.radians(a)), 34 * math.cos(math.radians(a))))
    assert c.gates == 0, c.gates


def test_edges():
    c = C.Counter(S)
    for i, r in enumerate([30, 37.5, 38, 36, 37.5, 30, 38, 30]):   # dip to 36 doesn't re-arm
        c.update(i / 60, i, (0, r))
    assert c.edges == 2, c.edges
    assert c.score == 0 - 2 * S.edge_penalty


def test_real_take(path):
    """Count a real labelled export (Center/North/NE/NW + unlabeled ball) and print the result."""
    frames, t, board, free = C.read_motive_csv(path)
    c = C.Counter(S)
    missing = 0
    for f, ti, b, fr in zip(frames, t, board, free):
        xy, _ = C.ball_on_map((b, fr))
        missing += xy is None
        c.update(ti, f, xy)
    print(f"{path}: {len(t)} frames, ball missing in {missing}, "
          f"gates {c.gates}, edge hits {c.edges}, score {c.score}")


if __name__ == "__main__":
    test_identify_any_orientation()
    test_missing()
    test_names()
    with tempfile.TemporaryDirectory() as d:
        test_csv(pathlib.Path(d))
    test_gates()
    test_edges()
    for p in sys.argv[1:]:
        test_real_take(p)
    print("all checks passed")
