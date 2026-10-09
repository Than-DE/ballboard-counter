"""Self-checks for counter.py. Run: python test_counter.py [path/to/motive_export.csv]"""
import math
import sys

import numpy as np

import counter as C

S = C.Settings()


def rot135(bu, bv):
    th = math.radians(C.ROT)
    return bu * math.cos(th) - bv * math.sin(th), bu * math.sin(th) + bv * math.cos(th)


def test_identify_any_orientation():
    """Shuffled, tilted board + ball in Z-up and Y-up worlds -> correct roles and map position."""
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
            bu, bv = rng.uniform(-250, 250, 2)
            ball = c + bu * u + bv * v + 15 * n
            pts = np.array([c, ne, nw, ball])
            if up == "y":                                # right-handed Y-up: (x, y, z) -> (x, z, -y)
                pts = pts[:, [0, 2, 1]] * [1, 1, -1]
            order = rng.permutation(4)
            xy, found = C.ball_on_map(pts[order], S)
            assert found and np.allclose(xy, rot135(bu / 10, bv / 10)), (up, xy)


def test_north_marker():
    """4-marker board: the north rim marker is never taken for the ball, even with the centre hidden."""
    c, u, v, up = np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    th = np.radians(47.7)                                # north: 47.7° from the 315° toward the 45° marker
    nw, ne, north = c + 390 * u, c - 390 * v, c + 390 * (np.cos(th) * u - np.sin(th) * v)
    ball = c + 200 * u + 150 * v + 15 * up
    xy, found = C.ball_on_map(np.array([north, ne, ball, c, nw]), S)
    assert found and np.allclose(xy, rot135(20, 15)), xy
    # start of a trial: the ball sits on the centre marker and hides it
    xy, _ = C.ball_on_map(np.array([north, ne, c + 5 * u + 25 * up, nw]), S)
    assert xy is None or math.hypot(*xy) < 5, xy


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
    """Same ball position as MATCH/analysis/board.py for frame 10, then count the trial."""
    frames, t, pos = C.read_motive_csv(path)
    xy, _ = C.ball_on_map(pos[10], S)
    assert np.allclose(xy, (-5.358, 4.830), atol=.01), xy
    c = C.Counter(S)
    missing = 0
    for f, ti, m in zip(frames, t, pos):
        xy, _ = C.ball_on_map(m, S)
        missing += xy is None
        c.update(ti, f, xy)
    print(f"{path}: {len(t)} frames, ball missing in {missing}, "
          f"gates {c.gates}, edge hits {c.edges}, score {c.score}")


if __name__ == "__main__":
    test_identify_any_orientation()
    test_north_marker()
    test_gates()
    test_edges()
    for p in sys.argv[1:]:
        test_real_take(p)
    print("all checks passed")
