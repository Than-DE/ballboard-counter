# Ball-board counter

Counts gate crossings and edge hits live while Motive is recording, from the board and
ball markers alone. No rigid body, no 3D model: every frame it finds the board triangle
among the unlabeled markers and treats the leftover marker on the board as the ball.

Score = gates × `gate_points` − edge hits × `edge_penalty` (1 and 0.5 by default).

| File | What it is |
|---|---|
| `app.py` | the window: connect to Motive or replay a CSV, live map, counts, log |
| `counter.py` | the counting logic (no UI, no network), same maths as `MATCH/analysis/board.py` |
| `test_counter.py` | self-checks: `python test_counter.py [some_export.csv]` |
| `settings.json` | scoring rules and thresholds, created next to the app on first start |
| `build_exe.bat` | builds `dist\BallBoardCounter.exe` on Windows |
| `.github/workflows/build-exe.yml` | the same build on GitHub's Windows machines |

## Getting the .exe

A Windows `.exe` has to be built on Windows (PyInstaller can't cross-compile from a Mac).
Two ways:

1. **GitHub releases (nothing to install anywhere):** tag a version and push the tag,
   e.g. `git tag v0.2.0 && git push origin v0.2.0`. GitHub builds the exe on Windows and
   publishes it as a Release (repository page > *Releases*). *Actions > build-exe > Run
   workflow* builds without a release; the exe is then under the run's *Artifacts*.
2. **Any Windows PC, once:** install Python 3.12 from python.org (tick *Add python.exe to
   PATH*; no pyenv needed, the `py` launcher comes with it), copy this folder over and
   double-click `build_exe.bat`. The exe lands in `dist\`.

The Motive PC then only needs `BallBoardCounter.exe`. It creates `settings.json` and a
`logs\` folder next to itself, so put it in a folder you can write to (not Program Files).

To run from source instead: `py -m venv .venv`, `.venv\Scripts\pip install -r requirements.txt`,
`.venv\Scripts\python app.py`.

## Motive setup

In Motive's streaming settings (Motive 3: *Settings > Streaming*; Motive 2: *Data Streaming* pane):

- **Enable** NatNet streaming.
- **Local interface:** *Loopback* (127.0.0.1) when the app runs on the Motive PC (recommended),
  otherwise the PC's network address.
- **Transmission type:** *Unicast* (leave *multicast* unticked in the app), or tick both.
- Stream **labeled and unlabeled markers**. The app reads the unlabeled ones.
- Up axis: either (the app works out which way is up from the board).

Motive streams while live and while playing back a take, so a recorded take is a good
way to test the connection.

## In the lab

1. Start the app, keep both IPs at `127.0.0.1`, press **Connect**. The status line should show
   ~the capture rate in Hz, `board ✓` and `ball ✓`. Windows may ask to allow network access: allow.
2. Type a trial name, press **Start** when the trial begins and **Stop** when it ends.
3. **Manual correction** buttons fix a missed or wrong count; they're logged as manual.

Every event is written immediately to `logs\<trial>_live_<date>_<time>.csv` (time, Motive frame
number, gate and direction, running totals). The frame number lines events up with the
recorded take.

**Replay CSV…** counts a previous trial: pick the speed (1× to 20×), then a Motive CSV export
(in Motive: export the take's tracking data as CSV, with markers). The whole file is counted as
one trial named after the file, and its log is saved like a live one. A timeline above the board
shows how far the replay is, with a tick per counted gate (green, top) and edge hit (red, bottom);
**Pause** / **Resume** stops and continues the playback without losing the count. **Click the
timeline** to jump to that moment: the counts are redone from the start of the recording up to
there, so they always match a straight play; manual corrections before that point are kept. The speed can be changed
while it plays.

The coloured banner under the toolbar always says what is on screen: **● LIVE** (red, data from
Motive), **▶ REPLAY** (blue, a recorded file) or **NOT CONNECTED** (grey). While live, replay and
speed are disabled so a stray click can't cut off a recording; while replaying, the connection
settings are disabled. Log files are named `<trial>_live_<date>_<time>.csv` or
`<trial>_replay_<date>_<time>.csv`. Use it to check the counts
against your manual ones without Motive.

## Rules and thresholds (`settings.json`)

Defaults, to confirm against the task's rules and your manual counts:

| Setting | Default | Meaning |
|---|---|---|
| `gate_r_min`, `gate_r_max` | 20, 30 cm | a crossing only counts inside the white band |
| `same_gate_twice` | false | rocking back and forth over one gate counts once; the next gate must be a different one |
| (direction) | both | both directions count; the log records cw/ccw |
| `edge_hit_r` | 37 cm | the ball counts as hitting the edge beyond this radius (the measured maximum in the practice trial was 38.4 cm) |
| `edge_rearm_r` | 35 cm | the ball must come back inside this before the next edge hit counts |
| `centre_rim_mm`, `rim_rim_mm`, `marker_tol_mm` | 390, 556, 30 mm | the board marker triangle and how exact it must be |
| `max_gap_s` | 0.5 s | if the ball is lost longer than this, no crossing is counted across the gap |

On `BallBoard_Marc_C_S1_T01_Practise.csv` the defaults give 78 gates and 19 edge hits (score 68.5).
The edge thresholds are the ones to tune against manual counts. Restart the app after editing.

## Not yet tested

The live connection has only been tested against the `natnet` library's own data objects,
not against a running Motive. First check in the lab: connect while a take plays back and
confirm `board ✓ ball ✓` and that the counts match a replay of the same take's CSV export.
