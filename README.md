# Ball-board counter

Counts gate crossings and edge hits live while Motive is recording. The board is a Motive
rigid body whose four markers are labelled **Center**, **North**, **NE** (45° rim) and **NW**
(315° rim) in Motive's Labels pane; the ball is the one unlabeled marker lying on the board.
The app takes the board's position from the labelled markers and projects the ball onto it,
so marker IDs changing mid-trial no longer matter. The label is matched on the last part of
the name (`Board:NE`, `NE`), upper or lower case; `Centre` also works.

Score = gates × `gate_points` − edge hits × `edge_penalty` (1 and 0.5 by default).

| File | What it is |
|---|---|
| `app.py` | the window: connect to Motive or replay a CSV, live map, counts, log |
| `counter.py` | the counting logic (no UI, no network), same maths as `MATCH/analysis/board.py` |
| `test_counter.py` | self-checks: `python test_counter.py [some_export.csv]` |
| `settings.json` | scoring rules and thresholds, created next to the app on first start |
| `build_exe.bat` | builds `dist\BallBoardCounter.exe` on Windows |
| `.github/workflows/build-exe.yml` | the same build on GitHub's Windows machines |
| `icon.png`, `make_icon.py` | the app icon (the board schematic) and the script that draws it |

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

**On a Mac** (for replaying CSVs; same steps, an `.app` instead of an exe):
`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pyinstaller pillow`, then
`.venv/bin/pyinstaller --noconfirm --windowed --name BallBoardCounter --hidden-import natnet --icon icon.png --add-data icon.png:. app.py`
and copy `dist/BallBoardCounter.app` to `/Applications`. The Mac app keeps `settings.json`,
`logs/` and `recent.txt` in `~/Documents/BallBoardCounter`.

To run from source instead: `py -m venv .venv`, `.venv\Scripts\pip install -r requirements.txt`,
`.venv\Scripts\python app.py`.

## Motive setup

In Motive's streaming settings (Motive 3: *Settings > Streaming*; Motive 2: *Data Streaming* pane):

- **Enable** NatNet streaming.
- **Local interface:** *Loopback* (127.0.0.1) when the app runs on the Motive PC (recommended),
  otherwise the PC's network address.
- **Transmission type:** *Unicast* (leave *multicast* unticked in the app), or tick both.
- Stream **labeled and unlabeled markers**, with the board's markers labelled Center, North, NE and NW. The app reads the unlabeled ones.
- Up axis: either (the app works out which way is up from the board).

A short version of this is in the app under *About* (Mac: app menu > About BallBoardCounter;
Windows: Help > About).

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

**Replay CSV…** (or the ▾ next to it, a menu of the last 10 loaded files) counts a previous trial: pick the speed (1× to 20×), then a Motive CSV export
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
| `max_gap_s` | 0.5 s | if the ball is lost longer than this, no crossing is counted across the gap |

Exports and streams from before the board markers were labelled (`Unlabeled NNNN` for all five) are
not read any more: the CSV reader says which labels are missing.

The edge thresholds are the ones to tune against manual counts. Restart the app after editing.

The geometry settings (`centre_rim_mm` and friends) are gone; an old `settings.json` with them still loads.

## Not yet tested

The live connection has only been tested against the `natnet` library's own data objects,
not against a running Motive. First check in the lab: connect while a take plays back and
confirm `board ✓ ball ✓` (the board marker names are read from Motive's model description, which
needs NatNet 4 or later, and are matched to the stream by marker number, 1-based) and that the counts match a replay of the same take's CSV export.
