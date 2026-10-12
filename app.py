"""Ball-board counter: live gate crossings and edge hits from Motive (NatNet), or a CSV replay.

Threads: frames arrive on a source thread (NatNet or replay), which runs the counter on
every frame and puts the results in a queue. Only the Tk main thread touches the window;
it drains the queue ~30 times a second.
"""
import csv
import datetime
import math
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, fields
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import counter as C

HERE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
if getattr(sys, "frozen", False) and sys.platform == "darwin":   # never write inside the .app bundle
    HERE = Path.home() / "Documents" / "BallBoardCounter"
    HERE.mkdir(parents=True, exist_ok=True)
ICON = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "icon.png"   # bundled by PyInstaller --add-data
SETTINGS = C.Settings.load(HERE / "settings.json")
LOGS = HERE / "logs"
RECENT = HERE / "recent.txt"            # last loaded CSVs, newest first, one path per line

PX = 250 / 45                   # canvas pixels per cm before the first resize
SIDE_W = 330                    # sidebar width in pixels at 96 dpi
TRAIL_S = 1.5
BALL, GATE_ON, EDGE_ON = "#eb6834", "#1baf7a", "#e0453a"
OK, BAD, IDLE = "#1baf7a", "#e0453a", "#9a9a9a"      # marker present / missing / no data
HEAD_BG = "#dcd9d2"
MODES = {"idle": ("#6b6b6b", "NOT CONNECTED"),       # banner colour, label
         "live": ("#c62828", "● LIVE"),
         "replay": ("#1c5cab", "▶ REPLAY")}
# the markers the app looks for: label, board row (see counter.ROLES), place on the board map (cm)
_r, _d = C.R_BOARD, math.sqrt(.5) * C.R_BOARD
BOARD_MARKERS = (("Center", 0, (0, 0)), ("North", 1, (0, _r)), ("NE", 2, (_d, _d)), ("NW", 3, (-_d, _d)))


class Section(ttk.Frame):
    """Collapsible sidebar section (like Blender's panels): click the header to fold it."""

    def __init__(self, parent, title, open=True):
        super().__init__(parent)
        self.title, self.open = title, open
        self.head = tk.Label(self, anchor="w", padx=8, pady=4, bg=HEAD_BG, fg="black", cursor="hand2",
                             font=("Segoe UI", 10, "bold"))
        self.head.pack(fill="x", pady=(8, 0))
        self.head.bind("<Button-1>", lambda _e: self.toggle())
        self.body = ttk.Frame(self, padding=(8, 6, 8, 2))
        self._sync()

    def toggle(self):
        self.open = not self.open
        self._sync()

    def _sync(self):
        self.head.configure(text=("▾  " if self.open else "▸  ") + self.title)
        if self.open:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()


# --- frame sources: call on_frame(frame, t, (k,3) mm) from their own thread ---------------

class LiveSource:
    """Frames streamed by Motive. NatNet sends metres; unlabeled markers have model ID 0.

    The board's markers arrive as labeled markers of the board rigid body, numbered 1..n in the
    order of the rigid body's marker list; their names (Center, North, NE, NW) come from
    Motive's model description (NatNet 4+), which is requested again until it is known.
    """

    def __init__(self, server, local, multicast, on_frame):
        from natnet import NatNetClient
        self.client = NatNetClient(server_ip_address=server, local_ip_address=local,
                                   use_multicast=multicast)
        self.client.on_data_frame_received_event.handlers.append(self._frame)
        self.client.on_data_description_received_event.handlers.append(self._descriptions)
        self.on_frame = on_frame
        self.board = None                       # (rigid body id, [board row per marker]) once known
        self.asked = 0.0

    def start(self):
        self.client.connect(timeout=5.0)        # raises TimeoutError if Motive doesn't answer
        self.client.run_async()
        self._ask()

    def stop(self):
        self.client.shutdown()

    def _ask(self):
        self.asked = time.monotonic()
        self.client.request_modeldef()

    def _descriptions(self, d):
        for rb in d.rigid_bodies:
            rows = [C.role(m.name) for m in rb.markers]
            if set(rows) == {0, 1, 2, 3}:
                self.board = (rb.id_num, rows)

    def _frame(self, df):
        if self.board is None and time.monotonic() - self.asked > 2:
            self._ask()
        board = np.full((4, 3), np.nan)
        if self.board is not None:
            mid, rows = self.board
            mine = [m for m in df.labeled_markers if m.model_id == mid]
            base = 0 if any(m.marker_id == 0 for m in mine) else 1          # 1-based unless a 0 shows up
            for m in mine:
                if 0 <= m.marker_id - base < len(rows):
                    board[rows[m.marker_id - base]] = m.pos
        free = [m.pos for m in df.labeled_markers if m.model_id == 0 and not (m.param or 0) & 0x01]
        if not free:                            # older NatNet versions: legacy unlabeled list
            free = list(df.unlabeled_marker_pos)
        self.on_frame(df.prefix.frame_number, df.suffix.timestamp,
                      (board * 1000, np.array(free, float).reshape(-1, 3) * 1000), df.suffix.is_recording)


class ReplaySource:
    """A Motive CSV export played back at `speed` x real time.

    Pausable, `speed` may change while playing, and seek(i) jumps to frame i. A jump
    calls on_seek(i) on the replay thread before frame i plays, so the app can recount.
    """

    def __init__(self, path, speed, on_frame, on_seek):
        self.frames, self.t, board, free = C.read_motive_csv(path)
        self.pos = list(zip(board, free))        # one (board, free) frame per row
        self.speed, self.on_frame, self.on_seek = speed, on_frame, on_seek
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._seek_to = None

    @property
    def paused(self):
        return self._paused.is_set()

    def toggle_pause(self):
        (self._paused.clear if self.paused else self._paused.set)()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()

    def seek(self, i):
        self._seek_to = int(np.clip(i, 0, len(self.t) - 1))

    def _run(self):
        i, resync = 0, True
        while i < len(self.t) and not self._stop.is_set():
            if self._seek_to is not None:
                i, self._seek_to = self._seek_to, None
                self.on_seek(i)
                resync = True                               # then show frame i, even while paused
            elif self._paused.is_set():
                time.sleep(.05)
                resync = True
                continue
            if resync or self.speed != speed:              # (re)start the clock from this frame
                speed, t0, wall0, resync = self.speed, self.t[i], time.perf_counter(), False
            lag = (self.t[i] - t0) / speed - (time.perf_counter() - wall0)
            if lag > 0:
                time.sleep(lag)
            self.on_frame(int(self.frames[i]), float(self.t[i]), self.pos[i])
            i += 1
        if not self._stop.is_set():
            self.on_frame(None, None, None)                 # end of file


# --- the app ------------------------------------------------------------------------------

class App:
    def __init__(self, root):
        self.root, self.q = root, queue.Queue()
        self.counter = C.Counter(SETTINGS)
        self.lock = threading.Lock()            # counter is used by the source thread and the UI
        self.source = None
        self.running = False
        self.log = None
        self.last = (0, 0.0)                    # latest (frame, t) seen
        self.t_start = None                     # source time of the trial's first frame
        self.trail = deque()
        self.flash = {}                         # canvas item -> monotonic time the flash ends
        self.rate = deque()                     # arrival times of recent frames, for Hz
        self.status_bits = ""
        self.mode = "idle"
        self.connecting = self.closing = False
        self.rec = None                         # Motive's recording flag, last frame (None: unknown / not live)
        self.auto = False                       # the running trial was started by Motive's recording
        self._tl_key = None
        self.dirty = True                       # board view needs re-applying (set by every redraw)
        self.xy, self.present = None, None      # latest ball on the map, and which of the 4 board markers are tracked
        self.n_free = 0
        self.ticks = []                         # (time, timeline item) per counted event, replay only
        self.tot = {"gate": 0, "edge": 0}       # running totals of the events shown
        self.replay_t = None                    # (first, last) time of the file being replayed
        root.title("Ball-board counter")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self._set_mode("idle")
        self.poll()

    # layout
    def _build(self):
        self.root.minsize(900, 600)
        self.root.geometry("1280x820")
        style = ttk.Style()
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=6)
        self.shown = {}                         # widget / canvas item -> last options set (skip no-op updates)

        # mode banner: which kind of data is on screen must be obvious at a glance
        self.banner = tk.Frame(self.root)
        self.banner.pack(fill="x")
        self.mode_label = tk.Label(self.banner, font=("Segoe UI", 12, "bold"), padx=10, pady=5)
        self.mode_label.pack(side="left")
        self.status = tk.Label(self.banner, text="", anchor="w")
        self.status.pack(side="left", fill="x", expand=True)

        self.replay_bar = ttk.Frame(self.root, padding=(8, 6, 8, 0))
        self.pause_btn = ttk.Button(self.replay_bar, text="⏸ Pause", width=10, command=self.toggle_pause)
        self.pause_btn.pack(side="left")
        self.timeline = tk.Canvas(self.replay_bar, height=28, bg="#e8e6e0", highlightthickness=0,
                                  cursor="hand2")
        self.timeline.pack(side="left", fill="x", expand=True, padx=8)
        self.tl_done = self.timeline.create_rectangle(0, 0, 0, 28, fill="#b9d3f2", width=0)
        self.tl_head = self.timeline.create_line(0, 0, 0, 28, fill="black", width=2)
        self.timeline.bind("<Button-1>", self.seek_click)
        self.tl_label = ttk.Label(self.replay_bar, text="", width=16, font=("Consolas", 10))
        self.tl_label.pack(side="left")

        mid = self.mid = ttk.Frame(self.root, padding=8)
        mid.pack(fill="both", expand=True)

        # the board takes all the room the sidebar leaves; it is redrawn to fit on every resize
        self.canvas = tk.Canvas(mid, width=400, height=400, bg="white", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self._redraw_id = None
        self.cx, self.cy, self.px = 200, 200, PX
        self._build_sidebar(mid)
        self._draw_board()
        self.root.bind("<F11>", lambda _e: self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen")))
        self.root.bind("<Escape>", lambda _e: self.root.attributes("-fullscreen", False))

    def _build_sidebar(self, parent):
        """Scrollable column of collapsible sections: Score, Trial, Markers, Connection, Replay, Manual, Events."""
        width = int(SIDE_W * self.root.winfo_fpixels("1i") / 96)
        box = ttk.Frame(parent)
        box.pack(side="right", fill="y", padx=(8, 0))
        sc = self.side_canvas = tk.Canvas(box, width=width, highlightthickness=0,
                                          bg=ttk.Style().lookup("TFrame", "background"))
        bar = ttk.Scrollbar(box, orient="vertical", command=sc.yview)
        sc.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        sc.pack(side="left", fill="y")
        inner = ttk.Frame(sc)
        sc.create_window((0, 0), window=inner, anchor="nw", width=width)
        inner.bind("<Configure>", lambda _e: sc.configure(scrollregion=sc.bbox("all")))

        def wheel(e):                           # scroll the sidebar when the pointer is over it
            w = self.root.winfo_containing(e.x_root, e.y_root)
            if w is not None and str(w).startswith(str(box)):
                sc.yview_scroll(-1 if e.delta > 0 else 1, "units")
        self.root.bind_all("<MouseWheel>", wheel)

        # score
        sec = Section(inner, "Score")
        sec.pack(fill="x")
        row = ttk.Frame(sec.body)
        row.pack(fill="x")
        self.numbers = {}
        for i, (key, label) in enumerate((("gates", "Gates"), ("edges", "Edge hits"), ("score", "Score"))):
            col = ttk.Frame(row)
            col.pack(side="left", expand=True, fill="x")
            ttk.Label(col, text=label).pack()
            self.numbers[key] = ttk.Label(col, text="0", font=("Segoe UI", 28, "bold"))
            self.numbers[key].pack()
        self.clear_btn = ttk.Button(sec.body, text="Clear score", command=self.clear)
        self.clear_btn.pack(fill="x", pady=(8, 0))

        # trial
        sec = Section(inner, "Trial")
        sec.pack(fill="x")
        ttk.Label(sec.body, text="Name (used for the log file)").pack(anchor="w")
        self.auto_name = self.now_name()        # default name; a fresh one for each trial unless you typed your own
        self.trial = tk.StringVar(value=self.auto_name)
        ttk.Entry(sec.body, textvariable=self.trial).pack(fill="x", pady=(2, 8))
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(sec.body, text="Start and stop with Motive's recording (live)",
                        variable=self.follow).pack(anchor="w", pady=(0, 2))
        self.rec_note = ttk.Label(sec.body, text="")
        self.rec_note.pack(anchor="w", pady=(0, 8))
        row = ttk.Frame(sec.body)
        row.pack(fill="x")
        self.start_btn = ttk.Button(row, text="Start", style="Big.TButton", command=self.start)
        self.start_btn.pack(side="left", expand=True, fill="x")
        self.stop_btn = ttk.Button(row, text="Stop", style="Big.TButton", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(6, 0))

        # markers: what the app looks for, and whether it sees it
        sec = Section(inner, "Markers")
        sec.pack(fill="x")
        self.marker_rows = {}
        for name in [m[0] for m in BOARD_MARKERS] + ["Ball"]:
            row = ttk.Frame(sec.body)
            row.pack(fill="x", pady=1)
            dot = tk.Label(row, text="●", fg=IDLE, font=("Segoe UI", 12))
            dot.pack(side="left")
            ttk.Label(row, text=name, width=8).pack(side="left", padx=(4, 0))
            state = ttk.Label(row, text="—")
            state.pack(side="left")
            self.marker_rows[name] = (dot, state)
        self.marker_note = ttk.Label(sec.body, text="", wraplength=width - 40)
        self.marker_note.pack(anchor="w", pady=(6, 0))

        # connection
        sec = Section(inner, "Motive connection")
        sec.pack(fill="x")
        self.server, self.local = tk.StringVar(value="127.0.0.1"), tk.StringVar(value="127.0.0.1")
        self.multicast = tk.BooleanVar(value=False)
        self.conn_widgets = []
        grid = ttk.Frame(sec.body)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)
        for r, (label, var) in enumerate((("Motive IP", self.server), ("This PC", self.local))):
            ttk.Label(grid, text=label).grid(row=r, column=0, sticky="w", pady=2)
            self.conn_widgets.append(ttk.Entry(grid, textvariable=var))
            self.conn_widgets[-1].grid(row=r, column=1, sticky="ew", padx=(8, 0), pady=2)
        self.conn_widgets.append(ttk.Checkbutton(sec.body, text="multicast", variable=self.multicast))
        self.conn_widgets[-1].pack(anchor="w", pady=2)
        self.connect_btn = ttk.Button(sec.body, text="Connect", style="Big.TButton", command=self.toggle_connect)
        self.connect_btn.pack(fill="x", pady=(4, 0))

        # replay
        sec = Section(inner, "Replay a CSV")
        sec.pack(fill="x")
        try:
            self.recent = RECENT.read_text().splitlines()
        except OSError:
            self.recent = []
        row = ttk.Frame(sec.body)
        row.pack(fill="x")
        self.replay_btn = ttk.Button(row, text="Open CSV…", command=self.toggle_replay)
        self.replay_btn.pack(side="left", expand=True, fill="x")
        self.recent_btn = ttk.Button(row, text="Recent ▾", command=self.recent_menu)
        self.recent_btn.pack(side="left", padx=(6, 0))
        row = ttk.Frame(sec.body)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="Speed").pack(side="left")
        self.speed = tk.StringVar(value="1x")
        self.speed_box = ttk.Combobox(row, textvariable=self.speed, values=["1x", "2x", "5x", "20x"],
                                      width=5, state="readonly")
        self.speed_box.pack(side="left", padx=8)
        self.speed_box.bind("<<ComboboxSelected>>", self.change_speed)

        # manual correction
        sec = Section(inner, "Manual correction")
        sec.pack(fill="x")
        self.manual_btns = []
        for r, kind in enumerate(("gate", "edge")):
            for c, delta in enumerate((+1, -1)):
                b = ttk.Button(sec.body, text=f"{delta:+d} {kind}", state="disabled",
                               command=lambda k=kind, d=delta: self.manual(k, d))
                b.grid(row=r, column=c, sticky="ew", padx=2, pady=2)
                self.manual_btns.append(b)
            sec.body.columnconfigure(r, weight=1)

        # events
        sec = Section(inner, "Events")
        sec.pack(fill="x", pady=(0, 8))
        self.events = tk.Listbox(sec.body, height=9, font=("Consolas", 10))
        self.events.pack(fill="x")

    # board canvas: everything is placed from cx, cy and px (pixels per cm), set on each resize
    def _xy(self, x, y):
        return self.cx + x * self.px, self.cy - y * self.px

    def _circle(self, r, centre=(0, 0), **kw):
        (x0, y0), (x1, y1) = self._xy(centre[0] - r, centre[1] + r), self._xy(centre[0] + r, centre[1] - r)
        return self.canvas.create_oval(x0, y0, x1, y1, **kw)

    def _on_resize(self, _event):
        if self._redraw_id:
            self.root.after_cancel(self._redraw_id)
        self._redraw_id = self.root.after(40, self._draw_board)     # once the dragging pauses

    def _draw_board(self):
        self._redraw_id = None
        c = self.canvas
        c.delete("all")
        self.flash.clear()
        w, h = max(c.winfo_width(), 100), max(c.winfo_height(), 100)
        self.cx, self.cy = w / 2, h / 2 - 8
        self.px = min(w / 2, self.cy) / 46                          # 39 cm board plus a margin
        self.rim = self._circle(C.R_BOARD, fill="#8a8477", outline="peru", width=max(2, self.px / 3))
        self._circle(SETTINGS.gate_r_max, outline="white", dash=(2, 3))
        self._circle(SETTINGS.gate_r_min, outline="white", dash=(2, 3))
        self._circle(SETTINGS.edge_hit_r, outline="#e0453a", dash=(1, 4))
        self.gate_lines = {}
        font = ("Segoe UI", max(8, int(self.px * 1.1)))
        for g in C.GATES:
            a = math.radians(g)
            p0 = self._xy(SETTINGS.gate_r_min * math.sin(a), SETTINGS.gate_r_min * math.cos(a))
            p1 = self._xy(SETTINGS.gate_r_max * math.sin(a), SETTINGS.gate_r_max * math.cos(a))
            self.gate_lines[g] = c.create_line(*p0, *p1, fill="white", width=max(3, self.px / 2))
            c.create_text(*self._xy(34 * math.sin(a), 34 * math.cos(a)), text=f"{g}°", fill="white", font=font)
        c.create_text(w / 2, h - 12, text="participant", fill="black", font=font)
        # the four board markers at their places on the rim, coloured by whether they are tracked
        self.marker_items = {name: self._circle(1.3, centre=pos, fill=IDLE, outline="black")
                             for name, _, pos in BOARD_MARKERS}
        self.trail_line = c.create_line(0, 0, 0, 0, fill=BALL, width=max(2, self.px / 2.5), state="hidden")
        self.ball = self._circle(1.6, fill=BALL, outline="black", state="hidden")
        self.dirty = True                                           # poll() re-applies ball, trail, markers

    # frames (source thread)
    def on_frame(self, frame, t, markers, recording=None):
        """recording: Motive's "is recording" flag for this frame (live only), else None."""
        if frame is None:
            self.q.put(("end",))
            return
        xy, board = C.ball_on_map(markers)
        present = tuple(~np.isnan(markers[0]).any(axis=1))
        with self.lock:
            events = self.counter.update(t, frame, xy) if self.running else []
        self.q.put(("frame", frame, t, xy, present, len(markers[1]), events, recording))

    def on_seek(self, i):
        """Replay jumped to frame i: recount frames 0..i-1 so the totals match a straight play.

        Manual corrections made before that point are kept. Runs on the replay thread.
        """
        src = self.source
        with self.lock:
            if self.running:
                manual = [e for e in self.counter.events if e.manual and e.t <= src.t[i]]
                self.counter.reset()
                # ponytail: recounts from the start, ~0.3 s for 2 min; cache per-frame positions if trials get long
                for k in range(i):
                    self.counter.update(float(src.t[k]), int(src.frames[k]), C.ball_on_map(src.pos[k])[0])
                for e in manual:
                    self.counter.manual(e.t, e.frame, e.kind, e.delta)
                self.counter.events.sort(key=lambda e: e.t)
            recounted = list(self.counter.events)       # later frames' events follow via the queue
        self.q.put(("seek", recounted))

    # UI loop (main thread)
    def _set(self, target, **kw):
        """Configure a widget (or canvas item) only if the options changed: no-op updates cost a relayout."""
        key = tuple(sorted(kw.items()))
        if self.shown.get(target) != key:
            self.shown[target] = key
            if isinstance(target, int):
                self.canvas.itemconfigure(target, **kw)
            else:
                target.configure(**kw)

    def poll(self):
        latest = None
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "end":
                    self.stop()
                    self._disconnect("replay finished")
                    latest = None               # frames queued before the end must not revive the view
                    continue
                if msg[0] == "seek":
                    self._rebuild(msg[1])
                    continue
                if msg[0] == "connected":
                    self._connected(*msg[1:])
                    continue
                if msg[0] == "connect_failed":
                    self._connect_failed(msg[1])
                    continue
                _, frame, t, xy, present, n, events, rec = msg
                latest = msg
                self._follow_recording(rec)
                if self.running and self.t_start is None:
                    self.t_start = t
                self.rate.append(time.monotonic())
                if xy is not None:
                    if self.trail and t < self.trail[-1][0]:    # time went back: new CSV or source, wipe the old trail
                        self.trail.clear()
                    self.trail.append((t, xy))
                for ev in events:
                    self._show_event(ev)
        except queue.Empty:
            pass

        now = time.monotonic()
        while self.rate and now - self.rate[0] > 1:
            self.rate.popleft()
        if latest:
            _, frame, t, self.xy, self.present, self.n_free, _, _ = latest
            self.last = (frame, t)
            while self.trail and t - self.trail[0][0] > TRAIL_S:
                self.trail.popleft()
        if latest or self.dirty:
            self._draw_view()
        self._show_markers(live_gap=self.mode == "live" and not self.rate)
        replaying = isinstance(self.source, ReplaySource)
        if self.connecting:
            self._set(self.status, text="connecting…")
        elif self.source is not None:
            state = "   ⏸ PAUSED" if replaying and self.source.paused else ("   ▶ COUNTING" if self.running else "")
            self._set(self.status, text=f"{self.source_name} · {len(self.rate)} Hz · frame {self.last[0]}{state}")
        if self.replay_t is not None:
            self._draw_timeline()
        for item, until in list(self.flash.items()):
            if now > until:
                self.canvas.itemconfigure(item, **({"outline": "peru"} if item == self.rim else {"fill": "white"}))
                del self.flash[item]
        self.root.after(33, self.poll)

    def _follow_recording(self, rec):
        """Motive started / stopped recording: start / stop the trial. A manual trial is never touched.

        Only a change counts, so connecting while Motive is already recording starts nothing.
        A trial that was started by hand keeps running through the recording and stops by hand.
        ponytail: acts on the UI thread, so the first ~30 ms of a recording are not counted.
        """
        prev, self.rec = self.rec, rec
        if rec is None or prev is None or rec == prev or self.mode != "live" or not self.follow.get():
            return
        if rec and not self.running:
            self.start()
            self.auto = True
        elif not rec and self.auto:
            self.stop()

    def _draw_view(self):
        """Ball, trail and board-marker dots on the canvas, from the latest frame."""
        self.dirty = False
        pts = [c for _, p in self.trail for c in self._xy(*p)]
        self._set(self.trail_line, state="normal" if len(pts) >= 4 else "hidden")
        if len(pts) >= 4:
            self.canvas.coords(self.trail_line, *pts)
        if self.xy is not None:
            (x0, y0), (x1, y1) = (self._xy(self.xy[0] - 1.6, self.xy[1] + 1.6),
                                  self._xy(self.xy[0] + 1.6, self.xy[1] - 1.6))
            self.canvas.coords(self.ball, x0, y0, x1, y1)
        self._set(self.ball, state="normal" if self.xy is not None else "hidden")

    def _show_markers(self, live_gap):
        """Markers section and the dots on the board: which of the markers we look for are tracked."""
        have = self.present is not None and not live_gap
        for name, row, _ in BOARD_MARKERS:
            ok = have and bool(self.present[row])
            colour = (OK if ok else BAD) if have else IDLE
            self._set(self.marker_rows[name][0], fg=colour)
            self._set(self.marker_rows[name][1], text="tracked" if ok else "MISSING" if have else "—")
            self._set(self.marker_items[name], fill=colour)
        ball = have and self.xy is not None
        self._set(self.marker_rows["Ball"][0], fg=(OK if ball else BAD) if have else IDLE)
        self._set(self.marker_rows["Ball"][1], text="on the board" if ball else "not found" if have else "—")
        if live_gap:
            note = "No frames arriving. Is Motive streaming (Settings > Streaming > Enable)?"
        elif isinstance(self.source, LiveSource) and self.source.board is None:
            note = "Waiting for Motive's marker names. Is the board a rigid body with its markers labelled Center, North, NE, NW?"
        elif have:
            note = f"Unlabeled markers seen: {self.n_free} (the ball is the one on the board)"
        else:
            note = "The board's rigid body markers must be labelled Center, North, NE, NW in Motive."
        self._set(self.marker_note, text=note)
        live = self.mode == "live" and not live_gap
        self._set(self.rec_note, text=("Motive is recording" if self.rec else "Motive is not recording") if live else "")

    def _show_event(self, ev):
        """A new event: add it everywhere and flash the gate or rim."""
        self._add_event(ev)
        self._show_totals()
        if self.log:
            self.log.flush()                    # every event hits the disk right away
        if not ev.manual:
            item = self.gate_lines[ev.gate] if ev.kind == "gate" else self.rim
            self.canvas.itemconfigure(item, **({"fill": GATE_ON} if ev.kind == "gate" else {"outline": EDGE_ON}))
            self.flash[item] = time.monotonic() + 0.6

    def _add_event(self, ev):
        """Running totals, event list, timeline tick and log row for one event."""
        self.tot[ev.kind] += ev.delta
        g, e = self.tot["gate"], self.tot["edge"]
        sc = g * SETTINGS.gate_points - e * SETTINGS.edge_penalty
        t_trial = ev.t - self.t_start if self.t_start is not None else 0.0
        what = f"gate {ev.gate}° {ev.direction}" if ev.kind == "gate" and not ev.manual else ev.kind
        self.events.insert(0, f"{t_trial:7.2f} s  {'manual ' if ev.manual else ''}{what} {ev.delta:+d}")
        if not ev.manual and isinstance(self.source, ReplaySource):
            y0, y1 = (2, 14) if ev.kind == "gate" else (14, 26)      # gates on top, edges below
            self.ticks.append((ev.t, self.timeline.create_line(0, y0, 0, y1, width=2,
                                                               fill=GATE_ON if ev.kind == "gate" else EDGE_ON)))
        if self.log:
            self.log_writer.writerow({**asdict(ev), "t_trial": round(t_trial, 3), "gates": g, "edges": e, "score": sc})

    def _show_totals(self):
        g, e = self.tot["gate"], self.tot["edge"]
        self.numbers["gates"].configure(text=str(g))
        self.numbers["edges"].configure(text=str(e))
        self.numbers["score"].configure(text=f"{g * SETTINGS.gate_points - e * SETTINGS.edge_penalty:g}")

    def _rebuild(self, events):
        """After a jump: redo the event list, ticks, totals and log file from the recounted events."""
        self.events.delete(0, "end")
        for _, item in self.ticks:
            self.timeline.delete(item)
        self.ticks, self.tot = [], {"gate": 0, "edge": 0}
        self.trail.clear()
        if self.log:
            self._open_log(self.log_path)
        for ev in events:
            self._add_event(ev)
        self._show_totals()
        if self.log:
            self.log.flush()

    def seek_click(self, event):
        if isinstance(self.source, ReplaySource):
            t0, t1 = self.replay_t
            target = t0 + (t1 - t0) * min(max(event.x / max(self.timeline.winfo_width(), 1), 0), 1)
            self.source.seek(np.searchsorted(self.source.t, target))

    # connecting
    def toggle_connect(self):
        if self.connecting:
            return
        if self.source is not None:
            self._disconnect("disconnected")
            return
        server, local = self.server.get().strip(), self.local.get().strip()
        if not server or not local:
            messagebox.showwarning("Connect", "Enter both IP addresses first (Motive IP and this PC).")
            return
        self.connecting = True
        self._set_mode("idle")
        threading.Thread(target=self._connect, args=(server, local, self.multicast.get()), daemon=True).start()

    def _connect(self, server, local, multicast):
        """Worker thread: connecting can take seconds (timeout 5 s) and must not freeze the window."""
        try:
            src = LiveSource(server, local, multicast, self.on_frame)
            src.start()
        except Exception as ex:
            self.q.put(("connect_failed", ex))
        else:
            self.q.put(("connected", src, server))

    def _connected(self, src, server):
        self.connecting = False
        if self.closing:
            src.stop()
            return
        self.source, self.source_name = src, f"Motive {server}"
        self.rec = None
        self._set_mode("live")
        self.replay_t = None
        self.replay_bar.pack_forget()

    def _connect_failed(self, ex):
        self.connecting = False
        self._set_mode("idle")
        self._set(self.status, text="not connected")
        if isinstance(ex, ImportError):
            messagebox.showerror("Missing package", "The 'natnet' package is not installed.")
        else:
            messagebox.showerror("Could not connect", f"{type(ex).__name__}: {ex}\n\nCheck that Motive is "
                                 "streaming (Settings > Streaming > Enable), that the IP addresses match "
                                 "Motive's streaming interface, the multicast setting matches, and that "
                                 "Windows Firewall allows this app.")

    def replay(self, path=None):
        """Count a recorded trial: plays the whole CSV export as a trial named after the file."""
        path = path or filedialog.askopenfilename(title="Motive CSV export", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        if self.source is not None:
            self._disconnect("")
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            src = ReplaySource(path, float(self.speed.get().rstrip("x")), self.on_frame, self.on_seek)
        except Exception as ex:
            messagebox.showerror("Could not read file", f"{type(ex).__name__}: {ex}")
            return
        finally:
            self.root.configure(cursor="")
        self.replay_name = Path(path).name
        self.recent = [path] + [f for f in self.recent if f != path][:9]
        try:
            RECENT.write_text("\n".join(self.recent))
        except OSError:
            pass                                # only a convenience, never block a replay on it
        self.source, self.source_name = src, f"{self.replay_name} · {self.speed.get()}"
        self._set_mode("replay")
        for _, item in self.ticks:
            self.timeline.delete(item)
        self.ticks = []
        self.replay_t = (src.t[0], src.t[-1])
        self.pause_btn.configure(text="⏸ Pause", state="normal")
        self.replay_bar.pack(fill="x", before=self.mid)
        self.stop()                             # a replay is a whole trial: count it from the start
        self.trial.set(Path(path).stem)
        self.start()
        src.start()

    def recent_menu(self):
        """Native popup menu of the last loaded CSVs, under the ▾ button."""
        menu = tk.Menu(self.root, tearoff=0)
        for f in self.recent:
            menu.add_command(label=Path(f).name, command=lambda f=f: self.replay(f))
        if not self.recent:
            menu.add_command(label="No recent files", state="disabled")
        b = self.recent_btn
        menu.tk_popup(b.winfo_rootx(), b.winfo_rooty() + b.winfo_height())

    def _set_mode(self, mode):
        """idle / live / replay: banner colour and which controls make sense."""
        self.mode = mode
        colour, text = MODES[mode]
        for w in (self.banner, self.mode_label, self.status):
            w.configure(bg=colour)
        self.mode_label.configure(text=text, fg="white")
        self.status.configure(fg="white")
        self.root.title(f"Ball-board counter — {text}")
        idle = mode == "idle" and not self.connecting
        for w in self.conn_widgets:
            w.configure(state="normal" if idle else "disabled")
        self.connect_btn.configure(text="Connecting…" if self.connecting else "Disconnect" if mode == "live" else "Connect",
                                   state="disabled" if mode == "replay" or self.connecting else "normal")
        self.replay_btn.configure(text="■ Stop replay" if mode == "replay" else "Open CSV…",
                                  state="normal" if idle or mode == "replay" else "disabled")
        self.speed_box.configure(state="readonly" if idle or mode == "replay" else "disabled")
        self.recent_btn.configure(state="normal" if idle else "disabled")
        self.clear_btn.configure(state="disabled" if mode == "replay" else "normal")

    def toggle_replay(self):
        if self.mode == "replay":
            self.stop()
            self._disconnect("replay stopped")
        else:
            self.replay()

    def change_speed(self, _event=None):
        if isinstance(self.source, ReplaySource):
            self.source.speed = float(self.speed.get().rstrip("x"))
            self.source_name = f"{self.replay_name} · {self.speed.get()}"

    def toggle_pause(self):
        if isinstance(self.source, ReplaySource):
            self.source.toggle_pause()
            self.pause_btn.configure(text="▶ Resume" if self.source.paused else "⏸ Pause")

    def _draw_timeline(self):
        t0, t1 = self.replay_t
        w = max(self.timeline.winfo_width(), 1)
        x = lambda t: (t - t0) / (t1 - t0) * w
        now = min(max(self.last[1], t0), t1)
        key = (round(now, 2), w, len(self.ticks))
        if key == self._tl_key:
            return
        self._tl_key = key
        self.timeline.coords(self.tl_done, 0, 0, x(now), 28)
        self.timeline.coords(self.tl_head, x(now), 0, x(now), 28)
        for t, item in self.ticks:                  # re-placed every time: survives window resizes
            _, y0, _, y1 = self.timeline.coords(item)
            self.timeline.coords(item, x(t), y0, x(t), y1)
        self.tl_label.configure(text=f"{now - t0:6.1f} / {t1 - t0:.1f} s")

    def _disconnect(self, why):
        if self.source is not None:
            self.source.stop()
        self.source = None
        self.present, self.xy = None, None
        self.dirty = True
        self.pause_btn.configure(text="⏸ Pause", state="disabled")
        self._set_mode("idle")
        self.status.configure(text=why or "not connected")

    # trial control
    @staticmethod
    def now_name():
        return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def start(self):
        if self.running:
            return
        if self.mode != "replay" and self.trial.get().strip() in ("", self.auto_name):
            self.auto_name = self.now_name()
            self.trial.set(self.auto_name)
        with self.lock:
            self.counter.reset()
            self.running = True
        self.t_start = None
        self.tot = {"gate": 0, "edge": 0}
        self.events.delete(0, "end")
        self._show_totals()
        LOGS.mkdir(exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = "".join(ch for ch in self.trial.get() if ch.isalnum() or ch in "-_") or "trial"
        kind = "replay" if self.mode == "replay" else "live"
        tag = "" if name == self.auto_name else f"_{stamp}"       # the default name already is a timestamp
        self._open_log(LOGS / f"{name}_{kind}{tag}.csv")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        for b in self.manual_btns:
            b.configure(state="normal")

    def _open_log(self, path):
        """(Re)write the log file from scratch: header now, rows as events come."""
        if self.log:
            self.log.close()
        self.log_path, self.log = path, open(path, "w", newline="")
        cols = [f.name for f in fields(C.Event)] + ["t_trial", "gates", "edges", "score"]
        self.log_writer = csv.DictWriter(self.log, fieldnames=cols)
        self.log_writer.writeheader()

    def clear(self):
        """Zero the score and the event list. A running trial restarts (new log file); the old log stays on disk."""
        if self.tot["gate"] or self.tot["edge"] or self.events.size():
            note = "\nThe running trial restarts with a new log file." if self.running else ""
            if not messagebox.askyesno("Clear score", "Clear the score and the event list?" + note):
                return
        was, auto = self.running, self.auto
        self.stop()
        with self.lock:
            self.counter.reset()
        self.tot = {"gate": 0, "edge": 0}
        self.events.delete(0, "end")
        self.trail.clear()
        self._show_totals()
        if was:
            self.start()
            self.auto = auto                    # still follows Motive's recording if it started the trial

    def stop(self):
        self.auto = False
        with self.lock:
            self.running = False
        if self.log:
            self.log.close()
            self.log = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        for b in self.manual_btns:
            b.configure(state="disabled")

    def manual(self, kind, delta):
        if not self.running:
            return
        with self.lock:
            ev = self.counter.manual(self.last[1], self.last[0], kind, delta)
        self._show_event(ev)

    def close(self):
        self.closing = True
        self.stop()
        if self.source is not None:
            self.source.stop()
        self.root.destroy()


ABOUT = """Part of MATCH (Motor Adaptation Through Complexity matching in Harmonized joint action), \
a Tilburg University research project on joint action as training for upper-body motor rehabilitation.

The ball-on-board task: participants, alone or in pairs, hold a large round board by its black \
handles and tilt it to roll a ball as fast and accurately as possible along the circular target \
track. Each pass through one of the four doors ("gates" here) scores 1 point, each time the ball \
hits the board's edge costs ½ point. A trial lasts 2 minutes.

This program does that counting, instead of the experimenter counting by eye. It reads the board \
and ball markers from OptiTrack Motive, live during a recording or from a CSV export, shows the \
score as the trial runs and logs every gate crossing and edge hit to a CSV per trial.

Connecting to a live recording, in Motive's streaming settings (Motive 3: Settings > Streaming; \
Motive 2: Data Streaming pane):
• enable NatNet streaming
• local interface: Loopback (127.0.0.1) if this app runs on the Motive PC, otherwise its network address
• transmission type: Unicast (leave multicast unticked here)
• stream labeled and unlabeled markers
• the board is a rigid body whose markers are labelled Center, North, NE and NW (Labels pane)
Then enter the same IPs under "Motive connection" and press Connect.\n\nPress F11 for full screen; the board grows with the window. The sidebar sections fold when you click their titles."""


def about():
    messagebox.showinfo("About Ball-board counter", "Ball-board counter", detail=ABOUT)


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)      # else Windows scales the whole window as a bitmap
        except Exception:
            pass
    root = tk.Tk()
    if sys.platform == "darwin":
        root.createcommand("tkAboutDialog", about)      # app menu > About BallBoardCounter
    else:
        menubar = tk.Menu(root)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=about)
        menubar.add_cascade(label="Help", menu=help_menu)
        root.configure(menu=menubar)
    try:
        root.iconphoto(True, tk.PhotoImage(file=ICON))
    except tk.TclError:
        pass                                    # no icon file (e.g. a source copy without it)
    App(root)
    root.mainloop()
