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

PX = 250 / 45                   # canvas pixels per cm
SIZE = 520
TRAIL_S = 1.5
BALL, GATE_ON, EDGE_ON = "#eb6834", "#1baf7a", "#e0453a"
MODES = {"idle": ("#6b6b6b", "NOT CONNECTED"),       # banner colour, label
         "live": ("#c62828", "● LIVE"),
         "replay": ("#1c5cab", "▶ REPLAY")}


# --- frame sources: call on_frame(frame, t, (k,3) mm) from their own thread ---------------

class LiveSource:
    """Frames streamed by Motive. NatNet sends metres; unlabeled markers have model ID 0."""

    def __init__(self, server, local, multicast, on_frame):
        from natnet import NatNetClient
        self.client = NatNetClient(server_ip_address=server, local_ip_address=local,
                                   use_multicast=multicast)
        self.client.on_data_frame_received_event.handlers.append(self._frame)
        self.on_frame = on_frame

    def start(self):
        self.client.connect(timeout=5.0)        # raises TimeoutError if Motive doesn't answer
        self.client.run_async()

    def stop(self):
        self.client.shutdown()

    def _frame(self, df):
        pts = [m.pos for m in df.labeled_markers if m.model_id == 0 and not (m.param or 0) & 0x01]
        if not pts:                             # older NatNet versions: legacy unlabeled list
            pts = list(df.unlabeled_marker_pos)
        self.on_frame(df.prefix.frame_number, df.suffix.timestamp, np.array(pts, float).reshape(-1, 3) * 1000)


class ReplaySource:
    """A Motive CSV export played back at `speed` x real time.

    Pausable, `speed` may change while playing, and seek(i) jumps to frame i. A jump
    calls on_seek(i) on the replay thread before frame i plays, so the app can recount.
    """

    def __init__(self, path, speed, on_frame, on_seek):
        self.frames, self.t, self.pos = C.read_motive_csv(path)
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
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Motive IP").pack(side="left")
        self.server = tk.StringVar(value="127.0.0.1")
        self.conn_widgets = [ttk.Entry(top, textvariable=self.server, width=15)]
        self.conn_widgets[-1].pack(side="left", padx=4)
        ttk.Label(top, text="this PC").pack(side="left")
        self.local = tk.StringVar(value="127.0.0.1")
        self.conn_widgets.append(ttk.Entry(top, textvariable=self.local, width=15))
        self.conn_widgets[-1].pack(side="left", padx=4)
        self.multicast = tk.BooleanVar(value=False)
        self.conn_widgets.append(ttk.Checkbutton(top, text="multicast", variable=self.multicast))
        self.conn_widgets[-1].pack(side="left", padx=4)
        self.connect_btn = ttk.Button(top, text="Connect", width=11, command=self.toggle_connect)
        self.connect_btn.pack(side="left", padx=4)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)
        self.replay_btn = ttk.Button(top, text="Replay CSV…", width=12, command=self.toggle_replay)
        self.replay_btn.pack(side="left")
        try:
            self.recent = RECENT.read_text().splitlines()
        except OSError:
            self.recent = []
        self.recent_btn = ttk.Button(top, text="▾", width=2, command=self.recent_menu)
        self.recent_btn.pack(side="left")
        self.speed = tk.StringVar(value="1x")
        self.speed_box = ttk.Combobox(top, textvariable=self.speed, values=["1x", "2x", "5x", "20x"],
                                      width=4, state="readonly")
        self.speed_box.pack(side="left", padx=4)
        self.speed_box.bind("<<ComboboxSelected>>", self.change_speed)

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
        self.canvas = tk.Canvas(mid, width=SIZE, height=SIZE, bg="white", highlightthickness=0)
        self.canvas.pack(side="left")
        self._draw_board()

        side = ttk.Frame(mid, padding=(16, 0))
        side.pack(side="left", fill="y")
        self.numbers = {}
        for key, label in (("gates", "Gates"), ("edges", "Edge hits"), ("score", "Score")):
            ttk.Label(side, text=label, font=("Segoe UI", 12)).pack(anchor="w")
            self.numbers[key] = ttk.Label(side, text="0", font=("Segoe UI", 34, "bold"))
            self.numbers[key].pack(anchor="w", pady=(0, 8))
        ttk.Label(side, text="Trial name").pack(anchor="w")
        self.trial = tk.StringVar(value="trial")
        ttk.Entry(side, textvariable=self.trial, width=22).pack(anchor="w", pady=(0, 8))
        row = ttk.Frame(side)
        row.pack(anchor="w")
        self.start_btn = ttk.Button(row, text="Start", command=self.start)
        self.start_btn.pack(side="left")
        ttk.Button(row, text="Stop", command=self.stop).pack(side="left", padx=4)
        ttk.Label(side, text="Manual correction", padding=(0, 14, 0, 2)).pack(anchor="w")
        for kind in ("gate", "edge"):
            row = ttk.Frame(side)
            row.pack(anchor="w", pady=2)
            for delta in (+1, -1):
                ttk.Button(row, text=f"{delta:+d} {kind}", width=8,
                           command=lambda k=kind, d=delta: self.manual(k, d)).pack(side="left", padx=(0, 4))

        self.events = tk.Listbox(self.root, height=6, font=("Consolas", 10))
        self.events.pack(fill="x", padx=8, pady=(0, 8))

    def _xy(self, x, y):
        return SIZE / 2 + x * PX, SIZE / 2 - y * PX

    def _circle(self, r, **kw):
        (x0, y0), (x1, y1) = self._xy(-r, r), self._xy(r, -r)
        return self.canvas.create_oval(x0, y0, x1, y1, **kw)

    def _draw_board(self):
        self.rim = self._circle(C.R_BOARD, fill="#8a8477", outline="peru", width=4)
        self._circle(SETTINGS.gate_r_max, outline="white", dash=(2, 3))
        self._circle(SETTINGS.gate_r_min, outline="white", dash=(2, 3))
        self._circle(SETTINGS.edge_hit_r, outline="#e0453a", dash=(1, 4))
        self.gate_lines = {}
        for g in C.GATES:
            a = math.radians(g)
            p0 = self._xy(SETTINGS.gate_r_min * math.sin(a), SETTINGS.gate_r_min * math.cos(a))
            p1 = self._xy(SETTINGS.gate_r_max * math.sin(a), SETTINGS.gate_r_max * math.cos(a))
            self.gate_lines[g] = self.canvas.create_line(*p0, *p1, fill="white", width=4)
            self.canvas.create_text(*self._xy(34 * math.sin(a), 34 * math.cos(a)), text=f"{g}°",
                                    fill="white", font=("Segoe UI", 9))
        self.canvas.create_text(SIZE / 2, SIZE - 12, text="participant", fill="black", font=("Segoe UI", 10))
        self.trail_line = self.canvas.create_line(0, 0, 0, 0, fill=BALL, width=3, state="hidden")
        self.ball = self._circle(1.6, fill=BALL, outline="black", state="hidden")

    # frames (source thread)
    def on_frame(self, frame, t, markers):
        if frame is None:
            self.q.put(("end",))
            return
        xy, board = C.ball_on_map(markers, SETTINGS)
        with self.lock:
            events = self.counter.update(t, frame, xy) if self.running else []
        self.q.put(("frame", frame, t, xy, board, len(markers), events))

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
                    self.counter.update(float(src.t[k]), int(src.frames[k]), C.ball_on_map(src.pos[k], SETTINGS)[0])
                for e in manual:
                    self.counter.manual(e.t, e.frame, e.kind, e.delta)
                self.counter.events.sort(key=lambda e: e.t)
            recounted = list(self.counter.events)       # later frames' events follow via the queue
        self.q.put(("seek", recounted))

    # UI loop (main thread)
    def poll(self):
        latest = None
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "end":
                    self.stop()
                    self._disconnect("replay finished")
                    continue
                if msg[0] == "seek":
                    self._rebuild(msg[1])
                    continue
                _, frame, t, xy, board, n, events = msg
                latest = msg
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
            _, frame, t, xy, board, n, _ = latest
            self.last = (frame, t)
            while self.trail and t - self.trail[0][0] > TRAIL_S:
                self.trail.popleft()
            pts = [c for _, p in self.trail for c in self._xy(*p)]
            self.canvas.itemconfigure(self.trail_line, state="normal" if len(pts) >= 4 else "hidden")
            if len(pts) >= 4:
                self.canvas.coords(self.trail_line, *pts)
            if xy is not None:
                (x0, y0), (x1, y1) = self._xy(xy[0] - 1.6, xy[1] + 1.6), self._xy(xy[0] + 1.6, xy[1] - 1.6)
                self.canvas.coords(self.ball, x0, y0, x1, y1)
            self.canvas.itemconfigure(self.ball, state="normal" if xy is not None else "hidden")
            self.status_bits = (f"{n} unlabeled markers · board {'✓' if board else '✗'} · "
                                f"ball {'✓' if xy is not None else '✗'} · frame {frame}")
        replaying = isinstance(self.source, ReplaySource)
        if self.source is not None:
            state = "   ⏸ PAUSED" if replaying and self.source.paused else ("   ▶ COUNTING" if self.running else "")
            self.status.configure(text=f"{self.source_name} · {len(self.rate)} Hz · {self.status_bits}{state}")
        if self.replay_t is not None:
            self._draw_timeline()
        for item, until in list(self.flash.items()):
            if now > until:
                self.canvas.itemconfigure(item, **({"outline": "peru"} if item == self.rim else {"fill": "white"}))
                del self.flash[item]
        self.root.after(33, self.poll)

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
        if self.source is not None:
            self._disconnect("disconnected")
            return
        try:
            src = LiveSource(self.server.get(), self.local.get(), self.multicast.get(), self.on_frame)
            self.status.configure(text="connecting…")
            self.root.update_idletasks()
            src.start()
        except ImportError:
            messagebox.showerror("Missing package", "The 'natnet' package is not installed.")
            return
        except Exception as ex:
            messagebox.showerror("Could not connect", f"{type(ex).__name__}: {ex}\n\nCheck that Motive is "
                                 "streaming (Settings > Streaming > Enable), that the IP addresses match "
                                 "Motive's streaming interface, the multicast setting matches, and that "
                                 "Windows Firewall allows this app.")
            self.status.configure(text="not connected")
            return
        self.source, self.source_name = src, f"Motive {self.server.get()}"
        self._set_mode("live")
        self.replay_t = None
        self.replay_bar.pack_forget()

    def replay(self, path=None):
        """Count a recorded trial: plays the whole CSV export as a trial named after the file."""
        path = path or filedialog.askopenfilename(title="Motive CSV export", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        if self.source is not None:
            self._disconnect("")
        try:
            src = ReplaySource(path, float(self.speed.get().rstrip("x")), self.on_frame, self.on_seek)
        except Exception as ex:
            messagebox.showerror("Could not read file", f"{type(ex).__name__}: {ex}")
            return
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
        for w in self.conn_widgets:
            w.configure(state="normal" if mode == "idle" else "disabled")
        self.connect_btn.configure(text="Disconnect" if mode == "live" else "Connect",
                                   state="disabled" if mode == "replay" else "normal")
        self.replay_btn.configure(text="■ Stop replay" if mode == "replay" else "Replay CSV…",
                                  state="disabled" if mode == "live" else "normal")
        self.speed_box.configure(state="disabled" if mode == "live" else "readonly")
        self.recent_btn.configure(state="disabled" if mode == "live" else "normal")

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
        self.pause_btn.configure(text="⏸ Pause", state="disabled")
        self._set_mode("idle")
        self.status.configure(text=why or "not connected")

    # trial control
    def start(self):
        if self.running:
            return
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
        self._open_log(LOGS / f"{name}_{kind}_{stamp}.csv")
        self.start_btn.configure(state="disabled")

    def _open_log(self, path):
        """(Re)write the log file from scratch: header now, rows as events come."""
        if self.log:
            self.log.close()
        self.log_path, self.log = path, open(path, "w", newline="")
        cols = [f.name for f in fields(C.Event)] + ["t_trial", "gates", "edges", "score"]
        self.log_writer = csv.DictWriter(self.log, fieldnames=cols)
        self.log_writer.writeheader()

    def stop(self):
        with self.lock:
            self.running = False
        if self.log:
            self.log.close()
            self.log = None
        self.start_btn.configure(state="normal")

    def manual(self, kind, delta):
        if not self.running:
            return
        with self.lock:
            ev = self.counter.manual(self.last[1], self.last[0], kind, delta)
        self._show_event(ev)

    def close(self):
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
Then enter the same IPs here and press Connect."""


def about():
    messagebox.showinfo("About Ball-board counter", "Ball-board counter", detail=ABOUT)


if __name__ == "__main__":
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
