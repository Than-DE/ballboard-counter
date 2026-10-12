"""Draws icon.png (1024 px): the board schematic of MATCH/analysis/board.py draw_board, without labels.

python make_icon.py   (needs matplotlib; only rerun when the icon should change)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Wedge
from matplotlib.transforms import Affine2D

R_BOARD, R_INNER, R_OUTER = 39.0, 20.0, 30.0      # cm, same as counter.py / board.py
pol = lambda r, a: (r * np.sin(np.radians(a)), r * np.cos(np.radians(a)))   # compass deg -> x, y

fig = plt.figure(figsize=(10, 10), dpi=102.4)
ax = fig.add_axes([0, 0, 1, 1])
ax.add_patch(Circle((0, 0), R_BOARD, fc="#8a8477", ec="peru", lw=24))
for a in (45, 135, 225, 315):
    ax.add_patch(Rectangle((-4, R_BOARD + 1), 8, 7, fc="k" if a in (135, 225) else "dimgray",
                           transform=Affine2D().rotate_deg(-a) + ax.transData))
ax.add_patch(Wedge((0, 0), R_OUTER, 0, 360, width=R_OUTER - R_INNER, fc="w", alpha=.12))
for r in (R_INNER, R_OUTER):
    ax.add_patch(Circle((0, 0), r, fill=False, ec="w", ls=(0, (2, 2)), lw=6))
for a in (0, 90, 180, 270):
    ax.plot(*zip(pol(R_INNER, a), pol(R_OUTER, a)), color="w", lw=22, solid_capstyle="round")
for x, y in [(0, 0)] + [pol(R_BOARD, a) for a in (315, 0, 45)]:
    ax.add_patch(Circle((x, y), 2.6, fc="crimson", zorder=5))
ax.add_patch(Circle(pol(25, 62), 4.2, fc="#f2c14e", ec="k", lw=6, zorder=6))   # the ball, in the band
ax.set_xlim(-49, 49); ax.set_ylim(-49, 49); ax.axis("off")
fig.savefig("icon.png", transparent=True)
