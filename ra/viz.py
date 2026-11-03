"""Plot helpers: consistent style, figure saving, and box-and-arrow diagrams drawn with matplotlib
(architecture, ReAct flow, multi-turn flow, decision tree, audit trail)."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config

PALETTE = ["#2E5A88", "#E07B39", "#3A9D5D", "#C0392B", "#8E44AD", "#7F8C8D", "#D4AC0D", "#16A085"]
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10, "axes.titleweight": "bold"})


def save(fig, name: str) -> None:
    fig.savefig(config.FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=150)


def bar_labels(ax, fmt="{:.1f}", fontsize=8):
    for c in ax.containers:
        ax.bar_label(c, fmt=fmt, fontsize=fontsize, padding=2)


def diagram(nodes: dict, edges: list, title: str = "", figsize=(12, 7), xlim=(0, 12), ylim=(0, 8)):
    """nodes: id -> (x, y, label, color[, w, h]); edges: (src, dst, label[, style])."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    boxes = {}
    for nid, spec in nodes.items():
        x, y, label, color = spec[:4]
        w = spec[4] if len(spec) > 4 else 2.0
        h = spec[5] if len(spec) > 5 else 0.8
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    fc=color, ec="#333333", lw=1.0, alpha=0.92))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, color="white" if color not in ("#F4F6F7", "#FDEBD0", "#D6EAF8", "#D5F5E3", "#FADBD8") else "#222",
                wrap=True, fontweight="bold")
        boxes[nid] = (x, y, w, h)
    for e in edges:
        a, b, lab = e[:3]
        style = e[3] if len(e) > 3 else "-"
        xa, ya, wa, ha = boxes[a]
        xb, yb, wb, hb = boxes[b]
        # attach to box borders
        dx, dy = xb - xa, yb - ya
        if abs(dx) * ha > abs(dy) * wa:
            pa = (xa + (wa / 2) * (1 if dx > 0 else -1), ya)
            pb = (xb - (wb / 2) * (1 if dx > 0 else -1), yb)
        else:
            pa = (xa, ya + (ha / 2) * (1 if dy > 0 else -1))
            pb = (xb, yb - (hb / 2) * (1 if dy > 0 else -1))
        ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#444",
                                     linestyle=style, connectionstyle="arc3,rad=0.0"))
        if lab:
            ax.text((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2 + 0.12, lab, fontsize=7.5, ha="center", color="#333",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
    if title:
        ax.set_title(title, fontsize=12)
    return fig, ax
