#!/usr/bin/env python3
"""Generate the 6 publication-grade charts and schematic diagrams for TokenFed."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_JSON = os.path.join(ROOT_DIR, "results", "dashboard_data.json")
DEFAULT_FIG_DIR = os.path.join(ROOT_DIR, "figures")
OUT_PNG = os.path.join(DEFAULT_FIG_DIR, "token_fed_chart.png")
TWO_BAR_PNG = os.path.join(DEFAULT_FIG_DIR, "token_fed_2bar.png")
BOOM_PNG = os.path.join(DEFAULT_FIG_DIR, "token_economy_boom.png")
DECOMP_PNG = os.path.join(DEFAULT_FIG_DIR, "token_fed_decomposition.png")
DECOMP_2BAR_PNG = os.path.join(DEFAULT_FIG_DIR, "token_fed_decomposition_2bar.png")
ARCH_PNG = os.path.join(DEFAULT_FIG_DIR, "token_fed_architecture.png")

REGIMES = [
    ("flat_today",  "1. Today\n(Flat + Locked)", "#94A3B8"),
    ("pure_market", "2. Pure\nMarket",           "#F59E0B"),
    ("llm_fed",     "3. TokenFed ★\n(Central Bank)", "#0284C7"),
]

WORLDS = [
    ("supply_shock",    "GPU\nCrunch"),
    ("slop_bubble",     "Spam\nBubble"),
    ("innovation_boom", "Science\nBoom"),
]


def _set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
    })


def main() -> None:
    with open(DATA_JSON) as f:
        d = json.load(f)

    ov = d["overall"]
    worlds_data = d["worlds"]

    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")

    # Panel 1: Total Economic + R&D Value (Pooled)
    ax0 = axes[0]
    ax0.set_facecolor("#FFFFFF")
    x = np.arange(len(REGIMES))
    net_vals = [ov[r[0]]["netVal"] for r in REGIMES]
    colors = [r[2] for r in REGIMES]
    labels = [r[1] for r in REGIMES]

    bars0 = ax0.bar(x, net_vals, width=0.50, color=colors, zorder=3)
    bars0[2].set_edgecolor("#0F172A")
    bars0[2].set_linewidth(1.6)
    ax0.set_ylim(0, 118)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=9.2, color="#1E293B", linespacing=1.15)
    ax0.get_xticklabels()[2].set_fontweight("bold")
    ax0.set_yticks([0, 25, 50, 75, 100])
    ax0.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=8.8, color="#64748B")
    ax0.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax0.set_title("Overall Score (Output + Lab R&D)", fontsize=11.3, fontweight="bold", color="#0F172A", pad=12)
    for i, (b, v) in enumerate(zip(bars0, net_vals)):
        ax0.text(
            b.get_x() + b.get_width() / 2.0,
            v + 2.2,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=10.4 if i == 2 else 9.6,
            fontweight="bold" if i == 2 else "normal",
            color="#0F172A",
        )

    # Panel 2: Cash != Value — Science Breakthroughs vs Speculative Swarms (%)
    ax1 = axes[1]
    ax1.set_facecolor("#FFFFFF")
    w = 0.32
    sci_vals = [ov[r[0]]["sciPct"] for r in REGIMES]
    swm_vals = [ov[r[0]]["swmPct"] for r in REGIMES]
    b_sci = ax1.bar(x - w / 2, sci_vals, width=w, color="#0284C7", label="Science Labs (High Value)", zorder=3)
    b_swm = ax1.bar(x + w / 2, swm_vals, width=w, color="#CBD5E1", edgecolor="#94A3B8", linewidth=0.8, label="Rich Swarms (Low Value)", zorder=3)
    ax1.set_ylim(0, 118)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.2, color="#1E293B", linespacing=1.15)
    ax1.get_xticklabels()[2].set_fontweight("bold")
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.8, color="#64748B")
    ax1.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax1.set_title("% of Jobs Done: Science vs. Rich Swarms", fontsize=11.3, fontweight="bold", color="#0F172A", pad=12)
    ax1.legend(loc="upper left", frameon=False, fontsize=8.5)

    for i, (bs, bw, vs, vw) in enumerate(zip(b_sci, b_swm, sci_vals, swm_vals)):
        ax1.text(
            bs.get_x() + bs.get_width() / 2.0,
            vs + 2.0,
            f"{vs}%",
            ha="center",
            va="bottom",
            fontsize=9.5 if i in (1, 2) else 8.8,
            fontweight="bold" if i in (1, 2) else "normal",
            color="#B45309" if i == 1 else "#0369A1",
        )
        ax1.text(
            bw.get_x() + bw.get_width() / 2.0,
            vw + 2.0,
            f"{vw}%",
            ha="center",
            va="bottom",
            fontsize=8.6,
            color="#475569",
        )

    # Panel 3: Total Value Across the 3 Macro Worlds
    ax2 = axes[2]
    ax2.set_facecolor("#FFFFFF")
    xw = np.arange(len(WORLDS))
    bw_w = 0.24
    offsets = [-bw_w, 0.0, bw_w]
    short_names = ["Today", "Pure Market", "TokenFed ★"]
    for idx, ((rk, _, col), sname) in enumerate(zip(REGIMES, short_names)):
        wvals = [worlds_data[wk]["regimes"][rk]["netVal"] for wk, _ in WORLDS]
        bars_w = ax2.bar(
            xw + offsets[idx],
            wvals,
            width=bw_w,
            color=col,
            edgecolor="#0F172A" if idx == 2 else "none",
            linewidth=1.2 if idx == 2 else 0.0,
            label=sname,
            zorder=3,
        )
        for bw_i, v in zip(bars_w, wvals):
            ax2.text(
                bw_i.get_x() + bw_i.get_width() / 2.0,
                v + 2.0,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=9.1 if idx == 2 else 8.4,
                fontweight="bold" if idx == 2 else "normal",
                color="#0F172A" if idx == 2 else ("#92400E" if idx == 1 else "#475569"),
            )

    ax2.set_ylim(0, 126)
    ax2.set_xticks(xw)
    ax2.set_xticklabels([wl for _, wl in WORLDS], fontsize=9.2, color="#1E293B", linespacing=1.15)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=8.8, color="#64748B")
    ax2.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax2.set_title("Overall Score Across Three Shocks", fontsize=11.3, fontweight="bold", color="#0F172A", pad=12)
    ax2.legend(loc="upper left", ncol=3, frameon=False, fontsize=8.3, columnspacing=1.0)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)

    fig.suptitle(
        "TokenFed Benchmark  (12 Matched Runs per System · 12 AI Firms)",
        fontsize=12.8,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.91], w_pad=2.4)
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


def plot_boom_chart() -> None:
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.5), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")

    # Panel 1: Falling Price per 1M Tokens (GPT-4-Class Capability, Epoch AI / API Pricing)
    ax0 = axes[0]
    ax0.set_facecolor("#FFFFFF")
    yrs_p = ["2023\n(GPT-4)", "2024", "2025", "2026"]
    prices = [45.0, 2.50, 0.20, 0.06]
    xp = np.arange(len(yrs_p))
    ax0.plot(xp, prices, marker="o", markersize=8, linewidth=2.6, color="#D97706", zorder=4)
    ax0.fill_between(xp, prices, 0.02, color="#FEF3C7", alpha=0.55, zorder=2)
    ax0.set_yscale("log")
    ax0.set_ylim(0.02, 160)
    ax0.set_xticks(xp)
    ax0.set_xticklabels(yrs_p, fontsize=9.0, color="#1E293B", linespacing=1.15)
    ax0.set_yticks([0.05, 0.5, 5, 50])
    ax0.set_yticklabels(["$0.05", "$0.50", "$5", "$50"], fontsize=8.8, color="#64748B")
    ax0.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax0.set_title("1. Unit Price Fell (~10x–40x / yr)", fontsize=11.2, fontweight="bold", color="#0F172A", pad=12)
    price_lbls = ["~$45 / 1M", "$2.50", "$0.20", "<$0.10*"]
    for i, (xi, yi, lbl) in enumerate(zip(xp, prices, price_lbls)):
        ax0.text(
            xi,
            yi * (1.42 if i < 3 else 1.55),
            lbl,
            ha="center",
            va="bottom",
            fontsize=9.2,
            fontweight="bold" if i in (0, 3) else "normal",
            color="#92400E" if i == 3 else "#1E293B",
        )
    ax0.text(
        0.02, 0.05,
        r"*2023-class capability; frontier reasoning stays \$5–\$25+/1M",
        transform=ax0.transAxes,
        fontsize=7.8,
        color="#64748B",
    )

    # Panel 2: Exploding Tokens per Unit of Work (Log Scale)
    ax1 = axes[1]
    ax1.set_facecolor("#FFFFFF")
    work_labels = ["2023\nChat Reply", "2024\nDoc / RAG", "2025\nAgent PR", "2026\nScience Run"]
    work_tokens = [500, 15_000, 500_000, 5_000_000]
    work_disp = ["~500", "~15k", "~500k", "5M+"]
    xw = np.arange(len(work_labels))
    b_colors = ["#94A3B8", "#64748B", "#6366F1", "#4F46E5"]
    bars1 = ax1.bar(xw, work_tokens, width=0.52, color=b_colors, zorder=3)
    ax1.set_yscale("log")
    ax1.set_ylim(150, 30_000_000)
    ax1.set_xticks(xw)
    ax1.set_xticklabels(work_labels, fontsize=9.0, color="#1E293B", linespacing=1.15)
    ax1.set_yticks([1_000, 100_000, 10_000_000])
    ax1.set_yticklabels(["1k", "100k", "10M"], fontsize=8.8, color="#64748B")
    ax1.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax1.set_title("2. Tokens per Job Exploded (10,000x)", fontsize=11.2, fontweight="bold", color="#0F172A", pad=12)
    for i, (b, v, dl) in enumerate(zip(bars1, work_tokens, work_disp)):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            v * 1.38,
            dl,
            ha="center",
            va="bottom",
            fontsize=9.4 if i >= 2 else 8.9,
            fontweight="bold" if i >= 2 else "normal",
            color="#312E81" if i >= 2 else "#334155",
        )

    # Panel 3: Total Monthly Token Volume at Google (Google I/O 2024–2026)
    ax2 = axes[2]
    ax2.set_facecolor("#FFFFFF")
    vol_labels = ["May 2024\n(Google I/O)", "May 2025\n(Google I/O)", "May 2026\n(Google I/O)"]
    vol_trillions = [9.7, 480.0, 3200.0]
    vol_disp = ["9.7T", "480T", "3,200T\n(3.2 Quadrillion)"]
    xv = np.arange(len(vol_labels))
    v_colors = ["#94A3B8", "#38BDF8", "#0284C7"]
    bars2 = ax2.bar(xv, vol_trillions, width=0.50, color=v_colors, zorder=3)
    bars2[2].set_edgecolor("#0F172A")
    bars2[2].set_linewidth(1.5)
    ax2.set_ylim(0, 4150)
    ax2.set_xticks(xv)
    ax2.set_xticklabels(vol_labels, fontsize=9.0, color="#1E293B", linespacing=1.15)
    ax2.get_xticklabels()[2].set_fontweight("bold")
    ax2.set_yticks([0, 1000, 2000, 3000])
    ax2.set_yticklabels(["0", "1,000T", "2,000T", "3,000T"], fontsize=8.8, color="#64748B")
    ax2.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax2.set_title("3. Total Volume Surged 330x in 2 Years", fontsize=11.2, fontweight="bold", color="#0F172A", pad=12)
    for i, (b, v, dl) in enumerate(zip(bars2, vol_trillions, vol_disp)):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            v + 95,
            dl,
            ha="center",
            va="bottom",
            fontsize=9.4 if i == 2 else 9.0,
            fontweight="bold" if i == 2 else "normal",
            color="#0F172A" if i == 2 else "#334155",
            linespacing=1.1,
        )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)

    fig.suptitle(
        "Jevons Paradox in AI: Why Cheaper Tokens Created More Compute Scarcity",
        fontsize=12.8,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.91], w_pad=2.4)
    fig.savefig(BOOM_PNG, dpi=220)
    plt.close(fig)
    print(f"Wrote {BOOM_PNG}")


def plot_decomposition_chart() -> None:
    """Plot the two +11.4 value jumps side-by-side plus the 100% physical cluster balance sheet."""
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.85), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")

    cats = ["Seed\nStartups", "Enterprise\n+ Swarms", "Internal\nLab R&D", "Science\nLabs"]
    xc = np.arange(len(cats))

    # Panel 1: Step 1 — Today -> Pure Market (+11.4 Net)
    ax0 = axes[0]
    ax0.set_facecolor("#FFFFFF")
    d1 = [12.0, 4.3, 0.0, -4.9]
    c1 = ["#059669", "#F59E0B", "#CBD5E1", "#DC2626"]
    bars1 = ax0.bar(xc, d1, width=0.52, color=c1, zorder=3)
    bars1[0].set_edgecolor("#064E3B")
    bars1[0].set_linewidth(1.2)
    bars1[3].set_edgecolor("#7F1D1D")
    bars1[3].set_linewidth(1.2)
    ax0.axhline(0, color="#64748B", linewidth=1.0, zorder=2)
    ax0.set_ylim(-9.0, 20.5)
    ax0.set_xticks(xc)
    ax0.set_xticklabels(cats, fontsize=8.9, color="#1E293B", linespacing=1.15)
    ax0.set_yticks([-5, 0, 5, 10, 15])
    ax0.set_yticklabels(["-5", "0", "+5", "+10", "+15"], fontsize=8.8, color="#64748B")
    ax0.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax0.set_title("1. Today → Pure Market (+11.4 Net)", fontsize=11.1, fontweight="bold", color="#0F172A", pad=12)

    for i, (b, v) in enumerate(zip(bars1, d1)):
        if abs(v) < 0.05:
            ax0.text(b.get_x() + b.get_width() / 2.0, 0.6, "0.0", ha="center", va="bottom", fontsize=8.8, color="#64748B")
        elif v > 0:
            ax0.text(
                b.get_x() + b.get_width() / 2.0, v + 0.55, f"+{v:.1f}",
                ha="center", va="bottom", fontsize=9.5 if i == 0 else 8.9,
                fontweight="bold" if i == 0 else "normal",
                color="#065F46" if i == 0 else "#92400E",
            )
        else:
            ax0.text(
                b.get_x() + b.get_width() / 2.0, v - 0.65, f"{v:.1f}",
                ha="center", va="top", fontsize=9.5, fontweight="bold", color="#B91C1C",
            )

    ax0.text(
        0.97, 0.93,
        "Unlocks idle blocks for startups,\nbut cash-rich bots crowd out science",
        transform=ax0.transAxes, ha="right", va="top", fontsize=8.2, color="#475569",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=0.8),
    )

    # Panel 2: Step 2 — Pure Market -> TokenFed (+11.4 Net)
    ax1 = axes[1]
    ax1.set_facecolor("#FFFFFF")
    d2 = [-1.2, 2.1, -5.5, 16.1]
    c2 = ["#94A3B8", "#38BDF8", "#475569", "#0284C7"]
    bars2 = ax1.bar(xc, d2, width=0.52, color=c2, zorder=3)
    bars2[3].set_edgecolor("#0F172A")
    bars2[3].set_linewidth(1.5)
    ax1.axhline(0, color="#64748B", linewidth=1.0, zorder=2)
    ax1.set_ylim(-9.0, 20.5)
    ax1.set_xticks(xc)
    ax1.set_xticklabels(cats, fontsize=8.9, color="#1E293B", linespacing=1.15)
    ax1.get_xticklabels()[3].set_fontweight("bold")
    ax1.set_yticks([-5, 0, 5, 10, 15])
    ax1.set_yticklabels(["-5", "0", "+5", "+10", "+15"], fontsize=8.8, color="#64748B")
    ax1.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax1.set_title("2. Pure Market → TokenFed ★ (+11.4 Net)", fontsize=11.1, fontweight="bold", color="#0F172A", pad=12)

    for i, (b, v) in enumerate(zip(bars2, d2)):
        if v > 0:
            ax1.text(
                b.get_x() + b.get_width() / 2.0, v + 0.55, f"+{v:.1f}",
                ha="center", va="bottom", fontsize=10.0 if i == 3 else 8.9,
                fontweight="bold" if i == 3 else "normal",
                color="#0F172A" if i == 3 else "#0369A1",
            )
        else:
            ax1.text(
                b.get_x() + b.get_width() / 2.0, v - 0.65, f"{v:.1f}",
                ha="center", va="top", fontsize=9.2 if i == 2 else 8.7,
                fontweight="bold" if i == 2 else "normal",
                color="#334155" if i == 2 else "#64748B",
            )

    ax1.text(
        0.03, 0.93,
        "Flexing lab R&D (-5.5) frees GPUs\nfor +16.1 Science & +2.6 Ent. P0s",
        transform=ax1.transAxes, ha="left", va="top", fontsize=8.2, color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0F9FF", edgecolor="#BAE6FD", linewidth=0.9),
    )

    # Panel 3: 100% Stacked Physical Cluster Split (59.7k tokens/run)
    ax2 = axes[2]
    ax2.set_facecolor("#FFFFFF")
    sys_lbls = ["1. Today\n(Flat + Locked)", "2. Pure\nMarket", "3. TokenFed ★\n(Central Bank)"]
    xs = np.arange(len(sys_lbls))
    pub_sh  = np.array([44, 52, 62])
    corp_sh = np.array([17, 18, 18])
    trn_sh  = np.array([28, 28, 18])
    idle_sh = np.array([11,  2,  2])

    bw = 0.48
    ax2.bar(xs, pub_sh, width=bw, color="#0284C7", label="Customer + Science Pool", zorder=3)
    ax2.bar(xs, corp_sh, width=bw, bottom=pub_sh, color="#94A3B8", label="Corporate Blocks Used", zorder=3)
    ax2.bar(xs, trn_sh, width=bw, bottom=pub_sh + corp_sh, color="#334155", label="Internal Lab R&D", zorder=3)
    ax2.bar(
        xs, idle_sh, width=bw, bottom=pub_sh + corp_sh + trn_sh,
        color="#FCA5A5", edgecolor="#DC2626", linewidth=0.9, label="Idle Corporate Waste", zorder=3,
    )

    for idx in range(3):
        ax2.text(xs[idx], pub_sh[idx] / 2.0, f"{pub_sh[idx]}%", ha="center", va="center", fontsize=9.4, fontweight="bold", color="#FFFFFF", zorder=4)
        ax2.text(xs[idx], pub_sh[idx] + corp_sh[idx] / 2.0, f"{corp_sh[idx]}%", ha="center", va="center", fontsize=8.5, color="#FFFFFF", zorder=4)
        ax2.text(xs[idx], pub_sh[idx] + corp_sh[idx] + trn_sh[idx] / 2.0, f"{trn_sh[idx]}%", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#FFFFFF", zorder=4)
        if idle_sh[idx] >= 8:
            ax2.text(xs[idx], 100 - idle_sh[idx] / 2.0, f"{idle_sh[idx]}%", ha="center", va="center", fontsize=8.6, fontweight="bold", color="#991B1B", zorder=4)
        else:
            ax2.text(xs[idx], 101.6, f"{idle_sh[idx]}% idle", ha="center", va="bottom", fontsize=8.0, color="#B91C1C", zorder=4)

    ax2.set_ylim(0, 123)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(sys_lbls, fontsize=9.0, color="#1E293B", linespacing=1.15)
    ax2.get_xticklabels()[2].set_fontweight("bold")
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.8, color="#64748B")
    ax2.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax2.set_title("3. Physical GPU Cluster Split (100%)", fontsize=11.1, fontweight="bold", color="#0F172A", pad=12)
    ax2.legend(loc="upper left", ncol=2, frameon=False, fontsize=7.9, columnspacing=0.9, handlelength=1.2)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)

    fig.suptitle(
        "Where the Gain Actually Came From: Commercial Waste vs. R&D Flex + Science Reserve",
        fontsize=12.6,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.91], w_pad=2.2)
    fig.savefig(DECOMP_PNG, dpi=220)
    plt.close(fig)
    print(f"Wrote {DECOMP_PNG}")


def plot_architecture_diagram() -> None:
    """Render a clean 3-layer editorial schematic showing how TokenFed unbundles in reality."""
    _set_style()
    fig, ax = plt.subplots(figsize=(15.8, 6.1), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_card(x: float, y: float, w: float, h: float, fc: str, ec: str, lw: float = 1.2, r: float = 1.6) -> None:
        shadow = FancyBboxPatch(
            (x + 0.25, y - 0.45), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor="#0F172A", edgecolor="none", alpha=0.05, zorder=1,
        )
        ax.add_patch(shadow)
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
        )
        ax.add_patch(box)

    def draw_arrow(x1: float, y1: float, x2: float, y2: float, col: str = "#64748B", lw: float = 1.6) -> None:
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>,head_length=5.5,head_width=3.4",
            linewidth=lw, color=col, zorder=3,
        )
        ax.add_patch(arr)

    # Title & subtitle
    ax.text(
        50, 96.2,
        "How 'TokenFed' Unbundles in the Real World  (No Single Government Central Bank)",
        ha="center", va="center", fontsize=13.2, fontweight="bold", color="#0F172A",
    )
    ax.text(
        50, 91.6,
        "Humans don't day-trade token tickers — software routers trade spot/batch spreads across private lab treasuries and public science reserves.",
        ha="center", va="center", fontsize=9.4, color="#475569",
    )

    # Layer 1 Kicker
    ax.text(
        2.0, 86.2,
        "1. WORKLOADS & BUYERS  ·  Set simple urgency & budget rules (no human ticker-watching)",
        fontsize=8.4, fontweight="bold", color="#64748B",
    )

    # Top 3 Buyer Cards (y=64..84)
    top_cards = [
        (
            2.0, 64.0, 30.0, 20.0, "#F8FAFC", "#CBD5E1",
            "Enterprises (Forward Blocks & P0s)",
            "#334155",
            "• Pre-buy reserved capacity (PTUs / Scale Tier)\n• Auto-sweep idle night & weekend slack for rebates\n• Burst onto 2x+ Priority lane during live P0 outages",
        ),
        (
            35.0, 64.0, 30.0, 20.0, "#F8FAFC", "#CBD5E1",
            "Startups & Agent Teams (Builders)",
            "#0F172A",
            "• Tag urgency: ship live PRs <=2x; batch evals at 0.5x\n• Keep automated tests/verifiers running during spikes\n• Tap fintech working-capital credit post-cloud-credits",
        ),
        (
            68.0, 64.0, 30.0, 20.0, "#F0F9FF", "#7DD3FC",
            "Science & Public-Good Labs (Cash != Value)",
            "#0369A1",
            "• High societal upside (oncology, climate, open safety)\n• Grant-capped wallets get outbid in pure cash surges\n• Submit accredited runs to ring-fenced compute lanes",
        ),
    ]
    for x, y, w, h, fc, ec, title, tc, body in top_cards:
        draw_card(x, y, w, h, fc, ec)
        ax.text(x + 1.8, y + h - 3.6, title, ha="left", va="top", fontsize=9.6, fontweight="bold", color=tc, zorder=4)
        ax.text(x + 1.8, y + h - 8.0, body, ha="left", va="top", fontsize=8.35, color="#334155", linespacing=1.42, zorder=4)

    # Arrows Layer 1 -> Layer 2
    draw_arrow(17.0, 63.6, 17.0, 55.4, "#64748B")
    draw_arrow(50.0, 63.6, 50.0, 55.4, "#0284C7")
    draw_arrow(83.0, 63.6, 83.0, 55.4, "#0284C7")
    ax.text(18.2, 59.5, "Idle PTU sublease\n+ P0 burst orders", fontsize=7.8, color="#475569", va="center")
    ax.text(51.2, 59.5, "Policy-tagged\nagent jobs", fontsize=7.8, color="#475569", va="center")
    ax.text(84.2, 59.5, "Verified grant\nworkloads", fontsize=7.8, color="#0369A1", va="center")

    # Layer 2: Automated Clearinghouse / Router Bar (y=38..55)
    draw_card(2.0, 38.5, 96.0, 16.2, "#EFF6FF", "#38BDF8", lw=1.5)
    ax.text(
        4.0, 51.3,
        "2. AUTOMATED TREASURY ROUTERS & MULTI-LAB BROKERS   (Software Becomes the Trader)",
        ha="left", va="top", fontsize=9.8, fontweight="bold", color="#0C4A6E", zorder=4,
    )
    ax.text(
        4.0, 46.2,
        "• Agent harnesses & cross-lab routers (future OpenRouter / Bedrock / infra desks) read live spot, priority & overnight batch spreads\n"
        "• Arbitrage equivalent capability across OpenAI, Anthropic & Google  ·  Clear secondary corporate PTU subleases  ·  Route science to public quotas",
        ha="left", va="top", fontsize=8.55, color="#1E293B", linespacing=1.45, zorder=4,
    )

    # Arrows Layer 2 -> Layer 3
    draw_arrow(31.0, 38.1, 31.0, 29.4, "#D97706", lw=1.8)
    draw_arrow(78.0, 38.1, 78.0, 29.4, "#0284C7", lw=1.8)
    ax.text(32.3, 33.8, "Commercial spot (2x+), standard (1x)\n& async batch (0.5x) traffic", fontsize=7.9, color="#92400E", va="center")
    ax.text(79.3, 33.8, "Ring-fenced ~0.5x\naccredited science lane", fontsize=7.9, color="#0369A1", va="center")

    # Layer 3: Two Supply Pillars (y=4..29)
    # Left Pillar: Private Lab Compute Treasuries (~85-90% of GPUs)
    draw_card(2.0, 4.5, 56.5, 24.4, "#FFFBEB", "#F59E0B", lw=1.5)
    ax.text(
        4.0, 25.5,
        "3A. FRONTIER LAB 'COMPUTE TREASURIES'  (~85–90% of Cluster GPUs)",
        ha="left", va="top", fontsize=9.7, fontweight="bold", color="#92400E", zorder=4,
    )
    ax.text(
        4.0, 21.3,
        "Private Monetary Policy — Adopted Voluntarily Because It Raises Lab Yield (+75% Net API Rev in Sim)",
        ha="left", va="top", fontsize=8.3, fontweight="bold", color="#B45309", zorder=4,
    )
    ax.text(
        4.0, 16.8,
        "• Replaces flat sticker pricing + blind 429 walls with dynamic Priority/Spot vs. 0.5x Flex/Batch spreads\n"
        "• Flexes async RL rollouts, post-training, synthetic data & eval GPUs into daytime peaks (pre-training stays fixed)\n"
        "• Keeps automated test/verification lanes affordable so builders don't ship untested code during surges",
        ha="left", va="top", fontsize=8.35, color="#334155", linespacing=1.42, zorder=4,
    )

    # Right Pillar: Public Strategic Compute Reserves (~10-15% of GPUs)
    draw_card(61.0, 4.5, 37.0, 24.4, "#F0F9FF", "#0284C7", lw=1.6)
    ax.text(
        63.0, 25.5,
        "3B. PUBLIC STRATEGIC COMPUTE RESERVES",
        ha="left", va="top", fontsize=9.7, fontweight="bold", color="#075985", zorder=4,
    )
    ax.text(
        63.0, 21.3,
        "Fiscal Policy — NAIRR, NIH & Philanthropic Science Pools",
        ha="left", va="top", fontsize=8.3, fontweight="bold", color="#0284C7", zorder=4,
    )
    ax.text(
        63.0, 16.8,
        "• Contracts physical in-kind GPU shares ahead of time\n"
        "  (cash stipends alone just bid up commercial surge)\n"
        "• Captures the headline welfare gain (+16.1 Science pts)\n"
        "  that profit-maximizing labs won't subsidize alone",
        ha="left", va="top", fontsize=8.35, color="#1E293B", linespacing=1.42, zorder=4,
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(ARCH_PNG, dpi=220)
    plt.close(fig)
    print(f"Wrote {ARCH_PNG}")


def plot_2bar_chart() -> None:
    """Render the clean 2-system (Today vs. TokenFed) essay chart across overall value, cohorts, and shocks."""
    with open(DATA_JSON) as f:
        d = json.load(f)

    ov = d["overall"]
    worlds_data = d["worlds"]

    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")

    two_regimes = [
        ("flat_today", "Today\n(Flat + Locked)", "#94A3B8"),
        ("llm_fed",    "TokenFed ★\n(Central Bank)", "#0284C7"),
    ]

    # Panel 1: Overall Value on Identical GPUs (76.8 -> 99.6, +30%)
    ax0 = axes[0]
    ax0.set_facecolor("#FFFFFF")
    x0 = np.array([0.0, 0.85])
    vals0 = [ov[r[0]]["netVal"] for r in two_regimes]
    cols0 = [r[2] for r in two_regimes]
    lbls0 = [r[1] for r in two_regimes]

    bars0 = ax0.bar(x0, vals0, width=0.42, color=cols0, zorder=3)
    bars0[1].set_edgecolor("#0F172A")
    bars0[1].set_linewidth(1.6)
    ax0.set_xlim(-0.48, 1.35)
    ax0.set_ylim(0, 134)
    ax0.set_xticks(x0)
    ax0.set_xticklabels(lbls0, fontsize=9.5, color="#1E293B", linespacing=1.15)
    ax0.get_xticklabels()[1].set_fontweight("bold")
    ax0.set_yticks([0, 25, 50, 75, 100])
    ax0.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=8.8, color="#64748B")
    ax0.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax0.set_title("1. Overall Value (Same GPUs)", fontsize=11.3, fontweight="bold", color="#0F172A", pad=12)

    ax0.text(x0[0], vals0[0] + 2.3, f"{vals0[0]:.1f}", ha="center", va="bottom", fontsize=10.0, color="#334155")
    ax0.text(
        x0[1], vals0[1] + 2.3, f"{vals0[1]:.1f} (+30%)",
        ha="center", va="bottom", fontsize=10.2, fontweight="bold", color="#0F172A",
    )
    ax0.text(
        0.02, 0.96,
        "+49% customer output · +75% API rev\n11% → 2% idle corporate GPU waste",
        transform=ax0.transAxes, ha="left", va="top", fontsize=8.0, color="#0F172A", linespacing=1.32,
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#F0F9FF", edgecolor="#BAE6FD", linewidth=0.9),
    )

    # Panel 2: Jobs Finished On Time Across Cohorts (%)
    ax1 = axes[1]
    ax1.set_facecolor("#FFFFFF")
    cohorts = [
        ("hiPct",   "All Urgent\nJobs"),
        ("strtPct", "Seed\nStartups"),
        ("sciPct",  "Science\nLabs"),
        ("entPct",  "Enterprise\nP0s"),
    ]
    xc = np.arange(len(cohorts))
    wc = 0.34
    t_vals = [ov["flat_today"][k] for k, _ in cohorts]
    f_vals = [ov["llm_fed"][k] for k, _ in cohorts]

    b_t = ax1.bar(xc - wc / 2, t_vals, width=wc, color="#94A3B8", label="Today (Flat + Locked)", zorder=3)
    b_f = ax1.bar(xc + wc / 2, f_vals, width=wc, color="#0284C7", edgecolor="#0F172A", linewidth=1.1, label="TokenFed ★", zorder=3)
    ax1.set_ylim(0, 122)
    ax1.set_xticks(xc)
    ax1.set_xticklabels([lbl for _, lbl in cohorts], fontsize=9.0, color="#1E293B", linespacing=1.15)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.8, color="#64748B")
    ax1.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax1.set_title("2. Jobs Finished On Time (%)", fontsize=11.3, fontweight="bold", color="#0F172A", pad=12)
    ax1.legend(loc="upper left", ncol=2, frameon=False, fontsize=8.4, columnspacing=1.0)

    for bt, bf, vt, vf in zip(b_t, b_f, t_vals, f_vals):
        ax1.text(bt.get_x() + bt.get_width() / 2.0, vt + 2.0, f"{vt}%", ha="center", va="bottom", fontsize=8.6, color="#475569")
        ax1.text(bf.get_x() + bf.get_width() / 2.0, vf + 2.0, f"{vf}%", ha="center", va="bottom", fontsize=9.3, fontweight="bold", color="#0F172A")

    # Panel 3: Score Across the 3 Macro Shocks
    ax2 = axes[2]
    ax2.set_facecolor("#FFFFFF")
    xw = np.arange(len(WORLDS))
    ws = 0.34
    wt_vals = [worlds_data[wk]["regimes"]["flat_today"]["netVal"] for wk, _ in WORLDS]
    wf_vals = [worlds_data[wk]["regimes"]["llm_fed"]["netVal"] for wk, _ in WORLDS]

    bw_t = ax2.bar(xw - ws / 2, wt_vals, width=ws, color="#94A3B8", label="Today", zorder=3)
    bw_f = ax2.bar(xw + ws / 2, wf_vals, width=ws, color="#0284C7", edgecolor="#0F172A", linewidth=1.2, label="TokenFed ★", zorder=3)
    ax2.set_ylim(0, 126)
    ax2.set_xticks(xw)
    ax2.set_xticklabels([wl for _, wl in WORLDS], fontsize=9.2, color="#1E293B", linespacing=1.15)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=8.8, color="#64748B")
    ax2.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax2.set_title("3. Score Across Three Shocks", fontsize=11.3, fontweight="bold", color="#0F172A", pad=12)
    ax2.legend(loc="upper left", ncol=2, frameon=False, fontsize=8.4, columnspacing=1.0)

    for bt, bf, vt, vf in zip(bw_t, bw_f, wt_vals, wf_vals):
        ax2.text(bt.get_x() + bt.get_width() / 2.0, vt + 2.0, f"{vt:.0f}", ha="center", va="bottom", fontsize=8.8, color="#475569")
        ax2.text(bf.get_x() + bf.get_width() / 2.0, vf + 2.0, f"{vf:.0f}", ha="center", va="bottom", fontsize=9.6, fontweight="bold", color="#0F172A")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)

    fig.suptitle(
        "Today vs. TokenFed  (Same Physical GPU Cluster · 12 Autonomous AI Firms)",
        fontsize=12.8,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.91], w_pad=2.4)
    fig.savefig(TWO_BAR_PNG, dpi=220)
    plt.close(fig)
    print(f"Wrote {TWO_BAR_PNG}")


def plot_decomposition_2bar_chart() -> None:
    """Render the 2-regime (Today -> TokenFed, no Pure Market) gain decomposition & GPU split chart."""
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.75), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")

    # Panel 1: Cohort Value Deltas Today -> TokenFed (+22.8 Net: 76.8 -> 99.6)
    ax0 = axes[0]
    ax0.set_facecolor("#FFFFFF")
    cats = ["Science\nLabs", "Seed\nStartups", "Enterprise\nP0s", "Slop\nSwarms", "Lab R&D\n(Flexed)"]
    xc = np.arange(len(cats))
    deltas = [11.2, 10.7, 3.7, 2.7, -5.5]
    cols = ["#0284C7", "#059669", "#38BDF8", "#F59E0B", "#475569"]

    bars = ax0.bar(xc, deltas, width=0.54, color=cols, zorder=3)
    bars[0].set_edgecolor("#0F172A")
    bars[0].set_linewidth(1.4)
    bars[1].set_edgecolor("#064E3B")
    bars[1].set_linewidth(1.2)
    ax0.axhline(0, color="#64748B", linewidth=1.0, zorder=2)
    ax0.set_ylim(-9.5, 19.5)
    ax0.set_xticks(xc)
    ax0.set_xticklabels(cats, fontsize=8.6, color="#1E293B", linespacing=1.15)
    ax0.get_xticklabels()[0].set_fontweight("bold")
    ax0.get_xticklabels()[1].set_fontweight("bold")
    ax0.set_yticks([-5, 0, 5, 10, 15])
    ax0.set_yticklabels(["-5", "0", "+5", "+10", "+15"], fontsize=8.8, color="#64748B")
    ax0.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax0.set_title("1. Value Gain by Group (+22.8 Net)", fontsize=11.2, fontweight="bold", color="#0F172A", pad=12)

    txt_cols = ["#0F172A", "#065F46", "#0369A1", "#92400E", "#334155"]
    for i, (b, v) in enumerate(zip(bars, deltas)):
        if v >= 0:
            ax0.text(
                b.get_x() + b.get_width() / 2.0, v + 0.55, f"+{v:.1f}",
                ha="center", va="bottom",
                fontsize=9.6 if i < 2 else 8.9,
                fontweight="bold" if i < 2 else "normal",
                color=txt_cols[i],
            )
        else:
            ax0.text(
                b.get_x() + b.get_width() / 2.0, v - 0.65, f"{v:.1f}",
                ha="center", va="top", fontsize=9.3, fontweight="bold", color=txt_cols[i],
            )

    ax0.text(
        0.98, 0.95,
        "+28.3 external output (+49%)\nfrom R&D flex (-5.5) + reclaimed GPUs",
        transform=ax0.transAxes, ha="right", va="top", fontsize=8.0, color="#0F172A", linespacing=1.32,
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#F0F9FF", edgecolor="#BAE6FD", linewidth=0.9),
    )

    # Panel 2: 100% Stacked Physical Cluster Split (Today vs. TokenFed)
    ax1 = axes[1]
    ax1.set_facecolor("#FFFFFF")
    sys_lbls = ["Today\n(Flat + Locked)", "TokenFed ★\n(Central Bank)"]
    xs = np.array([0.0, 0.85])
    pub_sh  = np.array([44, 62])
    corp_sh = np.array([17, 18])
    trn_sh  = np.array([28, 18])
    idle_sh = np.array([11,  2])

    bw = 0.44
    ax1.bar(xs, pub_sh, width=bw, color="#0284C7", label="Customer + Science Pool", zorder=3)
    ax1.bar(xs, corp_sh, width=bw, bottom=pub_sh, color="#94A3B8", label="Corporate Blocks Used", zorder=3)
    ax1.bar(xs, trn_sh, width=bw, bottom=pub_sh + corp_sh, color="#334155", label="Internal Lab R&D", zorder=3)
    ax1.bar(
        xs, idle_sh, width=bw, bottom=pub_sh + corp_sh + trn_sh,
        color="#FCA5A5", edgecolor="#DC2626", linewidth=0.9, label="Idle Corporate Waste", zorder=3,
    )

    for idx in range(2):
        ax1.text(xs[idx], pub_sh[idx] / 2.0, f"{pub_sh[idx]}%", ha="center", va="center", fontsize=9.6, fontweight="bold", color="#FFFFFF", zorder=4)
        ax1.text(xs[idx], pub_sh[idx] + corp_sh[idx] / 2.0, f"{corp_sh[idx]}%", ha="center", va="center", fontsize=8.7, color="#FFFFFF", zorder=4)
        ax1.text(xs[idx], pub_sh[idx] + corp_sh[idx] + trn_sh[idx] / 2.0, f"{trn_sh[idx]}%", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#FFFFFF", zorder=4)
        if idle_sh[idx] >= 8:
            ax1.text(xs[idx], 100 - idle_sh[idx] / 2.0, f"{idle_sh[idx]}%", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#991B1B", zorder=4)
        else:
            ax1.text(xs[idx], 101.6, f"{idle_sh[idx]}% idle (1/5th)", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#B91C1C", zorder=4)

    ax1.set_xlim(-0.48, 1.35)
    ax1.set_ylim(0, 126)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(sys_lbls, fontsize=9.4, color="#1E293B", linespacing=1.15)
    ax1.get_xticklabels()[1].set_fontweight("bold")
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.8, color="#64748B")
    ax1.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax1.set_title("2. Physical GPU Cluster Split (100%)", fontsize=11.2, fontweight="bold", color="#0F172A", pad=12)
    ax1.legend(loc="upper left", ncol=2, frameon=False, fontsize=7.9, columnspacing=0.9, handlelength=1.2)

    # Panel 3: Net Lab API Revenue on Identical GPUs (19.9k -> 34.8k, +75%)
    ax2 = axes[2]
    ax2.set_facecolor("#FFFFFF")
    xr = np.array([0.0, 0.85])
    rev_vals = [19.86, 34.83]
    rev_cols = ["#94A3B8", "#0284C7"]

    rbars = ax2.bar(xr, rev_vals, width=0.42, color=rev_cols, zorder=3)
    rbars[1].set_edgecolor("#0F172A")
    rbars[1].set_linewidth(1.6)
    ax2.set_xlim(-0.48, 1.35)
    ax2.set_ylim(0, 48.0)
    ax2.set_xticks(xr)
    ax2.set_xticklabels(sys_lbls, fontsize=9.4, color="#1E293B", linespacing=1.15)
    ax2.get_xticklabels()[1].set_fontweight("bold")
    ax2.set_yticks([0, 10, 20, 30, 40])
    ax2.set_yticklabels(["0", "10k", "20k", "30k", "40k"], fontsize=8.8, color="#64748B")
    ax2.grid(axis="y", linestyle="--", alpha=0.35, color="#CBD5E1", zorder=0)
    ax2.set_title("3. Net Lab API Revenue (Same GPUs)", fontsize=11.2, fontweight="bold", color="#0F172A", pad=12)

    ax2.text(xr[0], rev_vals[0] + 0.9, "19.9k", ha="center", va="bottom", fontsize=10.0, color="#334155")
    ax2.text(
        xr[1], rev_vals[1] + 0.9, "34.8k (+75%)",
        ha="center", va="bottom", fontsize=10.2, fontweight="bold", color="#0F172A",
    )
    ax2.text(
        0.02, 0.95,
        "~30% more earned per token (1.28x)\n44 → 0 blind 429 rate-limit lockouts",
        transform=ax2.transAxes, ha="left", va="top", fontsize=8.0, color="#0F172A", linespacing=1.32,
        bbox=dict(boxstyle="round,pad=0.34", facecolor="#F0F9FF", edgecolor="#BAE6FD", linewidth=0.9),
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)

    fig.suptitle(
        "Where the Gain Actually Came From: Today → TokenFed  (Same Physical GPU Cluster)",
        fontsize=12.7,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.0, 1, 0.91], w_pad=2.3)
    fig.savefig(DECOMP_2BAR_PNG, dpi=220)
    plt.close(fig)
    print(f"Wrote {DECOMP_2BAR_PNG}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render TokenFed figures.")
    parser.add_argument("--out-dir", default=DEFAULT_FIG_DIR, help="Output directory for generated PNGs")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    OUT_PNG = os.path.join(args.out_dir, "token_fed_chart.png")
    TWO_BAR_PNG = os.path.join(args.out_dir, "token_fed_2bar.png")
    BOOM_PNG = os.path.join(args.out_dir, "token_economy_boom.png")
    DECOMP_PNG = os.path.join(args.out_dir, "token_fed_decomposition.png")
    DECOMP_2BAR_PNG = os.path.join(args.out_dir, "token_fed_decomposition_2bar.png")
    ARCH_PNG = os.path.join(args.out_dir, "token_fed_architecture.png")
    main()
    plot_2bar_chart()
    plot_boom_chart()
    plot_decomposition_chart()
    plot_decomposition_2bar_chart()
    plot_architecture_diagram()
