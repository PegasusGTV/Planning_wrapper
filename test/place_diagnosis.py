"""
Visualize placement outcomes (success + failure) from eval log files.
Produces three grid heatmaps (auto, mpc, ppo) where each cell is colored
by failure rate and annotated with "fail/total" counts.

Usage:
    python place_diagnosis.py eval_results/eval_20260419_174125.txt
    python place_diagnosis.py <any_eval_log.txt>
"""

import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── parsing ───────────────────────────────────────────────────────────────────

RUN_HEADER   = re.compile(r'\[(\d+)/(\d+)\]\s+skill=(\w+)\s+env=(\S+)\s+seed=(\d+)')
PLACE_TARGET = re.compile(r'\s+place\s+\(obstacle-at\s+\S+\s+(r_(\d+)_(\d+))\)')
PLACE_FAIL   = re.compile(r'\s+→ FAIL')
PLACE_OK     = re.compile(r'\s+→ OK')


def parse_outcomes(path: str) -> dict[str, dict[str, dict[str, int]]]:
    """
    Return {skill: {cell: {'ok': int, 'fail': int}}}.
    Cell names are like 'r_ROW_COL'.
    """
    outcomes: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {'ok': 0, 'fail': 0})
    )

    current_skill: str | None = None
    pending_cell:  str | None = None

    with open(path) as f:
        for line in f:
            m = RUN_HEADER.match(line)
            if m:
                current_skill = m.group(3)
                pending_cell  = None
                continue

            if current_skill is None:
                continue

            m = PLACE_TARGET.match(line)
            if m:
                pending_cell = m.group(1)   # e.g. 'r_4_9'
                continue

            if pending_cell is not None:
                if PLACE_FAIL.match(line):
                    outcomes[current_skill][pending_cell]['fail'] += 1
                    pending_cell = None
                elif PLACE_OK.match(line):
                    outcomes[current_skill][pending_cell]['ok'] += 1
                    pending_cell = None

    return outcomes


# ── plotting ──────────────────────────────────────────────────────────────────

SKILL_ORDER = ['auto', 'mpc', 'ppo']


def cell_to_rc(cell: str) -> tuple[int, int]:
    """'r_4_9' → (4, 9)"""
    _, r, c = cell.split('_')
    return int(r), int(c)


def plot_heatmaps(outcomes: dict, log_path: str) -> None:
    # Determine global grid extents from all cells across all skills.
    all_cells = [cell for skill_data in outcomes.values() for cell in skill_data]
    if not all_cells:
        print("No place actions found in log.")
        return

    rows = [cell_to_rc(c)[0] for c in all_cells]
    cols = [cell_to_rc(c)[1] for c in all_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    # Grid layout: first number (r) → horizontal x-axis
    #              second number (c) → vertical y-axis, higher c = bottom (farther from robot)
    grid_h = max_c - min_c + 1   # vertical dimension (second number)
    grid_w = max_r - min_r + 1   # horizontal dimension (first number)

    n_skills = len(SKILL_ORDER)
    cell_px   = 0.8          # inches per grid cell
    fig_w     = grid_w * cell_px * n_skills + 3
    fig_h     = grid_h * cell_px + 1.5

    fig, axes = plt.subplots(1, n_skills, figsize=(fig_w, fig_h))
    if n_skills == 1:
        axes = [axes]

    # Colormap: white (0 % failure) → red (100 % failure).
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'fail_rate', ['#f7f7f7', '#fc8d59', '#d73027']
    )

    for ax, skill in zip(axes, SKILL_ORDER):
        skill_data = outcomes.get(skill, {})

        # Build 2-D arrays for failure rate and annotation text.
        # Shape: (grid_h, grid_w) = (second-number extent, first-number extent)
        # Row index  → second number (c), increases downward → higher c at bottom
        # Col index  → first number (r), increases rightward
        rate_grid  = np.full((grid_h, grid_w), np.nan)
        annot_grid = np.full((grid_h, grid_w), '', dtype=object)

        for cell, counts in skill_data.items():
            r, c = cell_to_rc(cell)
            ri, ci = c - min_c, r - min_r   # c drives vertical, r drives horizontal
            total = counts['ok'] + counts['fail']
            rate  = counts['fail'] / total if total > 0 else 0.0
            rate_grid[ri, ci]  = rate
            annot_grid[ri, ci] = f"{counts['fail']}/{total}"

        # Draw cells.
        im = ax.imshow(
            np.where(np.isnan(rate_grid), -0.05, rate_grid),
            cmap=cmap, vmin=0, vmax=1,
            aspect='equal', interpolation='none',
        )

        # Annotate each cell.
        for ri in range(grid_h):
            for ci in range(grid_w):
                txt = annot_grid[ri, ci]
                if txt:
                    rate = rate_grid[ri, ci]
                    color = 'white' if rate > 0.55 else '#333333'
                    ax.text(ci, ri, txt, ha='center', va='center',
                            fontsize=7, color=color, fontweight='bold')

        # Axes: first number (r) along x, second number (c) along y (0=top, max=bottom)
        ax.set_xticks(range(grid_w))
        ax.set_xticklabels([str(min_r + i) for i in range(grid_w)], fontsize=7)
        ax.set_yticks(range(grid_h))
        ax.set_yticklabels([str(min_c + i) for i in range(grid_h)], fontsize=7)
        ax.set_xlabel("r_X_*  (first index)", fontsize=8)
        ax.set_ylabel("r_*_Y  (second index, higher = farther)", fontsize=8)

        total_fail  = sum(v['fail'] for v in skill_data.values())
        total_place = sum(v['ok'] + v['fail'] for v in skill_data.values())
        ax.set_title(
            f"{skill.upper()}   {total_fail} fail / {total_place} total",
            fontsize=10, pad=6
        )

        # Colorbar.
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('failure rate', fontsize=7)
        cb.ax.tick_params(labelsize=7)
        # Mark NaN cells (never visited) with a hatched overlay.
        nan_mask = np.ma.masked_where(~np.isnan(rate_grid), np.ones_like(rate_grid))
        ax.imshow(nan_mask, cmap=mcolors.ListedColormap(['#cccccc']),
                  aspect='equal', interpolation='none', alpha=0.6)

    fig.suptitle(f"Place outcome heatmap — {log_path}", fontsize=11, y=1.01)
    plt.tight_layout()

    out = log_path.replace('.txt', '_place_heatmap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved → {out}")
    plt.show()


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python place_diagnosis.py <eval_log.txt>")
        sys.exit(1)

    log = sys.argv[1]
    outcomes = parse_outcomes(log)

    for skill in SKILL_ORDER:
        data = outcomes.get(skill, {})
        total_f = sum(v['fail'] for v in data.values())
        total_t = sum(v['ok'] + v['fail'] for v in data.values())
        print(f"[{skill:4s}]  {total_f:3d} fail / {total_t:3d} total  "
              f"({100*total_f/total_t:.1f}%)" if total_t else f"[{skill:4s}]  no data")
        for cell, v in sorted(data.items(), key=lambda kv: -kv[1]['fail'])[:10]:
            tot = v['ok'] + v['fail']
            print(f"        {cell:12s}  {v['fail']}/{tot}  "
                  f"({100*v['fail']/tot:.0f}%)")

    plot_heatmaps(outcomes, log)
