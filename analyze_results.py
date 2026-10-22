"""Studio 08: load scenario summaries and plot ridgeless regression risk.

Run after your teammate's simulation:
    python analyze_results.py --input simulation_results.pkl --output figure.pdf

The pickle must contain one pandas DataFrame row per scenario with columns
nsim, gamma, mse, se. The nsim column is normalized to n_sim internally;
the previously used n_sim spelling and common MSE/SE names are also accepted.
"""

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess


REGIMES = (1, 50, 1000)
SCENARIO_COUNTS = {1: 5000, 50: 100, 1000: 5}
COLUMN_NAMES = {
    "n_sim": ("n_sim", "nsim"),
    "gamma": ("gamma",),
    "mse": ("mse", "MSE", "mse_hat", "MSE_hat", "estimated_mse"),
    "se": ("se", "SE", "mse_se", "se_mse", "se_MSE", "se(MSE)"),
}


def load_results(path):
    """Read and validate all 5,105 scenario summaries without pooling rows."""
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("The pickle must contain a pandas DataFrame.")

    renamed = {}
    for canonical, alternatives in COLUMN_NAMES.items():
        matches = [name for name in alternatives if name in frame.columns]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {canonical!r} column from {alternatives}; "
                f"found {matches}. Columns present: {list(frame.columns)}"
            )
        renamed[matches[0]] = canonical
    data = frame.rename(columns=renamed)[list(COLUMN_NAMES)].copy()
    for column in data:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if not np.isfinite(data[["n_sim", "gamma", "mse"]].to_numpy(dtype=float)).all():
        raise ValueError("n_sim, gamma, and mse must all be finite.")
    if not data["n_sim"].isin(REGIMES).all():
        raise ValueError("n_sim must be one of 1, 50, or 1000.")
    data["n_sim"] = data["n_sim"].astype(int)
    if not data["gamma"].between(0.1 - 1e-12, 10 + 1e-12).all():
        raise ValueError("gamma must lie in [0.1, 10].")
    if (data["mse"] < 0).any():
        raise ValueError("MSE values cannot be negative.")

    for repetitions, expected in SCENARIO_COUNTS.items():
        observed = int(data["n_sim"].eq(repetitions).sum())
        if observed != expected:
            raise ValueError(
                f"n_sim={repetitions}: expected {expected} scenario rows, "
                f"found {observed}. Load the complete scenario summary, "
                "not the replicate-level table or several concatenated runs."
            )

    # A sample SD is undefined for one replicate. The handout specifies SE=0.
    single = data["n_sim"].eq(1)
    if not data.loc[single, "se"].eq(0).all():
        warnings.warn("Setting se=0 for n_sim=1, as required by the handout.",
                      UserWarning, stacklevel=2)
    data.loc[single, "se"] = 0.0
    repeated_se = data.loc[~single, "se"].to_numpy(dtype=float)
    if not np.isfinite(repeated_se).all() or (repeated_se < 0).any():
        raise ValueError("For n_sim>1, se must be finite and nonnegative.")

    fixed_grid = np.sort(data.loc[data["n_sim"].eq(1000), "gamma"].to_numpy())
    if not np.allclose(fixed_grid, [0.2, 0.5, 0.8, 2, 5], rtol=0, atol=1e-12):
        raise ValueError("n_sim=1000 requires gamma = {0.2, 0.5, 0.8, 2, 5}.")

    # Do not group by gamma: integer p can cause several distinct scenarios
    # to have the same actual p/n, especially in the 5,000-scenario regime.
    return data.sort_values(["n_sim", "gamma"], kind="stable").reset_index(drop=True)


def theoretical_risk(gamma):
    """Asymptotic risk for isotropic features, sigma^2=1, and r^2=5.

    At gamma=1 the curve is undefined; NaN prevents a line across the pole.
    """
    gamma = np.asarray(gamma, dtype=float)
    if not np.isfinite(gamma).all() or (gamma <= 0).any():
        raise ValueError("Theoretical risk requires finite, positive gamma.")
    risk = np.full_like(gamma, np.nan)
    left, right = gamma < 1, gamma > 1
    risk[left] = gamma[left] / (1 - gamma[left])
    risk[right] = 5 * (1 - 1 / gamma[right]) + 1 / (gamma[right] - 1)
    return risk


def smooth_single_replicates(data, frac=0.1):
    """Smooth original MSE versus original gamma separately on each side."""
    if not 0 < frac <= 1:
        raise ValueError("LOWESS frac must lie in (0, 1].")
    single = data.loc[data["n_sim"].eq(1)]
    curves = []
    for mask in (single["gamma"] < 1, single["gamma"] > 1):
        side = single.loc[mask].sort_values("gamma", kind="stable")
        if len(side) * frac < 3:
            raise ValueError("LOWESS needs at least three local points per side.")
        curves.append(lowess(
            endog=side["mse"].to_numpy(dtype=float),
            exog=side["gamma"].to_numpy(dtype=float),
            frac=frac, it=0, return_sorted=True,
        ))
    return curves


def make_figure(data, ymax=10.0, frac=0.1):
    """Return the required three-panel figure; keep raw risk values intact."""
    if not np.isfinite(ymax) or ymax <= 0:
        raise ValueError("ymax must be finite and positive.")
    smooth_curves = smooth_single_replicates(data, frac=frac)
    left = np.geomspace(0.1, 1 - 1e-4, 700)
    right = np.geomspace(1 + 1e-4, 10, 700)
    blue, orange = "#0072B2", "#D55E00"
    style = {
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.2), sharey=True)
        errorbar_handle = None
        for ax, repetitions in zip(axes, REGIMES):
            subset = data.loc[data["n_sim"].eq(repetitions)]
            # Always plot true values. The display limits only clip the view.
            ax.scatter(subset["gamma"], subset["mse"], color=blue,
                       s=4 if repetitions == 1 else 20,
                       alpha=0.12 if repetitions == 1 else 0.9,
                       linewidths=0, zorder=2)
            if repetitions == 1:
                for curve in smooth_curves:
                    ax.plot(curve[:, 0], curve[:, 1], color=orange,
                            linewidth=1.6, label="LOWESS", zorder=4)
            else:
                errorbar_handle = ax.errorbar(
                    subset["gamma"], subset["mse"],
                    yerr=2 * subset["se"], fmt="none", ecolor=blue,
                    elinewidth=0.9, capsize=2, zorder=3,
                )

            # Fit/plot the two branches separately, never across gamma=1.
            for grid in (left, right):
                ax.plot(grid, theoretical_risk(grid), color="0.12",
                        linestyle="--", linewidth=1.5,
                        label="Theoretical risk", zorder=5)
            ax.axvline(1, color="0.45", linestyle=":", linewidth=1.0,
                       label="Interpolation threshold", zorder=1)

            above = subset["mse"] > ymax
            if above.any():
                ax.scatter(subset.loc[above, "gamma"],
                           np.full(int(above.sum()), 0.985 * ymax),
                           color=blue, marker="^", s=12, alpha=0.55, zorder=6)
            ax.text(0.98, 0.96, f"Above {ymax:g}: {int(above.sum()):,} points",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color="0.35")
            ax.set_xscale("log")
            ax.set_xlim(0.1, 10)
            ax.set_ylim(0, ymax)
            ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10],
                          labels=["0.1", "0.2", "0.5", "1", "2", "5", "10"])
            ax.set_title(rf"$n_{{\mathrm{{sim}}}}={repetitions}$"
                         + f"\n{len(subset):,} scenarios")
            ax.set_xlabel(r"Aspect ratio $\gamma=p/n$ (log scale)")
            ax.grid(axis="y", color="0.9", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

        axes[0].set_ylabel(r"Estimated risk: $\|\widehat\beta-\beta\|_2^2$")
        handles = [
            Line2D([], [], color=blue, marker="o", linestyle="none",
                   markersize=4, label="Estimated risk"),
            Line2D([], [], color=orange, linewidth=1.6, label="LOWESS (one replicate)"),
            Line2D([], [], color="0.12", linestyle="--", label="Theoretical risk"),
            errorbar_handle,
        ]
        fig.legend(handles, ["Estimated risk", "LOWESS (one replicate)",
                             "Theoretical risk", r"$\pm2$ Monte Carlo SE"],
                   loc="lower center", bbox_to_anchor=(0.52, 0.105),
                   ncol=4, frameon=False, fontsize=8)
        fig.suptitle(r"Ridgeless least squares: $n=200$, $r^2=5$", y=0.98, fontsize=12)
        fig.text(0.52, 0.065,
                 f"LOWESS: frac={frac:g}, it=0; separate fits for gamma < 1 and gamma > 1.",
                 ha="center", fontsize=8, color="0.3")
        fig.text(0.52, 0.025,
                 f"View limited to risk <= {ymax:g}; triangles mark estimates above this limit. "
                 "Raw values are retained.", ha="center", fontsize=8, color="0.3")
        fig.subplots_adjust(left=0.07, right=0.985, bottom=0.30, top=0.80, wspace=0.12)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("simulation_results.pkl"))
    parser.add_argument("--output", type=Path, default=Path("figure.pdf"))
    parser.add_argument("--ymax", type=float, default=10.0,
                        help="Upper display limit; does not change data or LOWESS.")
    parser.add_argument("--frac", type=float, default=0.1,
                        help="Fraction of neighbors used by LOWESS (default: 0.1).")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf for the required deliverable.")

    # The analysis reads the saved summaries; it never fits a regression model.
    data = load_results(args.input)
    fig = make_figure(data, ymax=args.ymax, frac=args.frac)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    for repetitions in REGIMES:
        subset = data.loc[data["n_sim"].eq(repetitions)]
        print(f"n_sim={repetitions}: {len(subset):,} scenarios, "
              f"{int((subset['mse'] > args.ymax).sum()):,} estimates above the display limit")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
