"""Run the Studio 08 benign-overfitting simulation.

The script creates the three required simulation regimes and saves one
pandas DataFrame with columns

    nsim, gamma, mse, se

to ``simulation_results.pkl`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# Fixed experimental setup from the handout.
N = 200
R2 = 5.0


def make_gamma_grid(nsim: int) -> np.ndarray:
    """Return the required gamma values for one replication regime."""
    if nsim == 1:
        # 5000 scenarios, logarithmically spaced over [0.1, 10].
        return np.logspace(np.log10(0.1), np.log10(10.0), 5000)
    if nsim == 50:
        # 100 scenarios, logarithmically spaced over [0.1, 10].
        return np.logspace(np.log10(0.1), np.log10(10.0), 100)
    if nsim == 1000:
        # Five scenarios specified by the handout.
        return np.array([0.2, 0.5, 0.8, 2.0, 5.0])
    raise ValueError("nsim must be one of {1, 50, 1000}")


def simulate_once(n: int, gamma: float, r2: float, rng: np.random.Generator) -> float:
    """Simulate one data set, fit ridgeless least squares, and return risk.

    The model is
        y = X beta + epsilon,
    with iid N(0,1) entries in X and epsilon, and
        beta = sqrt(r2 / p) * 1_p.

    For isotropic features, the requested prediction risk is
        ||beta_hat - beta||_2^2.
    """
    # [gamma * n] in the handout.  round() also matches the convention used
    # in the previous studio solution.
    p = max(1, int(round(gamma * n)))

    beta = np.full(p, np.sqrt(r2 / p))
    X = rng.standard_normal((n, p))
    epsilon = rng.standard_normal(n)
    y = X @ beta + epsilon

    # sklearn returns ordinary least squares when p < n and the minimum-L2
    # norm least-squares solution when p >= n.
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    beta_hat = model.coef_

    return float(np.sum((beta_hat - beta) ** 2))


def run_scenario(
    nsim: int,
    gamma: float,
    n: int,
    r2: float,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Run all replications for one gamma value and summarize the risk."""
    risks = np.empty(nsim)

    for rep in range(nsim):
        risks[rep] = simulate_once(n=n, gamma=gamma, r2=r2, rng=rng)

    mse = float(risks.mean())
    se = 0.0 if nsim == 1 else float(risks.std(ddof=1) / np.sqrt(nsim))

    return {
        "nsim": nsim,
        "gamma": float(gamma),
        "mse": mse,
        "se": se,
    }


def run_simulation(seed: int = 20261022) -> pd.DataFrame:
    """Run all three regimes and return the complete results DataFrame."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    for nsim in (1, 50, 1000):
        gammas = make_gamma_grid(nsim)
        print(f"Running nsim={nsim}: {len(gammas)} scenarios")

        for i, gamma in enumerate(gammas, start=1):
            rows.append(run_scenario(nsim, gamma, N, R2, rng))

            # Occasional progress messages, useful for the 5000-scenario run.
            if i % 250 == 0 or i == len(gammas):
                print(f"  completed {i}/{len(gammas)} scenarios")

    return pd.DataFrame(rows, columns=["nsim", "gamma", "mse", "se"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=20261022,
        help="random seed (default: 20261022)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("simulation_results.pkl"),
        help="output pickle path (default: simulation_results.pkl)",
    )
    args = parser.parse_args()

    results = run_simulation(seed=args.seed)
    results.to_pickle(args.out)

    print(f"\nSaved {len(results)} rows to {args.out}")
    print(results.groupby("nsim").size())


if __name__ == "__main__":
    main()
