"""Hypothesis testing for scientific reasoning.

Statistical tests for comparing experimental results, validating models,
and testing quantum measurement distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class HypothesisTest:
    """Result of a hypothesis test."""

    test_name: str
    statistic: float
    p_value: float
    alpha: float = 0.05
    null_hypothesis: str = ""
    alt_hypothesis: str = ""

    @property
    def reject_null(self) -> bool:
        return self.p_value < self.alpha

    @property
    def conclusion(self) -> str:
        if self.reject_null:
            return f"Reject H₀ (p={self.p_value:.4f} < α={self.alpha})"
        return f"Fail to reject H₀ (p={self.p_value:.4f} ≥ α={self.alpha})"

    def __repr__(self) -> str:
        return (
            f"HypothesisTest({self.test_name}: "
            f"stat={self.statistic:.4f}, p={self.p_value:.4f}, "
            f"{'reject' if self.reject_null else 'fail to reject'} H₀)"
        )


def chi_squared_test(
    observed: np.ndarray | list[float],
    expected: np.ndarray | list[float],
    alpha: float = 0.05,
) -> HypothesisTest:
    """Chi-squared goodness-of-fit test.

    Tests whether observed frequencies match expected frequencies.
    Commonly used to test if quantum measurement outcomes match
    theoretical probability distributions.
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)

    if len(observed) != len(expected):
        raise ValueError("Observed and expected must have the same length")

    stat, p_value = stats.chisquare(observed, f_exp=expected)

    return HypothesisTest(
        test_name="Chi-squared",
        statistic=float(stat),
        p_value=float(p_value),
        alpha=alpha,
        null_hypothesis="Observed matches expected distribution",
        alt_hypothesis="Observed does not match expected distribution",
    )


def t_test(
    sample1: np.ndarray | list[float],
    sample2: np.ndarray | list[float] | None = None,
    mu: float = 0.0,
    alpha: float = 0.05,
) -> HypothesisTest:
    """Student's t-test.

    One-sample t-test (sample1 vs mu) or two-sample independent t-test.
    """
    sample1 = np.asarray(sample1, dtype=float)

    if sample2 is not None:
        sample2 = np.asarray(sample2, dtype=float)
        stat, p_value = stats.ttest_ind(sample1, sample2)
        null = "The two sample means are equal"
        alt = "The two sample means are different"
        name = "Two-sample t-test"
    else:
        stat, p_value = stats.ttest_1samp(sample1, mu)
        null = f"Sample mean equals {mu}"
        alt = f"Sample mean does not equal {mu}"
        name = "One-sample t-test"

    return HypothesisTest(
        test_name=name,
        statistic=float(stat),
        p_value=float(p_value),
        alpha=alpha,
        null_hypothesis=null,
        alt_hypothesis=alt,
    )


def ks_test(
    sample: np.ndarray | list[float],
    distribution: str = "norm",
    alpha: float = 0.05,
    **dist_params: Any,
) -> HypothesisTest:
    """Kolmogorov-Smirnov test for distribution fitting.

    Tests whether a sample follows a given distribution.
    """
    sample = np.asarray(sample, dtype=float)
    stat, p_value = stats.kstest(sample, distribution, args=tuple(dist_params.values()))

    return HypothesisTest(
        test_name="Kolmogorov-Smirnov",
        statistic=float(stat),
        p_value=float(p_value),
        alpha=alpha,
        null_hypothesis=f"Sample follows {distribution} distribution",
        alt_hypothesis=f"Sample does not follow {distribution} distribution",
    )


def quantum_distribution_test(
    counts: dict[str, int],
    expected_probs: dict[str, float],
    alpha: float = 0.05,
) -> HypothesisTest:
    """Test whether quantum measurement counts match expected probabilities.

    Takes a counts dict {bitstring: count} and expected probabilities
    {bitstring: probability} and runs a chi-squared test.
    """
    total_shots = sum(counts.values())
    all_keys = sorted(set(counts.keys()) | set(expected_probs.keys()))

    observed = np.array([counts.get(k, 0) for k in all_keys], dtype=float)
    expected = np.array([expected_probs.get(k, 0.0) * total_shots for k in all_keys], dtype=float)

    # Filter out zero-expected entries
    mask = expected > 0
    observed = observed[mask]
    expected = expected[mask]

    return chi_squared_test(observed, expected, alpha=alpha)
