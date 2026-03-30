"""Tests for hypothesis testing."""

import numpy as np
import pytest
from crowe_synapse.hypothesis import (
    HypothesisTest,
    chi_squared_test,
    quantum_distribution_test,
    t_test,
)


class TestChiSquared:
    def test_uniform_distribution(self):
        # Perfect uniform distribution should not reject
        observed = [25, 25, 25, 25]
        expected = [25, 25, 25, 25]
        result = chi_squared_test(observed, expected)
        assert not result.reject_null
        assert result.p_value == pytest.approx(1.0)

    def test_skewed_distribution(self):
        # Very skewed should reject
        observed = [100, 0, 0, 0]
        expected = [25, 25, 25, 25]
        result = chi_squared_test(observed, expected)
        assert result.reject_null

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            chi_squared_test([1, 2], [1, 2, 3])


class TestTTest:
    def test_one_sample(self):
        rng = np.random.default_rng(42)
        sample = rng.normal(loc=5.0, scale=1.0, size=100)
        result = t_test(sample, mu=5.0)
        assert not result.reject_null  # should not reject if mu is correct

    def test_one_sample_reject(self):
        rng = np.random.default_rng(42)
        sample = rng.normal(loc=10.0, scale=1.0, size=100)
        result = t_test(sample, mu=0.0)
        assert result.reject_null  # mean is far from 0

    def test_two_sample_same(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(loc=5.0, scale=1.0, size=100)
        s2 = rng.normal(loc=5.0, scale=1.0, size=100)
        result = t_test(s1, s2)
        assert not result.reject_null

    def test_two_sample_different(self):
        rng = np.random.default_rng(42)
        s1 = rng.normal(loc=0.0, scale=1.0, size=100)
        s2 = rng.normal(loc=10.0, scale=1.0, size=100)
        result = t_test(s1, s2)
        assert result.reject_null


class TestQuantumDistribution:
    def test_fair_coin(self):
        counts = {"0": 500, "1": 500}
        expected = {"0": 0.5, "1": 0.5}
        result = quantum_distribution_test(counts, expected)
        assert not result.reject_null

    def test_bell_state(self):
        # Bell state: |00⟩ and |11⟩ with equal probability
        counts = {"00": 502, "11": 498}
        expected = {"00": 0.5, "11": 0.5}
        result = quantum_distribution_test(counts, expected)
        assert result.test_name == "Chi-squared"
        assert not result.reject_null


class TestHypothesisResult:
    def test_conclusion_reject(self):
        h = HypothesisTest("test", 10.0, 0.001)
        assert h.reject_null
        assert "Reject" in h.conclusion

    def test_conclusion_fail(self):
        h = HypothesisTest("test", 0.5, 0.5)
        assert not h.reject_null
        assert "Fail" in h.conclusion
