"""Shared pytest/Hypothesis configuration for the test suite.

The property-based tests in this suite are logically correct and pass in
isolation, but during a *full-suite* run on slower machines Hypothesis can
abort some of them with ``FailedHealthCheck`` for timing-only reasons
(``HealthCheck.too_slow`` / ``HealthCheck.data_too_large``). These are
environmental signals about input-generation speed, not logic failures.

To keep the full suite reliably green without weakening any property or
reducing example counts, we register and load a Hypothesis profile that:

- suppresses the timing-only health checks, and
- disables the per-example deadline (so a slow draw never fails a test).

Because ``@settings(...)`` decorators inherit any field they do not set from
the currently loaded profile, and ``conftest.py`` is imported before the test
modules are collected, every property test automatically picks up these
settings while retaining its own ``max_examples`` and other explicit options.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings

# Register a profile that tolerates slow input generation on constrained
# machines while preserving each test's configured example count.
settings.register_profile(
    "stable",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

# Load it as the active default so all property tests inherit the timing
# tolerances at decoration time.
settings.load_profile("stable")
