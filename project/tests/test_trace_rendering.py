"""Unit tests for human-readable trace rendering and the latency-budget flag (Task 8.4).

These complement ``test_proof_trace_export.py`` (which constructs dataclasses directly)
by driving the :class:`~nsr.proof_trace.ProofTraceBuilder` end-to-end and focusing on
three behaviours called out by the task:

- a step with no applied production rule renders the explicit ``no-rule-applied``
  indicator (Req 8.2);
- repair attempts render in execution order (Req 8.3);
- the latency-budget-exceeded indication is set and rendered when the cumulative
  System-2 latency exceeds the configured budget (Req 11.4).
"""

from __future__ import annotations

from nsr.models.enums import ValidationStatus
from nsr.models.reasoning import SymbolicRepresentation
from nsr.proof_trace import NO_RULE_APPLIED, ProofTraceBuilder, applied_rule_label
from nsr.proof_trace_export import render_step, render_trace


# --------------------------------------------------------------------------- #
# Req 8.2 -- explicit no-rule-applied indicator
# --------------------------------------------------------------------------- #


def test_step_without_applied_rule_renders_no_rule_indicator():
    """A step recorded with no applied rule renders the explicit indicator (Req 8.2)."""

    builder = ProofTraceBuilder()
    step = builder.append_step(
        "premise restated",
        status=ValidationStatus.ACCEPTED,
        applied_rule_id=None,
    )

    # The helper resolves the missing rule id to the explicit indicator...
    assert applied_rule_label(step) == NO_RULE_APPLIED
    # ...and that indicator appears in the rendered step header.
    rendered_step = render_step(step)
    assert f"[rule: {NO_RULE_APPLIED}]" in rendered_step
    # The whole-trace rendering surfaces it too.
    assert f"[rule: {NO_RULE_APPLIED}]" in render_trace(builder.trace)


def test_applied_rule_id_renders_instead_of_indicator():
    """When a rule is applied, its id renders and the indicator does not (Req 8.2)."""

    builder = ProofTraceBuilder()
    builder.append_step(
        "apply modus ponens",
        status=ValidationStatus.ACCEPTED,
        applied_rule_id="MP",
    )
    rendered = render_trace(builder.trace)
    assert "[rule: MP]" in rendered
    assert NO_RULE_APPLIED not in rendered


def test_mixed_steps_render_rule_and_indicator_independently():
    """Per-step rule presentation is independent across steps (Req 8.2)."""

    builder = ProofTraceBuilder()
    builder.append_step("with rule", status=ValidationStatus.ACCEPTED, applied_rule_id="R1")
    builder.append_step("without rule", status=ValidationStatus.ACCEPTED, applied_rule_id=None)

    rendered = render_trace(builder.trace)
    # The rule-bearing step keeps its id; the bare step shows the indicator.
    assert "[rule: R1]" in rendered
    assert f"[rule: {NO_RULE_APPLIED}]" in rendered
    # Indicator follows the rule-bearing step in execution order.
    assert rendered.index("[rule: R1]") < rendered.index(f"[rule: {NO_RULE_APPLIED}]")


# --------------------------------------------------------------------------- #
# Req 8.3 -- repair attempts render in execution order
# --------------------------------------------------------------------------- #


def _rep(logic_form: str) -> SymbolicRepresentation:
    return SymbolicRepresentation(logic_form=logic_form)


def test_repair_attempts_render_in_execution_order():
    """Multiple repair attempts render in the order they were recorded (Req 8.3)."""

    builder = ProofTraceBuilder()
    step = builder.append_step(
        "faulty inference",
        status=ValidationStatus.REPAIRED,
        applied_rule_id=None,
        violated_rule_ids=["R2"],
    )
    # Record three attempts in execution order, each with a distinct repaired form.
    builder.record_repair_attempt(
        step,
        rejected_step=_rep("bad-v0"),
        violated_rule_ids=["R2"],
        repaired_step=_rep("fix-v0"),
    )
    builder.record_repair_attempt(
        step,
        rejected_step=_rep("bad-v1"),
        violated_rule_ids=["R2", "R3"],
        repaired_step=_rep("fix-v1"),
    )
    builder.record_repair_attempt(
        step,
        rejected_step=_rep("bad-v2"),
        violated_rule_ids=["R3"],
        repaired_step=_rep("fix-v2"),
    )

    # Attempt indices are assigned in execution order by the builder.
    assert [a.attempt_index for a in step.repair_attempts] == [0, 1, 2]

    rendered = render_step(step)
    pos0 = rendered.index("repair 0")
    pos1 = rendered.index("repair 1")
    pos2 = rendered.index("repair 2")
    assert pos0 < pos1 < pos2

    # Each attempt's repaired form appears in its own line, in order.
    assert rendered.index("fix-v0") < rendered.index("fix-v1") < rendered.index("fix-v2")


def test_repair_attempt_renders_violated_rules_and_repaired_form():
    """A repair line names the violated rules and the resulting repaired form (Req 8.3)."""

    builder = ProofTraceBuilder()
    step = builder.append_step(
        "step needing repair",
        status=ValidationStatus.REPAIRED,
        applied_rule_id="R5",
    )
    builder.record_repair_attempt(
        step,
        rejected_step=_rep("rejected-form"),
        violated_rule_ids=["R5", "R6"],
        repaired_step=_rep("repaired-form"),
    )
    rendered = render_step(step)
    assert "repair 0:" in rendered
    assert "R5, R6" in rendered
    assert "repaired-form" in rendered


def test_unrepaired_attempt_renders_placeholder():
    """An attempt with no repaired step renders an explicit unrepaired marker (Req 8.3)."""

    builder = ProofTraceBuilder()
    step = builder.append_step(
        "exhausted step",
        status=ValidationStatus.REJECTED,
        applied_rule_id=None,
    )
    builder.record_repair_attempt(
        step,
        rejected_step=_rep("still-bad"),
        violated_rule_ids=[],  # no rules -> indicator used for the violated list
        repaired_step=None,
    )
    rendered = render_step(step)
    assert "(unrepaired)" in rendered
    # With no violated rules, the repair line falls back to the no-rule indicator.
    assert NO_RULE_APPLIED in rendered


# --------------------------------------------------------------------------- #
# Req 11.4 -- latency-budget-exceeded indication
# --------------------------------------------------------------------------- #


def test_latency_budget_exceeded_flag_set_when_system2_over_budget():
    """System-2 latency above the budget sets the exceeded flag (Req 11.4)."""

    builder = ProofTraceBuilder(latency_budget_ms=100)
    builder.add_system2_latency(60.0)
    builder.add_system2_latency(50.0)  # cumulative 110 > 100
    record = builder.record_latency(pipeline_ms=130.0)

    assert record.system2_ms == 110.0
    assert record.latency_budget_exceeded is True
    assert builder.trace.latency is record


def test_latency_budget_not_exceeded_when_within_budget():
    """System-2 latency at or below the budget leaves the flag clear (Req 11.4)."""

    builder = ProofTraceBuilder(latency_budget_ms=100)
    builder.add_system2_latency(100.0)  # equal to budget, not exceeding
    record = builder.record_latency(pipeline_ms=140.0)

    assert record.latency_budget_exceeded is False


def test_no_budget_configured_never_flags_exceeded():
    """Without a configured budget, the indication is never set (Req 11.4)."""

    builder = ProofTraceBuilder()  # no budget
    builder.add_system2_latency(10_000.0)
    record = builder.record_latency(pipeline_ms=10_500.0)

    assert record.latency_budget_exceeded is False


def test_latency_budget_exceeded_is_rendered():
    """The exceeded indication appears in the human-readable rendering (Req 11.4)."""

    builder = ProofTraceBuilder(latency_budget_ms=50)
    builder.append_step("a step", status=ValidationStatus.ACCEPTED, applied_rule_id="R1")
    builder.add_system2_latency(75.0)
    builder.record_latency(pipeline_ms=90.0)

    rendered = render_trace(builder.trace)
    assert "budget exceeded" in rendered


def test_latency_within_budget_not_rendered_as_exceeded():
    """A within-budget query does not render the exceeded indication (Req 11.4)."""

    builder = ProofTraceBuilder(latency_budget_ms=200)
    builder.add_system2_latency(50.0)
    builder.record_latency(pipeline_ms=120.0)

    rendered = render_trace(builder.trace)
    assert "budget exceeded" not in rendered
