"""Unit tests for the Validation Engine (Task 6.1).

These verify the accept/reject semantics from Requirements 6.1, 6.2, 6.3 and 6.7:

- evaluation against every applicable production rule with a recorded outcome,
- acceptance only when all applicable rules are satisfied,
- rejection recording every violated rule, and
- the per-rule evaluation record suitable for journaling.
"""

from __future__ import annotations

from nsr import RuleEvaluation, ValidationEngine, ValidationOutcome
from nsr.models import ProductionRule, SymbolicRepresentation, ValidationStatus


def _rep(logic_form: str = "", source_text: str = "", predicates=None):
    return SymbolicRepresentation(
        logic_form=logic_form,
        source_text=source_text,
        predicates=predicates or {},
    )


def test_no_rules_accepts_vacuously():
    """With no rules, acceptance holds vacuously (Req 6.2)."""
    engine = ValidationEngine()
    outcome = engine.validate(_rep(logic_form="anything"), [])

    assert isinstance(outcome, ValidationOutcome)
    assert outcome.status is ValidationStatus.ACCEPTED
    assert outcome.accepted is True
    assert outcome.rejected is False
    assert outcome.applicable_rule_ids == []
    assert outcome.violated_rule_ids == []
    assert outcome.evaluations == []


def test_all_applicable_rules_satisfied_accepts():
    """A step satisfying every applicable rule is accepted (Req 6.2)."""
    engine = ValidationEngine()
    rules = [
        ProductionRule(rule_id="r1", condition="IF sum", action="THEN total"),
        ProductionRule(rule_id="r2", condition="IF total", action="THEN positive"),
    ]
    rep = _rep(logic_form="sum total positive", source_text="the total is positive")

    outcome = engine.validate(rep, rules)

    assert outcome.status is ValidationStatus.ACCEPTED
    assert outcome.applicable_rule_ids == ["r1", "r2"]
    assert outcome.violated_rule_ids == []
    assert outcome.violated_rules == []


def test_inapplicable_rules_are_neither_satisfied_nor_violated():
    """Rules whose condition does not match are not applicable (Req 6.1)."""
    engine = ValidationEngine()
    rules = [
        ProductionRule(rule_id="r1", condition="IF division", action="THEN nonzero"),
    ]
    rep = _rep(logic_form="addition step", source_text="2 + 2 = 4")

    outcome = engine.validate(rep, rules)

    assert outcome.status is ValidationStatus.ACCEPTED
    assert outcome.applicable_rule_ids == []
    assert outcome.violated_rule_ids == []
    assert outcome.evaluations == [
        RuleEvaluation(rule_id="r1", applicable=False, satisfied=True)
    ]


def test_partial_violation_rejects_and_records_every_violated_rule():
    """Rejection records every violated applicable rule (Req 6.3)."""
    engine = ValidationEngine()
    rules = [
        # applicable + satisfied
        ProductionRule(rule_id="r1", condition="IF sum", action="THEN total"),
        # applicable + violated (action term absent)
        ProductionRule(rule_id="r2", condition="IF sum", action="THEN normalized"),
        # applicable + violated (action term absent)
        ProductionRule(rule_id="r3", condition="IF total", action="THEN rounded"),
        # not applicable
        ProductionRule(rule_id="r4", condition="IF derivative", action="THEN slope"),
    ]
    rep = _rep(logic_form="sum total", source_text="computing the sum total")

    outcome = engine.validate(rep, rules)

    assert outcome.status is ValidationStatus.REJECTED
    assert outcome.rejected is True
    assert outcome.applicable_rule_ids == ["r1", "r2", "r3"]
    assert outcome.violated_rule_ids == ["r2", "r3"]
    # Repair Coordinator consumes the violated rule objects directly (Req 6.4 prep).
    assert [r.rule_id for r in outcome.violated_rules] == ["r2", "r3"]


def test_every_rule_has_an_evaluation_record():
    """Every supplied rule produces an evaluation record for journaling (Req 6.7)."""
    engine = ValidationEngine()
    rules = [
        ProductionRule(rule_id="r1", condition="IF sum", action="THEN total"),
        ProductionRule(rule_id="r2", condition="IF sum", action="THEN missing"),
        ProductionRule(rule_id="r3", condition="IF absent", action="THEN whatever"),
    ]
    rep = _rep(logic_form="sum total")

    outcome = engine.validate(rep, rules)

    by_id = {e.rule_id: e for e in outcome.evaluations}
    assert set(by_id) == {"r1", "r2", "r3"}
    assert by_id["r1"] == RuleEvaluation("r1", applicable=True, satisfied=True)
    assert by_id["r2"] == RuleEvaluation("r2", applicable=True, satisfied=False)
    assert by_id["r3"] == RuleEvaluation("r3", applicable=False, satisfied=True)


def test_empty_condition_rule_is_always_applicable():
    """A rule with an empty condition matches unconditionally."""
    engine = ValidationEngine()
    rules = [ProductionRule(rule_id="default", condition="", action="THEN grounded")]

    # action term present -> satisfied
    assert engine.validate(_rep(logic_form="grounded"), rules).accepted is True
    # action term absent -> violated
    rejected = engine.validate(_rep(logic_form="floating"), rules)
    assert rejected.status is ValidationStatus.REJECTED
    assert rejected.violated_rule_ids == ["default"]


def test_empty_action_rule_is_satisfied_when_applicable():
    """An applicable rule with an empty action is satisfied vacuously."""
    engine = ValidationEngine()
    rules = [ProductionRule(rule_id="r1", condition="IF premise", action="")]
    outcome = engine.validate(_rep(logic_form="premise holds"), rules)

    assert outcome.status is ValidationStatus.ACCEPTED
    assert outcome.applicable_rule_ids == ["r1"]
    assert outcome.violated_rule_ids == []


def test_conjunctive_condition_requires_all_terms():
    """An ``AND`` condition is applicable only when all its terms are present."""
    engine = ValidationEngine()
    rules = [
        ProductionRule(rule_id="r1", condition="IF a AND b", action="THEN c"),
    ]
    # only one of the two condition terms present -> not applicable
    not_applicable = engine.validate(_rep(logic_form="only a here"), rules)
    assert not_applicable.applicable_rule_ids == []
    assert not_applicable.accepted is True

    # both condition terms present, action absent -> applicable + violated
    applicable = engine.validate(_rep(logic_form="a and b together"), rules)
    assert applicable.status is ValidationStatus.REJECTED
    assert applicable.violated_rule_ids == ["r1"]


def test_predicates_contribute_to_matching():
    """Structured predicate fields are part of the searchable representation."""
    engine = ValidationEngine()
    rules = [ProductionRule(rule_id="r1", condition="IF operator", action="THEN plus")]
    rep = _rep(predicates={"operator": "plus", "operands": [1, 2]})

    outcome = engine.validate(rep, rules)
    assert outcome.status is ValidationStatus.ACCEPTED
    assert outcome.applicable_rule_ids == ["r1"]


def test_validate_rejects_none_representation():
    engine = ValidationEngine()
    try:
        engine.validate(None, [])  # type: ignore[arg-type]
    except ValueError as exc:
        assert "non-None" in str(exc)
    else:
        raise AssertionError("expected ValueError for None representation")


def test_validation_is_deterministic():
    """Repeated validation of the same inputs yields the same outcome."""
    engine = ValidationEngine()
    rules = [
        ProductionRule(rule_id="r1", condition="IF sum", action="THEN total"),
        ProductionRule(rule_id="r2", condition="IF sum", action="THEN missing"),
    ]
    rep = _rep(logic_form="sum total")

    first = engine.validate(rep, rules)
    second = engine.validate(rep, rules)
    assert first == second
