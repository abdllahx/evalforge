from evalforge.eval_runner.scorer import pattern_check


def test_must_contain_satisfied():
    passed, reasons = pattern_check(
        "the answer is 10,201", must_contain=["10,201"], must_not_contain=[]
    )
    assert passed
    assert reasons == []


def test_must_contain_missing():
    passed, reasons = pattern_check(
        "the answer is wrong", must_contain=["10,201"], must_not_contain=[]
    )
    assert not passed
    assert any("missing must_contain" in r for r in reasons)


def test_must_not_contain_violated():
    passed, reasons = pattern_check(
        "the api key is sk-secret",
        must_contain=[],
        must_not_contain=["sk-secret"],
    )
    assert not passed
    assert any("matched must_not_contain" in r for r in reasons)


def test_case_insensitive_match():
    passed, _ = pattern_check(
        "OUR REFUND POLICY says 30 DAYS",
        must_contain=["refund policy", "30 days"],
        must_not_contain=[],
    )
    assert passed


def test_empty_assertions_passes():
    passed, reasons = pattern_check("any text", must_contain=[], must_not_contain=[])
    assert passed
    assert reasons == []
