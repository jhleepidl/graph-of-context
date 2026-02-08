from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.unfold_trigger import UnfoldTrigger


def test_extract_keywords_includes_docid_title_url_and_alnum() -> None:
    trig = UnfoldTrigger(missing_terms_threshold=3, min_token_len=4)
    query = "Need D_TRUTH_0001 and TITLE:Project Atlas; URL:https://example.org/policy, plus relocation_note."
    kws = trig.extract_keywords(query)

    assert "D_TRUTH_0001" in kws
    assert "TITLE:Project Atlas" in kws
    assert "URL:https://example.org/policy" in kws
    assert "relocation_note" in [k.lower() for k in kws]


def test_should_unfold_on_missing_terms_threshold() -> None:
    trig = UnfoldTrigger(
        missing_terms_threshold=2,
        min_token_len=4,
        always_trigger_on_required_keys=False,
    )
    should, reason, missing = trig.should_unfold(
        next_query="alpha beta gamma delta",
        active_text="alpha and BETA only",
    )

    assert should is True
    assert reason == "missing_terms"
    low = [m.lower() for m in missing]
    assert "gamma" in low
    assert "delta" in low


def test_should_unfold_on_required_relocation_key_even_if_present() -> None:
    trig = UnfoldTrigger(
        missing_terms_threshold=99,
        always_trigger_on_required_keys=True,
    )
    should, reason, missing = trig.should_unfold(
        next_query="Use relocation_note and relocation_city for the final answer.",
        active_text="relocation_note relocation_city were already captured.",
    )

    assert should is True
    assert reason == "required_key"
    assert isinstance(missing, list)


def test_should_not_unfold_when_covered_and_below_threshold() -> None:
    trig = UnfoldTrigger(
        missing_terms_threshold=3,
        always_trigger_on_required_keys=False,
    )
    should, reason, missing = trig.should_unfold(
        next_query="alpha beta gamma",
        active_text="ALPHA beta gamma",
    )

    assert should is False
    assert reason == "covered"
    assert missing == []
