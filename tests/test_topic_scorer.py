from youtube import topic_scorer


def test_heuristic_scorer_is_deterministic_and_in_range():
    a = topic_scorer.score_topic_heuristic("The Mysterious Place Nobody Is Allowed to Visit")
    b = topic_scorer.score_topic_heuristic("The Mysterious Place Nobody Is Allowed to Visit")
    assert a == b
    for key in topic_scorer.CRITERIA:
        assert 0 <= a["scores"][key] <= 10
    assert 0 <= a["overall"] <= 10


def test_curiosity_words_raise_curiosity_score():
    low = topic_scorer.score_topic_heuristic("A regular Tuesday afternoon in a quiet town")
    high = topic_scorer.score_topic_heuristic("The unexplained mystery of a vanished, forbidden ancient city")
    assert high["scores"]["curiosity"] > low["scores"]["curiosity"]


def test_score_topic_defaults_to_heuristic_without_llm():
    result = topic_scorer.score_topic("Some topic", use_llm=False)
    assert result["backend"] == "heuristic"


def test_scorer_has_no_political_dimension():
    # Spec section 12: must never gain a political/ideological scoring axis.
    forbidden_substrings = ("politic", "ideolog", "partisan", "party")
    for key in topic_scorer.CRITERIA:
        assert not any(f in key.lower() for f in forbidden_substrings)
