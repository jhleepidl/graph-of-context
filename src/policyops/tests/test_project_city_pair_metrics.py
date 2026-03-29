from __future__ import annotations

from src.metrics import normalize_project_city_pair, robust_match


def test_normalize_project_city_pair_parses_pipe_and_prose() -> None:
    assert normalize_project_city_pair("Project_0052 | City_28") == "Project_0052 | City_28"
    prose = (
        "For handle Juniper-052-61, the project is Project_0052 with start_year 2014. "
        "The current operating city is City_28 as per the approval notice."
    )
    assert normalize_project_city_pair(prose) == "Project_0052 | City_28"


def test_robust_match_accepts_salvageable_prose_answer() -> None:
    pred = (
        "Therefore, the answer is based on Project_0052 and its current operating city City_28."
    )
    gold = "Project_0052 | City_28"
    assert robust_match(pred, gold) is True
