from __future__ import annotations

from msu_research_radar.radar.tagging import apply_profile_tags, match_text_rule


def test_match_text_rule_include_exclude_and_regex() -> None:
    matched, hits = match_text_rule(
        "Department of Systems Biology, Faculty of Biology",
        {
            "include_terms": ["systems biology"],
            "include_regex": ["Faculty\\s+of\\s+Biology"],
            "exclude_terms": ["chemistry"],
        },
    )

    assert matched is True
    assert "term:systems biology" in hits
    assert "regex:Faculty\\s+of\\s+Biology" in hits

    blocked, block_hits = match_text_rule(
        "Department of Systems Biology and Chemistry",
        {"include_terms": ["systems biology"], "exclude_terms": ["chemistry"]},
    )
    assert blocked is False
    assert block_hits == ["exclude_term:chemistry"]


def test_apply_profile_tags_returns_generic_tags_and_reasons() -> None:
    profile = {
        "affiliation_tags": {
            "target_unit": {"include_terms": ["Faculty of Biology"], "weight": 2.0},
        },
        "topic_tags": {
            "target_topic": {"include_terms": ["genomics"], "weight": 3.0},
        },
        "role_tags": {
            "senior": {
                "author_positions": ["last"],
                "corresponding_author": True,
                "position_terms": ["professor"],
                "weight": 1.0,
            },
        },
    }
    row = {
        "raw_affiliation_strings": ["Faculty of Biology, Lomonosov Moscow State University"],
        "title": "Comparative genomics of model organisms",
        "author_position": "last",
        "is_corresponding": True,
        "position": "Professor",
    }

    tagged = apply_profile_tags(row, profile)

    assert tagged["matched_affiliation_tags"] == ["target_unit"]
    assert tagged["matched_topic_tags"] == ["target_topic"]
    assert tagged["matched_role_tags"] == ["senior"]
    assert tagged["profile_match_score"] == 6.0
    assert any(reason.startswith("affiliation_tags.target_unit") for reason in tagged["match_reasons"])
