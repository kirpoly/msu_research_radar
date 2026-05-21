from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from msu_research_radar.istina import resolver


def _publications_html(title: str, authors: list[str]) -> str:
    authors_html = ", ".join(
        f'<a href="/workers/{index + 1}/">{author}</a>' for index, author in enumerate(authors)
    )
    return f"""
    <html>
      <body>
        <ul class="activity">
          <li><a href="/publications/article/1/">{title}</a></li>
          <li>{authors_html}</li>
        </ul>
      </body>
    </html>
    """


PROFILE_HTML = """
<html>
  <body>
    <div class="personal-info">
      <a class="fullname">Alexandrushkina Natalia Andreevna</a>
      <span class="badge badge-primary">N.Alexandrushkina</span>
      <p class="position">
        <a href="/organizations/department/58239330/">Regenerative Medicine Lab</a>
        <a href="/organizations/department/708037354/">Medical Research and Education Institute</a>
        <span class="name">researcher</span>
      </p>
      <div>IstinaResearcherID (IRID): 50264918</div>
    </div>
    <ul class="nav">
      <li id="profile.home"></li>
    </ul>
  </body>
</html>
"""


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.status_code = 200
        self.url = "https://istina.msu.ru/fake/"

    def raise_for_status(self) -> None:
        return None


def test_resolve_person_auto_selects_single_exact_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resolver, "PROFILE_CACHE_DIR", tmp_path / "profiles")
    monkeypatch.setattr(resolver, "INTERIM_PROFILE_DIR", tmp_path / "interim")
    monkeypatch.setattr(
        resolver,
        "search_istina_employees_normalized",
        Mock(
            return_value={
                "input_name": "Natalia A Alexandrushkina",
                "effective_queries": ["Alexandrushkina N", "Alexandrushkina N.A."],
                "candidates": [
                    {
                        "name": "Candidate 1",
                        "profile_url": "https://istina.msu.ru/profile/candidate1/",
                        "worker_url": "https://istina.msu.ru/workers/111/",
                        "snippet": "snippet 1",
                        "affiliation_hint": "aff1",
                    },
                    {
                        "name": "Candidate 2",
                        "profile_url": "https://istina.msu.ru/profile/candidate2/",
                        "worker_url": "https://istina.msu.ru/workers/222/",
                        "snippet": "snippet 2",
                        "affiliation_hint": "aff2",
                    },
                ],
            }
        ),
    )

    responses = {
        "https://istina.msu.ru/workers/111/publications/": _publications_html(
            "Target Article",
            ["Author A", "Author B"],
        ),
        "https://istina.msu.ru/workers/222/publications/": _publications_html(
            "Different Article",
            ["Author C", "Author D"],
        ),
        "https://istina.msu.ru/profile/candidate1/": PROFILE_HTML,
    }

    def fake_get(url: str, timeout: int) -> _Response:
        assert url in responses
        return _Response(responses[url])

    monkeypatch.setattr(resolver.requests, "get", fake_get)

    result = resolver.resolve_istina_person(
        query="Natalia A Alexandrushkina",
        article_title="Target Article",
        coauthors=["Author A"],
    )

    assert result["selection_mode"] == "single exact title match"
    assert result["effective_queries"] == ["Alexandrushkina N", "Alexandrushkina N.A."]
    assert result["chosen_candidate"]["name"] == "Candidate 1"
    assert result["chosen_match"]["title_exact_norm"] is True
    output_path = Path(result["output_json_path"])
    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["chosen_candidate"]["name"] == "Candidate 1"


def test_resolve_person_interactive_fallback_with_coauthor_bonus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resolver, "PROFILE_CACHE_DIR", tmp_path / "profiles")
    monkeypatch.setattr(resolver, "INTERIM_PROFILE_DIR", tmp_path / "interim")
    monkeypatch.setattr(
        resolver,
        "search_istina_employees_normalized",
        Mock(
            return_value={
                "input_name": "Alexandrushkina N",
                "effective_queries": ["Alexandrushkina N"],
                "candidates": [
                    {
                        "name": "Candidate 1",
                        "profile_url": "https://istina.msu.ru/profile/candidate1/",
                        "worker_url": "https://istina.msu.ru/workers/333/",
                        "snippet": "snippet 1",
                        "affiliation_hint": "aff1",
                    },
                    {
                        "name": "Candidate 2",
                        "profile_url": "https://istina.msu.ru/profile/candidate2/",
                        "worker_url": "https://istina.msu.ru/workers/444/",
                        "snippet": "snippet 2",
                        "affiliation_hint": "aff2",
                    },
                ],
            }
        ),
    )
    monkeypatch.setattr("builtins.input", Mock(return_value="1"))

    responses = {
        "https://istina.msu.ru/workers/333/publications/": _publications_html(
            "Mesenchymal Cells in Regenerative Medicine",
            ["Unrelated Author"],
        ),
        "https://istina.msu.ru/workers/444/publications/": _publications_html(
            "Mesenchymal Cell Regeneration and Medicine",
            ["Makarevich Pavel I.", "Another Author"],
        ),
        "https://istina.msu.ru/profile/candidate2/": PROFILE_HTML,
    }

    def fake_get(url: str, timeout: int) -> _Response:
        assert url in responses
        return _Response(responses[url])

    monkeypatch.setattr(resolver.requests, "get", fake_get)

    result = resolver.resolve_istina_person(
        query="Alexandrushkina N",
        article_title="Mesenchymal Cell Medicine",
        coauthors=["Makarevich P. I."],
    )

    assert result["selection_mode"] == "interactive selection"
    assert result["chosen_candidate"]["name"] == "Candidate 2"
    assert result["chosen_match"]["title_exact_norm"] is False
    assert result["chosen_match"]["coauthor_overlap"]
