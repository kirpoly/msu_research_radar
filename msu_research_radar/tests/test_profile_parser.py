from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from msu_research_radar.istina.profile_parser import (
    parse_istina_profile_html,
    parse_istina_profile_url,
)


def _profile_fixture_path() -> Path:
    fixtures = sorted(Path("data/raw/istina").glob("*.html"))
    assert fixtures, "Expected at least one Istina profile HTML fixture in data/raw/istina"
    return fixtures[0]


def _profile_fixture_html() -> str:
    return _profile_fixture_path().read_text(encoding="utf-8")


def test_parse_istina_profile_html_extracts_expected_fields() -> None:
    html = _profile_fixture_html()
    url = "https://istina.msu.ru/profile/N.Alexandrushkina/"

    parsed = parse_istina_profile_html(html, url=url)

    assert parsed["full_name"] == "Александрушкина Наталья Андреевна"
    assert parsed["profile_url"] == url
    assert parsed["username"] == "N.Alexandrushkina"
    assert parsed["irid"] == "50264918"
    assert parsed["researcher_id"] == "U-1044-2017"
    assert parsed["orcid"] == "0000-0003-4946-7843"
    assert parsed["spin"] == "3611-7954"
    assert parsed["scopus_author_id"] is None
    assert parsed["current_affiliations"] == [
        "Лаборатория медицинской биоинженерии",
        "Медицинский научно-образовательный институт",
    ]
    assert parsed["lab"] == "Лаборатория медицинской биоинженерии"
    assert parsed["department"] is None
    assert parsed["institute"] == "Медицинский научно-образовательный институт"
    assert parsed["position"] == "научный сотрудник"
    assert "profile.home" in parsed["visible_blocks"]
    assert "profile.all" in parsed["visible_blocks"]
    assert len(parsed["raw_text_snippets"]) >= 4


def test_parse_istina_profile_html_visible_blocks() -> None:
    parsed = parse_istina_profile_html(_profile_fixture_html())

    assert parsed["visible_blocks"] == [
        "profile.home",
        "profile.topics",
        "profile.publications",
        "profile.projects",
        "profile.talks",
        "profile.teaching",
        "profile.innovation",
        "profile.misc",
        "profile.all",
    ]


def test_parse_istina_profile_html_missing_optional_fields() -> None:
    html = """
    <html>
      <body>
        <ul class="nav">
          <li id="profile.home"></li>
        </ul>
        <div class="personal-info">
          <a class="fullname">Test User</a>
          <span class="badge badge-primary">test.user</span>
        </div>
      </body>
    </html>
    """

    parsed = parse_istina_profile_html(html, url="https://istina.msu.ru/profile/test.user/")

    assert parsed["full_name"] == "Test User"
    assert parsed["username"] == "test.user"
    assert parsed["irid"] is None
    assert parsed["researcher_id"] is None
    assert parsed["orcid"] is None
    assert parsed["spin"] is None
    assert parsed["scopus_author_id"] is None
    assert parsed["current_affiliations"] == []
    assert parsed["lab"] is None
    assert parsed["department"] is None
    assert parsed["institute"] is None
    assert parsed["position"] is None


def test_parse_istina_profile_url_uses_requests_get(monkeypatch: pytest.MonkeyPatch) -> None:
    html = _profile_fixture_html()
    response = Mock()
    response.text = html
    response.raise_for_status = Mock()

    mock_get = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", mock_get)

    url = "https://istina.msu.ru/profile/N.Alexandrushkina/"
    parsed = parse_istina_profile_url(url, timeout=12.5)

    mock_get.assert_called_once_with(url, timeout=12.5)
    response.raise_for_status.assert_called_once_with()
    assert parsed["profile_url"] == url
    assert parsed["irid"] == "50264918"


def test_parse_istina_profile_url_propagates_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("bad status")
    response.text = ""
    monkeypatch.setattr(requests, "get", Mock(return_value=response))

    with pytest.raises(requests.HTTPError):
        parse_istina_profile_url("https://istina.msu.ru/profile/N.Alexandrushkina/")

