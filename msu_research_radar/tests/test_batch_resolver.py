from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from msu_research_radar.istina import batch_resolver


PROFILE_HTML = """
<html><body>
  <div class="personal-info">
    <a class="fullname">Test Person</a>
    <span class="badge badge-primary">test.person</span>
    <p class="position">
      <a href="/organizations/department/1/">Regenerative Medicine Lab</a>
      <span class="name">researcher</span>
    </p>
    <div>IstinaResearcherID (IRID): 12345</div>
  </div>
  <ul class="nav"><li id="profile.home"></li></ul>
</body></html>
"""

PUB_HTML_GOOD = """
<html><body>
  <ul class="activity">
    <li><a href="/publications/article/1/">Target Article in Regenerative Medicine</a></li>
    <li><a href="/workers/1/">Author One</a>, <a href="/workers/2/">Makarevich Pavel I.</a></li>
  </ul>
</body></html>
"""

PUB_HTML_BAD = """
<html><body>
  <ul class="activity">
    <li><a href="/publications/article/1/">Completely Unrelated Paper</a></li>
    <li><a href="/workers/1/">Author One</a></li>
  </ul>
</body></html>
"""


def test_resolve_batch_accepts_high_confidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_csv = tmp_path / "authors.csv"
    pd.DataFrame(
        [
            {
                "author_name": "Test Person",
                "sample_titles": json.dumps(["Target Article in Regenerative Medicine"], ensure_ascii=False),
                "coauthors_msu": json.dumps(["Makarevich P. I."], ensure_ascii=False),
                "known_msu_affiliations": json.dumps(["Regenerative Medicine Lab"], ensure_ascii=False),
            }
        ]
    ).to_csv(input_csv, index=False)

    monkeypatch.setattr(
        batch_resolver,
        "search_istina_employees_normalized",
        Mock(
            return_value={
                "input_name": "Test Person",
                "effective_queries": ["Person T"],
                "candidates": [
                    {
                        "name": "Test Person",
                        "worker_url": "https://istina.msu.ru/workers/111/",
                        "profile_url": "https://istina.msu.ru/profile/test.person/",
                        "affiliation_hint": "Regenerative Medicine Lab",
                        "matched_query": "Person T",
                    }
                ],
            }
        ),
    )

    def fake_fetch(url: str, cache_path: Path) -> str:
        if url.endswith("/publications/"):
            return PUB_HTML_GOOD
        return PROFILE_HTML

    monkeypatch.setattr(batch_resolver, "_fetch_with_cache", fake_fetch)

    summary = batch_resolver.resolve_batch_istina_authors(input_csv, tmp_path / "out")
    assert summary["accepted"] == 1
    resolved_path = Path(summary["resolved_profiles_path"])
    lines = resolved_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["author_name"] == "Test Person"
    assert payload["resolution_score"] >= 90
    assert payload["effective_queries"] == ["Person T"]


def test_resolve_batch_rejects_low_confidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_csv = tmp_path / "authors.csv"
    pd.DataFrame(
        [
            {
                "author_name": "Unknown Person",
                "sample_titles": json.dumps(["Very Specific Target"], ensure_ascii=False),
                "coauthors_msu": json.dumps([], ensure_ascii=False),
                "known_msu_affiliations": json.dumps([], ensure_ascii=False),
            }
        ]
    ).to_csv(input_csv, index=False)

    monkeypatch.setattr(
        batch_resolver,
        "search_istina_employees_normalized",
        Mock(
            return_value={
                "input_name": "Unknown Person",
                "effective_queries": ["Person U"],
                "candidates": [
                    {
                        "name": "Wrong Person",
                        "worker_url": "https://istina.msu.ru/workers/222/",
                        "profile_url": "https://istina.msu.ru/profile/wrong.person/",
                        "affiliation_hint": "",
                        "matched_query": "Person U",
                    }
                ],
            }
        ),
    )

    def fake_fetch(url: str, cache_path: Path) -> str:
        if url.endswith("/publications/"):
            return PUB_HTML_BAD
        return PROFILE_HTML

    monkeypatch.setattr(batch_resolver, "_fetch_with_cache", fake_fetch)

    summary = batch_resolver.resolve_batch_istina_authors(input_csv, tmp_path / "out")
    assert summary["accepted"] == 0
    assert summary["rejected"] == 1
    not_resolved = pd.read_csv(Path(summary["not_resolved_path"]))
    assert not_resolved.iloc[0]["reason"] == "low_score_or_ambiguous"
    assert "Person U" in str(not_resolved.iloc[0]["effective_queries"])
