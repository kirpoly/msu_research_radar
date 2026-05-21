from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from msu_research_radar.export.master_builder import build_master_author_graph


def test_build_master_author_graph_uses_only_resolved_authors(tmp_path: Path) -> None:
    authorships = [
        {
            "author_position": "first",
            "is_corresponding": True,
            "author": {"display_name": "Resolved Author"},
            "institutions": [{"display_name": "MSU", "ror": "https://ror.org/010pmpe69"}],
        },
        {
            "author_position": "last",
            "is_corresponding": False,
            "author": {"display_name": "Other MSU Author"},
            "institutions": [{"display_name": "MSU", "ror": "https://ror.org/010pmpe69"}],
        },
    ]
    works_df = pd.DataFrame(
        [
            {
                "openalex_id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.1/test",
                "title": "Paper Title",
                "publication_date": "2026-05-09",
                "publication_year": 2026,
                "journal": "Journal A",
                "cited_by_count": 3,
                "topic": "Mesenchymal stem cell research",
                "subfield": "Genetics",
                "field": "Medicine",
                "domain": "Health Sciences",
                "authorships": json.dumps(authorships, ensure_ascii=False),
            }
        ]
    )
    works_path = tmp_path / "works.parquet"
    works_df.to_parquet(works_path, index=False, engine="pyarrow")

    authors_path = tmp_path / "authors.csv"
    pd.DataFrame([{"author_name": "Resolved Author"}]).to_csv(authors_path, index=False)

    resolved_path = tmp_path / "resolved.jsonl"
    resolved_payload = {
        "author_name": "Resolved Author",
        "profile_url": "https://istina.msu.ru/profile/resolved/",
        "matched_query": "Resolved Author",
        "resolution_score": 97.0,
        "matched_title": "Paper Title",
        "matched_coauthors": ["Other MSU Author"],
        "parsed_profile": {
            "irid": "12345",
            "orcid": "0000-0001-0000-0001",
            "spin": "1111-2222",
            "researcher_id": "X-0001-2026",
            "lab": "Лаборатория X",
            "department": None,
            "institute": "Институт Y",
            "position": "научный сотрудник",
        },
    }
    resolved_path.write_text(json.dumps(resolved_payload, ensure_ascii=False) + "\n", encoding="utf-8")

    output_path = tmp_path / "master.parquet"
    summary = build_master_author_graph(works_path, authors_path, resolved_path, output_path)

    assert summary["rows"] == 1
    out_df = pd.read_parquet(output_path)
    assert len(out_df) == 1
    assert out_df.iloc[0]["author_name"] == "Resolved Author"
    assert out_df.iloc[0]["irid"] == "12345"
    assert bool(out_df.iloc[0]["is_corresponding"]) is True
    assert out_df.iloc[0]["topic"] == "Mesenchymal stem cell research"
    assert out_df.iloc[0]["subfield"] == "Genetics"
    assert out_df.iloc[0]["field"] == "Medicine"
    assert out_df.iloc[0]["domain"] == "Health Sciences"
    assert "Other MSU Author" in out_df.iloc[0]["msu_coauthors"]
