from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from msu_research_radar.radar.shortlist import export_shortlist


def _profile_yaml(path: Path) -> None:
    path.write_text(
        """
profiles:
  test_profile:
    affiliation_tags:
      target_unit:
        include_terms: ["Faculty of Biology"]
        weight: 2.0
    topic_tags:
      target_topic:
        include_terms: ["genomics"]
        weight: 2.0
    role_tags:
      senior:
        author_positions: ["last"]
        corresponding_author: true
        weight: 1.0
      young:
        author_positions: ["first"]
        weight: 1.0
    scoring:
      recent_years: [2026]
      supervisor_weights:
        relevant_pub_count: 1.0
        last_author_count: 2.0
        corresponding_author_count: 1.0
        senior_role_tag_count: 3.0
      young_contact_weights:
        relevant_pub_count: 1.0
        first_author_count: 2.0
        young_role_tag_count: 3.0
""",
        encoding="utf-8",
    )


def _works(path: Path) -> None:
    works = pd.DataFrame(
        [
            {
                "openalex_id": "W1",
                "doi": "10.1/a",
                "title": "Genomics paper one",
                "publication_year": 2026,
                "journal": "Journal A",
                "authorships": json.dumps(
                    [
                        {
                            "author_position": "last",
                            "is_corresponding": True,
                            "author": {"id": "A1", "display_name": "Natalia Alexandrushkina"},
                            "raw_affiliation_strings": [
                                "Faculty of Biology, Lomonosov Moscow State University"
                            ],
                            "institutions": [{"ror": "https://ror.org/010pmpe69"}],
                        },
                        {
                            "author_position": "first",
                            "author": {"id": "A2", "display_name": "Young Author"},
                            "raw_affiliation_strings": [
                                "Faculty of Biology, Lomonosov Moscow State University"
                            ],
                            "institutions": [{"ror": "https://ror.org/010pmpe69"}],
                        },
                    ],
                    ensure_ascii=False,
                ),
            },
            {
                "openalex_id": "W2",
                "doi": "10.1/b",
                "title": "Genomics paper two",
                "publication_year": 2026,
                "journal": "Journal B",
                "authorships": json.dumps(
                    [
                        {
                            "author_position": "first",
                            "is_corresponding": False,
                            "author": {"id": "A1", "display_name": "Natalia A. Alexandrushkina"},
                            "raw_affiliation_strings": [
                                "Faculty of Biology, Lomonosov Moscow State University"
                            ],
                            "institutions": [{"ror": "https://ror.org/010pmpe69"}],
                        }
                    ],
                    ensure_ascii=False,
                ),
            },
        ]
    )
    works.to_parquet(path, index=False, engine="pyarrow")


def _resolved(path: Path) -> None:
    payload = {
        "author_name": "Natalia Alexandrushkina",
        "profile_url": "https://istina.msu.ru/profile/N.Alexandrushkina/",
        "resolution_score": 100,
        "parsed_profile": {
            "irid": "50264918",
            "lab": "Lab A",
            "department": "Department A",
            "institute": "Institute A",
            "position": "professor",
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_export_shortlist_has_required_sheets_and_merges_name_variants(tmp_path: Path) -> None:
    works_path = tmp_path / "works.parquet"
    profile_path = tmp_path / "profiles.yaml"
    resolved_path = tmp_path / "resolved.jsonl"
    output_path = tmp_path / "shortlist.xlsx"
    _works(works_path)
    _profile_yaml(profile_path)
    _resolved(resolved_path)

    summary = export_shortlist(
        works_path=works_path,
        output_path=output_path,
        profile_config=profile_path,
        profile_name="test_profile",
        resolved_path=resolved_path,
    )

    assert summary["papers_relevant"] == 3
    book = pd.ExcelFile(output_path)
    assert set(book.sheet_names) == {
        "supervisors",
        "young_contacts",
        "labs_or_units",
        "papers",
        "coauthor_edges",
        "unresolved_istina",
        "config_used",
    }
    supervisors = pd.read_excel(output_path, sheet_name="supervisors")
    young_contacts = pd.read_excel(output_path, sheet_name="young_contacts")
    assert supervisors["author_identity"].nunique() == 2
    assert young_contacts["young_contact_score"].max() != supervisors["supervisor_score"].max()
    merged = supervisors[supervisors["istina_profile_url"] == "https://istina.msu.ru/profile/N.Alexandrushkina/"]
    assert len(merged) == 1
    assert merged.iloc[0]["relevant_pub_count"] == 2
