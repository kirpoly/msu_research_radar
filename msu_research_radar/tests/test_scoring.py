from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from msu_research_radar.scoring.scorer import export_author_scores, export_lab_scores, score_authors, score_labs


def _weights_yaml(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "recent_years: [2025, 2026]",
                "sample_papers_limit: 3",
                "lab_weights:",
                "  pub_count: 1.0",
                "  recent_pub_count_2025_2026: 2.0",
                "  topic_aging_count: 1.0",
                "  topic_bioengineering_regeneration_count: 1.0",
                "  topic_systems_omics_count: 1.0",
                "  topic_genetics_epigenetics_count: 1.0",
                "  unique_active_authors: 0.5",
                "  senior_author_signals: 0.5",
                "  young_staff_signals_explicit: 0.5",
                "author_weights:",
                "  relevant_pub_count: 1.0",
                "  first_author_count: 2.0",
                "  last_author_count: 2.0",
                "  corresponding_author_count_explicit: 2.0",
                "  has_istina_profile: 2.0",
                "  has_lab: 1.0",
                "  topic_match_counts: 1.0",
            ]
        ),
        encoding="utf-8",
    )


def _master_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "openalex_id": "W1",
                "doi": "10.1/a",
                "title": "Mesenchymal stem cell engineering for regeneration",
                "publication_date": "2026-02-01",
                "publication_year": 2026,
                "journal": "Journal A",
                "cited_by_count": 12,
                "topic": "Mesenchymal stem cell research",
                "subfield": "Genetics",
                "field": "Medicine",
                "domain": "Health Sciences",
                "author_name": "Author One",
                "author_position": "first",
                "is_first_author": True,
                "is_last_author": False,
                "is_corresponding": True,
                "msu_coauthors": json.dumps(["Author Two"], ensure_ascii=False),
                "msu_coauthor_count": 1,
                "istina_profile_url": "https://istina.msu.ru/profile/a1/",
                "lab": "Lab A",
                "department": "Dept A",
                "institute": "Institute A",
                "position": "ведущий научный сотрудник",
                "matched_title": "Mesenchymal stem cell engineering for regeneration",
            },
            {
                "openalex_id": "W2",
                "doi": "10.1/b",
                "title": "Aging biomarkers in systems biology and omics",
                "publication_date": "2025-07-11",
                "publication_year": 2025,
                "journal": "Journal B",
                "cited_by_count": 6,
                "topic": "Aging biomarker analysis",
                "subfield": "Systems Biology",
                "field": "Medicine",
                "domain": "Health Sciences",
                "author_name": "Author Two",
                "author_position": "last",
                "is_first_author": False,
                "is_last_author": True,
                "is_corresponding": False,
                "msu_coauthors": json.dumps(["Author One"], ensure_ascii=False),
                "msu_coauthor_count": 1,
                "istina_profile_url": "https://istina.msu.ru/profile/a2/",
                "lab": "Lab A",
                "department": "Dept A",
                "institute": "Institute A",
                "position": "аспирант",
                "matched_title": "Aging biomarkers in systems biology and omics",
            },
        ]
    )


def test_score_labs_and_authors_components(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.yaml"
    _weights_yaml(weights_path)
    master_path = tmp_path / "master.parquet"
    _master_df().to_parquet(master_path, index=False, engine="pyarrow")

    lab_scores = score_labs(master_path, config_path=weights_path)
    assert len(lab_scores) == 1
    row = lab_scores.iloc[0]
    assert row["lab_group"] == "Lab A"
    assert row["pub_count"] == 2
    assert row["recent_pub_count_2025_2026"] == 2
    assert row["unique_active_authors"] == 2
    assert row["senior_author_signals"] == 1
    assert row["young_staff_signals_explicit"] == 1
    assert row["total_score"] > 0
    assert "component_pub_count" in lab_scores.columns
    assert "sample_papers" in lab_scores.columns

    author_scores = score_authors(master_path, config_path=weights_path)
    assert len(author_scores) == 2
    first = author_scores.loc[author_scores["author_name"] == "Author One"].iloc[0]
    assert first["first_author_count"] == 1
    assert first["corresponding_author_count_explicit"] == 1
    assert first["has_istina_profile"] == 1
    assert first["has_lab"] == 1
    assert first["topic_match_counts"] >= 1
    assert "component_relevant_pub_count" in author_scores.columns


def test_export_scores_to_excel(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.yaml"
    _weights_yaml(weights_path)
    master_path = tmp_path / "master.parquet"
    _master_df().to_parquet(master_path, index=False, engine="pyarrow")

    lab_xlsx = tmp_path / "lab_scores.xlsx"
    author_xlsx = tmp_path / "author_scores.xlsx"
    lab_summary = export_lab_scores(master_path, lab_xlsx, config_path=weights_path)
    author_summary = export_author_scores(master_path, author_xlsx, config_path=weights_path)

    assert lab_summary["rows"] == 1
    assert author_summary["rows"] == 2
    assert lab_xlsx.exists()
    assert author_xlsx.exists()

    lab_book = pd.ExcelFile(lab_xlsx)
    author_book = pd.ExcelFile(author_xlsx)
    assert "lab_scores" in lab_book.sheet_names
    assert "weights" in lab_book.sheet_names
    assert "author_scores" in author_book.sheet_names
    assert "weights" in author_book.sheet_names
