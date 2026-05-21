from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from msu_research_radar.authors.msu_author_extractor import extract_msu_authors


def test_extract_msu_authors_filters_and_aggregates(tmp_path: Path) -> None:
    authorships = [
        {
            "author_position": "first",
            "author": {"display_name": "Author MSU 1"},
            "raw_affiliation_strings": ["Lomonosov Moscow State University"],
            "institutions": [{"display_name": "MSU", "ror": "https://ror.org/010pmpe69"}],
        },
        {
            "author_position": "middle",
            "author": {"display_name": "Author External"},
            "raw_affiliation_strings": ["External University"],
            "institutions": [{"display_name": "Other", "ror": "https://ror.org/000000000"}],
        },
        {
            "author_position": "last",
            "author": {"display_name": "Author MSU 2"},
            "raw_affiliation_strings": ["Faculty of Biology, Lomonosov Moscow State University"],
            "institutions": [{"display_name": "MSU", "ror": "https://ror.org/010pmpe69"}],
        },
    ]
    works = pd.DataFrame(
        [
            {
                "openalex_id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.1/test",
                "title": "Test Work",
                "authorships": json.dumps(authorships, ensure_ascii=False),
            }
        ]
    )
    works_path = tmp_path / "works.parquet"
    output_csv = tmp_path / "msu_authors.csv"
    works.to_parquet(works_path, index=False, engine="pyarrow")

    summary = extract_msu_authors(works_path, output_csv)

    assert summary["authors_extracted"] == 2
    df = pd.read_csv(output_csv)
    assert sorted(df["author_name"].tolist()) == ["Author MSU 1", "Author MSU 2"]
    msu1 = df[df["author_name"] == "Author MSU 1"].iloc[0]
    assert msu1["paper_count"] == 1
    assert msu1["first_author_count"] == 1
    assert msu1["last_author_count"] == 0
    coauthors = json.loads(msu1["coauthors_msu"])
    assert coauthors == ["Author MSU 2"]

