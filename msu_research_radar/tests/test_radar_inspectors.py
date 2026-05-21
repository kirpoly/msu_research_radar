from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from msu_research_radar.radar.inspectors import inspect_affiliations, inspect_topics


def test_inspect_affiliations_exports_counts_and_samples(tmp_path: Path) -> None:
    input_path = tmp_path / "works.parquet"
    output_path = tmp_path / "affiliations.csv"
    pd.DataFrame(
        [
            {
                "author_name": "Author One",
                "title": "Paper One",
                "raw_affiliation_strings": json.dumps(
                    ["Faculty of Biology, Lomonosov Moscow State University"],
                    ensure_ascii=False,
                ),
            },
            {
                "author_name": "Author Two",
                "title": "Paper Two",
                "raw_affiliation_strings": json.dumps(
                    ["Faculty of Biology, Lomonosov Moscow State University"],
                    ensure_ascii=False,
                ),
            },
        ]
    ).to_parquet(input_path, index=False, engine="pyarrow")

    summary = inspect_affiliations(input_path, output_path)

    assert summary["rows"] == 1
    out = pd.read_csv(output_path)
    assert out.iloc[0]["count"] == 2
    assert "Author One" in out.iloc[0]["sample_authors"]


def test_inspect_topics_exports_hierarchy_and_topic_lists(tmp_path: Path) -> None:
    input_path = tmp_path / "works.parquet"
    output_path = tmp_path / "topics.csv"
    pd.DataFrame(
        [
            {
                "title": "Paper One",
                "journal": "Journal A",
                "topic": "Genomics",
                "subfield": "Systems Biology",
                "topics": json.dumps(["Genomics", "Transcriptomics"], ensure_ascii=False),
            },
        ]
    ).to_parquet(input_path, index=False, engine="pyarrow")

    summary = inspect_topics(input_path, output_path)

    assert summary["rows"] == 4
    out = pd.read_csv(output_path)
    assert set(out["topic_value"]) == {"Genomics", "Systems Biology", "Transcriptomics"}
    assert set(out.loc[out["topic_value"] == "Genomics", "source_field"]) == {"topic", "topics"}
    assert "Paper One" in out.iloc[0]["sample_titles"]
