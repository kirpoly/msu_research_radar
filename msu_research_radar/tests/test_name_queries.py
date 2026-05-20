from __future__ import annotations

import pytest

from msu_research_radar.istina.name_queries import generate_istina_name_queries


@pytest.mark.parametrize(
    ("raw_name", "must_include"),
    [
        ("Alexandrushkina N A", ["Alexandrushkina N", "Александрушкина Н"]),
        ("N. A. Alexandrushkina", ["Alexandrushkina N", "Александрушкина Н"]),
        ("Khocheva N", ["Khocheva N", "Хочева Н"]),
        ("Khokhlov A N", ["Khokhlov A", "Хохлов А"]),
        ("Разин С В", ["Разин С", "Razin S"]),
        ("Скулачев В", ["Скулачев В", "Skulachev V"]),
        ("Moskalev A", ["Moskalev A", "Москалев А"]),
        ("Eremichev R Yu", ["Eremichev R", "Еремичев Р"]),
        ("Григорьева О А", ["Григорьева О", "Grigoreva O"]),
        ("Vodopetova Maria A", ["Vodopetova M", "Водопетова М"]),
    ],
)
def test_generate_istina_name_queries_has_core_variants(
    raw_name: str,
    must_include: list[str],
) -> None:
    queries = generate_istina_name_queries(raw_name)
    assert queries
    for expected in must_include:
        assert expected in queries


def test_generate_istina_name_queries_ranks_short_form_first() -> None:
    queries = generate_istina_name_queries("Alexandrushkina N A")
    assert queries[0] == "Alexandrushkina N"

