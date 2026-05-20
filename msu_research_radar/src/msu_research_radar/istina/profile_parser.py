"""Strict parser for Istina MSU profile pages."""

from __future__ import annotations

import re
from typing import Any

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _first_current_position(personal_info: Tag) -> Tag | None:
    for position_block in personal_info.select("p.position"):
        if position_block.find_parent(id="non_current_employments") is None:
            return position_block
    return None


def _extract_irid(personal_info: Tag) -> tuple[str | None, str | None]:
    pattern = re.compile(r"IstinaResearcherID\s*\(IRID\)\s*:\s*(\d+)")
    for div in personal_info.select("div"):
        text = _clean_text(div.get_text(" ", strip=True))
        if not text:
            continue
        match = pattern.search(text)
        if match:
            return match.group(1), text
    return None, None


def _extract_scopus_author_id(personal_info: Tag) -> tuple[str | None, str | None]:
    for anchor in personal_info.select("a[href*='scopus']"):
        href = anchor.get("href", "")
        if "author" not in href.lower():
            continue
        text = _clean_text(anchor.get_text(" ", strip=True))
        if text:
            digits_match = re.search(r"\d+", text)
            if digits_match:
                return digits_match.group(0), text
        href_match = re.search(r"authorid=(\d+)", href, flags=re.IGNORECASE)
        if href_match:
            return href_match.group(1), href
    return None, None


def _classify_affiliations(affiliations: list[str]) -> tuple[str | None, str | None, str | None]:
    lab = None
    department = None
    institute = None

    for item in affiliations:
        lowered = item.lower()
        if lab is None and "лаборатор" in lowered:
            lab = item
        if department is None and "кафедр" in lowered:
            department = item
        if institute is None and "институт" in lowered:
            institute = item

    return lab, department, institute


def parse_istina_profile_html(html: str, url: str | None = None) -> dict[str, Any]:
    """Parse profile data from a saved Istina profile HTML."""
    soup = BeautifulSoup(html, "html.parser")
    personal_info = soup.select_one("div.personal-info")

    result: dict[str, Any] = {
        "full_name": None,
        "profile_url": url,
        "username": None,
        "irid": None,
        "researcher_id": None,
        "orcid": None,
        "spin": None,
        "scopus_author_id": None,
        "current_affiliations": [],
        "lab": None,
        "department": None,
        "institute": None,
        "position": None,
        "visible_blocks": [],
        "raw_text_snippets": [],
    }

    result["visible_blocks"] = [
        value for value in (li.get("id") for li in soup.select("li[id^='profile.']")) if value
    ]

    if personal_info is None:
        return result

    full_name = personal_info.select_one("a.fullname")
    result["full_name"] = _clean_text(full_name.get_text(" ", strip=True) if full_name else None)

    username = personal_info.select_one("span.badge.badge-primary")
    result["username"] = _clean_text(username.get_text(" ", strip=True) if username else None)

    current_position = _first_current_position(personal_info)
    if current_position is not None:
        role = current_position.select_one("span.name")
        result["position"] = _clean_text(role.get_text(" ", strip=True) if role else None)

        affiliations = [
            text
            for text in (
                _clean_text(anchor.get_text(" ", strip=True))
                for anchor in current_position.select("a[href*='/organizations/department/']")
            )
            if text
        ]
        result["current_affiliations"] = affiliations
        lab, department, institute = _classify_affiliations(affiliations)
        result["lab"] = lab
        result["department"] = department
        result["institute"] = institute

        position_text = _clean_text(current_position.get_text(" ", strip=True))
        if position_text:
            result["raw_text_snippets"].append(position_text)

    irid, irid_snippet = _extract_irid(personal_info)
    result["irid"] = irid
    if irid_snippet:
        result["raw_text_snippets"].append(irid_snippet)

    researcher = personal_info.select_one("a[href*='webofscience.com/wos/author/record/']")
    if researcher:
        researcher_text = _clean_text(researcher.get_text(" ", strip=True))
        result["researcher_id"] = researcher_text
        if researcher_text:
            result["raw_text_snippets"].append(f"ResearcherID: {researcher_text}")

    orcid = personal_info.select_one("a[href*='orcid.org/']")
    if orcid:
        orcid_text = _clean_text(orcid.get_text(" ", strip=True))
        result["orcid"] = orcid_text
        if orcid_text:
            result["raw_text_snippets"].append(f"ORCID: {orcid_text}")

    spin = personal_info.select_one("a[href*='elibrary.ru/authors.asp?spin=']")
    if spin:
        spin_text = _clean_text(spin.get_text(" ", strip=True))
        result["spin"] = spin_text
        if spin_text:
            result["raw_text_snippets"].append(f"SPIN: {spin_text}")

    scopus_author_id, scopus_snippet = _extract_scopus_author_id(personal_info)
    result["scopus_author_id"] = scopus_author_id
    if scopus_snippet:
        result["raw_text_snippets"].append(scopus_snippet)

    return result


def parse_istina_profile_url(url: str, timeout: float = 20.0) -> dict[str, Any]:
    """Download profile page and parse it without saving to disk."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return parse_istina_profile_html(response.text, url=url)

