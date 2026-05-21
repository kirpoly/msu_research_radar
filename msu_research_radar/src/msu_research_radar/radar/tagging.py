"""Generic configurable tag matching for radar profiles."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml


TAG_SECTIONS = ("affiliation_tags", "topic_tags", "role_tags")


def clean_text(value: Any) -> str:
    """Return a compact string representation."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def lower_text(value: Any) -> str:
    """Return normalized lowercase text."""
    return clean_text(value).lower()


def json_list(value: Any) -> list[Any]:
    """Parse list-like values stored as JSON strings in tabular exports."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return parsed if isinstance(parsed, list) else [parsed]
    return []


def json_dumps_list(values: list[Any]) -> str:
    """Serialize a list with stable ordering where possible."""
    return json.dumps(values, ensure_ascii=False)


def normalize_name(value: str) -> str:
    """Normalize a person name for conservative joins."""
    return " ".join(re.sub(r"[^\w\s]", " ", value.lower(), flags=re.UNICODE).split())


def load_research_profiles(config_path: str | Path) -> dict[str, Any]:
    """Load all configured research profiles from YAML."""
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    profiles = payload.get("profiles") or {}
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("research profile config must contain a non-empty 'profiles' mapping")
    return profiles


def load_research_profile(config_path: str | Path, profile_name: str | None = None) -> dict[str, Any]:
    """Load one research profile from YAML.

    If profile_name is omitted, the first profile in the YAML file is used.
    """
    profiles = load_research_profiles(config_path)
    selected_name = profile_name or next(iter(profiles))
    if selected_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown research profile {selected_name!r}. Available: {available}")
    profile = dict(profiles[selected_name] or {})
    profile["name"] = selected_name
    return profile


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _term_matches(text: str, term: str) -> bool:
    return bool(term.strip()) and term.lower() in text


def _regex_matches(text: str, pattern: str) -> bool:
    if not pattern.strip():
        return False
    return re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE) is not None


def match_text_rule(text: str, rule: dict[str, Any]) -> tuple[bool, list[str]]:
    """Match a text rule with include/exclude terms and regexes."""
    normalized = lower_text(text)
    include_terms = _as_list(rule.get("include_terms"))
    include_regex = _as_list(rule.get("include_regex"))
    exclude_terms = _as_list(rule.get("exclude_terms"))
    exclude_regex = _as_list(rule.get("exclude_regex"))

    exclude_hits = [f"exclude_term:{term}" for term in exclude_terms if _term_matches(normalized, term)]
    exclude_hits.extend(f"exclude_regex:{pattern}" for pattern in exclude_regex if _regex_matches(normalized, pattern))
    if exclude_hits:
        return False, exclude_hits

    include_hits = [f"term:{term}" for term in include_terms if _term_matches(normalized, term)]
    include_hits.extend(f"regex:{pattern}" for pattern in include_regex if _regex_matches(normalized, pattern))
    return bool(include_hits), include_hits


def match_role_rule(row: dict[str, Any], rule: dict[str, Any]) -> tuple[bool, list[str]]:
    """Match role tags against author role signals and Istina position text."""
    hits: list[str] = []
    author_position = lower_text(row.get("author_position"))
    position_text = lower_text(row.get("position"))

    wanted_positions = {lower_text(item) for item in _as_list(rule.get("author_positions"))}
    if author_position and author_position in wanted_positions:
        hits.append(f"author_position:{author_position}")

    for term in _as_list(rule.get("position_terms")):
        if _term_matches(position_text, term):
            hits.append(f"position_term:{term}")

    if rule.get("first_author") is True and bool(row.get("is_first_author")):
        hits.append("first_author:true")
    if rule.get("last_author") is True and bool(row.get("is_last_author")):
        hits.append("last_author:true")
    if rule.get("corresponding_author") is True and bool(row.get("is_corresponding")):
        hits.append("corresponding_author:true")

    return bool(hits), hits


def _affiliation_corpus(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("lab", "department", "institute", "position", "affiliation_hint"):
        parts.append(clean_text(row.get(key)))
    for key in ("raw_affiliation_strings", "known_msu_affiliations", "current_affiliations"):
        for item in json_list(row.get(key)):
            parts.append(clean_text(item))
    return " ".join(part for part in parts if part)


def _topic_corpus(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "abstract", "journal", "topic", "subfield", "field", "domain", "matched_title"):
        parts.append(clean_text(row.get(key)))
    for item in json_list(row.get("topics")):
        parts.append(clean_text(item))
    return " ".join(part for part in parts if part)


def _section_corpus(section: str, row: dict[str, Any]) -> str:
    if section == "affiliation_tags":
        return _affiliation_corpus(row)
    if section == "topic_tags":
        return _topic_corpus(row)
    return ""


def apply_profile_tags(row: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Apply generic profile tags to one row and return match metadata."""
    output: dict[str, Any] = {
        "matched_affiliation_tags": [],
        "matched_topic_tags": [],
        "matched_role_tags": [],
        "profile_match_score": 0.0,
        "match_reasons": [],
    }

    for section in TAG_SECTIONS:
        section_rules = profile.get(section) or {}
        if not isinstance(section_rules, dict):
            continue
        for tag_name, rule_payload in section_rules.items():
            rule = rule_payload or {}
            if section == "role_tags":
                matched, hits = match_role_rule(row, rule)
            else:
                matched, hits = match_text_rule(_section_corpus(section, row), rule)
            if not matched:
                continue
            output[f"matched_{section}"].append(str(tag_name))
            output["profile_match_score"] += float(rule.get("weight", 1.0))
            output["match_reasons"].extend(f"{section}.{tag_name}:{hit}" for hit in hits)

    for key in ("matched_affiliation_tags", "matched_topic_tags", "matched_role_tags", "match_reasons"):
        output[key] = sorted(set(output[key]))
    return output
