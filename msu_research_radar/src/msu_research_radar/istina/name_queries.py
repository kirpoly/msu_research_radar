"""Heuristic generator of Istina employee search queries."""

from __future__ import annotations

import re

CYR_TO_LAT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

LAT_MULTI_TO_CYR = [
    ("shch", "щ"),
    ("yo", "ё"),
    ("yu", "ю"),
    ("ya", "я"),
    ("zh", "ж"),
    ("kh", "х"),
    ("ts", "ц"),
    ("ch", "ч"),
    ("sh", "ш"),
]

LAT_SINGLE_TO_CYR = {
    "a": "а",
    "b": "б",
    "c": "к",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "х",
    "i": "и",
    "j": "й",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "к",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "w": "в",
    "x": "кс",
    "y": "й",
    "z": "з",
}


def _tokenize(name: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё]+", name)


def _is_cyrillic_token(token: str) -> bool:
    return any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in token)


def _translit_cyr_to_lat(text: str) -> str:
    output: list[str] = []
    for ch in text:
        lower = ch.lower()
        mapped = CYR_TO_LAT.get(lower, ch)
        if ch.isupper() and mapped:
            mapped = mapped[0].upper() + mapped[1:]
        output.append(mapped)
    return "".join(output)


def _translit_lat_to_cyr(text: str) -> str:
    lower = text.lower()
    i = 0
    out: list[str] = []
    while i < len(lower):
        matched = False
        for chunk, repl in LAT_MULTI_TO_CYR:
            if lower.startswith(chunk, i):
                out.append(repl)
                i += len(chunk)
                matched = True
                break
        if matched:
            continue
        ch = lower[i]
        out.append(LAT_SINGLE_TO_CYR.get(ch, ch))
        i += 1
    value = "".join(out)
    return value[:1].upper() + value[1:] if value else value


def _is_initial(token: str) -> bool:
    return len(token) == 1


def _extract_surname_and_initials(tokens: list[str]) -> tuple[str, list[str]]:
    if not tokens:
        return "", []

    candidates = [token for token in tokens if not _is_initial(token)]

    if not candidates:
        surname = tokens[-1]
        return surname, initials

    first = tokens[0]
    last = tokens[-1]
    if not _is_initial(first) and any(_is_initial(t) for t in tokens[1:]):
        surname = first
    elif not _is_initial(last) and any(_is_initial(t) for t in tokens[:-1]):
        surname = last
    else:
        surname = max(candidates, key=len)

    initials: list[str] = []
    skipped_surname = False
    for token in tokens:
        if token == surname and not skipped_surname:
            skipped_surname = True
            continue
        initials.append(token[0].upper())

    deduped: list[str] = []
    seen: set[str] = set()
    for initial in initials:
        if initial in seen:
            continue
        seen.add(initial)
        deduped.append(initial)
    initials = deduped[:2]

    return surname, initials


def _initial_to_script(initial: str, target_cyrillic: bool) -> str:
    if target_cyrillic:
        return _translit_lat_to_cyr(initial) if re.match(r"[A-Za-z]", initial) else initial
    if re.match(r"[А-Яа-яЁё]", initial):
        translit = _translit_cyr_to_lat(initial)
        return translit[:1].upper() if translit else initial
    return initial


def _build_variants(surname: str, initials: list[str], target_cyrillic: bool) -> list[str]:
    if not surname:
        return []

    script_surname = _translit_lat_to_cyr(surname) if target_cyrillic else _translit_cyr_to_lat(surname)
    if target_cyrillic and _is_cyrillic_token(surname):
        script_surname = surname
    if not target_cyrillic and not _is_cyrillic_token(surname):
        script_surname = surname

    script_initials = [_initial_to_script(value, target_cyrillic) for value in initials if value]
    script_initials = [value[:1].upper() for value in script_initials if value]

    if not script_initials:
        return [script_surname]

    i1 = script_initials[0]
    variants = [
        f"{script_surname} {i1}",
        f"{script_surname} {i1}.",
    ]
    if len(script_initials) > 1:
        i2 = script_initials[1]
        variants.extend(
            [
                f"{script_surname} {i1} {i2}",
                f"{script_surname} {i1}.{i2}.",
                f"{script_surname} {i1}.{i2}",
            ]
        )
    return variants


def generate_istina_name_queries(name: str) -> list[str]:
    """Generate ranked Istina employee search query variants."""
    tokens = _tokenize(name)
    surname, initials = _extract_surname_and_initials(tokens)
    if not surname:
        cleaned = " ".join(tokens).strip()
        return [cleaned] if cleaned else []

    original_is_cyr = _is_cyrillic_token(surname)
    variants = []
    variants.extend(_build_variants(surname, initials, target_cyrillic=original_is_cyr))
    variants.extend(_build_variants(surname, initials, target_cyrillic=not original_is_cyr))

    seen: set[str] = set()
    output: list[str] = []
    for variant in variants:
        normalized = " ".join(variant.split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output
