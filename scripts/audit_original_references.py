#!/usr/bin/env python3
"""Audit DOI metadata in the supplied original manuscript against Crossref."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "paper/original/paper_final.docx"
OUTPUT = ROOT / "reports/original_reference_audit.csv"
USER_AGENT = "dhaka-aqi-reference-audit/0.2 (mailto:awnonbhowmik@outlook.com)"

# These dispositions are editorial judgments based on relevance and source type;
# DOI metadata checks below are independent of this mapping.
DISPOSITIONS = {
    1: ("context_candidate", "Valid methods review; not needed in concise revision."),
    2: ("remove", "Preprint only; not needed for the revised station-specific claims."),
    3: ("remove", "AQICN is an aggregator and not an observational source for this study."),
    4: ("remove", "News report cannot support a monthly concentration series."),
    5: ("remove", "News report cannot support a monthly concentration series."),
    9: ("retain_revised", "Official Bangladesh policy source; cite exact publication page."),
    10: ("replace", "Vague title; replace with exact DoE 2018-2023 report metadata."),
    11: ("replace", "Vague recurring report title; cite a dated report if used."),
    12: ("remove", "News report cannot support a monthly concentration series."),
    14: ("remove", "Supplied title/DOI could not be matched as cited."),
    20: ("retain_revised", "Verified peer-reviewed Dhaka black-carbon context."),
    21: ("remove", "Dataset exists, but its measurement lineage is inadequate for trend use."),
    22: ("correct_retain", "Correct journal/DOI: Frontiers in Sustainable Cities, 10.3389/frsc.2021.681759."),
    23: ("correct_before_use", "Correct year/DOI are 2014 and 10.1016/j.atmosenv.2014.09.046."),
    24: ("retain_revised", "Verified peer-reviewed greater-Dhaka seasonality/forecast context."),
    26: ("retain_revised", "Verified peer-reviewed Dhaka winter PM2.5/black-carbon context."),
    29: ("remove", "The supplied Atmospheric Environment DOI does not resolve; title unverified."),
    35: ("remove", "News report cannot support a monthly concentration series."),
    36: ("remove", "Annual contextual HDI is outside revised inferential scope."),
    37: ("context_candidate", "Official contextual source, but not used for local burden estimation."),
    38: ("retain_revised", "Official health-guideline source."),
    39: ("remove", "Annual poverty context is outside revised inferential scope."),
    40: ("replace", "Secondary population site; use United Nations primary data if needed."),
}


def manuscript_markdown() -> str:
    result = subprocess.run(
        ["pandoc", str(ORIGINAL), "-t", "gfm", "--wrap=none"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_references(markdown: str) -> list[tuple[int, str]]:
    reference_text = markdown.split("**References**", maxsplit=1)[1]
    matches = list(re.finditer(r"(?m)^(\d+)\.\s+", reference_text))
    references: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(reference_text)
        citation = " ".join(reference_text[match.end() : end].split())
        references.append((int(match.group(1)), citation))
    return references


def citation_doi(citation: str) -> str:
    match = re.search(r"https?://doi\.org/([^\s]+)", citation, flags=re.IGNORECASE)
    return "" if match is None else match.group(1).rstrip(".,);]")


def citation_year(citation: str) -> str:
    match = re.search(r"\((20\d{2}|19\d{2})", citation)
    return "" if match is None else match.group(1)


def citation_title(citation: str) -> str:
    match = re.search(r"\((?:20\d{2}|19\d{2})[^)]*\)\.\s+(.+)", citation)
    if match is None:
        return ""
    return match.group(1).split(". ", maxsplit=1)[0].replace("\\[", "[").replace("\\]", "]")


def normalized_title(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    value = value.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    return "".join(character.lower() for character in value if character.isalnum())


def crossref_metadata(doi: str) -> tuple[str, dict[str, object] | None]:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return "resolved", json.load(response)["message"]
    except urllib.error.HTTPError as exc:
        return f"http_{exc.code}", None
    except (OSError, ValueError) as exc:
        return f"error_{type(exc).__name__}", None


def registry_year(metadata: dict[str, object]) -> str:
    for key in ("published-print", "published-online", "issued"):
        value = metadata.get(key)
        if isinstance(value, dict):
            parts = value.get("date-parts")
            if isinstance(parts, list) and parts and parts[0]:
                return str(parts[0][0])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Parse citations without Crossref")
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    retrieved = dt.date.today().isoformat()
    for number, citation in parse_references(manuscript_markdown()):
        doi = citation_doi(citation)
        expected_year = citation_year(citation)
        expected_title = citation_title(citation)
        status = "no_doi"
        metadata = None
        if doi and not args.offline:
            status, metadata = crossref_metadata(doi)
            time.sleep(0.1)
        elif doi:
            status = "not_queried"
        title = ""
        published_year = ""
        title_similarity = ""
        metadata_match = "not_checked"
        if metadata is not None:
            titles = metadata.get("title", [])
            title = titles[0] if isinstance(titles, list) and titles else ""
            published_year = registry_year(metadata)
            title_similarity = difflib.SequenceMatcher(
                None,
                normalized_title(expected_title),
                normalized_title(title),
            ).ratio()
            year_matches = not expected_year or expected_year == published_year
            title_matches = title_similarity >= 0.8
            if year_matches and title_matches:
                metadata_match = "yes"
            elif not year_matches and not title_matches:
                metadata_match = "year_and_title_mismatch"
            elif not year_matches:
                metadata_match = "year_mismatch"
            else:
                metadata_match = "title_mismatch"
        disposition, note = DISPOSITIONS.get(
            number,
            ("context_candidate", "Verify relevance before carrying into the concise revised paper."),
        )
        if metadata_match in {"title_mismatch", "year_and_title_mismatch"}:
            disposition = "correct_before_use"
            note = "The supplied DOI resolves to a different article title; correct or remove this citation."
        rows.append(
            {
                "reference_number": number,
                "original_citation": citation,
                "doi": doi,
                "resolver_status": status,
                "registry_title": title,
                "citation_title": expected_title,
                "title_similarity": title_similarity,
                "citation_year": expected_year,
                "registry_year": published_year,
                "metadata_match": metadata_match,
                "disposition": disposition,
                "notes": note,
                "retrieval_date": retrieved,
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} references to {OUTPUT}")


if __name__ == "__main__":
    main()
