import csv
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path


_WORLD_PORTS_FILE = Path(__file__).with_name("world_ports.csv")
_MIN_CANDIDATE_SCORE = 0.55
_AUTO_MATCH_SCORE = 0.78
_AMBIGUOUS_SCORE_MARGIN = 0.06
_GENERIC_PORT_WORDS = {"harbor", "harbour", "of", "port", "seaport", "terminal"}


def _normalize_port_name(value):
    """Make port names easier to compare despite spelling and accent differences."""
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    words = re.findall(r"[a-z0-9]+", ascii_value)
    return " ".join(word for word in words if word not in _GENERIC_PORT_WORDS)


def _port_name_variants(value):
    """Return the full port name and any names written in parentheses."""
    variants = {_normalize_port_name(value)}
    for alias in re.findall(r"\(([^()]*)\)", value or ""):
        normalized_alias = _normalize_port_name(alias)
        if normalized_alias:
            variants.add(normalized_alias)
    return variants - {""}


@lru_cache(maxsize=1)
def _load_world_ports():
    """Read and cache the bundled list of world ports."""
    try:
        with _WORLD_PORTS_FILE.open(encoding="utf-8-sig", newline="") as port_file:
            reader = csv.DictReader(port_file)
            expected_columns = {"port_name", "country", "port_code"}
            if not reader.fieldnames or not expected_columns.issubset(reader.fieldnames):
                raise RuntimeError(
                    "The world ports CSV must contain port_name, country, and port_code columns."
                )
            ports = tuple(
                {
                    "port": (row.get("port_name") or "").strip(),
                    "country": (row.get("country") or "").strip(),
                    "port_code": (row.get("port_code") or "").strip(),
                }
                for row in reader
                if (row.get("port_name") or "").strip()
                and (row.get("country") or "").strip()
            )
    except OSError as error:
        raise RuntimeError(f"Could not read the world ports CSV: {error}") from error

    if not ports:
        raise RuntimeError("The world ports CSV does not contain any port records.")
    return ports


def _score_port_name(query_variants, port_name):
    """Return the strongest fuzzy match score for two port names."""
    return max(
        SequenceMatcher(None, query, candidate).ratio()
        for query in query_variants
        for candidate in _port_name_variants(port_name)
    )


def _port_type_for_country(country):
    """Use Local for Sri Lanka ports and Foreign for all other countries."""
    return "Local" if (country or "").strip().casefold() == "sri lanka" else "Foreign"


def lookup_port_country(port_name):
    """Match a port to the CSV and return its country when the match is clear."""
    port_name = (port_name or "").strip()
    if not port_name:
        raise ValueError("Enter a port name before looking up its country.")
    if len(port_name) > 120:
        raise ValueError("Port name must be 120 characters or fewer.")

    query_variants = _port_name_variants(port_name)
    if not query_variants:
        raise ValueError("Enter a specific port name, not only the word port or harbour.")

    matched_ports = {}
    for port in _load_world_ports():
        score = _score_port_name(query_variants, port["port"])
        if score >= _MIN_CANDIDATE_SCORE:
            candidate = {
                **port,
                "country_code": port["port_code"][:2].upper(),
                "port_type": _port_type_for_country(port["country"]),
                "match_score": round(score, 3),
            }
            candidate_key = (
                _normalize_port_name(candidate["port"]),
                candidate["country"].casefold(),
            )
            previous = matched_ports.get(candidate_key)
            if previous is None or candidate["match_score"] > previous["match_score"]:
                matched_ports[candidate_key] = candidate

    matched_ports = list(matched_ports.values())
    matched_ports.sort(key=lambda candidate: candidate["match_score"], reverse=True)
    best = matched_ports[0] if matched_ports else None
    close_matches = [
        candidate for candidate in matched_ports
        if best and best["match_score"] - candidate["match_score"] <= _AMBIGUOUS_SCORE_MARGIN
    ]
    close_countries = {candidate["country"].casefold() for candidate in close_matches}
    confident = bool(best and best["match_score"] >= _AUTO_MATCH_SCORE)

    if confident and len(close_countries) == 1:
        result = {
            "status": "found",
            "query": port_name,
            "port": best["port"],
            "country": best["country"],
            "country_code": best["country_code"],
            "port_code": best["port_code"],
            "port_type": _port_type_for_country(best["country"]),
            "match_score": best["match_score"],
            "candidates": close_matches,
            "source": "world_ports.csv",
        }
    elif confident and len(close_countries) > 1:
        result = {
            "status": "ambiguous",
            "query": port_name,
            "port": None,
            "country": None,
            "country_code": None,
            "port_code": None,
            "port_type": "Foreign",
            "match_score": best["match_score"],
            "candidates": close_matches,
            "source": "world_ports.csv",
        }
    else:
        result = {
            "status": "not_found",
            "query": port_name,
            "port": None,
            "country": None,
            "country_code": None,
            "port_code": None,
            "port_type": "Foreign",
            "match_score": best["match_score"] if best else None,
            "candidates": [],
            "source": "world_ports.csv",
        }

    return result
