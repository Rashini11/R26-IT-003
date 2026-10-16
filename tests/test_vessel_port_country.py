from backend.vessel_detection.services import port_country_service


def test_lookup_fuzzy_matches_port_from_csv_and_returns_country():
    result = port_country_service.lookup_port_country("Port of Colomobo")

    assert result["status"] == "found"
    assert result["port"] == "Colombo"
    assert result["country"] == "Sri Lanka"
    assert result["country_code"] == "LK"
    assert result["port_type"] == "Local"
    assert result["source"] == "world_ports.csv"


def test_lookup_requires_choice_when_port_is_in_multiple_countries(monkeypatch):
    monkeypatch.setattr(
        port_country_service,
        "_load_world_ports",
        lambda: (
            {"port": "Victoria", "country": "Canada", "port_code": "CAVIC"},
            {"port": "Victoria", "country": "Hong Kong", "port_code": "HKVIC"},
        ),
    )

    result = port_country_service.lookup_port_country("Victoria")

    assert result["status"] == "ambiguous"
    assert result["country"] is None
    assert {item["country_code"] for item in result["candidates"]} == {"CA", "HK"}


def test_lookup_marks_known_non_sri_lanka_port_foreign():
    result = port_country_service.lookup_port_country("Hamburg")

    assert result["status"] == "found"
    assert result["country"] == "Germany"
    assert result["port_type"] == "Foreign"


def test_lookup_defaults_unmatched_port_to_foreign():
    result = port_country_service.lookup_port_country("zzqzxwqv")

    assert result["status"] == "not_found"
    assert result["country"] is None
    assert result["port_type"] == "Foreign"
