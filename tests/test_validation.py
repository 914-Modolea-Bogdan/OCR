import pytest

from validation import valid_cnp, parse_date, validate_fields


def test_valid_cnp_true():
    # This CNP was generated using the official checksum algorithm.
    assert valid_cnp("1960523456781") is True


def test_valid_cnp_false():
    assert valid_cnp("1960523456782") is False
    assert valid_cnp("") is False
    assert valid_cnp(None) is False


def test_parse_date_basic():
    d = parse_date("01.07.24")
    assert d.year == 2024
    assert d.month == 7
    assert d.day == 1


def test_parse_date_handles_dict_shape():
    d = parse_date({"value": "310725", "confidence": 0.9})
    assert d.year == 2025
    assert d.month == 7
    assert d.day == 31


def test_validate_fields_ok():
    fields = {
        "cnp": "1960523456781",
        "de_la": "010724",
        "pana_la": "100724",
        "nr_zile": "10",
    }
    issues = validate_fields(fields)
    assert issues == {}


def test_validate_fields_cnp_and_dates():
    fields = {
        "cnp": "1960523456782",  # bad checksum
        "de_la": "100724",
        "pana_la": "010724",  # end before start
        "nr_zile": "15",
    }
    issues = validate_fields(fields)

    assert "cnp" in issues
    assert issues["cnp"] == "invalid_checksum"

    assert "pana_la" in issues
    assert issues["pana_la"] == "end_before_start"

    # days mismatch message should mention expected value
    assert "nr_zile" in issues
    assert "days_mismatch_expected_" in issues["nr_zile"]
