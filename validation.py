from datetime import datetime
import re
from typing import Any, Dict


def _unwrap_field_value(v: Any) -> str:
    """
    Acceptă fie un șir normal, fie un dict de forma
    {"value": "...", "confidence": ...} și întoarce mereu un string.
    """
    if isinstance(v, dict):
        # ne uităm după cheia "value"
        v = v.get("value", "")

    if v is None:
        return ""

    return str(v)


def valid_cnp(cnp: Any) -> bool:
    # importăm aici ca să evităm probleme de import circular
    from ocr_utils import _cnp_ok

    cnp_str = _unwrap_field_value(cnp)
    if not cnp_str:
        return False
    return _cnp_ok(cnp_str)


def parse_date(s: Any):
    """
    Acceptă orice (string sau dict din results_with_confidence),
    extrage doar cifrele și întoarce un datetime sau None.
    Așteptat format: dd.mm.yy sau ddmmyy.
    """
    s_str = _unwrap_field_value(s)
    digits = re.sub(r"[^0-9]", "", s_str)

    if len(digits) < 6:
        return None

    dd = int(digits[:2])
    mm = int(digits[2:4])
    yy = int(digits[4:6])
    year = yy + (2000 if yy < 50 else 1900)

    try:
        return datetime(year, mm, dd)
    except ValueError:
        return None


def validate_fields(fields: Dict[str, Any]) -> Dict[str, str]:
    """
    Primește dict-ul de câmpuri, fie în format vechi (valori string),
    fie în format nou (valori dict cu {"value", "confidence"}).
    Întoarce un dict cu problemele găsite.
    """
    issues: Dict[str, str] = {}

    # --- CNP titular ---
    cnp = _unwrap_field_value(fields.get("cnp"))
    if cnp and not valid_cnp(cnp):
        issues["cnp"] = "invalid_checksum"

    # --- CNP copil ---
    cnp_copil = _unwrap_field_value(fields.get("cnp_copil"))
    if cnp_copil and not valid_cnp(cnp_copil):
        issues["cnp_copil"] = "invalid_checksum"

    # --- Date: de_la / pana_la ---
    d_start = parse_date(fields.get("de_la"))
    d_end = parse_date(fields.get("pana_la"))

    if d_start and d_end:
        if d_end < d_start:
            issues["pana_la"] = "end_before_start"

        # --- Număr zile ---
        nr_zile_raw = _unwrap_field_value(fields.get("nr_zile"))
        if nr_zile_raw.isdigit():
            diff = (d_end - d_start).days + 1
            if abs(diff - int(nr_zile_raw)) > 1:  # tolerăm +/- 1 zi
                issues["nr_zile"] = f"days_mismatch_expected_{diff}"

    return issues
