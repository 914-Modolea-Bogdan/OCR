# ai_corrector.py
from __future__ import annotations

import re
from typing import Any, Dict

from ocr_utils import _cnp_best13, format_date_from_digits
from validation import valid_cnp


def _unwrap(v: Any) -> tuple[str, float]:
    """
    Acceptă:
      - string simplu
      - dict {"value": "...", "confidence": ...}
    și întoarce (valoare, confidence).
    """
    if isinstance(v, dict):
        return str(v.get("value", "") or ""), float(v.get("confidence", 0.0) or 0.0)
    if v is None:
        return "", 0.0
    return str(v), 0.0


def ai_correct_fields(fields: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Mic „engine de auto-corect”:
      - normalizează câmpuri numerice (lungime fixă)
      - normalizează datele și le aliniază cu luna/anul din antet
      - corectează CNP (caută cea mai bună secvență de 13 cifre, cu checksum)
    Returnează: { name: {"value": ..., "confidence": ...}, ... }
    """

    out: Dict[str, Dict[str, Any]] = {}

    # === 0) Context: luna și anul din „Valabil pentru luna ... anul ...” ===
    ctx_month = None  # 1..12
    ctx_year = None   # ultimele 2 cifre (ex: 25 pentru 2025)

    raw_month, _ = _unwrap(fields.get("valabil_pentru_luna_digits", ""))
    m_digits = re.sub(r"[^0-9]", "", raw_month)
    if len(m_digits) >= 1:
        try:
            m = int(m_digits[:2])
            if 1 <= m <= 12:
                ctx_month = m
        except ValueError:
            pass

    raw_year, _ = _unwrap(fields.get("valabil_pentru_anul", ""))
    y_digits = re.sub(r"[^0-9]", "", raw_year)
    if len(y_digits) >= 2:
        try:
            ctx_year = int(y_digits[:2])
        except ValueError:
            pass

    # ținte de lungime pentru câmpurile strict numerice
    target_len = {
        "valabil_pentru_luna_digits": 2,
        "valabil_pentru_anul": 2,
        "cod_indemnizatie": 2,
        "nr_zile": 2,
        "cod_diagnostic": 5,
        "cnp": 13,
        "cnp_copil": 13,
    }

    for name, raw in fields.items():
        val, conf = _unwrap(raw)
        orig_val, orig_conf = val, conf  # dacă vrei să loghezi înainte/după

        # --- 1) câmpuri strict numerice (luăm DOAR cifre, tăiem la lungime) ---
        if name in {
            "valabil_pentru_luna_digits",
            "valabil_pentru_anul",
            "cod_indemnizatie",
            "nr_zile",
            "cod_diagnostic",
            "nr_inregistrare",
        }:
            digits = re.sub(r"[^0-9]", "", val)
            tlen = target_len.get(name)
            if tlen is not None and len(digits) >= tlen:
                digits = digits[:tlen]
            val = digits
            if val and conf < 0.8:
                conf = max(conf, 0.7)

        # --- 2) lună (01–12), forțată după context dacă există ---
        if name == "valabil_pentru_luna_digits" and val:
            try:
                m = int(val[:2])
                if not (1 <= m <= 12) and ctx_month is not None:
                    m = ctx_month
                m = max(1, min(12, m))
                val = f"{m:02d}"
                conf = max(conf, 0.85)
            except ValueError:
                pass

        # --- 3) date: Data acordării / De la / Până la ---
        if name in {"data_acordarii", "de_la", "pana_la"}:
            digits = re.sub(r"[^0-9]", "", val)

            # încercăm să împărțim în dd mm yy
            if len(digits) >= 6:
                dd = digits[0:2]
                mm = digits[2:4]
                yy = digits[4:6]

                # dacă luna e invalidă sau diferită de context, o forțăm
                if ctx_month is not None:
                    try:
                        m_int = int(mm)
                    except ValueError:
                        m_int = 0
                    if not (1 <= m_int <= 12) or m_int != ctx_month:
                        mm = f"{ctx_month:02d}"

                # dacă avem context de an, îl forțăm în yy
                if ctx_year is not None:
                    yy = f"{ctx_year:02d}"

                digits = dd + mm + yy

            pretty = format_date_from_digits(digits)
            if pretty:
                val = pretty          # de forma 31.07.2025
                conf = max(conf, 0.9)
            else:
                # fallback – păstrăm primele 6 cifre brute
                val = digits[:6]

        # --- 4) CNP / CNP copil ---
        if name in {"cnp", "cnp_copil"}:
            digits = re.sub(r"[^0-9]", "", val)
            best = _cnp_best13(digits)
            if best:
                val = best
                if valid_cnp(val):
                    conf = max(conf, 0.9)

        out[name] = {
            "value": val,
            "confidence": float(conf),
        }

    return out
