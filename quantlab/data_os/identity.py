"""内部 instrument_id。代码只是别名，退市代码继续留在主表里。"""

from __future__ import annotations

import pandas as pd

from quantlab.data_os.schema import ALIAS_COLUMNS, INSTRUMENT_COLUMNS, require_columns


def instrument_id(country: str, exchange: str, local_code: str) -> str:
    country = country.strip().upper()
    exchange = exchange.strip().upper()
    local_code = local_code.strip().upper()
    if not country or not exchange or not local_code:
        raise ValueError("instrument_id 需要 country、exchange、local_code")
    return f"{country}.{exchange}.{local_code}"


def a_share_identity(symbol: str) -> dict:
    text = symbol.strip().lower().replace(".", "")
    if text.startswith(("sh", "sz", "bj")) and len(text) >= 8:
        prefix, code = text[:2], text[2:8]
    else:
        code = text[-6:].zfill(6)
        if code.startswith(("4", "8", "92")):
            prefix = "bj"
        elif code.startswith(("5", "6", "9")):
            prefix = "sh"
        else:
            prefix = "sz"
    exchange = {"sh": "XSHG", "sz": "XSHE", "bj": "BJSE"}[prefix]
    return {
        "instrument_id": instrument_id("CN", exchange, code),
        "symbol": code,
        "exchange": exchange,
        "country": "CN",
        "asset_type": "stock",
        "currency": "CNY",
        "baostock_code": f"{prefix}.{code}",
        "tencent_code": f"{prefix}{code}",
        "akshare_code": code,
    }


def us_identity(symbol: str, exchange: str = "US") -> dict:
    code = symbol.strip().upper()
    venue = (exchange or "US").strip().upper()
    return {
        "instrument_id": instrument_id("US", venue, code),
        "symbol": code,
        "exchange": venue,
        "country": "US",
        "asset_type": "stock",
        "currency": "USD",
    }


def master_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in INSTRUMENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["status"] = frame["status"].fillna("listed")
    return require_columns(frame.drop_duplicates("instrument_id"), INSTRUMENT_COLUMNS, "instrument_master")


def alias_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in ALIAS_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return require_columns(frame, ALIAS_COLUMNS, "instrument_alias")
