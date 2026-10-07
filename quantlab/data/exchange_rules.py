"""Conservative limits for seasoned A shares; exceptional listing sessions stay unknown."""
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd

RULE_VERSION='cn_daily_v2'
RULE_VERSIONS=('cn_daily_v1','cn_daily_v2')
# SSE/SZSE trading rules (2026 revision, 上证发〔2026〕41号): main-board risk-warning shares move from 5% to 10%.
MAIN_BOARD_ST_10PCT=pd.Timestamp('2026-07-06')


def limit_ratio(code, date, is_st, listing_session):
    code=str(code).split('.')[-1];date=pd.Timestamp(date)
    if listing_session is None or listing_session<6 or date<pd.Timestamp('1996-12-16') or pd.isna(is_st):return None
    if code.startswith('688'):return .20
    if code.startswith(('300','301')):return .20 if date>=pd.Timestamp('2020-08-24') else (.05 if bool(is_st) else .10)
    if code.startswith(('600','601','603','605','000','001','002','003')):
        return .05 if bool(is_st) and date<MAIN_BOARD_ST_10PCT else .10
    return None


def limit_prices(reference, ratio, tick=.01):
    def rounded(x):return float((x/Decimal(str(tick))).quantize(Decimal('1'),rounding=ROUND_HALF_UP)*Decimal(str(tick)))
    ref=Decimal(str(reference));r=Decimal(str(ratio))
    return rounded(ref*(1+r)),rounded(ref*(1-r))
