"""Resident individual A-share cash dividend tax; all other tax profiles need explicit data."""
import pandas as pd


def dividend_tax_rate(acquired_date,sold_date,ex_date):
    acquired,sold,ex=map(pd.Timestamp,(acquired_date,sold_date,ex_date))
    if acquired>sold or ex<pd.Timestamp('2013-01-01'):raise ValueError('unsupported_dividend_tax_period')
    if sold<=acquired+pd.DateOffset(months=1):return .20
    if sold<=acquired+pd.DateOffset(years=1):return .10
    return .05 if ex<pd.Timestamp('2015-09-08') else 0.
