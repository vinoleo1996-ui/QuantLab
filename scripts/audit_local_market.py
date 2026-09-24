#!/usr/bin/env python3
"""Read-only inventory of an existing market catalog before reuse in QuantLab.

Requires the optional duckdb package. No current industry, listing date or raw
price flag is promoted into historical truth merely because it exists in a DB.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def audit(database, output, sample_raw=None, start='2020-01-02', end='2026-06-30'):
    import duckdb
    database=Path(database).resolve();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if database.is_relative_to(output.resolve()):raise ValueError('audit_output_must_not_contain_source_database')
    con=duckdb.connect(str(database),read_only=True)
    con.execute('BEGIN TRANSACTION')
    queries={
        'security_catalog_by_board':"SELECT asset_class,board,count(DISTINCT symbol) AS instruments FROM security_master GROUP BY ALL ORDER BY ALL",
        'security_catalog_sources':"SELECT source_name,count(DISTINCT symbol) AS instruments FROM security_master WHERE asset_class='stock' GROUP BY ALL",
        'bar_sources':"SELECT adjusted,source_name,count(*) AS observations,count(DISTINCT symbol) AS instruments,min(trading_day) AS first_date,max(trading_day) AS last_date FROM daily_bars GROUP BY ALL ORDER BY observations DESC",
        'stock_prices_in_window':"SELECT count(*) AS observations,count(DISTINCT symbol) AS instruments,count(*) FILTER(WHERE amount IS NULL) AS missing_amount FROM daily_bars WHERE NOT adjusted AND trading_day BETWEEN ? AND ? AND symbol IN (SELECT symbol FROM security_master WHERE asset_class='stock')",
        'stocks_per_year':"SELECT year(trading_day) AS year,count(DISTINCT symbol) AS instruments,count(*) AS observations FROM daily_bars WHERE NOT adjusted AND trading_day BETWEEN ? AND ? AND symbol IN (SELECT symbol FROM security_master WHERE asset_class='stock') GROUP BY ALL ORDER BY year",
        'state_coverage_by_year':"SELECT year(effective_on) AS year,count(DISTINCT symbol) AS instruments,count(*) AS observations FROM security_states GROUP BY ALL ORDER BY year",
        'industry_sources':"SELECT source_name,count(DISTINCT symbol) AS instruments,count(*) AS observations FROM industry_history GROUP BY ALL",
        'lifecycle_flags':"SELECT count(DISTINCT symbol) AS instruments,count(DISTINCT symbol) FILTER(WHERE delisted_on IS NOT NULL) AS with_delisting_date,count(DISTINCT symbol) FILTER(WHERE listed_on='2000-01-04') AS shared_2000_start_date FROM security_master WHERE asset_class='stock'",
    }
    result=dict(status='inventory_only_not_execution_qualified',as_of=datetime.now(timezone.utc).isoformat(),
        source_database=str(database),source_access='read_only',start=start,end=end,measurements={},
        limitations=['Catalog membership and bar coverage are not daily eligibility.',
                     'Qlib-derived listing dates and delisting dates need lifecycle verification.',
                     'Historical ST/suspension coverage is incomplete; missing states cannot be marked tradable.',
                     'Industry effective dates alone do not establish historical availability.',
                     'Adjustment factors do not replace separate cash-dividend and share-action accounting.',
                     'No all-market strategy performance has been calculated by this audit.'])
    try:
        for name,sql in queries.items():
            cursor=con.execute(sql,[start,end] if '?' in sql else [])
            result['measurements'][name]=[dict(zip([c[0] for c in cursor.description],row)) for row in cursor.fetchall()]
        # Export candidates without using current survival/ST/industry as filters.
        cursor=con.execute("SELECT symbol,listed_on,delisted_on,board,effective_on,source_name FROM security_master WHERE asset_class='stock' ORDER BY symbol,effective_on")
        import csv
        with (output/'membership_candidates.csv').open('w') as file:
            writer=csv.writer(file)
            writer.writerow(['code','listed_on_unverified','delisted_on_unverified','board','effective_on_unverified','source','usage'])
            for symbol,listed,delisted,board,effective,source in cursor.fetchall():
                code,exchange=symbol.split('.')
                writer.writerow([exchange.lower()+'.'+code,listed,delisted,board,effective,source,'catalog_candidate_only'])
        if sample_raw:
            files=sorted(Path(sample_raw).glob('*_daily.csv'))
            if not files:raise ValueError('no_original_sample_files')
            for path in files:
                metadata=json.loads(path.with_suffix('.json').read_text())
                if metadata.get('error_code')!='0' or hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['sha256']:
                    raise ValueError('original_sample_checksum_failed:'+str(path))
            sql="""WITH original AS (
                SELECT upper(split_part(code,'.',2)||'.'||split_part(code,'.',1)) AS symbol,
                       date::DATE AS trading_day,open,close,amount
                FROM read_csv_auto(?,union_by_name=true)), paired AS (
                SELECT o.*,d.open AS db_open,d.close AS db_close,d.amount AS db_amount
                FROM original o JOIN daily_bars d USING(symbol,trading_day)
                WHERE o.trading_day BETWEEN ? AND ?)
                SELECT count(*) AS paired_rows,count(DISTINCT symbol) AS paired_instruments,
                  sum(CASE WHEN abs(close-db_close)>.011 THEN 1 ELSE 0 END) AS close_differences_above_0011,
                  sum(CASE WHEN abs(open-db_open)>.011 THEN 1 ELSE 0 END) AS open_differences_above_0011,
                  max(abs(close-db_close)) AS max_absolute_close_difference,
                  sum(CASE WHEN amount>0 AND abs(amount-db_amount)/amount>.05 THEN 1 ELSE 0 END) AS amount_differences_above_5pct
                FROM paired"""
            cursor=con.execute(sql,[[str(p) for p in files],start,end])
            result['sample_comparison']=dict(zip([c[0] for c in cursor.description],cursor.fetchone()))
            result['sample_comparison']['provided_files']=len(files)
            result['sample_comparison']['proves_full_market_quality']=False
        result['membership_candidates_sha256']=hashlib.sha256((output/'membership_candidates.csv').read_bytes()).hexdigest()
    finally:
        con.close()
    (output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--database',required=True)
    parser.add_argument('--output',default='output/market_universe_audit')
    parser.add_argument('--sample-raw')
    parser.add_argument('--start',default='2020-01-02');parser.add_argument('--end',default='2026-06-30')
    args=parser.parse_args()
    print(json.dumps(audit(args.database,args.output,args.sample_raw,args.start,args.end),ensure_ascii=False,default=str,indent=2))
