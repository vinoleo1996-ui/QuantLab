import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from fixtures_v2 import dataset
from quantlab.config import CONFIG
from quantlab.data.contracts import load_dataset, dump_dataset, timestamp, timestamps_series
from quantlab.data.industry import resolve_industries
from quantlab.data.local_adapter import normalize_panel, export_local_dataset, verify_qlib_tree
from quantlab.data.market import universe_at, exposure_matrix, panels_for_research


class IndustryEvidence(unittest.TestCase):
    def test_dictionary_encoded_sector_can_resolve_missing_members_to_unknown(self):
        ds=dataset(days=80);ds.manifest['industry_mode']='effective_date_proxy'
        ds.industry_records=pd.DataFrame([dict(instrument_id='S000',sector='Finance',effective_at='2022-01-01',
            available_at=None,source='history',revision='1',temporal_basis='effective_only')])
        ds.industry_records['sector']=ds.industry_records.sector.astype('category')
        cfg=deepcopy(CONFIG);cfg.research.universe_mode='pit_liquidity'
        amount=pd.DataFrame({c:p.amount for c,p in ds.panels.items()})
        _,records=universe_at(ds,ds.calendar[70],amount,cfg)
        self.assertEqual(records.loc['S000','sector'],'Finance')
        self.assertEqual(records.loc['S001','sector'],'UNKNOWN')

    def test_vector_timestamp_parser_matches_scalar_naive_aware_mixed_and_missing(self):
        for values in [
            ['2024-01-02 18:00:00','2024-01-03 18:00:00',None],
            ['2024-01-02T10:00:00Z','2024-01-03T10:00:00Z',pd.NaT],
            ['2024-01-02 18:00:00','2024-01-02T10:00:00+00:00','2024-01-02T19:00:00+09:00',None],
            [pd.Timestamp('2024-01-02'),pd.Timestamp('2024-01-02',tz='US/Eastern'),pd.NaT],
        ]:
            series=pd.Series(values,index=[f'i{i}' for i in range(len(values))],name='available')
            scalar=pd.to_datetime(series.map(lambda x:timestamp(x) if pd.notna(x) else pd.NaT),utc=True).dt.tz_convert('Asia/Shanghai')
            actual=timestamps_series(series)
            self.assertEqual(list(actual),list(scalar))
            self.assertTrue(actual.index.equals(series.index))
        with self.assertRaises((ValueError,TypeError)):
            timestamps_series(pd.Series(['2024-01-02','not-a-time']))

    def test_effective_proxy_preserves_unknown_publication_and_rejects_future_rows(self):
        ds=dataset();ds.manifest['industry_mode']='effective_date_proxy'
        ds.industry_records=pd.DataFrame([
            dict(instrument_id='S000',sector='old',effective_at='2022-01-01',available_at=None,source='history',revision='1',temporal_basis='effective_only'),
            dict(instrument_id='S000',sector='future',effective_at='2025-01-01',available_at=None,source='history',revision='2',temporal_basis='effective_only')])
        resolved=resolve_industries(ds,'2023-01-03')
        self.assertEqual(resolved.loc['S000','sector'],'old')
        self.assertTrue(pd.isna(resolved.loc['S000','available_at']))
        self.assertTrue(resolve_industries(ds,'2023-01-03','verified_pit').empty)
        e=exposure_matrix(ds,'2023-01-03',ds.instruments.set_index('instrument_id'),['S000'])
        self.assertEqual(e.loc['old','S000'],1.)

    def test_verified_revision_waits_for_publication(self):
        ds=dataset();ds.industry_records=pd.DataFrame([
            dict(instrument_id='S000',sector='old',effective_at='2022-01-01',available_at='2022-01-02T00:00:00+08:00',source='history',revision='1',temporal_basis='verified_pit'),
            dict(instrument_id='S000',sector='new',effective_at='2022-01-01',available_at='2023-01-04T00:00:00+08:00',source='history',revision='2',temporal_basis='verified_pit')])
        self.assertEqual(resolve_industries(ds,'2023-01-03').loc['S000','sector'],'old')
        self.assertEqual(resolve_industries(ds,'2023-01-04').loc['S000','sector'],'new')

    def test_optional_tables_roundtrip_without_certifying_proxy_time(self):
        ds=dataset();ds.manifest['industry_mode']='effective_date_proxy'
        ds.industry_records=pd.DataFrame([dict(instrument_id='S000',sector='Finance',effective_at='2022-01-01',
            available_at=None,source='raw_history',revision='1',temporal_basis='effective_only')])
        ds.financials=pd.DataFrame([dict(instrument_id='S000',metric='net_profit',value=10.,
            report_period='2022-12-31',announced_on='2023-04-01',available_at=None,source='snapshot',revision='1')])
        with tempfile.TemporaryDirectory() as folder:
            path=dump_dataset(ds,folder);loaded=load_dataset(path,'free_research')
            pd.testing.assert_frame_equal(ds.industry_records,loaded.industry_records)
            pd.testing.assert_frame_equal(ds.financials,loaded.financials)
            self.assertEqual(path,dump_dataset(ds,folder))

    def test_daily_unknown_state_does_not_inherit_an_earlier_non_st_flag(self):
        ds=dataset(days=80);cfg=deepcopy(CONFIG);cfg.research.universe_mode='pit_liquidity'
        ds.instruments['state_basis']='daily_observation_required'
        for p in ds.panels.values():p['is_st_at_open']=0.
        ds.panels['S000'].loc[ds.calendar[70],'is_st_at_open']=np.nan
        amount=pd.DataFrame({c:p.amount for c,p in ds.panels.items()})
        selected,_=universe_at(ds,ds.calendar[70],amount,cfg)
        self.assertNotIn('S000',selected)
        self.assertIn('S001',selected)

    def test_observed_age_tracks_in_place_missing_prices_and_panel_replacement(self):
        ds=dataset(days=80,stocks=1);cfg=deepcopy(CONFIG)
        cfg.research.universe_mode='pit_liquidity';cfg.universe.a_min_listing_days=75
        ds.instruments['listing_date']=None
        ds.instruments['observation_start']=ds.calendar[0]
        amount=pd.DataFrame({c:p.amount for c,p in ds.panels.items()})
        date=ds.calendar[74];original=ds.panels['S000'].copy()
        self.assertIn('S000',universe_at(ds,date,amount,cfg)[0])
        ds.panels['S000'].loc[ds.calendar[0],'close']=np.nan
        self.assertNotIn('S000',universe_at(ds,date,amount,cfg)[0])
        ds.panels['S000']=original
        self.assertIn('S000',universe_at(ds,date,amount,cfg)[0])

    def test_observed_age_tracks_appended_and_deleted_rows(self):
        ds=dataset(days=74,stocks=1);cfg=deepcopy(CONFIG)
        cfg.research.universe_mode='pit_liquidity';cfg.universe.a_min_listing_days=75
        ds.instruments['listing_date']=None
        ds.instruments['observation_start']=ds.calendar[0]
        amount=pd.DataFrame({c:p.amount for c,p in ds.panels.items()})
        self.assertNotIn('S000',universe_at(ds,ds.calendar[73],amount,cfg)[0])
        panel=ds.panels['S000'];panel.loc[ds.calendar[74]]=panel.iloc[-1]
        amount.loc[ds.calendar[74]]=amount.iloc[-1]
        self.assertIn('S000',universe_at(ds,ds.calendar[74],amount,cfg)[0])
        panel.drop(index=ds.calendar[0],inplace=True)
        self.assertNotIn('S000',universe_at(ds,ds.calendar[74],amount,cfg)[0])


class LocalSources(unittest.TestCase):
    def test_decoded_prices_must_match_the_hashed_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'qlib_bin';(root/'calendars').mkdir(parents=True);(root/'features'/'sz000001').mkdir(parents=True)
            (root/'calendars'/'day.txt').write_text('2024-01-02\n')
            price=root/'features'/'sz000001'/'close.day.bin';price.write_bytes(b'original')
            archive=Path(directory)/'source.tar.gz'
            with tarfile.open(archive,'w:gz') as bundle:bundle.add(root,arcname='qlib_bin')
            self.assertEqual(verify_qlib_tree(root,archive),2)
            price.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'decoded_checksum_mismatch'):verify_qlib_tree(root,archive)

    def test_no_turnover_or_state_is_invented_from_price_and_volume(self):
        dates=pd.bdate_range('2024-01-02',periods=10)
        raw=pd.DataFrame(dict(open=10.,high=10.,low=10.,close=10.,amount=1e8,volume=1e7,reference_price=10.,feature_close=10.),index=dates)
        p,basic=normalize_panel('sz.000001',raw,None,{},dates)
        self.assertTrue(p.turnover.isna().all())
        self.assertTrue(p.is_st_at_open.isna().all())
        self.assertTrue(p.open_status.eq('unknown').all())
        self.assertIsNone(basic)

    def test_adjustment_snapshot_is_feature_only_and_truncates_causally(self):
        ds=dataset();ds.manifest['feature_price_basis']='historical_adjustment_snapshot'
        for p in ds.panels.values():p['feature_close']=np.arange(len(p))+100.
        prices,_,_=panels_for_research(ds)
        self.assertAlmostEqual(prices.iloc[1,0],101.)
        self.assertNotEqual(ds.panels['S000'].close.iloc[1],101.)
        short=deepcopy(ds);short.manifest['as_of']=str(ds.calendar[35].date())
        pd.testing.assert_frame_equal(prices.loc[:ds.calendar[35]],panels_for_research(short)[0])

    @unittest.skipUnless(importlib.util.find_spec('duckdb'),'optional local lake reader not installed')
    def test_export_freezes_all_observed_codes_and_preserves_unknown_lifecycle(self):
        import duckdb
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);db=root/'lake.duckdb';cache=root/'raw';cache.mkdir()
            con=duckdb.connect(str(db))
            con.execute("CREATE TABLE security_master AS SELECT '000001.SZ' AS symbol,'stock' AS asset_class UNION ALL SELECT '600001.SH','stock' UNION ALL SELECT '688001.SH','stock'")
            dates=pd.bdate_range('2024-01-02',periods=10)
            rows=[]
            for code in ['000001.SZ','600001.SH','688001.SH']:
                for d in dates:rows.append(dict(symbol=code,trading_day=d,open=10.,high=10.,low=10.,close=10.,previous_close=10.,volume=100000,amount=1000000.,source_name='raw',raw_hash='r1',adjusted=False))
            con.register('input_bars',pd.DataFrame(rows));con.execute('CREATE TABLE daily_bars AS SELECT * FROM input_bars')
            con.execute("CREATE TABLE adjustment_factors AS SELECT symbol,trading_day,1.0 AS factor FROM daily_bars")
            con.register('input_calendar',pd.DataFrame(dict(exchange='CN',calendar_day=dates,is_open=True)))
            con.execute('CREATE TABLE trading_calendar AS SELECT * FROM input_calendar')
            con.execute('CREATE TABLE security_states(symbol VARCHAR,effective_on DATE,is_st BOOLEAN,is_suspended BOOLEAN,source_name VARCHAR,raw_hash VARCHAR)')
            con.execute("INSERT INTO security_states VALUES ('000001.SZ','2024-01-10',false,false,'observed','s1')")
            con.execute('CREATE TABLE industry_history(symbol VARCHAR,industry VARCHAR,effective_on DATE,source_name VARCHAR,raw_hash VARCHAR,classification VARCHAR)')
            con.execute("INSERT INTO industry_history VALUES ('000001.SZ','Finance','2022-01-01','history','i1','csrc_2012')")
            con.execute('CREATE TABLE financial_records(symbol VARCHAR,metric VARCHAR,value DOUBLE,report_period DATE,announced_on DATE,source_name VARCHAR,raw_hash VARCHAR)')
            con.close()
            pd.DataFrame({'date':dates,'close':100.}).to_csv(cache/'benchmark.csv',index=False)
            digest=hashlib.sha256((cache/'benchmark.csv').read_bytes()).hexdigest()
            (cache/'benchmark.json').write_text(json.dumps(dict(error_code='0',sha256=digest)))
            original=hashlib.sha256(db.read_bytes()).hexdigest()
            result=export_local_dataset(db,root/'snapshots',raw_cache_roots=[cache],start=dates[0],end=dates[-1])
            self.assertEqual(result['securities'],3)
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(),original)
            ds=load_dataset(result['manifest'],'free_research')
            self.assertEqual(len(ds.panels),3)
            rules=ds.instruments.set_index('instrument_id')
            self.assertEqual(rules.loc['sh.600001','sell_lot'],100)
            self.assertEqual(rules.loc['sh.688001','sell_lot'],1)
            self.assertEqual(rules.loc['sh.688001','min_sell_quantity'],200)
            self.assertTrue(ds.instruments.listing_date.isna().all())
            self.assertTrue(ds.industry_records.available_at.isna().all())
            self.assertEqual(ds.validate('free_research'),[])
            self.assertIn('industry_proxy_not_PIT',ds.validate('strict'))
            repeat=export_local_dataset(db,root/'snapshots',raw_cache_roots=[cache],start=dates[0],end=dates[-1])
            self.assertEqual(result['manifest'],repeat['manifest'])

            # An observed unresolved distribution remains in the snapshot even
            # when its cash amount is below the price-reference gap threshold.
            code='sz.000001';exdates=['2024-01-05','2024-01-09','2024-01-11']
            actions=[]
            for ex,record in zip(exdates,['2024-01-04','2024-01-08','2024-01-10']):
                actions.append(dict(action_id=code+':'+ex,instrument_id=code,record_date=record,ex_date=ex,
                    pay_date=ex,share_date=ex,cash_per_share=1.,share_ratio=0.,available_at='2024-01-03 00:00:00',
                    source='complete_second_source',revision='a1',quality_status='ok',tax_policy='gross_cn_individual',action_type='distribution'))
            action_path=cache/'corporate_actions.csv';pd.DataFrame(actions).to_csv(action_path,index=False)
            action_path.with_suffix('.json').write_text(json.dumps(dict(error_code='0',sha256=hashlib.sha256(action_path.read_bytes()).hexdigest())))
            partial=cache/(code+'_dividend_2023.csv')
            pd.DataFrame([
                dict(dividOperateDate='2024-01-05',dividRegistDate='2024-01-04',dividPlanDate='2024-01-02',dividPayDate=None,dividCashPsBeforeTax=1.),
                dict(dividOperateDate='2024-01-05',dividRegistDate='2024-01-03',dividPlanDate='2024-01-02',dividPayDate=None,dividCashPsBeforeTax=1.),
                dict(dividOperateDate='2024-01-09',dividRegistDate='2024-01-08',dividPlanDate='2024-01-02',dividPayDate=None,dividCashPsBeforeTax=2.)
            ]).to_csv(partial,index=False)
            partial.with_suffix('.json').write_text(json.dumps(dict(error_code='0',sha256=hashlib.sha256(partial.read_bytes()).hexdigest())))
            gaps=[
                dict(instrument_id=code,date='2024-01-05',reason='distribution_payment_or_share_date_missing',cash_per_share_reported=1.,share_ratio_reported=0.),
                dict(instrument_id=code,date='2024-01-09',reason='distribution_payment_or_share_date_missing',cash_per_share_reported=2.,share_ratio_reported=0.),
                dict(instrument_id=code,date='2024-01-11',reason='multiple_distribution_records_same_day_not_reconciled',cash_per_share_reported=1.,share_ratio_reported=0.),
                dict(instrument_id=code,date='2024-01-12',reason='distribution_payment_or_share_date_missing',cash_per_share_reported=.005,share_ratio_reported=0.)]
            gap_path=cache/'action_gaps.json';gap_path.write_text(json.dumps(gaps))
            gap_hash=hashlib.sha256(gap_path.read_bytes()).hexdigest()
            gap_path.with_suffix('.meta.json').write_text(json.dumps(dict(sha256=gap_hash)))
            with_gaps=export_local_dataset(db,root/'snapshots',raw_cache_roots=[cache],start=dates[0],end=dates[-1])
            revised=load_dataset(with_gaps['manifest'],'free_research')
            source=revised.manifest['source_files']['accounting_gap_sources'][0]
            self.assertEqual(source['sha256'],gap_hash)
            reported=[g for g in revised.manifest['accounting_gaps'] if 'gap_source_sha256' in g]
            self.assertEqual({g['date'] for g in reported},{'2024-01-09','2024-01-11','2024-01-12'})
            remaining=[g for g in revised.manifest['data_gaps'] if g['reason']=='incomplete_distribution_dates']
            self.assertEqual([g['date'] for g in remaining],['2024-01-05','2024-01-09'])
            self.assertEqual(remaining[0]['record_date'],'2024-01-03')
            gap_path.write_text('[]')
            with self.assertRaisesRegex(ValueError,'action_gap_checksum_mismatch'):
                export_local_dataset(db,root/'snapshots',raw_cache_roots=[cache],start=dates[0],end=dates[-1])


if __name__=='__main__':unittest.main()
