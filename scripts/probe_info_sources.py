"""Probe Eastmoney datacenter reports for history depth (earliest/latest date, row count)."""
import json
import sys
import time

import requests

URL = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
HEAD = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://data.eastmoney.com/'}

REPORTS = {
    'margin_stock': ('RPTA_WEB_RZRQ_GGMX', 'DATE'),
    'margin_market': ('RPTA_RZRQ_LSHJ', 'DIM_DATE'),
    'billboard': ('RPT_DAILYBILLBOARD_DETAILSNEW', 'TRADE_DATE'),
    'billboard_inst': ('RPT_ORGANIZATION_TRADE_DETAILSNEW', 'TRADE_DATE'),
    'holder_num': ('RPT_HOLDERNUM_DET', 'HOLD_NOTICE_DATE'),
    'forecast': ('RPT_PUBLIC_OP_NEWPREDICT', 'NOTICE_DATE'),
    'express': ('RPT_FCI_PERFORMANCEE', 'UPDATE_DATE'),
    'survey': ('RPT_ORG_SURVEYNEW', 'NOTICE_DATE'),
    'block_trade': ('RPT_DATA_BLOCKTRADE', 'TRADE_DATE'),
    'unlock': ('RPT_LIFT_STAGE', 'FREE_DATE'),
    'holder_change': ('RPT_SHARE_HOLDER_INCREASE', 'NOTICE_DATE'),
    'buyback': ('RPTA_WEB_GETHGLIST', 'UPD'),
    'north_hold': ('RPT_MUTUAL_HOLDSTOCKNORTH_STA', 'TRADE_DATE'),
    'north_flow': ('RPT_MUTUAL_DEAL_HISTORY', 'TRADE_DATE'),
    'exec_trade': ('RPT_EXECUTIVE_HOLD_DETAILS', 'CHANGE_DATE'),
    'pledge': ('RPT_CSDC_LIST', 'TRADE_DATE'),
}


def get(report, sort_col, order):
    params = {'reportName': report, 'columns': 'ALL', 'pageNumber': 1, 'pageSize': 1,
              'sortColumns': sort_col, 'sortTypes': order, 'source': 'WEB', 'client': 'WEB'}
    r = requests.get(URL, params=params, headers=HEAD, timeout=20)
    return r.json()


out = {}
for key, (report, col) in REPORTS.items():
    row = {'report': report}
    try:
        for order, tag in ((1, 'first'), (-1, 'last')):
            j = get(report, col, order)
            res = j.get('result') or {}
            data = res.get('data') or []
            row['count'] = res.get('count')
            row[tag] = data[0].get(col) if data else j.get('message')
            if tag == 'last' and data:
                row['fields'] = sorted(data[0])[:60]
            time.sleep(0.4)
    except Exception as e:
        row['error'] = repr(e)
    out[key] = row
    print(key, row.get('count'), row.get('first'), row.get('last'), row.get('error', ''), flush=True)

json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
