#!/usr/bin/env python3
"""从 AKShare/yfinance 拉取候选池日线数据并缓存(增量)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quantlab.cli import cmd_fetch, run_backtest  # noqa: E402


def main() -> None:
    source = sys.argv[1] if len(sys.argv) > 1 else "auto"
    market = sys.argv[2] if len(sys.argv) > 2 else "a"
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    from quantlab.cli import build_panels
    panels, meta, bench = build_panels(source, market, limit)
    print(f"[fetch_data] {market} panels={len(panels)} meta={meta.shape} bench_len={len(bench)}")
    print("[fetch_data] 下一步: python -m quantlab.cli backtest --source akshare --market a")


if __name__ == "__main__":
    main()