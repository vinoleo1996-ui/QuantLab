#!/usr/bin/env python3
"""运行回测并生成 HTML 报告。用法: python run_backtest.py [source] [market]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quantlab.cli import cmd_backtest  # noqa: E402


class _A:
    source = sys.argv[1] if len(sys.argv) > 1 else "auto"
    market = sys.argv[2] if len(sys.argv) > 2 else "a"


if __name__ == "__main__":
    cmd_backtest(_A())