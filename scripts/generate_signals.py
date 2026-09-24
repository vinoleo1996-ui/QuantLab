#!/usr/bin/env python3
"""输出本期买卖建议清单(信号级，人工执行)。用法: python generate_signals.py [source] [market]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quantlab.cli import cmd_signals  # noqa: E402


class _A:
    source = sys.argv[1] if len(sys.argv) > 1 else "auto"
    market = sys.argv[2] if len(sys.argv) > 2 else "a"


if __name__ == "__main__":
    cmd_signals(_A())