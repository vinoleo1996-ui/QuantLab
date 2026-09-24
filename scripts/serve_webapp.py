#!/usr/bin/env python3
"""启动 Web 看板。用法: python serve_webapp.py [port]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quantlab.cli import cmd_web  # noqa: E402


class _A:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8899


if __name__ == "__main__":
    cmd_web(_A())