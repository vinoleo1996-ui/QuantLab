#!/usr/bin/env python3
"""Compatible wrapper for the versioned research CLI."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _cli_compat import forward
if __name__=='__main__':forward('backtest',sys.argv[1:])
