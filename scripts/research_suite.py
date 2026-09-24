#!/usr/bin/env python3
"""Compatible entrypoint for the registered research pipeline."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quantlab.cli import main

if __name__=='__main__':main(['research',*sys.argv[1:]])
