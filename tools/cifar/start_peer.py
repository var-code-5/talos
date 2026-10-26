#!/usr/bin/env python3
"""Cross-platform entry point, used after installing requirements-cifar.txt."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from mtcp.cifar.__main__ import main

if __name__=='__main__':
    sys.argv.insert(1,'peer')
    main()
