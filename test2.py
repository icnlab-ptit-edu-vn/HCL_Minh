"""Compatibility wrapper.

The old evaluator assumed inconsistent action semantics between training and
inference. Use test3.py for the overhead-aware reconfiguration experiments.
"""

from test3 import main

if __name__ == "__main__":
    main()
