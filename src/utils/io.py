from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def step(name: str):
    start = time.time()
    print(f"\n=== {name} ===")
    try:
        yield
    finally:
        secs = time.time() - start
        print(f"--- done: {name} ({secs:.1f}s) ---")
