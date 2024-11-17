# src/utils.py

import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f"[{name}] Elapsed time: {elapsed_time:.2f} seconds")
