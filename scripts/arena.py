"""smallest tensor arena tflite-micro can allocate for each model (bisection).

run with the tflite-micro wheel overlaid, e.g.
uv run --with tflite-micro python scripts/arena.py model.tflite ...
"""

import sys

from tflite_micro.python.tflite_micro import runtime


def fits(model: bytes, size: int) -> bool:
    try:
        runtime.Interpreter.from_bytes(model, arena_size=size)
        return True
    except Exception:
        return False


for path in sys.argv[1:]:
    model = open(path, "rb").read()
    lo, hi = 1024, 1 << 20
    if not fits(model, hi):
        print(f"{path}: does not fit in {hi} bytes")
        continue
    while hi - lo > 16:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if fits(model, mid) else (mid, hi)
    print(f"{hi:7d} bytes  {path}")
