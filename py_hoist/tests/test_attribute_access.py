"""
Small benchmark to compare attribute access vs local variable access.
Run: python -u tests/test_attribute_access.py
"""
import time


class Holder:

    def __init__(self, data):
        self.data = data


def run_benchmark(list_size = 1000, iterations = 1000000, repeats = 5):
    h = Holder(list(range(list_size)))
    n = iterations

    def method_attr():
        s = 0
        for i in range(n):
            s += h.data[i % list_size]
        return s

    def method_local():
        s = 0
        data = h.data
        for i in range(n):
            s += data[i % list_size]
        return s

    a_times = []
    b_times = []
    # warmup
    method_attr()
    method_local()

    # Run benchmarks separately to avoid cross-run interference
    for _ in range(repeats):
        t0 = time.perf_counter()
        method_attr()
        a_times.append(time.perf_counter() - t0)

    for _ in range(repeats):
        t0 = time.perf_counter()
        method_local()
        b_times.append(time.perf_counter() - t0)

    def stats(lst):
        return min(lst), sum(lst) / len(lst)

    a_min, a_avg = stats(a_times)
    b_min, b_avg = stats(b_times)

    print(f"settings: list_size={list_size}, iterations={iterations}, repeats={repeats}")
    print(f"attr access: min={a_min:.6f}s avg={a_avg:.6f}s")
    print(f"local var : min={b_min:.6f}s avg={b_avg:.6f}s")
    print(f"ratio (attr/local) min={a_min/b_min:.3f} avg={a_avg/b_avg:.3f}")
    # 百分比（相对于 local）：正数表示 attr 更慢，负数表示 attr 更快
    pct_min = (a_min - b_min) / b_min * 100 if b_min != 0 else float('inf')
    pct_avg = (a_avg - b_avg) / b_avg * 100 if b_avg != 0 else float('inf')
    print(f"percent slower (attr vs local): min={pct_min:.2f}% avg={pct_avg:.2f}%")


def run_max_benchmark(n = 1000000, repeats = 5):
    """Compare calling built-in max vs local alias _max in a tight loop."""
    data = list(range(100))

    def call_builtin():
        s = 0
        for i in range(n):
            s += max(data)
        return s

    def call_local():
        s = 0
        _max = max
        for i in range(n):
            s += _max(data)
        return s

    a_times = []
    b_times = []
    # warmup
    call_builtin()
    call_local()

    for _ in range(repeats):
        t0 = time.perf_counter()
        call_builtin()
        a_times.append(time.perf_counter() - t0)

    for _ in range(repeats):
        t0 = time.perf_counter()
        call_local()
        b_times.append(time.perf_counter() - t0)

    def stats(lst):
        return min(lst), sum(lst) / len(lst)

    a_min, a_avg = stats(a_times)
    b_min, b_avg = stats(b_times)

    print(f"\nmax-call benchmark: n={n}, repeats={repeats}")
    print(f"builtin max : min={a_min:.6f}s avg={a_avg:.6f}s")
    print(f"local _max  : min={b_min:.6f}s avg={b_avg:.6f}s")
    pct_min = (a_min - b_min) / b_min * 100 if b_min != 0 else float('inf')
    pct_avg = (a_avg - b_avg) / b_avg * 100 if b_avg != 0 else float('inf')
    print(f"percent slower (builtin vs local): min={pct_min:.2f}% avg={pct_avg:.2f}%")


if __name__ == '__main__':
    run_benchmark()
    run_max_benchmark()
