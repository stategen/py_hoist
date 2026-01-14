"""
Benchmark: compare attribute access when the field is defined on the parent
vs defined on the child (child uses @dataclass(slots=True)).

Run: python -u tests/test_dataclass_slots_access.py
"""
import time
from dataclasses import dataclass
import statistics


@dataclass
class ParentWithField:
    val: int = 1


@dataclass(slots = True)
class ChildSlotsInherit(ParentWithField):
    # child uses slots=True but parent field is declared in parent (no slots)
    # attribute 'val' is not added to child's __slots__.
    val2: int = 2


@dataclass
class ParentEmpty:
    pass


@dataclass(slots = True)
class ChildSlotsOwnField(ParentEmpty):
    # here 'val' is declared on the child and will be placed into child's __slots__
    val: int = 1
    val2: int = 2


def bench_access(obj, iterations: int):
    s = 0
    for i in range(iterations):
        s += obj.val
    return s


def run_dataclass_slots_benchmark(iterations: int = 20_000_0000, repeats: int = 1):
    # Prepare instances
    obj_inherited = ChildSlotsInherit()  # field defined on parent
    obj_own = ChildSlotsOwnField()  # field defined on child (slots)

    inh_times = []
    own_times = []

    # Run timings: first measure inherited (parent-defined) attribute access,
    # then measure child-defined (stored in __slots__) attribute access.
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(iterations):
            val1 = obj_inherited.val
        inh_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for ii in range(iterations):
            val2 = obj_own.val
        own_times.append(time.perf_counter() - t0)

    def stats(lst):
        return {
                    'min': min(lst),
                    'median': statistics.median(lst),
                    'avg': sum(lst) / len(lst),
                    'std': statistics.pstdev(lst),
        }

    inh = stats(inh_times)
    own = stats(own_times)

    print(f"settings: iterations={iterations}, repeats={repeats}")
    print(f"attr on parent (inherited, not in child's __slots__) : min={inh['min']}s avg={inh['avg']}s avg={inh['avg']:}s std={inh['std']:}s"
          )
    print(f"attr on child  (declared in child's __slots__)       : min={own['min']}s avg={own['avg']}s avg={own['avg']}s std={own['std']}s"
          )
    pct_median = (own['avg'] - inh['avg']) / inh['avg'] * 100 if inh['avg'] != 0 else float('inf')
    print(f"percent change by avg (child-slot vs parent-inherited): {pct_median:.2f}%")

    if inh['avg'] < own['avg']:
        print('Result by avg: parent-declared attribute is FASTER')
    elif own['avg'] < inh['avg']:
        print('Result by avg: child-slot attribute is FASTER')
    else:
        print('Result by avg: equal')


if __name__ == '__main__':
    run_dataclass_slots_benchmark()
