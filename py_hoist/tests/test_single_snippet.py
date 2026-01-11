#!/usr/bin/env python3
"""Manual test: run hoist_vars_cst on an inline snippet containing the
user-provided `get_pure_raw_dfs` method. This does not read any production
files and will not modify code.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ensure tools dir importable
repo_root = Path(__file__).resolve().parents[1]
tools_dir = repo_root / 'tools'
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

from py_hoist import hoist_src

SNIPPET = '''


class Dummy:
    def test2(items):
        """ 
        Expect: same as test1; `util.do_something` may be bound locally without behavior change.
        Do not: alter the truthiness or side effects of `it.deep`.
        """ 
        total = 0
        for it in items:
            if it.deep:
                total+= util.do_something(it.deep)
        return total 


'''


def main():
    print('--- ORIGINAL SNIPPET ---')
    print(SNIPPET)
    print('\\n--- TRANSFORMED BY HOIST ---')
    out = hoist_src(SNIPPET, src_path = Path(__file__).resolve())
    print(out)


if __name__ == '__main__':
    main()
