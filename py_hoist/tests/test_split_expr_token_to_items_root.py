#!/usr/bin/env python3

from py_hoist import hoist_ast_util as u

TESTS = [
            "a...b",
            "[x][y].z",
            "func(arg1, arg2)",
            "obj..prop",
            "a[b].c[d].e",
]
TESTS = [
            "QDate(current_date.year(), current_date.month(), 1).addMonths",
]


def main():
    for s in TESTS:
        print('INPUT:', s)
        res = u.split_expr_token_to_items__(s)
        print('keys():', res.keys())
        print('values():', res.values())
        print('items():', res.items())
        print()


if __name__ == '__main__':
    main()
