#!/usr/bin/env python3
from py_hoist import hoist_ast_util as u

TESTS = [
            "QDate(current_date.year(), current_date.month(), 1).addMonths",
]


def main():
    for s in TESTS:
        print('INPUT:', s)
        print(u.split_by_brackets__(s))
        print()


if __name__ == '__main__':
    main()
