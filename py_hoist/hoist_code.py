"""Run the attribute-hoisting transformer on a single file.

Usage:
    python tools/run_hoist_file.py <input.py> [output.py]

If only <input.py> is provided, the script will write the transformed
output to the same directory with `_hoist` appended to the stem, e.g.
src/foo.py -> src/foo_hoist.py

Examples (from repo root):
    python tools/run_hoist_file.py src/module.py
    python tools/run_hoist_file.py src/module.py src/module_hoist.py

On Windows PowerShell (from repo root):
    python ./tools/run_hoist_file.py ./src/module.py

This script is intended for quick local testing of the hoist transformer
and does not modify the original input file unless `--write` is used.
"""

import sys
from pathlib import Path
import argparse
from typing import Optional, List

# allow running from repo root; ensure tools dir is importable
_tools_dir = Path(__file__).parent
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

# prefer libcst-backed hoister if available
import py_hoist.hoist_ast as hoist


def _hoist_file(src_path: Path, backup_suffix: Optional[str] = None):
    # skip backup files (we use .back suffix for backups)
    # skip backup files (we use .py.back suffix for backups)
    # create b
    # ackup unless it already exists; use .py.back extension
    if backup_suffix:
        suf = backup_suffix if backup_suffix.startswith('.') else f'.{backup_suffix}'
        backup = src_path.with_suffix(suf)
        if backup.exists():
            print(f"Backup already exists: {backup}")
        else:
            backup.write_text(src_path.read_text(encoding = "utf-8"), encoding = "utf-8")
            print(f"Created backup: {backup}")

    src = src_path.read_text(encoding = "utf-8")
    # run hoister with debug=True to surface internal analysis for troubleshooting
    new = hoist.hoist_src(src, src_path = src_path)
    src_path.write_text(new, encoding = "utf-8")


def hoist_path(input_path: str, create_backup: bool = True):
    """Process a file or directory using already-processed parameters.

    - `input_path`: path or Path to file or directory to process.
    - `create_backup`: if True, create `.py.back` backups; if False, do not create backups.
    """
    backup_suffix = 'py.back' if create_backup else None
    inp = Path(input_path)
    if not inp.exists():
        print(f"Input path not found: {inp}")
        raise SystemExit(2)

    if inp.is_dir():
        # Collect all files first, then process them one-by-one while
        # printing an inline progress status that updates the same line.
        files = [p for p in sorted(inp.rglob('*.py'), key = lambda p: str(p)) if '__pycache__' not in p.parts]
        total = len(files)
        for idx, p in enumerate(files, start = 1):
            rel = str(p)
            # Print a 4-letter status (matches 'Done' length) and keep on the same line;
            # after processing overwrite the line with "Done" status.
            print(f"[ {idx:4d}/{total:4d} ] Work       {rel}", end = '\r', flush = True)
            _hoist_file(p, backup_suffix = backup_suffix)
            # overwrite with completed status and move to next line
            print(f"[ {idx:4d}/{total:4d} ] Done       {rel}" + ' ' * 10)
    else:
        _hoist_file(inp, backup_suffix = backup_suffix)


#python -m py_hoist.hoist_code .\src
def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description = 'Run attribute hoister on file or directory')
    parser.add_argument('input_path', help = 'file or directory to process')
    parser.add_argument('-noback',
                        action = 'store_true',
                        dest = 'no_backup',
                        help = 'Do not create backups (by default a .py.back backup is created).')
    args = parser.parse_args(argv)

    create_backup = not args.no_backup
    hoist_path(args.input_path, create_backup = create_backup)


if __name__ == '__main__':
    main()
