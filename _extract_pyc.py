"""
Extract structure from a .pyc file using dis + marshal.
Usage: python _extract_pyc.py <file.pyc>
"""
import dis, marshal, struct, sys, types
from pathlib import Path

def read_pyc(path):
    with open(path, "rb") as f:
        f.read(16)          # magic (4) + flags (4) + timestamp (4) + size (4)
        return marshal.loads(f.read())

def walk(code, indent=0):
    pad = "  " * indent
    name = code.co_name
    args = list(code.co_varnames[:code.co_argcount])
    doc  = code.co_consts[0] if code.co_consts and isinstance(code.co_consts[0], str) else None

    print(f"{pad}{'='*60}")
    print(f"{pad}CODE: {name}  args={args}  locals={list(code.co_varnames)}")
    if doc:
        first_line = doc.split('\n')[0].strip()[:120]
        print(f"{pad}  DOC: {first_line!r}")
    # constants (non-code, non-None, non-trivial)
    consts = [c for c in code.co_consts
              if c is not None and not isinstance(c, types.CodeType)
              and c != doc and c != 0 and c != 1 and c != -1]
    if consts:
        print(f"{pad}  CONSTS: {[repr(c)[:80] for c in consts[:20]]}")
    # names used (globals, attributes)
    if code.co_names:
        print(f"{pad}  NAMES: {list(code.co_names)}")
    # free/cell vars
    if code.co_freevars or code.co_cellvars:
        print(f"{pad}  FREE={list(code.co_freevars)} CELL={list(code.co_cellvars)}")

    # recurse into nested code objects
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            walk(c, indent + 1)

if __name__ == "__main__":
    path = sys.argv[1]
    print(f"\n{'#'*70}")
    print(f"# FILE: {path}")
    print(f"{'#'*70}\n")
    code = read_pyc(path)
    walk(code)
