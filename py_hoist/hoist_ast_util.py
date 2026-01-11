from __future__ import annotations

import ast
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Set

from py_hoist.multi_items import MultiItems

# Reusable local helpers to avoid repeating the same isinstance tuples.
LOOP_TYPES = (ast.For, getattr(ast, 'AsyncFor', type(None)))
LOOP_OR_WHILE = (ast.For, ast.While, getattr(ast, 'AsyncFor', type(None)))
# from px_util.profile_util import profile


def _is_loop_or_while_node(node: ast.AST) -> bool:
    return isinstance(node, LOOP_OR_WHILE)


def _target_contains_base(t: ast.AST, base_text: str) -> bool:
    """Return True if target `t` (Name/Tuple/List/other) contains `base_text`.

    This consolidates the repeated checks for Assign/AnnAssign/For targets.
    """
    if isinstance(t, (ast.Tuple, ast.List)):
        for el in t.elts:
            if isinstance(el, ast.Name) and el.id == base_text:
                return True
            el_unparsed = safe_unparse__(el)
            if el_unparsed == base_text:
                return True
        return False
    if isinstance(t, ast.Name) and t.id == base_text:
        return True
    t_unparsed = safe_unparse__(t)
    if t_unparsed == base_text:
        return True
    return False


def attr_chain_text_ast__(node: ast.Attribute) -> str:
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        parts.append(ast.unparse(cur))
    return ".".join(reversed(parts))


def safe_unparse__(node: ast.AST) -> Optional[str]:
    """Unparse `node` to a compact string (no spaces).

    Note: removed exception swallowing — allow exceptions to propagate so
    callers can observe and fix unexpected AST shapes.
    """
    return ast.unparse(node).replace(' ', '')


def add_line_numbers__(source: str) -> str:
    """Add 4-digit line numbers to each line of source code.
    
    Line numbers are right-aligned in a 4-character field followed by a space.
    Does not catch exceptions - they propagate to caller.
    
    Args:
        source: Source code string
        
    Returns:
        String with line numbers prepended to each line
    """
    lines = source.splitlines(keepends = True)
    result = []
    for i, line in enumerate(lines, 1):
        result.append(f"{i:4d} {line}")
    return ''.join(result)


def sanitize_ident__(s: str) -> str:
    """Sanitize an arbitrary string into a valid Python identifier fragment.

    Replaces non identifier characters with underscores and ensures the
    result starts with a letter by prefixing `const_` when necessary.

    This function intentionally does not catch exceptions; callers should
    ensure input is a string.
    """
    # Encode each non-identifier character as a hex token `0xHH` (easy to paste
    # into online hex decoders). Leave letters/digits/underscore unchanged.
    parts: List[str] = []
    for ch in s:
        # map quotes to underscore for readability and to avoid heavy encoding
        if ch == '"' or ch == "'" or ch == '[' or ch == ']':
            parts.append('_')
        elif ch == '.':
            parts.append('_dot_')
        elif re.match(r'[0-9A-Za-z_]', ch):
            parts.append(ch)
        else:
            parts.append(f"0x{ord(ch):02X}")
    res = ''.join(parts)
    # If the first character of the resulting identifier would be a digit,
    # prefix with `v_` to ensure it's a valid Python identifier start.
    if res and res[0].isdigit():
        res = 'v_' + res

    return res


def short_hash__(s: str) -> str:
    """Return an 8-character hex short hash for string `s`.

    Uses SHA-256 and returns the first 8 hex digits (32 bits).
    """
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[: 8]


def find_name_ident_used_or_new__(
            name: str,
            ident_name: str,
            used_name_ident_map: dict[str, str],
            used_ident_counters: dict[str, int],
) -> str:
    used_ident = used_name_ident_map.get(name)
    if used_ident is not None:
        return used_ident

    used_cnt = used_ident_counters.get(ident_name)
    if used_cnt is None or used_cnt == 0:
        used_cnt = 0
    else:
        used_cnt += 1

    new_ident_name = ident_name
    max_len = 30

    if len(ident_name) > max_len + 9:
        new_ident_name = ident_name[: max_len] + '_' + short_hash__(ident_name)

    if used_cnt > 0:
        new_ident_name = new_ident_name + '_' + str(used_cnt)

    used_ident_counters[ident_name] = used_cnt
    used_name_ident_map = new_ident_name
    return new_ident_name


def node_matches_chain(node: ast.AST,
                       target_chain: str,
                       target_prefixes: List[str],
                       base_name: str,
                       *,
                       allow_prefixes: bool = True) -> bool:
    """Return True if `node` matches the `target_chain`.

    If `allow_prefixes` is True also match any prefix in `target_prefixes`.
    Additionally, treat a `Name` node that assigns to `base_name` as a match
    (this covers rebinding the base variable itself).
    """
    t_unparsed = safe_unparse__(node)
    if t_unparsed is not None:
        if t_unparsed == target_chain:
            return True
        if allow_prefixes and t_unparsed in target_prefixes:
            return True

    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute):
            # Only consider inner Attribute nodes when prefixes are allowed.
            if not allow_prefixes:
                continue
            sub_unparsed = safe_unparse__(sub)
            if sub_unparsed is not None and sub_unparsed in target_prefixes:
                return True
        elif isinstance(sub, ast.Name):
            if sub.id == base_name and isinstance(sub.ctx, (ast.Store, ast.Del)):
                return True
    return False


def is_in_first_operand_of_boolop__(attr_node: ast.AST, parent_map: Dict[ast.AST, ast.AST]) -> bool:
    """Return True if `attr_node` is located in the first (left-most)
    operand of the nearest enclosing `ast.BoolOp`.

    This is a conservative check used to preserve short-circuit semantics
    when considering hoisting from boolean `and`/`or` expressions.
    """
    # Walk upwards to find nearest BoolOp ancestor
    cur = attr_node
    boolop_node = None
    while cur is not None:
        p = parent_map.get(cur)
        if isinstance(p, ast.BoolOp):
            boolop_node = p
            break
        cur = p
    if boolop_node is None:
        return False

    # Descend into left-most operand
    left = boolop_node.values[0] if boolop_node.values else None
    # Walk up from attr_node and ensure we reach `left` before leaving the BoolOp
    cur = attr_node
    while cur is not None and cur is not boolop_node:
        if cur is left:
            return True
        cur = parent_map.get(cur)
    return left is boolop_node and (attr_node is boolop_node or parent_map.get(attr_node) is boolop_node)


def _node_defines_base(node: ast.AST, base_text: str) -> bool:
    """Return True if `node` defines `base_text` (class name, assign target, import alias, loop target).

    Consolidates repeated checks used when searching for a base definition.
    """
    if isinstance(node, ast.ClassDef):
        return node.name == base_text
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if _target_contains_base(t, base_text):
                return True
    if isinstance(node, ast.AnnAssign):
        t = node.target
        if _target_contains_base(t, base_text):
            return True

    if isinstance(node, (ast.For, getattr(ast, 'AsyncFor', type(None)))):
        t = node.target
        if _target_contains_base(t, base_text):
            return True

    # comprehensions have `ast.comprehension` nodes with `target` fields
    if isinstance(node, ast.comprehension):
        t = node.target
        if _target_contains_base(t, base_text):
            return True

    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            if alias.asname == base_text:
                return True
            if alias.name.split('.')[0] == base_text:
                return True
    return False


def build_parent_map__(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    """Build a mapping from child -> parent for all nodes under `tree`.

    Use this map for upward navigation instead of mutating AST nodes.
    """
    parent_map: Dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent
    return parent_map


def get_up_loop_node__(node: Optional[ast.AST], func_parent_map: Dict[ast.AST, ast.AST]) -> ast.AST:
    """Return True if `node` is inside any loop/while ancestor according to parent_map."""
    cur = node
    while cur is not None:
        if _is_loop_or_while_node(cur):
            return cur
        cur = func_parent_map.get(cur)
    return None


def get_up_comprehension_expr__(node: Optional[ast.AST], func_parent_map: Dict[ast.AST, ast.AST]) -> Optional[ast.AST]:
    """Return the nearest enclosing comprehension expression node (`ListComp`,
    `DictComp`, `SetComp`, or `GeneratorExp`) or None.

    Many comprehension internals (like the f-string expression) are children of
    the comprehension expression node rather than of the `ast.comprehension`
    generator nodes. Use this helper when you want the whole comprehension
    expression as the loop-like ancestor.
    """
    cur = node
    while cur is not None:
        if isinstance(cur, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            return cur
        cur = func_parent_map.get(cur)
    return None


def is_type_annotation__(node: ast.AST, parent_map: Dict[ast.AST, ast.AST]) -> bool:
    """Check if a node is part of a type annotation (AnnAssign, arg, or FunctionDef returns)."""
    curr = node
    while curr is not None:
        parent = parent_map.get(curr)
        if parent is None:
            break

        # Check if curr is the annotation field of its parent
        if isinstance(parent, ast.AnnAssign) and curr is parent.annotation:
            return True
        if isinstance(parent, ast.arg) and curr is parent.annotation:
            return True
        if isinstance(parent, (ast.FunctionDef, getattr(ast, 'AsyncFunctionDef', type(None)))) and curr is parent.returns:
            return True

        curr = parent
    return False


def walk_no_nested__(root: ast.AST):
    """Yield nodes under `root` but do not descend into nested FunctionDef/ClassDef nodes.

    This prevents counting occurrences inside nested functions or classes as
    belonging to the outer function.
    """
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        # Push children in reverse order so they are processed in forward order (LIFO)
        children = list(ast.iter_child_nodes(node))
        for child in reversed(children):
            # do not descend into nested defs that are not the root
            if child is root:
                stack.append(child)
            elif isinstance(child, (ast.FunctionDef, getattr(ast, 'AsyncFunctionDef', type(None)), ast.ClassDef)):
                continue
            else:
                stack.append(child)


def find_stmt_ancestor__(node: ast.stmt, parent_map: Dict[ast.AST, ast.AST]) -> Optional[ast.stmt]:
    cur = node
    while cur is not None and not isinstance(cur, ast.stmt):
        cur = parent_map.get(cur)
    return cur


def get_if_info__(stmt: ast.AST, parent_map: Dict[ast.AST, ast.AST]) -> Tuple[Optional[ast.If], Tuple[Optional[ast.If], str]]:
    """Returns (root_if, (immediate_if, branch_type))
    branch_type is 'test', 'body', or 'orelse'.
    """
    # If the statement itself is an If, it's the 'test' of that If
    if isinstance(stmt, ast.If):
        root_if = stmt
        while True:
            p = parent_map.get(root_if)
            if isinstance(p, ast.If) and root_if in p.orelse:
                root_if = p
                continue
            break
        return root_if, (stmt, 'test')

    curr = stmt
    immediate_if = None
    branch_type = 'none'

    # Find the immediate If parent
    p = parent_map.get(curr)
    while p:
        if isinstance(p, ast.If):
            immediate_if = p
            if curr is p.test:
                branch_type = 'test'
            elif curr in p.body:
                branch_type = 'body'
            elif curr in p.orelse:
                branch_type = 'orelse'
            break
        curr = p
        p = parent_map.get(curr)

    if not immediate_if:
        return None, (None, 'none')

    # Find root of if-elif chain
    root_if = immediate_if
    while True:
        p_of_root = parent_map.get(root_if)
        if isinstance(p_of_root, ast.If) and root_if in p_of_root.orelse:
            root_if = p_of_root
            continue
        break
    return root_if, (immediate_if, branch_type)


def is_in_first_logical_path__(attr_node: ast.AST, immediate_if: ast.If, parent_map: Dict[ast.AST, ast.AST]) -> bool:
    """Return True if attr_node is located inside the first-evaluated subexpression
    of `immediate_if.test` (the left-most operand for BoolOp)."""
    if immediate_if is None:
        return False
    test_expr = immediate_if.test
    # If the test expression is a BoolOp with multiple operands, do NOT allow hoisting
    # from it (conservative choice: avoid changing short-circuit semantics).
    if isinstance(test_expr, ast.BoolOp) and len(test_expr.values) > 1:
        return False

    # Descend into left-most operands of BoolOp chains (single-operand BoolOp rare)
    left = test_expr
    while isinstance(left, ast.BoolOp) and left.values:
        left = left.values[0]

    # Walk upwards from attr_node and check whether we reach `left` before leaving the test_expr
    cur = attr_node
    while cur is not None and cur is not test_expr:
        if cur is left:
            return True
        cur = parent_map.get(cur)
    # If attr_node is exactly the test_expr or contained within it but not within left, return whether left is test_expr
    return left is test_expr and (attr_node is test_expr or parent_map.get(attr_node) is test_expr)


def get_func_args__(func_node: ast.AST) -> Dict[str, ast.arg]:
    """Return a mapping from argument name -> `ast.arg` for `func_node`.

    Includes posonly, args, kwonly, vararg and kwarg (where present).
    """
    if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    func = func_node
    args_map: Dict[str, ast.arg] = {}
    if getattr(func.args, 'posonlyargs', None):
        for a in func.args.posonlyargs:
            args_map[getattr(a, 'arg', None)] = a
    for a in func.args.args:
        args_map[getattr(a, 'arg', None)] = a
    for a in func.args.kwonlyargs:
        args_map[getattr(a, 'arg', None)] = a
    if func.args.vararg:
        args_map[getattr(func.args.vararg, 'arg', None)] = func.args.vararg
    if func.args.kwarg:
        args_map[getattr(func.args.kwarg, 'arg', None)] = func.args.kwarg
    return args_map


# deprecated: find_base_def removed in favor of find_base_def_no_map


#@profile
def find_stmt_list_path__(root: ast.AST, target: ast.AST) -> List[Tuple[List[ast.AST], int, ast.AST]]:
    """Return a list of (list_ref, index, parent_node) from the target
    statement up to the root. The list is ordered bottom->top.
    """
    path: List[Tuple[List[ast.AST], int, ast.AST]] = []

    #@profile
    def recurse(node: ast.AST) -> bool:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for i, el in enumerate(value):
                    if el is target:
                        path.append((value, i, node))
                        return True
                    if isinstance(el, ast.AST):
                        if recurse(el):
                            path.append((value, i, node))
                            return True
            elif isinstance(value, ast.AST):
                if recurse(value):
                    return True
        return False

    recurse(root)
    return path


#@profile
def find_prev_upward_no_map__(start_stmt: ast.AST, func: ast.AST, func_parent_map: Dict[ast.AST, ast.AST],
                              predicate) -> Optional[ast.AST]:
    """Strict forward-then-up search for nodes matching `predicate`.

    `root` is typically the module node so the search can climb out of the
    function to module-level statements (imports, classes).
    """
    # Determine the top-most node we should search up to: prefer the
    # containing function (FunctionDef/AsyncFunctionDef) of `start_stmt` if
    # available, otherwise fall back to the supplied `root`.
    cur = start_stmt
    func_root = None
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, getattr(ast, 'AsyncFunctionDef', type(None)))):
            func_root = cur
            break
        cur = func_parent_map.get(cur)

    top_root = func_root if func_root is not None else func

    path = find_stmt_list_path__(top_root, start_stmt)
    if not path:
        return None

    # build start_stmt ancestor set for visibility checks
    start_path = find_stmt_list_path__(top_root, start_stmt)
    start_anc_parents = {pnode for (_list, _idx, pnode) in start_path}

    # helper: determine immediate enclosing If and which branch ('test'/'body'/'orelse')
    #@profile
    def _immediate_if_branch(node: ast.AST) -> Tuple[Optional[ast.If], Optional[str]]:
        # Walk upwards using parent_map_local (O(depth) instead of O(N) search).
        cur = node
        while cur is not None and cur is not top_root:
            parent = func_parent_map.get(cur)
            if parent is None:
                break
            if isinstance(parent, ast.If):
                # determine which part of the If this `cur` belongs to
                if cur is parent.test:
                    return parent, 'test'
                # membership checks on lists are O(n) in the number of statements
                # but typically each list is short; this is still cheaper than
                # a full downward traversal from root for each node.
                if cur in parent.body:
                    return parent, 'body'
                if cur in parent.orelse:
                    return parent, 'orelse'
            cur = parent
        return None, None

    # Precompute the immediate-if branch for the start statement once
    # and cache results for candidate nodes to avoid repeated expensive
    # calls to `find_stmt_list_path__` inside the hot loop.
    start_if, start_branch = _immediate_if_branch(start_stmt)
    cand_if_cache: dict[ast.AST, Tuple[Optional[ast.If], Optional[str]]] = {}

    # path is bottom->top: iterate levels; for each statement list, scan
    # previous siblings and, if nothing matches, ascend to the next parent
    # list until `top_root` is reached.
    for list_ref, idx, parent_node in path:
        # scan previous siblings only
        for i in range(idx - 1, -1, -1):
            cand = list_ref[i]
            # If candidate and start_stmt are enclosed by the same immediate
            # If but in different branches, treat candidate as not visible.
            cand_if_branch = cand_if_cache.get(cand)
            if cand_if_branch is None:
                cand_if_branch = _immediate_if_branch(cand)
                cand_if_cache[cand] = cand_if_branch
            cand_if, cand_branch = cand_if_branch

            # If candidate is inside some If: ensure start_stmt is also
            # enclosed by the same If (i.e., the If is an ancestor of
            # start_stmt). Otherwise the candidate is not visible.
            if cand_if is not None:
                if cand_if not in start_anc_parents:
                    continue
                # If both are in the same immediate If but different
                # branches, skip as well.
                if cand_branch != start_branch:
                    continue

            res = predicate(cand)
            if res == 'STOP':
                return None
            if res:
                return cand
    return None


def loop_has_prior_definition__(func: ast.AST, loop_node: ast.AST, base_names: set[str]) -> bool:
    """Return True if a statement prior to `loop_node` defines any name in `base_names`.

    This checks preceding statements for definitions or occurrences of any
    name in `base_names`. Exceptions and missing-path cases are not guarded
    here — callers must pass valid arguments and handle any exceptions.
    """
    path = find_stmt_list_path__(func, loop_node)
    if not path:
        # Unable to locate loop_node within `func` — conservatively assume
        # there is no prior definition here.
        return False

    list_ref, loop_idx, _parent = path[0]
    prior_stmts = list_ref[: loop_idx]

    def _root_name(node: ast.AST) -> Optional[str]:
        cur = node
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        if isinstance(cur, ast.Name):
            return cur.id
        return None

    for stmt in reversed(prior_stmts):
        # check any base name using the consolidated definition helper
        for b in base_names:
            if _node_defines_base(stmt, b):
                return True

        # Fallback: scan for actual definitions (Assign/AnnAssign/For/Import/comprehension)
        # inside the statement rather than mere uses. This avoids false positives
        # where a base name only appears in a read context.
        for node in ast.walk(stmt):
            for b in base_names:
                if _node_defines_base(node, b):
                    return True
    # Additionally, scan the loop body itself for any definition/occurrence of
    # the base names (covers cases like `bars = ...` or `bars: T = ...` that
    # appear as the first statements inside the loop). This is a conservative
    # local check to detect definitions that occur within the loop prior to
    # uses when the caller only has the loop node available.
    body_stmts = getattr(loop_node, 'body', []) or []
    for stmt in body_stmts:
        # check top-level statement definitions
        for b in base_names:
            if _node_defines_base(stmt, b):
                return True
        # also scan nested nodes for real definitions (not mere uses)
        for node in ast.walk(stmt):
            for b in base_names:
                if _node_defines_base(node, b):
                    return True

    return False


def get_loop_assign_attribute_names__(target: ast.expr) -> list[str]:
    """Return the loop target expressions of interest.

    Historically this returned only the base `Name` (e.g. `self`) for
    an attribute target like `self.x`. For hoisting we need the full
    target text (e.g. `self.x`) so callers can detect loop-target
    conflicts precisely. Therefore:
    - For a `Name` target return the `Name` node.
    - For an `Attribute` target return the full `Attribute` node
      (preserving the chained attribute structure).
    - For `Tuple`/`List` targets recurse into elements.
    """
    roots: list[ast.expr] = []
    if isinstance(target, ast.Name):
        roots.append(target.id)
    elif isinstance(target, ast.Attribute):
        # Return the full attribute expression (e.g. `self.x`) rather than
        # just the base name. This preserves the left-hand chain for
        # accurate comparisons with hoisted names.
        roots.append(ast.unparse(target))
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            roots.extend(get_loop_assign_attribute_names__(elt))
    return roots


def find_base_def_no_map__(attr: ast.Attribute,
                           func: ast.AST,
                           func_parent_map: Dict[ast.AST, ast.AST],
                           func_args: Optional[dict[str, ast.arg]] = None) -> Optional[ast.AST]:
    """Find defining node for attribute base without using cached parent maps.

    Checks function parameters first, then scans previous siblings up to the
    containing function only. If not found within the function, returns None
    (meaning the base is at module scope).
    """
    # resolve root name/value
    root = attr
    while isinstance(root, ast.Attribute):
        root = root.value

    if isinstance(root, ast.Name):
        base_text = root.id
    else:
        base_text = ast.unparse(root)

    if not base_text:
        return None

    # 1) check function parameters (common case)
    arg = func_args.get(base_text, None)
    if arg is not None:
        return arg

    # 2) locate the containing statement and ensure it's within the function
    # scope. Use a local parent map to find the statement ancestor, then
    # verify the statement lies under `func_node`.
    parent_stmt = find_stmt_ancestor__(attr, func_parent_map)
    if parent_stmt is None:
        return None
    # ensure the parent statement is actually contained in the function
    parent_path = find_stmt_list_path__(func, parent_stmt)
    if not parent_path:
        return None

    def _pred(node: ast.AST) -> bool:
        return _node_defines_base(node, base_text)

    # search within function (previous siblings then up within function)
    for list_ref, idx, parent_node in parent_path:
        # also consider the parent_node itself (e.g., a For/AsyncFor defines loop targets)
        if _pred(parent_node):
            return parent_node
        for i in range(idx - 1, -1, -1):
            cand = list_ref[i]
            if _pred(cand):
                return cand
    return None


def get_with_conflicts__(func: ast.AST) -> Set[str]:
    """Find attribute chains that are both read and written within the same 'with' block."""
    conflicts = set()
    for node in ast.walk(func):
        if isinstance(node, ast.With):
            loads = set()
            stores = set()
            # Walk the with block body and items (context managers)
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute):
                    chain = attr_chain_text_ast__(sub)
                    if isinstance(sub.ctx, ast.Load):
                        loads.add(chain)
                    elif isinstance(sub.ctx, (ast.Store, ast.Del)):
                        stores.add(chain)
                elif isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Attribute):
                    chain = attr_chain_text_ast__(sub.target)
                    loads.add(chain)  # AugAssign reads then writes
                    stores.add(chain)
            conflicts.update(loads & stores)
    return conflicts


def get_subattr_for_chain__(full_attr: ast.Attribute, hoist_chains: list[str]) -> ast.Attribute:
    """Given a full Attribute node and a target_chain like 'it.some', return
    the Attribute node within `full_attr` that corresponds to that chain.
    """
    # collect attribute nodes from inner->outer
    stack: List[ast.Attribute] = []
    cur = full_attr
    while isinstance(cur, ast.Attribute):
        stack.append(cur)
        cur = cur.value
    rev = list(reversed(stack))  # left-to-right attr nodes
    # For a target_chain like 'it.some' we want the attribute node
    # corresponding to the last name in the chain ('some'). In `rev`
    # (left-to-right attr nodes) that is at index `len(parts) - 2`.
    idx = len(hoist_chains) - 2
    if idx < 0:
        idx = 0
    if 0 <= idx < len(rev):
        return rev[idx]
    # fallback to the outermost attribute
    return stack[0] if stack else full_attr


def get_attr_name_and_pos__(node: ast.AST) -> tuple[tuple[int, int], str]:
    """Return (attr_name, lineno, col_offset) for common AST shapes.

    Uses direct attribute access after isinstance checks (no getattr/hasattr).
    Raises AttributeError if expected lineno/col_offset are missing — let callers observe failures.
    """
    if isinstance(node, ast.Attribute):
        return (node.lineno, node.col_offset), node.attr
    if isinstance(node, ast.Name):
        return (node.lineno, node.col_offset), node.id
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            return (func.lineno, func.col_offset), func.attr
        if isinstance(func, ast.Name):
            return (func.lineno, func.col_offset), func.id
        # fallback: use call text and call node position
        return (node.lineno, node.col_offset), safe_unparse__(node)
    # fallback for other nodes
    return (node.lineno, node.col_offset), safe_unparse__(node)


def split_on_delims__(s: str, delims: List[str] | None = None) -> List[str]:
    """Split string `s` on given single-character delimiters and drop empty tokens.

    Default delimiters: ['.', '[', ']']
    Example: 'symbol_position_map[pos_contract.symbol].append' ->
             ['symbol_position_map', 'pos_contract', 'symbol', 'append']
    """
    import re
    if delims is None:
        delims = ['.', '[', ']']
    pattern = '[' + re.escape(''.join(delims)) + ']+'.replace(' ', '')
    return [tok for tok in re.split(pattern, s) if tok]


def split_base_ident__(name: str) -> list[str]:
    spls = name.split('[', 1)
    spl_0 = spls[0].split('.')[0]
    result = [spl_0]
    if len(spls) > 1:
        inner = spls[1]
        inner = inner.rsplit(']', 1)[0]
        if inner.isidentifier():
            result.append(inner)

    return result


def split_by_brackets__(s: str) -> List[str]:
    """Split `s` by bracketed segments, keeping bracketed parts intact.

    Example:
        'symbol_position_map[pos_contract.symbol].append' ->
        ['symbol_position_map', '[pos_contract.symbol]', '.append']

    Notes:
    - Keeps each top-level bracketed segment ("[...]") as a single token.
    - Keeps each top-level bracketed segment ("[...]" or "(...)") as a single token.
    - Handles nested brackets of `[]` and `()` correctly and preserves their
      contents as a single token. If a matching closer is not found the
      remainder of the string is treated as the bracket contents (same as
      previous behavior).
    """
    # do not perform defensive type/emptiness checks here; let callers
    # provide valid input and let any exceptions propagate.
    res: List[str] = []
    n = len(s)
    last = 0
    i = 0
    while i < n:
        ch = s[i]
        if ch in ('[', '('):
            # compute prefix (text before this bracket)
            prefix = s[last : i] if i > last else ''

            # determine matching closer and scan with a stack to handle nesting
            if ch == '[':
                expected = ']'
            else:
                expected = ')'
            stack = [expected]
            j = i + 1
            while j < n and stack:
                cj = s[j]
                if cj == '[':
                    stack.append(']')
                elif cj == '(':
                    stack.append(')')
                elif stack and cj == stack[-1]:
                    stack.pop()
                j += 1

            # j is one past the matching closer (or end of string)
            bracket_part = s[i : j]

            # If prefix ends with an identifier-like character or a closing
            # bracket/paren, merge prefix+bracket into a single token so
            # calls like `QDate(...).addMonths` keep `QDate(...)` together.
            if prefix and (prefix[-1].isalnum() or prefix[-1] in '_])'):
                res.append(prefix + bracket_part)
            else:
                if prefix:
                    res.append(prefix)
                res.append(bracket_part)

            last = j
            i = j
        else:
            i += 1

    # append remaining tail
    if last < n:
        tail = s[last :]
        if tail:
            res.append(tail)
    return res


def split_expr_token_to_items__(s: str) -> MultiItems[str, str]:
    tokens: list[str] = split_by_brackets__(s)
    result = MultiItems()
    for tok in tokens:
        is_bracket = tok.startswith('[') and tok.endswith(']')
        if is_bracket:
            name = tok[1 :-1]
            result.add(name, tok)
        else:
            if '(' in tok and ')' in tok:
                # function call: treat entire token as single item
                spls = [tok]
            else:
                spls = tok.split('.')
            for i, sub_sp in enumerate(spls):
                if sub_sp:
                    if i == 0:
                        result.add(sub_sp, sub_sp)
                    else:
                        result.add(sub_sp, '.' + sub_sp)

    return result
