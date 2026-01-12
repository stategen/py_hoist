from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional

from py_hoist.multi_items import MultiItems


@dataclass(slots = True)
class CodeAssignStmtInfo:
    assign_in_loop: bool = False  # 在循环内赋值
    stmt_lineno: int = None
    assign_node: ast.Assign = None
    assign_names: set[str] = field(default_factory = set)


class AttrUsedInType(Enum):

    def __new__(cls, _value: int, can_hoist: bool = True):
        obj = object.__new__(cls)
        obj._value_ = _value
        return obj

    def __init__(self, _value: int, can_hoist: bool = True):
        self.can_hoist = can_hoist

    IN_FIRST_BOOLOP = (1, True)
    IN_OTHER_BOOLOP = (2, False)
    IN_IF_TEST_FIRST = (3, True)
    IN_IF_TEST_OTHER = (4, False)
    IN_ELIF_TEST_FIRST = (5, True)
    IN_ELIF_TEST_OTHER = (6, False)
    IN_NORMAL = (7, True)


@dataclass(slots = True)
class AttrUsageInfo:
    attr_node: ast.Attribute = None  # 使用的属性节点
    base_def: Optional[cst.AST] = None  # 属性的基础定义节点
    base_text: str = None  # 属性的基础定义节点
    calc_var_name: str = None  # 属性的基础定义节点
    used_in_loop: bool = False
    base_loop_re_assigned: bool = False  # 是否有基于该属性的重新赋值（如 base_deep =base.deep ,但base在循环中被重新赋值 base =Base()  ）
    used_type: AttrUsedInType = AttrUsedInType.IN_NORMAL
    chain_text: str = None  # 赋值语句的源代码文本
    chain_parts: MultiItems[str, str] = None  # 赋值语句的源代码文本
    chain_2_text: str = None  # 赋值语句的源代码文本 （仅前两段） self.xxxx, map[xxx]

    has_more3_chains: bool = False  # 是否包含多层链
    hoist_sub_attr_node: Optional[ast.Attribute] = None  # 该赋值语句所绑定的子属性节点
    under_with_conflict: bool = False  # 是否在 with 语句的冲突变量上下文中使用
    assignStmtInfo: 'AssignStmtInfo' = None
    in_index: bool = None  #是否在索引中使用


@dataclass(slots = True)
class AssignStmtInfo:
    assign_node: ast.Assign = None  # 赋值语句节点
    bar_name: str = None  #变量名 gooos_main_holder
    chain_text: str = None  # 赋值语句的源代码文本
    chain_2_value: str = None  #提升的属性链 gooos_main.holder
    gen_by_orig_code: bool = False  # 是否原本就在代码中（否则是新创建的）
    attr_infos: List[AttrUsageInfo] = field(default_factory = list)  # 该赋值语句所绑定的属性列表
    create_outside_loop: bool = False  # 是否需要提升到循环外部
    use_by_with_conflict: bool = False  # 是否被 with 语句的冲突变量使用

    def _sub_can_hoist(self) -> bool:
        # if self.assign_name == 'UsePriceType_default':
        #     print(f'self.assign_name')

        if self.use_by_with_conflict:
            return False  #with 不做

        if self.create_outside_loop:
            return True  #循环内使用在循环外创建，要插入

        valid_cnt = 0
        attr_in_index = False
        find_can_hoist = False
        for attr_info in self.attr_infos:
            if attr_info.in_index:
                attr_in_index = True
                break

            if attr_info.used_type.can_hoist:
                find_can_hoist = True
                valid_cnt += 1

            elif find_can_hoist:
                valid_cnt += 1

        if attr_in_index or valid_cnt > 1:
            return True  #多次使用要插入

        return False

    def can_replaced(self) -> bool:
        """判断属性语句是否要被替换。
        """
        if self.gen_by_orig_code:
            return True

        return self._sub_can_hoist()

    def need_insert(self) -> bool:
        """判断赋值语话是否要被插入
        """
        if self.gen_by_orig_code:
            return False  #已有，不要插入

        return self._sub_can_hoist()


def transform_code_tree(source: str, pos_map: Dict[(int, int, int), AttrUsageInfo],
                        insert_map: Dict[Tuple[int, int, str], AssignStmtInfo]) -> tuple[bool, str]:
    sorted_pop_items: list[tuple[(int, int, int), AttrUsageInfo]] = sorted(pos_map.items(),
                                                                           key = lambda item: (item[0][0], item[0][1]),
                                                                           reverse = True)
    insert_items: list[tuple[Tuple[int, int, str], AssignStmtInfo]] = sorted(insert_map.items(),
                                                                             key = lambda item: item[0],
                                                                             reverse = True)

    ################ --- inline: 文本替换（使用 pos_map）
    # Perform replacements directly on the per-line list to avoid absolute-index drift.
    lines_src = source.splitlines(keepends = True)

    has_any_replce_or_insert = False

    for (lineno, col, end_col), attr_usage_info in sorted_pop_items:

        # if assignStmtInfo.chain_text == 'symbol_position_map[pos_contract_symbol].append':
        #     print(f"chain_text: {assignStmtInfo.chain_text}")

        # lineno is 1-based; map to 0-based index
        idx = lineno - 1
        # intentionally do NOT perform bounds/clamp checks here; allow errors to surface
        line = lines_src[idx]

        # Treat the line as a byte-mapped single-character string so AST byte offsets
        # align with slicing. This avoids misalignment on lines containing multibyte
        # characters such as Chinese text.
        byte_line = line.encode('utf-8').decode('latin-1')

        # use offsets as provided by AST (do not clamp or catch errors)
        start = col
        end = end_col

        snippet = byte_line[start : end]
        var_name_and_left_attr = attr_usage_info.assignStmtInfo.bar_name + ''.join(attr_usage_info.chain_parts.values()[2 :])
        replacement = var_name_and_left_attr.encode('utf-8').decode('latin-1')
        new_byte_line = byte_line[: start] + replacement + byte_line[end :]
        new_line = new_byte_line.encode('latin-1').decode('utf-8')
        lines_src[idx] = new_line
        has_any_replce_or_insert = True

        # print(f"[HOIST-DEBUG] pos_map entry: {(lineno, col, end_col)} :{snippet!r}=>{var_name_and_left_attr!r} \n{lines_src[idx]}")

    for (lineno, col, assign_name), assignStmtInfo in insert_items:
        idx = lineno - 1
        # Ensure the inserted statement is indented to match the owner statement's column
        indent = (' ' * col) if (col is not None and col > 0) else ''
        assign_line = f"{assignStmtInfo.bar_name} = {assignStmtInfo.chain_2_value}"

        new_line = indent + assign_line + '\n'
        lines_src.insert(idx, new_line)
        has_any_replce_or_insert = True

    if not has_any_replce_or_insert:
        new_source = source
    else:

        new_source = ''.join(lines_src)

    return has_any_replce_or_insert, new_source
