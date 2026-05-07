"""Immutable calculator tool — evaluates a basic arithmetic expression."""
from __future__ import annotations

import ast
import operator
from typing import Any

from platform_core.tools import register_tool

NAME = "calculate"

SCHEMA = {
    "name": NAME,
    "description": (
        "Evaluate a basic arithmetic expression and return the numeric result "
        "as a string. Supports +, -, *, /, //, %, **, and parentheses over "
        "integer and float literals."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "An arithmetic expression, e.g. '2 + 3 * 4'.",
            },
        },
        "required": ["expression"],
    },
}


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def run(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    result = _eval(tree)
    return str(result)


register_tool(NAME, SCHEMA, run)
