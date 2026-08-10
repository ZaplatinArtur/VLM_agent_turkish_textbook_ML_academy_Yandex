"""Fail-closed calculator and SymPy gates for answer-blind experiments.

The module never calls ``eval`` or ``sympy.sympify`` on model text.  A bounded
Python AST is translated node by node into exact arithmetic.  SymPy is optional
at import time, but callers can require it before labelling a run as the
``calculator_sympy`` treatment.
"""

from __future__ import annotations

import ast
import math
import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from operator import mul
from typing import Any, Callable

from vlm_judge.normalization import (
    normalize_multiple_choice,
    normalize_text,
    parse_numeric,
)


@dataclass(frozen=True, slots=True)
class ProgramResult:
    ok: bool
    value: str | None = None
    error: str | None = None
    engine: str | None = None
    operation_count: int = 0
    numeric_literal_count: int = 0
    nontrivial: bool = False


@dataclass(frozen=True, slots=True)
class GateDecision:
    switch: bool
    audit_required: bool
    reasons: tuple[str, ...]


def sympy_available() -> bool:
    try:
        import sympy  # noqa: F401
    except ImportError:
        return False
    return True


def _product(values: list[Fraction]) -> Fraction:
    return reduce(mul, values, Fraction(1))


def _mean(values: list[Fraction]) -> Fraction:
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values, Fraction(0)) / len(values)


def _sqrt(value: Fraction) -> Fraction | float:
    if value < 0:
        raise ValueError("sqrt requires a non-negative value")
    numerator = math.isqrt(value.numerator)
    denominator = math.isqrt(value.denominator)
    if numerator * numerator == value.numerator and denominator * denominator == value.denominator:
        return Fraction(numerator, denominator)
    return math.sqrt(float(value))


def _integer(value: Fraction) -> int:
    if not isinstance(value, Fraction) or value.denominator != 1:
        raise ValueError("integer argument required")
    return int(value)


_MAX_INTEGER_ARGUMENT = 10**9
_MAX_COMBINATORIC_N = 1000
_MAX_INTEGER_RESULT_BITS = 4096


def _bounded_integer(value: int) -> int:
    if abs(value) > _MAX_INTEGER_ARGUMENT:
        raise ValueError("integer argument is too large")
    return value


def _bounded_integer_result(value: int) -> int:
    if abs(value).bit_length() > _MAX_INTEGER_RESULT_BITS:
        raise ValueError("integer result is too large")
    return value


def _power_size_ok(numerator_bits: int, denominator_bits: int, exponent: int) -> bool:
    multiplier = max(1, abs(exponent))
    return (
        numerator_bits * multiplier <= _MAX_INTEGER_RESULT_BITS
        and denominator_bits * multiplier <= _MAX_INTEGER_RESULT_BITS
    )


def _bounded_gcd_int(*values: int) -> int:
    if not 1 <= len(values) <= 16:
        raise ValueError("gcd requires 1..16 arguments")
    return _bounded_integer_result(math.gcd(*(_bounded_integer(value) for value in values)))


def _bounded_lcm_int(*values: int) -> int:
    if not 1 <= len(values) <= 16:
        raise ValueError("lcm requires 1..16 arguments")
    return _bounded_integer_result(math.lcm(*(_bounded_integer(value) for value in values)))


def _bounded_comb_int(n: int, k: int) -> int:
    if not 0 <= n <= _MAX_COMBINATORIC_N or not 0 <= k <= n:
        raise ValueError("comb arguments are outside the safe range")
    return _bounded_integer_result(math.comb(n, k))


def _bounded_perm_int(n: int, k: int | None = None) -> int:
    if not 0 <= n <= _MAX_COMBINATORIC_N:
        raise ValueError("perm n is outside the safe range")
    if k is not None and not 0 <= k <= n:
        raise ValueError("perm k is outside the safe range")
    return _bounded_integer_result(math.perm(n, k))


_EXACT_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "len": len,
    "min": min,
    "max": max,
    "sum": lambda values: sum(values, Fraction(0)),
    "product": _product,
    "mean": _mean,
    "sqrt": _sqrt,
    "percent": lambda value, rate: value * rate / 100,
    "gcd": lambda *values: _bounded_gcd_int(*(_integer(value) for value in values)),
    "lcm": lambda *values: _bounded_lcm_int(*(_integer(value) for value in values)),
    "comb": lambda n, k: _bounded_comb_int(_integer(n), _integer(k)),
    "perm": lambda n, k=None: _bounded_perm_int(
        _integer(n), None if k is None else _integer(k)
    ),
}


class _ExactEvaluator:
    def __init__(self, *, max_nodes: int) -> None:
        self.max_nodes = max_nodes
        self.visited = 0

    def visit(self, node: ast.AST) -> Any:
        self.visited += 1
        if self.visited > self.max_nodes:
            raise ValueError("program is too large")
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise ValueError(f"unsupported syntax: {type(node).__name__}")
        return method(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Fraction:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are allowed")
        if isinstance(node.value, int):
            if abs(node.value) > 10**12:
                raise ValueError("integer constant is too large")
            return Fraction(node.value)
        if not math.isfinite(node.value):
            raise ValueError("numeric constant must be finite")
        return Fraction(str(node.value))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("unsupported unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            if not isinstance(right, Fraction) or right.denominator != 1:
                raise ValueError("power exponent must be an integer")
            exponent = int(right)
            if abs(exponent) > 12:
                raise ValueError("power exponent is too large")
            if isinstance(left, Fraction) and not _power_size_ok(
                abs(left.numerator).bit_length(),
                left.denominator.bit_length(),
                exponent,
            ):
                raise ValueError("power result is too large")
            return left**exponent
        raise ValueError("unsupported binary operator")

    def visit_List(self, node: ast.List) -> list[Any]:
        if len(node.elts) > 40:
            raise ValueError("list is too large")
        return [self.visit(value) for value in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> tuple[Any, ...]:
        if len(node.elts) > 40:
            raise ValueError("tuple is too large")
        return tuple(self.visit(value) for value in node.elts)

    def visit_Call(self, node: ast.Call) -> Any:
        if node.keywords or not isinstance(node.func, ast.Name):
            raise ValueError("only direct positional function calls are allowed")
        function = _EXACT_FUNCTIONS.get(node.func.id)
        if function is None:
            raise ValueError(f"function is not allowed: {node.func.id}")
        return function(*(self.visit(argument) for argument in node.args))


def _format_exact(value: Any) -> str:
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite result")
        return f"{value:.12g}"
    raise ValueError(f"program must return one scalar, got {type(value).__name__}")


def _clean_program(source: str) -> str:
    cleaned = str(source or "").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1]).strip()
    if cleaned.casefold().startswith("python\n"):
        cleaned = cleaned[7:].strip()
    if not cleaned:
        raise ValueError("empty program")
    if len(cleaned) > 500:
        raise ValueError("program is too long")
    return cleaned


def _program_expression(source: str) -> tuple[ast.Expression, int, int]:
    module = ast.parse(_clean_program(source), mode="exec")
    if len(module.body) != 1:
        raise ValueError("program must contain exactly one statement")
    statement = module.body[0]
    if isinstance(statement, ast.Expr):
        expression = ast.Expression(statement.value)
    elif (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "result"
    ):
        expression = ast.Expression(statement.value)
    else:
        raise ValueError("use one expression or `result = <expression>`")
    operation_count = sum(
        isinstance(node, (ast.BinOp, ast.Call)) for node in ast.walk(expression)
    )
    numeric_literal_count = sum(
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and isinstance(node.value, (int, float))
        for node in ast.walk(expression)
    )
    return expression, operation_count, numeric_literal_count


class _SympyEvaluator:
    """Translate a bounded AST into SymPy without parsing model text in SymPy."""

    def __init__(
        self,
        *,
        max_nodes: int,
        allow_symbols: bool,
        max_symbols: int = 8,
    ) -> None:
        self.max_nodes = max_nodes
        self.allow_symbols = allow_symbols
        self.max_symbols = max_symbols
        self.visited = 0
        self.symbols: dict[str, Any] = {}

    def visit(self, node: ast.AST) -> Any:
        self.visited += 1
        if self.visited > self.max_nodes:
            raise ValueError("symbolic expression is too large")
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise ValueError(f"unsupported symbolic syntax: {type(node).__name__}")
        return method(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        from sympy import Rational

        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric symbolic constants are allowed")
        if isinstance(node.value, int):
            if abs(node.value) > 10**9:
                raise ValueError("symbolic integer is too large")
            return Rational(node.value)
        if not math.isfinite(node.value):
            raise ValueError("symbolic float must be finite")
        return Rational(str(node.value))

    def visit_Name(self, node: ast.Name) -> Any:
        from sympy import Symbol, pi

        if node.id == "pi":
            return pi
        if not self.allow_symbols:
            raise ValueError(f"free name is not allowed in a calculator program: {node.id}")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,7}", node.id):
            raise ValueError("invalid symbolic name")
        if node.id not in self.symbols:
            if len(self.symbols) >= self.max_symbols:
                raise ValueError("too many symbolic names")
            self.symbols[node.id] = Symbol(node.id, real=True)
        return self.symbols[node.id]

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("unsupported symbolic unary operator")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        from sympy import Mod, floor

        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return floor(left / right)
        if isinstance(node.op, ast.Mod):
            return Mod(left, right)
        if isinstance(node.op, ast.Pow):
            if not getattr(right, "is_Integer", False):
                raise ValueError("symbolic exponent must be an integer")
            exponent = int(right)
            if abs(exponent) > 12:
                raise ValueError("symbolic exponent is too large")
            if getattr(left, "is_Rational", False):
                numerator, denominator = left.as_numer_denom()
                if not _power_size_ok(
                    abs(int(numerator)).bit_length(),
                    abs(int(denominator)).bit_length(),
                    exponent,
                ):
                    raise ValueError("symbolic power result is too large")
            return left**exponent
        raise ValueError("unsupported symbolic binary operator")

    def visit_List(self, node: ast.List) -> list[Any]:
        if len(node.elts) > 40:
            raise ValueError("symbolic list is too large")
        return [self.visit(value) for value in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> tuple[Any, ...]:
        if len(node.elts) > 40:
            raise ValueError("symbolic tuple is too large")
        return tuple(self.visit(value) for value in node.elts)

    def visit_Call(self, node: ast.Call) -> Any:
        from sympy import Abs, Max, Min, Mul, Rational, sqrt

        if node.keywords or not isinstance(node.func, ast.Name):
            raise ValueError("only direct positional symbolic calls are allowed")
        name = node.func.id
        arguments = [self.visit(argument) for argument in node.args]
        if name == "abs" and len(arguments) == 1:
            return Abs(arguments[0])
        if name == "sqrt" and len(arguments) == 1:
            return sqrt(arguments[0])
        if name == "min" and arguments:
            return Min(*arguments)
        if name == "max" and arguments:
            return Max(*arguments)
        if name in {"sum", "product", "mean"} and len(arguments) == 1:
            values = list(arguments[0])
            if not values and name == "mean":
                raise ValueError("mean requires at least one value")
            if name == "sum":
                return sum(values, Rational(0))
            if name == "product":
                return Mul(*values)
            return sum(values, Rational(0)) / len(values)
        if name == "len" and len(arguments) == 1:
            return Rational(len(arguments[0]))
        if name == "percent" and len(arguments) == 2:
            return arguments[0] * arguments[1] / 100
        if name in {"gcd", "lcm", "comb", "perm"}:
            integers: list[int] = []
            for value in arguments:
                if not getattr(value, "is_Integer", False):
                    raise ValueError(f"{name} requires integer arguments")
                integers.append(int(value))
            if name == "gcd" and integers:
                return Rational(_bounded_gcd_int(*integers))
            if name == "lcm" and integers:
                return Rational(_bounded_lcm_int(*integers))
            if name == "comb" and len(integers) == 2:
                return Rational(_bounded_comb_int(integers[0], integers[1]))
            if name == "perm" and len(integers) in {1, 2}:
                return Rational(
                    _bounded_perm_int(
                        integers[0],
                        None if len(integers) == 1 else integers[1],
                    )
                )
            raise ValueError(f"invalid arguments for {name}")
        raise ValueError(f"symbolic function is not allowed: {name}")


def execute_program(source: str, *, max_nodes: int = 96) -> ProgramResult:
    """Execute one bounded calculator expression and return a canonical scalar."""

    operation_count = numeric_literal_count = 0
    try:
        expression, operation_count, numeric_literal_count = _program_expression(source)
        if sum(1 for _ in ast.walk(expression)) > max_nodes:
            raise ValueError("program is too large")
        nontrivial = operation_count > 0
        if sympy_available():
            from sympy import cancel, simplify

            value = _SympyEvaluator(
                max_nodes=max_nodes,
                allow_symbols=False,
            ).visit(expression)
            if isinstance(value, (list, tuple)):
                raise ValueError("program must return one scalar")
            if getattr(value, "free_symbols", set()):
                raise ValueError("calculator result has free symbols")
            value = cancel(simplify(value))
            if value.is_finite is not True:
                raise ValueError("calculator result is not finite")
            if getattr(value, "is_Rational", False):
                numerator, denominator = value.as_numer_denom()
                if max(
                    abs(int(numerator)).bit_length(),
                    abs(int(denominator)).bit_length(),
                ) > _MAX_INTEGER_RESULT_BITS:
                    raise ValueError("calculator result is too large")
            return ProgramResult(
                True,
                value=str(value),
                engine="sympy_ast",
                operation_count=operation_count,
                numeric_literal_count=numeric_literal_count,
                nontrivial=nontrivial,
            )
        value = _ExactEvaluator(max_nodes=max_nodes).visit(expression)
        return ProgramResult(
            True,
            value=_format_exact(value),
            engine="fraction_ast",
            operation_count=operation_count,
            numeric_literal_count=numeric_literal_count,
            nontrivial=nontrivial,
        )
    except (ArithmeticError, ImportError, SyntaxError, TypeError, ValueError) as exc:
        return ProgramResult(
            False,
            error=f"{type(exc).__name__}: {exc}",
            operation_count=operation_count,
            numeric_literal_count=numeric_literal_count,
            nontrivial=operation_count > 0,
        )


_LATEX_FRACTION = re.compile(r"\\frac\s*\{([^{}]{1,80})\}\s*\{([^{}]{1,80})\}")
_DEGREE_ATOM = re.compile(
    r"(?P<atom>(?:\d+(?:\.\d+)?|\([^()]{1,80}\)))\s*(?:\u00b0|degrees?|derece)",
    re.IGNORECASE,
)
_MATH_TOKEN = re.compile(
    r"\*\*|(?:\d+(?:\.\d+)?|\.\d+)|(?:[A-Za-z_][A-Za-z_0-9]*)|[()+\-*/]"
)
_ALLOWED_MATH_SOURCE = re.compile(r"^[0-9A-Za-z_+\-*/^().\s]*$")


def _normalize_symbolic_source(source: str) -> str:
    value = unicodedata.normalize("NFKC", str(source or "")).strip()
    if not value or len(value) > 240:
        raise ValueError("symbolic expression is empty or too long")
    value = (
        value.replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00d7", "*")
        .replace("\u00b7", "*")
        .replace("\u00f7", "/")
        .replace("\\pi", "pi")
        .replace("\u03c0", "pi")
        .replace("\\left", "")
        .replace("\\right", "")
        .replace("^", "**")
    )
    for _ in range(3):
        replaced = _LATEX_FRACTION.sub(r"((\1)/(\2))", value)
        if replaced == value:
            break
        value = replaced
    value = re.sub(r"(?<=\d),(?=\d)", ".", value)
    value = _DEGREE_ATOM.sub(r"((\g<atom>)*pi/180)", value)
    value = re.sub(r"(?i)\s*(?:radians?|radyan)\b", "", value)
    value = value.replace("$", "").strip()
    if not _ALLOWED_MATH_SOURCE.fullmatch(value):
        raise ValueError("symbolic expression contains unsupported characters")
    tokens = _MATH_TOKEN.findall(value)
    if not tokens or "".join(tokens) != re.sub(r"\s+", "", value):
        raise ValueError("symbolic expression contains an unsupported token")
    output: list[str] = []
    previous: str | None = None
    for token in tokens:
        previous_is_atom = bool(
            previous
            and (
                previous == ")"
                or previous[0].isdigit()
                or previous[0] == "."
                or previous[0].isalpha()
                or previous[0] == "_"
            )
        )
        current_starts_atom = bool(
            token == "("
            or token[0].isdigit()
            or token[0] == "."
            or token[0].isalpha()
            or token[0] == "_"
        )
        is_allowed_call = bool(
            previous
            and token == "("
            and previous
            in {
                "abs",
                "len",
                "min",
                "max",
                "sum",
                "product",
                "mean",
                "sqrt",
                "percent",
                "gcd",
                "lcm",
                "comb",
                "perm",
            }
        )
        if previous_is_atom and current_starts_atom and not is_allowed_call:
            output.append("*")
        output.append(token)
        previous = token
    return "".join(output)


def _symbolic_expression(source: str, *, allow_symbols: bool) -> Any:
    tree = ast.parse(_normalize_symbolic_source(source), mode="eval")
    expression = _SympyEvaluator(
        max_nodes=80,
        allow_symbols=allow_symbols,
    ).visit(tree)
    if int(expression.count_ops()) > 120:
        raise ValueError("symbolic result is too complex")
    return expression


def safe_math_equivalent(left: str, right: str) -> bool:
    """Compare bounded numeric or algebraic expressions without code execution."""

    left_text = str(left)
    right_text = str(right)
    has_equation = left_text.count("=") == 1 or right_text.count("=") == 1
    if not has_equation:
        left_number = parse_numeric(left_text)
        right_number = parse_numeric(right_text)
        if left_number is not None and right_number is not None:
            return left_number == right_number
    if not sympy_available():
        return False
    try:
        from sympy import cancel, simplify

        def equation(value: str) -> Any:
            if value.count("=") == 1:
                lhs, rhs = value.split("=", maxsplit=1)
                return _symbolic_expression(lhs, allow_symbols=True) - _symbolic_expression(
                    rhs, allow_symbols=True
                )
            return _symbolic_expression(value, allow_symbols=True)

        left_expression = equation(left_text)
        right_expression = equation(right_text)
        difference = cancel(left_expression - right_expression)
        if difference == 0 or simplify(difference) == 0:
            return True
        if left_text.count("=") == 1 and right_text.count("=") == 1:
            if right_expression == 0:
                return left_expression == 0
            ratio = cancel(left_expression / right_expression)
            return bool(ratio != 0 and not ratio.free_symbols)
        return False
    except (ArithmeticError, ImportError, SyntaxError, TypeError, ValueError):
        return False


def answer_equivalent(left: Any, right: Any, answer_type: str) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return False
    if answer_type == "choice":
        left_choice = normalize_multiple_choice(left_text)
        right_choice = normalize_multiple_choice(right_text)
        return left_choice is not None and left_choice == right_choice
    if answer_type == "numeric":
        return safe_math_equivalent(left_text, right_text)
    if normalize_text(left_text) == normalize_text(right_text):
        return True
    return safe_math_equivalent(left_text, right_text)


def answer_parseable(value: Any, answer_type: str) -> bool:
    text = str(value or "").strip()
    if not text or len(text) > 120:
        return False
    if answer_type == "choice":
        return normalize_multiple_choice(text) is not None
    if answer_type == "numeric":
        if parse_numeric(text) is not None:
            return True
        if not sympy_available():
            return False
        try:
            expression = _symbolic_expression(text, allow_symbols=False)
            return not expression.free_symbols and expression.is_finite is True
        except (ArithmeticError, ImportError, SyntaxError, TypeError, ValueError):
            return False
    return bool(normalize_text(text))


def decide_calculator_switch(
    *,
    baseline_answer: Any,
    answer_type: str,
    draft: dict[str, Any],
    program: ProgramResult,
    audit: dict[str, Any] | None = None,
) -> GateDecision:
    """Require deterministic tool evidence and two independent answer reads.

    The first draft must be produced without seeing the frozen candidate.  The
    optional audit is the only stage that compares the independent result with
    that candidate.  Every malformed, low-confidence or contradictory state
    preserves the frozen answer.
    """

    reasons: list[str] = []
    if draft.get("applicable") is not True:
        reasons.append("not_applicable")
    if str(draft.get("confidence") or "") != "high":
        reasons.append("draft_not_high_confidence")
    if not program.ok:
        reasons.append("program_failed")
    elif not program.nontrivial:
        reasons.append("trivial_program")
    predicted_value = str(draft.get("predicted_program_value") or "").strip()
    if program.ok and (
        not predicted_value
        or not safe_math_equivalent(str(program.value or ""), predicted_value)
    ):
        reasons.append("predicted_value_mismatch")
    for check in ("unit_check", "constraint_check"):
        check_value = str(draft.get(check) or "")
        if check_value not in {"pass", "not_applicable"}:
            reasons.append(f"draft_{check}_failed")
    independent_answer = str(draft.get("independent_answer") or "").strip()
    if not answer_parseable(independent_answer, answer_type):
        reasons.append("independent_answer_unparseable")
    elif answer_type == "numeric":
        if str(draft.get("problem_kind") or "") == "equation_substitution":
            # Equation verification intentionally returns a zero residual rather
            # than the candidate root itself.
            if not safe_math_equivalent(str(program.value or ""), "0"):
                reasons.append("substitution_residual_nonzero")
        elif not safe_math_equivalent(
            independent_answer,
            str(program.value or ""),
        ):
            # For every other numeric response the calculator value must be the
            # claimed answer. A second model cannot overrule this deterministic
            # contradiction by calling its own audit tool-consistent.
            reasons.append("independent_program_disagree")
    if reasons:
        return GateDecision(False, False, tuple(dict.fromkeys(reasons)))
    if answer_equivalent(independent_answer, baseline_answer, answer_type):
        return GateDecision(False, False, ("independent_agrees_with_baseline",))
    if audit is None:
        return GateDecision(False, True, ("audit_required",))

    if audit.get("switch_recommended") is not True:
        reasons.append("audit_rejected_switch")
    if audit.get("tool_consistent") is not True:
        reasons.append("audit_tool_inconsistent")
    if audit.get("question_consistent") is not True:
        reasons.append("audit_question_inconsistent")
    if str(audit.get("confidence") or "") != "high":
        reasons.append("audit_not_high_confidence")
    for check in ("unit_check", "constraint_check"):
        check_value = str(audit.get(check) or "")
        if check_value not in {"pass", "not_applicable"}:
            reasons.append(f"audit_{check}_failed")
    final_answer = str(audit.get("final_answer") or "").strip()
    if not answer_parseable(final_answer, answer_type):
        reasons.append("audit_answer_unparseable")
    elif not answer_equivalent(final_answer, independent_answer, answer_type):
        reasons.append("independent_audit_disagree")
    if reasons:
        return GateDecision(False, False, tuple(dict.fromkeys(reasons)))
    return GateDecision(True, False, ())
