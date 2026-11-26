"""Bounded arithmetic interpreter. No Python eval or executable expressions."""
import ast
import math
import operator


_FUNCTIONS = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum,
              "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
              "tan": math.tan, "log": math.log, "log10": math.log10}
_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e100:
        raise ValueError("результат вне допустимого числового диапазона")
    return value


def _power(base, exponent):
    _number(base); _number(exponent)
    if abs(exponent) > 100:
        raise ValueError("модуль степени не должен превышать 100")
    return _number(base ** exponent)


def _evaluate(node, depth=0):
    if depth > 20:
        raise ValueError("слишком глубокое выражение")
    visit = lambda child: _evaluate(child, depth + 1)
    if isinstance(node, ast.Constant):
        return _number(node.value)
    if isinstance(node, ast.Name) and node.id in ("pi", "e"):
        return getattr(math, node.id)
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        value = _number(visit(node.operand))
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = _number(visit(node.left)), _number(visit(node.right))
        if isinstance(node.op, ast.Pow):
            return _power(left, right)
        if type(node.op) in _OPERATORS:
            return _number(_OPERATORS[type(node.op)](left, right))
    if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) <= 32:
        return [_number(visit(child)) for child in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        if len(node.args) > 32:
            raise ValueError("слишком много аргументов")
        args = [visit(child) for child in node.args]
        if name == "pow" and len(args) == 2:
            return _power(*args)
        if name in _FUNCTIONS:
            if name not in ("sum", "min", "max"):
                for arg in args:
                    _number(arg)
            if name == "round" and len(args) == 2 and abs(args[1]) > 100:
                raise ValueError("слишком большая точность округления")
            return _number(_FUNCTIONS[name](*args))
    raise ValueError("недопустимая конструкция выражения")


def calculate(expression: str) -> str:
    """Return a formatted finite arithmetic result, or a controlled error."""
    try:
        if not isinstance(expression, str) or not expression.strip() or len(expression) > 512:
            raise ValueError("выражение должно содержать от 1 до 512 символов")
        tree = ast.parse(expression.strip(), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            raise ValueError("слишком сложное выражение")
        result = _number(_evaluate(tree.body))
        return f"Результат: {expression} = {result}"
    except ZeroDivisionError:
        return "Ошибка: деление на ноль"
    except (SyntaxError, TypeError, ValueError, OverflowError, RecursionError) as error:
        return f"Ошибка вычисления: {error}"

def solve_equation(equation: str) -> str:
    """
    Решение уравнения (упрощенная версия)
    
    Args:
        equation: Уравнение (например: "2*x + 5 = 15")
    
    Returns:
        str: Решение уравнения
    """
    
    # В реальной реализации можно использовать sympy
    return f"""
    🧮 Решение уравнения: {equation}
    
    Для полноценного решения уравнений установите sympy:
    pip install sympy
    
    Пример использования:
    from sympy import symbols, solve
    x = symbols('x')
    solution = solve(equation, x)
    """


if __name__ == "__main__":
    # Тестирование
    test_cases = [
        "(15 * 23) + 42",
        "sqrt(144)",
        "sin(pi / 2)",
        "log10(1000)",
        "2 ** 10",
        "100 / 0"  # Ошибка
    ]
    
    for expr in test_cases:
        result = calculate(expr)
        print(f"{expr} => {result}\n")
