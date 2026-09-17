"""
Calculator Tool
================
Инструмент для математических вычислений
"""

import math
from typing import Union


def calculate(expression: str) -> Union[float, str]:
    """
    Вычисление математического выражения
    
    Args:
        expression: Математическое выражение (например: "(15 * 23) + 42")
    
    Returns:
        Результат вычисления или сообщение об ошибке
    """
    
    # Безопасные функции для eval
    safe_dict = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'pi': math.pi,
        'e': math.e
    }
    
    try:
        # Проверка на опасные символы
        dangerous_chars = ['import', 'exec', 'eval', '__', 'lambda']
        if any(char in expression for char in dangerous_chars):
            return "Ошибка: недопустимые символы в выражении"
        
        # Вычисление
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"Результат: {expression} = {result}"
        
    except SyntaxError:
        return "Ошибка: неверный синтаксис выражения"
    except ZeroDivisionError:
        return "Ошибка: деление на ноль"
    except Exception as e:
        return f"Ошибка вычисления: {str(e)}"


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
