"""
Chain-of-Thought Prompting
===========================
Техника пошагового рассуждения для сложных задач.

Этот пример демонстрирует:
- Basic CoT prompting
- Zero-shot CoT
- Self-consistency
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def basic_chain_of_thought(problem: str) -> str:
    """
    Basic Chain-of-Thought - явное указание на пошаговое рассуждение
    """
    prompt = ChatPromptTemplate.from_template(
        """Решите следующую задачу, рассуждая шаг за шагом.

Задача: {problem}

Решение:
Шаг 1: Сначала определите...
Шаг 2: Затем проанализируйте...
Шаг 3: После этого вычислите...
Шаг 4: Наконец, сделайте вывод...

Ответ:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"problem": problem})


def zero_shot_cot(problem: str) -> str:
    """
    Zero-shot CoT - использование магической фразы "Let's think step by step"
    """
    prompt = ChatPromptTemplate.from_template(
        "{problem}\n\nLet's think step by step."
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"problem": problem})


def self_consistency(problem: str, num_paths: int = 3) -> str:
    """
    Self-consistency - генерация нескольких путей рассуждения и выбор наиболее частого ответа
    """
    prompt = ChatPromptTemplate.from_template(
        """Решите задачу, рассуждая шаг за шагом:

{problem}

Предоставьте подробное решение:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    # Генерируем несколько решений
    solutions = []
    for i in range(num_paths):
        solution = chain.invoke({"problem": problem})
        solutions.append(solution)
        print(f"\nПуть рассуждения {i+1}:")
        print(solution)
    
    # В реальном приложении здесь был бы анализ и выбор наиболее согласованного ответа
    return f"\nСгенерировано {num_paths} путей рассуждения. В реальном приложении здесь был бы анализ согласованности."


def tree_of_thought(problem: str) -> str:
    """
    Tree of Thoughts - исследование нескольких путей рассуждения
    """
    prompt = ChatPromptTemplate.from_template(
        """Представьте трех разных экспертов, решающих эту задачу.

Задача: {problem}

Каждый эксперт:
1. Делает первый шаг рассуждения
2. Оценивает его правильность (0-100%)
3. Если <80%, пробует другой подход
4. Продолжает, пока не придет к решению

Покажите рассуждения всех экспертов и итоговый ответ:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"problem": problem})


if __name__ == "__main__":
    test_problem = """
    В магазине было 45 яблок. Продали 12 яблок утром и 18 яблок днем.
    Затем привезли еще 30 яблок. Сколько яблок стало в магазине?
    """
    
    print("=" * 60)
    print("BASIC CHAIN-OF-THOUGHT")
    print("=" * 60)
    print(basic_chain_of_thought(test_problem))
    print()
    
    print("=" * 60)
    print("ZERO-SHOT COT")
    print("=" * 60)
    print(zero_shot_cot(test_problem))
    print()
    
    print("=" * 60)
    print("SELF-CONSISTENCY")
    print("=" * 60)
    print(self_consistency(test_problem, num_paths=3))
    print()
    
    print("=" * 60)
    print("TREE OF THOUGHTS")
    print("=" * 60)
    print(tree_of_thought(test_problem))
