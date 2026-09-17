"""
Prompt Optimizer
=================
Автоматическая оптимизация промптов на основе результатов.

Этот пример демонстрирует:
- Итеративное улучшение промптов
- A/B тестирование вариантов
- Метрики качества
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from typing import List, Dict, Tuple

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


class PromptOptimizer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.history = []
    
    def generate_prompt_variants(self, base_prompt: str, task: str) -> List[str]:
        """
        Генерация вариантов промпта для тестирования
        """
        meta_prompt = ChatPromptTemplate.from_template(
            """Вы - эксперт по prompt engineering. Создайте 3 варианта следующего промпта для задачи: {task}

Базовый промпт:
{base_prompt}

Создайте 3 улучшенных варианта с разными подходами:
1. Более конкретный
2. С примерами
3. С ограничениями и правилами

Верните только варианты промптов, разделенные '---':"""
        )
        
        chain = meta_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"task": task, "base_prompt": base_prompt})
        
        variants = [v.strip() for v in result.split("---") if v.strip()]
        return variants[:3]
    
    def evaluate_prompt(self, prompt_template: str, test_cases: List[Dict]) -> float:
        """
        Оценка качества промпта на тестовых случаях
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        scores = []
        for case in test_cases:
            try:
                result = chain.invoke({"input": case["input"]})
                
                # Простая оценка: длина ответа и наличие ключевых слов
                score = 0.0
                
                # Длина ответа (оптимально 50-500 символов)
                length = len(result)
                if 50 <= length <= 500:
                    score += 0.3
                elif length > 0:
                    score += 0.1
                
                # Наличие ответа (не пустой)
                if result.strip():
                    score += 0.3
                
                # Наличие ключевых слов из ожидаемого ответа
                if "expected" in case:
                    expected_keywords = case["expected"].lower().split()
                    result_lower = result.lower()
                    matches = sum(1 for kw in expected_keywords if kw in result_lower)
                    score += 0.4 * (matches / max(len(expected_keywords), 1))
                
                scores.append(score)
            except Exception as e:
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def optimize(self, base_prompt: str, task: str, test_cases: List[Dict], iterations: int = 2) -> Tuple[str, float]:
        """
        Итеративная оптимизация промпта
        """
        current_prompt = base_prompt
        best_score = self.evaluate_prompt(current_prompt, test_cases)
        
        print(f"Начальная оценка: {best_score:.2f}")
        
        for iteration in range(iterations):
            print(f"\nИтерация {iteration + 1}/{iterations}")
            
            # Генерируем варианты
            variants = self.generate_prompt_variants(current_prompt, task)
            
            # Тестируем каждый вариант
            for i, variant in enumerate(variants):
                score = self.evaluate_prompt(variant, test_cases)
                print(f"  Вариант {i+1}: {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    current_prompt = variant
                    print(f"  ✓ Новый лучший вариант!")
        
        return current_prompt, best_score


def ab_test_prompts(prompt_a: str, prompt_b: str, test_cases: List[Dict]) -> Dict:
    """
    A/B тестирование двух вариантов промптов
    """
    optimizer = PromptOptimizer()
    
    score_a = optimizer.evaluate_prompt(prompt_a, test_cases)
    score_b = optimizer.evaluate_prompt(prompt_b, test_cases)
    
    winner = "A" if score_a > score_b else "B"
    
    return {
        "prompt_a_score": score_a,
        "prompt_b_score": score_b,
        "winner": winner,
        "improvement": abs(score_a - score_b)
    }


if __name__ == "__main__":
    # Тестовые случаи
    test_cases = [
        {
            "input": "Объясни, что такое API",
            "expected": "API это интерфейс программирования приложений набор правил"
        },
        {
            "input": "Как работает машинное обучение?",
            "expected": "машинное обучение алгоритмы данные обучение модель предсказание"
        },
        {
            "input": "Что такое нейронная сеть?",
            "expected": "нейронная сеть узлы слои обучение веса активация"
        },
    ]
    
    # Базовый промпт
    base_prompt = "Ответь на вопрос: {input}"
    
    print("=" * 60)
    print("PROMPT OPTIMIZATION")
    print("=" * 60)
    
    optimizer = PromptOptimizer()
    optimized_prompt, score = optimizer.optimize(
        base_prompt=base_prompt,
        task="Объяснение технических концепций",
        test_cases=test_cases,
        iterations=2
    )
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ")
    print("=" * 60)
    print(f"Финальная оценка: {score:.2f}")
    print(f"\nОптимизированный промпт:\n{optimized_prompt}")
    
    # A/B тестирование
    print("\n" + "=" * 60)
    print("A/B ТЕСТ")
    print("=" * 60)
    
    prompt_a = "Ответь кратко: {input}"
    prompt_b = "Предоставь подробное объяснение с примерами для: {input}"
    
    result = ab_test_prompts(prompt_a, prompt_b, test_cases)
    
    print(f"Prompt A оценка: {result['prompt_a_score']:.2f}")
    print(f"Prompt B оценка: {result['prompt_b_score']:.2f}")
    print(f"Победитель: Prompt {result['winner']}")
    print(f"Улучшение: {result['improvement']:.2f}")
