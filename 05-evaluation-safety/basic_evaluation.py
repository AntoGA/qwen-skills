"""
Basic AI Evaluation
====================
Базовые метрики для оценки качества AI-систем.

Этот пример демонстрирует:
- ROUGE scores (для генерации текста)
- BLEU scores (для перевода)
- Accuracy/Precision/Recall (для классификации)
- Perplexity (для языковых моделей)
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

load_dotenv()

# Загрузка NLTK данных
nltk.download('punkt', quiet=True)


class TextEvaluator:
    """Оценщик качества генерации текста"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        self.smoothie = SmoothingFunction().method1
    
    def rouge_score(self, reference: str, generated: str) -> Dict[str, float]:
        """
        Вычисление ROUGE scores
        
        Args:
            reference: Эталонный текст
            generated: Сгенерированный текст
        
        Returns:
            Dict с ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        scores = self.rouge_scorer.score(reference, generated)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def bleu_score(self, reference: List[str], generated: str) -> float:
        """
        Вычисление BLEU score
        
        Args:
            reference: Список эталонных токенов
            generated: Сгенерированный текст
        
        Returns:
            BLEU score (0-1)
        """
        generated_tokens = nltk.word_tokenize(generated.lower())
        reference_tokens = [ref.lower().split() for ref in reference]
        
        score = sentence_bleu(
            reference_tokens, 
            generated_tokens,
            smoothing_function=self.smoothie
        )
        return score
    
    def batch_evaluate(self, references: List[str], generated: List[str]) -> Dict[str, float]:
        """
        Пакетная оценка нескольких текстов
        
        Args:
            references: Список эталонных текстов
            generated: Список сгенерированных текстов
        
        Returns:
            Dict со средними метриками
        """
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        bleu_scores = []
        
        for ref, gen in zip(references, generated):
            rouge = self.rouge_score(ref, gen)
            for key in rouge_scores:
                rouge_scores[key].append(rouge[key])
            
            bleu = self.bleu_score([ref], gen)
            bleu_scores.append(bleu)
        
        return {
            'avg_rouge1': np.mean(rouge_scores['rouge1']),
            'avg_rouge2': np.mean(rouge_scores['rouge2']),
            'avg_rougeL': np.mean(rouge_scores['rougeL']),
            'avg_bleu': np.mean(bleu_scores)
        }


class ClassificationEvaluator:
    """Оценщик качества классификации"""
    
    @staticmethod
    def evaluate(
        y_true: List[int], 
        y_pred: List[int],
        average: str = 'macro'
    ) -> Dict[str, float]:
        """
        Вычисление метрик классификации
        
        Args:
            y_true: Истинные метки
            y_pred: Предсказанные метки
            average: Метод усреднения ('macro', 'micro', 'weighted')
        
        Returns:
            Dict с accuracy, precision, recall, f1
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average),
            'recall': recall_score(y_true, y_pred, average=average),
            'f1': f1_score(y_true, y_pred, average=average)
        }


def calculate_perplexity(log_probs: List[float]) -> float:
    """
    Вычисление perplexity языковой модели
    
    Args:
        log_probs: Список логарифмических вероятностей
    
    Returns:
        Perplexity score
    """
    avg_log_prob = np.mean(log_probs)
    perplexity = np.exp(-avg_log_prob)
    return perplexity


if __name__ == "__main__":
    # Пример использования
    print("=" * 60)
    print("AI Evaluation Examples")
    print("=" * 60)
    
    # 1. ROUGE evaluation
    print("\n1. ROUGE Score Example:")
    evaluator = TextEvaluator()
    
    reference = "Искусственный интеллект революционизирует многие отрасли промышленности"
    generated = "ИИ революционизирует различные отрасли промышленности"
    
    rouge = evaluator.rouge_score(reference, generated)
    print(f"Reference: {reference}")
    print(f"Generated: {generated}")
    print(f"ROUGE-1: {rouge['rouge1']:.3f}")
    print(f"ROUGE-2: {rouge['rouge2']:.3f}")
    print(f"ROUGE-L: {rouge['rougeL']:.3f}")
    
    # 2. BLEU evaluation
    print("\n2. BLEU Score Example:")
    reference_tokens = ["искусственный", "интеллект", "меняет", "мир"]
    generated_text = "искусственный интеллект изменяет мир"
    
    bleu = evaluator.bleu_score([" ".join(reference_tokens)], generated_text)
    print(f"Reference: {' '.join(reference_tokens)}")
    print(f"Generated: {generated_text}")
    print(f"BLEU: {bleu:.3f}")
    
    # 3. Classification metrics
    print("\n3. Classification Metrics Example:")
    y_true = [0, 1, 1, 0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 0, 1, 1, 1, 1]
    
    classifier = ClassificationEvaluator()
    metrics = classifier.evaluate(y_true, y_pred)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    
    # 4. Perplexity
    print("\n4. Perplexity Example:")
    log_probs = [-2.5, -1.8, -3.2, -2.1, -1.9]
    perplexity = calculate_perplexity(log_probs)
    print(f"Log probabilities: {log_probs}")
    print(f"Perplexity: {perplexity:.2f}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)
