"""
Bias Detection
===============
Обнаружение и анализ предвзятости в AI-системах.

Этот пример демонстрирует:
- Demographic parity
- Equalized odds
- Disparate impact
- Bias mitigation techniques
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class BiasAnalyzer:
    """Анализатор предвзятости для классификационных моделей"""
    
    def __init__(self):
        self.metrics = {}
    
    def demographic_parity(
        self, 
        y_pred: List[int], 
        protected_attribute: List[str]
    ) -> Dict[str, float]:
        """
        Вычисление demographic parity
        
        Demographic parity: P(Y=1 | A=a) = P(Y=1 | A=b)
        для всех групп a, b
        
        Args:
            y_pred: Предсказания модели (0 или 1)
            protected_attribute: Защищенный атрибут (например, пол, раса)
        
        Returns:
            Dict с positive rate для каждой группы
        """
        groups = defaultdict(list)
        
        for pred, attr in zip(y_pred, protected_attribute):
            groups[attr].append(pred)
        
        positive_rates = {}
        for group, predictions in groups.items():
            positive_rates[group] = np.mean(predictions)
        
        self.metrics['demographic_parity'] = positive_rates
        return positive_rates
    
    def equalized_odds(
        self,
        y_true: List[int],
        y_pred: List[int],
        protected_attribute: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Вычисление equalized odds
        
        Equalized odds: TPR и FPR одинаковы для всех групп
        
        Args:
            y_true: Истинные метки
            y_pred: Предсказания модели
            protected_attribute: Защищенный атрибут
        
        Returns:
            Dict с TPR и FPR для каждой группы
        """
        groups = defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})
        
        for true, pred, attr in zip(y_true, y_pred, protected_attribute):
            if true == 1 and pred == 1:
                groups[attr]['tp'] += 1
            elif true == 0 and pred == 1:
                groups[attr]['fp'] += 1
            elif true == 0 and pred == 0:
                groups[attr]['tn'] += 1
            elif true == 1 and pred == 0:
                groups[attr]['fn'] += 1
        
        metrics = {}
        for group, counts in groups.items():
            tpr = counts['tp'] / (counts['tp'] + counts['fn']) if (counts['tp'] + counts['fn']) > 0 else 0
            fpr = counts['fp'] / (counts['fp'] + counts['tn']) if (counts['fp'] + counts['tn']) > 0 else 0
            metrics[group] = {'tpr': tpr, 'fpr': fpr}
        
        self.metrics['equalized_odds'] = metrics
        return metrics
    
    def disparate_impact(
        self,
        y_pred: List[int],
        protected_attribute: List[str],
        privileged_group: str,
        unprivileged_group: str
    ) -> float:
        """
        Вычисление disparate impact ratio
        
        DI = P(Y=1 | A=unprivileged) / P(Y=1 | A=privileged)
        
        Приемлемый диапазон: 0.8 - 1.25
        
        Args:
            y_pred: Предсказания модели
            protected_attribute: Защищенный атрибут
            privileged_group: Привилегированная группа
            unprivileged_group: Непривилегированная группа
        
        Returns:
            Disparate impact ratio
        """
        privileged_preds = [p for p, a in zip(y_pred, protected_attribute) if a == privileged_group]
        unprivileged_preds = [p for p, a in zip(y_pred, protected_attribute) if a == unprivileged_group]
        
        privileged_rate = np.mean(privileged_preds)
        unprivileged_rate = np.mean(unprivileged_preds)
        
        di = unprivileged_rate / privileged_rate if privileged_rate > 0 else 0
        
        self.metrics['disparate_impact'] = di
        return di
    
    def statistical_parity_difference(
        self,
        y_pred: List[int],
        protected_attribute: List[str],
        privileged_group: str,
        unprivileged_group: str
    ) -> float:
        """
        Вычисление statistical parity difference
        
        SPD = P(Y=1 | A=privileged) - P(Y=1 | A=unprivileged)
        
        Идеальное значение: 0
        
        Args:
            y_pred: Предсказания модели
            protected_attribute: Защищенный атрибут
            privileged_group: Привилегированная группа
            unprivileged_group: Непривилегированная группа
        
        Returns:
            Statistical parity difference
        """
        privileged_preds = [p for p, a in zip(y_pred, protected_attribute) if a == privileged_group]
        unprivileged_preds = [p for p, a in zip(y_pred, protected_attribute) if a == unprivileged_group]
        
        privileged_rate = np.mean(privileged_preds)
        unprivileged_rate = np.mean(unprivileged_preds)
        
        spd = privileged_rate - unprivileged_rate
        
        self.metrics['statistical_parity_difference'] = spd
        return spd
    
    def generate_report(self) -> str:
        """Генерация отчета о предвзятости"""
        report = []
        report.append("=" * 60)
        report.append("BIAS ANALYSIS REPORT")
        report.append("=" * 60)
        
        if 'demographic_parity' in self.metrics:
            report.append("\n1. Demographic Parity:")
            for group, rate in self.metrics['demographic_parity'].items():
                report.append(f"   {group}: {rate:.3f}")
        
        if 'equalized_odds' in self.metrics:
            report.append("\n2. Equalized Odds:")
            for group, metrics in self.metrics['equalized_odds'].items():
                report.append(f"   {group}:")
                report.append(f"     TPR: {metrics['tpr']:.3f}")
                report.append(f"     FPR: {metrics['fpr']:.3f}")
        
        if 'disparate_impact' in self.metrics:
            di = self.metrics['disparate_impact']
            report.append(f"\n3. Disparate Impact: {di:.3f}")
            if 0.8 <= di <= 1.25:
                report.append("   ✓ Приемлемый диапазон (0.8-1.25)")
            else:
                report.append("   ✗ Вне приемлемого диапазона")
        
        if 'statistical_parity_difference' in self.metrics:
            spd = self.metrics['statistical_parity_difference']
            report.append(f"\n4. Statistical Parity Difference: {spd:.3f}")
            if abs(spd) < 0.1:
                report.append("   ✓ Низкая предвзятость")
            else:
                report.append("   ✗ Высокая предвзятость")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)


class BiasMitigation:
    """Техники снижения предвзятости"""
    
    @staticmethod
    def reweighting(
        y_true: List[int],
        protected_attribute: List[str]
    ) -> List[float]:
        """
        Reweighting: присвоение весов примерам для балансировки
        
        Args:
            y_true: Истинные метки
            protected_attribute: Защищенный атрибут
        
        Returns:
            Список весов для каждого примера
        """
        groups = defaultdict(list)
        for i, attr in enumerate(protected_attribute):
            groups[attr].append(i)
        
        weights = [1.0] * len(y_true)
        total_samples = len(y_true)
        num_groups = len(groups)
        
        for group, indices in groups.items():
            group_size = len(indices)
            weight = total_samples / (num_groups * group_size)
            for idx in indices:
                weights[idx] = weight
        
        return weights
    
    @staticmethod
    def threshold_adjustment(
        y_scores: List[float],
        protected_attribute: List[str],
        target_positive_rate: float = 0.5
    ) -> List[int]:
        """
        Threshold adjustment: разные пороги для разных групп
        
        Args:
            y_scores: Вероятности предсказаний
            protected_attribute: Защищенный атрибут
            target_positive_rate: Целевая положительная rate
        
        Returns:
            Скорректированные предсказания
        """
        groups = defaultdict(list)
        for i, attr in enumerate(protected_attribute):
            groups[attr].append(i)
        
        predictions = [0] * len(y_scores)
        
        for group, indices in groups.items():
            group_scores = [y_scores[i] for i in indices]
            threshold = np.percentile(group_scores, (1 - target_positive_rate) * 100)
            
            for idx in indices:
                predictions[idx] = 1 if y_scores[idx] >= threshold else 0
        
        return predictions


if __name__ == "__main__":
    print("=" * 60)
    print("Bias Detection Examples")
    print("=" * 60)
    
    # Генерация тестовых данных
    np.random.seed(42)
    n_samples = 100
    
    # Защищенный атрибут (пол)
    gender = ['male'] * 50 + ['female'] * 50
    
    # Истинные метки (с небольшой предвзятостью)
    y_true = [1] * 30 + [0] * 20 + [1] * 25 + [0] * 25
    
    # Предсказания модели (с предвзятостью)
    y_pred = [1] * 35 + [0] * 15 + [1] * 20 + [0] * 30
    
    # Вероятности предсказаний
    y_scores = np.random.uniform(0.3, 0.9, n_samples).tolist()
    
    # Анализ предвзятости
    analyzer = BiasAnalyzer()
    
    print("\n1. Demographic Parity:")
    dp = analyzer.demographic_parity(y_pred, gender)
    for group, rate in dp.items():
        print(f"   {group}: {rate:.3f}")
    
    print("\n2. Equalized Odds:")
    eo = analyzer.equalized_odds(y_true, y_pred, gender)
    for group, metrics in eo.items():
        print(f"   {group}: TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}")
    
    print("\n3. Disparate Impact:")
    di = analyzer.disparate_impact(y_pred, gender, 'male', 'female')
    print(f"   DI = {di:.3f}")
    print(f"   Status: {'✓ Acceptable' if 0.8 <= di <= 1.25 else '✗ Unacceptable'}")
    
    print("\n4. Statistical Parity Difference:")
    spd = analyzer.statistical_parity_difference(y_pred, gender, 'male', 'female')
    print(f"   SPD = {spd:.3f}")
    print(f"   Status: {'✓ Low bias' if abs(spd) < 0.1 else '✗ High bias'}")
    
    # Полный отчет
    print("\n" + analyzer.generate_report())
    
    # Mitigation techniques
    print("\n5. Bias Mitigation:")
    
    print("\n   a) Reweighting:")
    weights = BiasMitigation.reweighting(y_true, gender)
    print(f"   Sample weights: {weights[:5]}")
    
    print("\n   b) Threshold Adjustment:")
    adjusted_pred = BiasMitigation.threshold_adjustment(y_scores, gender, target_positive_rate=0.5)
    print(f"   Original positive rate: {np.mean(y_pred):.3f}")
    print(f"   Adjusted positive rate: {np.mean(adjusted_pred):.3f}")
    
    print("\n" + "=" * 60)
    print("Bias analysis complete!")
    print("=" * 60)
