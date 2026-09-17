# 🛡️ Навык 5: AI Evaluation & Safety

## Что это?

**AI Evaluation & Safety** — это практики оценки качества AI-систем и обеспечения их безопасной работы.

## Почему это важно?

- ✅ **Качество**: измерение и улучшение точности модели
- ✅ **Безопасность**: защита от вредного контента и атак
- ✅ **Доверие**: прозрачность и объяснимость решений
- ✅ **Соответствие**: соблюдение регуляторных требований

## Ключевые концепции

### 1. **Evaluation Metrics**
- **ROUGE/BLEU**: для задач генерации текста
- **Accuracy/Precision/Recall**: для классификации
- **Perplexity**: для языковых моделей
- **Human evaluation**: экспертная оценка

### 2. **Safety Guardrails**
- Content filtering (токсичность, насилие)
- Prompt injection protection
- Output validation
- Rate limiting

### 3. **Bias Detection**
- Demographic parity
- Equalized odds
- Bias mitigation techniques

### 4. **Monitoring**
- Logging и аудит
- Performance tracking
- Anomaly detection
- User feedback collection

## Примеры в этом разделе

- `basic_evaluation.py` — базовые метрики оценки
- `safety_guardrails.py` — защитные механизмы
- `bias_detection.py` — обнаружение предвзятости

## Ресурсы

- [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation)
- [Anthropic Constitutional AI](https://www.anthropic.com/news/constitutional-ai-harmlessness-from-ai-feedback)
- [Google AI Safety](https://ai.google/responsibility/safety/)

## Быстрый старт

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск примеров
python basic_evaluation.py
python safety_guardrails.py
```

## License

MIT
