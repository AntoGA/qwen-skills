# 🤖 Навык 2: AI Agents & Tool Use

## Что это?

**AI Agents** — это автономные системы, которые могут:
- Принимать решения на основе контекста
- Вызывать внешние инструменты (API, базы данных, поиск)
- Выполнять многошаговые задачи
- Взаимодействовать друг с другом (multi-agent systems)

## Почему это важно?

- ✅ **Автономность**: агенты могут работать без постоянного вмешательства человека
- ✅ **Интеграция**: подключение к реальным системам через API
- ✅ **Сложные задачи**: решение проблем, требующих множества шагов
- ✅ **Масштабируемость**: координация нескольких агентов

## Ключевые концепции

### 1. ReAct (Reasoning + Acting)
Агент чередует рассуждения и действия:
```
Thought: Мне нужно найти информацию о погоде
Action: вызвать weather_api(город="Москва")
Observation: Температура +20°C
Thought: Теперь я могу ответить пользователю
Answer: В Москве сейчас +20°C
```

### 2. Function Calling
Модель может вызывать функции с параметрами:
```python
tools = [
    {
        "name": "search_web",
        "description": "Поиск в интернете",
        "parameters": {
            "query": "строка для поиска"
        }
    }
]
```

### 3. Multi-Agent Systems
Несколько агентов работают вместе:
- **Researcher**: собирает информацию
- **Analyst**: анализирует данные
- **Writer**: создает отчет

## Структура проекта

```
02-agents/
├── README.md              # Этот файл
├── basic_agent.py         # Простой агент с инструментами
├── multi_agent.py         # Multi-agent система
├── tools/
│   ├── __init__.py
│   ├── web_search.py      # Инструмент поиска
│   ├── calculator.py      # Калькулятор
│   └── database.py        # Работа с БД
└── requirements.txt       # Зависимости
```

## Быстрый старт

```bash
cd 02-agents
pip install -r requirements.txt
python basic_agent.py
```

## Примеры использования

### Простой агент
```python
from basic_agent import create_agent

agent = create_agent()
result = agent.run("Найди информацию о квантовых компьютерах и объясни, как они работают")
print(result)
```

### Multi-agent система
```python
from multi_agent import create_research_team

team = create_research_team()
report = team.run("Подготовь отчет о трендах AI в 2026 году")
print(report)
```

## Инструменты

### Web Search
```python
from tools.web_search import search_web

results = search_web("latest AI news")
```

### Calculator
```python
from tools.calculator import calculate

result = calculate("(15 * 23) + 42")
```

### Database
```python
from tools.database import query_database

data = query_database("SELECT * FROM users WHERE age > 25")
```

## Ресурсы

### Документация
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [AutoGen](https://microsoft.github.io/autogen/)
- [CrewAI](https://github.com/joaomdmoura/crewAI)

### Статьи
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)

### Курсы
- [AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/)
- [Multi AI Agent Systems with crewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)

## Следующие шаги

1. Изучи `basic_agent.py` — простой агент с инструментами
2. Попробуй `multi_agent.py` — координация нескольких агентов
3. Создай свои инструменты в папке `tools/`

---

[← Назад к RAG](../01-rag/README.md) | [К главному README](../README.md) | [Далее: Multi-modal →](../03-multimodal/README.md)
