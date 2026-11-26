# Qwen Skills — учебные AI-пайплайны

Пять разделов с примерами RAG, инструментальных агентов, мультимодальных API,
prompt engineering и evaluation. **Статус: учебный PoC.** Offline-тесты проверяют
код и контракты с подставными моделями; качество реальных моделей не измерено.

Название репозитория не означает, что примеры уже проверены на Qwen. Текстовые
примеры используют OpenAI-compatible Chat Completions через LangChain. Для
совместимого Qwen endpoint задайте OPENAI_BASE_URL, OPENAI_API_KEY и DEFAULT_MODEL;
фактическую совместимость проверьте отдельно. Audio/image/moderation примеры
используют отдельные провайдерские API и не становятся совместимыми с Qwen
автоматически. Anthropic-интеграция и локальный inference здесь не реализованы.

| Раздел | Содержимое |
| --- | --- |
| [01-rag](01-rag/) | FAISS retrieval, RAG с источниками, Cohere reranking |
| [02-agents](02-agents/) | ReAct, последовательный LangGraph, calculator/read-only SQL |
| [03-multimodal](03-multimodal/) | Vision, transcription и image API примеры |
| [04-prompt-engineering](04-prompt-engineering/) | Prompt chains, few-shot, schema-validated JSON, heuristic optimizer |
| [05-evaluation-safety](05-evaluation-safety/) | Локальные метрики и демонстрационные guardrails |

## Установка и offline-проверка

Проверяемая платформа: Linux, Python **3.11/3.12**. SQL resource limits используют
sqlite3.setlimit из Python 3.11+. Для других ОС нужно проверить наличие FAISS wheels.

```bash
python -m venv .venv
source .venv/bin/activate       # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip check
python -m compileall -q 01-rag 02-agents 03-multimodal 04-prompt-engineering 05-evaluation-safety
python -m pytest -q tests --junitxml=artifacts/tests.xml
```

API-ключи для тестов не нужны. В тестах заблокированы исходящие socket connections,
используются FakeListLLM/FakeListChatModel и локальные deterministic embeddings.
Проверяется реальный FAISS retrieval, сборка ReAct/LangGraph и передача данных между
этапами. Это не оценка semantic quality, безопасности модели или live-provider SLA.

Зависимости ограничены совместимой legacy-когортой LangChain **0.3** и LangGraph
**0.6**. Не обновляйте один пакет до 1.x независимо от остальных. Все разделы
подключают общий requirements.txt. Это диапазоны версий, не полный lock-файл;
CI сохраняет pip freeze и JUnit для каждого окружения. Локальный snapshot пакетов
есть в [отчётах](docs/reports/2026-11-26-audit.md).

## Live-примеры

Только при осознанном запуске: они могут обращаться к платному API.

```bash
cp .env.example .env
# Заполните .env локально; не коммитьте ключи.
python 01-rag/basic_rag.py
python 02-agents/basic_agent.py
python 02-agents/multi_agent.py
```

Клиенты создаются при вызове операции, а не при импорте. RAG дополнительно требует
работающий embeddings endpoint/модель; reranking требует COHERE_API_KEY. Включённый
ReAct prompt локален, поэтому remote prompt hub не требуется.

- Web/news search — **явная заглушка**, не возвращает реальные новости или погоду.
- Calculator не использует eval: ограниченные AST, размер/глубина, finite results,
  ограниченная степень. Это арифметический пример, не sandbox для Python-кода.
- SQL tool — только чтение, файл открывается mode=ro; есть authorizer, лимиты длины,
  шагов запроса и строк. create_sample_database — отдельная trusted demo-функция.
  Подключайте инструмент только к отдельной разрешённой БД: чтение тоже раскрывает данные.
- Multi-agent пример последовательно обрабатывает текст; researcher без retrieval
  не проверяет факты. Реальная автономность и актуальность не заявляются.
- Guardrails — эвристики/учебные проверки; regex не доказывает защиту от injection.
- BLEU использует одинаковую локальную wordpunct tokenization обеих сторон;
  ROUGE использует локальные Unicode word tokens без стемминга. Это метрики
  перекрытия текста, а не доказательство смысловой корректности или качества модели.
- Prompt optimizer использует length/keyword score, а не независимую quality metric.
- Мультимодальные live API не проверены; доступность указанных model IDs зависит от провайдера.

[Отчёт об исправлениях и проверках](docs/reports/2026-11-26-audit.md).
MIT — [LICENSE](LICENSE).
