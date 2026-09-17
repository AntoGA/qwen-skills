# 🎯 Навык 1: RAG (Retrieval-Augmented Generation)

## Что это?

**RAG** — это техника, которая позволяет AI-моделям работать с актуальными данными, выходящими за рамки их обучающей выборки. Модель получает контекст из внешних источников перед генерацией ответа.

## Почему это важно?

- ✅ **Снижение галлюцинаций**: модель опирается на реальные документы
- ✅ **Актуальность**: работа с данными, которых нет в обучающей выборке
- ✅ **Прозрачность**: можно указать источники информации
- ✅ **Контроль**: вы решаете, какие данные использовать

## Архитектура RAG

```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Chunking   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Embedding  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Vector Store │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retrieval  │ ← Query
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     LLM     │ ← Context + Query
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Answer    │
└─────────────┘
```

## Что вы изучите

1. **Basic RAG** — простая реализация с FAISS
2. **Advanced RAG** — reranking, query transformation
3. **Hybrid Search** — комбинация semantic + keyword search
4. **Evaluation** — метрики качества RAG-систем

## Быстрый старт

```bash
cd 01-rag
pip install -r requirements.txt
python basic_rag.py
```

## Ресурсы

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Chroma Vector Database](https://docs.trychroma.com/)
