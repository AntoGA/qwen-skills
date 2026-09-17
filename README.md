# 🤖 Qwen Skills — 5 ключевых навыков для AI-разработчика

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Практический проект, демонстрирующий **5 самых востребованных навыков** для работы с AI-моделями (Qwen, GPT, Claude) в 2025–2026 году.

Каждый навык содержит теорию, рабочий код и примеры использования.

---

## 📋 Содержание

| # | Навык | Описание | Сложность |
|---|-------|----------|----------|
| 1 | 🎯 [RAG](./01-rag-systems/) | Retrieval-Augmented Generation — поиск и генерация по документам | ⭐⭐ |
| 2 | 🤖 [AI Agents](./02-ai-agents/) | Автономные агенты с инструментами и мульти-агентные системы | ⭐⭐⭐ |
| 3 | 🎨 [Multi-modal AI](./03-multi-modal/) | Работа с изображениями, аудио и видео | ⭐⭐ |
| 4 | ✍️ [Prompt Engineering](./04-prompt-engineering/) | Продвинутое проектирование промптов | ⭐ |
| 5 | 🛡️ [Evaluation & Safety](./05-evaluation-safety/) | Оценка качества и безопасность AI-систем | ⭐⭐⭐ |

---

## 🚀 Быстрый старт

```bash
# Клонировать репозиторий
git clone https://github.com/AntoGA/qwen-skills.git
cd qwen-skills

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Установить зависимости
pip install -r requirements.txt

# Настроить API-ключи
cp .env.example .env
# Отредактировать .env — вписать свои ключи
```

## ⚙️ Требования

- Python 3.9+
- OpenAI API key (или совместимый endpoint)
- (Опционально) Anthropic API key
- (Опционально) Hugging Face token

## 🛠️ Технологии

- **LangChain** — фреймворк для LLM-приложений
- **LangGraph** — оркестрация AI-агентов
- **FAISS / ChromaDB** — векторные базы данных
- **OpenAI / Anthropic** — LLM-провайдеры
- **Sentence Transformers** — эмбеддинги

## 📄 Лицензия

MIT License — см. [LICENSE](./LICENSE)

---

> 👤 **Автор:** [AntoGA](https://github.com/AntoGA)  
> 📅 **Дата:** Сентябрь 2026
