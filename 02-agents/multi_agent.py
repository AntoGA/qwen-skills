"""
Multi-Agent System
===================
Система из нескольких агентов, работающих вместе для решения сложных задач.

Агенты:
- Researcher: собирает информацию
- Analyst: анализирует данные
- Writer: создает отчет
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

load_dotenv()


class AgentState(TypedDict):
    """Состояние системы агентов"""
    task: str
    research: str
    analysis: str
    report: str
    messages: Annotated[list, operator.add]


def create_researcher():
    """Создание агента-исследователя"""
    
    prompt = ChatPromptTemplate.from_template("""
    Ты - эксперт-исследователь. Твоя задача - собрать информацию по теме.
    
    Тема: {task}
    
    Собери ключевые факты, статистику и важные детали.
    Будь краток и конкретен.
    """)
    
    llm = ChatOpenAI(
        model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        temperature=0.3
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain


def create_analyst():
    """Создание агента-аналитика"""
    
    prompt = ChatPromptTemplate.from_template("""
    Ты - эксперт-аналитик. Проанализируй собранную информацию.
    
    Тема: {task}
    
    Собранная информация:
    {research}
    
    Выдели:
    1. Ключевые тренды
    2. Важные инсайты
    3. Потенциальные проблемы
    4. Возможности
    """)
    
    llm = ChatOpenAI(
        model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        temperature=0.3
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain


def create_writer():
    """Создание агента-писателя"""
    
    prompt = ChatPromptTemplate.from_template("""
    Ты - профессиональный писатель. Создай структурированный отчет.
    
    Тема: {task}
    
    Анализ:
    {analysis}
    
    Создай отчет со следующей структурой:
    1. Введение (2-3 предложения)
    2. Основные выводы (3-5 пунктов)
    3. Рекомендации (2-3 пункта)
    4. Заключение (1-2 предложения)
    
    Стиль: профессиональный, но понятный.
    """)
    
    llm = ChatOpenAI(
        model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        temperature=0.5
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain


def research_node(state: AgentState):
    """Узел исследования"""
    print("🔍 Researcher: собираю информацию...")
    researcher = create_researcher()
    research = researcher.invoke({"task": state["task"]})
    return {
        "research": research,
        "messages": ["Research completed"]
    }


def analysis_node(state: AgentState):
    """Узел анализа"""
    print("📊 Analyst: анализирую данные...")
    analyst = create_analyst()
    analysis = analyst.invoke({
        "task": state["task"],
        "research": state["research"]
    })
    return {
        "analysis": analysis,
        "messages": ["Analysis completed"]
    }


def writing_node(state: AgentState):
    """Узел написания отчета"""
    print("✍️ Writer: создаю отчет...")
    writer = create_writer()
    report = writer.invoke({
        "task": state["task"],
        "analysis": state["analysis"]
    })
    return {
        "report": report,
        "messages": ["Report completed"]
    }


def create_research_team():
    """
    Создание команды агентов
    
    Returns:
        Compiled graph: Готовая к работе multi-agent система
    """
    
    # Создание графа
    workflow = StateGraph(AgentState)
    
    # Добавление узлов
    workflow.add_node("research", research_node)
    workflow.add_node("analyze", analysis_node)
    workflow.add_node("write", writing_node)
    
    # Определение потоков
    workflow.set_entry_point("research")
    workflow.add_edge("research", "analyze")
    workflow.add_edge("analyze", "write")
    workflow.add_edge("write", END)
    
    # Компиляция
    return workflow.compile()


def main():
    """Пример использования multi-agent системы"""
    
    print("🤖 Создание команды агентов...")
    team = create_research_team()
    
    # Примеры задач
    tasks = [
        "Тренды в AI индустрии 2026 года",
        "Влияние квантовых компьютеров на криптографию",
        "Будущее автономных автомобилей"
    ]
    
    print("\n" + "="*60)
    print("📝 Запуск multi-agent системы")
    print("="*60 + "\n")
    
    for i, task in enumerate(tasks, 1):
        print(f"\n{'─'*60}")
        print(f"Задача {i}: {task}")
        print(f"{'─'*60}\n")
        
        # Запуск агентов
        result = team.invoke({
            "task": task,
            "research": "",
            "analysis": "",
            "report": "",
            "messages": []
        })
        
        print(f"\n{'═'*60}")
        print("📄 ИТОГОВЫЙ ОТЧЕТ")
        print(f"{'═'*60}\n")
        print(result["report"])
        print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
