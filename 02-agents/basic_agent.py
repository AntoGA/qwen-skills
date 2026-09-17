"""
Basic AI Agent with Tools
==========================
Простой агент, использующий ReAct подход для выполнения задач с помощью инструментов.

Этот пример демонстрирует:
- Создание агента с инструментами
- ReAct reasoning loop
- Вызов внешних API
- Обработку ошибок
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub

# Load environment variables
load_dotenv()

# Import custom tools
from tools.web_search import search_web
from tools.calculator import calculate


def create_tools():
    """Создание набора инструментов для агента"""
    
    tools = [
        Tool(
            name="WebSearch",
            func=search_web,
            description="Поиск информации в интернете. Используй для получения актуальных данных."
        ),
        Tool(
            name="Calculator",
            func=calculate,
            description="Калькулятор для математических вычислений. Вводи выражение, например: (15 * 23) + 42"
        )
    ]
    
    return tools


def create_agent(verbose: bool = True):
    """
    Создание AI агента с инструментами
    
    Args:
        verbose: Выводить ли подробные логи рассуждений агента
    
    Returns:
        AgentExecutor: Готовый к работе агент
    """
    
    # Инициализация LLM
    llm = ChatOpenAI(
        model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        temperature=0,
        verbose=verbose
    )
    
    # Создание инструментов
    tools = create_tools()
    
    # Загрузка промпта для ReAct агента
    prompt = hub.pull("hwchase17/react")
    
    # Создание агента
    agent = create_react_agent(llm, tools, prompt)
    
    # Оборачивание в AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent_executor


def main():
    """Пример использования агента"""
    
    print("🤖 Создание AI агента...")
    agent = create_agent(verbose=True)
    
    # Примеры задач
    queries = [
        "Какая сейчас погода в Москве?",
        "Сколько будет (123 * 456) + 789?",
        "Найди информацию о последних достижениях в AI и посчитай, сколько лет прошло с 2020 года"
    ]
    
    print("\n" + "="*60)
    print("📝 Запуск агента с примерами задач")
    print("="*60 + "\n")
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─'*60}")
        print(f"Задача {i}: {query}")
        print(f"{'─'*60}\n")
        
        try:
            result = agent.invoke({"input": query})
            print(f"\n✅ Результат:\n{result['output']}\n")
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")
    
    # Интерактивный режим
    print("\n" + "="*60)
    print("🎮 Интерактивный режим (введите 'quit' для выхода)")
    print("="*60 + "\n")
    
    while True:
        query = input("Ваш вопрос: ").strip()
        
        if query.lower() in ['quit', 'exit', 'выход']:
            print("\n👋 До свидания!")
            break
        
        if not query:
            continue
        
        try:
            result = agent.invoke({"input": query})
            print(f"\n{result['output']}\n")
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")


if __name__ == "__main__":
    main()
