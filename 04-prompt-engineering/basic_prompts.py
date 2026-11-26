"""
Basic Prompt Engineering Techniques
====================================
Базовые техники создания эффективных промптов.

Этот пример демонстрирует:
- Zero-shot prompting
- Few-shot prompting
- Role-based prompting
- Structured output
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, ConfigDict



def get_llm():
    load_dotenv()
    return ChatOpenAI(model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"), temperature=0.7)



def zero_shot_prompt(question: str) -> str:
    """
    Zero-shot prompting - прямой запрос без примеров
    """
    prompt = ChatPromptTemplate.from_template(
        "Ответьте на следующий вопрос четко и кратко:\n\n{question}"
    )
    
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"question": question})


def few_shot_prompt(question: str) -> str:
    """
    Few-shot prompting - предоставление примеров для обучения
    """
    examples = [
        {"input": "Что такое Python?", "output": "Python - это высокоуровневый язык программирования, известный своей простотой и читаемостью."},
        {"input": "Что такое API?", "output": "API (Application Programming Interface) - это набор правил и протоколов, позволяющий различным приложениям взаимодействовать друг с другом."},
    ]
    
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Вы - полезный ассистент. Отвечайте в стиле примеров."),
        few_shot_prompt,
        ("human", "{input}"),
    ])
    
    chain = final_prompt | get_llm() | StrOutputParser()
    return chain.invoke({"input": question})


def role_based_prompt(question: str, role: str = "expert") -> str:
    """
    Role-based prompting - задание роли для модели
    """
    roles = {
        "expert": "Вы - эксперт в области искусственного интеллекта с 20-летним опытом.",
        "teacher": "Вы - терпеливый учитель, объясняющий сложные концепции простым языком.",
        "critic": "Вы - критический аналитик, который находит недостатки в любых аргументах.",
    }
    
    system_message = roles.get(role, roles["expert"])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}"),
    ])
    
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"question": question})


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    main_concept: str
    key_points: list[str]
    confidence: float = Field(ge=0, le=1)
    sources: list[str]


def structured_output_prompt(question: str) -> dict:
    """
    Structured output - получение ответа в структурированном формате
    """
    prompt = ChatPromptTemplate.from_template(
        """Проанализируйте следующий вопрос и предоставьте ответ в формате JSON:
        
Вопрос: {question}

Формат ответа:
{{
    "main_concept": "основная концепция",
    "key_points": ["пункт 1", "пункт 2", "пункт 3"],
    "confidence": 0.95,
    "sources": ["источник 1", "источник 2"]
}}

Ответ в формате JSON:"""
    )
    
    chain = prompt | get_llm() | StrOutputParser()
    result = chain.invoke({"question": question})
    
    return StructuredAnswer.model_validate_json(result).model_dump()


if __name__ == "__main__":
    load_dotenv()
    test_question = "Объясните, что такое машинное обучение"
    
    print("=" * 60)
    print("ZERO-SHOT PROMPTING")
    print("=" * 60)
    print(zero_shot_prompt(test_question))
    print()
    
    print("=" * 60)
    print("FEW-SHOT PROMPTING")
    print("=" * 60)
    print(few_shot_prompt(test_question))
    print()
    
    print("=" * 60)
    print("ROLE-BASED PROMPTING (Expert)")
    print("=" * 60)
    print(role_based_prompt(test_question, "expert"))
    print()
    
    print("=" * 60)
    print("ROLE-BASED PROMPTING (Teacher)")
    print("=" * 60)
    print(role_based_prompt(test_question, "teacher"))
    print()
    
    print("=" * 60)
    print("STRUCTURED OUTPUT")
    print("=" * 60)
    print(structured_output_prompt(test_question))
