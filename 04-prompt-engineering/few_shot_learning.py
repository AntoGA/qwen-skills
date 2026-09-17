"""
Few-Shot Learning
==================
Обучение модели на примерах прямо в промпте.

Этот пример демонстрирует:
- Static few-shot examples
- Dynamic example selection
- Example formatting
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# База примеров
EXAMPLES = [
    {
        "input": "Переведи 'Hello' на русский",
        "output": "Привет",
        "category": "translation"
    },
    {
        "input": "Сделай текст более формальным: 'Привет, как дела?'",
        "output": "Здравствуйте! Как ваши дела?",
        "category": "style_transfer"
    },
    {
        "input": "Исправь грамматику: 'Я пошол в магазин'",
        "output": "Я пошёл в магазин",
        "category": "grammar_correction"
    },
    {
        "input": "Сократи текст: 'В связи с тем, что погода была плохая, мы решили остаться дома'",
        "output": "Из-за плохой погоды мы остались дома",
        "category": "summarization"
    },
]


def static_few_shot(user_input: str) -> str:
    """
    Static few-shot - использование фиксированного набора примеров
    """
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=EXAMPLES,
    )
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Вы - полезный ассистент для работы с текстом. Следуйте стилю примеров."),
        few_shot_prompt,
        ("human", "{input}"),
    ])
    
    chain = final_prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_input})


def dynamic_few_shot(user_input: str, k: int = 2) -> str:
    """
    Dynamic few-shot - выбор наиболее релевантных примеров на основе сходства
    """
    # Создаем embeddings для примеров
    embeddings = OpenAIEmbeddings()
    
    # Подготавливаем тексты для индексации
    texts = [f"{ex['input']} -> {ex['output']}" for ex in EXAMPLES]
    metadatas = [{"input": ex["input"], "output": ex["output"]} for ex in EXAMPLES]
    
    # Создаем vector store
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    # Ищем наиболее релевантные примеры
    docs = vectorstore.similarity_search(user_input, k=k)
    
    # Извлекаем примеры
    selected_examples = [
        {"input": doc.metadata["input"], "output": doc.metadata["output"]}
        for doc in docs
    ]
    
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=selected_examples,
    )
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "Вы - полезный ассистент для работы с текстом. Следуйте стилю примеров."),
        few_shot_prompt,
        ("human", "{input}"),
    ])
    
    chain = final_prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_input})


def formatted_few_shot(user_input: str) -> str:
    """
    Formatted few-shot - структурированное форматирование примеров
    """
    formatted_examples = "\n\n".join([
        f"ЗАДАЧА: {ex['input']}\nРЕШЕНИЕ: {ex['output']}\nКАТЕГОРИЯ: {ex['category']}"
        for ex in EXAMPLES
    ])
    
    prompt = ChatPromptTemplate.from_template(
        """Вы - ассистент для работы с текстом. Вот примеры вашей работы:

{examples}

Теперь обработайте следующий запрос:

ЗАДАЧА: {input}
РЕШЕНИЕ:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"examples": formatted_examples, "input": user_input})


if __name__ == "__main__":
    test_input = "Сделай текст более вежливым: 'Дай мне это'"
    
    print("=" * 60)
    print("STATIC FEW-SHOT")
    print("=" * 60)
    print(static_few_shot(test_input))
    print()
    
    print("=" * 60)
    print("DYNAMIC FEW-SHOT (k=2)")
    print("=" * 60)
    print(dynamic_few_shot(test_input, k=2))
    print()
    
    print("=" * 60)
    print("FORMATTED FEW-SHOT")
    print("=" * 60)
    print(formatted_few_shot(test_input))
