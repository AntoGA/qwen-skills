"""
Basic RAG Implementation
========================
Простая реализация Retrieval-Augmented Generation с использованием FAISS.

Этот пример демонстрирует:
- Загрузку документов
- Разбиение на чанки
- Создание эмбеддингов
- Семантический поиск
- Генерацию ответа с контекстом
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

# Загрузка переменных окружения
load_dotenv()

# Пример документов для демонстрации
SAMPLE_DOCUMENTS = [
    {
        "content": """
        Qwen — это серия больших языковых моделей, разработанных Alibaba Cloud.
        Модели Qwen обучены на обширном корпусе текстов и обладают сильными 
        способностями в понимании естественного языка, генерации текста и 
        решении различных задач. Qwen поддерживает множество языков и 
        демонстрирует выдающуюся производительность в задачах рассуждения.
        """,
        "metadata": {"source": "about_qwen.txt"}
    },
    {
        "content": """
        RAG (Retrieval-Augmented Generation) — это техника, которая улучшает 
        качество генерации LLM за счет предоставления релевантного контекста 
        из внешних источников. Вместо того чтобы полагаться только на параметры 
        модели, RAG извлекает информацию из базы знаний и передает её в промпт.
        Это значительно снижает галлюцинации и повышает актуальность ответов.
        """,
        "metadata": {"source": "about_rag.txt"}
    },
    {
        "content": """
        Векторные базы данных хранят эмбеддинги — числовые представления 
        текста в многомерном пространстве. Похожие тексты находятся близко 
        друг к другу в этом пространстве. FAISS (Facebook AI Similarity Search) 
        — это библиотека для эффективного поиска похожих векторов. Она 
        поддерживает различные индексы и работает очень быстро даже на 
        миллионах документов.
        """,
        "metadata": {"source": "about_vectors.txt"}
    }
]


def create_documents(sample_docs):
    """Создание объектов Document из словарей."""
    from langchain.schema import Document
    return [
        Document(page_content=doc["content"], metadata=doc["metadata"])
        for doc in sample_docs
    ]


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Разбиение документов на чанки."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_documents(documents)


def create_vector_store(chunks, embedding_model="text-embedding-3-small"):
    """Создание векторного хранилища из чанков."""
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def create_rag_chain(vector_store, model_name="gpt-4o-mini"):
    """Создание RAG-цепочки."""
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=500
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
    
    return qa_chain


def main():
    """Основная функция демонстрации RAG."""
    print("=" * 60)
    print("🎯 Basic RAG Implementation")
    print("=" * 60)
    
    # 1. Создание документов
    print("\n📄 Шаг 1: Создание документов...")
    documents = create_documents(SAMPLE_DOCUMENTS)
    print(f"   Создано {len(documents)} документов")
    
    # 2. Разбиение на чанки
    print("\n✂️  Шаг 2: Разбиение на чанки...")
    chunks = split_documents(documents)
    print(f"   Создано {len(chunks)} чанков")
    
    # 3. Создание векторного хранилища
    print("\n🗄️  Шаг 3: Создание векторного хранилища...")
    vector_store = create_vector_store(chunks)
    print("   ✓ Векторное хранилище создано")
    
    # 4. Создание RAG-цепочки
    print("\n🔗 Шаг 4: Создание RAG-цепочки...")
    qa_chain = create_rag_chain(vector_store)
    print("   ✓ RAG-цепочка готова")
    
    # 5. Тестирование
    print("\n" + "=" * 60)
    print("🧪 Тестирование RAG-системы")
    print("=" * 60)
    
    test_queries = [
        "Что такое Qwen?",
        "Как работает RAG?",
        "Что такое векторные базы данных?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 60}")
        print(f"Вопрос {i}: {query}")
        print(f"{'─' * 60}")
        
        result = qa_chain.invoke({"query": query})
        
        print(f"\n💬 Ответ:")
        print(result["result"])
        
        print(f"\n📚 Источники ({len(result['source_documents'])}):")
        for j, doc in enumerate(result["source_documents"], 1):
            print(f"   {j}. {doc.metadata.get('source', 'unknown')}")
    
    print("\n" + "=" * 60)
    print("✅ Демонстрация завершена!")
    print("=" * 60)


if __name__ == "__main__":
    main()
