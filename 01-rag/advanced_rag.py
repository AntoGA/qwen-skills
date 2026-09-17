"""
Advanced RAG with Reranking
============================
Продвинутая реализация RAG с использованием reranking для улучшения релевантности.

Техники:
- Query expansion
- Reranking с cross-encoder
- Hybrid search (semantic + keyword)
"""

import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.schema import Document

load_dotenv()


class AdvancedRAG:
    """Продвинутая RAG-система с reranking."""
    
    def __init__(self, documents, embedding_model="text-embedding-3-small"):
        self.documents = documents
        self.embedding_model = embedding_model
        self.vector_store = None
        self.retriever = None
        
    def preprocess_documents(self, chunk_size=500, chunk_overlap=50):
        """Предобработка документов с продвинутым разбиением."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
        )
        chunks = text_splitter.split_documents(self.documents)
        print(f"✓ Создано {len(chunks)} чанков из {len(self.documents)} документов")
        return chunks
    
    def create_vector_store(self, chunks):
        """Создание векторного хранилища."""
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        print("✓ Векторное хранилище создано")
        return self.vector_store
    
    def setup_retriever_with_reranking(self, k=10, rerank_top_n=3):
        """Настройка retriever с reranking."""
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Reranker для улучшения релевантности
        reranker = CohereRerank(
            model="rerank-english-v3.0",
            top_n=rerank_top_n
        )
        
        self.retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )
        
        print(f"✓ Retriever с reranking готов (k={k}, top_n={rerank_top_n})")
        return self.retriever
    
    def expand_query(self, query, llm=None):
        """Расширение запроса для улучшения поиска."""
        if llm is None:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        expansion_prompt = f"""Expand the following query with related terms and concepts 
to improve search results. Return only the expanded query.

Original query: {query}

Expanded query:"""
        
        expanded = llm.invoke(expansion_prompt).content.strip()
        return expanded
    
    def retrieve_with_expansion(self, query, use_expansion=True):
        """Поиск с опциональным расширением запроса."""
        search_query = query
        
        if use_expansion:
            search_query = self.expand_query(query)
            print(f"\n🔍 Expanded query: {search_query}")
        
        docs = self.retriever.invoke(search_query)
        return docs
    
    def generate_answer(self, query, context_docs, llm=None):
        """Генерация ответа на основе контекста."""
        if llm is None:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = f"""Answer the question based on the following context. 
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        response = llm.invoke(prompt)
        return response.content


def demo_advanced_rag():
    """Демонстрация продвинутого RAG."""
    print("=" * 60)
    print("🚀 Advanced RAG with Reranking")
    print("=" * 60)
    
    # Пример документов
    sample_docs = [
        Document(
            page_content="""
            LangChain is a framework for developing applications powered by 
            language models. It provides tools to connect LLMs with external 
            data sources, create chains of operations, and build agents that 
            can use tools. LangChain supports multiple LLM providers including 
            OpenAI, Anthropic, and local models.
            """,
            metadata={"source": "langchain.txt"}
        ),
        Document(
            page_content="""
            Vector databases like FAISS, Chroma, and Pinecone store embeddings 
            which are numerical representations of text. These databases enable 
            semantic search by finding documents with similar meanings, not just 
            similar keywords. This is crucial for RAG systems.
            """,
            metadata={"source": "vector_db.txt"}
        ),
        Document(
            page_content="""
            Reranking is a technique that improves search results by using a 
            more sophisticated model to score relevance. Cross-encoders are 
            commonly used for reranking because they can better understand the 
            relationship between query and document compared to bi-encoders.
            """,
            metadata={"source": "reranking.txt"}
        )
    ]
    
    # Создание Advanced RAG
    rag = AdvancedRAG(sample_docs)
    
    # Предобработка
    print("\n📄 Шаг 1: Предобработка документов...")
    chunks = rag.preprocess_documents(chunk_size=400, chunk_overlap=50)
    
    # Векторное хранилище
    print("\n🗄️  Шаг 2: Создание векторного хранилища...")
    rag.create_vector_store(chunks)
    
    # Retriever с reranking
    print("\n🔧 Шаг 3: Настройка retriever с reranking...")
    rag.setup_retriever_with_reranking(k=10, rerank_top_n=3)
    
    # Тестирование
    print("\n" + "=" * 60)
    print("🧪 Тестирование")
    print("=" * 60)
    
    query = "What is LangChain and how does it work with vector databases?"
    
    print(f"\n❓ Query: {query}")
    
    # Поиск с расширением
    print("\n🔎 Поиск с query expansion и reranking...")
    docs = rag.retrieve_with_expansion(query, use_expansion=True)
    
    print(f"\n📚 Найдено {len(docs)} релевантных документов:")
    for i, doc in enumerate(docs, 1):
        print(f"\n   [{i}] Source: {doc.metadata.get('source', 'unknown')}")
        print(f"       Content: {doc.page_content[:150]}...")
    
    # Генерация ответа
    print("\n💬 Генерация ответа...")
    answer = rag.generate_answer(query, docs)
    print(f"\n✅ Answer:\n{answer}")
    
    print("\n" + "=" * 60)
    print("✅ Демонстрация завершена!")
    print("=" * 60)


if __name__ == "__main__":
    demo_advanced_rag()
