from types import SimpleNamespace
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]
    def embed_query(self, text):
        return [1.0, 0.0] if 'qwen' in text.lower() else [0.0, 1.0]


def test_basic_rag_retrieval_and_source_documents(load_module):
    mod = load_module('01-rag/basic_rag.py')
    docs = mod.create_documents([
        {'content':'Qwen example document', 'metadata':{'source':'qwen.txt'}},
        {'content':'Other example', 'metadata':{'source':'other.txt'}}])
    chunks = mod.split_documents(docs, chunk_size=100, chunk_overlap=10)
    store = mod.create_vector_store(chunks, embeddings=LocalEmbeddings())
    chain = mod.create_rag_chain(store, llm=FakeListLLM(responses=['Offline answer']))
    result = chain.invoke({'query':'Qwen'})
    assert result['result'] == 'Offline answer'
    assert result['source_documents'][0].metadata['source'] == 'qwen.txt'


def test_advanced_rag_reranker_uses_retrieved_documents(load_module, monkeypatch):
    mod = load_module('01-rag/advanced_rag.py')
    class FirstDocument(BaseDocumentCompressor):
        def compress_documents(self, documents, query, callbacks=None):
            return documents[:1]
    monkeypatch.setattr(mod, 'CohereRerank', lambda **kwargs: FirstDocument())
    rag = mod.AdvancedRAG([Document(page_content='Qwen example', metadata={'source':'one'})])
    rag.create_vector_store(rag.preprocess_documents(), embeddings=LocalEmbeddings())
    rag.setup_retriever_with_reranking(k=2, rerank_top_n=1)
    docs = rag.retrieve_with_expansion('Qwen', use_expansion=False)
    assert len(docs) == 1 and docs[0].metadata['source'] == 'one'
    assert rag.generate_answer('Qwen?', docs, llm=FakeListChatModel(responses=['Answer'])) == 'Answer'


def test_react_agent_calls_calculator_without_prompt_hub(load_module):
    mod = load_module('02-agents/basic_agent.py')
    llm = FakeListLLM(responses=['Action: Calculator\nAction Input: 2 + 3', 'Final Answer: 5'])
    agent = mod.create_agent(verbose=False, llm=llm)
    assert agent.invoke({'input':'Calculate 2+3'})['output'] == '5'


def test_research_graph_compiles_and_passes_state(load_module, monkeypatch):
    mod = load_module('02-agents/multi_agent.py')
    monkeypatch.setattr(mod, 'create_researcher', lambda: SimpleNamespace(invoke=lambda data: 'R:'+data['task']))
    monkeypatch.setattr(mod, 'create_analyst', lambda: SimpleNamespace(invoke=lambda data: 'A:'+data['research']))
    monkeypatch.setattr(mod, 'create_writer', lambda: SimpleNamespace(invoke=lambda data: 'W:'+data['analysis']))
    team = mod.create_research_team()
    result = team.invoke({'task':'Example', 'research':'', 'analysis':'', 'report':'', 'messages':[]})
    assert result['report'] == 'W:A:R:Example'
    assert len(result['messages']) == 3


def test_prompt_chain_uses_injected_fake_model(load_module, monkeypatch):
    mod = load_module('04-prompt-engineering/basic_prompts.py')
    monkeypatch.setattr(mod, 'get_llm', lambda: FakeListChatModel(responses=['Example response']))
    assert mod.zero_shot_prompt('Question') == 'Example response'
    assert mod.few_shot_prompt('Question') == 'Example response'
    assert mod.role_based_prompt('Question') == 'Example response'


def test_optimizer_tie_is_not_declared_a_winner(load_module, monkeypatch):
    mod = load_module('04-prompt-engineering/prompt_optimizer.py')
    monkeypatch.setattr(mod, 'PromptOptimizer', lambda: SimpleNamespace(evaluate_prompt=lambda *a:0.5))
    assert mod.ab_test_prompts('A', 'B', [])['winner'] == 'tie'


def test_structured_output_validates_json_schema(load_module, monkeypatch):
    import pytest
    from pydantic import ValidationError
    mod = load_module('04-prompt-engineering/basic_prompts.py')
    json_text = '{"main_concept":"RAG","key_points":["retrieve"],"confidence":0.8,"sources":[]}'
    monkeypatch.setattr(mod, 'get_llm', lambda:FakeListChatModel(responses=[json_text]))
    assert mod.structured_output_prompt('Example')['main_concept'] == 'RAG'
    monkeypatch.setattr(mod, 'get_llm', lambda:FakeListChatModel(responses=['not JSON']))
    with pytest.raises(ValidationError): mod.structured_output_prompt('Example')


def test_dynamic_few_shot_uses_fake_embeddings(load_module, monkeypatch):
    mod = load_module('04-prompt-engineering/few_shot_learning.py')
    monkeypatch.setattr(mod, 'OpenAIEmbeddings', lambda: LocalEmbeddings())
    monkeypatch.setattr(mod, 'get_llm', lambda:FakeListChatModel(responses=['Result']))
    assert mod.dynamic_few_shot('example', k=2) == 'Result'
