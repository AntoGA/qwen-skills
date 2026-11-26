from pathlib import Path
import pytest

EXAMPLES = sorted(str(p.relative_to(Path(__file__).resolve().parents[1]))
    for p in Path(__file__).resolve().parents[1].glob('0*/*.py'))


@pytest.mark.parametrize('path', EXAMPLES)
def test_examples_import_without_keys_or_client_creation(load_module, monkeypatch, path):
    import openai
    import nltk
    import langchain_openai
    def forbidden(*args, **kwargs):
        raise AssertionError('Provider clients/downloads must not be created at import')
    monkeypatch.setattr(openai.OpenAI, '__init__', forbidden)
    monkeypatch.setattr(langchain_openai.ChatOpenAI, '__init__', forbidden)
    monkeypatch.setattr(nltk, 'download', forbidden)
    load_module(path)
