import importlib.util
from pathlib import Path
import socket
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '02-agents'))


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    for key in ('OPENAI_API_KEY', 'COHERE_API_KEY', 'LANGCHAIN_API_KEY', 'LANGSMITH_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('LANGCHAIN_TRACING_V2', 'false')
    monkeypatch.setenv('LANGSMITH_TRACING', 'false')
    def blocked(*args, **kwargs):
        raise AssertionError('Network is disabled in offline tests')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)


@pytest.fixture
def load_module():
    def load(relative):
        path = ROOT / relative
        name = 'qwen_test_' + path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    return load
