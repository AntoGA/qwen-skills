from types import SimpleNamespace
from unittest.mock import Mock
import math
import pytest


def test_audio_segments_follow_sdk_attributes(load_module, monkeypatch, tmp_path):
    mod = load_module('03-multimodal/audio_transcription.py')
    audio = tmp_path/'audio.wav'; audio.write_bytes(b'test fixture')
    create = Mock(return_value=SimpleNamespace(segments=[SimpleNamespace(start=0.0, end=1.0, text='hello')]))
    monkeypatch.setattr(mod,'get_client',lambda:SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))))
    assert mod.transcribe_with_timestamps(str(audio)) == [{'start':0.0,'end':1.0,'text':'hello'}]
    assert create.call_args.kwargs['file'].closed


def test_vision_png_has_correct_data_url(load_module, monkeypatch, tmp_path):
    mod = load_module('03-multimodal/vision_analysis.py')
    path = tmp_path/'test.png';path.write_bytes(b'PNG fixture')
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='answer'))]))
    monkeypatch.setattr(mod, 'get_client', lambda:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    assert mod.analyze_image(str(path)) == 'answer'
    url = create.call_args.kwargs['messages'][0]['content'][1]['image_url']['url']
    assert url.startswith('data:image/png;base64,')


def test_image_count_rejected_before_client_creation(load_module, monkeypatch):
    mod = load_module('03-multimodal/image_generation.py')
    monkeypatch.setattr(mod, 'get_client', lambda:pytest.fail('must not create provider client'))
    with pytest.raises(ValueError):
        mod.generate_image('example', n=2)


def test_failed_image_download_preserves_existing_file(load_module, monkeypatch, tmp_path):
    import httpx
    mod = load_module('03-multimodal/image_generation.py')
    path = tmp_path/'existing.png';path.write_bytes(b'original')
    class FailedResponse:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self):
            raise httpx.HTTPStatusError('not found', request=httpx.Request('GET','https://example.org/image'),response=httpx.Response(404))
    monkeypatch.setattr(mod.httpx, 'stream', lambda *args,**kwargs:FailedResponse())
    with pytest.raises(httpx.HTTPStatusError):
        mod.download_image('https://example.org/image', str(path))
    assert path.read_bytes() == b'original'


def test_local_evaluation_requires_no_nltk_download(load_module):
    mod = load_module('05-evaluation-safety/basic_evaluation.py')
    evaluator = mod.TextEvaluator()
    text = 'A sample sentence, with punctuation.'
    assert evaluator.bleu_score([text], text) == pytest.approx(1)
    report = evaluator.batch_evaluate([text],[text])
    assert all(math.isfinite(v) for v in report.values())
    with pytest.raises(ValueError): evaluator.batch_evaluate([],[])
    with pytest.raises(ValueError): evaluator.batch_evaluate([text],[])
    assert mod.calculate_perplexity([math.log(0.5)]) == pytest.approx(2)
    with pytest.raises(ValueError): mod.calculate_perplexity([])


def test_interrupted_download_does_not_replace_destination(load_module, monkeypatch, tmp_path):
    import httpx
    mod = load_module('03-multimodal/image_generation.py')
    destination = tmp_path/'image.png'; destination.write_bytes(b'original')
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def raise_for_status(self): pass
        def iter_bytes(self):
            yield b'partial'
            raise httpx.ReadError('interrupted')
    monkeypatch.setattr(mod.httpx, 'stream', lambda *args,**kwargs:Response())
    with pytest.raises(httpx.ReadError): mod.download_image('https://example.org/image', str(destination))
    assert destination.read_bytes() == b'original'
    assert list(tmp_path.iterdir()) == [destination]


def test_audio_temporary_chunks_removed_after_failure(load_module, monkeypatch, tmp_path):
    import sys
    from pathlib import Path
    mod = load_module('03-multimodal/audio_transcription.py')
    paths = []
    class Audio:
        def __len__(self): return 1000
        def __getitem__(self, key): return self
        def export(self,path,format): paths.append(Path(path)); Path(path).write_bytes(b'fixture')
    monkeypatch.setitem(sys.modules, 'pydub', SimpleNamespace(AudioSegment=SimpleNamespace(from_file=lambda _:Audio())))
    def fail(path): raise RuntimeError('provider unavailable')
    monkeypatch.setattr(mod, 'transcribe_audio', fail)
    with pytest.raises(RuntimeError): mod.transcribe_long_audio('fixture', chunk_duration=1)
    assert paths and not paths[0].exists() and not paths[0].parent.exists()
    with pytest.raises(ValueError): mod.transcribe_long_audio('fixture', chunk_duration=0)


def test_rouge_does_not_discard_cyrillic(load_module):
    mod = load_module('05-evaluation-safety/basic_evaluation.py')
    evaluator = mod.TextEvaluator()
    assert evaluator.rouge_score('Русский пример текста', 'Русский пример текста')['rouge1'] == pytest.approx(1)
    assert evaluator.rouge_score('Пример нового текста', 'Пример другого текста')['rouge1'] > 0
