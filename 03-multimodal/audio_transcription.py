"""
Audio Transcription
====================
Транскрибация аудио с использованием Whisper API.

Этот пример демонстрирует:
- Загрузку аудиофайлов
- Транскрибацию речи в текст
- Перевод аудио на английский
- Обработку длинных файлов
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI



def get_client():
    """Create a provider client only when a live operation is requested."""
    load_dotenv()
    return OpenAI()



def transcribe_audio(audio_path: str, language: str = "ru") -> str:
    """
    Транскрибация аудиофайла в текст
    
    Args:
        audio_path: Путь к аудиофайлу
        language: Язык аудио (ISO 639-1 код)
    
    Returns:
        str: Транскрибированный текст
    """
    
    with open(audio_path, "rb") as audio_file:
        transcript = get_client().audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            response_format="text"
        )
    
    return transcript


def translate_audio(audio_path: str) -> str:
    """
    Транскрибация и перевод аудио на английский
    
    Args:
        audio_path: Путь к аудиофайлу
    
    Returns:
        str: Переведенный текст на английском
    """
    
    with open(audio_path, "rb") as audio_file:
        translation = get_client().audio.translations.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    
    return translation


def transcribe_long_audio(audio_path: str, chunk_duration: int = 600) -> str:
    """
    Транскрибация длинных аудиофайлов с разбиением на чанки
    
    Args:
        audio_path: Путь к аудиофайлу
        chunk_duration: Длительность чанка в секундах (по умолчанию 10 минут)
    
    Returns:
        str: Полная транскрибация
    
    Note:
        Требует установки pydub: pip install pydub
    """
    
    if type(chunk_duration) is not int or chunk_duration < 1:
        raise ValueError('chunk_duration must be a positive integer')
    try:
        from pydub import AudioSegment
    except ImportError:
        print("⚠️  Установите pydub: pip install pydub")
        return ""
    
    # Загружаем аудио
    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)
    chunk_ms = chunk_duration * 1000
    
    transcripts = []
    
    print(f"🎵 Обрабатываем аудио длительностью {duration_ms / 1000:.1f} сек")
    
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="qwen-audio-") as directory:
        for i in range(0, duration_ms, chunk_ms):
            chunk = audio[i:i + chunk_ms]
            temp_path = str(Path(directory) / f"chunk_{i}.wav")
            chunk.export(temp_path, format="wav")
            transcripts.append(transcribe_audio(temp_path))

    return " ".join(transcripts)


def transcribe_with_timestamps(audio_path: str) -> list:
    """
    Транскрибация с временными метками
    
    Args:
        audio_path: Путь к аудиофайлу
    
    Returns:
        list: Список сегментов с текстом и временем
    """
    
    with open(audio_path, "rb") as audio_file:
        response = get_client().audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    
    segments = []
    for segment in response.segments:
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
    
    return segments


if __name__ == "__main__":
    load_dotenv()
    # Пример использования
    
    test_audio = "test_audio.mp3"
    
    if not Path(test_audio).exists():
        print(f"⚠️  Аудиофайл {test_audio} не найден.")
        print("Пожалуйста, добавьте аудиофайл для тестирования.")
        print("\nПоддерживаемые форматы: mp3, mp4, mpeg, mpga, m4a, wav, webm")
        print("\nПример использования:")
        print("  python audio_transcription.py")
    else:
        print("🎵 Транскрибация аудио")
        print("=" * 50)
        
        # Простая транскрибация
        print("\n📝 Транскрибация:")
        transcript = transcribe_audio(test_audio)
        print(transcript)
        
        # Перевод
        print("\n🌐 Перевод на английский:")
        translation = translate_audio(test_audio)
        print(translation)
        
        # С временными метками
        print("\n⏱️  Транскрибация с временными метками:")
        segments = transcribe_with_timestamps(test_audio)
        for seg in segments[:5]:  # Первые 5 сегментов
            print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
