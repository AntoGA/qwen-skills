"""
Vision Analysis
================
Анализ изображений с использованием Vision-Language моделей.

Этот пример демонстрирует:
- Загрузку изображений
- Описание содержимого
- Visual question answering
- OCR (распознавание текста)
"""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()


def encode_image(image_path: str) -> str:
    """
    Кодирует изображение в base64 для отправки в API
    
    Args:
        image_path: Путь к изображению
    
    Returns:
        str: Base64 строка
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_image(image_path: str, question: str = "What's in this image?") -> str:
    """
    Анализ изображения с помощью GPT-4 Vision
    
    Args:
        image_path: Путь к изображению
        question: Вопрос об изображении
    
    Returns:
        str: Ответ модели
    """
    
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content


def describe_image(image_path: str) -> str:
    """
    Подробное описание изображения
    """
    return analyze_image(
        image_path,
        "Please describe this image in detail. Include objects, colors, actions, and any text visible."
    )


def extract_text(image_path: str) -> str:
    """
    OCR - извлечение текста из изображения
    """
    return analyze_image(
        image_path,
        "Extract all visible text from this image. Return only the text."
    )


def answer_visual_question(image_path: str, question: str) -> str:
    """
    Ответ на вопрос об изображении
    """
    return analyze_image(image_path, question)


if __name__ == "__main__":
    # Пример использования
    
    # Создадим тестовое изображение (в реальном сценарии используйте свое)
    test_image = "test_image.jpg"
    
    if not Path(test_image).exists():
        print(f"⚠️  Изображение {test_image} не найдено.")
        print("Пожалуйста, добавьте изображение для тестирования.")
        print("\nПример использования:")
        print("  python vision_analysis.py")
    else:
        print("🖼️  Анализ изображения")
        print("=" * 50)
        
        # Описание
        print("\n📝 Описание:")
        description = describe_image(test_image)
        print(description)
        
        # Извлечение текста
        print("\n📄 Извлеченный текст:")
        text = extract_text(test_image)
        print(text if text else "Текст не найден")
        
        # Вопрос
        print("\n❓ Вопрос: Сколько людей на изображении?")
        answer = answer_visual_question(test_image, "How many people are in this image?")
        print(answer)
