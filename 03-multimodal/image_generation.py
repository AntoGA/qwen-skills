"""
Image Generation
=================
Генерация изображений с использованием DALL-E API.

Этот пример демонстрирует:
- Генерацию изображений по текстовому описанию
- Создание вариаций существующих изображений
- Редактирование изображений (inpainting)
"""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import httpx

load_dotenv()

client = OpenAI()


def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    n: int = 1
) -> list:
    """
    Генерация изображений по текстовому описанию
    
    Args:
        prompt: Текстовое описание изображения
        size: Размер ("1024x1024", "1792x1024", "1024x1792")
        quality: Качество ("standard" или "hd")
        n: Количество изображений
    
    Returns:
        list: Список URL сгенерированных изображений
    """
    
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality=quality,
        n=n
    )
    
    urls = [image.url for image in response.data]
    return urls


def download_image(url: str, save_path: str) -> None:
    """
    Скачивание изображения по URL
    
    Args:
        url: URL изображения
        save_path: Путь для сохранения
    """
    
    with httpx.stream("GET", url) as response:
        with open(save_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    
    print(f"✅ Изображение сохранено: {save_path}")


def create_variation(image_path: str, n: int = 1) -> list:
    """
    Создание вариаций существующего изображения
    
    Args:
        image_path: Путь к исходному изображению
        n: Количество вариаций
    
    Returns:
        list: Список URL вариаций
    
    Note:
        Требует DALL-E 2 и изображений размером до 4MB
    """
    
    with open(image_path, "rb") as image_file:
        response = client.images.create_variation(
            image=image_file,
            n=n,
            size="1024x1024"
        )
    
    urls = [image.url for image in response.data]
    return urls


def edit_image(image_path: str, mask_path: str, prompt: str) -> list:
    """
    Редактирование изображения (inpainting)
    
    Args:
        image_path: Путь к исходному изображению
        mask_path: Путь к маске (область для редактирования)
        prompt: Описание желаемых изменений
    
    Returns:
        list: Список URL отредактированных изображений
    
    Note:
        Маска должна быть PNG с прозрачностью
    """
    
    with open(image_path, "rb") as image_file, \
         open(mask_path, "rb") as mask_file:
        
        response = client.images.edit(
            image=image_file,
            mask=mask_file,
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
    
    urls = [image.url for image in response.data]
    return urls


def generate_with_style(prompt: str, style: str = "vivid") -> list:
    """
    Генерация изображения с определенным стилем
    
    Args:
        prompt: Текстовое описание
        style: Стиль ("vivid" или "natural")
    
    Returns:
        list: Список URL изображений
    """
    
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="hd",
        style=style,
        n=1
    )
    
    return [image.url for image in response.data]


if __name__ == "__main__":
    # Пример использования
    
    print("🎨 Генерация изображений")
    print("=" * 50)
    
    # Генерация изображения
    prompt = "A futuristic city with flying cars and neon lights at sunset"
    
    print(f"\n🖼️  Генерация по запросу: '{prompt}'")
    urls = generate_image(prompt, quality="hd")
    
    for i, url in enumerate(urls):
        print(f"\n📷 Изображение {i + 1}:")
        print(url)
        
        # Скачивание
        save_path = f"generated_image_{i + 1}.png"
        download_image(url, save_path)
    
    # Генерация с разными стилями
    print("\n\n🎨 Генерация с разными стилями:")
    
    prompt2 = "A serene mountain landscape"
    
    print(f"\nЗапрос: '{prompt2}'")
    
    print("\n🌟 Vivid стиль:")
    vivid_urls = generate_with_style(prompt2, style="vivid")
    print(vivid_urls[0])
    
    print("\n🌿 Natural стиль:")
    natural_urls = generate_with_style(prompt2, style="natural")
    print(natural_urls[0])
    
    print("\n" + "=" * 50)
    print("✅ Готово!")
