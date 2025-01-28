import pytest
import torch
from PIL import Image
from io import BytesIO
import requests

# Импортируем наши функции из app.py (меняйте названия файла и функций, если отличаются)
from main.main import load_model, load_labels, get_preprocess_transform


def test_load_model():
    """Проверяем, что модель загружается и отдает выход."""
    model = load_model()
    assert model is not None, "Модель не должна быть None."

    # Создаем случайный тензор 1 x 3 x 224 x 224 (соответствие формату [batch, channels, height, width])
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    # У MobilenetV2 1000 выходных классов(предположения)
    assert output.shape == (1, 1000), "Выход модели должен иметь размер [1, 1000]."


def test_load_labels():
    """Проверяем, что метки загружаются и в списке 1000 элементов (ImageNet)."""
    labels = load_labels()
    assert isinstance(labels, list), "Метки должны возвращаться в виде списка."
    assert len(labels) == 1000, "Для ImageNet ожидается 1000 меток."
    # Дополнительно можно проверить, что в списке есть ожидаемые строки
    # Например, первая метка в официальном списке часто 'tench'
    assert any("tench" in label.lower() for label in labels), "Слово 'tench' не найдено среди меток."


def test_inference_on_image():
    """Проверяем, что модель корректно обрабатывает реальное изображение."""
    # Берем тестовое изображение (можно взять любое публичное)
    url = "https://avatars.mds.yandex.net/i?id=6b3edad500bc852b309883bebdcdeeb318a673c4-4727095-images-thumbs&n=13"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    # Применяем те же самые трансформации
    preprocess = get_preprocess_transform()
    image_tensor = preprocess(image).unsqueeze(0)  # [1, 3, 224, 224]

    model = load_model()
    with torch.no_grad():
        output = model(image_tensor)
    assert output.shape == (1, 1000), "Размер выхода при распознавании реального изображения должен быть [1, 1000]."
